"""Training panel — guided workflow for training data preparation and model training.

Workflow:
1. Import unlabeled point cloud (LAS/PCD)
2. Run initial segmentation (bootstrap labels with existing model)
3. Review & correct labels in-app
4. Export labeled data
5. Configure & run training
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QCheckBox,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QGroupBox,
    QScrollArea,
    QFileDialog,
    QProgressBar,
    QMessageBox,
    QStackedWidget,
)

from understory.gui.tooltips import get_tooltip


class TrainingWorker(QThread):
    """Runs model training in a background thread."""

    progress = Signal(int, float, float)  # epoch, loss, accuracy
    finished = Signal(str)  # model path
    error = Signal(str)

    def __init__(self, parameters: dict, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._params = parameters

    def run(self) -> None:
        try:
            scripts_dir = str(Path(__file__).parent.parent.parent.parent / "scripts")
            if scripts_dir not in sys.path:
                sys.path.insert(0, scripts_dir)

            from train import TrainModel
            trainer = TrainModel(self._params)
            trainer.run_training()

            from tools import get_fsct_path
            model_path = os.path.join(get_fsct_path("model"), self._params["model_filename"])
            self.finished.emit(model_path)
        except Exception as e:
            self.error.emit(str(e))


class TrainingPanel(QWidget):
    """Training data pipeline panel with guided workflow."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._worker: Optional[TrainingWorker] = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(12, 12, 12, 12)

        title = QLabel("Training Workflow")
        title.setObjectName("sectionHeader")
        layout.addWidget(title)

        # Step 1: Data import
        step1 = QGroupBox("Step 1: Import Training Data")
        s1_layout = QVBoxLayout(step1)

        s1_layout.addWidget(QLabel(
            "Import labeled point clouds (.las) into the training data directory.\n"
            "Labels should be: 1=Terrain, 2=Vegetation, 3=CWD, 4=Stem"
        ))

        import_row = QHBoxLayout()
        self._train_dir_label = QLineEdit()
        self._train_dir_label.setReadOnly(True)
        self._train_dir_label.setPlaceholderText("data/train/")
        import_row.addWidget(self._train_dir_label)

        import_btn = QPushButton("Import Files...")
        import_btn.clicked.connect(self._import_training_data)
        import_row.addWidget(import_btn)
        s1_layout.addLayout(import_row)

        layout.addWidget(step1)

        # Step 2: Bootstrap labels
        step2 = QGroupBox("Step 2: Bootstrap Labels (Optional)")
        s2_layout = QVBoxLayout(step2)
        s2_layout.addWidget(QLabel(
            "Run inference with an existing model to generate initial labels\n"
            "for unlabeled point clouds. Then correct them in Step 3."
        ))

        bootstrap_btn = QPushButton("Bootstrap Labels...")
        bootstrap_btn.clicked.connect(self._bootstrap_labels)
        s2_layout.addWidget(bootstrap_btn)
        layout.addWidget(step2)

        # Step 3: Label correction
        step3 = QGroupBox("Step 3: Review & Correct Labels")
        s3_layout = QVBoxLayout(step3)
        s3_layout.addWidget(QLabel(
            "Open the label editor to review and correct point labels.\n"
            "Use keyboard shortcuts 1-4 for quick class assignment."
        ))

        edit_btn = QPushButton("Open Label Editor...")
        edit_btn.clicked.connect(self._open_label_editor)
        s3_layout.addWidget(edit_btn)
        layout.addWidget(step3)

        # Step 4: Training configuration
        step4 = QGroupBox("Step 4: Configure Training")
        s4_form = QFormLayout(step4)

        self._model_name = QLineEdit("modelV2.pth")
        s4_form.addRow("Model filename:", self._model_name)

        self._epochs = QSpinBox()
        self._epochs.setRange(1, 100000)
        self._epochs.setValue(2000)
        self._epochs.setToolTip(get_tooltip("epochs"))
        s4_form.addRow("Epochs:", self._epochs)

        self._lr = QDoubleSpinBox()
        self._lr.setRange(0.0000001, 0.01)
        self._lr.setDecimals(7)
        self._lr.setSingleStep(0.000005)
        self._lr.setValue(0.000025)
        self._lr.setToolTip(get_tooltip("learning_rate"))
        s4_form.addRow("Learning rate:", self._lr)

        self._train_batch = QSpinBox()
        self._train_batch.setRange(1, 64)
        self._train_batch.setValue(2)
        self._train_batch.setToolTip(get_tooltip("train_batch_size"))
        s4_form.addRow("Training batch size:", self._train_batch)

        self._val_batch = QSpinBox()
        self._val_batch.setRange(1, 64)
        self._val_batch.setValue(2)
        self._val_batch.setToolTip(get_tooltip("validation_batch_size"))
        s4_form.addRow("Validation batch size:", self._val_batch)

        self._device_combo = QComboBox()
        self._device_combo.addItems(["cuda", "cpu"])
        s4_form.addRow("Device:", self._device_combo)

        self._load_existing = QCheckBox("Load existing model weights")
        self._load_existing.setChecked(True)
        s4_form.addRow(self._load_existing)

        self._validate = QCheckBox("Run validation during training")
        self._validate.setChecked(True)
        s4_form.addRow(self._validate)

        self._class_weights = QComboBox()
        self._class_weights.addItems(["Auto (recommended)", "None (uniform)"])
        self._class_weights.setToolTip(get_tooltip("class_weights"))
        s4_form.addRow("Class weights:", self._class_weights)

        layout.addWidget(step4)

        # Step 5: Run training
        step5 = QGroupBox("Step 5: Train Model")
        s5_layout = QVBoxLayout(step5)

        self._train_progress = QProgressBar()
        self._train_progress.setVisible(False)
        s5_layout.addWidget(self._train_progress)

        self._train_status = QLabel("Ready to train")
        s5_layout.addWidget(self._train_status)

        self._train_btn = QPushButton("Start Training")
        self._train_btn.setObjectName("runButton")
        self._train_btn.clicked.connect(self._start_training)
        s5_layout.addWidget(self._train_btn)

        layout.addWidget(step5)
        layout.addStretch()

        scroll.setWidget(content)
        outer.addWidget(scroll)

    def _import_training_data(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "Import Training Data",
            "", "Point Clouds (*.las *.laz *.pcd);;All Files (*)",
        )
        if not files:
            return

        scripts_dir = str(Path(__file__).parent.parent.parent.parent / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from tools import get_fsct_path

        import shutil
        train_dir = Path(get_fsct_path("data")) / "train"
        train_dir.mkdir(parents=True, exist_ok=True)

        copied = 0
        for f in files:
            dest = train_dir / Path(f).name
            if Path(f).resolve() == dest.resolve():
                continue  # already in train dir
            shutil.copy2(f, dest)
            copied += 1

        self._train_dir_label.setText(str(train_dir))
        skipped = len(files) - copied
        msg = f"Imported {copied} file(s) to {train_dir}"
        if skipped:
            msg += f"\n({skipped} already in training directory, skipped)"
        QMessageBox.information(self, "Import Complete", msg)

    def _bootstrap_labels(self) -> None:
        QMessageBox.information(
            self, "Bootstrap Labels",
            "To bootstrap labels:\n"
            "1. Run the pipeline on your unlabeled point cloud\n"
            "2. Use the segmented output as initial labels\n"
            "3. Open in the Label Editor to correct mistakes",
        )

    def _open_label_editor(self) -> None:
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open Point Cloud for Label Editing",
            "", "Point Clouds (*.las *.laz *.pcd);;All Files (*)",
        )
        if not filepath:
            return

        from understory.gui.viewer.label_editor import LabelEditor

        scripts_dir = str(Path(__file__).parent.parent.parent.parent / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from tools import load_file

        pc, headers = load_file(filepath, headers_of_interest=["x", "y", "z", "label", "confidence"])

        editor = LabelEditor()
        labels = None
        confidence = None
        if headers and "label" in headers:
            label_idx = headers.index("label")
            labels = pc[:, label_idx].astype(np.int32)
        if headers and "confidence" in headers:
            conf_idx = headers.index("confidence")
            confidence = pc[:, conf_idx].astype(np.float32)
        editor.load_points(pc[:, :3], labels=labels, confidence=confidence)
        editor.setWindowTitle(f"Label Editor — {os.path.basename(filepath)}")
        editor.resize(1200, 800)
        editor.show()
        # Keep a reference to prevent garbage collection
        self._label_editor = editor

    def _start_training(self) -> None:
        if self._worker and self._worker.isRunning():
            QMessageBox.warning(self, "Training Running", "Training is already in progress.")
            return

        parameters = dict(
            preprocess_train_datasets=True,
            preprocess_validation_datasets=True,
            clean_sample_directories=True,
            perform_validation_during_training=self._validate.isChecked(),
            generate_point_cloud_vis=False,
            load_existing_model=self._load_existing.isChecked(),
            num_epochs=self._epochs.value(),
            learning_rate=self._lr.value(),
            model_filename=self._model_name.text(),
            sample_box_size_m=np.array([6, 6, 6]),
            sample_box_overlap=[0.5, 0.5, 0.5],
            min_points_per_box=1000,
            max_points_per_box=20000,
            subsample=False,
            subsampling_min_spacing=0.025,
            num_cpu_cores_preprocessing=0,
            num_cpu_cores_deep_learning=1,
            train_batch_size=self._train_batch.value(),
            validation_batch_size=self._val_batch.value(),
            device=self._device_combo.currentText(),
            class_weights="auto" if self._class_weights.currentIndex() == 0 else None,
        )

        self._train_btn.setEnabled(False)
        self._train_progress.setVisible(True)
        self._train_progress.setRange(0, parameters["num_epochs"])
        self._train_status.setText("Training...")

        self._worker = TrainingWorker(parameters)
        self._worker.progress.connect(self._on_train_progress)
        self._worker.finished.connect(self._on_train_finished)
        self._worker.error.connect(self._on_train_error)
        self._worker.start()

    @Slot(int, float, float)
    def _on_train_progress(self, epoch: int, loss: float, acc: float) -> None:
        self._train_progress.setValue(epoch)
        self._train_status.setText(f"Epoch {epoch} — Loss: {loss:.4f}, Acc: {acc:.4f}")

    @Slot(str)
    def _on_train_finished(self, model_path: str) -> None:
        self._train_btn.setEnabled(True)
        self._train_progress.setVisible(False)
        self._train_status.setText(f"Training complete! Model saved: {model_path}")
        QMessageBox.information(self, "Training Complete", f"Model saved to:\n{model_path}")

    @Slot(str)
    def _on_train_error(self, msg: str) -> None:
        self._train_btn.setEnabled(True)
        self._train_progress.setVisible(False)
        self._train_status.setText(f"Error: {msg}")
        QMessageBox.critical(self, "Training Error", msg)
