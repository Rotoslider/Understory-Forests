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
import threading
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


class _SignalStream:
    """File-like stream that emits a Qt signal on write, for stdout capture."""

    def __init__(self, signal):
        self._signal = signal
        self._buf = ""

    def write(self, text: str) -> int:
        if not text:
            return 0
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line:
                self._signal.emit(line)
        return len(text)

    def flush(self) -> None:
        if self._buf:
            self._signal.emit(self._buf)
            self._buf = ""


class TrainingWorker(QThread):
    """Runs model training in a background thread."""

    progress = Signal(int, float, float, float, float)  # epoch, train_loss, train_acc, val_loss, val_acc
    finished = Signal(str)  # model path
    error = Signal(str)
    log_output = Signal(str)  # captured stdout/stderr line

    def __init__(self, parameters: dict, cancel_event: threading.Event,
                 pause_event: threading.Event, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._params = parameters
        self._cancel_event = cancel_event
        self._pause_event = pause_event

    def _on_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc):
        self.progress.emit(epoch, train_loss, train_acc, val_loss, val_acc)

    def run(self) -> None:
        stream = _SignalStream(self.log_output)
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = stream
        sys.stderr = stream
        try:
            scripts_dir = str(Path(__file__).parent.parent.parent.parent / "scripts")
            if scripts_dir not in sys.path:
                sys.path.insert(0, scripts_dir)

            from train import TrainModel
            trainer = TrainModel(
                self._params,
                progress_callback=self._on_epoch,
                cancel_event=self._cancel_event,
                pause_event=self._pause_event,
            )
            trainer.run_training()

            stream.flush()
            from tools import get_fsct_path
            model_path = os.path.join(get_fsct_path("model"), self._params["model_filename"])
            self.finished.emit(model_path)
        except Exception as e:
            stream.flush()
            self.error.emit(str(e))
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


class TrainingChartCanvas:
    """Lazy-loaded matplotlib chart for training loss/accuracy curves."""

    def __init__(self, parent=None):
        self._widget = None
        self._parent = parent
        self._epochs = []
        self._train_loss = []
        self._val_loss = []

    def _ensure_widget(self):
        if self._widget is not None:
            return
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        fig = Figure(figsize=(5, 3), dpi=100)
        fig.set_facecolor("#1a2e26")
        self._ax = fig.add_subplot(1, 1, 1)
        self._style_axis()
        fig.tight_layout(pad=1.5)
        self._widget = FigureCanvasQTAgg(fig)
        self._widget.setParent(self._parent)
        self._widget.setMinimumHeight(200)

    def _style_axis(self):
        ax = self._ax
        ax.set_facecolor("#0d1b16")
        ax.set_title("Training Loss", fontsize=11, fontweight="bold", color="#a8d8c0")
        ax.set_xlabel("Epoch", fontsize=9, color="#a8d8c0")
        ax.set_ylabel("Loss", fontsize=9, color="#a8d8c0")
        ax.tick_params(colors="#a8d8c0", labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#2d7a5e")
        ax.spines["bottom"].set_color("#2d7a5e")
        ax.grid(True, alpha=0.3, color="#2d7a5e")

    def widget(self):
        self._ensure_widget()
        return self._widget

    def add_epoch(self, epoch, train_loss, val_loss):
        self._ensure_widget()
        self._epochs.append(epoch)
        self._train_loss.append(train_loss)
        self._val_loss.append(val_loss)

        self._ax.clear()
        self._style_axis()
        self._ax.plot(self._epochs, self._train_loss, color="#4a9e7e", linewidth=1.5, label="Train")
        if any(v > 0 for v in self._val_loss):
            self._ax.plot(self._epochs, self._val_loss, color="#e8a838", linewidth=1.5, label="Validation")
        self._ax.legend(fontsize=8, facecolor="#1a2e26", edgecolor="#2d7a5e", labelcolor="#a8d8c0")
        self._widget.draw_idle()

    def reset(self):
        self._epochs.clear()
        self._train_loss.clear()
        self._val_loss.clear()
        if self._widget is not None:
            self._ax.clear()
            self._style_axis()
            self._widget.draw_idle()


class TrainingPanel(QWidget):
    """Training data pipeline panel with guided workflow."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._worker: Optional[TrainingWorker] = None
        self._setup_ui()

    @staticmethod
    def _add_help_button(group_box: QGroupBox, help_text: str) -> None:
        """Insert a small '?' button into a group box title area."""
        btn = QPushButton("?")
        btn.setFixedSize(20, 20)
        btn.setToolTip(help_text)
        btn.setStyleSheet(
            "QPushButton { font-weight: bold; font-size: 11px; padding: 0; "
            "border-radius: 10px; min-height: 0; min-width: 0; }"
        )
        btn.clicked.connect(lambda: QMessageBox.information(btn.window(), "Help", help_text))
        # Insert at the top of the group box layout
        existing_layout = group_box.layout()
        if existing_layout and existing_layout.count() > 0:
            existing_layout.insertWidget(0, btn)

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

        # Quick Start Guide — collapsible
        guide = QGroupBox("Quick Start Guide")
        guide.setCheckable(True)
        guide.setChecked(False)
        guide_layout = QVBoxLayout(guide)
        guide_text = QLabel(
            "<b>1.</b> Import labeled point clouds (.las) with labels 1-4.<br>"
            "<b>2.</b> Optionally bootstrap labels using an existing model — run "
            "the pipeline, then correct the output in the Label Editor.<br>"
            "<b>3.</b> Open the Label Editor to review and correct labels. "
            "Use keys 1-4 to quickly assign classes to selected points.<br>"
            "<b>4.</b> Configure training parameters. Start with defaults for "
            "fine-tuning an existing model, or increase epochs for training "
            "from scratch.<br>"
            "<b>5.</b> Click 'Start Training' and monitor the loss chart. "
            "Lower loss = better model. The model is saved after every epoch."
        )
        guide_text.setWordWrap(True)
        guide_text.setTextFormat(Qt.RichText)
        guide_layout.addWidget(guide_text)
        layout.addWidget(guide)

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

        self._add_help_button(step1,
            "Import labeled .las files into the data/train/ directory.\n"
            "Each file should have a 'label' column with values:\n"
            "  1 = Terrain, 2 = Vegetation, 3 = CWD, 4 = Stem\n\n"
            "You can also place files directly in the data/train/ folder."
        )
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
        self._add_help_button(step2,
            "If you don't have labeled data yet, use an existing model\n"
            "to auto-generate initial labels. Run the pipeline on your\n"
            "point cloud, then open the segmented output in the Label\n"
            "Editor (Step 3) to correct any mistakes."
        )
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
        self._add_help_button(step3,
            "The Label Editor lets you visually correct point labels.\n\n"
            "Key shortcuts:\n"
            "  R = Toggle box selection\n"
            "  B = Toggle brush selection\n"
            "  1-4 = Paint selection with that class\n"
            "  C = Toggle confidence heatmap\n"
            "  Ctrl+Z = Undo, Ctrl+Y = Redo\n\n"
            "Focus on low-confidence points first (red in confidence view)."
        )
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

        self._chart = TrainingChartCanvas(parent=None)
        s5_layout.addWidget(self._chart.widget())

        self._train_progress = QProgressBar()
        self._train_progress.setVisible(False)
        s5_layout.addWidget(self._train_progress)

        self._train_status = QLabel("Ready to train")
        s5_layout.addWidget(self._train_status)

        # GPU monitor
        self._gpu_label = QLabel("GPU: N/A")
        self._gpu_label.setStyleSheet("font-size: 11px; color: #2d7a5e;")
        self._gpu_label.setVisible(False)
        s5_layout.addWidget(self._gpu_label)

        # Control buttons
        btn_row = QHBoxLayout()

        self._train_btn = QPushButton("Start Training")
        self._train_btn.setObjectName("runButton")
        self._train_btn.clicked.connect(self._start_training)
        btn_row.addWidget(self._train_btn)

        self._pause_btn = QPushButton("Pause")
        self._pause_btn.setEnabled(False)
        self._pause_btn.clicked.connect(self._toggle_pause)
        btn_row.addWidget(self._pause_btn)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._stop_training)
        btn_row.addWidget(self._stop_btn)

        s5_layout.addLayout(btn_row)

        # Console output
        from PySide6.QtWidgets import QTextEdit
        self._console = QTextEdit()
        self._console.setReadOnly(True)
        self._console.setMaximumHeight(120)
        self._console.setPlaceholderText("Training output...")
        s5_layout.addWidget(self._console)

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
        self._pause_btn.setEnabled(True)
        self._stop_btn.setEnabled(True)
        self._train_progress.setVisible(True)
        self._train_progress.setRange(0, parameters["num_epochs"])
        self._train_status.setText("Training...")
        self._console.clear()
        self._chart.reset()

        # Start GPU monitor
        from understory.gui.main_window import GpuMonitor
        if not hasattr(self, "_gpu_monitor"):
            self._gpu_monitor = GpuMonitor(interval_ms=2000, parent=self)
            self._gpu_monitor.updated.connect(self._gpu_label.setText)
        self._gpu_label.setVisible(True)
        self._gpu_monitor.start()

        self._cancel_event = threading.Event()
        self._pause_event = threading.Event()

        self._worker = TrainingWorker(parameters, self._cancel_event, self._pause_event)
        self._worker.progress.connect(self._on_train_progress)
        self._worker.log_output.connect(self._on_log_output)
        self._worker.finished.connect(self._on_train_finished)
        self._worker.error.connect(self._on_train_error)
        self._worker.start()

    def _toggle_pause(self) -> None:
        if not self._worker or not self._worker.isRunning():
            return
        if self._pause_event.is_set():
            self._pause_event.clear()
            self._pause_btn.setText("Pause")
            self._train_status.setText("Resumed...")
        else:
            self._pause_event.set()
            self._pause_btn.setText("Resume")
            self._train_status.setText("Paused")

    def _stop_training(self) -> None:
        if not self._worker or not self._worker.isRunning():
            return
        self._cancel_event.set()
        # Also clear pause so the loop can exit
        self._pause_event.clear()
        self._stop_btn.setEnabled(False)
        self._pause_btn.setEnabled(False)
        self._train_status.setText("Stopping (after current epoch)...")

    @Slot(str)
    def _on_log_output(self, text: str) -> None:
        self._console.append(text)

    @Slot(int, float, float, float, float)
    def _on_train_progress(self, epoch: int, loss: float, acc: float, val_loss: float = 0, val_acc: float = 0) -> None:
        self._train_progress.setValue(epoch)
        status = f"Epoch {epoch} — Loss: {loss:.4f}, Acc: {acc:.4f}"
        if val_loss > 0:
            status += f" | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        self._train_status.setText(status)
        self._chart.add_epoch(epoch, loss, val_loss)

    def _stop_monitoring(self) -> None:
        """Stop GPU monitor and reset control buttons."""
        if hasattr(self, "_gpu_monitor"):
            self._gpu_monitor.stop()
        self._gpu_label.setVisible(False)
        self._train_btn.setEnabled(True)
        self._pause_btn.setEnabled(False)
        self._pause_btn.setText("Pause")
        self._stop_btn.setEnabled(False)

    @Slot(str)
    def _on_train_finished(self, model_path: str) -> None:
        self._stop_monitoring()
        self._train_progress.setVisible(False)
        self._train_status.setText(f"Training complete! Model saved: {model_path}")
        QMessageBox.information(self, "Training Complete", f"Model saved to:\n{model_path}")

    @Slot(str)
    def _on_train_error(self, msg: str) -> None:
        self._stop_monitoring()
        self._train_progress.setVisible(False)
        self._train_status.setText(f"Error: {msg}")
        QMessageBox.critical(self, "Training Error", msg)
