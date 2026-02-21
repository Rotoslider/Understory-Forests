"""Batch processing panel -- run the pipeline on multiple point cloud files.

Provides a QDialog with a file list, per-file and overall progress tracking,
console log output, and cancellation support.  All files share the same
pipeline settings (taken from the current ProjectConfig) but each gets its
own output directory derived from its filename.
"""

from __future__ import annotations

import gc
import os
import threading
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QTextEdit,
    QMessageBox,
    QGroupBox,
    QWidget,
)

from understory.config.settings import ProjectConfig


# ---------------------------------------------------------------------------
# Status constants
# ---------------------------------------------------------------------------

class FileStatus:
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"
    CANCELLED = "cancelled"


# Unicode icons used as visual status indicators in the file list.
_STATUS_ICONS = {
    FileStatus.PENDING: "\u23F3",    # hourglass
    FileStatus.RUNNING: "\u25B6",    # play triangle
    FileStatus.DONE: "\u2705",       # check mark
    FileStatus.ERROR: "\u274C",      # cross mark
    FileStatus.CANCELLED: "\u23F9",  # stop square
}

_STATUS_COLORS = {
    FileStatus.PENDING: QColor(180, 180, 180),
    FileStatus.RUNNING: QColor(50, 150, 255),
    FileStatus.DONE: QColor(50, 200, 80),
    FileStatus.ERROR: QColor(220, 50, 50),
    FileStatus.CANCELLED: QColor(180, 140, 40),
}


# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------

class BatchPipelineWorker(QThread):
    """Iterates over a list of files, running ``run_pipeline()`` for each.

    Signals
    -------
    file_started(int, str)
        Index and path of the file about to be processed.
    file_progress(int, str, float)
        Index, stage name, and overall fraction (0-1) for the current file.
    file_finished(int, str)
        Index and output directory of the completed file.
    file_error(int, str, str)
        Index, short user message, and full traceback.
    file_cancelled(int)
        Index of the file that was being processed when cancellation occurred.
    overall_progress(float)
        Overall batch fraction (0-1).
    all_done(int, int, int)
        Total files, succeeded count, failed count.
    log_message(str)
        Free-form log text to append to the console.
    """

    file_started = Signal(int, str)         # index, filepath
    file_progress = Signal(int, str, float) # index, stage, fraction
    file_finished = Signal(int, str)        # index, output_dir
    file_error = Signal(int, str, str)      # index, user_msg, traceback
    file_cancelled = Signal(int)            # index
    overall_progress = Signal(float)        # 0.0 - 1.0
    all_done = Signal(int, int, int)        # total, succeeded, failed
    log_message = Signal(str)               # text

    def __init__(
        self,
        file_paths: list[str],
        base_config: ProjectConfig,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._file_paths = list(file_paths)
        self._base_config = base_config
        self._cancel_event = threading.Event()

    def request_stop(self) -> None:
        """Request cooperative cancellation."""
        self._cancel_event.set()

    # ---- internal helpers ----

    def _clone_config(self, filepath: str) -> ProjectConfig:
        """Create a deep copy of the base config with the input file replaced."""
        # Use asdict + load round-trip for a clean deep copy
        data = asdict(self._base_config)
        clone = ProjectConfig(
            project_name=data.get("project_name", ""),
            operator=data.get("operator", ""),
            notes=data.get("notes", ""),
            point_cloud_filename=filepath,
            prepared_point_cloud="",
            preprocess=data.get("preprocess", True),
            segmentation=data.get("segmentation", True),
            postprocessing=data.get("postprocessing", True),
            measure_plot=data.get("measure_plot", True),
            make_report=data.get("make_report", True),
            clean_up_files=data.get("clean_up_files", False),
        )
        # Copy sub-configs field-by-field from the dict so defaults stay correct
        from understory.config.settings import (
            ProcessingConfig,
            ModelConfig,
            MeasurementConfig,
            OutputConfig,
        )
        clone.processing = ProcessingConfig(**data.get("processing", {}))
        clone.model = ModelConfig(**data.get("model", {}))
        clone.measurement = MeasurementConfig(**data.get("measurement", {}))
        clone.output = OutputConfig(**data.get("output", {}))

        # Override input and clear output so pipeline auto-computes it
        clone.point_cloud_filename = filepath
        clone.output.output_directory = None
        return clone

    def _cleanup_gpu(self) -> None:
        """Release GPU memory between pipeline runs."""
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # ---- main loop ----

    def run(self) -> None:  # noqa: C901  (complexity acceptable for a worker)
        # Force non-interactive Matplotlib backend for background thread
        import matplotlib
        matplotlib.use("Agg")

        from understory.core.pipeline import run_pipeline, PipelineCancelled, PipelineStageError

        total = len(self._file_paths)
        succeeded = 0
        failed = 0

        for idx, filepath in enumerate(self._file_paths):
            # --- check cancellation before starting next file ---
            if self._cancel_event.is_set():
                self.file_cancelled.emit(idx)
                self.log_message.emit(f"Batch cancelled before file {idx + 1}/{total}.")
                break

            filename = os.path.basename(filepath)
            self.file_started.emit(idx, filepath)
            self.log_message.emit(
                f"\n{'='*60}\n"
                f"[{idx + 1}/{total}] Processing: {filename}\n"
                f"{'='*60}"
            )

            config = self._clone_config(filepath)

            def _progress_cb(stage: str, fraction: float, _idx: int = idx) -> None:
                self.file_progress.emit(_idx, stage, fraction)
                # Overall: completed files + fraction of current file
                overall = (_idx + fraction) / total
                self.overall_progress.emit(overall)

            try:
                result = run_pipeline(
                    config,
                    progress_callback=_progress_cb,
                    cancel_event=self._cancel_event,
                )
                output_dir = result.get("output_dir", "")
                self.file_finished.emit(idx, output_dir)
                self.log_message.emit(f"[{idx + 1}/{total}] Done: {filename} -> {output_dir}")
                succeeded += 1

            except PipelineCancelled:
                self.file_cancelled.emit(idx)
                self.log_message.emit(f"[{idx + 1}/{total}] Cancelled: {filename}")
                break

            except Exception as e:
                tb = traceback.format_exc()
                if isinstance(e, PipelineStageError):
                    user_msg = e.user_message
                else:
                    user_msg = str(e)
                self.file_error.emit(idx, user_msg, tb)
                self.log_message.emit(f"[{idx + 1}/{total}] ERROR: {filename} -- {user_msg}")
                failed += 1

            finally:
                self._cleanup_gpu()

        self.overall_progress.emit(1.0)
        self.all_done.emit(total, succeeded, failed)
        self.log_message.emit(
            f"\nBatch complete: {succeeded} succeeded, {failed} failed "
            f"out of {total} files."
        )


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------

class BatchPanel(QDialog):
    """Dialog for multi-file batch processing."""

    # Accepted point cloud extensions (case-insensitive matching)
    ACCEPTED_EXTENSIONS = {".las", ".laz", ".pcd"}

    def __init__(
        self,
        base_config: Optional[ProjectConfig] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Batch Processing")
        self.setMinimumSize(700, 560)
        self.resize(780, 640)

        self._base_config = base_config or ProjectConfig()
        self._worker: Optional[BatchPipelineWorker] = None

        self._setup_ui()

    # ------------------------------------------------------------------ UI

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)

        # -- File list group --
        file_group = QGroupBox("Input Files")
        file_layout = QVBoxLayout(file_group)

        self._file_list = QListWidget()
        self._file_list.setSelectionMode(QListWidget.ExtendedSelection)
        self._file_list.setAlternatingRowColors(True)
        file_layout.addWidget(self._file_list)

        btn_row = QHBoxLayout()
        self._add_btn = QPushButton("Add Files...")
        self._add_btn.clicked.connect(self._add_files)
        btn_row.addWidget(self._add_btn)

        self._add_dir_btn = QPushButton("Add Folder...")
        self._add_dir_btn.clicked.connect(self._add_folder)
        btn_row.addWidget(self._add_dir_btn)

        self._remove_btn = QPushButton("Remove Selected")
        self._remove_btn.clicked.connect(self._remove_selected)
        btn_row.addWidget(self._remove_btn)

        self._clear_btn = QPushButton("Clear All")
        self._clear_btn.clicked.connect(self._clear_files)
        btn_row.addWidget(self._clear_btn)

        file_layout.addLayout(btn_row)
        root.addWidget(file_group)

        # -- Progress group --
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)

        # Per-file progress
        self._file_progress_label = QLabel("Per-file progress:")
        progress_layout.addWidget(self._file_progress_label)
        self._file_progress_bar = QProgressBar()
        self._file_progress_bar.setRange(0, 100)
        self._file_progress_bar.setValue(0)
        progress_layout.addWidget(self._file_progress_bar)

        # Overall progress
        self._overall_progress_label = QLabel("Overall progress:")
        progress_layout.addWidget(self._overall_progress_label)
        self._overall_progress_bar = QProgressBar()
        self._overall_progress_bar.setRange(0, 100)
        self._overall_progress_bar.setValue(0)
        progress_layout.addWidget(self._overall_progress_bar)

        root.addWidget(progress_group)

        # -- Run / Stop buttons --
        action_row = QHBoxLayout()
        self._run_btn = QPushButton("Run Batch")
        self._run_btn.setObjectName("runButton")
        self._run_btn.clicked.connect(self._run_batch)
        action_row.addWidget(self._run_btn)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._stop_batch)
        action_row.addWidget(self._stop_btn)

        action_row.addStretch()

        self._close_btn = QPushButton("Close")
        self._close_btn.clicked.connect(self.close)
        action_row.addWidget(self._close_btn)

        root.addLayout(action_row)

        # -- Console log --
        self._console = QTextEdit()
        self._console.setReadOnly(True)
        self._console.setMaximumHeight(180)
        self._console.setPlaceholderText("Batch console output...")
        root.addWidget(self._console)

    # -------------------------------------------------------- file management

    def _is_accepted(self, filepath: str) -> bool:
        return Path(filepath).suffix.lower() in self.ACCEPTED_EXTENSIONS

    def _file_already_listed(self, filepath: str) -> bool:
        for i in range(self._file_list.count()):
            item = self._file_list.item(i)
            if item.data(Qt.UserRole) == filepath:
                return True
        return False

    def _add_file_item(self, filepath: str) -> None:
        """Add a single file to the list widget with pending status."""
        if not self._is_accepted(filepath):
            return
        if self._file_already_listed(filepath):
            return
        item = QListWidgetItem()
        item.setData(Qt.UserRole, filepath)
        self._update_item_status(item, FileStatus.PENDING)
        self._file_list.addItem(item)

    def _update_item_status(self, item: QListWidgetItem, status: str) -> None:
        filepath = item.data(Qt.UserRole)
        filename = os.path.basename(filepath)
        icon = _STATUS_ICONS.get(status, "")
        item.setText(f"{icon}  {filename}")
        item.setForeground(_STATUS_COLORS.get(status, QColor(180, 180, 180)))
        item.setData(Qt.UserRole + 1, status)

    def _add_files(self) -> None:
        filepaths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Point Cloud Files",
            "",
            "Point Clouds (*.las *.laz *.pcd);;All Files (*)",
        )
        for fp in filepaths:
            self._add_file_item(fp)

    def _add_folder(self) -> None:
        dirpath = QFileDialog.getExistingDirectory(self, "Select Folder")
        if not dirpath:
            return
        added = 0
        for entry in sorted(Path(dirpath).iterdir()):
            if entry.is_file() and entry.suffix.lower() in self.ACCEPTED_EXTENSIONS:
                self._add_file_item(str(entry))
                added += 1
        if added == 0:
            QMessageBox.information(
                self,
                "No Files Found",
                f"No .las, .laz, or .pcd files found in:\n{dirpath}",
            )

    def _remove_selected(self) -> None:
        for item in self._file_list.selectedItems():
            row = self._file_list.row(item)
            self._file_list.takeItem(row)

    def _clear_files(self) -> None:
        self._file_list.clear()

    def _get_file_paths(self) -> list[str]:
        paths = []
        for i in range(self._file_list.count()):
            item = self._file_list.item(i)
            paths.append(item.data(Qt.UserRole))
        return paths

    # ---------------------------------------------------------- config access

    def set_base_config(self, config: ProjectConfig) -> None:
        """Update the base config used for all batch runs."""
        self._base_config = config

    # -------------------------------------------------------- batch execution

    def _run_batch(self) -> None:
        if self._worker and self._worker.isRunning():
            QMessageBox.warning(self, "Batch Running", "A batch is already running.")
            return

        file_paths = self._get_file_paths()
        if not file_paths:
            QMessageBox.warning(self, "No Files", "Add at least one point cloud file to process.")
            return

        # Validate that all files still exist
        missing = [fp for fp in file_paths if not os.path.exists(fp)]
        if missing:
            names = "\n".join(os.path.basename(f) for f in missing[:10])
            QMessageBox.warning(
                self,
                "Missing Files",
                f"The following files could not be found:\n\n{names}",
            )
            return

        # Reset all items to pending
        for i in range(self._file_list.count()):
            self._update_item_status(self._file_list.item(i), FileStatus.PENDING)

        self._console.clear()
        self._log(f"Starting batch processing of {len(file_paths)} file(s)...")

        self._file_progress_bar.setValue(0)
        self._overall_progress_bar.setValue(0)
        self._file_progress_label.setText("Per-file progress:")
        self._overall_progress_label.setText("Overall progress: 0 / {0}".format(len(file_paths)))

        self._run_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._add_btn.setEnabled(False)
        self._add_dir_btn.setEnabled(False)
        self._remove_btn.setEnabled(False)
        self._clear_btn.setEnabled(False)

        self._worker = BatchPipelineWorker(file_paths, self._base_config)
        self._worker.file_started.connect(self._on_file_started)
        self._worker.file_progress.connect(self._on_file_progress)
        self._worker.file_finished.connect(self._on_file_finished)
        self._worker.file_error.connect(self._on_file_error)
        self._worker.file_cancelled.connect(self._on_file_cancelled)
        self._worker.overall_progress.connect(self._on_overall_progress)
        self._worker.all_done.connect(self._on_all_done)
        self._worker.log_message.connect(self._log)
        self._worker.start()

    def _stop_batch(self) -> None:
        if self._worker and self._worker.isRunning():
            self._log("Requesting batch stop (will finish current file's operation)...")
            self._worker.request_stop()
            self._stop_btn.setEnabled(False)

    # ------------------------------------------------------------ slot handlers

    @Slot(int, str)
    def _on_file_started(self, index: int, filepath: str) -> None:
        if index < self._file_list.count():
            self._update_item_status(self._file_list.item(index), FileStatus.RUNNING)
            self._file_list.scrollToItem(self._file_list.item(index))
        total = self._file_list.count()
        self._file_progress_bar.setValue(0)
        self._file_progress_label.setText(
            f"Per-file progress: {os.path.basename(filepath)}"
        )
        self._overall_progress_label.setText(
            f"Overall progress: {index + 1} / {total}"
        )

    @Slot(int, str, float)
    def _on_file_progress(self, index: int, stage: str, fraction: float) -> None:
        self._file_progress_bar.setValue(int(fraction * 100))
        filepath = ""
        if index < self._file_list.count():
            filepath = os.path.basename(
                self._file_list.item(index).data(Qt.UserRole)
            )
        self._file_progress_label.setText(
            f"Per-file progress: {filepath} -- {stage}"
        )

    @Slot(int, str)
    def _on_file_finished(self, index: int, output_dir: str) -> None:
        if index < self._file_list.count():
            self._update_item_status(self._file_list.item(index), FileStatus.DONE)

    @Slot(int, str, str)
    def _on_file_error(self, index: int, user_msg: str, tb: str) -> None:
        if index < self._file_list.count():
            self._update_item_status(self._file_list.item(index), FileStatus.ERROR)
        if tb:
            self._log(f"--- Traceback ---\n{tb}")

    @Slot(int)
    def _on_file_cancelled(self, index: int) -> None:
        # Mark the cancelled file and all remaining files
        for i in range(index, self._file_list.count()):
            item = self._file_list.item(i)
            if item.data(Qt.UserRole + 1) == FileStatus.PENDING:
                self._update_item_status(item, FileStatus.CANCELLED)
            elif i == index:
                self._update_item_status(item, FileStatus.CANCELLED)

    @Slot(float)
    def _on_overall_progress(self, fraction: float) -> None:
        self._overall_progress_bar.setValue(int(fraction * 100))

    @Slot(int, int, int)
    def _on_all_done(self, total: int, succeeded: int, failed: int) -> None:
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._add_btn.setEnabled(True)
        self._add_dir_btn.setEnabled(True)
        self._remove_btn.setEnabled(True)
        self._clear_btn.setEnabled(True)

        self._overall_progress_label.setText(
            f"Overall progress: {succeeded + failed} / {total} -- complete"
        )

        skipped = total - succeeded - failed
        summary = f"Batch finished: {succeeded} succeeded, {failed} failed"
        if skipped > 0:
            summary += f", {skipped} skipped/cancelled"
        self._log(summary)

        # Show a summary dialog
        icon = QMessageBox.Information if failed == 0 else QMessageBox.Warning
        box = QMessageBox(self)
        box.setIcon(icon)
        box.setWindowTitle("Batch Complete")
        box.setText(summary)
        box.exec()

    # ------------------------------------------------------------ helpers

    def _log(self, msg: str) -> None:
        self._console.append(msg)

    def closeEvent(self, event) -> None:
        """Prevent closing while a batch is still running."""
        if self._worker and self._worker.isRunning():
            reply = QMessageBox.question(
                self,
                "Batch Running",
                "A batch is still running. Stop it and close?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.No:
                event.ignore()
                return
            self._worker.request_stop()
            self._worker.wait(10000)
        super().closeEvent(event)
