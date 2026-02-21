"""Processing panel — tabbed sidebar with all pipeline controls.

Provides project setup, point cloud preparation, parameter configuration,
model selection, pipeline stage controls, and output configuration
in a tabbed interface.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Optional

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
    QTextEdit,
    QProgressBar,
    QMessageBox,
    QTabWidget,
    QTableView,
    QHeaderView,
)
from PySide6.QtGui import QPixmap, QAction
from PySide6.QtCore import QAbstractTableModel, QModelIndex

from understory.config.settings import ProjectConfig
from understory.gui.tooltips import get_tooltip


class _SignalStream:
    """File-like stream that emits a Qt signal on write, for stdout capture."""

    def __init__(self, signal):
        self._signal = signal
        self._buf = ""

    def write(self, text: str) -> int:
        if not text:
            return 0
        # Buffer partial lines; emit on newline
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line:  # skip empty lines from trailing newlines
                self._signal.emit(line)
        return len(text)

    def flush(self) -> None:
        if self._buf:
            self._signal.emit(self._buf)
            self._buf = ""


class PipelineWorker(QThread):
    """Runs the FSCT pipeline in a background thread."""

    progress = Signal(str, float)  # stage_name, fraction
    finished = Signal(str)  # output_dir
    error = Signal(str, str)  # short user message, full traceback
    cancelled = Signal()
    log_output = Signal(str)  # captured stdout/stderr line

    def __init__(self, config: ProjectConfig, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._config = config
        self._cancel_event = threading.Event()

    def request_stop(self):
        self._cancel_event.set()

    def run(self) -> None:
        import traceback
        # Force non-interactive Matplotlib backend for background thread
        import matplotlib
        matplotlib.use("Agg")

        # Redirect stdout/stderr so print() output appears in console
        stream = _SignalStream(self.log_output)
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = stream
        sys.stderr = stream
        try:
            from understory.core.pipeline import run_pipeline, PipelineStageError, PipelineCancelled
            result = run_pipeline(self._config, progress_callback=self._emit_progress, cancel_event=self._cancel_event)
            stream.flush()
            self.finished.emit(result.get("output_dir", ""))
        except PipelineCancelled:
            stream.flush()
            self.cancelled.emit()
        except Exception as e:
            stream.flush()
            tb = traceback.format_exc()
            from understory.core.pipeline import PipelineStageError
            if isinstance(e, PipelineStageError):
                self.error.emit(e.user_message, tb)
            else:
                self.error.emit(str(e), tb)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            # Always release GPU memory when pipeline thread exits
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    def _emit_progress(self, stage: str, fraction: float) -> None:
        self.progress.emit(stage, fraction)


class PandasTableModel(QAbstractTableModel):
    """Read-only Qt table model backed by a pandas DataFrame."""

    def __init__(self, df=None, parent=None):
        super().__init__(parent)
        self._df = df

    def set_dataframe(self, df):
        self.beginResetModel()
        self._df = df
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return len(self._df) if self._df is not None else 0

    def columnCount(self, parent=QModelIndex()):
        return len(self._df.columns) if self._df is not None else 0

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and self._df is not None:
            value = self._df.iloc[index.row(), index.column()]
            if isinstance(value, float):
                return f"{value:.3f}"
            return str(value)
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and self._df is not None:
            if orientation == Qt.Horizontal:
                return str(self._df.columns[section])
            return str(section + 1)
        return None


class ProcessingPanel(QWidget):
    """Tabbed sidebar panel with all processing controls."""

    file_loaded = Signal(str)  # filepath
    pipeline_started = Signal()  # emitted when the pipeline worker starts
    pipeline_finished = Signal(str)  # output_dir
    pipeline_error = Signal()  # emitted when pipeline errors out
    project_saved = Signal(str)  # filepath where project was saved
    load_output_layers = Signal(list)  # list of .las file paths
    tree_selected = Signal(int)  # tree ID selected in the results table
    plot_centre_changed = Signal(object)  # (x, y) tuple or None for auto
    swap_axes_requested = Signal(str)  # mode: yz, xz, xy, rot90z, reset
    crop_outliers_requested = Signal()
    reset_crop_requested = Signal()
    subsample_requested = Signal(float)  # voxel spacing in metres
    save_cloud_requested = Signal(str)  # filepath to save modified cloud
    trim_select_requested = Signal()  # start rectangle selection for trimming
    trim_apply_requested = Signal(bool)  # True=keep, False=remove
    trim_cancel_requested = Signal()  # cancel trim selection

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._config = ProjectConfig()
        self._worker: Optional[PipelineWorker] = None
        self._updating_from_viewer: bool = False
        self._last_save_path: Optional[str] = None
        self._prepared_cloud_path: Optional[str] = None

        self._widget_defaults: dict = {}  # widget -> default value
        self._setup_ui()
        self._register_defaults()

    def _register_default(self, widget, default_value) -> None:
        """Record a default value and install a right-click 'Reset to Default' menu."""
        self._widget_defaults[widget] = default_value
        widget.setContextMenuPolicy(Qt.CustomContextMenu)
        widget.customContextMenuRequested.connect(
            lambda pos, w=widget, d=default_value: self._show_reset_menu(w, d, pos)
        )

    def _show_reset_menu(self, widget, default_value, pos) -> None:
        """Show context menu with Reset to Default action."""
        from PySide6.QtWidgets import QMenu
        menu = QMenu(self)
        if isinstance(widget, QCheckBox):
            label = "checked" if default_value else "unchecked"
        else:
            label = str(default_value)
        action = QAction(f"Reset to Default ({label})", self)
        action.triggered.connect(lambda: self._reset_widget(widget, default_value))
        menu.addAction(action)
        menu.exec(widget.mapToGlobal(pos))

    def _reset_widget(self, widget, default_value) -> None:
        """Reset a widget to its default value."""
        if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            widget.setValue(default_value)
        elif isinstance(widget, QCheckBox):
            widget.setChecked(default_value)
        elif isinstance(widget, QComboBox):
            idx = widget.findText(str(default_value))
            if idx >= 0:
                widget.setCurrentIndex(idx)
        elif isinstance(widget, QLineEdit):
            widget.setText(str(default_value))

    def _setup_ui(self) -> None:
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        # Header — always use styled text for readability
        header = QWidget()
        header.setObjectName("sidebarHeader")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(12, 12, 12, 8)

        # Small icon for branding
        icon_path = Path(__file__).parent.parent.parent / "resources" / "icons" / "understory-icon.png"
        if icon_path.exists():
            icon_label = QLabel()
            pixmap = QPixmap(str(icon_path))
            icon_label.setPixmap(pixmap.scaledToHeight(48, Qt.SmoothTransformation))
            icon_label.setAlignment(Qt.AlignCenter)
            header_layout.addWidget(icon_label)

        title = QLabel("Understory")
        title.setObjectName("sectionHeader")
        title.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(title)

        subtitle = QLabel("Forest Structural Complexity Tool")
        subtitle.setObjectName("subtitle")
        subtitle.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(subtitle)

        outer_layout.addWidget(header)

        # Tab widget
        self._tabs = QTabWidget()
        self._setup_project_tab()
        self._setup_prepare_tab()
        self._setup_process_tab()
        self._setup_advanced_tab()
        self._setup_results_tab()
        outer_layout.addWidget(self._tabs, 1)

        # Persistent controls below tabs
        bottom = QWidget()
        bottom_layout = QVBoxLayout(bottom)
        bottom_layout.setContentsMargins(12, 4, 12, 4)

        # Progress — dual bar: overall (top, thin) + stage (bottom, main)
        self._progress_container = QWidget()
        self._progress_container.setVisible(False)
        prog_layout = QVBoxLayout(self._progress_container)
        prog_layout.setContentsMargins(0, 0, 0, 0)
        prog_layout.setSpacing(2)

        self._overall_bar = QProgressBar()
        self._overall_bar.setFixedHeight(6)
        self._overall_bar.setTextVisible(False)
        self._overall_bar.setStyleSheet("""
            QProgressBar { background: #e0e8e4; border: none; border-radius: 3px; }
            QProgressBar::chunk { background: #1a4a3a; border-radius: 3px; }
        """)
        prog_layout.addWidget(self._overall_bar)

        self._progress_bar = QProgressBar()
        self._progress_bar.setFixedHeight(20)
        self._progress_bar.setFormat("%p%")
        self._progress_bar.setStyleSheet("""
            QProgressBar { background: #e0e8e4; border: none; border-radius: 3px;
                           font-size: 11px; color: #1a2e26; }
            QProgressBar::chunk { background: #4a9e7e; border-radius: 3px; }
        """)
        prog_layout.addWidget(self._progress_bar)

        self._progress_label = QLabel("")
        self._progress_label.setStyleSheet("font-size: 11px; color: #2d7a5e;")
        prog_layout.addWidget(self._progress_label)

        bottom_layout.addWidget(self._progress_container)

        # Run / Stop buttons
        btn_row = QHBoxLayout()
        self._run_btn = QPushButton("Run Pipeline")
        self._run_btn.setObjectName("runButton")
        self._run_btn.clicked.connect(self.run_pipeline)
        btn_row.addWidget(self._run_btn)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self.stop_pipeline)
        btn_row.addWidget(self._stop_btn)
        bottom_layout.addLayout(btn_row)

        # Console log
        self._console = QTextEdit()
        self._console.setReadOnly(True)
        self._console.setMaximumHeight(150)
        self._console.setPlaceholderText("Console output...")
        bottom_layout.addWidget(self._console)

        self._save_log_btn = QPushButton("Save Log")
        self._save_log_btn.setToolTip("Save console output to a text file")
        self._save_log_btn.clicked.connect(self._save_console_log)
        bottom_layout.addWidget(self._save_log_btn)

        outer_layout.addWidget(bottom)

    def _register_defaults(self) -> None:
        """Register default values for all parameter widgets (enables right-click reset)."""
        # Prepare tab
        self._register_default(self._chk_subsample, False)
        self._register_default(self._subsample_spacing, 0.01)
        # Process tab — Plot geometry
        self._register_default(self._plot_radius, 0.0)
        self._register_default(self._plot_buffer, 0.0)
        self._register_default(self._auto_centre, True)
        self._register_default(self._centre_x, 0.0)
        self._register_default(self._centre_y, 0.0)
        # Process tab — Performance
        self._register_default(self._batch_size, 2)
        self._register_default(self._cpu_cores, 0)
        self._register_default(self._cpu_only, False)
        # Process tab — Pipeline stages
        self._register_default(self._chk_preprocess, True)
        self._register_default(self._chk_segmentation, True)
        self._register_default(self._chk_postprocessing, True)
        self._register_default(self._chk_measure, True)
        self._register_default(self._chk_report, True)
        # Advanced tab — Measurement
        self._register_default(self._grid_resolution, 0.5)
        self._register_default(self._minimum_cci, 0.3)
        self._register_default(self._min_tree_cyls, 10)
        self._register_default(self._min_cluster_size, 30)
        self._register_default(self._height_percentile, 100.0)
        self._register_default(self._tree_base_cutoff, 5.0)
        self._register_default(self._ground_veg_cutoff, 3.0)
        self._register_default(self._veg_sorting_range, 1.5)
        self._register_default(self._stem_sorting_range, 1.0)
        # Advanced tab — Slicing
        self._register_default(self._slice_thickness, 0.15)
        self._register_default(self._slice_increment, 0.05)
        # Advanced tab — Taper
        self._register_default(self._taper_height_min, 0.0)
        self._register_default(self._taper_height_max, 30.0)
        self._register_default(self._taper_height_inc, 0.2)
        self._register_default(self._taper_slice, 0.4)
        # Advanced tab — Segmentation boxes
        self._register_default(self._box_dim, 6.0)
        self._register_default(self._box_overlap, 0.5)
        self._register_default(self._min_pts_box, 1000)
        self._register_default(self._max_pts_box, 20000)
        # Advanced tab — Cleanup
        self._register_default(self._chk_delete_working, True)
        self._register_default(self._chk_minimise, False)

    # ===== Tab 1: Project =====

    def _setup_project_tab(self) -> None:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(8, 8, 8, 8)

        # File input
        group = QGroupBox("Input File")
        glayout = QVBoxLayout(group)
        file_row = QHBoxLayout()
        self._file_input = QLineEdit()
        self._file_input.setPlaceholderText("Select a point cloud file...")
        self._file_input.setReadOnly(True)
        file_row.addWidget(self._file_input)
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._browse_file)
        file_row.addWidget(browse_btn)
        glayout.addLayout(file_row)
        layout.addWidget(group)

        # Project info
        group = QGroupBox("Project Info")
        form = QFormLayout(group)

        self._project_name = QLineEdit()
        self._project_name.setPlaceholderText("My Forest Plot")
        form.addRow("Project:", self._project_name)

        self._operator = QLineEdit()
        self._operator.setPlaceholderText("Operator name")
        form.addRow("Operator:", self._operator)

        self._notes = QTextEdit()
        self._notes.setMaximumHeight(60)
        self._notes.setPlaceholderText("Project notes...")
        form.addRow("Notes:", self._notes)
        layout.addWidget(group)

        # Output
        group = QGroupBox("Output")
        glayout = QVBoxLayout(group)
        dir_row = QHBoxLayout()
        self._output_dir = QLineEdit()
        self._output_dir.setPlaceholderText("Auto (next to input file)")
        dir_row.addWidget(self._output_dir)
        dir_btn = QPushButton("...")
        dir_btn.setMaximumWidth(30)
        dir_btn.clicked.connect(self._browse_output_dir)
        dir_row.addWidget(dir_btn)
        glayout.addLayout(dir_row)
        layout.addWidget(group)

        # Photos
        group = QGroupBox("Field Photos")
        glayout = QVBoxLayout(group)
        self._photos_list = QLabel("No photos attached")
        self._photos_list.setWordWrap(True)
        glayout.addWidget(self._photos_list)
        photo_btn = QPushButton("Attach Photos...")
        photo_btn.clicked.connect(self._attach_photos)
        glayout.addWidget(photo_btn)
        layout.addWidget(group)

        layout.addStretch()
        scroll.setWidget(content)
        self._tabs.addTab(scroll, "Project")

    # ===== Tab 2: Prepare =====

    def _setup_prepare_tab(self) -> None:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(8, 8, 8, 8)

        # Orientation
        group = QGroupBox("Orientation")
        glayout = QVBoxLayout(group)
        info = QLabel("Fix point cloud orientation so Z points up.")
        info.setWordWrap(True)
        glayout.addWidget(info)

        swap_row = QHBoxLayout()
        for label, mode in [("Y\u2194Z", "yz"), ("X\u2194Z", "xz"), ("X\u2194Y", "xy")]:
            btn = QPushButton(f"Swap {label}")
            btn.clicked.connect(lambda checked=False, m=mode: self.swap_axes_requested.emit(m))
            swap_row.addWidget(btn)
        glayout.addLayout(swap_row)

        swap_row2 = QHBoxLayout()
        rot_btn = QPushButton("Rotate 90\u00b0 Z")
        rot_btn.clicked.connect(lambda: self.swap_axes_requested.emit("rot90z"))
        swap_row2.addWidget(rot_btn)
        reset_btn = QPushButton("Reset Orientation")
        reset_btn.clicked.connect(lambda: self.swap_axes_requested.emit("reset"))
        swap_row2.addWidget(reset_btn)
        glayout.addLayout(swap_row2)
        layout.addWidget(group)

        # Cleaning
        group = QGroupBox("Clean")
        glayout = QVBoxLayout(group)
        info = QLabel("Remove outlier points beyond the 99.5th percentile per axis.")
        info.setWordWrap(True)
        glayout.addWidget(info)
        btn_row = QHBoxLayout()
        crop_btn = QPushButton("Crop Outliers")
        crop_btn.clicked.connect(self.crop_outliers_requested.emit)
        btn_row.addWidget(crop_btn)
        self._reset_crop_btn = QPushButton("Reset Crop")
        self._reset_crop_btn.setEnabled(False)
        self._reset_crop_btn.clicked.connect(self.reset_crop_requested.emit)
        btn_row.addWidget(self._reset_crop_btn)
        glayout.addLayout(btn_row)
        layout.addWidget(group)

        # Subsampling
        group = QGroupBox("Subsample")
        glayout = QVBoxLayout(group)
        info = QLabel("Reduce point density before processing to speed up the pipeline.")
        info.setWordWrap(True)
        glayout.addWidget(info)
        self._chk_subsample = QCheckBox("Enable subsampling")
        self._chk_subsample.setToolTip(get_tooltip("subsample"))
        glayout.addWidget(self._chk_subsample)
        form = QFormLayout()
        self._subsample_spacing = QDoubleSpinBox()
        self._subsample_spacing.setRange(0.001, 1.0)
        self._subsample_spacing.setSingleStep(0.005)
        self._subsample_spacing.setValue(0.01)
        self._subsample_spacing.setDecimals(3)
        self._subsample_spacing.setSuffix(" m")
        self._subsample_spacing.setToolTip(get_tooltip("subsampling_min_spacing"))
        form.addRow("Min spacing:", self._subsample_spacing)
        glayout.addLayout(form)
        preview_subsample_btn = QPushButton("Preview Subsample")
        preview_subsample_btn.setToolTip("Apply voxel-grid subsampling to the loaded point cloud preview")
        preview_subsample_btn.clicked.connect(
            lambda: self.subsample_requested.emit(self._subsample_spacing.value())
        )
        glayout.addWidget(preview_subsample_btn)
        layout.addWidget(group)

        # Trim
        group = QGroupBox("Trim Point Cloud")
        glayout = QVBoxLayout(group)
        info = QLabel(
            "Draw a rectangle to select a region, then choose to keep or remove those points."
        )
        info.setWordWrap(True)
        glayout.addWidget(info)

        self._trim_select_btn = QPushButton("Select Region")
        self._trim_select_btn.setToolTip("Draw a rectangle on the 3D view to select points")
        self._trim_select_btn.clicked.connect(self._on_trim_select)
        glayout.addWidget(self._trim_select_btn)

        trim_action_row = QHBoxLayout()
        self._trim_keep_btn = QPushButton("Keep Selected")
        self._trim_keep_btn.setToolTip("Keep only the selected points, remove everything else")
        self._trim_keep_btn.setEnabled(False)
        self._trim_keep_btn.clicked.connect(lambda: self.trim_apply_requested.emit(True))
        trim_action_row.addWidget(self._trim_keep_btn)

        self._trim_remove_btn = QPushButton("Remove Selected")
        self._trim_remove_btn.setToolTip("Remove the selected points, keep everything else")
        self._trim_remove_btn.setEnabled(False)
        self._trim_remove_btn.clicked.connect(lambda: self.trim_apply_requested.emit(False))
        trim_action_row.addWidget(self._trim_remove_btn)
        glayout.addLayout(trim_action_row)

        self._trim_cancel_btn = QPushButton("Cancel")
        self._trim_cancel_btn.setEnabled(False)
        self._trim_cancel_btn.clicked.connect(self._on_trim_cancel)
        glayout.addWidget(self._trim_cancel_btn)

        layout.addWidget(group)

        # Save
        group = QGroupBox("Save Prepared Cloud")
        glayout = QVBoxLayout(group)
        info = QLabel(
            "Save the modified point cloud (after orientation, crop, and/or "
            "subsample changes) as a .las file for future use."
        )
        info.setWordWrap(True)
        glayout.addWidget(info)
        save_btn = QPushButton("Save Point Cloud As...")
        save_btn.clicked.connect(self._save_prepared_cloud)
        glayout.addWidget(save_btn)
        layout.addWidget(group)

        layout.addStretch()
        scroll.setWidget(content)
        self._tabs.addTab(scroll, "Prepare")

    # ===== Tab 3: Process =====

    def _setup_process_tab(self) -> None:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(8, 8, 8, 8)

        # Plot geometry
        group = QGroupBox("Plot Geometry")
        form = QFormLayout(group)

        self._plot_radius = QDoubleSpinBox()
        self._plot_radius.setRange(0, 1000)
        self._plot_radius.setSuffix(" m")
        self._plot_radius.setToolTip(get_tooltip("plot_radius"))
        self._plot_radius.valueChanged.connect(self._on_radius_changed)
        form.addRow("Plot radius:", self._plot_radius)

        self._plot_buffer = QDoubleSpinBox()
        self._plot_buffer.setRange(0, 100)
        self._plot_buffer.setSuffix(" m")
        self._plot_buffer.setToolTip(get_tooltip("plot_radius_buffer"))
        form.addRow("Plot buffer:", self._plot_buffer)

        self._auto_centre = QCheckBox("Auto (bounding box centre)")
        self._auto_centre.setChecked(True)
        self._auto_centre.setToolTip(get_tooltip("plot_centre"))
        self._auto_centre.toggled.connect(self._on_auto_centre_toggled)
        form.addRow("Plot centre:", self._auto_centre)

        self._centre_x = QDoubleSpinBox()
        self._centre_x.setRange(-1e8, 1e8)
        self._centre_x.setDecimals(3)
        self._centre_x.setSuffix(" m")
        self._centre_x.setEnabled(False)
        self._centre_x.valueChanged.connect(self._on_centre_changed)
        form.addRow("  Centre X:", self._centre_x)

        self._centre_y = QDoubleSpinBox()
        self._centre_y.setRange(-1e8, 1e8)
        self._centre_y.setDecimals(3)
        self._centre_y.setSuffix(" m")
        self._centre_y.setEnabled(False)
        self._centre_y.valueChanged.connect(self._on_centre_changed)
        form.addRow("  Centre Y:", self._centre_y)
        layout.addWidget(group)

        # Model
        group = QGroupBox("Model")
        glayout = QVBoxLayout(group)
        form = QFormLayout()
        self._model_combo = QComboBox()
        self._scan_models()
        self._model_combo.setToolTip(get_tooltip("model_filename"))
        form.addRow("Model:", self._model_combo)
        glayout.addLayout(form)
        import_btn = QPushButton("Import Model...")
        import_btn.clicked.connect(self._import_model)
        glayout.addWidget(import_btn)
        layout.addWidget(group)

        # Performance
        group = QGroupBox("Performance")
        form = QFormLayout(group)

        self._batch_size = QSpinBox()
        self._batch_size.setRange(1, 64)
        self._batch_size.setValue(2)
        self._batch_size.setToolTip(get_tooltip("batch_size"))
        form.addRow("Batch size:", self._batch_size)

        self._cpu_cores = QSpinBox()
        self._cpu_cores.setRange(0, os.cpu_count() or 16)
        self._cpu_cores.setValue(0)
        self._cpu_cores.setToolTip(get_tooltip("num_cpu_cores"))
        form.addRow("CPU cores (0=all):", self._cpu_cores)

        self._cpu_only = QCheckBox("CPU only (no GPU)")
        self._cpu_only.setToolTip(get_tooltip("use_CPU_only"))
        form.addRow(self._cpu_only)
        layout.addWidget(group)

        # Pipeline stages
        group = QGroupBox("Pipeline Stages")
        glayout = QVBoxLayout(group)

        self._chk_preprocess = QCheckBox("Preprocessing")
        self._chk_preprocess.setChecked(True)
        glayout.addWidget(self._chk_preprocess)

        self._chk_segmentation = QCheckBox("Semantic Segmentation")
        self._chk_segmentation.setChecked(True)
        glayout.addWidget(self._chk_segmentation)

        self._chk_postprocessing = QCheckBox("Post-processing")
        self._chk_postprocessing.setChecked(True)
        glayout.addWidget(self._chk_postprocessing)

        self._chk_measure = QCheckBox("Measurement")
        self._chk_measure.setChecked(True)
        glayout.addWidget(self._chk_measure)

        self._chk_report = QCheckBox("Generate Report")
        self._chk_report.setChecked(True)
        glayout.addWidget(self._chk_report)
        layout.addWidget(group)

        layout.addStretch()
        scroll.setWidget(content)
        self._tabs.addTab(scroll, "Process")

    # ===== Tab 4: Advanced =====

    def _setup_advanced_tab(self) -> None:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(8, 8, 8, 8)

        # Measurement parameters
        group = QGroupBox("Measurement")
        form = QFormLayout(group)

        self._grid_resolution = QDoubleSpinBox()
        self._grid_resolution.setRange(0.1, 5.0)
        self._grid_resolution.setSingleStep(0.1)
        self._grid_resolution.setValue(0.5)
        self._grid_resolution.setSuffix(" m")
        self._grid_resolution.setToolTip(get_tooltip("grid_resolution"))
        form.addRow("DTM resolution:", self._grid_resolution)

        self._minimum_cci = QDoubleSpinBox()
        self._minimum_cci.setRange(0.0, 1.0)
        self._minimum_cci.setSingleStep(0.05)
        self._minimum_cci.setValue(0.3)
        self._minimum_cci.setToolTip(get_tooltip("minimum_CCI"))
        form.addRow("Min CCI:", self._minimum_cci)

        self._min_tree_cyls = QSpinBox()
        self._min_tree_cyls.setRange(1, 100)
        self._min_tree_cyls.setValue(10)
        self._min_tree_cyls.setToolTip(get_tooltip("min_tree_cyls"))
        form.addRow("Min tree cylinders:", self._min_tree_cyls)

        self._min_cluster_size = QSpinBox()
        self._min_cluster_size.setRange(5, 500)
        self._min_cluster_size.setValue(30)
        self._min_cluster_size.setToolTip(get_tooltip("min_cluster_size"))
        form.addRow("Min cluster size:", self._min_cluster_size)

        self._height_percentile = QDoubleSpinBox()
        self._height_percentile.setRange(90, 100)
        self._height_percentile.setSingleStep(0.5)
        self._height_percentile.setValue(100)
        self._height_percentile.setToolTip(get_tooltip("height_percentile"))
        form.addRow("Height percentile:", self._height_percentile)

        self._tree_base_cutoff = QDoubleSpinBox()
        self._tree_base_cutoff.setRange(0, 30)
        self._tree_base_cutoff.setSingleStep(0.5)
        self._tree_base_cutoff.setValue(5)
        self._tree_base_cutoff.setSuffix(" m")
        self._tree_base_cutoff.setToolTip(get_tooltip("tree_base_cutoff_height"))
        form.addRow("Tree base cutoff:", self._tree_base_cutoff)

        self._ground_veg_cutoff = QDoubleSpinBox()
        self._ground_veg_cutoff.setRange(0, 20)
        self._ground_veg_cutoff.setSingleStep(0.5)
        self._ground_veg_cutoff.setValue(3)
        self._ground_veg_cutoff.setSuffix(" m")
        self._ground_veg_cutoff.setToolTip(get_tooltip("ground_veg_cutoff_height"))
        form.addRow("Ground veg cutoff:", self._ground_veg_cutoff)

        self._veg_sorting_range = QDoubleSpinBox()
        self._veg_sorting_range.setRange(0.1, 10)
        self._veg_sorting_range.setSingleStep(0.1)
        self._veg_sorting_range.setValue(1.5)
        self._veg_sorting_range.setSuffix(" m")
        self._veg_sorting_range.setToolTip(get_tooltip("veg_sorting_range"))
        form.addRow("Veg sorting range:", self._veg_sorting_range)

        self._stem_sorting_range = QDoubleSpinBox()
        self._stem_sorting_range.setRange(0.1, 10)
        self._stem_sorting_range.setSingleStep(0.1)
        self._stem_sorting_range.setValue(1.0)
        self._stem_sorting_range.setSuffix(" m")
        self._stem_sorting_range.setToolTip(get_tooltip("stem_sorting_range"))
        form.addRow("Stem sorting range:", self._stem_sorting_range)
        layout.addWidget(group)

        # Slicing
        group = QGroupBox("Slicing")
        form = QFormLayout(group)

        self._slice_thickness = QDoubleSpinBox()
        self._slice_thickness.setRange(0.01, 1.0)
        self._slice_thickness.setSingleStep(0.05)
        self._slice_thickness.setValue(0.15)
        self._slice_thickness.setSuffix(" m")
        self._slice_thickness.setToolTip(get_tooltip("slice_thickness"))
        form.addRow("Slice thickness:", self._slice_thickness)

        self._slice_increment = QDoubleSpinBox()
        self._slice_increment.setRange(0.01, 0.5)
        self._slice_increment.setSingleStep(0.01)
        self._slice_increment.setValue(0.05)
        self._slice_increment.setSuffix(" m")
        self._slice_increment.setToolTip(get_tooltip("slice_increment"))
        form.addRow("Slice increment:", self._slice_increment)
        layout.addWidget(group)

        # Taper
        group = QGroupBox("Taper Measurement")
        form = QFormLayout(group)

        self._taper_height_min = QDoubleSpinBox()
        self._taper_height_min.setRange(0, 50)
        self._taper_height_min.setSingleStep(0.5)
        self._taper_height_min.setValue(0)
        self._taper_height_min.setSuffix(" m")
        self._taper_height_min.setToolTip(get_tooltip("taper_measurement_height_min"))
        form.addRow("Min height:", self._taper_height_min)

        self._taper_height_max = QDoubleSpinBox()
        self._taper_height_max.setRange(0, 100)
        self._taper_height_max.setSingleStep(1)
        self._taper_height_max.setValue(30)
        self._taper_height_max.setSuffix(" m")
        self._taper_height_max.setToolTip(get_tooltip("taper_measurement_height_max"))
        form.addRow("Max height:", self._taper_height_max)

        self._taper_height_inc = QDoubleSpinBox()
        self._taper_height_inc.setRange(0.05, 2.0)
        self._taper_height_inc.setSingleStep(0.1)
        self._taper_height_inc.setValue(0.2)
        self._taper_height_inc.setSuffix(" m")
        self._taper_height_inc.setToolTip(get_tooltip("taper_measurement_height_increment"))
        form.addRow("Height increment:", self._taper_height_inc)

        self._taper_slice = QDoubleSpinBox()
        self._taper_slice.setRange(0.1, 2.0)
        self._taper_slice.setSingleStep(0.1)
        self._taper_slice.setValue(0.4)
        self._taper_slice.setSuffix(" m")
        self._taper_slice.setToolTip(get_tooltip("taper_slice_thickness"))
        form.addRow("Slice thickness:", self._taper_slice)
        layout.addWidget(group)

        # Model / Box parameters
        group = QGroupBox("Segmentation Boxes")
        form = QFormLayout(group)

        self._box_dim = QDoubleSpinBox()
        self._box_dim.setRange(1, 20)
        self._box_dim.setSingleStep(1)
        self._box_dim.setValue(6)
        self._box_dim.setSuffix(" m")
        self._box_dim.setToolTip(get_tooltip("box_dimensions"))
        form.addRow("Box size:", self._box_dim)

        self._box_overlap = QDoubleSpinBox()
        self._box_overlap.setRange(0.0, 0.9)
        self._box_overlap.setSingleStep(0.1)
        self._box_overlap.setValue(0.5)
        self._box_overlap.setToolTip(get_tooltip("box_overlap"))
        form.addRow("Box overlap:", self._box_overlap)

        self._min_pts_box = QSpinBox()
        self._min_pts_box.setRange(10, 10000)
        self._min_pts_box.setSingleStep(100)
        self._min_pts_box.setValue(1000)
        self._min_pts_box.setToolTip(get_tooltip("min_points_per_box"))
        form.addRow("Min points/box:", self._min_pts_box)

        self._max_pts_box = QSpinBox()
        self._max_pts_box.setRange(1000, 100000)
        self._max_pts_box.setSingleStep(1000)
        self._max_pts_box.setValue(20000)
        self._max_pts_box.setToolTip(get_tooltip("max_points_per_box"))
        form.addRow("Max points/box:", self._max_pts_box)
        layout.addWidget(group)

        # Cleanup options
        group = QGroupBox("Cleanup")
        form = QFormLayout(group)
        self._chk_delete_working = QCheckBox("Delete working directory after run")
        self._chk_delete_working.setChecked(True)
        self._chk_delete_working.setToolTip(get_tooltip("delete_working_directory"))
        form.addRow(self._chk_delete_working)
        self._chk_minimise = QCheckBox("Minimise output size")
        self._chk_minimise.setToolTip(get_tooltip("minimise_output_size_mode"))
        form.addRow(self._chk_minimise)
        layout.addWidget(group)

        layout.addStretch()
        scroll.setWidget(content)
        self._tabs.addTab(scroll, "Advanced")

    # ===== Tab 5: Results =====

    # Output layer definitions: (label, filename, tooltip)
    OUTPUT_LAYERS = [
        ("DTM", "DTM.las", "Digital Terrain Model — interpolated ground surface"),
        ("Cropped DTM", "cropped_DTM.las", "DTM cropped to plot radius (tree-aware mode)"),
        ("Terrain Points", "terrain_points.las", "Points classified as ground/terrain"),
        ("Vegetation Points", "vegetation_points.las", "Points classified as vegetation"),
        ("CWD Points", "cwd_points.las", "Coarse Woody Debris points (may be empty if none detected)"),
        ("Stem Points", "stem_points.las", "Points classified as tree stems"),
        ("Ground Vegetation", "ground_veg.las", "Vegetation below the canopy cutoff height"),
        ("Segmented", "segmented.las", "Full point cloud with semantic labels"),
        ("Segmented Cleaned", "segmented_cleaned.las", "Cleaned segmentation output"),
        ("Stem Points Sorted", "stem_points_sorted.las", "Stem points sorted by tree ID"),
        ("Veg Points Sorted", "veg_points_sorted.las", "Vegetation points sorted by tree ID"),
        ("Skeleton Clusters", "skeleton_cluster_visualisation.las", "Tree skeleton cluster visualization"),
        ("Cylinder Model", "full_cyl_array.las", "Fitted cylinder array for all trees"),
        ("Sorted Cylinders", "sorted_full_cyl_array.las", "Cylinder array sorted by tree ID"),
        ("Cleaned Cylinders", "cleaned_cyls.las", "Cleaned cylinders after filtering"),
        ("Cleaned Cyl Vis", "cleaned_cyl_vis.las", "Cleaned cylinder visualization"),
        ("Interpolated Cylinders", "interpolated_full_cyl_array.las", "Gap-filled interpolated cylinder array"),
        ("Text Labels", "text_point_cloud.las", "3D text labels for tree IDs"),
        ("Tree-Aware Crop", "tree_aware_cropped_point_cloud.las", "Point cloud cropped with tree-aware buffer"),
    ]

    def _setup_results_tab(self) -> None:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(8, 8, 8, 8)

        # --- Pipeline Run Selector ---
        group = QGroupBox("Pipeline Run History")
        group.setToolTip(
            "Browse previous pipeline runs for this project.\n"
            "Select a run to view its output layers, tree data, and report."
        )
        glayout = QVBoxLayout(group)
        run_info = QLabel("Select a previous run to compare results:")
        run_info.setWordWrap(True)
        glayout.addWidget(run_info)
        self._run_combo = QComboBox()
        self._run_combo.setPlaceholderText("Save a project and run the pipeline to see runs here")
        self._run_combo.setToolTip("Each pipeline run is saved with a timestamp. Select one to load its results.")
        self._run_combo.currentIndexChanged.connect(self._on_run_selected)
        glayout.addWidget(self._run_combo)
        layout.addWidget(group)

        # --- Output Layers ---
        group = QGroupBox("Output Layers")
        glayout = QVBoxLayout(group)
        info = QLabel("Select output layers to load into the viewer.")
        info.setWordWrap(True)
        glayout.addWidget(info)

        self._layer_checkboxes: list[tuple[QCheckBox, str]] = []
        for label, filename, tooltip in self.OUTPUT_LAYERS:
            cb = QCheckBox(label)
            cb.setToolTip(tooltip)
            cb.setEnabled(False)
            self._layer_checkboxes.append((cb, filename))
            glayout.addWidget(cb)

        btn_row = QHBoxLayout()
        self._select_all_layers_btn = QPushButton("Select All")
        self._select_all_layers_btn.setEnabled(False)
        self._select_all_layers_btn.clicked.connect(self._select_all_layers)
        btn_row.addWidget(self._select_all_layers_btn)

        self._load_layers_btn = QPushButton("Load Layers")
        self._load_layers_btn.setEnabled(False)
        self._load_layers_btn.clicked.connect(self._load_selected_layers)
        btn_row.addWidget(self._load_layers_btn)
        glayout.addLayout(btn_row)
        layout.addWidget(group)

        # --- Tree Measurements ---
        group = QGroupBox("Tree Measurements")
        glayout = QVBoxLayout(group)
        self._tree_table_model = PandasTableModel()
        self._tree_table = QTableView()
        self._tree_table.setModel(self._tree_table_model)
        self._tree_table.setAlternatingRowColors(True)
        self._tree_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._tree_table.horizontalHeader().setStretchLastSection(True)
        self._tree_table.horizontalHeader().setMinimumSectionSize(60)
        self._tree_table.setMinimumHeight(180)
        self._tree_table.setSelectionBehavior(QTableView.SelectRows)
        self._tree_table.clicked.connect(self._on_tree_row_clicked)
        glayout.addWidget(self._tree_table)

        self._export_tree_btn = QPushButton("Export Tree Data...")
        self._export_tree_btn.setEnabled(False)
        self._export_tree_btn.clicked.connect(self._export_tree_data)
        glayout.addWidget(self._export_tree_btn)
        layout.addWidget(group)

        # --- Report ---
        group = QGroupBox("Report")
        glayout = QVBoxLayout(group)
        report_row = QHBoxLayout()
        self._open_report_btn = QPushButton("Open Report")
        self._open_report_btn.setEnabled(False)
        self._open_report_btn.clicked.connect(self._open_report)
        report_row.addWidget(self._open_report_btn)
        self._export_pdf_btn = QPushButton("Export PDF")
        self._export_pdf_btn.setEnabled(False)
        self._export_pdf_btn.clicked.connect(self._export_pdf)
        report_row.addWidget(self._export_pdf_btn)
        glayout.addLayout(report_row)
        layout.addWidget(group)

        # --- Compare Runs ---
        group = QGroupBox("Analysis")
        glayout = QVBoxLayout(group)
        self._compare_runs_btn = QPushButton("Compare Runs...")
        self._compare_runs_btn.setToolTip("Compare tree measurements across two pipeline runs")
        self._compare_runs_btn.setEnabled(False)
        self._compare_runs_btn.clicked.connect(self._compare_runs)
        glayout.addWidget(self._compare_runs_btn)

        self._gis_export_btn = QPushButton("Export to GIS...")
        self._gis_export_btn.setToolTip("Export tree data as GeoJSON or Shapefile")
        self._gis_export_btn.setEnabled(False)
        self._gis_export_btn.clicked.connect(self._export_gis)
        glayout.addWidget(self._gis_export_btn)
        layout.addWidget(group)

        layout.addStretch()
        scroll.setWidget(content)
        self._tabs.addTab(scroll, "Results")

        self._results_output_dir: Optional[str] = None

    def _select_all_layers(self) -> None:
        for cb, _ in self._layer_checkboxes:
            if cb.isEnabled():
                cb.setChecked(True)

    def _load_selected_layers(self) -> None:
        if not self._results_output_dir:
            QMessageBox.information(
                self, "No Run Selected",
                "Please select a pipeline run from the Run History dropdown first."
            )
            return
        paths = []
        for cb, filename in self._layer_checkboxes:
            if cb.isChecked() and cb.isEnabled():
                paths.append(os.path.join(self._results_output_dir, filename))
        if paths:
            self.load_output_layers.emit(paths)
        else:
            QMessageBox.information(self, "No Layers", "Please select at least one output layer.")

    def _export_tree_data(self) -> None:
        if self._tree_table_model._df is None:
            return
        # Default save location: reports/ folder for the current run
        default_path = ""
        reports_dir = self._get_reports_dir()
        if reports_dir:
            default_path = os.path.join(reports_dir, "tree_data.csv")
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Tree Data", default_path, "CSV Files (*.csv);;All Files (*)",
        )
        if filepath:
            if not filepath.endswith(".csv"):
                filepath += ".csv"
            self._tree_table_model._df.to_csv(filepath, index=False)
            self._log(f"Tree data exported to: {filepath}")

    def _on_tree_row_clicked(self, index) -> None:
        """When a row is clicked in the tree table, emit the tree ID."""
        if self._tree_table_model._df is None:
            return
        row = index.row()
        df = self._tree_table_model._df
        if "TreeId" in df.columns:
            tree_id = int(df.iloc[row]["TreeId"])
            self.tree_selected.emit(tree_id)

    def _open_report(self) -> None:
        if not self._results_output_dir:
            return
        # Look in reports/ first (files are moved there), fall back to output/
        reports_dir = self._get_reports_dir()
        report_path = None
        for candidate_dir in [reports_dir, self._results_output_dir]:
            if candidate_dir:
                p = os.path.join(candidate_dir, "Plot_Report.html")
                if os.path.exists(p):
                    report_path = p
                    break
        if report_path:
            import webbrowser
            webbrowser.open(f"file://{report_path}")
        else:
            QMessageBox.warning(self, "No Report", "Report file not found.")

    def _get_reports_dir(self) -> str | None:
        """Return the reports/ folder for the current run, or None."""
        if not self._results_output_dir:
            return None
        reports_dir = os.path.join(os.path.dirname(self._results_output_dir), "reports")
        if os.path.isdir(reports_dir):
            return reports_dir
        return None

    def _export_pdf(self) -> None:
        if not self._results_output_dir:
            return
        # Look for the HTML report in reports/ first, fall back to output/
        reports_dir = self._get_reports_dir()
        html_path = None
        for candidate_dir in [reports_dir, self._results_output_dir]:
            if candidate_dir:
                p = os.path.join(candidate_dir, "Plot_Report.html")
                if os.path.exists(p):
                    html_path = p
                    break
        if not html_path:
            QMessageBox.warning(self, "No Report", "HTML report not found. Run the pipeline first.")
            return
        # Save PDF to reports/ folder if available, otherwise next to HTML
        dest_dir = reports_dir or self._results_output_dir
        pdf_path = os.path.join(dest_dir, "Plot_Report.pdf")
        try:
            from understory.core.report import export_pdf
            export_pdf(html_path, pdf_path)
            self._log(f"PDF exported: {pdf_path}")
            import webbrowser
            webbrowser.open(f"file://{pdf_path}")
        except Exception as e:
            QMessageBox.critical(self, "PDF Export Error", f"Failed to export PDF: {e}")

    def _compare_runs(self) -> None:
        """Open comparison dialog for two pipeline runs."""
        if self._run_combo.count() < 2:
            QMessageBox.information(self, "Need Two Runs", "At least two pipeline runs are needed for comparison.")
            return
        try:
            from understory.core.comparison import compare_runs, generate_comparison_report
            # Use current run as run_b, previous run as run_a
            idx_b = self._run_combo.currentIndex()
            idx_a = idx_b - 1 if idx_b > 0 else idx_b + 1
            dir_a = self._run_combo.itemData(idx_a)
            dir_b = self._run_combo.itemData(idx_b)
            if not dir_a or not dir_b:
                return
            report_path = generate_comparison_report(dir_a, dir_b)
            if report_path:
                import webbrowser
                webbrowser.open(f"file://{report_path}")
                self._log(f"Comparison report: {report_path}")
            else:
                QMessageBox.warning(self, "Comparison Failed", "Could not generate comparison report.")
        except Exception as e:
            QMessageBox.critical(self, "Comparison Error", f"Failed to compare runs: {e}")

    def _export_gis(self) -> None:
        """Export tree data to GIS format."""
        if not self._results_output_dir:
            return
        tree_csv = os.path.join(self._results_output_dir, "tree_data.csv")
        if not os.path.exists(tree_csv):
            QMessageBox.warning(self, "No Data", "No tree_data.csv found in the output directory.")
            return
        filepath, selected_filter = QFileDialog.getSaveFileName(
            self, "Export to GIS", "",
            "GeoJSON (*.geojson);;Shapefile (*.shp);;CSV with Coordinates (*.csv);;All Files (*)",
        )
        if not filepath:
            return
        try:
            import pandas as pd
            tree_data = pd.read_csv(tree_csv)
            if filepath.endswith(".geojson") or "GeoJSON" in selected_filter:
                from understory.core.gis_export import export_geojson
                if not filepath.endswith(".geojson"):
                    filepath += ".geojson"
                export_geojson(tree_data, filepath)
            elif filepath.endswith(".shp") or "Shapefile" in selected_filter:
                from understory.core.gis_export import export_shapefile
                if not filepath.endswith(".shp"):
                    filepath += ".shp"
                export_shapefile(tree_data, filepath)
            else:
                from understory.core.gis_export import export_csv_with_coords
                if not filepath.endswith(".csv"):
                    filepath += ".csv"
                export_csv_with_coords(tree_data, filepath)
            self._log(f"GIS export saved: {filepath}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export GIS data: {e}")

    def _populate_runs(self) -> None:
        """Populate the run selector combo from the project's run history."""
        self._run_combo.blockSignals(True)
        self._run_combo.clear()
        if self._last_save_path:
            from understory.core.paths import ProjectPaths
            project_dir = Path(self._last_save_path).parent
            project_paths = ProjectPaths(project_dir)
            runs = project_paths.list_runs()
            for run_dir in runs:
                output_dir = str(ProjectPaths.run_output_dir(run_dir))
                # Display as run timestamp (strip "run_" prefix)
                display = run_dir.name.replace("run_", "").replace("_", " ", 1).replace("-", ":", 3)
                self._run_combo.addItem(display, userData=output_dir)
        self._run_combo.blockSignals(False)

    def _on_run_selected(self, index: int) -> None:
        """When a run is selected from the combo, refresh results and restore settings."""
        if index < 0:
            return
        output_dir = self._run_combo.itemData(index)
        if output_dir and os.path.isdir(output_dir):
            self._populate_results(output_dir)
            # Restore settings from the run's config snapshot
            run_dir = Path(output_dir).parent
            run_config = run_dir / "run_config.yaml"
            if run_config.exists():
                try:
                    config = ProjectConfig.load(str(run_config))
                    self._apply_config(config)
                    self._log(f"Settings restored from {run_dir.name}")
                except Exception:
                    pass  # silently skip if config is corrupt

    def _populate_results(self, output_dir: str) -> None:
        """Enable results tab controls based on available output files."""
        self._results_output_dir = output_dir

        # Enable layer checkboxes for existing files
        any_layer = False
        for cb, filename in self._layer_checkboxes:
            filepath = os.path.join(output_dir, filename)
            exists = os.path.exists(filepath)
            cb.setEnabled(exists)
            if not exists:
                cb.setChecked(False)
            if exists:
                any_layer = True

        self._select_all_layers_btn.setEnabled(any_layer)
        self._load_layers_btn.setEnabled(any_layer)

        # Load tree data CSV
        tree_csv = os.path.join(output_dir, "tree_data.csv")
        if os.path.exists(tree_csv):
            try:
                import pandas as pd
                df = pd.read_csv(tree_csv)
                self._tree_table_model.set_dataframe(df)
                self._export_tree_btn.setEnabled(True)
            except Exception as e:
                self._log(f"Could not load tree data: {e}")

        # Enable report buttons — check reports/ folder first, then output/
        report_exists = False
        reports_dir = self._get_reports_dir()
        for candidate_dir in [reports_dir, output_dir]:
            if candidate_dir and os.path.exists(os.path.join(candidate_dir, "Plot_Report.html")):
                report_exists = True
                break
        self._open_report_btn.setEnabled(report_exists)
        self._export_pdf_btn.setEnabled(report_exists)

        self._compare_runs_btn.setEnabled(self._run_combo.count() >= 2)
        self._gis_export_btn.setEnabled(self._export_tree_btn.isEnabled())

        # Switch to Results tab
        self._tabs.setCurrentIndex(self._tabs.count() - 1)

    # --- Actions ---

    def _browse_file(self) -> None:
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Open Point Cloud",
            "",
            "Point Clouds (*.las *.laz *.pcd);;All Files (*)",
        )
        if filepath:
            self.set_input_file(filepath)

    def set_input_file(self, filepath: str) -> None:
        self._file_input.setText(filepath)
        self._config.point_cloud_filename = filepath
        self.file_loaded.emit(filepath)

    def _browse_output_dir(self) -> None:
        dirpath = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dirpath:
            self._output_dir.setText(dirpath)

    def _attach_photos(self) -> None:
        """Attach field photos to the project."""
        filepaths, _ = QFileDialog.getOpenFileNames(
            self, "Select Field Photos", "",
            "Images (*.jpg *.jpeg *.png *.tiff *.bmp);;All Files (*)",
        )
        if filepaths:
            self._config.photos = filepaths
            self._photos_list.setText(f"{len(filepaths)} photo(s) attached")
            self._log(f"Attached {len(filepaths)} field photo(s)")

    def _on_auto_centre_toggled(self, checked: bool) -> None:
        self._centre_x.setEnabled(not checked)
        self._centre_y.setEnabled(not checked)
        if checked:
            self.plot_centre_changed.emit(None)
        else:
            self._on_centre_changed()

    def _on_centre_changed(self) -> None:
        if not self._auto_centre.isChecked():
            self.plot_centre_changed.emit((self._centre_x.value(), self._centre_y.value()))

    def _on_radius_changed(self) -> None:
        """Re-emit plot centre when radius changes so the circle updates."""
        if self._auto_centre.isChecked():
            self.plot_centre_changed.emit(None)
        else:
            self.plot_centre_changed.emit((self._centre_x.value(), self._centre_y.value()))

    def set_plot_centre(self, x: float, y: float) -> None:
        """Set plot centre from external source (e.g. viewer drag widget)."""
        self._updating_from_viewer = True
        self._auto_centre.setChecked(False)
        self._centre_x.setValue(x)
        self._centre_y.setValue(y)
        self._updating_from_viewer = False

    def _on_trim_select(self) -> None:
        """Start rectangle selection for trimming."""
        self._trim_select_btn.setEnabled(False)
        self._trim_cancel_btn.setEnabled(True)
        self.trim_select_requested.emit()

    def _on_trim_cancel(self) -> None:
        """Cancel trim selection."""
        self._trim_select_btn.setEnabled(True)
        self._trim_keep_btn.setEnabled(False)
        self._trim_remove_btn.setEnabled(False)
        self._trim_cancel_btn.setEnabled(False)
        self.trim_cancel_requested.emit()

    def on_trim_region_selected(self) -> None:
        """Called when the viewer finishes rectangle selection for trim."""
        self._trim_keep_btn.setEnabled(True)
        self._trim_remove_btn.setEnabled(True)

    def on_trim_applied(self) -> None:
        """Called after a trim operation completes — reset button states."""
        self._trim_select_btn.setEnabled(True)
        self._trim_keep_btn.setEnabled(False)
        self._trim_remove_btn.setEnabled(False)
        self._trim_cancel_btn.setEnabled(False)

    def _save_prepared_cloud(self) -> None:
        # Default to project folder if a project has been saved
        default_dir = ""
        if self._last_save_path:
            default_dir = str(Path(self._last_save_path).parent)
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Save Prepared Point Cloud",
            default_dir,
            "LAS Files (*.las);;LAZ Compressed (*.laz);;All Files (*)",
        )
        if filepath:
            if not filepath.endswith((".las", ".laz")):
                filepath += ".las"
            self._prepared_cloud_path = filepath
            self.save_cloud_requested.emit(filepath)

    def _scan_models(self) -> None:
        self._model_combo.clear()
        model_dir = Path(__file__).parent.parent.parent.parent / "model"
        if model_dir.exists():
            for pth in sorted(model_dir.glob("*.pth")):
                self._model_combo.addItem(pth.name)
        if self._model_combo.count() == 0:
            self._model_combo.addItem("model.pth")

    def _import_model(self) -> None:
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Import Model", "", "PyTorch Models (*.pth);;All Files (*)"
        )
        if filepath:
            import shutil
            model_dir = Path(__file__).parent.parent.parent.parent / "model"
            model_dir.mkdir(exist_ok=True)
            dest = model_dir / Path(filepath).name
            shutil.copy2(filepath, dest)
            self._scan_models()
            idx = self._model_combo.findText(dest.name)
            if idx >= 0:
                self._model_combo.setCurrentIndex(idx)

    # --- Config build/apply ---

    def _build_config(self) -> ProjectConfig:
        """Build ProjectConfig from current UI state."""
        config = ProjectConfig()
        config.point_cloud_filename = self._file_input.text()
        config.prepared_point_cloud = self._prepared_cloud_path or ""
        config.project_name = self._project_name.text()
        config.operator = self._operator.text()
        config.notes = self._notes.toPlainText()
        config.photos = getattr(self._config, 'photos', [])

        config.preprocess = self._chk_preprocess.isChecked()
        config.segmentation = self._chk_segmentation.isChecked()
        config.postprocessing = self._chk_postprocessing.isChecked()
        config.measure_plot = self._chk_measure.isChecked()
        config.make_report = self._chk_report.isChecked()

        # Processing
        config.processing.plot_radius = self._plot_radius.value()
        config.processing.plot_radius_buffer = self._plot_buffer.value()
        if self._auto_centre.isChecked():
            config.processing.plot_centre = None
        else:
            config.processing.plot_centre = [self._centre_x.value(), self._centre_y.value()]
        config.processing.slice_thickness = self._slice_thickness.value()
        config.processing.slice_increment = self._slice_increment.value()
        config.processing.batch_size = self._batch_size.value()
        config.processing.num_cpu_cores = self._cpu_cores.value()
        config.processing.use_CPU_only = self._cpu_only.isChecked()
        config.processing.subsample = self._chk_subsample.isChecked()
        config.processing.subsampling_min_spacing = self._subsample_spacing.value()
        config.processing.delete_working_directory = self._chk_delete_working.isChecked()
        config.processing.minimise_output_size_mode = self._chk_minimise.isChecked()

        # Measurement (in MeasurementConfig)
        config.measurement.grid_resolution = self._grid_resolution.value()
        config.measurement.minimum_CCI = self._minimum_cci.value()
        config.measurement.min_tree_cyls = self._min_tree_cyls.value()
        config.measurement.min_cluster_size = self._min_cluster_size.value()

        # Measurement-related fields in ProcessingConfig
        config.processing.height_percentile = self._height_percentile.value()
        config.processing.tree_base_cutoff_height = self._tree_base_cutoff.value()
        config.processing.ground_veg_cutoff_height = self._ground_veg_cutoff.value()
        config.processing.veg_sorting_range = self._veg_sorting_range.value()
        config.processing.stem_sorting_range = self._stem_sorting_range.value()

        # Taper
        config.processing.taper_measurement_height_min = self._taper_height_min.value()
        config.processing.taper_measurement_height_max = self._taper_height_max.value()
        config.processing.taper_measurement_height_increment = self._taper_height_inc.value()
        config.processing.taper_slice_thickness = self._taper_slice.value()

        # Model
        config.model.model_filename = self._model_combo.currentText()
        box_dim = self._box_dim.value()
        config.model.box_dimensions = [box_dim, box_dim, box_dim]
        overlap = self._box_overlap.value()
        config.model.box_overlap = [overlap, overlap, overlap]
        config.model.min_points_per_box = self._min_pts_box.value()
        config.model.max_points_per_box = self._max_pts_box.value()

        # Output
        out_dir = self._output_dir.text().strip()
        if out_dir:
            config.output.output_directory = out_dir

        return config

    def _apply_config(self, config: ProjectConfig) -> None:
        """Apply a ProjectConfig to the UI widgets."""
        self._file_input.setText(config.point_cloud_filename)
        self._prepared_cloud_path = config.prepared_point_cloud or None
        self._project_name.setText(config.project_name)
        self._operator.setText(config.operator)
        self._notes.setPlainText(config.notes)

        if config.photos:
            self._photos_list.setText(f"{len(config.photos)} photo(s) attached")
            self._config.photos = config.photos

        self._chk_preprocess.setChecked(config.preprocess)
        self._chk_segmentation.setChecked(config.segmentation)
        self._chk_postprocessing.setChecked(config.postprocessing)
        self._chk_measure.setChecked(config.measure_plot)
        self._chk_report.setChecked(config.make_report)

        # Processing
        self._plot_radius.setValue(config.processing.plot_radius)
        self._plot_buffer.setValue(config.processing.plot_radius_buffer)
        if config.processing.plot_centre is not None:
            self._auto_centre.setChecked(False)
            self._centre_x.setValue(config.processing.plot_centre[0])
            self._centre_y.setValue(config.processing.plot_centre[1])
        else:
            self._auto_centre.setChecked(True)
        self._slice_thickness.setValue(config.processing.slice_thickness)
        self._slice_increment.setValue(config.processing.slice_increment)
        self._batch_size.setValue(config.processing.batch_size)
        self._cpu_cores.setValue(config.processing.num_cpu_cores)
        self._cpu_only.setChecked(config.processing.use_CPU_only)
        self._chk_subsample.setChecked(config.processing.subsample)
        self._subsample_spacing.setValue(config.processing.subsampling_min_spacing)
        self._chk_delete_working.setChecked(config.processing.delete_working_directory)
        self._chk_minimise.setChecked(config.processing.minimise_output_size_mode)

        # Measurement (MeasurementConfig)
        self._grid_resolution.setValue(config.measurement.grid_resolution)
        self._minimum_cci.setValue(config.measurement.minimum_CCI)
        self._min_tree_cyls.setValue(config.measurement.min_tree_cyls)
        self._min_cluster_size.setValue(config.measurement.min_cluster_size)

        # Measurement-related (ProcessingConfig)
        self._height_percentile.setValue(config.processing.height_percentile)
        self._tree_base_cutoff.setValue(config.processing.tree_base_cutoff_height)
        self._ground_veg_cutoff.setValue(config.processing.ground_veg_cutoff_height)
        self._veg_sorting_range.setValue(config.processing.veg_sorting_range)
        self._stem_sorting_range.setValue(config.processing.stem_sorting_range)

        # Taper
        self._taper_height_min.setValue(config.processing.taper_measurement_height_min)
        self._taper_height_max.setValue(config.processing.taper_measurement_height_max)
        self._taper_height_inc.setValue(config.processing.taper_measurement_height_increment)
        self._taper_slice.setValue(config.processing.taper_slice_thickness)

        # Model
        idx = self._model_combo.findText(config.model.model_filename)
        if idx >= 0:
            self._model_combo.setCurrentIndex(idx)
        if config.model.box_dimensions:
            self._box_dim.setValue(config.model.box_dimensions[0])
        if config.model.box_overlap:
            self._box_overlap.setValue(config.model.box_overlap[0])
        self._min_pts_box.setValue(config.model.min_points_per_box)
        self._max_pts_box.setValue(config.model.max_points_per_box)

        # Output
        if config.output.output_directory:
            self._output_dir.setText(config.output.output_directory)

    # --- Pipeline execution ---

    def run_pipeline(self) -> None:
        if self._worker and self._worker.isRunning():
            QMessageBox.warning(self, "Pipeline Running", "A pipeline is already running.")
            return

        filepath = self._file_input.text()
        if not filepath or not os.path.exists(filepath):
            QMessageBox.warning(self, "No Input", "Please select a valid point cloud file.")
            return

        # Validate project name
        if not self._project_name.text().strip():
            reply = QMessageBox.question(
                self,
                "Project Name Missing",
                "No project name has been set. The output folder will use the "
                "input filename.\n\nContinue anyway?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.No:
                self._project_name.setFocus()
                return

        # Offer to save project before running
        reply = QMessageBox.question(
            self,
            "Save Project?",
            "Would you like to save the project configuration before running?",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            QMessageBox.Yes,
        )
        if reply == QMessageBox.Cancel:
            return
        if reply == QMessageBox.Yes:
            if self._last_save_path:
                # Quick save to known location
                self.save_project(self._last_save_path)
            else:
                # No previous save — show file dialog
                save_path, _ = QFileDialog.getSaveFileName(
                    self, "Save Project", "", "Understory Projects (*.yaml);;All Files (*)",
                )
                if save_path:
                    if not save_path.endswith((".yaml", ".yml")):
                        save_path += ".yaml"
                    self.save_project(save_path)
                else:
                    return  # User cancelled save dialog

        # Clear any stale run-specific output path before building config
        # so _build_config() doesn't carry over a previous run's directory
        if self._last_save_path and self._output_dir.text().strip():
            # Check if it's a run-specific auto path (contains /runs/)
            if "/runs/" in self._output_dir.text():
                self._output_dir.clear()

        config = self._build_config()

        # Use prepared (subsampled/cropped) cloud as pipeline input when available
        if self._prepared_cloud_path and os.path.exists(self._prepared_cloud_path):
            config.point_cloud_filename = self._prepared_cloud_path
            self._log(f"Using prepared cloud: {os.path.basename(self._prepared_cloud_path)}")

        # If project has been saved, create a timestamped run folder
        if self._last_save_path:
            from understory.core.paths import ProjectPaths
            project_dir = Path(self._last_save_path).parent
            project_paths = ProjectPaths(project_dir)
            if project_paths.runs_dir.exists() or project_paths.config_file.exists():
                run_dir = project_paths.create_run()
                run_output = str(ProjectPaths.run_output_dir(run_dir))
                config.output.output_directory = run_output
                # Don't set _output_dir text — it's for user overrides only.
                # The auto-generated run path is transient per run.
                # Save a config snapshot for this run
                config.save(str(run_dir / "run_config.yaml"))
                self._log(f"Run folder: {run_dir.name}")

        self._console.clear()
        self._log("Starting pipeline...")

        self._progress_container.setVisible(True)
        self._overall_bar.setValue(0)
        self._progress_bar.setValue(0)
        self._progress_label.setText("Starting...")
        self._run_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)

        self._worker = PipelineWorker(config)
        self._worker.progress.connect(self._on_progress)
        self._worker.log_output.connect(self._log)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.cancelled.connect(self._on_cancelled)
        self._worker.start()
        self.pipeline_started.emit()

    def stop_pipeline(self) -> None:
        """Request the pipeline worker to stop."""
        if self._worker and self._worker.isRunning():
            self._log("Stopping pipeline (will finish current stage)...")
            self._worker.request_stop()
            self._stop_btn.setEnabled(False)
            # Worker will emit cancelled signal when it exits
        else:
            self._log("No pipeline is running.")

    # Stage weights matching pipeline.py for within-stage progress
    _STAGE_WEIGHTS = [
        ("Starting", 0), ("Preprocessing", 15), ("Semantic Segmentation", 45),
        ("Post-processing", 20), ("Measurement", 15), ("Report Generation", 5),
        ("Complete", 0),
    ]

    @Slot(str, float)
    def _on_progress(self, stage: str, fraction: float) -> None:
        # Overall bar (dark green, thin)
        self._overall_bar.setValue(int(fraction * 100))
        # Stage bar — estimate within-stage fraction from overall fraction
        stage_frac = self._estimate_stage_fraction(stage, fraction)
        self._progress_bar.setValue(int(stage_frac * 100))
        self._progress_label.setText(f"{stage}  —  {fraction*100:.0f}% overall")

    def _estimate_stage_fraction(self, stage: str, overall: float) -> float:
        """Estimate within-stage fraction from overall pipeline fraction."""
        # Build stage ranges from weights
        total = sum(w for _, w in self._STAGE_WEIGHTS) or 1
        cum = 0.0
        for name, weight in self._STAGE_WEIGHTS:
            start = cum / total
            end = (cum + weight) / total
            if name == stage and weight > 0:
                rng = end - start
                if rng > 0:
                    return min(1.0, max(0.0, (overall - start) / rng))
                return 0.0
            cum += weight
        return overall  # fallback

    @Slot(str)
    def _on_finished(self, output_dir: str) -> None:
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._progress_container.setVisible(False)
        self._log(f"Pipeline complete! Output: {output_dir}")

        # Refresh run selector and auto-select the completed run.
        # Use resolved paths for comparison since FSCTPaths resolves them.
        self._populate_runs()
        resolved = str(Path(output_dir).resolve())
        matched = False
        for i in range(self._run_combo.count()):
            combo_path = str(Path(self._run_combo.itemData(i)).resolve())
            if combo_path == resolved:
                self._run_combo.setCurrentIndex(i)
                matched = True
                break
        if not matched and self._run_combo.count() > 0:
            # Fallback: select the newest run (index 0)
            self._run_combo.setCurrentIndex(0)

        self._populate_results(output_dir)
        self.pipeline_finished.emit(output_dir)

    @Slot(str, str)
    def _on_error(self, msg: str, traceback_str: str = "") -> None:
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._progress_container.setVisible(False)
        self._log(f"ERROR: {msg}")
        if traceback_str:
            self._log(f"--- Full Traceback ---\n{traceback_str}")
        self.pipeline_error.emit()
        QMessageBox.critical(self, "Pipeline Error", msg)

    @Slot()
    def _on_cancelled(self) -> None:
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._progress_container.setVisible(False)
        self._log("Pipeline cancelled by user.")

    def _log(self, msg: str) -> None:
        self._console.append(msg)

    def _save_console_log(self) -> None:
        """Save the console output to a text file."""
        text = self._console.toPlainText()
        if not text.strip():
            QMessageBox.information(self, "Empty Log", "No console output to save.")
            return
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Console Log", "", "Text Files (*.txt);;All Files (*)",
        )
        if filepath:
            if not filepath.endswith(".txt"):
                filepath += ".txt"
            with open(filepath, "w") as f:
                f.write(text)
            self._log(f"Log saved to: {filepath}")

    # --- Project save/load ---

    def save_project(self, filepath: str) -> None:
        config = self._build_config()
        config.save(filepath)
        self._last_save_path = filepath
        self._log(f"Project saved to: {filepath}")
        self.project_saved.emit(filepath)

    def load_project(self, filepath: str) -> None:
        try:
            config = ProjectConfig.load(filepath)
            self._apply_config(config)
            self._last_save_path = filepath
            self._log(f"Project loaded from: {filepath}")

            # Populate run selector with previous runs
            self._populate_runs()

            # Prefer the prepared (subsampled/cropped) cloud over the original
            cloud_to_load = ""
            if config.prepared_point_cloud and os.path.exists(config.prepared_point_cloud):
                cloud_to_load = config.prepared_point_cloud
                self._log(f"Using prepared point cloud: {os.path.basename(cloud_to_load)}")
            elif config.point_cloud_filename and os.path.exists(config.point_cloud_filename):
                cloud_to_load = config.point_cloud_filename

            if cloud_to_load:
                self.file_loaded.emit(cloud_to_load)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load project: {e}")
