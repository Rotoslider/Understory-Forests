"""Understory main application window.

Layout:
    +------------------------------------------------------------------+
    |  [Understory Logo]  File  View  Tools  Help                       |
    +------------------------------------------------------------------+
    |  [Sidebar/Tool Panel]   |  [3D Point Cloud Viewer]                |
    |                          |                                         |
    |  Project  Prepare       |                                         |
    |  Process  Advanced      |                                         |
    |  Results                |                                         |
    |  ─────────────────      |                                         |
    |  [Run]  [Stop]          |                                         |
    |  [Console Log]          |                                         |
    +-------------------------+-----------------------------------------+
    |  [Status Bar: Progress, GPU info, point count]                    |
    +------------------------------------------------------------------+
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from PySide6.QtCore import Qt, QThread, Signal, Slot, QSize, QTimer, QProcess, QObject, QSettings
from PySide6.QtGui import QAction, QActionGroup, QIcon, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QLabel,
    QStatusBar,
    QMenuBar,
    QMenu,
    QFileDialog,
    QMessageBox,
    QProgressBar,
)

from understory.gui.panels.processing_panel import ProcessingPanel
from understory.gui.viewer.point_cloud_viewer import PointCloudViewer


class GpuMonitor(QObject):
    """Polls GPU utilization and memory usage periodically."""

    updated = Signal(str)  # formatted status string

    def __init__(self, interval_ms: int = 2000, parent: QObject | None = None):
        super().__init__(parent)
        self._timer = QTimer(self)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self._poll)

    def start(self) -> None:
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()

    def _poll(self) -> None:
        try:
            import subprocess
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=3,
            )
            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split(",")
                util = int(parts[0].strip())
                mem_used = float(parts[1].strip()) / 1024  # MiB -> GiB
                mem_total = float(parts[2].strip()) / 1024
                self.updated.emit(f"GPU: {util}% | {mem_used:.1f}/{mem_total:.0f} GB")
                return
        except Exception:
            pass

        # Fallback: torch.cuda.mem_get_info
        try:
            import torch
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info(0)
                used = (total - free) / (1024**3)
                total_gb = total / (1024**3)
                self.updated.emit(f"GPU: {used:.1f}/{total_gb:.0f} GB")
                return
        except Exception:
            pass

        self.updated.emit("GPU: N/A")


class MainWindow(QMainWindow):
    """Main Understory application window."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Understory")
        self.setMinimumSize(1200, 800)

        self._current_project_path: Optional[str] = None
        self._settings = QSettings("Understory", "Understory")
        self._last_directory: str = self._settings.value("last_directory", "")

        self._load_stylesheet()
        self._setup_icon()
        self._setup_menu_bar()
        self._setup_central_widget()
        self._setup_status_bar()

        # GPU live monitor (started/stopped with pipeline)
        self._gpu_monitor = GpuMonitor(parent=self)
        self._gpu_monitor.updated.connect(self._gpu_label.setText)
        self.setAcceptDrops(True)

        self._restore_settings()

    def _load_stylesheet(self) -> None:
        dark = self._settings.value("theme/dark", False, type=bool)
        if dark:
            qss_path = Path(__file__).parent.parent / "resources" / "styles" / "understory_dark.qss"
        else:
            qss_path = Path(__file__).parent.parent / "resources" / "styles" / "understory.qss"
        if qss_path.exists():
            self.setStyleSheet(qss_path.read_text())

    def _setup_icon(self) -> None:
        icon_path = Path(__file__).parent.parent / "resources" / "icons" / "understory-icon.png"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

    def _setup_menu_bar(self) -> None:
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        new_project_action = QAction("New Project", self)
        new_project_action.setShortcut("Ctrl+N")
        new_project_action.triggered.connect(self._new_project)
        file_menu.addAction(new_project_action)

        file_menu.addSeparator()

        open_action = QAction("Open Point Cloud...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._open_file)
        file_menu.addAction(open_action)

        open_project_action = QAction("Open Project...", self)
        open_project_action.setShortcut("Ctrl+Shift+O")
        open_project_action.triggered.connect(self._open_project)
        file_menu.addAction(open_project_action)

        file_menu.addSeparator()

        close_cloud_action = QAction("Close Point Cloud", self)
        close_cloud_action.setShortcut("Ctrl+W")
        close_cloud_action.triggered.connect(self._close_point_cloud)
        file_menu.addAction(close_cloud_action)

        file_menu.addSeparator()

        save_project_action = QAction("Save Project", self)
        save_project_action.setShortcut("Ctrl+S")
        save_project_action.triggered.connect(self._save_project)
        file_menu.addAction(save_project_action)

        save_as_action = QAction("Save Project As...", self)
        save_as_action.setShortcut("Ctrl+Shift+S")
        save_as_action.triggered.connect(self._save_project_as)
        file_menu.addAction(save_as_action)

        file_menu.addSeparator()

        self._recent_menu = QMenu("Recent Projects", self)
        file_menu.addMenu(self._recent_menu)
        self._update_recent_menu()

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View menu
        view_menu = menubar.addMenu("View")

        reset_view_action = QAction("Reset Camera", self)
        reset_view_action.setShortcut("Home")
        reset_view_action.triggered.connect(self._reset_camera)
        view_menu.addAction(reset_view_action)

        view_menu.addSeparator()

        top_view_action = QAction("Top View", self)
        top_view_action.setShortcut("Ctrl+1")
        top_view_action.triggered.connect(lambda: self._viewer.set_camera_view("top"))
        view_menu.addAction(top_view_action)

        front_view_action = QAction("Front View", self)
        front_view_action.setShortcut("Ctrl+2")
        front_view_action.triggered.connect(lambda: self._viewer.set_camera_view("front"))
        view_menu.addAction(front_view_action)

        right_view_action = QAction("Right View", self)
        right_view_action.setShortcut("Ctrl+3")
        right_view_action.triggered.connect(lambda: self._viewer.set_camera_view("right"))
        view_menu.addAction(right_view_action)

        iso_view_action = QAction("Isometric View", self)
        iso_view_action.setShortcut("Ctrl+4")
        iso_view_action.triggered.connect(lambda: self._viewer.set_camera_view("iso"))
        view_menu.addAction(iso_view_action)

        view_menu.addSeparator()

        screenshot_action = QAction("Export Screenshot...", self)
        screenshot_action.setShortcut("Ctrl+Shift+E")
        screenshot_action.triggered.connect(self._export_screenshot)
        view_menu.addAction(screenshot_action)

        view_menu.addSeparator()

        undo_action = QAction("Undo Prepare", self)
        undo_action.setShortcut("Ctrl+Z")
        undo_action.triggered.connect(self._undo_prepare)
        view_menu.addAction(undo_action)

        redo_action = QAction("Redo Prepare", self)
        redo_action.setShortcut("Ctrl+Shift+Z")
        redo_action.triggered.connect(self._redo_prepare)
        view_menu.addAction(redo_action)

        view_menu.addSeparator()

        flythrough_action = QAction("Flythrough Editor...", self)
        flythrough_action.triggered.connect(self._open_flythrough)
        view_menu.addAction(flythrough_action)

        view_menu.addSeparator()

        # Units submenu
        units_menu = QMenu("Units", self)
        units_group = QActionGroup(self)
        units_group.setExclusive(True)

        metric_action = QAction("Metric (m)", self)
        metric_action.setCheckable(True)
        metric_action.setChecked(True)
        metric_action.triggered.connect(lambda: self._set_units("Metric"))
        units_group.addAction(metric_action)
        units_menu.addAction(metric_action)

        imperial_action = QAction("Imperial (ft)", self)
        imperial_action.setCheckable(True)
        imperial_action.triggered.connect(lambda: self._set_units("Imperial"))
        units_group.addAction(imperial_action)
        units_menu.addAction(imperial_action)

        view_menu.addMenu(units_menu)

        # Restore saved unit preference
        saved_units = self._settings.value("viewer/units", "Metric")
        if saved_units == "Imperial":
            imperial_action.setChecked(True)
        self._units_actions = {"Metric": metric_action, "Imperial": imperial_action}

        view_menu.addSeparator()

        self._dark_mode_action = QAction("Dark Mode", self)
        self._dark_mode_action.setCheckable(True)
        self._dark_mode_action.setChecked(self._settings.value("theme/dark", False, type=bool))
        self._dark_mode_action.triggered.connect(self._toggle_theme)
        view_menu.addAction(self._dark_mode_action)

        # Tools menu
        tools_menu = menubar.addMenu("Tools")

        run_pipeline_action = QAction("Run Pipeline", self)
        run_pipeline_action.setShortcut("F5")
        run_pipeline_action.triggered.connect(self._run_pipeline)
        tools_menu.addAction(run_pipeline_action)

        stop_pipeline_action = QAction("Stop Pipeline", self)
        stop_pipeline_action.setShortcut("Shift+F5")
        stop_pipeline_action.triggered.connect(self._stop_pipeline)
        tools_menu.addAction(stop_pipeline_action)

        tools_menu.addSeparator()

        training_action = QAction("Training Workflow...", self)
        training_action.triggered.connect(self._open_training)
        tools_menu.addAction(training_action)

        label_editor_action = QAction("Label Editor...", self)
        label_editor_action.triggered.connect(self._open_label_editor)
        tools_menu.addAction(label_editor_action)

        tools_menu.addSeparator()

        batch_action = QAction("Batch Processing...", self)
        batch_action.triggered.connect(self._open_batch)
        tools_menu.addAction(batch_action)

        growth_action = QAction("Growth Dashboard...", self)
        growth_action.triggered.connect(self._open_growth_dashboard)
        tools_menu.addAction(growth_action)

        allometry_action = QAction("Allometric Equations...", self)
        allometry_action.triggered.connect(self._open_allometry)
        tools_menu.addAction(allometry_action)

        tools_menu.addSeparator()

        measure_menu = QMenu("Measure", self)
        distance_action = QAction("Distance", self)
        distance_action.triggered.connect(lambda: self._start_measure("distance"))
        measure_menu.addAction(distance_action)
        height_action = QAction("Height", self)
        height_action.triggered.connect(lambda: self._start_measure("height"))
        measure_menu.addAction(height_action)
        measure_menu.addSeparator()
        stop_measure_action = QAction("Stop Measuring (Escape)", self)
        stop_measure_action.triggered.connect(self._stop_measure)
        measure_menu.addAction(stop_measure_action)
        clear_measure_action = QAction("Clear Measurements", self)
        clear_measure_action.triggered.connect(self._clear_measurements)
        measure_menu.addAction(clear_measure_action)
        tools_menu.addMenu(measure_menu)

        compare_clouds_action = QAction("Compare Point Clouds...", self)
        compare_clouds_action.triggered.connect(self._compare_clouds)
        tools_menu.addAction(compare_clouds_action)

        # Help menu
        help_menu = menubar.addMenu("Help")

        about_action = QAction("About Understory", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _setup_central_widget(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        self._splitter = QSplitter(Qt.Horizontal)
        splitter = self._splitter

        # Left panel: processing controls
        self._processing_panel = ProcessingPanel()
        self._processing_panel.setMinimumWidth(380)
        self._processing_panel.setMaximumWidth(500)
        self._processing_panel.file_loaded.connect(self._on_file_loaded)
        self._processing_panel.pipeline_started.connect(self._on_pipeline_started)
        self._processing_panel.pipeline_finished.connect(self._on_pipeline_finished)
        self._processing_panel.pipeline_error.connect(self._on_pipeline_error)
        self._processing_panel.plot_centre_changed.connect(self._on_plot_centre_changed)
        self._processing_panel.swap_axes_requested.connect(self._on_swap_axes)
        self._processing_panel.crop_outliers_requested.connect(self._on_crop_outliers)
        self._processing_panel.reset_crop_requested.connect(self._on_reset_crop)
        self._processing_panel.subsample_requested.connect(self._on_subsample_preview)
        self._processing_panel.save_cloud_requested.connect(self._on_save_cloud)
        self._processing_panel.trim_select_requested.connect(self._on_trim_select)
        self._processing_panel.trim_apply_requested.connect(self._on_trim_apply)
        self._processing_panel.trim_cancel_requested.connect(self._on_trim_cancel)
        self._processing_panel.project_saved.connect(self._on_project_saved)
        self._processing_panel.load_output_layers.connect(self._on_load_output_layers)
        self._processing_panel.tree_selected.connect(self._on_tree_selected)
        splitter.addWidget(self._processing_panel)

        # Right panel: 3D viewer
        self._viewer = PointCloudViewer()
        self._viewer.point_picked.connect(self._on_point_picked)
        self._viewer.plot_centre_dragged.connect(self._on_plot_centre_dragged)
        self._viewer.crop_state_changed.connect(self._on_crop_state_changed)
        self._viewer.trim_region_selected.connect(self._processing_panel.on_trim_region_selected)
        splitter.addWidget(self._viewer)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([380, 820])

        layout.addWidget(splitter)

    def _setup_status_bar(self) -> None:
        status_bar = QStatusBar()
        self.setStatusBar(status_bar)

        # GPU info
        gpu_info = self._get_gpu_info()
        self._gpu_label = QLabel(gpu_info)
        status_bar.addWidget(self._gpu_label)

        # Spacer
        status_bar.addWidget(QLabel("  |  "))

        # Point count
        self._point_count_status = QLabel("")
        status_bar.addWidget(self._point_count_status)

        status_bar.addWidget(QLabel("  |  "))

        # Progress bar
        self._progress_bar = QProgressBar()
        self._progress_bar.setMaximumWidth(200)
        self._progress_bar.setMaximumHeight(16)
        self._progress_bar.setVisible(False)
        status_bar.addWidget(self._progress_bar)

        # Status message
        self._status_label = QLabel("Ready")
        status_bar.addPermanentWidget(self._status_label)

    @staticmethod
    def _get_gpu_info() -> str:
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                return f"GPU: {name} ({mem:.0f} GB)"
            else:
                return "GPU: None (CPU mode)"
        except ImportError:
            return "GPU: PyTorch not installed"

    def _restore_settings(self) -> None:
        """Restore window geometry, splitter state, and preferences from QSettings."""
        geom = self._settings.value("window/geometry")
        if geom:
            self.restoreGeometry(geom)
        state = self._settings.value("window/state")
        if state:
            self.restoreState(state)
        splitter_state = self._settings.value("window/splitter")
        if splitter_state:
            self._splitter.restoreState(splitter_state)
        # Restore color mode
        saved_color = self._settings.value("viewer/color_mode")
        if saved_color:
            for i in range(self._viewer._color_combo.count()):
                if self._viewer._color_combo.itemData(i).value == saved_color:
                    self._viewer._color_combo.setCurrentIndex(i)
                    break
        # Restore unit system
        saved_units = self._settings.value("viewer/units", "Metric")
        if saved_units == "Imperial":
            from understory.gui.viewer.point_cloud_viewer import UnitSystem
            self._viewer.set_unit_system(UnitSystem.IMPERIAL)

    def closeEvent(self, event) -> None:
        """Save settings on close."""
        self._settings.setValue("window/geometry", self.saveGeometry())
        self._settings.setValue("window/state", self.saveState())
        self._settings.setValue("window/splitter", self._splitter.saveState())
        self._settings.setValue("last_directory", self._last_directory)
        if self._viewer._color_mode:
            self._settings.setValue("viewer/color_mode", self._viewer._color_mode.value)
        super().closeEvent(event)

    def _remember_directory(self, filepath: str) -> None:
        """Remember the directory of the given file for future dialogs."""
        self._last_directory = os.path.dirname(filepath)

    # --- Menu actions ---

    def _open_file(self) -> None:
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Open Point Cloud",
            self._last_directory,
            "Point Clouds (*.las *.laz *.pcd);;All Files (*)",
        )
        if filepath:
            self._remember_directory(filepath)
            self._processing_panel.set_input_file(filepath)

    def _open_project(self) -> None:
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Open Project",
            self._last_directory,
            "Understory Projects (*.yaml *.yml);;All Files (*)",
        )
        if filepath:
            self._remember_directory(filepath)
            self._processing_panel.load_project(filepath)
            self._current_project_path = filepath
            self._update_title()
            self._add_recent_project(filepath)

    def _save_project(self) -> None:
        if self._current_project_path:
            self._processing_panel.save_project(self._current_project_path)
            self._status_label.setText(f"Saved: {os.path.basename(self._current_project_path)}")
        else:
            self._save_project_as()

    def _save_project_as(self) -> None:
        # Get project name from the UI
        project_name = self._processing_panel._project_name.text().strip()
        if not project_name:
            QMessageBox.warning(
                self, "Project Name Required",
                "Please enter a project name in the Project tab before saving.",
            )
            self._processing_panel._project_name.setFocus()
            return

        # Ask user where to create the project folder
        parent_dir = QFileDialog.getExistingDirectory(
            self,
            "Choose location for project folder",
            str(Path.home()),
        )
        if not parent_dir:
            return

        # Create project folder structure: <parent>/<project_name>/
        from understory.core.paths import ProjectPaths
        project_dir = Path(parent_dir) / project_name
        project_paths = ProjectPaths(project_dir)
        project_paths.ensure_dirs()

        # Save project.yaml inside the project folder
        yaml_path = str(project_paths.config_file)
        self._processing_panel.save_project(yaml_path)
        self._current_project_path = yaml_path
        self._update_title()
        self._status_label.setText(f"Project saved to: {project_dir}")

    def _new_project(self) -> None:
        """Reset all settings and clear the viewer for a fresh project."""
        reply = QMessageBox.question(
            self,
            "New Project",
            "Clear all settings and start a new project?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self._viewer.clear()
            from understory.config.settings import ProjectConfig
            self._processing_panel._apply_config(ProjectConfig())
            self._processing_panel._file_input.clear()
            self._processing_panel._output_dir.clear()
            self._processing_panel._notes.clear()
            self._processing_panel._console.clear()
            self._current_project_path = None
            self._processing_panel._last_save_path = None
            self._processing_panel._prepared_cloud_path = None
            self._update_title()
            self._processing_panel._tabs.setCurrentIndex(0)  # Switch to Project tab
            self._status_label.setText("Ready — New Project")

    def _close_point_cloud(self) -> None:
        self._viewer.clear()
        self._status_label.setText("Ready")
        self._update_point_count()

    def _reset_camera(self) -> None:
        self._viewer._reset_view()

    def _run_pipeline(self) -> None:
        self._processing_panel.run_pipeline()

    def _stop_pipeline(self) -> None:
        self._processing_panel.stop_pipeline()

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "About Understory",
            "<h2>Understory</h2>"
            "<p>Forest Structural Complexity Tool</p>"
            "<p>Modernized GUI for forest LiDAR point cloud processing.</p>"
            "<p>Performs semantic segmentation, tree measurement, "
            "and report generation from terrestrial laser scanning data.</p>"
            f"<p>Version {self._get_version()}</p>",
        )

    @staticmethod
    def _get_version() -> str:
        try:
            from understory import __version__
            return __version__
        except ImportError:
            return "unknown"

    def _update_title(self) -> None:
        if self._current_project_path:
            name = os.path.basename(self._current_project_path)
            self.setWindowTitle(f"Understory — {name}")
        else:
            self.setWindowTitle("Understory")

    def _update_point_count(self) -> None:
        """Update the status bar point count label."""
        viewer = self._viewer
        if viewer._points_full is None:
            self._point_count_status.setText("")
            return
        total = viewer._points_full.shape[0]
        displayed = len(viewer._lod_indices) if viewer._lod_indices is not None else total
        if displayed < total:
            self._point_count_status.setText(f"{displayed:,} / {total:,} points")
        else:
            self._point_count_status.setText(f"{total:,} points")

    # --- Slots ---

    @Slot(str)
    def _on_file_loaded(self, filepath: str) -> None:
        """Load and display a point cloud file in the viewer."""
        self._remember_directory(filepath)
        self._viewer.clear()
        self._status_label.setText(f"Loading: {os.path.basename(filepath)}")
        QApplication.processEvents()

        try:
            # Add scripts to path for load_file
            scripts_dir = str(Path(__file__).parent.parent.parent / "scripts")
            if scripts_dir not in sys.path:
                sys.path.insert(0, scripts_dir)
            from tools import load_file

            pc, headers = load_file(
                filepath,
                headers_of_interest=["x", "y", "z", "red", "green", "blue"],
            )

            if pc.shape[0] == 0:
                self._status_label.setText("Error: Empty point cloud")
                return

            points = pc[:, :3]
            colors = None
            if headers and len(headers) >= 6:
                color_cols = []
                for h in ("red", "green", "blue"):
                    if h in headers:
                        color_cols.append(headers.index(h))
                if len(color_cols) == 3:
                    colors = pc[:, color_cols]

            self._viewer.load_points(points, colors=colors)
            self._add_recent_project(filepath)
            self._status_label.setText(f"Loaded: {os.path.basename(filepath)} ({points.shape[0]:,} points)")
            self._update_point_count()

        except Exception as e:
            self._status_label.setText(f"Error loading file: {e}")

    @Slot(int, float, float, float)
    def _on_point_picked(self, index: int, x: float, y: float, z: float) -> None:
        """When a point is picked in focus mode, offer it as plot centre."""
        self._status_label.setText(f"Point picked: ({x:.3f}, {y:.3f}, {z:.3f})")

    @Slot(object)
    def _on_plot_centre_changed(self, centre) -> None:
        """Update the viewer plot circle when plot centre changes."""
        # Skip if this was triggered by the viewer drag updating the spinboxes
        if self._processing_panel._updating_from_viewer:
            return

        radius = self._processing_panel._plot_radius.value()
        if radius <= 0:
            self._viewer.disable_plot_circle_interaction()
            self._viewer.clear_plot_circle()
            return

        if centre is None:
            # Auto mode — compute centre from loaded cloud
            if self._viewer._points_full is not None:
                cx = float(np.mean(self._viewer._points_full[:, 0]))
                cy = float(np.mean(self._viewer._points_full[:, 1]))
                z = float(np.median(self._viewer._points_full[:, 2]))
                self._viewer.show_plot_circle(cx, cy, radius, z)
                self._viewer.enable_plot_circle_interaction(cx, cy, radius, z)
            else:
                self._viewer.disable_plot_circle_interaction()
                self._viewer.clear_plot_circle()
        else:
            z = 0
            if self._viewer._points_full is not None:
                z = float(np.median(self._viewer._points_full[:, 2]))
            self._viewer.show_plot_circle(centre[0], centre[1], radius, z)
            self._viewer.enable_plot_circle_interaction(centre[0], centre[1], radius, z)

    @Slot(float, float)
    def _on_plot_centre_dragged(self, x: float, y: float) -> None:
        """When the user drags the plot circle widget in the viewer."""
        self._processing_panel.set_plot_centre(x, y)

    @Slot(str)
    def _on_swap_axes(self, mode: str) -> None:
        """Forward axis swap request from Prepare tab to the viewer."""
        self._viewer.apply_axis_swap(mode)

    @Slot()
    def _on_crop_outliers(self) -> None:
        """Forward crop request from Prepare tab to the viewer."""
        self._viewer._crop_to_bounds()

    @Slot()
    def _on_reset_crop(self) -> None:
        """Forward reset crop request from Prepare tab to the viewer."""
        self._viewer._reset_crop()

    @Slot()
    def _on_trim_select(self) -> None:
        """Forward trim selection request to the viewer."""
        self._viewer.enable_trim_selection()

    @Slot(bool)
    def _on_trim_apply(self, keep: bool) -> None:
        """Apply trim and update panel state."""
        n_before = self._viewer._points_full.shape[0] if self._viewer._points_full is not None else 0
        self._viewer.apply_trim(keep)
        n_after = self._viewer._points_full.shape[0] if self._viewer._points_full is not None else 0
        action = "Kept" if keep else "Removed"
        self._status_label.setText(f"{action}: {n_before:,} -> {n_after:,} points")
        self._update_point_count()
        self._processing_panel.on_trim_applied()

    @Slot()
    def _on_trim_cancel(self) -> None:
        """Cancel trim selection."""
        self._viewer.cancel_trim()

    @Slot(float)
    def _on_subsample_preview(self, spacing: float) -> None:
        """Apply voxel-grid subsampling to the loaded point cloud in the viewer."""
        if self._viewer._points_full is None:
            QMessageBox.warning(self, "No Data", "No point cloud is loaded.")
            return

        pts = self._viewer._points_full
        n_before = pts.shape[0]
        self._status_label.setText(f"Subsampling ({n_before:,} points, spacing={spacing:.3f}m)...")
        QApplication.processEvents()

        # Voxel-grid subsampling via np.unique on quantized coordinates
        voxel_coords = np.floor(pts / spacing).astype(np.int64)
        _, unique_idx = np.unique(voxel_coords, axis=0, return_index=True)
        unique_idx.sort()

        self._viewer._points_full = pts[unique_idx]
        self._viewer._points_original = self._viewer._points_full.copy()
        if self._viewer._colors_full is not None:
            self._viewer._colors_full = self._viewer._colors_full[unique_idx]
        if self._viewer._labels is not None:
            self._viewer._labels = self._viewer._labels[unique_idx]
        if self._viewer._tree_ids is not None:
            self._viewer._tree_ids = self._viewer._tree_ids[unique_idx]
        # Update originals so Reset Crop doesn't undo the subsample
        self._viewer._colors_original = self._viewer._colors_full.copy() if self._viewer._colors_full is not None else None
        self._viewer._labels_original = self._viewer._labels.copy() if self._viewer._labels is not None else None
        self._viewer._tree_ids_original = self._viewer._tree_ids.copy() if self._viewer._tree_ids is not None else None
        self._viewer._build_lod()
        self._viewer._render()

        n_after = self._viewer._points_full.shape[0]
        self._status_label.setText(
            f"Subsampled: {n_before:,} -> {n_after:,} points ({n_after/n_before*100:.1f}%)"
        )
        self._update_point_count()

    @Slot(bool)
    def _on_crop_state_changed(self, cropped: bool) -> None:
        """Update the Reset Crop button state in the Prepare tab."""
        self._processing_panel._reset_crop_btn.setEnabled(cropped)

    @Slot(str)
    def _on_save_cloud(self, filepath: str) -> None:
        """Save the current (possibly modified) point cloud to a .las file."""
        if self._viewer._points_full is None:
            QMessageBox.warning(self, "No Data", "No point cloud is loaded.")
            return

        try:
            scripts_dir = str(Path(__file__).parent.parent.parent / "scripts")
            if scripts_dir not in sys.path:
                sys.path.insert(0, scripts_dir)
            from tools import save_file

            points = self._viewer._points_full
            colors = self._viewer._colors_full

            if colors is not None:
                # Convert back to 0-65535 range for LAS
                colors_out = (colors * 65535).astype(np.uint16).astype(np.float64)
                data = np.hstack([points, colors_out])
                headers = ["x", "y", "z", "red", "green", "blue"]
            else:
                data = points
                headers = ["x", "y", "z"]

            save_file(filepath, data, headers_of_interest=headers)
            self._processing_panel._log(f"Point cloud saved to: {filepath}")
            self._status_label.setText(f"Saved: {os.path.basename(filepath)}")

            # Auto-save project config with updated prepared cloud path
            if self._current_project_path:
                self._processing_panel.save_project(self._current_project_path)
                self._processing_panel._log(
                    f"Project updated with prepared cloud: {os.path.basename(filepath)}"
                )

            # Offer to reload saved cloud as the active project input
            reply = QMessageBox.question(
                self,
                "Reload Cloud",
                "Reload this cloud as the active project input?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply == QMessageBox.Yes:
                self._processing_panel.set_input_file(filepath)
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save point cloud: {e}")

    @Slot(str)
    def _on_project_saved(self, filepath: str) -> None:
        """Track current project path when saved from any source."""
        self._current_project_path = filepath
        self._update_title()

    @Slot(list)
    def _on_load_output_layers(self, paths: list) -> None:
        """Load multiple output .las files and merge them into the viewer."""
        self._viewer.clear()
        self._status_label.setText("Loading output layers...")
        QApplication.processEvents()

        try:
            scripts_dir = str(Path(__file__).parent.parent.parent / "scripts")
            if scripts_dir not in sys.path:
                sys.path.insert(0, scripts_dir)
            from tools import load_file

            all_points = []
            all_labels = []
            all_colors = []
            all_tree_ids = []
            for filepath in paths:
                if not os.path.exists(filepath):
                    continue
                pc, headers = load_file(
                    filepath,
                    headers_of_interest=["x", "y", "z", "label", "tree_id", "red", "green", "blue"],
                )
                if pc.shape[0] == 0:
                    continue
                all_points.append(pc[:, :3])

                if headers and "label" in headers:
                    all_labels.append(pc[:, headers.index("label")].astype(np.int32))
                else:
                    all_labels.append(np.zeros(pc.shape[0], dtype=np.int32))

                if headers and "tree_id" in headers:
                    all_tree_ids.append(pc[:, headers.index("tree_id")].astype(np.int32))
                else:
                    all_tree_ids.append(np.full(pc.shape[0], -1, dtype=np.int32))

                color_cols = []
                for h in ("red", "green", "blue"):
                    if headers and h in headers:
                        color_cols.append(headers.index(h))
                if len(color_cols) == 3:
                    all_colors.append(pc[:, color_cols])
                else:
                    all_colors.append(None)

            if not all_points:
                self._status_label.setText("No points found in selected layers")
                return

            points = np.vstack(all_points)
            labels = np.concatenate(all_labels)

            # Auto-detect 0-indexed labels from inference (0-3) and convert
            # to post-processing scheme (1-4) so CLASS_COLORS maps correctly.
            if 0 in labels and 4 not in labels:
                labels = labels + 1

            # Merge tree IDs
            tree_ids = np.concatenate(all_tree_ids)
            has_tree_ids = np.any(tree_ids >= 0)

            # Merge colors
            colors = None
            if all(c is not None for c in all_colors):
                colors = np.vstack(all_colors)

            self._viewer.load_points(
                points, colors=colors, labels=labels,
                tree_ids=tree_ids if has_tree_ids else None,
            )

            # Switch to classification color mode for layer viewing
            from understory.gui.viewer.point_cloud_viewer import ColorMode
            idx = self._viewer._color_combo.findData(ColorMode.CLASSIFICATION)
            if idx >= 0:
                self._viewer._color_combo.setCurrentIndex(idx)

            n_files = sum(1 for p in paths if os.path.exists(p))
            self._status_label.setText(
                f"Loaded {n_files} layer(s): {points.shape[0]:,} points"
            )
            self._update_point_count()
        except Exception as e:
            self._status_label.setText(f"Error loading layers: {e}")

    @Slot(int)
    def _on_tree_selected(self, tree_id: int) -> None:
        """Highlight a specific tree in the viewer when selected in the results table."""
        if self._viewer._tree_ids is None:
            self._status_label.setText(
                "Tree highlighting requires sorted layers (Stem/Veg Points Sorted)"
            )
            return

        # Switch to Tree ID color mode so the selected tree is visible
        from understory.gui.viewer.point_cloud_viewer import ColorMode
        idx = self._viewer._color_combo.findData(ColorMode.TREE_ID)
        if idx >= 0:
            self._viewer._color_combo.setCurrentIndex(idx)

        # Find the tree's points and focus the camera on them
        mask = self._viewer._tree_ids == tree_id
        if np.any(mask):
            tree_pts = self._viewer._points_full[mask]
            center = tree_pts.mean(axis=0)
            self._viewer._plotter.set_focus(center)
            self._status_label.setText(
                f"Tree {tree_id}: {mask.sum():,} points at ({center[0]:.1f}, {center[1]:.1f})"
            )

    @Slot()
    def _on_pipeline_started(self) -> None:
        """Handle pipeline start — begin GPU monitoring."""
        self._gpu_monitor.start()

    @Slot(str)
    def _on_pipeline_finished(self, output_dir: str) -> None:
        """Handle pipeline completion — load results into viewer."""
        self._gpu_monitor.stop()
        self._status_label.setText(f"Pipeline complete. Output: {output_dir}")
        self._progress_bar.setVisible(False)

    @Slot()
    def _on_pipeline_error(self) -> None:
        """Handle pipeline error — stop GPU monitor."""
        self._gpu_monitor.stop()

    def _open_training(self) -> None:
        """Open the training workflow panel in a separate window."""
        from understory.gui.panels.training_panel import TrainingPanel
        self._training_window = TrainingPanel()
        self._training_window.setWindowTitle("Understory — Training Workflow")
        self._training_window.resize(600, 800)
        self._training_window.show()

    def _open_label_editor(self) -> None:
        """Open the label editor for a selected file."""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Open Point Cloud for Label Editing",
            self._last_directory,
            "Point Clouds (*.las *.laz *.pcd);;All Files (*)",
        )
        if not filepath:
            return

        from understory.gui.viewer.label_editor import LabelEditor

        scripts_dir = str(Path(__file__).parent.parent.parent / "scripts")
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
        self._label_editor = editor

    # --- Screenshot Export ---

    def _export_screenshot(self) -> None:
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Screenshot", self._last_directory,
            "PNG (*.png);;JPEG (*.jpg);;TIFF (*.tiff);;All Files (*)",
        )
        if filepath:
            try:
                self._viewer._plotter.screenshot(filepath, scale=2)
                self._status_label.setText(f"Screenshot saved: {os.path.basename(filepath)}")
            except Exception as e:
                QMessageBox.critical(self, "Screenshot Error", f"Failed to save screenshot: {e}")

    # --- Recent Projects ---

    def _update_recent_menu(self) -> None:
        self._recent_menu.clear()
        recent = self._settings.value("recent_projects", [])
        if not recent:
            action = QAction("(No recent projects)", self)
            action.setEnabled(False)
            self._recent_menu.addAction(action)
            return
        for path in recent:
            action = QAction(os.path.basename(path), self)
            action.setToolTip(path)
            action.setData(path)
            action.triggered.connect(lambda checked=False, p=path: self._open_recent(p))
            self._recent_menu.addAction(action)

    def _add_recent_project(self, filepath: str) -> None:
        recent = self._settings.value("recent_projects", [])
        if not isinstance(recent, list):
            recent = []
        if filepath in recent:
            recent.remove(filepath)
        recent.insert(0, filepath)
        recent = recent[:10]
        self._settings.setValue("recent_projects", recent)
        self._update_recent_menu()

    def _open_recent(self, filepath: str) -> None:
        if os.path.exists(filepath):
            if filepath.endswith(('.yaml', '.yml')):
                self._processing_panel.load_project(filepath)
                self._current_project_path = filepath
                self._update_title()
            else:
                self._processing_panel.set_input_file(filepath)
            self._add_recent_project(filepath)
        else:
            QMessageBox.warning(self, "File Not Found", f"Could not find: {filepath}")

    # --- Drag-and-Drop ---

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                path = url.toLocalFile().lower()
                if path.endswith(('.las', '.laz', '.pcd', '.yaml', '.yml')):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event) -> None:
        for url in event.mimeData().urls():
            filepath = url.toLocalFile()
            lower = filepath.lower()
            if lower.endswith(('.yaml', '.yml')):
                self._processing_panel.load_project(filepath)
                self._current_project_path = filepath
                self._update_title()
                self._add_recent_project(filepath)
            elif lower.endswith(('.las', '.laz', '.pcd')):
                self._processing_panel.set_input_file(filepath)
            break  # Only handle the first file

    # --- Undo/Redo ---

    def _undo_prepare(self) -> None:
        if hasattr(self._viewer, 'undo') and self._viewer.undo():
            self._status_label.setText("Undo")
        else:
            self._status_label.setText("Nothing to undo")

    def _redo_prepare(self) -> None:
        if hasattr(self._viewer, 'redo') and self._viewer.redo():
            self._status_label.setText("Redo")
        else:
            self._status_label.setText("Nothing to redo")

    # --- Phase 3 Tools ---

    def _open_batch(self) -> None:
        try:
            from understory.gui.panels.batch_panel import BatchPanel
            dlg = BatchPanel(self._processing_panel._build_config(), parent=self)
            dlg.exec()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not open batch processing: {e}")

    def _open_growth_dashboard(self) -> None:
        try:
            from understory.gui.panels.growth_panel import GrowthPanel
            project_dir = None
            if self._current_project_path:
                project_dir = str(Path(self._current_project_path).parent)
            dlg = GrowthPanel(project_dir=project_dir, parent=self)
            dlg.exec()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not open growth dashboard: {e}")

    def _open_allometry(self) -> None:
        try:
            from understory.gui.panels.allometry_panel import AllometryPanel
            output_dir = None
            if hasattr(self._processing_panel, '_results_output_dir'):
                output_dir = self._processing_panel._results_output_dir
            dlg = AllometryPanel(output_dir=output_dir, parent=self)
            dlg.exec()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not open allometry panel: {e}")

    def _toggle_theme(self, dark: bool) -> None:
        """Toggle between light and dark themes."""
        self._settings.setValue("theme/dark", dark)
        self._load_stylesheet()
        # Update PyVista viewer background
        bg = "#0d1b16" if dark else "#1a2e26"
        self._viewer._plotter.set_background(bg)
        self._viewer._plotter.render()

    def _set_units(self, system_name: str) -> None:
        """Set the measurement unit system."""
        from understory.gui.viewer.point_cloud_viewer import UnitSystem
        system = UnitSystem(system_name)
        self._viewer.set_unit_system(system)
        self._settings.setValue("viewer/units", system_name)
        self._status_label.setText(f"Units: {system_name}")

    def _start_measure(self, mode: str) -> None:
        if hasattr(self._viewer, 'start_measurement'):
            self._viewer.start_measurement(mode)
            self._status_label.setText(
                f"Measure {mode}: click first point, then second point. "
                f"Escape or Tools > Measure > Stop to exit."
            )
        else:
            self._status_label.setText("Measurement tools not available in this viewer version")

    def _stop_measure(self) -> None:
        if hasattr(self._viewer, 'cancel_measurement'):
            self._viewer.cancel_measurement()
            self._status_label.setText("Measurement mode stopped")

    def _clear_measurements(self) -> None:
        if hasattr(self._viewer, 'clear_measurements'):
            self._viewer.clear_measurements()
            self._status_label.setText("Measurements cleared")

    def _compare_clouds(self) -> None:
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open Second Point Cloud for Comparison", self._last_directory,
            "Point Clouds (*.las *.laz *.pcd);;All Files (*)",
        )
        if filepath and hasattr(self._viewer, 'compare_with_cloud'):
            self._viewer.compare_with_cloud(filepath)
            self._status_label.setText(f"Comparing with: {os.path.basename(filepath)}")

    def _open_flythrough(self) -> None:
        try:
            from understory.gui.panels.flythrough import FlythroughEditor
            # Non-modal so user can interact with the 3D viewer between keyframes
            self._flythrough_dlg = FlythroughEditor(plotter=self._viewer._plotter, parent=self)
            self._flythrough_dlg.show()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not open flythrough editor: {e}")

    def set_progress(self, stage: str, fraction: float) -> None:
        """Update the status bar progress."""
        self._progress_bar.setVisible(True)
        self._progress_bar.setValue(int(fraction * 100))
        self._status_label.setText(stage)
        if fraction >= 1.0:
            self._progress_bar.setVisible(False)
