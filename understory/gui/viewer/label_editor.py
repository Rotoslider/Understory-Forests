"""Label correction tool extending the point cloud viewer.

Provides selection tools and class painting for training data preparation.
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

import numpy as np

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QButtonGroup,
    QRadioButton,
    QGroupBox,
    QCheckBox,
    QDoubleSpinBox,
    QScrollArea,
    QSplitter,
)
from PySide6.QtGui import QKeySequence, QShortcut

try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False


# Class definitions matching FSCT conventions
CLASSES = {
    1: ("Terrain", "#8B6914"),      # brown
    2: ("Vegetation", "#2d7a5e"),   # green
    3: ("CWD", "#DAA520"),          # goldenrod
    4: ("Stem", "#CC3333"),         # red
}


class _VTKPropsFilter(logging.Filter):
    """Filter out the non-fatal 'Too many props' VTK error from Python logging."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "Too many props" not in record.getMessage()


class UndoEntry:
    """Single undo/redo state."""
    def __init__(self, indices: np.ndarray, old_labels: np.ndarray, new_label: int):
        self.indices = indices.copy()
        self.old_labels = old_labels.copy()
        self.new_label = new_label


class LabelEditor(QWidget):
    """In-app label correction tool for training data preparation."""

    labels_changed = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        if not HAS_PYVISTA:
            raise ImportError("pyvista and pyvistaqt required for label editor")

        self._points: Optional[np.ndarray] = None
        self._labels: Optional[np.ndarray] = None
        self._confidence: Optional[np.ndarray] = None
        self._selected_indices: Optional[np.ndarray] = None
        self._current_class: int = 1
        self._undo_stack: list[UndoEntry] = []
        self._redo_stack: list[UndoEntry] = []
        self._brush_radius: float = 0.5
        self._picking_active: bool = False
        self._brush_mode: bool = False
        self._focus_mode: bool = False
        self._show_confidence: bool = False
        self._kdtree = None

        self._setup_ui()
        self._setup_shortcuts()

    def _setup_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Left toolbar
        toolbar = QVBoxLayout()
        toolbar.setContentsMargins(8, 8, 8, 8)

        # Navigation
        nav_group = QGroupBox("Navigation")
        nav_layout = QVBoxLayout(nav_group)
        nav_label = QLabel(
            "Left-drag: Rotate\n"
            "Middle-drag / Shift+Left: Pan\n"
            "Scroll: Zoom"
        )
        nav_label.setWordWrap(True)
        nav_layout.addWidget(nav_label)

        self._focus_btn = QPushButton("Set Focus (F)")
        self._focus_btn.setToolTip(
            "Right-click a point to set the camera orbit centre.\n"
            "Rotation will pivot around the picked point."
        )
        self._focus_btn.setCheckable(True)
        self._focus_btn.toggled.connect(self._on_focus_toggled)
        nav_layout.addWidget(self._focus_btn)

        self._edl_btn = QPushButton("EDL")
        self._edl_btn.setToolTip("Eye-Dome Lighting — enhances depth perception")
        self._edl_btn.setCheckable(True)
        self._edl_btn.setChecked(True)
        self._edl_btn.toggled.connect(self._on_edl_toggled)
        nav_layout.addWidget(self._edl_btn)

        # Camera views row
        views_row = QHBoxLayout()
        for label, view_id in [("Top", "top"), ("Front", "front"), ("Right", "right"), ("Iso", "iso")]:
            btn = QPushButton(label)
            btn.clicked.connect(lambda checked=False, v=view_id: self._set_camera_view(v))
            views_row.addWidget(btn)
        nav_layout.addLayout(views_row)

        reset_btn = QPushButton("Reset View (Home)")
        reset_btn.clicked.connect(self._reset_view)
        nav_layout.addWidget(reset_btn)

        toolbar.addWidget(nav_group)

        # Class selection + visibility
        class_group = QGroupBox("Class")
        class_layout = QVBoxLayout(class_group)
        self._class_buttons = QButtonGroup(self)
        self._visibility_checks: dict[int, QCheckBox] = {}

        for class_id, (name, color) in CLASSES.items():
            row = QHBoxLayout()
            vis = QCheckBox()
            vis.setChecked(True)
            vis.setToolTip(f"Show/hide {name} points")
            vis.toggled.connect(self._on_visibility_changed)
            self._visibility_checks[class_id] = vis
            row.addWidget(vis)

            btn = QRadioButton(f"{class_id}: {name}")
            btn.setStyleSheet(f"QRadioButton {{ color: {color}; font-weight: bold; }}")
            self._class_buttons.addButton(btn, class_id)
            row.addWidget(btn)
            class_layout.addLayout(row)
            if class_id == 1:
                btn.setChecked(True)

        self._class_buttons.idClicked.connect(self._on_class_changed)
        toolbar.addWidget(class_group)

        # Selection tool
        tool_group = QGroupBox("Selection")
        tool_layout = QVBoxLayout(tool_group)

        self._select_btn = QPushButton("Enable Box Select (R)")
        self._select_btn.setCheckable(True)
        self._select_btn.setChecked(False)
        self._select_btn.toggled.connect(self._toggle_picking)
        tool_layout.addWidget(self._select_btn)

        self._brush_btn = QPushButton("Enable Brush (B)")
        self._brush_btn.setToolTip("Click on points to select nearby points within the brush radius")
        self._brush_btn.setCheckable(True)
        self._brush_btn.setChecked(False)
        self._brush_btn.toggled.connect(self._toggle_brush)
        tool_layout.addWidget(self._brush_btn)

        brush_row = QHBoxLayout()
        brush_row.addWidget(QLabel("Radius:"))
        self._brush_spin = QDoubleSpinBox()
        self._brush_spin.setRange(0.01, 10.0)
        self._brush_spin.setValue(0.5)
        self._brush_spin.setSingleStep(0.1)
        self._brush_spin.setSuffix(" m")
        self._brush_spin.valueChanged.connect(self._on_brush_radius_changed)
        brush_row.addWidget(self._brush_spin)
        tool_layout.addLayout(brush_row)

        toolbar.addWidget(tool_group)

        # Confidence visualization
        conf_group = QGroupBox("Confidence")
        conf_layout = QVBoxLayout(conf_group)

        self._conf_btn = QPushButton("Color by Confidence (C)")
        self._conf_btn.setToolTip(
            "Color points by model confidence instead of class.\n"
            "Green = high confidence (likely correct)\n"
            "Yellow = medium confidence (spot-check)\n"
            "Red = low confidence (likely wrong — correct these first)"
        )
        self._conf_btn.setCheckable(True)
        self._conf_btn.toggled.connect(self._on_confidence_toggled)
        self._conf_btn.setEnabled(False)
        conf_layout.addWidget(self._conf_btn)

        thresh_row = QHBoxLayout()
        thresh_row.addWidget(QLabel("Threshold:"))
        self._conf_threshold = QDoubleSpinBox()
        self._conf_threshold.setRange(0.0, 1.0)
        self._conf_threshold.setValue(0.5)
        self._conf_threshold.setSingleStep(0.05)
        self._conf_threshold.setToolTip(
            "Points below this confidence are selected by\n"
            "'Select Low Confidence'. Default 0.5 = 50%."
        )
        thresh_row.addWidget(self._conf_threshold)
        conf_layout.addLayout(thresh_row)

        self._select_low_conf_btn = QPushButton("Select Low Confidence")
        self._select_low_conf_btn.setToolTip(
            "Auto-select all points with confidence below the threshold.\n"
            "These are the points most likely to be misclassified."
        )
        self._select_low_conf_btn.clicked.connect(self._select_low_confidence)
        self._select_low_conf_btn.setEnabled(False)
        conf_layout.addWidget(self._select_low_conf_btn)

        self._conf_stats = QLabel("")
        conf_layout.addWidget(self._conf_stats)

        toolbar.addWidget(conf_group)

        # Actions
        action_group = QGroupBox("Actions")
        action_layout = QVBoxLayout(action_group)

        self._paint_btn = QPushButton("Paint Selected")
        self._paint_btn.setToolTip(
            "Assign the selected class to all highlighted (white) points.\n"
            "Shortcut: press 1-4 to paint selection with that class directly."
        )
        self._paint_btn.clicked.connect(self._paint_selected)
        self._paint_btn.setEnabled(False)
        action_layout.addWidget(self._paint_btn)

        self._clear_sel_btn = QPushButton("Clear Selection (Esc)")
        self._clear_sel_btn.setToolTip("Deselect all highlighted points without painting.")
        self._clear_sel_btn.clicked.connect(self._clear_selection)
        self._clear_sel_btn.setEnabled(False)
        action_layout.addWidget(self._clear_sel_btn)

        undo_btn = QPushButton("Undo (Ctrl+Z)")
        undo_btn.clicked.connect(self.undo)
        action_layout.addWidget(undo_btn)

        redo_btn = QPushButton("Redo (Ctrl+Y)")
        redo_btn.clicked.connect(self.redo)
        action_layout.addWidget(redo_btn)

        save_btn = QPushButton("Save Labels...")
        save_btn.clicked.connect(self._save_labels)
        action_layout.addWidget(save_btn)

        toolbar.addWidget(action_group)

        # Stats
        self._stats_label = QLabel("No data loaded")
        toolbar.addWidget(self._stats_label)

        toolbar.addStretch()

        toolbar_widget = QWidget()
        toolbar_widget.setLayout(toolbar)

        scroll = QScrollArea()
        scroll.setWidget(toolbar_widget)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(200)

        # 3D viewer — start in navigation mode (no picking)
        pv.global_theme.background = "#1a2e26"
        self._plotter = QtInteractor(self)
        self._plotter.set_background("#1a2e26")
        self._plotter.enable_eye_dome_lighting()

        # Suppress VTK hardware selector "Too many props" error — non-fatal
        # on large point clouds, picking still works despite the warning.
        import vtk
        vtk.vtkObject.GlobalWarningDisplayOff()
        # Also suppress the Python logging version of the same error
        logging.getLogger("vtkmodules.vtkRenderingOpenGL2").setLevel(logging.CRITICAL)
        logging.getLogger().addFilter(_VTKPropsFilter())

        # Use a splitter so the user can drag to resize the sidebar
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(scroll)
        splitter.addWidget(self._plotter.interactor)
        splitter.setStretchFactor(0, 0)  # sidebar doesn't stretch
        splitter.setStretchFactor(1, 1)  # viewer takes remaining space
        splitter.setSizes([380, 820])    # initial sidebar 380px
        layout.addWidget(splitter)

    def _toggle_picking(self, enabled: bool) -> None:
        """Toggle between navigation and box-select modes."""
        self._picking_active = enabled
        if enabled:
            # Deactivate focus and brush modes — only one picking mode at a time
            if self._focus_mode:
                self._focus_btn.setChecked(False)
            if self._brush_mode:
                self._brush_btn.setChecked(False)
            self._select_btn.setText("Disable Box Select (R)")
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message=".*orig_extract_id.*", category=DeprecationWarning
                )
                self._plotter.enable_rectangle_through_picking(
                    callback=self._on_box_select, start=True, show_message=False,
                )
        else:
            self._select_btn.setText("Enable Box Select (R)")
            self._plotter.disable_picking()
            # Restore standard 3D navigation (trackball rotate/pan/zoom)
            self._plotter.enable_trackball_style()

    def _toggle_brush(self, enabled: bool) -> None:
        """Toggle brush selection mode."""
        self._brush_mode = enabled
        if enabled:
            # Deactivate box select and focus mode
            if self._picking_active:
                self._select_btn.setChecked(False)
            if self._focus_mode:
                self._focus_btn.setChecked(False)
            self._brush_btn.setText("Disable Brush (B)")
            # Build KDTree if needed
            if self._points is not None and self._kdtree is None:
                from scipy.spatial import cKDTree
                self._kdtree = cKDTree(self._points)
            self._plotter.enable_surface_point_picking(
                callback=self._on_brush_pick,
                show_message=False,
                show_point=False,
                picker="cell",
            )
        else:
            self._brush_btn.setText("Enable Brush (B)")
            self._plotter.disable_picking()
            self._plotter.enable_trackball_style()

    def _on_brush_pick(self, point: np.ndarray, *_args) -> None:
        """Handle a brush pick — select all points within brush radius."""
        if point is None or len(point) < 3 or self._kdtree is None:
            return
        indices = self._kdtree.query_ball_point(point[:3], r=self._brush_radius)
        if not indices:
            return
        new_indices = np.array(indices, dtype=int)
        # Merge with existing selection (additive)
        if self._selected_indices is not None:
            new_indices = np.unique(np.concatenate([self._selected_indices, new_indices]))
        self._selected_indices = new_indices
        self._paint_btn.setEnabled(True)
        self._clear_sel_btn.setEnabled(True)
        self._render()

    # --- Focus point ---

    def _on_focus_toggled(self, checked: bool) -> None:
        """Toggle focus-picking mode (click to set camera orbit centre)."""
        self._focus_mode = checked
        if checked:
            # Deactivate box select and brush — only one picking mode at a time
            if self._picking_active:
                self._select_btn.setChecked(False)
            if self._brush_mode:
                self._brush_btn.setChecked(False)
            self._plotter.enable_surface_point_picking(
                callback=self._on_point_picked_for_focus,
                show_message=False,
                show_point=True,
                color="yellow",
                point_size=12,
                picker="cell",
            )
        else:
            self._plotter.disable_picking()
            self._plotter.enable_trackball_style()

    def _on_point_picked_for_focus(self, point: np.ndarray, *_args) -> None:
        """Set the camera focal point to the picked point."""
        if point is not None and len(point) >= 3:
            self._plotter.set_focus(point[:3])

    # --- EDL ---

    def _on_edl_toggled(self, checked: bool) -> None:
        if checked:
            self._plotter.enable_eye_dome_lighting()
        else:
            self._release_edl()
        self._plotter.render()

    def _release_edl(self) -> None:
        """Disable EDL, releasing GPU resources first to avoid VTK warnings."""
        try:
            edl_pass = self._plotter.renderer._render_passes._edl_pass
            if edl_pass is not None:
                edl_pass.ReleaseGraphicsResources(self._plotter.render_window)
        except Exception:
            pass
        self._plotter.disable_eye_dome_lighting()

    # --- Camera views ---

    def _reset_view(self) -> None:
        if self._plotter and self._points is not None:
            self._plotter.suppress_rendering = True
            center = self._points.mean(axis=0)
            self._plotter.set_focus(center)
            self._plotter.reset_camera()
            self._plotter.suppress_rendering = False
            self._plotter.render()

    def _set_camera_view(self, view: str) -> None:
        """Switch camera direction while preserving focal point and zoom distance."""
        if self._points is None:
            return
        cam = self._plotter.camera
        focal = np.array(cam.focal_point)
        dist = cam.distance

        if view == "top":
            position = focal + np.array([0.0, 0.0, dist])
            viewup = (0, 1, 0)
        elif view == "front":
            position = focal + np.array([0.0, -dist, 0.0])
            viewup = (0, 0, 1)
        elif view == "right":
            position = focal + np.array([dist, 0.0, 0.0])
            viewup = (0, 0, 1)
        elif view == "iso":
            d = dist / np.sqrt(3)
            position = focal + np.array([d, d, d])
            viewup = (0, 0, 1)
        else:
            return

        self._plotter.camera_position = [tuple(position), tuple(focal), viewup]
        # Restore 3D trackball interaction so user can still pan/rotate
        self._plotter.enable_trackball_style()
        self._plotter.render()

    def _setup_shortcuts(self) -> None:
        for class_id in CLASSES:
            shortcut = QShortcut(QKeySequence(str(class_id)), self)
            shortcut.activated.connect(lambda cid=class_id: self._quick_paint(cid))

        QShortcut(QKeySequence("Ctrl+Z"), self).activated.connect(self.undo)
        QShortcut(QKeySequence("Ctrl+Y"), self).activated.connect(self.redo)
        QShortcut(QKeySequence("R"), self).activated.connect(
            lambda: self._select_btn.setChecked(not self._select_btn.isChecked())
        )
        QShortcut(QKeySequence("B"), self).activated.connect(
            lambda: self._brush_btn.setChecked(not self._brush_btn.isChecked())
        )
        QShortcut(QKeySequence("F"), self).activated.connect(
            lambda: self._focus_btn.setChecked(not self._focus_btn.isChecked())
        )
        QShortcut(QKeySequence("Home"), self).activated.connect(self._reset_view)
        QShortcut(QKeySequence("C"), self).activated.connect(
            lambda: self._conf_btn.isEnabled() and self._conf_btn.setChecked(not self._conf_btn.isChecked())
        )
        QShortcut(QKeySequence("Escape"), self).activated.connect(self._clear_selection)

    def _clear_selection(self) -> None:
        """Deselect all points without painting."""
        if self._selected_indices is None:
            return
        self._selected_indices = None
        self._paint_btn.setEnabled(False)
        self._clear_sel_btn.setEnabled(False)
        self._render()

    def load_points(
        self,
        points: np.ndarray,
        labels: Optional[np.ndarray] = None,
        confidence: Optional[np.ndarray] = None,
    ) -> None:
        """Load points for label editing.

        Args:
            points: Nx3 array of XYZ.
            labels: N array of class labels (1-4). If None, defaults to Vegetation (2).
            confidence: N array of model confidence (0-1). If None, confidence features disabled.
        """
        self._points = points[:, :3].astype(np.float32)

        if labels is not None:
            self._labels = labels.astype(np.int32).copy()
            # Auto-detect 0-indexed labels from inference (0-3) and convert to 1-4
            if 0 in self._labels and 4 not in self._labels:
                self._labels += 1
        else:
            self._labels = np.full(self._points.shape[0], 2, dtype=np.int32)

        if confidence is not None:
            self._confidence = confidence.astype(np.float32).copy()
            self._conf_btn.setEnabled(True)
            self._select_low_conf_btn.setEnabled(True)
            low = np.sum(self._confidence < 0.5)
            self._conf_stats.setText(
                f"Mean: {self._confidence.mean():.2f}\n"
                f"<0.5: {low:,} ({100*low/len(self._confidence):.1f}%)"
            )
        else:
            self._confidence = None
            self._conf_btn.setEnabled(False)
            self._conf_btn.setChecked(False)
            self._select_low_conf_btn.setEnabled(False)
            self._conf_stats.setText("No confidence data")

        self._selected_indices = None
        self._undo_stack.clear()
        self._redo_stack.clear()
        # Build KDTree for brush selection
        from scipy.spatial import cKDTree
        self._kdtree = cKDTree(self._points)

        self._render(reset_camera=True)
        self._update_stats()

    def _on_visibility_changed(self) -> None:
        """Re-render when layer visibility changes."""
        self._render()

    def _on_confidence_toggled(self, checked: bool) -> None:
        """Toggle between class colors and confidence heatmap."""
        self._show_confidence = checked
        self._render()

    def _select_low_confidence(self) -> None:
        """Select all points with confidence below the threshold."""
        if self._confidence is None or self._labels is None:
            return
        threshold = self._conf_threshold.value()
        vis_mask = self._get_visible_mask()
        low_conf = self._confidence < threshold
        combined = vis_mask & low_conf
        indices = np.where(combined)[0]
        if len(indices) == 0:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(
                self, "No Points Found",
                f"No visible points have confidence below {threshold:.2f}."
            )
            return
        self._selected_indices = indices
        self._paint_btn.setEnabled(True)
        self._clear_sel_btn.setEnabled(True)
        self._render()

    @staticmethod
    def _confidence_to_rgb(conf: np.ndarray) -> np.ndarray:
        """Map confidence values (0-1) to a red-yellow-green gradient.

        0.0 → red (255, 0, 0)
        0.5 → yellow (255, 255, 0)
        1.0 → green (0, 200, 0)
        """
        colors = np.zeros((len(conf), 3), dtype=np.uint8)
        # Red channel: full at 0, fades above 0.5
        colors[:, 0] = np.clip((1.0 - np.maximum(conf - 0.5, 0) * 2) * 255, 0, 255).astype(np.uint8)
        # Green channel: rises from 0 to 0.5, full above 0.5
        colors[:, 1] = np.clip(np.minimum(conf * 2, 1.0) * 200, 0, 200).astype(np.uint8)
        # Blue: always 0
        return colors

    def _get_visible_mask(self) -> np.ndarray:
        """Return boolean mask of points whose class is currently visible."""
        mask = np.zeros(self._labels.shape[0], dtype=bool)
        for class_id, chk in self._visibility_checks.items():
            if chk.isChecked():
                mask |= self._labels == class_id
        return mask

    def _render(self, *, reset_camera: bool = False) -> None:
        if self._points is None or self._labels is None:
            return

        # Preserve camera across re-render so the view doesn't jump
        saved_camera = None
        if not reset_camera:
            try:
                saved_camera = self._plotter.camera_position
            except Exception:
                pass

        self._plotter.clear()

        # Filter to visible classes only
        vis_mask = self._get_visible_mask()
        vis_points = self._points[vis_mask]
        vis_labels = self._labels[vis_mask]

        if vis_points.shape[0] > 0:
            cloud = pv.PolyData(vis_points)

            if self._show_confidence and self._confidence is not None:
                # Confidence heatmap: red → yellow → green
                vis_conf = self._confidence[vis_mask]
                colors = self._confidence_to_rgb(vis_conf)
            else:
                # Build colors from labels
                colors = np.zeros((vis_points.shape[0], 3), dtype=np.uint8)
                for class_id, (_, hex_color) in CLASSES.items():
                    mask = vis_labels == class_id
                    r = int(hex_color[1:3], 16)
                    g = int(hex_color[3:5], 16)
                    b = int(hex_color[5:7], 16)
                    colors[mask] = [r, g, b]

            cloud["RGB"] = colors
            self._plotter.add_mesh(
                cloud, scalars="RGB", rgb=True,
                point_size=3, render_points_as_spheres=False,
            )

        # Highlight selection (only visible selected points)
        if self._selected_indices is not None and len(self._selected_indices) > 0:
            sel_vis = self._selected_indices[vis_mask[self._selected_indices]]
            if len(sel_vis) > 0:
                sel_cloud = pv.PolyData(self._points[sel_vis])
                self._plotter.add_mesh(
                    sel_cloud, color="white", point_size=5,
                    render_points_as_spheres=True, opacity=0.7,
                )

        # Restore EDL state — plotter.clear() resets render passes
        if self._edl_btn.isChecked():
            self._plotter.enable_eye_dome_lighting()
        else:
            self._release_edl()

        # Restore camera or reset on first load
        if saved_camera is not None:
            self._plotter.camera_position = saved_camera
        else:
            self._plotter.reset_camera()

    def _on_box_select(self, selected) -> None:
        """Handle rectangle-through picking selection."""
        if self._points is None or selected is None:
            return

        # PyVista 0.47+ passes an UnstructuredGrid; extract points
        if hasattr(selected, "points"):
            pts = np.asarray(selected.points)
        else:
            pts = np.asarray(selected)

        # Ensure pts is always 2D (N, 3+)
        if pts.ndim == 1:
            if pts.shape[0] >= 3:
                pts = pts.reshape(-1, 3)
            else:
                return

        if pts.shape[0] == 0:
            return

        # Find which of our points are in the selection
        from scipy.spatial import cKDTree
        tree = cKDTree(self._points)
        _, indices = tree.query(pts[:, :3], k=1)
        self._selected_indices = np.unique(indices)
        self._paint_btn.setEnabled(True)
        self._clear_sel_btn.setEnabled(True)
        self._render()

    def _on_class_changed(self, class_id: int) -> None:
        self._current_class = class_id

    def _on_brush_radius_changed(self, value: float) -> None:
        self._brush_radius = value

    def _paint_selected(self) -> None:
        """Apply current class to selected points."""
        if self._selected_indices is None or self._labels is None:
            return

        old_labels = self._labels[self._selected_indices].copy()
        self._undo_stack.append(UndoEntry(self._selected_indices, old_labels, self._current_class))
        self._redo_stack.clear()

        self._labels[self._selected_indices] = self._current_class
        self._selected_indices = None
        self._paint_btn.setEnabled(False)
        self._clear_sel_btn.setEnabled(False)
        self._render()
        self._update_stats()
        self.labels_changed.emit()

    def _quick_paint(self, class_id: int) -> None:
        """Quick paint: set class and immediately paint selection."""
        self._current_class = class_id
        btn = self._class_buttons.button(class_id)
        if btn:
            btn.setChecked(True)
        self._paint_selected()

    def undo(self) -> None:
        if not self._undo_stack or self._labels is None:
            return
        entry = self._undo_stack.pop()
        current_labels = self._labels[entry.indices].copy()
        self._redo_stack.append(UndoEntry(entry.indices, current_labels, entry.new_label))
        self._labels[entry.indices] = entry.old_labels
        self._render()
        self._update_stats()
        self.labels_changed.emit()

    def redo(self) -> None:
        if not self._redo_stack or self._labels is None:
            return
        entry = self._redo_stack.pop()
        old_labels = self._labels[entry.indices].copy()
        self._undo_stack.append(UndoEntry(entry.indices, old_labels, entry.new_label))
        self._labels[entry.indices] = entry.new_label
        self._render()
        self._update_stats()
        self.labels_changed.emit()

    def _update_stats(self) -> None:
        if self._labels is None:
            self._stats_label.setText("No data loaded")
            return

        lines = [f"Total: {self._labels.shape[0]:,}"]
        for class_id, (name, _) in CLASSES.items():
            count = np.sum(self._labels == class_id)
            lines.append(f"{name}: {count:,}")
        self._stats_label.setText("\n".join(lines))

    def get_labeled_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the current points and labels."""
        return self._points.copy(), self._labels.copy()

    def _save_labels(self) -> None:
        if self._points is None:
            return
        from PySide6.QtWidgets import QFileDialog
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Labeled Point Cloud", "",
            "LAS Files (*.las);;LAZ Compressed (*.laz);;All Files (*)",
        )
        if filepath:
            if not filepath.endswith((".las", ".laz")):
                filepath += ".las"
            self.export_labeled_las(filepath)

    def export_labeled_las(self, filepath: str) -> None:
        """Export labeled point cloud as LAS file."""
        import sys
        from pathlib import Path
        scripts_dir = str(Path(__file__).parent.parent.parent.parent / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from tools import save_file

        columns = [self._points, self._labels.astype(np.float64)]
        headers = ["x", "y", "z", "label"]
        if self._confidence is not None:
            columns.append(self._confidence.astype(np.float64))
            headers.append("confidence")
        data = np.column_stack(columns)
        save_file(filepath, data, headers_of_interest=headers)

    def closeEvent(self, event) -> None:
        """Clean up the plotter to prevent crashes on close."""
        try:
            self._plotter.disable_picking()
            self._plotter.clear()
            self._plotter.close()
        except Exception:
            pass
        super().closeEvent(event)
