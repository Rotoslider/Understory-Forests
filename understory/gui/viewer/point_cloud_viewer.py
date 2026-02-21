"""3D point cloud viewer with Level-of-Detail (LOD) for handling 500M+ points.

Uses PyVista + pyvistaqt for rendering inside PySide6.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
    from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLabel, QPushButton, QDoubleSpinBox
    from PySide6.QtCore import Signal, Qt
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False


class ColorMode(Enum):
    RGB = "RGB"
    HEIGHT = "Height Gradient"
    CLASSIFICATION = "Classification"
    TREE_ID = "Tree ID"
    COMPARISON = "Comparison"


class UnitSystem(Enum):
    METRIC = "Metric"
    IMPERIAL = "Imperial"


METERS_TO_FEET = 3.28084


class MeasureMode(Enum):
    OFF = "off"
    DISTANCE = "distance"
    HEIGHT = "height"


# Classification colors (terrain, vegetation, CWD, stem)
CLASS_COLORS = {
    0: [0.5, 0.5, 0.5],    # noise — grey
    1: [0.6, 0.4, 0.2],    # terrain — brown
    2: [0.2, 0.7, 0.2],    # vegetation — green
    3: [0.8, 0.7, 0.1],    # CWD — yellow
    4: [0.8, 0.2, 0.2],    # stem — red
}

# LOD thresholds
LOD_LEVELS = {
    0: 1_000_000,    # overview: ~1M points
    1: 5_000_000,    # medium: ~5M points
    2: 20_000_000,   # close: ~20M points
}


@dataclass
class PrepareSnapshot:
    """Snapshot of viewer state for undo/redo."""
    points: np.ndarray
    colors: Optional[np.ndarray]
    labels: Optional[np.ndarray]
    tree_ids: Optional[np.ndarray]
    description: str


class PointCloudViewer(QWidget):
    """3D point cloud viewer with LOD and multiple color modes."""

    point_picked = Signal(int, float, float, float)  # index, x, y, z
    plot_centre_dragged = Signal(float, float)  # x, y from interactive widget
    crop_state_changed = Signal(bool)  # True when cropped, False when reset
    trim_region_selected = Signal()  # emitted when user finishes rectangle selection

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        if not HAS_PYVISTA:
            raise ImportError("pyvista and pyvistaqt are required for the point cloud viewer")

        self._points_full: Optional[np.ndarray] = None  # full resolution XYZ
        self._points_original: Optional[np.ndarray] = None  # original orientation
        self._colors_full: Optional[np.ndarray] = None  # full resolution RGB (0-1)
        self._colors_original: Optional[np.ndarray] = None  # pre-crop colors for reset
        self._labels: Optional[np.ndarray] = None        # classification labels
        self._labels_original: Optional[np.ndarray] = None  # pre-crop labels for reset
        self._tree_ids: Optional[np.ndarray] = None       # tree IDs
        self._tree_ids_original: Optional[np.ndarray] = None  # pre-crop tree IDs for reset
        self._current_lod: int = 0
        self._lod_indices: Optional[np.ndarray] = None
        self._color_mode: ColorMode = ColorMode.RGB
        self._plot_circle: Optional[pv.PolyData] = None
        self._crop_mask: Optional[np.ndarray] = None  # boolean mask for outlier crop
        self._focus_mode: bool = False
        self._plot_circle_widget_active: bool = False
        self._dragging_circle: bool = False
        self._circle_actor = None

        self._undo_stack: list[PrepareSnapshot] = []
        self._redo_stack: list[PrepareSnapshot] = []
        self._max_undo: int = 5

        self._slice_mode: str = "off"  # "off", "horizontal", "vertical_x", "vertical_y"
        self._slice_pos: float = 0.0
        self._slice_thickness: float = 2.0

        # Measurement state
        self._measure_mode: MeasureMode = MeasureMode.OFF
        self._measure_point_a: Optional[np.ndarray] = None
        self._measure_actors: list = []

        # Comparison state
        self._comparison_distances: Optional[np.ndarray] = None

        # Trim selection state
        self._trim_selection: Optional[np.ndarray] = None  # indices into _points_full
        self._trim_active: bool = False

        # Unit system
        self._unit_system: UnitSystem = UnitSystem.METRIC

        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Toolbar — viewing controls only
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(4, 4, 4, 4)

        toolbar.addWidget(QLabel("Color:"))
        self._color_combo = QComboBox()
        for mode in ColorMode:
            self._color_combo.addItem(mode.value, mode)
        self._color_combo.currentIndexChanged.connect(self._on_color_mode_changed)
        toolbar.addWidget(self._color_combo)

        # EDL shader toggle
        self._edl_btn = QPushButton("EDL")
        self._edl_btn.setToolTip("Eye-Dome Lighting — enhances depth perception")
        self._edl_btn.setCheckable(True)
        self._edl_btn.setChecked(True)
        self._edl_btn.toggled.connect(self._on_edl_toggled)
        toolbar.addWidget(self._edl_btn)

        toolbar.addStretch()

        self._point_count_label = QLabel("No data loaded")
        toolbar.addWidget(self._point_count_label)

        toolbar.addStretch()

        # Focus point toggle
        self._focus_btn = QPushButton("Set Focus")
        self._focus_btn.setToolTip("Right-click on a point to set the focus point")
        self._focus_btn.setCheckable(True)
        self._focus_btn.toggled.connect(self._on_focus_toggled)
        toolbar.addWidget(self._focus_btn)

        toolbar.addStretch()

        # Camera views
        for label, view_id in [("Top", "top"), ("Front", "front"), ("Right", "right"), ("Iso", "iso")]:
            btn = QPushButton(label)
            btn.clicked.connect(lambda checked=False, v=view_id: self.set_camera_view(v))
            toolbar.addWidget(btn)

        toolbar.addWidget(QLabel("|"))
        toolbar.addWidget(QLabel("Slice:"))
        self._slice_combo = QComboBox()
        self._slice_combo.setMaximumWidth(95)
        self._slice_combo.addItem("Off", "off")
        self._slice_combo.addItem("Horiz", "horizontal")
        self._slice_combo.addItem("Vert X", "vertical_x")
        self._slice_combo.addItem("Vert Y", "vertical_y")
        self._slice_combo.currentIndexChanged.connect(self._on_slice_mode_changed)
        toolbar.addWidget(self._slice_combo)

        self._slice_pos_spin = QDoubleSpinBox()
        self._slice_pos_spin.setMaximumWidth(100)
        self._slice_pos_spin.setRange(-1e6, 1e6)
        self._slice_pos_spin.setDecimals(1)
        self._slice_pos_spin.setSingleStep(0.5)
        self._slice_pos_spin.setPrefix("P:")
        self._slice_pos_spin.setEnabled(False)
        self._slice_pos_spin.valueChanged.connect(self._on_slice_pos_changed)
        toolbar.addWidget(self._slice_pos_spin)

        self._slice_thick_spin = QDoubleSpinBox()
        self._slice_thick_spin.setMaximumWidth(90)
        self._slice_thick_spin.setRange(0.1, 100)
        self._slice_thick_spin.setDecimals(1)
        self._slice_thick_spin.setSingleStep(0.5)
        self._slice_thick_spin.setValue(2.0)
        self._slice_thick_spin.setPrefix("T:")
        self._slice_thick_spin.setSuffix("m")
        self._slice_thick_spin.setEnabled(False)
        self._slice_thick_spin.valueChanged.connect(self._on_slice_thick_changed)
        toolbar.addWidget(self._slice_thick_spin)

        self._reset_btn = QPushButton("Reset View")
        self._reset_btn.clicked.connect(self._reset_view)
        toolbar.addWidget(self._reset_btn)

        layout.addLayout(toolbar)

        # PyVista interactor
        pv.global_theme.background = "#1a2e26"
        pv.global_theme.font.color = "#a8d8c0"
        self._plotter = QtInteractor(self)
        self._plotter.set_background("#1a2e26")
        self._plotter.enable_eye_dome_lighting()
        layout.addWidget(self._plotter.interactor)

        # Listen for Escape key via VTK (QWidget.keyPressEvent won't fire
        # because the VTK interactor consumes key events first).
        self._plotter.iren.interactor.AddObserver("KeyPressEvent", self._on_vtk_key_press)

    def load_points(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        tree_ids: Optional[np.ndarray] = None,
    ) -> None:
        """Load a point cloud into the viewer.

        Args:
            points: Nx3 array of XYZ coordinates.
            colors: Nx3 array of RGB values (0-255 or 0-1 scale).
            labels: N array of classification labels.
            tree_ids: N array of tree IDs.
        """
        self._points_full = np.ascontiguousarray(points[:, :3].astype(np.float32))
        self._points_original = self._points_full.copy()

        if colors is not None:
            colors = colors.astype(np.float64)
            if colors.max() > 1.0:
                colors = colors / colors.max()
            self._colors_full = colors
        else:
            self._colors_full = None

        self._labels = labels
        self._tree_ids = tree_ids

        # Store originals for crop reset
        self._colors_original = self._colors_full.copy() if self._colors_full is not None else None
        self._labels_original = self._labels.copy() if self._labels is not None else None
        self._tree_ids_original = self._tree_ids.copy() if self._tree_ids is not None else None
        self._crop_mask = None

        n = self._points_full.shape[0]
        self._point_count_label.setText(f"{n:,} points loaded")

        # Build LOD indices
        self._build_lod()
        self._render()

    def _build_lod(self) -> None:
        """Build LOD index arrays via random subsampling, respecting crop mask."""
        if self._points_full is None:
            return

        # Get eligible indices (respecting crop mask)
        if self._crop_mask is not None:
            eligible = np.where(self._crop_mask)[0]
        else:
            eligible = np.arange(self._points_full.shape[0])

        # Apply cross-section slice filter
        axis = self._get_slice_axis()
        if axis is not None and self._points_full is not None:
            pts = self._points_full[eligible]
            half = self._slice_thickness / 2
            mask = np.abs(pts[:, axis] - self._slice_pos) < half
            eligible = eligible[mask]
            n = len(eligible)

        n = len(eligible)

        # Determine appropriate LOD level
        if n <= LOD_LEVELS[0]:
            self._current_lod = 2  # show all
            self._lod_indices = eligible
        elif n <= LOD_LEVELS[1]:
            self._current_lod = 1
            chosen = np.random.choice(n, size=min(n, LOD_LEVELS[1]), replace=False)
            self._lod_indices = eligible[chosen]
        elif n <= LOD_LEVELS[2]:
            self._current_lod = 1
            chosen = np.random.choice(n, size=LOD_LEVELS[1], replace=False)
            self._lod_indices = eligible[chosen]
        else:
            self._current_lod = 0
            chosen = np.random.choice(n, size=LOD_LEVELS[0], replace=False)
            self._lod_indices = eligible[chosen]

        self._lod_indices.sort()

    def _render(self, preserve_camera: bool = False) -> None:
        """Render the current LOD view.

        Args:
            preserve_camera: If True, save and restore the camera position
                instead of resetting it (useful for color mode change).
        """
        if self._points_full is None or self._lod_indices is None:
            return

        saved_camera = None
        if preserve_camera:
            try:
                saved_camera = self._plotter.camera_position
            except Exception:
                pass

        self._plotter.clear()

        pts = self._points_full[self._lod_indices]
        cloud = pv.PolyData(pts)

        scalars = self._get_scalars()
        kwargs = {"point_size": 2, "render_points_as_spheres": False}

        if scalars is not None and scalars.ndim == 2:
            # Direct RGB
            cloud["RGB"] = (scalars * 255).astype(np.uint8)
            kwargs["scalars"] = "RGB"
            kwargs["rgb"] = True
        elif scalars is not None:
            cloud["values"] = scalars
            kwargs["scalars"] = "values"
            if self._color_mode == ColorMode.COMPARISON:
                kwargs["cmap"] = "RdYlBu_r"  # blue=close, red=far
                kwargs["scalar_bar_args"] = {
                    "color": "#ffffff",
                    "title": "Distance (m)",
                    "title_font_size": 14,
                    "label_font_size": 12,
                    "shadow": True,
                    "fmt": "%.2f",
                }
            else:
                kwargs["cmap"] = "viridis"
                kwargs["scalar_bar_args"] = {
                    "color": "#ffffff",
                    "title_font_size": 14,
                    "label_font_size": 12,
                    "shadow": True,
                    "fmt": "%.1f",
                }
        else:
            kwargs["color"] = "#4a9e7e"

        self._plotter.add_mesh(cloud, **kwargs)

        # Add legend/annotation for current color mode
        if self._color_mode == ColorMode.CLASSIFICATION and self._labels is not None:
            CLASS_NAMES = {0: "Noise", 1: "Terrain", 2: "Vegetation", 3: "CWD", 4: "Stem"}
            legend_entries = []
            unique_labels = np.unique(self._labels[self._lod_indices].astype(int))
            for lbl in sorted(unique_labels):
                if lbl in CLASS_COLORS:
                    name = CLASS_NAMES.get(lbl, f"Class {lbl}")
                    legend_entries.append([name, CLASS_COLORS[lbl]])
            if legend_entries:
                self._plotter.add_legend(
                    legend_entries,
                    bcolor=(0.1, 0.18, 0.15, 0.8),
                    face="circle",
                    size=(0.15, 0.2),
                )
        elif self._color_mode == ColorMode.TREE_ID and self._tree_ids is not None:
            self._plotter.add_text(
                "Colored by Tree ID",
                position="upper_right",
                font_size=10,
                color="#a8d8c0",
                shadow=True,
            )

        # Re-add plot circle if set
        self._circle_actor = None
        if self._plot_circle is not None:
            self._circle_actor = self._plotter.add_mesh(
                self._plot_circle, color=self.PLOT_CIRCLE_COLOR, line_width=3,
            )

        n_displayed = len(self._lod_indices)
        n_total = self._points_full.shape[0]
        if n_displayed < n_total:
            pct = n_displayed / n_total * 100
            self._point_count_label.setText(
                f"{n_total:,} points ({n_displayed:,} displayed, {pct:.1f}%)"
            )
        else:
            self._point_count_label.setText(f"{n_total:,} points")

        # Apply EDL state — clear() does NOT reset render passes in PyVista 0.47,
        # so we explicitly sync state here.
        if self._edl_btn.isChecked():
            self._plotter.enable_eye_dome_lighting()
        else:
            self._release_edl()

        self._plotter.reset_camera()

        # Then restore saved camera if requested
        if saved_camera is not None:
            self._plotter.camera_position = saved_camera

    def _get_scalars(self) -> Optional[np.ndarray]:
        """Get scalar values for the current color mode and LOD subset."""
        idx = self._lod_indices

        if self._color_mode == ColorMode.RGB:
            if self._colors_full is not None:
                return self._colors_full[idx]
            return None

        elif self._color_mode == ColorMode.HEIGHT:
            z = self._points_full[idx, 2]
            return z

        elif self._color_mode == ColorMode.CLASSIFICATION:
            if self._labels is not None:
                # Convert labels to RGB colors using CLASS_COLORS map
                lbl = self._labels[idx].astype(int)
                colors = np.zeros((len(lbl), 3), dtype=np.float64)
                for class_id, color in CLASS_COLORS.items():
                    mask = lbl == class_id
                    if mask.any():
                        colors[mask] = color
                return colors
            return None

        elif self._color_mode == ColorMode.TREE_ID:
            if self._tree_ids is not None:
                return self._tree_ids[idx].astype(np.float32)
            return None

        elif self._color_mode == ColorMode.COMPARISON:
            if self._comparison_distances is not None:
                return self._comparison_distances[idx].astype(np.float32)
            return None

        return None

    def _on_color_mode_changed(self, index: int) -> None:
        self._color_mode = self._color_combo.itemData(index)
        self._render(preserve_camera=True)

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

    def _reset_view(self) -> None:
        if self._plotter:
            self._plotter.suppress_rendering = True
            if self._points_full is not None:
                center = self._points_full.mean(axis=0)
                self._plotter.set_focus(center)
            self._plotter.reset_camera()
            self._plotter.suppress_rendering = False
            self._plotter.render()

    # Plot circle color — bright magenta for high contrast on any palette
    PLOT_CIRCLE_COLOR = "#ff00ff"

    def show_plot_circle(self, centre_x: float, centre_y: float, radius: float, z: float = 0) -> None:
        """Display a circular plot boundary in the viewer."""
        theta = np.linspace(0, 2 * np.pi, 200)
        x = centre_x + radius * np.cos(theta)
        y = centre_y + radius * np.sin(theta)
        z_arr = np.full_like(x, z)
        pts = np.column_stack([x, y, z_arr])
        lines = np.zeros((199, 3), dtype=int)
        lines[:, 0] = 2
        lines[:, 1] = np.arange(199)
        lines[:, 2] = np.arange(1, 200)
        self._plot_circle = pv.PolyData(pts, lines=lines.ravel())
        if self._dragging_circle:
            # During drag, just update the circle actor without full re-render
            self._update_circle_actor()
        else:
            self._render()

    def _update_circle_actor(self) -> None:
        """Update just the circle mesh without clearing the whole scene."""
        if self._plot_circle is None or self._plotter is None:
            return
        if self._circle_actor is not None:
            self._plotter.remove_actor(self._circle_actor)
        self._circle_actor = self._plotter.add_mesh(
            self._plot_circle, color=self.PLOT_CIRCLE_COLOR, line_width=3,
        )
        self._plotter.render()

    def clear_plot_circle(self) -> None:
        if self._circle_actor is not None:
            self._plotter.remove_actor(self._circle_actor)
            self._circle_actor = None
        self._plot_circle = None
        self._render()

    def clear(self) -> None:
        """Clear all data from the viewer."""
        self._points_full = None
        self._points_original = None
        self._colors_full = None
        self._colors_original = None
        self._labels = None
        self._labels_original = None
        self._tree_ids = None
        self._tree_ids_original = None
        self._lod_indices = None
        self._plot_circle = None
        self._crop_mask = None
        self._comparison_distances = None
        self._dragging_circle = False
        self._circle_actor = None
        # Clear interactive widgets (plot circle sphere)
        if self._plot_circle_widget_active:
            try:
                self._plotter.clear_sphere_widgets()
            except Exception:
                pass
            self._plot_circle_widget_active = False
        # Cancel any active measurement
        self.cancel_measurement()
        self._plotter.clear()
        self._point_count_label.setText("No data loaded")

    def export_screenshot(self, filepath: str, scale: int = 2) -> str:
        """Save a screenshot of the current view.

        Args:
            filepath: Output file path (PNG, JPEG, or TIFF).
            scale: Resolution multiplier (default 2x).

        Returns:
            The filepath that was written.
        """
        self._plotter.screenshot(filepath, transparent_background=False, scale=scale)
        return filepath

    # --- Crop outliers ---

    def _crop_to_bounds(self) -> None:
        """Remove outlier points beyond the 99.5th percentile per axis."""
        if self._points_full is None:
            return
        self.push_undo("Crop outliers")

        pts = self._points_full
        mask = np.ones(pts.shape[0], dtype=bool)
        for axis in range(3):
            lo = np.percentile(pts[:, axis], 0.25)
            hi = np.percentile(pts[:, axis], 99.75)
            mask &= (pts[:, axis] >= lo) & (pts[:, axis] <= hi)

        # Actually remove outlier points from the data
        self._points_full = self._points_full[mask]
        if self._colors_full is not None:
            self._colors_full = self._colors_full[mask]
        if self._labels is not None:
            self._labels = self._labels[mask]
        if self._tree_ids is not None:
            self._tree_ids = self._tree_ids[mask]
        self._crop_mask = None

        self.crop_state_changed.emit(True)
        self._build_lod()
        self._render()

    def _reset_crop(self) -> None:
        """Restore the full unfiltered point cloud from originals."""
        if self._points_original is None:
            return
        self._points_full = self._points_original.copy()
        self._colors_full = self._colors_original.copy() if self._colors_original is not None else None
        self._labels = self._labels_original.copy() if self._labels_original is not None else None
        self._tree_ids = self._tree_ids_original.copy() if self._tree_ids_original is not None else None
        self._crop_mask = None
        self.crop_state_changed.emit(False)
        self._build_lod()
        self._render()

    # --- Focus point ---

    def _on_focus_toggled(self, checked: bool) -> None:
        # Cancel measurement mode if activating focus
        if checked and self._measure_mode != MeasureMode.OFF:
            self.cancel_measurement()
        self._focus_mode = checked
        if checked:
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

    def _on_point_picked_for_focus(self, point: np.ndarray, *_args) -> None:
        """Set the camera focal point to the picked point."""
        if point is not None and len(point) >= 3:
            self._plotter.set_focus(point[:3])
            self.point_picked.emit(-1, float(point[0]), float(point[1]), float(point[2]))

    # --- Axis swap ---

    def apply_axis_swap(self, mode: str) -> None:
        """Apply an axis transformation to the point cloud.

        Args:
            mode: One of 'yz', 'xz', 'xy', 'rot90z', 'reset'.
        """
        if self._points_original is None:
            return
        self.push_undo(f"Axis swap: {mode}")

        if mode == "reset":
            self._points_full = self._points_original.copy()
        elif mode == "yz":
            self._points_full = self._points_original[:, [0, 2, 1]].copy()
        elif mode == "xz":
            self._points_full = self._points_original[:, [2, 1, 0]].copy()
        elif mode == "xy":
            self._points_full = self._points_original[:, [1, 0, 2]].copy()
        elif mode == "rot90z":
            pts = self._points_original.copy()
            x, y = pts[:, 0].copy(), pts[:, 1].copy()
            pts[:, 0] = y
            pts[:, 1] = -x
            self._points_full = pts

        self._points_full = np.ascontiguousarray(self._points_full)
        self._crop_mask = None
        self.crop_state_changed.emit(False)
        self._build_lod()
        self._render()

    # --- Interactive plot circle ---

    def enable_plot_circle_interaction(self, centre_x: float, centre_y: float, radius: float, z: float = 0) -> None:
        """Enable a draggable sphere widget to move the plot circle centre."""
        if self._plotter is None:
            return

        # Clear any existing widgets first
        if self._plot_circle_widget_active:
            self._plotter.clear_sphere_widgets()

        self._plot_circle_radius = radius
        self._plot_circle_z = z
        self._plot_circle_widget_active = True

        # Use a visible handle size — at least 1m or 8% of radius
        handle_radius = max(radius * 0.08, 1.0)
        self._plotter.add_sphere_widget(
            callback=self._on_plot_circle_widget_moved,
            center=(centre_x, centre_y, z),
            radius=handle_radius,
            color=self.PLOT_CIRCLE_COLOR,
            style="wireframe",
            interaction_event="always",
        )

    def disable_plot_circle_interaction(self) -> None:
        """Disable the interactive plot circle widget."""
        if self._plotter and self._plot_circle_widget_active:
            self._plotter.clear_sphere_widgets()
            self._plot_circle_widget_active = False
            self._dragging_circle = False

    def _on_plot_circle_widget_moved(self, point: np.ndarray) -> None:
        """Callback when the plot circle sphere widget is dragged."""
        cx, cy = float(point[0]), float(point[1])
        z = getattr(self, "_plot_circle_z", 0)
        radius = getattr(self, "_plot_circle_radius", 10)
        self._dragging_circle = True
        self.show_plot_circle(cx, cy, radius, z)
        self._dragging_circle = False
        self.plot_centre_dragged.emit(cx, cy)

    # --- Undo / Redo ---

    def push_undo(self, description: str = "") -> None:
        """Save the current state to the undo stack before a destructive operation."""
        if self._points_full is None:
            return
        snapshot = PrepareSnapshot(
            points=self._points_full.copy(),
            colors=self._colors_full.copy() if self._colors_full is not None else None,
            labels=self._labels.copy() if self._labels is not None else None,
            tree_ids=self._tree_ids.copy() if self._tree_ids is not None else None,
            description=description,
        )
        self._undo_stack.append(snapshot)
        if len(self._undo_stack) > self._max_undo:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def undo(self) -> Optional[str]:
        """Restore the previous state. Returns description of undone action, or None."""
        if not self._undo_stack:
            return None
        # Save current state to redo stack
        if self._points_full is not None:
            redo_snap = PrepareSnapshot(
                points=self._points_full.copy(),
                colors=self._colors_full.copy() if self._colors_full is not None else None,
                labels=self._labels.copy() if self._labels is not None else None,
                tree_ids=self._tree_ids.copy() if self._tree_ids is not None else None,
                description="",
            )
            self._redo_stack.append(redo_snap)

        snap = self._undo_stack.pop()
        self._points_full = snap.points
        self._colors_full = snap.colors
        self._labels = snap.labels
        self._tree_ids = snap.tree_ids
        self._crop_mask = None
        self._build_lod()
        self._render()
        return snap.description

    def redo(self) -> Optional[str]:
        """Re-apply the last undone action. Returns description, or None."""
        if not self._redo_stack:
            return None
        # Save current to undo
        if self._points_full is not None:
            undo_snap = PrepareSnapshot(
                points=self._points_full.copy(),
                colors=self._colors_full.copy() if self._colors_full is not None else None,
                labels=self._labels.copy() if self._labels is not None else None,
                tree_ids=self._tree_ids.copy() if self._tree_ids is not None else None,
                description="",
            )
            self._undo_stack.append(undo_snap)

        snap = self._redo_stack.pop()
        self._points_full = snap.points
        self._colors_full = snap.colors
        self._labels = snap.labels
        self._tree_ids = snap.tree_ids
        self._crop_mask = None
        self._build_lod()
        self._render()
        return snap.description

    @property
    def can_undo(self) -> bool:
        return len(self._undo_stack) > 0

    @property
    def can_redo(self) -> bool:
        return len(self._redo_stack) > 0

    # --- Cross-section slice ---

    def _on_slice_mode_changed(self, index: int) -> None:
        mode = self._slice_combo.itemData(index)
        self._slice_mode = mode
        enabled = mode != "off"
        self._slice_pos_spin.setEnabled(enabled)
        self._slice_thick_spin.setEnabled(enabled)

        # Auto-set position to midpoint of point cloud along slice axis
        if enabled and self._points_full is not None:
            axis = self._get_slice_axis()
            if axis is not None:
                mid = float(np.median(self._points_full[:, axis]))
                self._slice_pos_spin.blockSignals(True)
                self._slice_pos_spin.setValue(mid)
                self._slice_pos_spin.blockSignals(False)
                self._slice_pos = mid

        self._build_lod()
        self._render(preserve_camera=True)

    def _on_slice_pos_changed(self, value: float) -> None:
        self._slice_pos = value
        self._build_lod()
        self._render(preserve_camera=True)

    def _on_slice_thick_changed(self, value: float) -> None:
        self._slice_thickness = value
        self._build_lod()
        self._render(preserve_camera=True)

    def _get_slice_axis(self) -> Optional[int]:
        """Return the axis index for the current slice mode, or None."""
        if self._slice_mode == "horizontal":
            return 2  # Z axis
        elif self._slice_mode == "vertical_x":
            return 0  # X axis
        elif self._slice_mode == "vertical_y":
            return 1  # Y axis
        return None

    # --- Camera views ---

    def set_camera_view(self, view: str) -> None:
        """Set the camera to a predefined view.

        Args:
            view: One of 'top', 'front', 'right', 'iso'.
        """
        if view == "top":
            self._plotter.view_xy()
        elif view == "front":
            self._plotter.view_xz()
        elif view == "right":
            self._plotter.view_yz()
        elif view == "iso":
            self._plotter.view_isometric()

    # --- Unit system ---

    @property
    def unit_factor(self) -> float:
        return METERS_TO_FEET if self._unit_system == UnitSystem.IMPERIAL else 1.0

    @property
    def unit_suffix(self) -> str:
        return "ft" if self._unit_system == UnitSystem.IMPERIAL else "m"

    def set_unit_system(self, system: UnitSystem) -> None:
        self._unit_system = system

    # --- Measurement tools ---

    def start_measurement(self, mode: str) -> None:
        """Start a measurement interaction.

        Args:
            mode: One of 'distance', 'height'.
        """
        # Disable focus mode if active (avoids conflicting pickers)
        if self._focus_mode:
            self._focus_btn.setChecked(False)

        self.cancel_measurement()
        try:
            self._measure_mode = MeasureMode(mode)
        except ValueError:
            return
        self._measure_point_a = None
        self._plotter.enable_point_picking(
            callback=self._on_measure_pick,
            show_message=False,
            show_point=True,
            color="yellow",
            point_size=12,
            tolerance=0.025,
            use_picker=True,
        )

    def cancel_measurement(self) -> None:
        """Cancel active measurement mode (keeps existing visual markers)."""
        if self._measure_mode != MeasureMode.OFF:
            try:
                self._plotter.disable_picking()
            except Exception:
                pass
        self._measure_mode = MeasureMode.OFF
        self._measure_point_a = None

    def clear_measurements(self) -> None:
        """Remove all measurement visual markers from the scene."""
        self._measure_actors.clear()
        # Re-render to clean up all stray actors (measurement lines,
        # labels, and VTK picker point representations)
        self._render(preserve_camera=True)

    def _on_measure_pick(self, point: np.ndarray, *_args) -> None:
        """Handle a point pick during measurement."""
        if point is None or len(point) < 3:
            return
        pt = np.asarray(point[:3], dtype=float)

        if self._measure_point_a is None:
            # First point
            self._measure_point_a = pt
            return

        # Second point — compute and display
        a = self._measure_point_a
        b = pt

        if self._measure_mode == MeasureMode.DISTANCE:
            value = float(np.linalg.norm(b - a)) * self.unit_factor
            label = f"{value:.2f} {self.unit_suffix}"
        elif self._measure_mode == MeasureMode.HEIGHT:
            value = abs(float(b[2] - a[2])) * self.unit_factor
            label = f"dZ = {value:.2f} {self.unit_suffix}"
        else:
            label = ""

        # Draw measurement line
        line = pv.Line(a, b)
        actor_line = self._plotter.add_mesh(line, color="yellow", line_width=3)
        self._measure_actors.append(actor_line)

        # Draw label at midpoint
        mid = (a + b) / 2.0
        actor_label = self._plotter.add_point_labels(
            pv.PolyData(mid.reshape(1, 3)),
            [label],
            font_size=16,
            text_color="yellow",
            point_color="yellow",
            point_size=0,
            shape=None,
            render_points_as_spheres=False,
            always_visible=True,
        )
        self._measure_actors.append(actor_label)
        self._plotter.render()

        # Reset for next measurement (keep mode active)
        self._measure_point_a = None

    def _on_vtk_key_press(self, obj, event) -> None:
        """Handle VTK key press events (Escape cancels measurement)."""
        key = self._plotter.iren.interactor.GetKeySym()
        if key == "Escape" and self._measure_mode != MeasureMode.OFF:
            self.cancel_measurement()

    # --- Point cloud comparison ---

    def compare_with_cloud(self, filepath: str) -> None:
        """Load a second point cloud and color by nearest-neighbor distance.

        Args:
            filepath: Path to the comparison point cloud file.
        """
        if self._points_full is None:
            return

        from scipy.spatial import cKDTree
        from tools import load_file

        other_pc, _ = load_file(
            filepath,
            headers_of_interest=["x", "y", "z", "red", "green", "blue"],
        )
        other_xyz = other_pc[:, :3].astype(np.float32)

        # Build KD-tree of the currently loaded cloud
        tree = cKDTree(self._points_full)
        distances, _ = tree.query(other_xyz, k=1)

        # Also compute distances from current cloud to the other cloud
        other_tree = cKDTree(other_xyz)
        dist_self, _ = other_tree.query(self._points_full, k=1)

        self._comparison_distances = dist_self
        self._color_mode = ColorMode.COMPARISON
        # Update combo without triggering re-render
        for i in range(self._color_combo.count()):
            if self._color_combo.itemData(i) == ColorMode.COMPARISON:
                self._color_combo.blockSignals(True)
                self._color_combo.setCurrentIndex(i)
                self._color_combo.blockSignals(False)
                break
        else:
            self._color_combo.blockSignals(True)
            self._color_combo.addItem(ColorMode.COMPARISON.value, ColorMode.COMPARISON)
            self._color_combo.setCurrentIndex(self._color_combo.count() - 1)
            self._color_combo.blockSignals(False)

        self._render(preserve_camera=True)

    # --- Trim Tool ---

    def enable_trim_selection(self) -> None:
        """Activate rectangle-through-picking for point cloud trimming."""
        if self._points_full is None:
            return
        # Cancel any active focus/measure mode
        if self._focus_mode:
            self._focus_btn.setChecked(False)
        if self._measure_mode != MeasureMode.OFF:
            self.cancel_measurement()

        self._trim_active = True
        self._trim_selection = None
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*orig_extract_id.*", category=DeprecationWarning
            )
            self._plotter.enable_rectangle_through_picking(
                callback=self._on_trim_select, start=True, show_message=False,
            )

    def _on_trim_select(self, selected) -> None:
        """Handle rectangle-through picking for the trim tool."""
        if self._points_full is None or selected is None:
            return

        # Extract points from selection
        if hasattr(selected, "points"):
            pts = np.asarray(selected.points)
        else:
            pts = np.asarray(selected)

        if pts.ndim == 1:
            if pts.shape[0] >= 3:
                pts = pts.reshape(-1, 3)
            else:
                return
        if pts.shape[0] == 0:
            return

        # Map picked mesh points back to full-resolution indices via cKDTree
        from scipy.spatial import cKDTree
        tree = cKDTree(self._points_full)
        _, indices = tree.query(pts[:, :3], k=1)
        self._trim_selection = np.unique(indices)

        # Highlight selected region with white overlay
        self._render(preserve_camera=True)
        if self._trim_selection is not None and len(self._trim_selection) > 0:
            highlight_pts = self._points_full[self._trim_selection]
            highlight_cloud = pv.PolyData(highlight_pts)
            self._plotter.add_mesh(
                highlight_cloud, color="white", point_size=4,
                render_points_as_spheres=False, opacity=0.6, name="trim_highlight",
            )
            self._plotter.render()

        self.trim_region_selected.emit()

    def apply_trim(self, keep: bool) -> None:
        """Apply the trim operation.

        Args:
            keep: If True, keep only selected points. If False, remove selected points.
        """
        if self._trim_selection is None or self._points_full is None:
            return

        self.push_undo("Trim region")

        if keep:
            mask = np.zeros(self._points_full.shape[0], dtype=bool)
            mask[self._trim_selection] = True
        else:
            mask = np.ones(self._points_full.shape[0], dtype=bool)
            mask[self._trim_selection] = False

        self._points_full = self._points_full[mask]
        if self._colors_full is not None:
            self._colors_full = self._colors_full[mask]
        if self._labels is not None:
            self._labels = self._labels[mask]
        if self._tree_ids is not None:
            self._tree_ids = self._tree_ids[mask]
        self._crop_mask = None

        self._trim_selection = None
        self._trim_active = False
        self._plotter.disable_picking()
        self._plotter.enable_trackball_style()

        self.crop_state_changed.emit(True)
        self._build_lod()
        self._render()

    def cancel_trim(self) -> None:
        """Cancel trim selection and restore normal interaction."""
        self._trim_selection = None
        self._trim_active = False
        self._plotter.disable_picking()
        self._plotter.enable_trackball_style()
        self._render(preserve_camera=True)
