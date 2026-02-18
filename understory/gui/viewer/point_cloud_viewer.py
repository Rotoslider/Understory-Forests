"""3D point cloud viewer with Level-of-Detail (LOD) for handling 500M+ points.

Uses PyVista + pyvistaqt for rendering inside PySide6.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np

try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
    from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLabel, QPushButton
    from PySide6.QtCore import Signal
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False


class ColorMode(Enum):
    RGB = "RGB"
    HEIGHT = "Height Gradient"
    CLASSIFICATION = "Classification"
    TREE_ID = "Tree ID"


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


class PointCloudViewer(QWidget):
    """3D point cloud viewer with LOD and multiple color modes."""

    point_picked = Signal(int, float, float, float)  # index, x, y, z
    plot_centre_dragged = Signal(float, float)  # x, y from interactive widget
    crop_state_changed = Signal(bool)  # True when cropped, False when reset

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
                return self._labels[idx].astype(np.float32)
            return None

        elif self._color_mode == ColorMode.TREE_ID:
            if self._tree_ids is not None:
                return self._tree_ids[idx].astype(np.float32)
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
        self._plot_circle_widget_active = False
        self._dragging_circle = False
        self._circle_actor = None
        self._plotter.clear()
        self._point_count_label.setText("No data loaded")

    # --- Crop outliers ---

    def _crop_to_bounds(self) -> None:
        """Remove outlier points beyond the 99.5th percentile per axis."""
        if self._points_full is None:
            return

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
