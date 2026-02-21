"""Animation/Flythrough Editor — Feature 20.

Captures camera keyframes from a PyVista plotter and generates smooth
flythrough animations using cubic spline interpolation.  Supports preview
in the live viewer and export to PNG image sequence, GIF, or MP4.

Constructor takes a reference to the PyVista ``QtInteractor`` plotter.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QLabel,
    QPushButton,
    QListWidget,
    QSpinBox,
    QGroupBox,
    QFileDialog,
    QMessageBox,
    QProgressBar,
    QComboBox,
    QApplication,
)

try:
    from pyvistaqt import QtInteractor
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False


# ---------------------------------------------------------------------------
# Keyframe data structure
# ---------------------------------------------------------------------------

class Keyframe:
    """A single camera keyframe storing position, focal point, and view-up."""

    __slots__ = ("position", "focal_point", "view_up", "label")

    def __init__(
        self,
        position: tuple[float, float, float],
        focal_point: tuple[float, float, float],
        view_up: tuple[float, float, float],
        label: str = "",
    ):
        self.position = tuple(float(v) for v in position)
        self.focal_point = tuple(float(v) for v in focal_point)
        self.view_up = tuple(float(v) for v in view_up)
        self.label = label

    def __repr__(self) -> str:
        return (
            f"Keyframe(pos=({self.position[0]:.1f}, {self.position[1]:.1f}, "
            f"{self.position[2]:.1f}), label={self.label!r})"
        )


# ---------------------------------------------------------------------------
# Interpolation helpers
# ---------------------------------------------------------------------------

def _interpolate_camera_path(
    keyframes: list[Keyframe],
    total_frames: int,
) -> list[tuple[tuple, tuple, tuple]]:
    """Interpolate between *keyframes* using cubic splines.

    Returns a list of ``(position, focal_point, view_up)`` tuples — one per
    output frame — suitable for assigning directly to
    ``plotter.camera_position``.

    When fewer than 2 keyframes are provided the function returns an empty
    list.  With exactly 2 keyframes a simple linear interpolation is used
    (``CubicSpline`` requires at least 2 data points, but linear gives
    identical results for 2 points and avoids edge-case wiggle).
    """
    n_kf = len(keyframes)
    if n_kf < 2 or total_frames < 1:
        return []

    # Parameter t for each keyframe, evenly spaced on [0, 1].
    t_kf = np.linspace(0.0, 1.0, n_kf)
    t_out = np.linspace(0.0, 1.0, total_frames)

    # Build arrays (n_kf x 3) for each camera component.
    pos_arr = np.array([kf.position for kf in keyframes], dtype=np.float64)
    foc_arr = np.array([kf.focal_point for kf in keyframes], dtype=np.float64)
    vup_arr = np.array([kf.view_up for kf in keyframes], dtype=np.float64)

    if n_kf == 2:
        # Linear interpolation — avoids scipy dependency for the trivial case.
        pos_interp = np.outer(1 - t_out, pos_arr[0]) + np.outer(t_out, pos_arr[1])
        foc_interp = np.outer(1 - t_out, foc_arr[0]) + np.outer(t_out, foc_arr[1])
        vup_interp = np.outer(1 - t_out, vup_arr[0]) + np.outer(t_out, vup_arr[1])
    else:
        from scipy.interpolate import CubicSpline

        cs_pos = CubicSpline(t_kf, pos_arr, bc_type="clamped")
        cs_foc = CubicSpline(t_kf, foc_arr, bc_type="clamped")
        cs_vup = CubicSpline(t_kf, vup_arr, bc_type="clamped")

        pos_interp = cs_pos(t_out)
        foc_interp = cs_foc(t_out)
        vup_interp = cs_vup(t_out)

    # Normalise view-up vectors (spline interpolation can shrink/grow them).
    norms = np.linalg.norm(vup_interp, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vup_interp = vup_interp / norms

    path: list[tuple[tuple, tuple, tuple]] = []
    for i in range(total_frames):
        path.append((
            tuple(pos_interp[i]),
            tuple(foc_interp[i]),
            tuple(vup_interp[i]),
        ))
    return path


# ---------------------------------------------------------------------------
# Main dialog
# ---------------------------------------------------------------------------

class FlythroughEditor(QDialog):
    """Modal-free dialog for creating camera flythrough animations.

    Parameters
    ----------
    plotter : QtInteractor
        The active PyVista plotter whose camera will be read and driven.
    parent : QWidget, optional
        Parent widget (typically the main window).
    """

    # Emitted when the preview or export finishes (or is cancelled).
    animation_finished = Signal()

    def __init__(self, plotter: "QtInteractor", parent=None):
        super().__init__(parent)
        if not HAS_PYVISTA:
            raise ImportError("pyvistaqt is required for the flythrough editor")

        self._plotter = plotter
        self._keyframes: list[Keyframe] = []
        self._preview_timer: Optional[QTimer] = None
        self._preview_path: list[tuple] = []
        self._preview_index: int = 0
        self._is_previewing: bool = False
        self._is_exporting: bool = False

        self.setWindowTitle("Flythrough Editor")
        self.setMinimumSize(420, 600)
        # Non-modal so the user can interact with the viewer to move the camera
        self.setModal(False)
        self.setWindowFlags(
            self.windowFlags() | Qt.WindowStaysOnTopHint
        )
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        title = QLabel("Animation / Flythrough Editor")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #1a4a3a; padding: 4px 0;")
        layout.addWidget(title)

        # Ensure readable text in case the system theme has light backgrounds
        self.setStyleSheet("""
            QDialog { background: #f0f7f4; }
            QGroupBox { font-weight: bold; color: #1a4a3a; }
            QLabel { color: #1a2e26; }
            QPushButton { color: #1a2e26; }
        """)

        # ---- Keyframe list ----
        kf_group = QGroupBox("Keyframes")
        kf_layout = QVBoxLayout(kf_group)

        self._kf_list = QListWidget()
        self._kf_list.setAlternatingRowColors(True)
        self._kf_list.setDragDropMode(QListWidget.NoDragDrop)
        self._kf_list.currentRowChanged.connect(self._on_keyframe_selected)
        kf_layout.addWidget(self._kf_list)

        # Buttons row 1: capture / delete
        btn_row1 = QHBoxLayout()

        self._capture_btn = QPushButton("Capture Current View")
        self._capture_btn.setToolTip(
            "Save the current camera position, focal point, and view-up "
            "as a new keyframe at the end of the list."
        )
        self._capture_btn.clicked.connect(self._capture_keyframe)
        btn_row1.addWidget(self._capture_btn)

        self._delete_btn = QPushButton("Delete")
        self._delete_btn.setToolTip("Remove the selected keyframe")
        self._delete_btn.clicked.connect(self._delete_keyframe)
        self._delete_btn.setEnabled(False)
        btn_row1.addWidget(self._delete_btn)

        kf_layout.addLayout(btn_row1)

        # Buttons row 2: reorder
        btn_row2 = QHBoxLayout()

        self._move_up_btn = QPushButton("Move Up")
        self._move_up_btn.clicked.connect(self._move_up)
        self._move_up_btn.setEnabled(False)
        btn_row2.addWidget(self._move_up_btn)

        self._move_down_btn = QPushButton("Move Down")
        self._move_down_btn.clicked.connect(self._move_down)
        self._move_down_btn.setEnabled(False)
        btn_row2.addWidget(self._move_down_btn)

        self._goto_btn = QPushButton("Go To")
        self._goto_btn.setToolTip("Jump the camera to the selected keyframe")
        self._goto_btn.clicked.connect(self._goto_keyframe)
        self._goto_btn.setEnabled(False)
        btn_row2.addWidget(self._goto_btn)

        kf_layout.addLayout(btn_row2)

        layout.addWidget(kf_group)

        # ---- Animation settings ----
        settings_group = QGroupBox("Animation Settings")
        settings_form = QFormLayout(settings_group)

        self._total_frames_spin = QSpinBox()
        self._total_frames_spin.setRange(2, 100000)
        self._total_frames_spin.setValue(120)
        self._total_frames_spin.setToolTip(
            "Total number of frames in the animation. "
            "Duration = frames / FPS."
        )
        settings_form.addRow("Total frames:", self._total_frames_spin)

        self._fps_spin = QSpinBox()
        self._fps_spin.setRange(1, 120)
        self._fps_spin.setValue(30)
        self._fps_spin.setToolTip("Frames per second for preview and video export.")
        settings_form.addRow("FPS:", self._fps_spin)

        self._duration_label = QLabel(self._duration_text())
        settings_form.addRow("Duration:", self._duration_label)

        self._total_frames_spin.valueChanged.connect(self._update_duration)
        self._fps_spin.valueChanged.connect(self._update_duration)

        self._resolution_combo = QComboBox()
        self._resolution_combo.addItems([
            "Window size",
            "1280 x 720 (720p)",
            "1920 x 1080 (1080p)",
            "2560 x 1440 (1440p)",
            "3840 x 2160 (4K)",
        ])
        self._resolution_combo.setToolTip(
            "Output resolution for exported frames. "
            "'Window size' uses the current plotter dimensions."
        )
        settings_form.addRow("Resolution:", self._resolution_combo)

        layout.addWidget(settings_group)

        # ---- Preview / Export ----
        action_group = QGroupBox("Preview && Export")
        action_layout = QVBoxLayout(action_group)

        preview_row = QHBoxLayout()
        self._preview_btn = QPushButton("Preview")
        self._preview_btn.setToolTip(
            "Play the flythrough in the viewer at the configured FPS."
        )
        self._preview_btn.clicked.connect(self._toggle_preview)
        preview_row.addWidget(self._preview_btn)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setToolTip("Stop the running preview or export.")
        self._stop_btn.clicked.connect(self._stop_animation)
        self._stop_btn.setEnabled(False)
        preview_row.addWidget(self._stop_btn)

        action_layout.addLayout(preview_row)

        # Export format selector
        export_form = QFormLayout()
        self._export_format_combo = QComboBox()
        self._export_format_combo.addItems([
            "PNG image sequence",
            "GIF (requires imageio)",
            "MP4 (requires imageio-ffmpeg)",
        ])
        export_form.addRow("Format:", self._export_format_combo)
        action_layout.addLayout(export_form)

        self._export_btn = QPushButton("Export...")
        self._export_btn.setToolTip("Render every frame and save to disk.")
        self._export_btn.clicked.connect(self._export)
        action_layout.addWidget(self._export_btn)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        action_layout.addWidget(self._progress)

        self._status_label = QLabel("")
        action_layout.addWidget(self._status_label)

        layout.addWidget(action_group)

    # ------------------------------------------------------------------
    # Duration helper
    # ------------------------------------------------------------------

    def _duration_text(self) -> str:
        frames = self._total_frames_spin.value() if hasattr(self, "_total_frames_spin") else 120
        fps = self._fps_spin.value() if hasattr(self, "_fps_spin") else 30
        secs = frames / max(fps, 1)
        return f"{secs:.2f} s ({frames} frames at {fps} FPS)"

    @Slot()
    def _update_duration(self) -> None:
        self._duration_label.setText(self._duration_text())

    # ------------------------------------------------------------------
    # Keyframe management
    # ------------------------------------------------------------------

    def _refresh_list(self) -> None:
        """Rebuild the QListWidget from the internal keyframes list."""
        self._kf_list.clear()
        for i, kf in enumerate(self._keyframes):
            label = kf.label or f"Keyframe {i + 1}"
            text = (
                f"{label}  —  pos=({kf.position[0]:.1f}, "
                f"{kf.position[1]:.1f}, {kf.position[2]:.1f})"
            )
            self._kf_list.addItem(text)

    def _update_button_states(self) -> None:
        row = self._kf_list.currentRow()
        has_sel = row >= 0
        self._delete_btn.setEnabled(has_sel)
        self._goto_btn.setEnabled(has_sel)
        self._move_up_btn.setEnabled(has_sel and row > 0)
        self._move_down_btn.setEnabled(has_sel and row < len(self._keyframes) - 1)

    @Slot(int)
    def _on_keyframe_selected(self, row: int) -> None:
        self._update_button_states()

    @Slot()
    def _capture_keyframe(self) -> None:
        """Capture the current camera state from the plotter."""
        try:
            cam = self._plotter.camera_position
        except Exception as exc:
            QMessageBox.warning(
                self, "Capture Failed",
                f"Could not read camera position:\n{exc}",
            )
            return

        # cam is ((pos), (focal), (viewup))
        position, focal_point, view_up = cam
        idx = len(self._keyframes) + 1
        kf = Keyframe(
            position=position,
            focal_point=focal_point,
            view_up=view_up,
            label=f"Keyframe {idx}",
        )
        self._keyframes.append(kf)
        self._refresh_list()
        self._kf_list.setCurrentRow(len(self._keyframes) - 1)
        self._status_label.setText(f"Captured keyframe {idx}")

    @Slot()
    def _delete_keyframe(self) -> None:
        row = self._kf_list.currentRow()
        if row < 0 or row >= len(self._keyframes):
            return
        self._keyframes.pop(row)
        self._refresh_list()
        # Select the nearest remaining item.
        if self._keyframes:
            self._kf_list.setCurrentRow(min(row, len(self._keyframes) - 1))
        self._update_button_states()

    @Slot()
    def _move_up(self) -> None:
        row = self._kf_list.currentRow()
        if row <= 0:
            return
        self._keyframes[row], self._keyframes[row - 1] = (
            self._keyframes[row - 1],
            self._keyframes[row],
        )
        self._refresh_list()
        self._kf_list.setCurrentRow(row - 1)

    @Slot()
    def _move_down(self) -> None:
        row = self._kf_list.currentRow()
        if row < 0 or row >= len(self._keyframes) - 1:
            return
        self._keyframes[row], self._keyframes[row + 1] = (
            self._keyframes[row + 1],
            self._keyframes[row],
        )
        self._refresh_list()
        self._kf_list.setCurrentRow(row + 1)

    @Slot()
    def _goto_keyframe(self) -> None:
        """Jump the plotter camera to the selected keyframe."""
        row = self._kf_list.currentRow()
        if row < 0 or row >= len(self._keyframes):
            return
        kf = self._keyframes[row]
        self._plotter.camera_position = (kf.position, kf.focal_point, kf.view_up)
        self._plotter.render()
        self._status_label.setText(f"Camera moved to keyframe {row + 1}")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_for_animation(self) -> bool:
        """Return True if we have enough keyframes to animate."""
        if len(self._keyframes) < 2:
            QMessageBox.warning(
                self,
                "Not Enough Keyframes",
                "At least 2 keyframes are required to create an animation.",
            )
            return False
        return True

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    @Slot()
    def _toggle_preview(self) -> None:
        if self._is_previewing:
            self._stop_animation()
            return

        if not self._validate_for_animation():
            return

        total_frames = self._total_frames_spin.value()
        fps = self._fps_spin.value()

        self._preview_path = _interpolate_camera_path(self._keyframes, total_frames)
        if not self._preview_path:
            return

        self._preview_index = 0
        self._is_previewing = True
        self._preview_btn.setText("Pause")
        self._stop_btn.setEnabled(True)
        self._capture_btn.setEnabled(False)
        self._export_btn.setEnabled(False)
        self._progress.setRange(0, total_frames)
        self._progress.setValue(0)
        self._progress.setVisible(True)

        interval_ms = max(1, int(1000.0 / fps))
        self._preview_timer = QTimer(self)
        self._preview_timer.setInterval(interval_ms)
        self._preview_timer.timeout.connect(self._preview_step)
        self._preview_timer.start()

    @Slot()
    def _preview_step(self) -> None:
        if self._preview_index >= len(self._preview_path):
            self._stop_animation()
            return

        cam = self._preview_path[self._preview_index]
        self._plotter.camera_position = cam
        self._plotter.render()

        self._progress.setValue(self._preview_index + 1)
        self._status_label.setText(
            f"Preview: frame {self._preview_index + 1}/{len(self._preview_path)}"
        )
        self._preview_index += 1

    @Slot()
    def _stop_animation(self) -> None:
        """Stop any running preview or export."""
        if self._preview_timer is not None:
            self._preview_timer.stop()
            self._preview_timer.deleteLater()
            self._preview_timer = None

        self._is_previewing = False
        self._is_exporting = False
        self._preview_btn.setText("Preview")
        self._stop_btn.setEnabled(False)
        self._capture_btn.setEnabled(True)
        self._export_btn.setEnabled(True)
        self._progress.setVisible(False)
        self._status_label.setText("Stopped")
        self.animation_finished.emit()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _parse_resolution(self) -> tuple[int, int] | None:
        """Parse the resolution combo into (width, height) or None for window."""
        text = self._resolution_combo.currentText()
        if text.startswith("Window"):
            return None
        # Format: "1920 x 1080 (1080p)"
        try:
            parts = text.split("(")[0].strip().split("x")
            return int(parts[0].strip()), int(parts[1].strip())
        except (ValueError, IndexError):
            return None

    @Slot()
    def _export(self) -> None:
        if not self._validate_for_animation():
            return

        fmt_index = self._export_format_combo.currentIndex()

        if fmt_index == 0:
            self._export_png_sequence()
        elif fmt_index == 1:
            self._export_gif()
        elif fmt_index == 2:
            self._export_mp4()

    # ---- PNG sequence ----

    def _export_png_sequence(self) -> None:
        out_dir = QFileDialog.getExistingDirectory(
            self, "Select Output Folder for PNG Frames"
        )
        if not out_dir:
            return

        total_frames = self._total_frames_spin.value()
        path = _interpolate_camera_path(self._keyframes, total_frames)
        if not path:
            return

        resolution = self._parse_resolution()
        self._begin_export(total_frames)

        for i, cam in enumerate(path):
            if not self._is_exporting:
                break
            self._plotter.camera_position = cam
            self._plotter.render()

            filename = os.path.join(out_dir, f"frame_{i:06d}.png")
            if resolution is not None:
                self._plotter.screenshot(
                    filename,
                    transparent_background=False,
                    window_size=resolution,
                )
            else:
                self._plotter.screenshot(filename, transparent_background=False)

            self._progress.setValue(i + 1)
            self._status_label.setText(f"Exporting frame {i + 1}/{total_frames}")
            QApplication.processEvents()

        self._finish_export(out_dir)

    # ---- GIF ----

    def _export_gif(self) -> None:
        try:
            import imageio.v3 as iio
        except ImportError:
            try:
                import imageio as iio
            except ImportError:
                QMessageBox.critical(
                    self,
                    "Missing Dependency",
                    "GIF export requires the 'imageio' package.\n\n"
                    "Install it with:  pip install imageio",
                )
                return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export GIF", "", "GIF Files (*.gif);;All Files (*)"
        )
        if not filepath:
            return
        if not filepath.lower().endswith(".gif"):
            filepath += ".gif"

        total_frames = self._total_frames_spin.value()
        fps = self._fps_spin.value()
        path = _interpolate_camera_path(self._keyframes, total_frames)
        if not path:
            return

        resolution = self._parse_resolution()
        self._begin_export(total_frames)

        frames: list[np.ndarray] = []
        for i, cam in enumerate(path):
            if not self._is_exporting:
                break
            self._plotter.camera_position = cam
            self._plotter.render()

            if resolution is not None:
                img = self._plotter.screenshot(
                    transparent_background=False,
                    window_size=resolution,
                    return_img=True,
                )
            else:
                img = self._plotter.screenshot(
                    transparent_background=False,
                    return_img=True,
                )
            frames.append(img)

            self._progress.setValue(i + 1)
            self._status_label.setText(f"Capturing frame {i + 1}/{total_frames}")
            QApplication.processEvents()

        if frames and self._is_exporting:
            self._status_label.setText("Writing GIF...")
            QApplication.processEvents()
            duration_ms = int(1000.0 / max(fps, 1))
            # imageio v3 API
            try:
                iio.imwrite(
                    filepath,
                    frames,
                    duration=duration_ms,
                    loop=0,
                )
            except TypeError:
                # Fallback for imageio v2 API
                import imageio
                with imageio.get_writer(filepath, mode="I", duration=duration_ms / 1000.0, loop=0) as writer:
                    for frame in frames:
                        writer.append_data(frame)

        self._finish_export(filepath)

    # ---- MP4 ----

    def _export_mp4(self) -> None:
        try:
            import imageio.v3 as iio
        except ImportError:
            try:
                import imageio as iio
            except ImportError:
                QMessageBox.critical(
                    self,
                    "Missing Dependency",
                    "MP4 export requires 'imageio' and 'imageio-ffmpeg'.\n\n"
                    "Install with:  pip install imageio imageio-ffmpeg",
                )
                return

        # Verify ffmpeg plugin is available.
        try:
            import imageio_ffmpeg
        except ImportError:
            QMessageBox.critical(
                self,
                "Missing Dependency",
                "MP4 export requires the 'imageio-ffmpeg' package.\n\n"
                "Install it with:  pip install imageio-ffmpeg",
            )
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export MP4", "", "MP4 Video (*.mp4);;All Files (*)"
        )
        if not filepath:
            return
        if not filepath.lower().endswith(".mp4"):
            filepath += ".mp4"

        total_frames = self._total_frames_spin.value()
        fps = self._fps_spin.value()
        path = _interpolate_camera_path(self._keyframes, total_frames)
        if not path:
            return

        resolution = self._parse_resolution()
        self._begin_export(total_frames)

        # Use imageio writer for ffmpeg-backed MP4
        try:
            import imageio
            writer = imageio.get_writer(
                filepath,
                fps=fps,
                codec="libx264",
                quality=8,
                pixelformat="yuv420p",
            )
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Export Error",
                f"Failed to create MP4 writer:\n{exc}",
            )
            self._stop_animation()
            return

        try:
            for i, cam in enumerate(path):
                if not self._is_exporting:
                    break
                self._plotter.camera_position = cam
                self._plotter.render()

                if resolution is not None:
                    img = self._plotter.screenshot(
                        transparent_background=False,
                        window_size=resolution,
                        return_img=True,
                    )
                else:
                    img = self._plotter.screenshot(
                        transparent_background=False,
                        return_img=True,
                    )
                writer.append_data(img)

                self._progress.setValue(i + 1)
                self._status_label.setText(f"Encoding frame {i + 1}/{total_frames}")
                QApplication.processEvents()
        finally:
            writer.close()

        self._finish_export(filepath)

    # ---- Export helpers ----

    def _begin_export(self, total_frames: int) -> None:
        self._is_exporting = True
        self._stop_btn.setEnabled(True)
        self._capture_btn.setEnabled(False)
        self._export_btn.setEnabled(False)
        self._preview_btn.setEnabled(False)
        self._progress.setRange(0, total_frames)
        self._progress.setValue(0)
        self._progress.setVisible(True)

    def _finish_export(self, output_path: str) -> None:
        was_cancelled = not self._is_exporting
        self._is_exporting = False
        self._stop_btn.setEnabled(False)
        self._capture_btn.setEnabled(True)
        self._export_btn.setEnabled(True)
        self._preview_btn.setEnabled(True)
        self._progress.setVisible(False)

        if was_cancelled:
            self._status_label.setText("Export cancelled")
        else:
            self._status_label.setText(f"Export complete: {output_path}")
            QMessageBox.information(
                self,
                "Export Complete",
                f"Animation exported to:\n{output_path}",
            )
        self.animation_finished.emit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def keyframes(self) -> list[Keyframe]:
        """Read-only access to the current keyframe list."""
        return list(self._keyframes)

    def add_keyframe(self, kf: Keyframe) -> None:
        """Programmatically add a keyframe."""
        self._keyframes.append(kf)
        self._refresh_list()

    def clear_keyframes(self) -> None:
        """Remove all keyframes."""
        self._keyframes.clear()
        self._refresh_list()
        self._update_button_states()
