"""Entry point for Understory.

Usage:
    python -m understory              # Launch GUI
    python -m understory --cli FILE   # Headless CLI mode
    python -m understory --cli FILE --config project.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="understory",
        description="Understory — Forest Structural Complexity Tool",
    )
    parser.add_argument(
        "--cli",
        metavar="FILE",
        nargs="?",
        const="",
        default=None,
        help="Run in headless CLI mode. Optionally provide an input point cloud file.",
    )
    parser.add_argument(
        "--config",
        metavar="YAML",
        default=None,
        help="Path to a YAML project configuration file.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_get_version()}",
    )

    args = parser.parse_args()

    if args.cli is not None:
        _run_cli(args)
    else:
        _run_gui()


def _get_version() -> str:
    try:
        from understory import __version__
        return __version__
    except ImportError:
        return "unknown"


def _run_cli(args: argparse.Namespace) -> None:
    """Run the pipeline in headless CLI mode."""
    from understory.config.settings import ProjectConfig
    from understory.core.pipeline import run_pipeline

    if args.config:
        config = ProjectConfig.load(args.config)
        # CLI file argument overrides config if provided
        if args.cli:
            config.point_cloud_filename = args.cli
    elif args.cli:
        config = ProjectConfig(point_cloud_filename=args.cli)
    else:
        # Interactive file selection (legacy tkinter mode)
        try:
            import tkinter as tk
            import tkinter.filedialog as fd
            root = tk.Tk()
            root.withdraw()
            files = fd.askopenfilenames(
                title="Choose point cloud files",
                filetypes=[("LAS", "*.las"), ("LAZ", "*.laz"), ("PCD", "*.pcd"), ("CSV", "*.csv")],
            )
            root.destroy()
            if not files:
                print("No files selected. Exiting.")
                sys.exit(0)
            for f in files:
                config = ProjectConfig(point_cloud_filename=f)
                result = run_pipeline(config, progress_callback=_cli_progress)
                print(f"\nResults saved to: {result['output_dir']}")
            return
        except ImportError:
            print("Error: No input file specified. Use: python -m understory --cli FILE")
            sys.exit(1)

    result = run_pipeline(config, progress_callback=_cli_progress)
    print(f"\nResults saved to: {result['output_dir']}")


def _cli_progress(stage: str, fraction: float) -> None:
    """Print progress to stdout."""
    bar_len = 30
    filled = int(bar_len * fraction)
    bar = "=" * filled + "-" * (bar_len - filled)
    print(f"\r[{bar}] {stage}: {fraction*100:.0f}%", end="", flush=True)
    if fraction >= 1.0:
        print()


def _run_gui() -> None:
    """Launch the PySide6 GUI application."""
    import os

    # Suppress PyVista deprecation warning about orig_extract_id renaming
    import warnings
    warnings.filterwarnings("ignore", message=".*orig_extract_id.*")

    # Force X11 backend — VTK's render window doesn't work under Wayland
    if "QT_QPA_PLATFORM" not in os.environ:
        os.environ["QT_QPA_PLATFORM"] = "xcb"

    try:
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import QTimer
        from understory.gui.main_window import MainWindow
    except ImportError as e:
        print(f"GUI dependencies not available: {e}")
        print("Install with: pip install PySide6 pyvista pyvistaqt")
        print("Or run in CLI mode: python -m understory --cli FILE")
        sys.exit(1)

    app = QApplication(sys.argv)
    app.setApplicationName("Understory")
    app.setOrganizationName("Understory")

    # Set app icon
    from PySide6.QtGui import QIcon
    icon_path = Path(__file__).parent / "resources" / "icons" / "understory-icon.png"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    window = MainWindow()
    # Show first, then maximize on next event loop tick to avoid X11 BadWindow
    # race condition when VTK's render window is configured before the Qt
    # window is fully mapped.
    window.show()
    QTimer.singleShot(0, window.showMaximized)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
