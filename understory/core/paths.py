"""Standardized path construction for FSCT pipeline outputs.

Replaces the duplicated output_dir/working_dir construction logic found in
preprocessing.py, inference.py, post_segmentation_script.py, measure.py,
and report_writer.py.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path


class ProjectPaths:
    """Manages the standardized Understory project folder structure.

    Structure::

        Projects/
          <project_name>/
            project.yaml              # master project config
            runs/
              run_<YYYY-MM-DD_HH-MM-SS>/
                run_config.yaml       # snapshot of settings for this run
                output/               # pipeline .las, .csv output
                  working_directory/  # intermediate files
                reports/              # HTML/PDF report, figures

    Supports multiple pipeline runs per project (different settings, different
    scans) with each run's results preserved separately for comparison.
    """

    def __init__(self, project_dir: str | Path):
        self._root = Path(project_dir).resolve()

    @property
    def root(self) -> Path:
        return self._root

    @property
    def config_file(self) -> Path:
        return self._root / "project.yaml"

    @property
    def runs_dir(self) -> Path:
        return self._root / "runs"

    def ensure_dirs(self) -> None:
        """Create the project root and runs directory."""
        self._root.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(exist_ok=True)

    def create_run(self) -> Path:
        """Create a new timestamped run folder and return its path."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = self.runs_dir / f"run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "output").mkdir(exist_ok=True)
        (run_dir / "reports").mkdir(exist_ok=True)
        return run_dir

    def list_runs(self) -> list[Path]:
        """Return list of run directories, newest first."""
        if not self.runs_dir.exists():
            return []
        runs = sorted(
            [r for r in self.runs_dir.iterdir() if r.is_dir() and r.name.startswith("run_")],
            reverse=True,
        )
        return runs

    def latest_run(self) -> Path | None:
        """Return the most recent run directory, or None."""
        runs = self.list_runs()
        return runs[0] if runs else None

    @staticmethod
    def run_output_dir(run_dir: Path) -> Path:
        """Get the output subdirectory for a run."""
        return run_dir / "output"

    @staticmethod
    def run_reports_dir(run_dir: Path) -> Path:
        """Get the reports subdirectory for a run."""
        return run_dir / "reports"


class FSCTPaths:
    """Manages all path construction for a single point cloud processing run."""

    def __init__(self, point_cloud_filename: str, output_directory: str | None = None):
        """
        Args:
            point_cloud_filename: Full path to the input point cloud file.
            output_directory: Override output location. If None, output goes
                next to the input file in a ``<name>_FSCT_output/`` folder.
        """
        self._input_path = Path(point_cloud_filename).resolve()
        self._input_dir = self._input_path.parent
        self._stem = self._input_path.stem  # filename without extension
        self._ext = self._input_path.suffix

        if output_directory:
            self._output_dir = Path(output_directory).resolve()
        else:
            self._output_dir = self._input_dir / f"{self._stem}_FSCT_output"

        self._working_dir = self._output_dir / "working_directory"

    # --- Properties ---

    @property
    def input_file(self) -> Path:
        return self._input_path

    @property
    def input_dir(self) -> Path:
        return self._input_dir

    @property
    def input_filename(self) -> str:
        return self._input_path.name

    @property
    def input_stem(self) -> str:
        return self._stem

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    @property
    def working_dir(self) -> Path:
        return self._working_dir

    # --- Commonly used output paths ---

    @property
    def plot_summary_csv(self) -> Path:
        return self._output_dir / "plot_summary.csv"

    @property
    def segmented_las(self) -> Path:
        return self._output_dir / "segmented.las"

    @property
    def dtm_las(self) -> Path:
        return self._output_dir / "DTM.las"

    @property
    def terrain_points_las(self) -> Path:
        return self._output_dir / "terrain_points.las"

    @property
    def vegetation_points_las(self) -> Path:
        return self._output_dir / "vegetation_points.las"

    @property
    def cwd_points_las(self) -> Path:
        return self._output_dir / "cwd_points.las"

    @property
    def stem_points_las(self) -> Path:
        return self._output_dir / "stem_points.las"

    @property
    def tree_data_csv(self) -> Path:
        return self._output_dir / "tree_data.csv"

    @property
    def taper_data_csv(self) -> Path:
        return self._output_dir / "taper_data.csv"

    @property
    def report_html(self) -> Path:
        return self._output_dir / "Plot_Report.html"

    @property
    def stem_map_png(self) -> Path:
        return self._output_dir / "Stem_Map.png"

    def output_file(self, name: str) -> Path:
        """Get path for an arbitrary output file."""
        return self._output_dir / name

    # --- Directory management ---

    def ensure_output_dirs(self, clean_working: bool = True) -> None:
        """Create output and working directories.

        Args:
            clean_working: If True and working_dir exists, delete and recreate it.
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)

        if self._working_dir.exists() and clean_working:
            shutil.rmtree(self._working_dir, ignore_errors=True)
        self._working_dir.mkdir(parents=True, exist_ok=True)

    def clean_working_dir(self) -> None:
        """Remove the working directory."""
        if self._working_dir.exists():
            shutil.rmtree(self._working_dir, ignore_errors=True)

    # --- Legacy compatibility ---

    @property
    def output_dir_str(self) -> str:
        """Output dir as string with trailing slash (legacy format)."""
        return str(self._output_dir) + "/"

    @property
    def working_dir_str(self) -> str:
        """Working dir as string with trailing slash (legacy format)."""
        return str(self._working_dir) + "/"

    @classmethod
    def get_project_root(cls) -> Path:
        """Get the FSCT project root directory."""
        return Path(__file__).resolve().parent.parent.parent

    @classmethod
    def get_model_dir(cls) -> Path:
        """Get the model weights directory."""
        return cls.get_project_root() / "model"

    @classmethod
    def get_scripts_dir(cls) -> Path:
        """Get the scripts directory."""
        return cls.get_project_root() / "scripts"

    @classmethod
    def get_data_dir(cls) -> Path:
        """Get the data directory."""
        return cls.get_project_root() / "data"
