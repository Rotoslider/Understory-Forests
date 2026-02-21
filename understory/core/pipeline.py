"""Modern pipeline wrapper around the FSCT processing pipeline.

Provides a clean interface for running the pipeline with ProjectConfig,
optional progress callbacks for GUI integration, and CLI compatibility.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Callable, Optional

from understory.config.settings import ProjectConfig


class PipelineStageError(Exception):
    """Wraps an error from a specific pipeline stage with a user-friendly message."""

    def __init__(self, stage: str, user_message: str, original_error: Exception):
        self.stage = stage
        self.user_message = user_message
        self.original_error = original_error
        super().__init__(user_message)


class PipelineCancelled(Exception):
    """Raised when the user cancels a running pipeline."""


# Known error patterns → user-friendly translations
_ERROR_TRANSLATIONS = [
    (
        ("NearestNeighbors", "0 sample(s)"),
        "Point cloud too sparse or plot radius too small. "
        "Try increasing the plot radius or loading a denser cloud.",
    ),
    (
        ("CUDA out of memory",),
        "GPU ran out of memory. Reduce the batch size in Process settings "
        "or enable 'CPU only' mode.",
    ),
    (
        ("Not compiled with CUDA support",),
        "PyG extensions (torch-scatter/cluster) were installed without CUDA. "
        "Fix by running:\n"
        "  pip uninstall -y torch-scatter torch-sparse torch-cluster torch-spline-conv\n"
        "  sudo apt install nvidia-cuda-toolkit\n"
        "  pip install torch-scatter torch-sparse torch-cluster torch-spline-conv --no-build-isolation\n"
        "Or switch to CPU mode in Process settings.",
    ),
    (
        ("CUDA error",),
        "A GPU error occurred. Try reducing batch size or switching to CPU mode.",
    ),
    (
        ("No such file or directory",),
        "A required file was not found. Check that the input file exists "
        "and that previous pipeline stages completed successfully.",
    ),
]


def _translate_error(stage: str, error: Exception) -> str:
    """Translate an exception into a user-friendly message."""
    error_str = str(error)
    for patterns, message in _ERROR_TRANSLATIONS:
        if all(p in error_str for p in patterns):
            return f"{stage} failed: {message}"
    return f"{stage} failed: {error_str}"


# Add scripts directory to path so existing modules can be imported
_scripts_dir = str(Path(__file__).resolve().parent.parent.parent / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)


def run_pipeline(
    config: ProjectConfig,
    progress_callback: Optional[Callable[[str, float], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> dict:
    """Run the FSCT processing pipeline.

    Args:
        config: Project configuration.
        progress_callback: Optional callback ``(stage_name, fraction)`` for
            progress reporting. ``fraction`` is 0.0–1.0 within each stage.
        cancel_event: Optional :class:`threading.Event`; when set, the
            pipeline will raise :class:`PipelineCancelled` before the next
            stage begins.

    Returns:
        dict with keys like ``"tree_data_csv"``, ``"plot_summary_csv"``
        pointing to output file paths.
    """
    # If input is a PCD file, convert to LAS first so all downstream code
    # (which assumes LAS extension for intermediate files) works correctly.
    input_path = Path(config.point_cloud_filename)
    if input_path.suffix.lower() == ".pcd":
        from tools import load_file, save_file

        las_copy = input_path.with_suffix(".las")
        if not las_copy.exists():
            pc, headers = load_file(
                str(input_path),
                headers_of_interest=["x", "y", "z", "red", "green", "blue"],
            )
            save_file(str(las_copy), pc, headers_of_interest=headers)
        config.point_cloud_filename = str(las_copy)

    # Ensure tree-aware plot cropping is active when a plot radius is set.
    # The legacy code requires plot_radius_buffer > 0 to enable cropping;
    # if the user left buffer at 0 we apply a minimal 0.5 m default so the
    # pipeline doesn't process points far outside the plot circle.
    if config.processing.plot_radius > 0 and config.processing.plot_radius_buffer == 0:
        config.processing.plot_radius_buffer = 0.5

    # Convert to legacy params
    parameters = config.to_legacy_params()

    # Ensure scripts dir is on path
    if _scripts_dir not in sys.path:
        sys.path.insert(0, _scripts_dir)

    from preprocessing import Preprocessing
    from inference import SemanticSegmentation
    from post_segmentation_script import PostProcessing
    from measure import MeasureTree
    from report_writer import ReportWriter
    from understory.core.report import generate_report
    from understory.core.paths import FSCTPaths

    if parameters["num_cpu_cores"] == 0:
        parameters["num_cpu_cores"] = os.cpu_count()

    def _report(stage: str, frac: float = 0.0) -> None:
        if progress_callback:
            progress_callback(stage, frac)

    # Build list of enabled stages with their approximate time-weight
    stages = []
    if config.preprocess:
        stages.append(("Preprocessing", 15))
    if config.segmentation:
        stages.append(("Semantic Segmentation", 45))
    if config.postprocessing:
        stages.append(("Post-processing", 20))
    if config.measure_plot:
        stages.append(("Measurement", 15))
    if config.make_report:
        stages.append(("Report Generation", 5))
    total_weight = sum(w for _, w in stages) or 1

    # Compute cumulative start fractions for each stage
    cumulative = 0.0
    stage_ranges = {}
    for name, weight in stages:
        start = cumulative / total_weight
        end = (cumulative + weight) / total_weight
        stage_ranges[name] = (start, end)
        cumulative += weight

    def _stage_report(name: str, frac: float) -> None:
        """Report overall progress based on stage and within-stage fraction."""
        if name in stage_ranges:
            start, end = stage_ranges[name]
            overall = start + (end - start) * frac
        else:
            overall = frac
        _report(name, overall)

    _report("Starting", 0.0)

    def _run_stage(name: str, func: Callable[[], None]) -> None:
        """Run a pipeline stage with error translation."""
        if cancel_event and cancel_event.is_set():
            raise PipelineCancelled("Pipeline cancelled by user")
        try:
            _stage_report(name, 0.0)
            func()
            _stage_report(name, 1.0)
        except PipelineCancelled:
            raise
        except PipelineStageError:
            raise
        except Exception as e:
            user_msg = _translate_error(name, e)
            raise PipelineStageError(name, user_msg, e) from e

    if config.preprocess:
        def _preprocess():
            prep = Preprocessing(parameters)
            prep.preprocess_point_cloud()
        _run_stage("Preprocessing", _preprocess)

    if config.segmentation:
        def _segmentation():
            seg = SemanticSegmentation(parameters)
            def _seg_progress(batch_frac):
                _stage_report("Semantic Segmentation", batch_frac)
            seg.inference(progress_callback=_seg_progress)
            del seg
        _run_stage("Semantic Segmentation", _segmentation)
        # Release GPU memory — inference is the only GPU-using stage
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if config.postprocessing:
        def _postprocess():
            post = PostProcessing(parameters)
            post.process_point_cloud()
        _run_stage("Post-processing", _postprocess)

    if config.measure_plot:
        def _measure():
            meas = MeasureTree(parameters)
            meas.run_measurement_extraction()
        _run_stage("Measurement", _measure)

    # Build output paths (needed for report, tree registry, and return value)
    paths = FSCTPaths(config.point_cloud_filename, output_directory=config.output.output_directory)

    # Apply tree registry for consistent IDs across runs within a project
    if config.measure_plot and paths.tree_data_csv.exists():
        try:
            import pandas as pd
            from understory.core.tree_registry import TreeRegistry

            # Locate registry in the project folder (two levels above output/)
            # Project structure: <project>/runs/run_<ts>/output/
            project_dir = paths.output_dir.parent.parent.parent
            registry_path = project_dir / "tree_registry.json"

            if registry_path.parent.exists():
                registry = TreeRegistry(registry_path)
                tree_df = pd.read_csv(paths.tree_data_csv)
                tree_df = registry.match_trees(tree_df)
                tree_df.to_csv(paths.tree_data_csv, index=False)
        except Exception:
            pass  # Non-critical — fall back to pipeline-assigned IDs

    if config.make_report:
        def _gen_report():
            generate_report(
                output_dir=str(paths.output_dir),
                point_cloud_filename=config.point_cloud_filename,
                project_name=config.project_name,
                operator=config.operator,
                notes=config.notes,
                photos=config.photos if config.photos else None,
            )
            # Move report files to the reports/ subdirectory if it exists
            # (present when running under a project with timestamped runs)
            reports_dir = paths.output_dir.parent / "reports"
            if reports_dir.exists():
                import shutil
                report_files = [paths.output_dir / "Plot_Report.html",
                                paths.output_dir / "understory-logo.png"]
                report_files.extend(paths.output_dir.glob("*.png"))
                for src in report_files:
                    if src.exists():
                        shutil.move(str(src), str(reports_dir / src.name))
        _run_stage("Report Generation", _gen_report)

    if config.clean_up_files:
        rpt = ReportWriter(parameters)
        rpt.clean_up_files()
        del rpt

    _report("Complete", 1.0)

    return {
        "output_dir": str(paths.output_dir),
        "tree_data_csv": str(paths.tree_data_csv),
        "taper_data_csv": str(paths.taper_data_csv),
        "plot_summary_csv": str(paths.plot_summary_csv),
        "report_html": str(paths.report_html),
    }
