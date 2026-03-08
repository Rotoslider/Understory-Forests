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


def _remap_output_tree_ids(paths, id_mapping: dict[int, int]) -> None:
    """Remap tree_id values in LAS output files and regenerate text labels.

    Called after TreeRegistry matching to keep point cloud tree IDs in sync
    with the updated tree_data.csv.

    Args:
        paths: FSCTPaths instance for the current run.
        id_mapping: Dict mapping {old_tree_id: new_tree_id}.
    """
    import numpy as np
    import pandas as pd
    from tools import load_file, save_file

    stem_veg_headers = ["x", "y", "z", "red", "green", "blue", "label", "height_above_dtm", "tree_id"]
    cyl_headers = ["x", "y", "z", "nx", "ny", "nz", "radius", "CCI", "branch_id",
                   "parent_branch_id", "tree_id", "tree_volume", "segment_angle_to_horiz", "height_above_dtm"]

    # LAS files that have tree_id as a column field
    las_files_with_tree_id = [
        ("stem_points_sorted.las", stem_veg_headers),
        ("veg_points_sorted.las", stem_veg_headers),
        ("tree_aware_cropped_point_cloud.las", stem_veg_headers),
        ("cleaned_cyls.las", cyl_headers),
        ("cleaned_cyl_vis.las", cyl_headers),
        ("sorted_full_cyl_array.las", cyl_headers),
        ("interpolated_full_cyl_array.las", cyl_headers),
    ]

    for filename, headers in las_files_with_tree_id:
        filepath = paths.output_dir / filename
        if not filepath.exists():
            continue
        try:
            data, loaded_headers = load_file(str(filepath), headers_of_interest=headers)
            if data.shape[0] == 0:
                continue
            tid_col = headers.index("tree_id")
            changed = False
            for old_id, new_id in id_mapping.items():
                mask = data[:, tid_col] == old_id
                if np.any(mask):
                    data[mask, tid_col] = new_id
                    changed = True
            if changed:
                save_file(str(filepath), data, headers_of_interest=headers)
        except Exception:
            continue

    # Also remap TreeId in taper_data.csv
    taper_path = paths.output_dir / "taper_data.csv"
    if taper_path.exists():
        try:
            taper_df = pd.read_csv(taper_path)
            if "TreeId" in taper_df.columns:
                taper_df["TreeId"] = taper_df["TreeId"].map(
                    lambda x: id_mapping.get(int(x), x))
                taper_df.to_csv(taper_path, index=False)
        except Exception:
            pass

    # Regenerate text_point_cloud.las with updated IDs
    _regenerate_text_labels(paths, id_mapping)


def _regenerate_text_labels(paths, id_mapping: dict[int, int]) -> None:
    """Regenerate the 3D text label point cloud with updated tree IDs.

    The text_point_cloud.las contains 3D points arranged as readable text
    (tree ID, DBH, CCI, height, volume). Since the text is baked into the
    geometry, we must regenerate it when IDs change.
    """
    import os
    import numpy as np
    import pandas as pd
    from tools import load_file, save_file, get_fsct_path

    tree_data_path = paths.tree_data_csv
    text_path = paths.output_dir / "text_point_cloud.las"
    if not tree_data_path.exists():
        return

    tree_data = pd.read_csv(tree_data_path)
    if tree_data.empty:
        return

    # Load character grids for text rendering
    characters = [
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "dot", "m", "space", "_", "-", "semiC",
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
        "_M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    ]
    numbers_dir = get_fsct_path("tools/numbers")
    character_viz = []
    for name in characters:
        csv_path = os.path.join(numbers_dir, name + ".csv")
        character_viz.append(np.genfromtxt(csv_path, delimiter=","))

    def _get_character(char):
        if char == ":":
            return character_viz[characters.index("semiC")]
        elif char == ".":
            return character_viz[characters.index("dot")]
        elif char == " ":
            return character_viz[characters.index("space")]
        elif char == "M":
            return character_viz[characters.index("_M")]
        else:
            return character_viz[characters.index(char)]

    def _make_text(character_size, xpos, ypos, zpos, offset, text):
        text_points = np.zeros((11, 0))
        for ch in text:
            text_points = np.hstack((text_points, np.array(_get_character(str(ch)))))
        indices = np.argwhere(np.rot90(text_points, axes=(1, 0)) == 1)
        if indices.shape[0] == 0:
            return np.zeros((0, 3))
        points = np.column_stack((
            indices[:, 0].astype(float), indices[:, 1].astype(float),
            np.zeros(indices.shape[0]),
        ))
        roll_mat = np.array([
            [1, 0, 0],
            [0, np.cos(-np.pi / 4), -np.sin(-np.pi / 4)],
            [0, np.sin(-np.pi / 4), np.cos(-np.pi / 4)],
        ])
        points = np.dot(points, roll_mat)
        points = points * character_size + [xpos + 0.2 + 0.5 * offset, ypos, zpos]
        return points

    def _points_along_line(x0, y0, z0, x1, y1, z1, resolution=0.05):
        n = int(np.linalg.norm(np.array([x1, y1, z1]) - np.array([x0, y0, z0])) / resolution)
        if n < 2:
            n = 2
        return np.column_stack((
            np.linspace(x0, x1, n), np.linspace(y0, y1, n), np.linspace(z0, z1, n),
        ))

    def _circle_points(x, y, z, r, n=100):
        angles = np.linspace(0, 2 * np.pi, n)
        return np.column_stack((r * np.cos(angles) + x, r * np.sin(angles) + y, np.full(n, z)))

    text_size = 0.00256
    line_height = 0.025
    all_parts = []

    for _, row in tree_data.iterrows():
        tree_id = int(row["TreeId"])
        dbh = float(row["DBH"])
        cci = float(row["CCI_at_BH"])
        height = float(row["Height"])
        vol1 = float(row["Volume_1"])
        vol2 = float(row["Volume_2"])
        x_base = float(row["x_tree_base"])
        y_base = float(row["y_tree_base"])
        z_base = float(row["z_tree_base"])

        if dbh == 0 or x_base == 0 or y_base == 0:
            continue

        # DBH position (approximate — use crown_mean or base offset)
        dbh_x = x_base
        dbh_y = y_base
        dbh_z = z_base + 1.3

        all_parts.append(_make_text(text_size, dbh_x, dbh_y + 2 * line_height, dbh_z + 2 * line_height, dbh * 0.5,
                                    "         TREE ID: " + str(tree_id)))
        all_parts.append(_make_text(text_size, dbh_x, dbh_y + line_height, dbh_z + line_height, dbh * 0.5,
                                    "            DIAM: " + str(np.around(dbh, 2)) + "m"))
        all_parts.append(_make_text(text_size, dbh_x, dbh_y, dbh_z, dbh * 0.5,
                                    "       CCI AT BH: " + str(np.around(cci, 2))))
        all_parts.append(_make_text(text_size, dbh_x, dbh_y - 2 * line_height, dbh_z - 2 * line_height, dbh * 0.5,
                                    "          HEIGHT: " + str(np.around(height, 2)) + "m"))
        all_parts.append(_make_text(text_size, dbh_x, dbh_y - 3 * line_height, dbh_z - 3 * line_height, dbh * 0.5,
                                    "          VOLUME 1: " + str(np.around(vol1, 2)) + "m3"))
        all_parts.append(_make_text(text_size, dbh_x, dbh_y - 4 * line_height, dbh_z - 4 * line_height, dbh * 0.5,
                                    "          VOLUME 2: " + str(np.around(vol2, 2)) + "m3"))
        # Height measurement line
        all_parts.append(_points_along_line(x_base, y_base, z_base, x_base, y_base, z_base + height, resolution=0.025))
        # DBH circle
        all_parts.append(_circle_points(dbh_x, dbh_y, dbh_z, dbh / 2))

    if all_parts:
        text_cloud = np.vstack([p for p in all_parts if p.shape[0] > 0])
        save_file(str(text_path), text_cloud)


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
                tree_df, id_mapping = registry.match_trees(tree_df)
                tree_df.to_csv(paths.tree_data_csv, index=False)

                # If any IDs changed, update them in all output files
                if id_mapping:
                    _remap_output_tree_ids(paths, id_mapping)
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
