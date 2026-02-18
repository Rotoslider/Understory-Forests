"""Structured configuration dataclasses for Understory.

Replaces the flat parameter dicts from run.py and other_parameters.py with
typed, documented configuration objects that can serialize to/from YAML.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ProcessingConfig:
    """Parameters controlling the processing pipeline."""

    # Plot geometry
    plot_centre: Optional[list[float]] = None  # [X, Y] metres; None = auto-compute from bounding box
    plot_radius: float = 0  # 0 = no crop; otherwise cylindrical crop radius (m)
    plot_radius_buffer: float = 0  # extra buffer for tree-aware cropping (m)

    # Slicing
    slice_thickness: float = 0.15  # height of each horizontal slice (m)
    slice_increment: float = 0.05  # vertical step between slices (m)

    # Stem sorting
    sort_stems: bool = True
    height_percentile: float = 100  # percentile for max height (98 filters noise)
    tree_base_cutoff_height: float = 5  # max height above DTM for a tree base (m)
    generate_output_point_cloud: bool = True

    # Vegetation
    ground_veg_cutoff_height: float = 3  # below this = understory (m)
    veg_sorting_range: float = 1.5  # max horizontal distance to match veg to tree (m)
    stem_sorting_range: float = 1.0  # max 3D distance to match stem to tree (m)

    # Taper
    taper_measurement_height_min: float = 0
    taper_measurement_height_max: float = 30
    taper_measurement_height_increment: float = 0.2
    taper_slice_thickness: float = 0.4

    # Performance
    batch_size: int = 2  # inference batch size (lower if CUDA OOM)
    num_cpu_cores: int = 0  # 0 = all cores
    use_CPU_only: bool = False

    # Cleanup
    delete_working_directory: bool = True
    minimise_output_size_mode: bool = False

    # Subsampling
    subsample: bool = False
    subsampling_min_spacing: float = 0.01

    # Low resolution hack
    low_resolution_point_cloud_hack_mode: int = 0


@dataclass
class ModelConfig:
    """Parameters for the deep learning model."""

    model_filename: str = "model.pth"
    box_dimensions: list[float] = field(default_factory=lambda: [6, 6, 6])
    box_overlap: list[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    min_points_per_box: int = 1000
    max_points_per_box: int = 20000

    # Class labels (fixed)
    noise_class: int = 0
    terrain_class: int = 1
    vegetation_class: int = 2
    cwd_class: int = 3
    stem_class: int = 4


@dataclass
class MeasurementConfig:
    """Parameters for tree measurement algorithms."""

    grid_resolution: float = 0.5  # DTM resolution (m)
    vegetation_coverage_resolution: float = 0.2
    num_neighbours: int = 5
    sorting_search_angle: float = 20
    sorting_search_radius: float = 1
    sorting_angle_tolerance: float = 90
    max_search_radius: float = 3
    max_search_angle: float = 30
    min_cluster_size: int = 30  # HDBSCAN clustering
    cleaned_measurement_radius: float = 0.2
    minimum_CCI: float = 0.3  # min Circumferential Completeness Index
    min_tree_cyls: int = 10  # min cylinders per tree


@dataclass
class OutputConfig:
    """Parameters controlling what outputs are generated."""

    output_directory: Optional[str] = None  # None = auto (next to input file)
    project_directory: Optional[str] = None  # Understory project folder root
    generate_report: bool = True
    generate_stem_map: bool = True
    generate_histograms: bool = True
    generate_point_clouds: bool = True
    generate_csvs: bool = True


@dataclass
class TrainingConfig:
    """Parameters for model training."""

    preprocess_train_datasets: bool = True
    preprocess_validation_datasets: bool = True
    clean_sample_directories: bool = True
    perform_validation_during_training: bool = True
    generate_point_cloud_vis: bool = False
    load_existing_model: bool = True
    num_epochs: int = 2000
    learning_rate: float = 0.000025
    sample_box_size_m: list[float] = field(default_factory=lambda: [6, 6, 6])
    sample_box_overlap: list[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    train_batch_size: int = 2
    validation_batch_size: int = 2
    num_cpu_cores_preprocessing: int = 0
    num_cpu_cores_deep_learning: int = 1
    device: str = "cuda"


@dataclass
class ProjectConfig:
    """Top-level project configuration that ties everything together."""

    # Project metadata
    project_name: str = ""
    operator: str = ""
    notes: str = ""
    photos: list[str] = field(default_factory=list)  # attached field photo paths

    # Input
    point_cloud_filename: str = ""
    prepared_point_cloud: str = ""  # subsampled/cropped version saved by user

    # Pipeline stage toggles
    preprocess: bool = True
    segmentation: bool = True
    postprocessing: bool = True
    measure_plot: bool = True
    make_report: bool = True
    clean_up_files: bool = False

    # Sub-configs
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    measurement: MeasurementConfig = field(default_factory=MeasurementConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_legacy_params(self) -> dict:
        """Convert structured config back to the flat dict that all existing
        pipeline classes expect, enabling incremental migration."""
        params = {}

        # From ProjectConfig
        params["point_cloud_filename"] = self.point_cloud_filename

        # From OutputConfig — output directory override for project folder structure
        if self.output.output_directory:
            params["output_dir"] = self.output.output_directory

        # From ProcessingConfig
        p = self.processing
        params["plot_centre"] = p.plot_centre
        params["plot_radius"] = p.plot_radius
        params["plot_radius_buffer"] = p.plot_radius_buffer
        params["slice_thickness"] = p.slice_thickness
        params["slice_increment"] = p.slice_increment
        params["sort_stems"] = int(p.sort_stems)
        params["height_percentile"] = p.height_percentile
        params["tree_base_cutoff_height"] = p.tree_base_cutoff_height
        params["generate_output_point_cloud"] = int(p.generate_output_point_cloud)
        params["ground_veg_cutoff_height"] = p.ground_veg_cutoff_height
        params["veg_sorting_range"] = p.veg_sorting_range
        params["stem_sorting_range"] = p.stem_sorting_range
        params["taper_measurement_height_min"] = p.taper_measurement_height_min
        params["taper_measurement_height_max"] = p.taper_measurement_height_max
        params["taper_measurement_height_increment"] = p.taper_measurement_height_increment
        params["taper_slice_thickness"] = p.taper_slice_thickness
        params["batch_size"] = p.batch_size
        params["num_cpu_cores"] = p.num_cpu_cores
        params["use_CPU_only"] = p.use_CPU_only
        params["delete_working_directory"] = p.delete_working_directory
        params["minimise_output_size_mode"] = int(p.minimise_output_size_mode)
        params["subsample"] = int(p.subsample)
        params["subsampling_min_spacing"] = p.subsampling_min_spacing
        params["low_resolution_point_cloud_hack_mode"] = p.low_resolution_point_cloud_hack_mode

        # From ModelConfig
        m = self.model
        params["model_filename"] = m.model_filename
        params["box_dimensions"] = m.box_dimensions
        params["box_overlap"] = m.box_overlap
        params["min_points_per_box"] = m.min_points_per_box
        params["max_points_per_box"] = m.max_points_per_box
        params["noise_class"] = m.noise_class
        params["terrain_class"] = m.terrain_class
        params["vegetation_class"] = m.vegetation_class
        params["cwd_class"] = m.cwd_class
        params["stem_class"] = m.stem_class

        # From MeasurementConfig
        me = self.measurement
        params["grid_resolution"] = me.grid_resolution
        params["vegetation_coverage_resolution"] = me.vegetation_coverage_resolution
        params["num_neighbours"] = me.num_neighbours
        params["sorting_search_angle"] = me.sorting_search_angle
        params["sorting_search_radius"] = me.sorting_search_radius
        params["sorting_angle_tolerance"] = me.sorting_angle_tolerance
        params["max_search_radius"] = me.max_search_radius
        params["max_search_angle"] = me.max_search_angle
        params["min_cluster_size"] = me.min_cluster_size
        params["cleaned_measurement_radius"] = me.cleaned_measurement_radius
        params["minimum_CCI"] = me.minimum_CCI
        params["min_tree_cyls"] = me.min_tree_cyls

        return params

    def save(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        path = Path(path)
        data = asdict(self)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load(cls, path: str | Path) -> "ProjectConfig":
        """Load configuration from a YAML file."""
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)

        # Reconstruct nested dataclasses
        config = cls(
            project_name=data.get("project_name", ""),
            operator=data.get("operator", ""),
            notes=data.get("notes", ""),
            point_cloud_filename=data.get("point_cloud_filename", ""),
            prepared_point_cloud=data.get("prepared_point_cloud", ""),
            preprocess=data.get("preprocess", True),
            segmentation=data.get("segmentation", True),
            postprocessing=data.get("postprocessing", True),
            measure_plot=data.get("measure_plot", True),
            make_report=data.get("make_report", True),
            clean_up_files=data.get("clean_up_files", False),
            processing=ProcessingConfig(**data.get("processing", {})),
            model=ModelConfig(**data.get("model", {})),
            measurement=MeasurementConfig(**data.get("measurement", {})),
            output=OutputConfig(**data.get("output", {})),
            training=TrainingConfig(**data.get("training", {})),
        )
        return config

    @classmethod
    def from_legacy_params(cls, params: dict) -> "ProjectConfig":
        """Create a ProjectConfig from a legacy flat parameter dict."""
        config = cls()
        config.point_cloud_filename = params.get("point_cloud_filename", "")

        p = config.processing
        p.plot_centre = params.get("plot_centre")
        p.plot_radius = params.get("plot_radius", 0)
        p.plot_radius_buffer = params.get("plot_radius_buffer", 0)
        p.slice_thickness = params.get("slice_thickness", 0.15)
        p.slice_increment = params.get("slice_increment", 0.05)
        p.sort_stems = bool(params.get("sort_stems", 1))
        p.height_percentile = params.get("height_percentile", 100)
        p.tree_base_cutoff_height = params.get("tree_base_cutoff_height", 5)
        p.generate_output_point_cloud = bool(params.get("generate_output_point_cloud", 1))
        p.ground_veg_cutoff_height = params.get("ground_veg_cutoff_height", 3)
        p.veg_sorting_range = params.get("veg_sorting_range", 1.5)
        p.stem_sorting_range = params.get("stem_sorting_range", 1)
        p.taper_measurement_height_min = params.get("taper_measurement_height_min", 0)
        p.taper_measurement_height_max = params.get("taper_measurement_height_max", 30)
        p.taper_measurement_height_increment = params.get("taper_measurement_height_increment", 0.2)
        p.taper_slice_thickness = params.get("taper_slice_thickness", 0.4)
        p.batch_size = params.get("batch_size", 2)
        p.num_cpu_cores = params.get("num_cpu_cores", 0)
        p.use_CPU_only = params.get("use_CPU_only", False)
        p.delete_working_directory = params.get("delete_working_directory", True)
        p.minimise_output_size_mode = bool(params.get("minimise_output_size_mode", 0))
        p.subsample = bool(params.get("subsample", 0))
        p.subsampling_min_spacing = params.get("subsampling_min_spacing", 0.01)
        p.low_resolution_point_cloud_hack_mode = params.get("low_resolution_point_cloud_hack_mode", 0)

        m = config.model
        m.model_filename = params.get("model_filename", "model.pth")
        m.box_dimensions = params.get("box_dimensions", [6, 6, 6])
        m.box_overlap = params.get("box_overlap", [0.5, 0.5, 0.5])
        m.min_points_per_box = params.get("min_points_per_box", 1000)
        m.max_points_per_box = params.get("max_points_per_box", 20000)

        me = config.measurement
        me.grid_resolution = params.get("grid_resolution", 0.5)
        me.vegetation_coverage_resolution = params.get("vegetation_coverage_resolution", 0.2)
        me.num_neighbours = params.get("num_neighbours", 5)
        me.sorting_search_angle = params.get("sorting_search_angle", 20)
        me.sorting_search_radius = params.get("sorting_search_radius", 1)
        me.sorting_angle_tolerance = params.get("sorting_angle_tolerance", 90)
        me.max_search_radius = params.get("max_search_radius", 3)
        me.max_search_angle = params.get("max_search_angle", 30)
        me.min_cluster_size = params.get("min_cluster_size", 30)
        me.cleaned_measurement_radius = params.get("cleaned_measurement_radius", 0.2)
        me.minimum_CCI = params.get("minimum_CCI", 0.3)
        me.min_tree_cyls = params.get("min_tree_cyls", 10)

        return config
