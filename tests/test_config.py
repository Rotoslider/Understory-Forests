"""Tests for the configuration system."""

import tempfile
from pathlib import Path

import pytest

from understory.config.settings import (
    ProcessingConfig,
    ModelConfig,
    MeasurementConfig,
    OutputConfig,
    ProjectConfig,
)


class TestProcessingConfig:
    def test_defaults(self):
        cfg = ProcessingConfig()
        assert cfg.plot_centre is None
        assert cfg.plot_radius == 0
        assert cfg.slice_thickness == 0.15
        assert cfg.batch_size == 2
        assert cfg.use_CPU_only is False

    def test_custom_values(self):
        cfg = ProcessingConfig(plot_radius=10, batch_size=4)
        assert cfg.plot_radius == 10
        assert cfg.batch_size == 4


class TestModelConfig:
    def test_defaults(self):
        cfg = ModelConfig()
        assert cfg.model_filename == "model.pth"
        assert cfg.box_dimensions == [6, 6, 6]
        assert cfg.terrain_class == 1
        assert cfg.stem_class == 4


class TestProjectConfig:
    def test_to_legacy_params(self):
        cfg = ProjectConfig(point_cloud_filename="/tmp/test.las")
        cfg.processing.plot_radius = 15
        cfg.processing.batch_size = 4
        cfg.model.model_filename = "custom.pth"

        params = cfg.to_legacy_params()

        assert params["point_cloud_filename"] == "/tmp/test.las"
        assert params["plot_radius"] == 15
        assert params["batch_size"] == 4
        assert params["model_filename"] == "custom.pth"
        assert params["noise_class"] == 0
        assert params["stem_class"] == 4

    def test_from_legacy_params(self):
        params = {
            "point_cloud_filename": "/tmp/test.las",
            "plot_radius": 20,
            "batch_size": 8,
            "model_filename": "v3.pth",
            "box_dimensions": [8, 8, 8],
            "min_cluster_size": 50,
        }

        cfg = ProjectConfig.from_legacy_params(params)

        assert cfg.point_cloud_filename == "/tmp/test.las"
        assert cfg.processing.plot_radius == 20
        assert cfg.processing.batch_size == 8
        assert cfg.model.model_filename == "v3.pth"
        assert cfg.model.box_dimensions == [8, 8, 8]
        assert cfg.measurement.min_cluster_size == 50

    def test_roundtrip_legacy(self):
        """Verify: config -> legacy -> config preserves values."""
        original = ProjectConfig(point_cloud_filename="/tmp/test.las")
        original.processing.plot_radius = 12.5
        original.processing.slice_thickness = 0.2
        original.model.model_filename = "modelV2.pth"
        original.measurement.min_cluster_size = 40

        params = original.to_legacy_params()
        restored = ProjectConfig.from_legacy_params(params)

        assert restored.processing.plot_radius == 12.5
        assert restored.processing.slice_thickness == 0.2
        assert restored.model.model_filename == "modelV2.pth"
        assert restored.measurement.min_cluster_size == 40

    def test_yaml_save_load(self, tmp_path):
        cfg = ProjectConfig(
            project_name="Test Plot",
            operator="Tester",
            point_cloud_filename="/tmp/test.las",
        )
        cfg.processing.plot_radius = 25.0
        cfg.processing.batch_size = 4

        yaml_path = tmp_path / "test_project.yaml"
        cfg.save(yaml_path)

        assert yaml_path.exists()

        loaded = ProjectConfig.load(yaml_path)
        assert loaded.project_name == "Test Plot"
        assert loaded.operator == "Tester"
        assert loaded.point_cloud_filename == "/tmp/test.las"
        assert loaded.processing.plot_radius == 25.0
        assert loaded.processing.batch_size == 4


class TestLegacyParamsCompleteness:
    """Ensure to_legacy_params produces all keys the pipeline expects."""

    def test_all_run_py_keys_present(self):
        cfg = ProjectConfig(point_cloud_filename="test.las")
        params = cfg.to_legacy_params()

        expected_keys = [
            "point_cloud_filename",
            "plot_centre", "plot_radius", "plot_radius_buffer",
            "batch_size", "num_cpu_cores", "use_CPU_only",
            "slice_thickness", "slice_increment",
            "sort_stems", "height_percentile", "tree_base_cutoff_height",
            "generate_output_point_cloud",
            "ground_veg_cutoff_height", "veg_sorting_range", "stem_sorting_range",
            "taper_measurement_height_min", "taper_measurement_height_max",
            "taper_measurement_height_increment", "taper_slice_thickness",
            "delete_working_directory", "minimise_output_size_mode",
            "model_filename", "box_dimensions", "box_overlap",
            "min_points_per_box", "max_points_per_box",
            "noise_class", "terrain_class", "vegetation_class", "cwd_class", "stem_class",
            "grid_resolution", "num_neighbours",
            "sorting_search_angle", "sorting_search_radius", "sorting_angle_tolerance",
            "max_search_radius", "max_search_angle",
            "min_cluster_size", "cleaned_measurement_radius",
            "minimum_CCI", "min_tree_cyls",
            "subsample", "subsampling_min_spacing",
            "low_resolution_point_cloud_hack_mode",
            "vegetation_coverage_resolution",
        ]

        for key in expected_keys:
            assert key in params, f"Missing key: {key}"
