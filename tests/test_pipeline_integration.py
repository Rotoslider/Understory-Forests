"""Integration tests for the Understory pipeline wrapper and report generation."""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from understory.config.settings import ProjectConfig
from understory.core.paths import FSCTPaths


class TestPipelineConfig:
    """Test that pipeline config correctly bridges to legacy params."""

    def test_config_produces_valid_params(self):
        cfg = ProjectConfig(point_cloud_filename="/tmp/test.las")
        params = cfg.to_legacy_params()

        # Pipeline classes access these keys
        assert "point_cloud_filename" in params
        assert "batch_size" in params
        assert "model_filename" in params
        assert "box_dimensions" in params
        assert isinstance(params["box_dimensions"], list)

    def test_config_gpu_settings(self):
        cfg = ProjectConfig(point_cloud_filename="/tmp/test.las")
        cfg.processing.use_CPU_only = False
        cfg.processing.batch_size = 4
        params = cfg.to_legacy_params()

        assert params["use_CPU_only"] is False
        assert params["batch_size"] == 4


class TestPipelinePreprocessing:
    """Test the preprocessing stage with real data."""

    @pytest.fixture
    def output_dir(self, tmp_path, example_las):
        """Set up a fresh output directory for testing."""
        paths = FSCTPaths(example_las, output_directory=str(tmp_path / "output"))
        paths.ensure_output_dirs()
        return paths

    def test_load_example_las(self, example_las):
        """Verify example.las loads correctly."""
        from tools import load_file
        pc, headers = load_file(example_las, silent=True)
        assert pc.shape[0] > 1000
        assert pc.shape[1] >= 3
        print(f"Loaded {pc.shape[0]:,} points with headers: {headers}")

    def test_preprocessing_class(self, example_las, tmp_path):
        """Test Preprocessing class initializes and can process."""
        cfg = ProjectConfig(point_cloud_filename=example_las)
        cfg.output = cfg.output  # defaults
        params = cfg.to_legacy_params()
        params["num_cpu_cores"] = os.cpu_count()

        # Just verify initialization doesn't crash
        from preprocessing import Preprocessing
        prep = Preprocessing(params)
        assert prep is not None


class TestReportTemplate:
    """Test the Jinja2 report template renders correctly."""

    def test_template_exists(self):
        template_path = Path(__file__).parent.parent / "understory" / "resources" / "report_template.html"
        assert template_path.exists()

    def test_template_renders(self):
        from jinja2 import Environment, FileSystemLoader
        template_dir = str(Path(__file__).parent.parent / "understory" / "resources")
        env = Environment(loader=FileSystemLoader(template_dir))
        template = env.get_template("report_template.html")

        html = template.render(
            filename="test.las",
            project_name="Test Project",
            operator="Tester",
            date="2026-02-15",
            notes="Test run",
            plot_centre_x=100.0,
            plot_centre_y=200.0,
            plot_radius=15.0,
            plot_radius_buffer=3.0,
            plot_area=706.86,
            num_trees=5,
            stems_per_ha=100,
            mean_dbh=0.3,
            median_dbh=0.28,
            min_dbh=0.1,
            max_dbh=0.6,
            mean_height=15.0,
            total_volume=5.2,
            canopy_cover=0.65,
            trees=[
                {"TreeId": 1, "DBH": 0.3, "Height": 15, "Volume_1": 1.2, "Volume_2": 1.0,
                 "CCI_at_BH": 0.85, "x_tree_base": 100.5, "y_tree_base": 200.3},
                {"TreeId": 2, "DBH": 0.5, "Height": 22, "Volume_1": 2.5, "Volume_2": 2.1,
                 "CCI_at_BH": 0.92, "x_tree_base": 103.2, "y_tree_base": 198.7},
            ],
            show_stem_map=False,
            show_histograms=False,
            logo_path=None,
            num_points_original=500000,
            num_points_trimmed=480000,
            num_points_subsampled=200000,
            num_terrain_points=80000,
            num_vegetation_points=60000,
            num_cwd_points=5000,
            num_stem_points=55000,
            understory_veg_coverage=0.35,
            cwd_coverage=0.05,
            avg_gradient=8.2,
            preprocessing_time=10,
            segmentation_time=30,
            postprocessing_time=5,
            measurement_time=20,
            total_time=65,
        )

        assert "Test Project" in html
        assert "test.las" in html
        assert "Tester" in html
        assert len(html) > 500


class TestFSCTPathsIntegration:
    def test_paths_with_real_file(self, example_las):
        paths = FSCTPaths(example_las)
        assert paths.input_stem == "example"
        assert paths.input_filename == "example.las"
        assert "FSCT_output" in str(paths.output_dir)

    def test_ensure_dirs_creates_structure(self, tmp_path):
        paths = FSCTPaths(str(tmp_path / "test.las"))
        paths.ensure_output_dirs()
        assert paths.output_dir.exists()
        assert paths.working_dir.exists()

    def test_model_dir_exists(self, project_root):
        model_dir = FSCTPaths.get_model_dir()
        assert model_dir.exists()
        assert (model_dir / "model.pth").exists()
