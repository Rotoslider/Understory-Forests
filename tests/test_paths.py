"""Tests for the FSCTPaths class."""

from pathlib import Path

import pytest

from understory.core.paths import FSCTPaths


class TestFSCTPaths:
    def test_default_output_dir(self):
        paths = FSCTPaths("/data/plots/plot1.las")
        assert paths.input_stem == "plot1"
        assert paths.input_filename == "plot1.las"
        assert paths.output_dir == Path("/data/plots/plot1_FSCT_output")
        assert paths.working_dir == Path("/data/plots/plot1_FSCT_output/working_directory")

    def test_custom_output_dir(self):
        paths = FSCTPaths("/data/plots/plot1.las", output_directory="/results/plot1")
        assert paths.output_dir == Path("/results/plot1")
        assert paths.working_dir == Path("/results/plot1/working_directory")

    def test_output_file_paths(self):
        paths = FSCTPaths("/data/plots/my_plot.las")
        assert paths.segmented_las.name == "segmented.las"
        assert paths.dtm_las.name == "DTM.las"
        assert paths.tree_data_csv.name == "tree_data.csv"
        assert paths.plot_summary_csv.name == "plot_summary.csv"

    def test_legacy_str_format(self):
        paths = FSCTPaths("/data/plots/plot1.las")
        assert paths.output_dir_str.endswith("/")
        assert paths.working_dir_str.endswith("/")

    def test_project_root(self):
        root = FSCTPaths.get_project_root()
        # Should be the FSCT directory
        assert root.name == "FSCT" or (root / "scripts").exists()

    def test_ensure_output_dirs(self, tmp_path):
        paths = FSCTPaths(tmp_path / "test.las")
        paths.ensure_output_dirs()
        assert paths.output_dir.exists()
        assert paths.working_dir.exists()

    def test_arbitrary_output_file(self):
        paths = FSCTPaths("/data/plot.las")
        custom = paths.output_file("my_custom_output.csv")
        assert custom.name == "my_custom_output.csv"
        assert custom.parent == paths.output_dir
