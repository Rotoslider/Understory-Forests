"""Tests for scripts/tools.py — file I/O, clustering, subsampling."""

import os
import sys
from pathlib import Path

import laspy
import numpy as np
import pytest

# Ensure scripts/ is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from tools import (
    get_fsct_path,
    load_file,
    save_file,
    subsample_point_cloud,
    cluster_dbscan,
    cluster_hdbscan,
    get_heights_above_DTM,
)


class TestGetFSCTPath:
    def test_returns_project_root(self):
        root = get_fsct_path()
        assert os.path.isdir(root)
        assert os.path.isdir(os.path.join(root, "scripts"))

    def test_subpath(self):
        model_dir = get_fsct_path("model")
        assert model_dir.endswith("model")

    def test_no_backslashes(self):
        path = get_fsct_path("model")
        assert "\\" not in path


class TestLoadFile:
    def test_load_las(self, example_las):
        pc, headers = load_file(example_las, silent=True)
        assert pc.shape[0] > 0
        assert pc.shape[1] >= 3
        assert headers[:3] == ["x", "y", "z"]

    def test_load_las_with_headers(self, example_las):
        pc, headers = load_file(
            example_las,
            headers_of_interest=["x", "y", "z", "red", "green", "blue"],
            silent=True,
        )
        assert pc.shape[0] > 0
        # May or may not have color headers depending on file
        assert len(headers) >= 3

    def test_load_with_crop(self, example_las):
        pc_full, _ = load_file(example_las, silent=True)
        centre = np.mean(pc_full[:, :2], axis=0).tolist()
        pc_crop, _ = load_file(
            example_las,
            plot_centre=centre,
            plot_radius=5.0,
            silent=True,
        )
        assert pc_crop.shape[0] < pc_full.shape[0]
        assert pc_crop.shape[0] > 0

    def test_load_with_return_num_points(self, example_las):
        pc, headers, num = load_file(example_las, silent=True, return_num_points=True)
        assert num > 0
        assert num >= pc.shape[0]

    def test_load_missing_file(self):
        pc, headers = load_file("/nonexistent/file.las", silent=True)
        assert pc.shape[0] == 0

    def test_load_pcd(self, tmp_path, example_las):
        """Round-trip: LAS -> PCD -> load PCD."""
        import open3d as o3d
        pc, _ = load_file(example_las, silent=True)
        # Create a small PCD
        pcd_path = str(tmp_path / "test.pcd")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc[:100, :3])
        o3d.io.write_point_cloud(pcd_path, pcd)

        pc_pcd, headers = load_file(pcd_path, silent=True)
        assert pc_pcd.shape[0] == 100
        assert pc_pcd.shape[1] == 3


class TestSaveFile:
    def test_save_las(self, tmp_path):
        pc = np.random.rand(500, 3) * 100
        out_path = str(tmp_path / "output.las")
        save_file(out_path, pc, silent=True)
        assert os.path.exists(out_path)

        # Verify we can load it back
        loaded, headers = load_file(out_path, silent=True)
        assert loaded.shape[0] == 500
        np.testing.assert_allclose(loaded[:, :3], pc, atol=0.002)  # LAS precision

    def test_save_las_with_headers(self, tmp_path):
        pc = np.random.rand(100, 6) * 100
        headers = ["x", "y", "z", "red", "green", "blue"]
        out_path = str(tmp_path / "rgb.las")
        save_file(out_path, pc, headers_of_interest=headers, silent=True)
        assert os.path.exists(out_path)

    def test_save_pcd(self, tmp_path):
        pc = np.random.rand(200, 3) * 50
        out_path = str(tmp_path / "output.pcd")
        save_file(out_path, pc, silent=True)
        assert os.path.exists(out_path)

    def test_save_empty(self, tmp_path, capsys):
        pc = np.zeros((0, 3))
        out_path = str(tmp_path / "empty.las")
        save_file(out_path, pc, silent=False)
        captured = capsys.readouterr()
        assert "empty" in captured.out.lower()


class TestClustering:
    def test_dbscan(self):
        # Two clusters
        cluster1 = np.random.rand(50, 3) * 0.01
        cluster2 = np.random.rand(50, 3) * 0.01 + 10
        points = np.vstack([cluster1, cluster2])
        result = cluster_dbscan(points, eps=0.1, min_samples=5)
        assert result.shape[0] == 100
        assert result.shape[1] == 4  # x, y, z, label

    def test_hdbscan(self):
        cluster1 = np.random.rand(50, 3) * 0.01
        cluster2 = np.random.rand(50, 3) * 0.01 + 10
        points = np.vstack([cluster1, cluster2])
        result = cluster_hdbscan(points, min_cluster_size=10)
        assert result.shape[0] == 100
        assert result.shape[1] == 4


class TestSubsampling:
    def test_subsample_reduces_points(self):
        np.random.seed(42)
        pc = np.random.rand(1000, 3) * 10
        subsampled = subsample_point_cloud(pc, min_spacing=0.5, num_cpu_cores=1)
        assert subsampled.shape[0] < 1000
        assert subsampled.shape[0] > 0


class TestHeightsAboveDTM:
    def test_basic(self):
        # Create a flat DTM
        dtm_x = np.linspace(0, 10, 20)
        dtm_y = np.linspace(0, 10, 20)
        xx, yy = np.meshgrid(dtm_x, dtm_y)
        dtm = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(400)])

        # Points 5m above
        points = np.array([[5.0, 5.0, 5.0, 0.0], [3.0, 3.0, 10.0, 0.0]])
        result = get_heights_above_DTM(points, dtm)
        np.testing.assert_allclose(result[0, -1], 5.0, atol=0.1)
        np.testing.assert_allclose(result[1, -1], 10.0, atol=0.1)
