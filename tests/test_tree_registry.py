"""Tests for the persistent tree numbering system."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from understory.core.tree_registry import TreeRegistry


@pytest.fixture
def registry_path(tmp_path):
    return tmp_path / "tree_registry.json"


@pytest.fixture
def sample_tree_data():
    """Create a simple tree_data DataFrame."""
    return pd.DataFrame({
        "TreeId": [1, 2, 3],
        "x_tree_base": [10.0, 20.0, 30.0],
        "y_tree_base": [10.0, 20.0, 30.0],
        "DBH": [0.3, 0.5, 0.2],
        "Height": [15.0, 25.0, 10.0],
    })


class TestTreeRegistry:
    def test_new_registry(self, registry_path, sample_tree_data):
        """First scan — all trees get new IDs."""
        reg = TreeRegistry(registry_path)
        result = reg.match_trees(sample_tree_data)

        assert len(result) == 3
        assert set(result["TreeId"]) == {1, 2, 3}
        assert registry_path.exists()
        assert reg.num_trees == 3

    def test_same_plot_matches(self, registry_path, sample_tree_data):
        """Process same plot twice — IDs should match."""
        reg1 = TreeRegistry(registry_path)
        result1 = reg1.match_trees(sample_tree_data)
        ids1 = set(result1["TreeId"])

        # Second scan of same plot
        reg2 = TreeRegistry(registry_path)
        result2 = reg2.match_trees(sample_tree_data)
        ids2 = set(result2["TreeId"])

        assert ids1 == ids2

    def test_shifted_plot_matches(self, registry_path, sample_tree_data):
        """Slightly shifted positions should still match."""
        reg = TreeRegistry(registry_path)
        reg.match_trees(sample_tree_data)

        # Shift positions slightly (within 2m default radius)
        shifted = sample_tree_data.copy()
        shifted["x_tree_base"] += 0.5
        shifted["y_tree_base"] -= 0.3

        reg2 = TreeRegistry(registry_path)
        result = reg2.match_trees(shifted)

        # Same tree IDs should be assigned
        assert set(result["TreeId"]) == {1, 2, 3}

    def test_new_tree_gets_new_id(self, registry_path, sample_tree_data):
        """A new tree should get a new ID without affecting existing ones."""
        reg = TreeRegistry(registry_path)
        reg.match_trees(sample_tree_data)

        # Add a new tree far from existing ones
        with_new = sample_tree_data.copy()
        new_tree = pd.DataFrame({
            "TreeId": [99],
            "x_tree_base": [100.0],
            "y_tree_base": [100.0],
            "DBH": [0.4],
            "Height": [20.0],
        })
        with_new = pd.concat([with_new, new_tree], ignore_index=True)

        reg2 = TreeRegistry(registry_path)
        result = reg2.match_trees(with_new)

        # Original 3 should keep IDs 1-3, new one gets 4
        original_ids = set(result.iloc[:3]["TreeId"])
        new_id = result.iloc[3]["TreeId"]

        assert original_ids == {1, 2, 3}
        assert new_id == 4

    def test_removed_tree_doesnt_affect_others(self, registry_path, sample_tree_data):
        """Removing a tree should not affect other tree IDs."""
        reg = TreeRegistry(registry_path)
        reg.match_trees(sample_tree_data)

        # Remove tree 2 (middle one)
        reduced = sample_tree_data[sample_tree_data["TreeId"].isin([1, 3])].copy()

        reg2 = TreeRegistry(registry_path)
        result = reg2.match_trees(reduced)

        assert set(result["TreeId"]) == {1, 3}

    def test_scan_history_accumulates(self, registry_path, sample_tree_data):
        """Multiple scans should build up scan history."""
        reg = TreeRegistry(registry_path)
        reg.match_trees(sample_tree_data)
        reg.match_trees(sample_tree_data)

        reg3 = TreeRegistry(registry_path)
        tree1 = reg3.get_tree(1)
        assert tree1 is not None
        assert len(tree1["scan_history"]) == 2

    def test_growth_data(self, registry_path, sample_tree_data):
        """Growth data should return scan history as DataFrame."""
        reg = TreeRegistry(registry_path)
        reg.match_trees(sample_tree_data)

        growth = reg.get_growth_data(1)
        assert growth is not None
        assert "dbh" in growth.columns
        assert len(growth) == 1

    def test_to_dataframe(self, registry_path, sample_tree_data):
        """Registry should export as DataFrame."""
        reg = TreeRegistry(registry_path)
        reg.match_trees(sample_tree_data)

        df = reg.to_dataframe()
        assert len(df) == 3
        assert "TreeId" in df.columns
        assert "x_base" in df.columns
        assert "num_scans" in df.columns

    def test_empty_tree_data(self, registry_path):
        """Empty tree data should not crash."""
        reg = TreeRegistry(registry_path)
        empty = pd.DataFrame(columns=["TreeId", "x_tree_base", "y_tree_base", "DBH"])
        result = reg.match_trees(empty)
        assert len(result) == 0
