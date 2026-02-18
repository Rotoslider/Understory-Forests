"""Persistent tree numbering via spatial matching.

Maintains a JSON registry of tree IDs and positions so that IDs remain
consistent across repeated scans of the same plot.

Matching algorithm:
    1. Build KD-tree from registered tree base positions
    2. For each newly measured tree, find nearest registered tree within
       configurable match_radius (default 2m)
    3. If multiple candidates, use DBH similarity as tiebreaker
    4. Unmatched new trees get max_id + 1
    5. Unmatched old trees remain in registry (may reappear in future scans)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


class TreeRegistry:
    """Persistent tree ID registry backed by a JSON file."""

    def __init__(self, registry_path: str | Path):
        self._path = Path(registry_path)
        self._trees: dict[int, dict] = {}
        self._next_id: int = 1

        if self._path.exists():
            self._load()

    def _load(self) -> None:
        with open(self._path) as f:
            data = json.load(f)
        # JSON keys are strings, convert to int
        self._trees = {int(k): v for k, v in data.get("trees", {}).items()}
        self._next_id = data.get("next_id", 1)
        if self._trees:
            self._next_id = max(self._next_id, max(self._trees.keys()) + 1)

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "next_id": self._next_id,
            "trees": {str(k): v for k, v in self._trees.items()},
        }
        with open(self._path, "w") as f:
            json.dump(data, f, indent=2)

    def match_trees(
        self,
        tree_data: pd.DataFrame,
        match_radius: float = 2.0,
        dbh_weight: float = 0.3,
    ) -> pd.DataFrame:
        """Match newly measured trees against the registry.

        Args:
            tree_data: DataFrame with columns: TreeId, x_tree_base, y_tree_base, DBH, etc.
            match_radius: Maximum distance (m) between tree bases for a match.
            dbh_weight: Weight of DBH similarity (0-1) in tiebreaking. 0 = pure distance.

        Returns:
            Updated tree_data DataFrame with persistent TreeId values.
        """
        if tree_data.empty:
            return tree_data

        tree_data = tree_data.copy()
        new_positions = np.column_stack([
            tree_data["x_tree_base"].values,
            tree_data["y_tree_base"].values,
        ])
        new_dbhs = tree_data["DBH"].values if "DBH" in tree_data.columns else np.zeros(len(tree_data))

        # Build mapping: new_index -> persistent_id
        persistent_ids = np.zeros(len(tree_data), dtype=int)
        used_registry_ids = set()

        if self._trees:
            # Build arrays from registry
            reg_ids = list(self._trees.keys())
            reg_positions = np.array([[t["x_base"], t["y_base"]] for t in self._trees.values()])
            reg_dbhs = np.array([t.get("dbh", 0) for t in self._trees.values()])

            from scipy.spatial import cKDTree
            reg_tree = cKDTree(reg_positions)

            for i in range(len(tree_data)):
                pos = new_positions[i]
                dbh = new_dbhs[i]

                # Find candidates within match_radius
                candidate_indices = reg_tree.query_ball_point(pos, r=match_radius)

                if candidate_indices:
                    # Filter out already-used registry entries
                    available = [j for j in candidate_indices if reg_ids[j] not in used_registry_ids]

                    if available:
                        # Score: distance + dbh_weight * |dbh_diff|
                        distances = np.linalg.norm(reg_positions[available] - pos, axis=1)
                        dbh_diffs = np.abs(reg_dbhs[available] - dbh) if dbh > 0 else np.zeros(len(available))

                        # Normalize
                        max_dist = max(distances.max(), 1e-6)
                        max_dbh_diff = max(dbh_diffs.max(), 1e-6)
                        scores = (1 - dbh_weight) * (distances / max_dist) + dbh_weight * (dbh_diffs / max_dbh_diff)

                        best_idx = available[np.argmin(scores)]
                        best_reg_id = reg_ids[best_idx]

                        persistent_ids[i] = best_reg_id
                        used_registry_ids.add(best_reg_id)
                        continue

                # No match — assign new ID
                persistent_ids[i] = self._next_id
                self._next_id += 1

        else:
            # No registry yet — assign sequential IDs
            for i in range(len(tree_data)):
                persistent_ids[i] = self._next_id
                self._next_id += 1

        # Update registry with new/updated tree data
        scan_time = datetime.now().isoformat()
        for i, pid in enumerate(persistent_ids):
            entry = self._trees.get(pid, {"scan_history": []})
            entry["x_base"] = float(new_positions[i, 0])
            entry["y_base"] = float(new_positions[i, 1])
            entry["dbh"] = float(new_dbhs[i])

            scan_record = {
                "date": scan_time,
                "dbh": float(new_dbhs[i]),
                "x_base": float(new_positions[i, 0]),
                "y_base": float(new_positions[i, 1]),
            }
            if "Height" in tree_data.columns:
                scan_record["height"] = float(tree_data.iloc[i]["Height"])

            entry.setdefault("scan_history", []).append(scan_record)
            self._trees[pid] = entry

        # Update DataFrame
        tree_data["TreeId"] = persistent_ids

        self._save()
        return tree_data

    def get_tree(self, tree_id: int) -> Optional[dict]:
        """Get registry data for a specific tree."""
        return self._trees.get(tree_id)

    def get_all_trees(self) -> dict[int, dict]:
        """Get all registered trees."""
        return dict(self._trees)

    def get_growth_data(self, tree_id: int) -> Optional[pd.DataFrame]:
        """Get DBH/height history for a tree across scans."""
        tree = self._trees.get(tree_id)
        if not tree or not tree.get("scan_history"):
            return None
        return pd.DataFrame(tree["scan_history"])

    @property
    def num_trees(self) -> int:
        return len(self._trees)

    def to_dataframe(self) -> pd.DataFrame:
        """Export the full registry as a DataFrame."""
        rows = []
        for tid, data in self._trees.items():
            row = {
                "TreeId": tid,
                "x_base": data["x_base"],
                "y_base": data["y_base"],
                "dbh": data.get("dbh", 0),
                "num_scans": len(data.get("scan_history", [])),
            }
            rows.append(row)
        return pd.DataFrame(rows)
