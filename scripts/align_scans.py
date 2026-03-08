"""Align and compare tree data from two scans of the same plot.

Finds the optimal rotation and translation to align scan B's tree positions
to scan A's coordinate frame, then generates an overlay stem map.

Usage:
    python align_scans.py <output_dir_A> <output_dir_B>
    python align_scans.py <output_dir_A> <output_dir_B> --match-radius 2.5
    python align_scans.py <tree_data_A.csv> <tree_data_B.csv>
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.spatial.distance import cdist

BRAND_COLORS = {
    "dark_forest": "#1a4a3a",
    "medium_forest": "#2d7a5e",
    "medium_green": "#4a9e7e",
    "light_mint": "#a8d8c0",
    "pale_mint": "#f0f7f4",
}

SCAN_A_COLOR = "#2d7a5e"  # medium forest green
SCAN_B_COLOR = "#c45a2d"  # burnt orange (distinct from green)
MATCH_LINE_COLOR = "#888888"


def load_tree_data(path: str) -> tuple[pd.DataFrame, str]:
    """Load tree_data.csv from a path or output directory."""
    p = Path(path)
    if p.is_dir():
        csv_path = p / "tree_data.csv"
    else:
        csv_path = p

    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    label = df["PlotId"].iloc[0] if "PlotId" in df.columns else csv_path.stem
    return df, label


def find_rotation(xy_a: np.ndarray, xy_b: np.ndarray, match_radius: float) -> dict:
    """Find the rotation angle that best aligns xy_b to xy_a.

    Uses brute-force angle search (1-degree steps) then refines to 0.1 degrees.
    Returns dict with angle, rotation matrix, translation, and match count.
    """
    best_angle = 0
    best_matches = 0

    # Coarse search: 1-degree steps
    for angle_deg in range(360):
        angle = np.radians(angle_deg)
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle),  np.cos(angle)]])
        b_rot = xy_b @ R.T
        dists = cdist(xy_a, b_rot)
        matches = np.sum(dists.min(axis=1) < match_radius)
        if matches > best_matches:
            best_matches = matches
            best_angle = angle_deg

    # Fine search: 0.1-degree steps around best
    for angle_deg in np.arange(best_angle - 2, best_angle + 2, 0.1):
        angle = np.radians(angle_deg)
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle),  np.cos(angle)]])
        b_rot = xy_b @ R.T
        dists = cdist(xy_a, b_rot)
        matches = np.sum(dists.min(axis=1) < match_radius)
        if matches > best_matches:
            best_matches = matches
            best_angle = angle_deg

    # Compute rotation matrix and estimate translation from matched pairs
    angle = np.radians(best_angle)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    b_rot = xy_b @ R.T

    dists = cdist(xy_a, b_rot)
    translation = np.array([0.0, 0.0])
    pairs = []
    for i in range(len(xy_a)):
        j = dists[i].argmin()
        if dists[i, j] < match_radius:
            pairs.append((i, j))

    if pairs:
        a_matched = xy_a[[p[0] for p in pairs]]
        b_matched = b_rot[[p[1] for p in pairs]]
        translation = np.mean(a_matched - b_matched, axis=0)

    return {
        "angle_deg": best_angle,
        "rotation_matrix": R,
        "translation": translation,
        "matches_before_translation": best_matches,
        "pairs": pairs,
    }


def apply_transform(xy: np.ndarray, result: dict) -> np.ndarray:
    """Apply the discovered rotation + translation to coordinates."""
    return xy @ result["rotation_matrix"].T + result["translation"]


def find_matched_pairs(xy_a: np.ndarray, xy_b_aligned: np.ndarray,
                       match_radius: float) -> list[tuple[int, int, float]]:
    """Find matched tree pairs after alignment. Returns (idx_a, idx_b, distance)."""
    dists = cdist(xy_a, xy_b_aligned)
    pairs = []
    used_b = set()
    # Greedy matching: closest pairs first
    flat_indices = np.argsort(dists, axis=None)
    for flat_idx in flat_indices:
        i, j = divmod(flat_idx, dists.shape[1])
        if dists[i, j] > match_radius:
            break
        if i not in {p[0] for p in pairs} and j not in used_b:
            pairs.append((int(i), int(j), float(dists[i, j])))
            used_b.add(j)
    return pairs


def generate_overlay_stem_map(
    df_a: pd.DataFrame, df_b: pd.DataFrame, xy_b_aligned: np.ndarray,
    pairs: list[tuple[int, int, float]],
    label_a: str, label_b: str, result: dict,
    output_path: Path,
):
    """Generate a stem map overlay showing both scans and matched pairs."""
    xy_a = df_a[["x_tree_base", "y_tree_base"]].values

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title("Scan Alignment — Stem Map Overlay",
                 fontsize=16, color=BRAND_COLORS["dark_forest"],
                 fontweight="bold", pad=12)
    ax.set_aspect("equal")

    # Draw match lines
    for i_a, i_b, dist in pairs:
        ax.plot([xy_a[i_a, 0], xy_b_aligned[i_b, 0]],
                [xy_a[i_a, 1], xy_b_aligned[i_b, 1]],
                color=MATCH_LINE_COLOR, linewidth=0.8, alpha=0.5, zorder=2)

    # Plot scan A trees
    ax.scatter(xy_a[:, 0], xy_a[:, 1], marker="o", s=60,
               facecolor=SCAN_A_COLOR, edgecolor="white",
               linewidth=1.2, zorder=5, label=label_a)

    # Plot scan B trees (aligned)
    ax.scatter(xy_b_aligned[:, 0], xy_b_aligned[:, 1], marker="^", s=60,
               facecolor=SCAN_B_COLOR, edgecolor="white",
               linewidth=1.2, zorder=5, label=label_b)

    # Label trees with IDs
    ids_a = df_a["TreeId"].values
    ids_b = df_b["TreeId"].values
    for i in range(len(xy_a)):
        ax.annotate(str(int(ids_a[i])), (xy_a[i, 0], xy_a[i, 1]),
                     textcoords="offset points", xytext=(5, 5),
                     fontsize=5, color=SCAN_A_COLOR, alpha=0.7, zorder=6)
    for i in range(len(xy_b_aligned)):
        ax.annotate(str(int(ids_b[i])), (xy_b_aligned[i, 0], xy_b_aligned[i, 1]),
                     textcoords="offset points", xytext=(5, -8),
                     fontsize=5, color=SCAN_B_COLOR, alpha=0.7, zorder=6)

    ax.set_xlabel("X (m)", fontsize=11)
    ax.set_ylabel("Y (m)", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend with stats
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=SCAN_A_COLOR,
               markersize=8, label=f"Scan A: {label_a} ({len(xy_a)} trees)"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor=SCAN_B_COLOR,
               markersize=8, label=f"Scan B: {label_b} ({len(xy_b_aligned)} trees)"),
        Line2D([0], [0], color=MATCH_LINE_COLOR, linewidth=1,
               label=f"Matched pairs: {len(pairs)}"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9,
              framealpha=0.9, edgecolor=BRAND_COLORS["light_mint"])

    # Add rotation info as text
    info_text = (f"Rotation: {result['angle_deg']:.1f}°\n"
                 f"Translation: ({result['translation'][0]:.2f}, "
                 f"{result['translation'][1]:.2f}) m")
    ax.text(0.02, 0.02, info_text, transform=ax.transAxes, fontsize=9,
            verticalalignment="bottom", color=BRAND_COLORS["dark_forest"],
            bbox=dict(boxstyle="round,pad=0.4", facecolor=BRAND_COLORS["pale_mint"],
                      edgecolor=BRAND_COLORS["light_mint"], alpha=0.9))

    fig.savefig(output_path, dpi=200, bbox_inches="tight",
                pad_inches=0.1, facecolor="white")
    plt.close(fig)
    print(f"Stem map overlay saved to: {output_path}")


def generate_match_report(
    df_a: pd.DataFrame, df_b: pd.DataFrame,
    pairs: list[tuple[int, int, float]], output_path: Path,
):
    """Save a CSV of matched tree pairs with measurements from both scans."""
    rows = []
    for i_a, i_b, dist in pairs:
        a = df_a.iloc[i_a]
        b = df_b.iloc[i_b]
        rows.append({
            "TreeId_A": int(a["TreeId"]),
            "TreeId_B": int(b["TreeId"]),
            "Position_Offset_m": round(dist, 3),
            "DBH_A": a.get("DBH", np.nan),
            "DBH_B": b.get("DBH", np.nan),
            "DBH_Diff": round(b.get("DBH", np.nan) - a.get("DBH", np.nan), 4)
            if pd.notna(a.get("DBH")) and pd.notna(b.get("DBH")) else np.nan,
            "Height_A": round(a.get("Height", np.nan), 2),
            "Height_B": round(b.get("Height", np.nan), 2),
            "x_base_A": round(a["x_tree_base"], 3),
            "y_base_A": round(a["y_tree_base"], 3),
            "x_base_B": round(b["x_tree_base"], 3),
            "y_base_B": round(b["y_tree_base"], 3),
        })

    match_df = pd.DataFrame(rows)
    match_df.to_csv(output_path, index=False)
    print(f"Match report saved to: {output_path}")
    return match_df


def main():
    parser = argparse.ArgumentParser(
        description="Align and compare tree data from two scans of the same plot."
    )
    parser.add_argument("scan_a", help="Output directory or tree_data.csv for scan A (reference)")
    parser.add_argument("scan_b", help="Output directory or tree_data.csv for scan B (to be aligned)")
    parser.add_argument("--match-radius", type=float, default=2.0,
                        help="Max distance (m) to consider two trees a match (default: 2.0)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for output files (default: parent of scan_a)")
    args = parser.parse_args()

    df_a, label_a = load_tree_data(args.scan_a)
    df_b, label_b = load_tree_data(args.scan_b)

    xy_a = df_a[["x_tree_base", "y_tree_base"]].values
    xy_b = df_b[["x_tree_base", "y_tree_base"]].values

    print(f"Scan A: {label_a} — {len(df_a)} trees")
    print(f"Scan B: {label_b} — {len(df_b)} trees")
    print(f"Match radius: {args.match_radius} m")
    print()

    # Check how many match without any rotation (already aligned?)
    dists_raw = cdist(xy_a, xy_b)
    raw_matches = np.sum(dists_raw.min(axis=1) < args.match_radius)
    print(f"Matches with no rotation: {raw_matches} / {len(df_a)}")

    # Find optimal rotation
    print("Searching for best rotation angle...")
    result = find_rotation(xy_a, xy_b, args.match_radius)
    xy_b_aligned = apply_transform(xy_b, result)

    # Find final matched pairs
    pairs = find_matched_pairs(xy_a, xy_b_aligned, args.match_radius)

    print(f"Best rotation: {result['angle_deg']:.1f}°")
    print(f"Translation: dx={result['translation'][0]:.2f} m, "
          f"dy={result['translation'][1]:.2f} m")
    print(f"Matched trees: {len(pairs)} / {len(df_a)} (scan A) "
          f"and {len(pairs)} / {len(df_b)} (scan B)")

    if pairs:
        match_dists = [p[2] for p in pairs]
        print(f"Match distance — mean: {np.mean(match_dists):.2f} m, "
              f"max: {np.max(match_dists):.2f} m")

    # Output directory
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        p = Path(args.scan_a)
        out_dir = p if p.is_dir() else p.parent

    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate overlay stem map
    generate_overlay_stem_map(
        df_a, df_b, xy_b_aligned, pairs,
        label_a, label_b, result,
        out_dir / "Scan_Alignment_Overlay.png",
    )

    # Generate match report CSV
    if pairs:
        match_df = generate_match_report(df_a, df_b, pairs, out_dir / "scan_match_report.csv")

        print()
        print("=== DBH Comparison (matched trees) ===")
        dbh_diffs = match_df["DBH_Diff"].dropna()
        if len(dbh_diffs) > 0:
            print(f"  Mean DBH difference (B - A): {dbh_diffs.mean():.4f} m")
            print(f"  Std DBH difference:          {dbh_diffs.std():.4f} m")
            print(f"  Mean absolute difference:    {dbh_diffs.abs().mean():.4f} m")


if __name__ == "__main__":
    main()
