"""Run comparison report: compare tree_data.csv from two pipeline runs.

Computes per-tree deltas for DBH, Height, and Volume, identifies new and
removed trees, and renders a branded HTML comparison report using Jinja2.
"""

from __future__ import annotations

import base64
import io
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader
from matplotlib import pyplot as plt

# Re-use brand colors from main report
BRAND_COLORS = {
    "dark_forest": "#1a4a3a",
    "medium_forest": "#2d7a5e",
    "medium_green": "#4a9e7e",
    "light_mint": "#a8d8c0",
    "pale_mint": "#f0f7f4",
}


def compare_runs(run_a_dir: str, run_b_dir: str) -> dict:
    """Compare tree_data.csv from two pipeline output directories.

    Args:
        run_a_dir: Path to the baseline (earlier) run directory.
        run_b_dir: Path to the comparison (later) run directory.

    Returns:
        A dict containing all comparison data:
            run_a_dir, run_b_dir          -- original paths
            run_a_date, run_b_date        -- last-modified timestamps of tree_data.csv
            trees_a_count, trees_b_count  -- number of trees in each run
            matched                       -- list of dicts with TreeId and deltas
            new_trees                     -- list of dicts for trees only in B
            removed_trees                 -- list of dicts for trees only in A
            dbh_deltas, height_deltas, volume_deltas -- numpy arrays of deltas
            summary stats (mean/median/std for each delta)
    """
    path_a = Path(run_a_dir) / "tree_data.csv"
    path_b = Path(run_b_dir) / "tree_data.csv"

    if not path_a.exists():
        raise FileNotFoundError(f"tree_data.csv not found in run A: {run_a_dir}")
    if not path_b.exists():
        raise FileNotFoundError(f"tree_data.csv not found in run B: {run_b_dir}")

    df_a = pd.read_csv(path_a)
    df_b = pd.read_csv(path_b)

    # Timestamps from file modification times
    run_a_date = datetime.fromtimestamp(path_a.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    run_b_date = datetime.fromtimestamp(path_b.stat().st_mtime).strftime("%Y-%m-%d %H:%M")

    # Directory names as labels
    run_a_label = Path(run_a_dir).name
    run_b_label = Path(run_b_dir).name

    # Match trees by TreeId
    ids_a = set(df_a["TreeId"]) if "TreeId" in df_a.columns else set()
    ids_b = set(df_b["TreeId"]) if "TreeId" in df_b.columns else set()

    common_ids = sorted(ids_a & ids_b)
    new_ids = sorted(ids_b - ids_a)
    removed_ids = sorted(ids_a - ids_b)

    # Index dataframes by TreeId for fast lookup
    a_indexed = df_a.set_index("TreeId") if not df_a.empty else pd.DataFrame()
    b_indexed = df_b.set_index("TreeId") if not df_b.empty else pd.DataFrame()

    # Helper to safely get a numeric value
    def _val(df: pd.DataFrame, tree_id: int, col: str) -> float:
        if col not in df.columns or tree_id not in df.index:
            return 0.0
        v = df.loc[tree_id, col]
        if isinstance(v, pd.Series):
            v = v.iloc[0]
        return float(v) if pd.notna(v) else 0.0

    # Compute deltas for matched trees
    matched = []
    dbh_deltas = []
    height_deltas = []
    volume_deltas = []

    for tid in common_ids:
        dbh_a = _val(a_indexed, tid, "DBH")
        dbh_b = _val(b_indexed, tid, "DBH")
        h_a = _val(a_indexed, tid, "Height")
        h_b = _val(b_indexed, tid, "Height")
        v_a = _val(a_indexed, tid, "Volume_1")
        v_b = _val(b_indexed, tid, "Volume_1")

        d_dbh = dbh_b - dbh_a
        d_h = h_b - h_a
        d_v = v_b - v_a

        pct_dbh = (d_dbh / dbh_a * 100) if dbh_a != 0 else 0.0
        pct_h = (d_h / h_a * 100) if h_a != 0 else 0.0
        pct_v = (d_v / v_a * 100) if v_a != 0 else 0.0

        matched.append({
            "TreeId": int(tid),
            "DBH_A": dbh_a,
            "DBH_B": dbh_b,
            "DBH_delta": d_dbh,
            "DBH_pct": pct_dbh,
            "Height_A": h_a,
            "Height_B": h_b,
            "Height_delta": d_h,
            "Height_pct": pct_h,
            "Volume_A": v_a,
            "Volume_B": v_b,
            "Volume_delta": d_v,
            "Volume_pct": pct_v,
        })

        dbh_deltas.append(d_dbh)
        height_deltas.append(d_h)
        volume_deltas.append(d_v)

    dbh_deltas = np.array(dbh_deltas) if dbh_deltas else np.array([])
    height_deltas = np.array(height_deltas) if height_deltas else np.array([])
    volume_deltas = np.array(volume_deltas) if volume_deltas else np.array([])

    # New trees (in B but not A)
    new_trees = []
    for tid in new_ids:
        new_trees.append({
            "TreeId": int(tid),
            "DBH": _val(b_indexed, tid, "DBH"),
            "Height": _val(b_indexed, tid, "Height"),
            "Volume": _val(b_indexed, tid, "Volume_1"),
        })

    # Removed trees (in A but not B)
    removed_trees = []
    for tid in removed_ids:
        removed_trees.append({
            "TreeId": int(tid),
            "DBH": _val(a_indexed, tid, "DBH"),
            "Height": _val(a_indexed, tid, "Height"),
            "Volume": _val(a_indexed, tid, "Volume_1"),
        })

    # Summary statistics
    def _stats(arr: np.ndarray) -> dict:
        if len(arr) == 0:
            return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    return {
        "run_a_dir": str(run_a_dir),
        "run_b_dir": str(run_b_dir),
        "run_a_label": run_a_label,
        "run_b_label": run_b_label,
        "run_a_date": run_a_date,
        "run_b_date": run_b_date,
        "trees_a_count": len(df_a),
        "trees_b_count": len(df_b),
        "matched_count": len(common_ids),
        "new_count": len(new_ids),
        "removed_count": len(removed_ids),
        "matched": matched,
        "new_trees": new_trees,
        "removed_trees": removed_trees,
        "dbh_deltas": dbh_deltas,
        "height_deltas": height_deltas,
        "volume_deltas": volume_deltas,
        "dbh_stats": _stats(dbh_deltas),
        "height_stats": _stats(height_deltas),
        "volume_stats": _stats(volume_deltas),
    }


def _generate_delta_histogram(
    data: np.ndarray,
    title: str,
    xlabel: str,
    bin_width: float,
) -> str:
    """Generate a delta histogram and return it as a base64-encoded PNG data URI.

    The histogram is centred on zero with negative deltas in a muted tone and
    positive deltas in the brand green.

    Args:
        data: Array of delta values.
        title: Chart title.
        xlabel: X-axis label.
        bin_width: Width of histogram bins.

    Returns:
        A data URI string (``data:image/png;base64,...``) suitable for an
        ``<img src="...">`` attribute.
    """
    if len(data) == 0:
        return ""

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.set_title(title, fontsize=14, color=BRAND_COLORS["dark_forest"], fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Count", fontsize=11)

    # Compute bins centred on zero
    abs_max = max(abs(np.min(data)), abs(np.max(data)), bin_width)
    edge = np.ceil(abs_max / bin_width) * bin_width + bin_width
    bins = np.arange(-edge, edge + bin_width, bin_width)

    # Color bars by sign
    n, bin_edges, patches = ax.hist(
        data, bins=bins, linewidth=0.5,
        edgecolor=BRAND_COLORS["dark_forest"],
        facecolor=BRAND_COLORS["medium_green"],
        alpha=0.85,
    )
    for patch, left_edge in zip(patches, bin_edges[:-1]):
        mid = left_edge + bin_width / 2
        if mid < 0:
            patch.set_facecolor(BRAND_COLORS["light_mint"])
        else:
            patch.set_facecolor(BRAND_COLORS["medium_green"])

    # Zero line
    ax.axvline(0, color=BRAND_COLORS["dark_forest"], linewidth=1.2, linestyle="--", alpha=0.6)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BRAND_COLORS["light_mint"])
    ax.spines["bottom"].set_color(BRAND_COLORS["light_mint"])

    # Add summary annotation
    mean_val = np.mean(data)
    ax.annotate(
        f"mean = {mean_val:+.4f}",
        xy=(0.97, 0.92), xycoords="axes fraction",
        ha="right", fontsize=10, color=BRAND_COLORS["dark_forest"],
        bbox=dict(boxstyle="round,pad=0.3", facecolor=BRAND_COLORS["pale_mint"], edgecolor=BRAND_COLORS["light_mint"]),
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight", pad_inches=0.1, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def generate_comparison_report(comparison_data: dict, output_path: str) -> str:
    """Render a branded HTML comparison report from comparison data.

    Args:
        comparison_data: Dict returned by :func:`compare_runs`.
        output_path: Filesystem path where the HTML report will be written.

    Returns:
        The path to the generated HTML file (same as *output_path*).
    """
    # Generate histogram data URIs
    dbh_hist = _generate_delta_histogram(
        comparison_data["dbh_deltas"],
        "DBH Change Distribution",
        "DBH Delta (m)",
        bin_width=0.02,
    )
    height_hist = _generate_delta_histogram(
        comparison_data["height_deltas"],
        "Height Change Distribution",
        "Height Delta (m)",
        bin_width=0.5,
    )
    volume_hist = _generate_delta_histogram(
        comparison_data["volume_deltas"],
        "Volume Change Distribution",
        "Volume Delta (m\u00b3)",
        bin_width=0.1,
    )

    # Build template context
    report_date = datetime.now().strftime("%Y-%m-%d %H:%M")

    context = {
        **comparison_data,
        "report_date": report_date,
        "dbh_hist_uri": dbh_hist,
        "height_hist_uri": height_hist,
        "volume_hist_uri": volume_hist,
    }

    # Remove numpy arrays (not needed in template, and not JSON-serializable)
    context.pop("dbh_deltas", None)
    context.pop("height_deltas", None)
    context.pop("volume_deltas", None)

    # Copy logo if available
    logo_uri = ""
    logo_src = Path(__file__).parent.parent / "resources" / "icons" / "understory-logo.png"
    if logo_src.exists():
        logo_bytes = logo_src.read_bytes()
        logo_b64 = base64.b64encode(logo_bytes).decode("ascii")
        logo_uri = f"data:image/png;base64,{logo_b64}"
    context["logo_uri"] = logo_uri

    # Render template
    template_dir = str(Path(__file__).parent.parent / "resources")
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template("comparison_template.html")
    html = template.render(**context)

    output_path = str(output_path)
    Path(output_path).write_text(html, encoding="utf-8")
    return output_path
