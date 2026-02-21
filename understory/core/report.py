"""Modernized report generation using Jinja2 HTML templates.

Replaces the mdutils/markdown-based report in scripts/report_writer.py
with a branded HTML report using the Understory color scheme.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

# Understory brand colors for matplotlib
BRAND_COLORS = {
    "dark_forest": "#1a4a3a",
    "medium_forest": "#2d7a5e",
    "medium_green": "#4a9e7e",
    "light_mint": "#a8d8c0",
    "pale_mint": "#f0f7f4",
}


def generate_report(
    output_dir: str,
    point_cloud_filename: str,
    project_name: str = "",
    operator: str = "",
    notes: str = "",
    show_stem_map: bool = True,
    show_histograms: bool = True,
    photos: list[str] | None = None,
) -> str:
    """Generate a branded HTML report from pipeline outputs.

    Args:
        output_dir: Path to the FSCT output directory.
        point_cloud_filename: Original point cloud filename.
        project_name: Optional project name for the report.
        operator: Optional operator name.
        notes: Optional notes text.
        show_stem_map: Whether to include the stem map image.
        show_histograms: Whether to include histogram images.
        photos: Optional list of field photo file paths to include.

    Returns:
        Path to the generated HTML report.
    """
    output_dir = Path(output_dir)

    # Load data
    plot_summary = pd.read_csv(output_dir / "plot_summary.csv", index_col=False)
    tree_data_path = output_dir / "tree_data.csv"
    tree_data = pd.read_csv(tree_data_path) if tree_data_path.exists() else pd.DataFrame()

    # Generate plots with brand colors
    _generate_branded_plots(output_dir, tree_data)

    # Generate stem map with brand colors
    _generate_branded_stem_map(output_dir, plot_summary, tree_data)

    # Generate taper profile chart if data exists
    has_taper = _generate_taper_chart(output_dir, tree_data)

    # Generate crown projection map
    has_crown_map = _generate_crown_map(output_dir, plot_summary, tree_data)

    # Build template context
    filename = os.path.basename(point_cloud_filename)
    num_trees = tree_data.shape[0] if not tree_data.empty else 0

    context = {
        "filename": filename,
        "project_name": project_name,
        "operator": operator,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "notes": notes,
        "plot_centre_x": float(plot_summary["Plot Centre X"].iloc[0]),
        "plot_centre_y": float(plot_summary["Plot Centre Y"].iloc[0]),
        "plot_radius": float(plot_summary["Plot Radius"].iloc[0]),
        "plot_radius_buffer": float(plot_summary["Plot Radius Buffer"].iloc[0]),
        "plot_area": float(plot_summary["Plot Area"].iloc[0]),
        "num_trees": num_trees,
        "stems_per_ha": int(plot_summary["Stems/ha"].iloc[0]) if num_trees > 0 else 0,
        "show_stem_map": show_stem_map,
        "show_histograms": show_histograms,
        "logo_path": None,  # relative path if logo is copied to output
    }

    if num_trees > 0:
        context.update({
            "mean_dbh": float(plot_summary["Mean DBH"].iloc[0]),
            "median_dbh": float(plot_summary["Median DBH"].iloc[0]),
            "min_dbh": float(plot_summary["Min DBH"].iloc[0]),
            "max_dbh": float(plot_summary["Max DBH"].iloc[0]),
            "mean_height": float(plot_summary["Mean Height"].iloc[0]),
            "total_volume": float(plot_summary.get("Total Volume 1", pd.Series([0])).iloc[0]),
            "canopy_cover": float(plot_summary.get("Canopy Cover Fraction", pd.Series([0])).iloc[0]),
            "trees": tree_data.to_dict("records"),
        })
    else:
        context.update({
            "mean_dbh": 0, "median_dbh": 0, "min_dbh": 0, "max_dbh": 0,
            "mean_height": 0, "total_volume": 0, "canopy_cover": 0,
            "trees": [],
        })

    # Stand metrics (computed from tree_data)
    if num_trees > 0 and "DBH" in tree_data.columns:
        dbh_vals = tree_data["DBH"].values
        plot_area_ha = context["plot_area"]  # already in hectares

        # Basal Area (m2/ha)
        ba_per_tree = np.pi * (dbh_vals / 2) ** 2  # m2
        total_ba = float(np.sum(ba_per_tree))
        basal_area_ha = total_ba / plot_area_ha if plot_area_ha > 0 else 0

        # Quadratic Mean Diameter
        qmd = float(np.sqrt(np.mean(dbh_vals ** 2)))

        # Lorey's Height = sum(Height_i * BA_i) / sum(BA_i)
        if "Height" in tree_data.columns:
            loreys_height = float(np.sum(tree_data["Height"].values * ba_per_tree) / total_ba) if total_ba > 0 else 0
        else:
            loreys_height = 0

        # Stand Density Index = stems_per_ha * (QMD_cm / 25.4)^1.605
        stems_per_ha = context["stems_per_ha"]
        qmd_cm = qmd * 100  # convert m to cm
        sdi = stems_per_ha * (qmd_cm / 25.4) ** 1.605 if qmd_cm > 0 else 0

        context.update({
            "basal_area_ha": basal_area_ha,
            "qmd": qmd,
            "loreys_height": loreys_height,
            "sdi": sdi,
        })
    else:
        context.update({
            "basal_area_ha": 0, "qmd": 0, "loreys_height": 0, "sdi": 0,
        })

    # Taper and crown map flags (with fallback: if image exists, include it)
    if not has_taper and (output_dir / "Taper_Profiles.png").exists():
        has_taper = True
    if not has_crown_map and (output_dir / "Crown_Projection_Map.png").exists():
        has_crown_map = True
    context["has_taper"] = has_taper
    context["has_crown_map"] = has_crown_map

    # Point cloud statistics
    context["num_points_original"] = int(plot_summary.get("Num Points Original PC", pd.Series([0])).iloc[0])
    context["num_points_trimmed"] = int(plot_summary.get("Num Points Trimmed PC", pd.Series([0])).iloc[0])
    context["num_points_subsampled"] = int(plot_summary.get("Num Points Subsampled PC", pd.Series([0])).iloc[0])
    context["num_terrain_points"] = int(plot_summary.get("Num Terrain Points", pd.Series([0])).iloc[0])
    context["num_vegetation_points"] = int(plot_summary.get("Num Vegetation Points", pd.Series([0])).iloc[0])
    context["num_cwd_points"] = int(plot_summary.get("Num CWD Points", pd.Series([0])).iloc[0])
    context["num_stem_points"] = int(plot_summary.get("Num Stem Points", pd.Series([0])).iloc[0])

    # Coverage & terrain
    context["understory_veg_coverage"] = float(plot_summary.get("Understory Veg Coverage Fraction", pd.Series([0])).iloc[0])
    context["cwd_coverage"] = float(plot_summary.get("CWD Coverage Fraction", pd.Series([0])).iloc[0])
    context["avg_gradient"] = float(plot_summary.get("Avg Gradient", pd.Series([0])).iloc[0])

    # Timing
    context["preprocessing_time"] = float(plot_summary.get("Preprocessing Time (s)", pd.Series([0])).iloc[0])
    context["segmentation_time"] = float(plot_summary.get("Semantic Segmentation Time (s)", pd.Series([0])).iloc[0])
    context["postprocessing_time"] = float(plot_summary.get("Post processing time (s)", pd.Series([0])).iloc[0])
    context["measurement_time"] = float(plot_summary.get("Measurement Time (s)", pd.Series([0])).iloc[0])
    context["total_time"] = float(plot_summary.get("Total Run Time (s)", pd.Series([0])).iloc[0])

    # Copy logo to output
    logo_src = Path(__file__).parent.parent / "resources" / "icons" / "understory-logo.png"
    if logo_src.exists():
        import shutil
        logo_dest = output_dir / "understory-logo.png"
        shutil.copy2(logo_src, logo_dest)
        context["logo_path"] = "understory-logo.png"

    # Copy field photos if provided
    photo_filenames = []
    if photos:
        import shutil as _shutil
        for photo_path in photos:
            p = Path(photo_path)
            if p.exists():
                dest = output_dir / p.name
                _shutil.copy2(p, dest)
                photo_filenames.append(p.name)
    context["photos"] = photo_filenames

    # Render template
    template_dir = str(Path(__file__).parent.parent / "resources")
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template("report_template.html")
    html = template.render(**context)

    report_path = output_dir / "Plot_Report.html"
    report_path.write_text(html, encoding="utf-8")

    return str(report_path)


def export_pdf(html_path: str, pdf_path: str) -> str:
    """Export an HTML report to PDF using QWebEngineView.

    Args:
        html_path: Path to the HTML report file.
        pdf_path: Desired output PDF path.

    Returns:
        Path to the generated PDF file.
    """
    from PySide6.QtCore import QUrl, QMarginsF, QEventLoop
    from PySide6.QtGui import QPageLayout, QPageSize
    from PySide6.QtWebEngineWidgets import QWebEngineView
    from PySide6.QtWidgets import QApplication

    # Ensure a QApplication exists (needed for QWebEngineView)
    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    view = QWebEngineView()
    loop = QEventLoop()

    # Wait for page to finish loading
    def on_load_finished(ok):
        if not ok:
            loop.quit()
            return
        # Set up A4 portrait with 10mm margins
        margins = QMarginsF(10, 10, 10, 10)
        page_layout = QPageLayout(QPageSize(QPageSize.A4), QPageLayout.Portrait, margins)

        def on_pdf_done(path):
            loop.quit()

        view.printToPdf(str(pdf_path), page_layout)
        # printToPdf with path arg is synchronous-ish but we use pdfPrintingFinished signal
        view.page().pdfPrintingFinished.connect(on_pdf_done)

    view.loadFinished.connect(on_load_finished)
    view.load(QUrl.fromLocalFile(str(Path(html_path).resolve())))
    loop.exec()

    return str(pdf_path)


def _generate_branded_plots(output_dir: Path, tree_data: pd.DataFrame) -> None:
    """Generate histogram plots with Understory brand colors."""
    if tree_data.empty:
        return

    plot_configs = [
        ("DBH", "Diameter at Breast Height Distribution", "DBH (m)", 0.1),
        ("Height", "Tree Height Distribution", "Height (m)", 1),
        ("Volume_1", "Tree Volume 1 Distribution", "Volume 1 (m\u00b3)", 0.1),
        ("Volume_2", "Tree Volume 2 Distribution", "Volume 2 (m\u00b3)", 0.1),
    ]

    for col, title, xlabel, bin_width in plot_configs:
        if col not in tree_data.columns:
            continue
        data = np.array(tree_data[col])
        if data.shape[0] == 0:
            continue

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.set_title(title, fontsize=14, color=BRAND_COLORS["dark_forest"], fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("Count", fontsize=11)

        bins = np.arange(0, np.ceil(np.max(data) / bin_width) * bin_width + bin_width, bin_width)
        ax.hist(
            data, bins=bins,
            linewidth=0.5,
            edgecolor=BRAND_COLORS["dark_forest"],
            facecolor=BRAND_COLORS["medium_green"],
            alpha=0.85,
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(BRAND_COLORS["light_mint"])
        ax.spines["bottom"].set_color(BRAND_COLORS["light_mint"])

        fig.savefig(
            output_dir / f"{title}.png",
            dpi=200, bbox_inches="tight", pad_inches=0.1,
            facecolor="white",
        )
        plt.close(fig)


def _generate_branded_stem_map(
    output_dir: Path, plot_summary: pd.DataFrame, tree_data: pd.DataFrame
) -> None:
    """Generate stem map with Understory brand colors."""
    # Add scripts to path for load_file
    scripts_dir = str(Path(__file__).parent.parent.parent / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    try:
        from tools import load_file
    except ImportError:
        return

    dtm_path = output_dir / "DTM.las"
    if not dtm_path.exists():
        return

    DTM, _ = load_file(str(dtm_path))
    if DTM.shape[0] == 0:
        return

    plot_centre_x = float(plot_summary["Plot Centre X"].iloc[0])
    plot_centre_y = float(plot_summary["Plot Centre Y"].iloc[0])
    plot_centre = np.array([plot_centre_x, plot_centre_y])
    plot_radius = float(plot_summary["Plot Radius"].iloc[0])

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(
        "Stem Map",
        fontsize=16, color=BRAND_COLORS["dark_forest"], fontweight="bold", pad=12,
    )
    ax.set_xlabel(f"Easting + {plot_centre_x:.2f} (m)", fontsize=11)
    ax.set_ylabel(f"Northing + {plot_centre_y:.2f} (m)", fontsize=11)
    ax.set_aspect("equal")

    # Contours
    import warnings
    zmin = np.floor(np.min(DTM[:, 2]))
    zmax = np.ceil(np.max(DTM[:, 2]))
    zrange = int(np.ceil((zmax - zmin))) + 1
    levels = np.linspace(zmin, zmax, zrange)

    if plot_radius > 0:
        ax.set_facecolor("#f5f5f0")
        circle_face = plt.Circle(
            xy=(0, 0), radius=plot_radius,
            facecolor="white", edgecolor=None, zorder=1,
        )
        ax.add_patch(circle_face)
        DTM_plot = DTM[np.linalg.norm(DTM[:, :2] - plot_centre, axis=1) < plot_radius]
    else:
        DTM_plot = DTM

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if DTM_plot.shape[0] > 3:
            ax.tricontour(
                DTM_plot[:, 0] - plot_centre[0],
                DTM_plot[:, 1] - plot_centre[1],
                DTM_plot[:, 2],
                levels=levels,
                colors=BRAND_COLORS["medium_forest"],
                linewidths=1.5,
                zorder=3,
            )

    if plot_radius > 0:
        circle_outline = plt.Circle(
            xy=(0, 0), radius=plot_radius,
            fill=False, edgecolor=BRAND_COLORS["dark_forest"],
            linewidth=2, zorder=8,
        )
        ax.add_patch(circle_outline)

    # Tree positions
    if not tree_data.empty and "x_tree_base" in tree_data.columns:
        x_base = np.array(tree_data["x_tree_base"])
        y_base = np.array(tree_data["y_tree_base"])
        tree_ids = np.array(tree_data["TreeId"])

        ax.scatter(
            x_base - plot_centre[0],
            y_base - plot_centre[1],
            marker="o", s=60,
            facecolor=BRAND_COLORS["medium_green"],
            edgecolor=BRAND_COLORS["dark_forest"],
            linewidth=1.5, zorder=9,
        )

        # Plot centre marker
        ax.scatter([0], [0], marker="+", s=80, c=BRAND_COLORS["dark_forest"], linewidth=2, zorder=10)

        # Tree labels
        for i in range(len(x_base)):
            ax.annotate(
                str(int(tree_ids[i])),
                (x_base[i] - plot_centre[0], y_base[i] - plot_centre[1]),
                textcoords="offset points", xytext=(5, 5),
                fontsize=7, color=BRAND_COLORS["dark_forest"],
                zorder=11,
            )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(
        output_dir / "Stem_Map.png",
        dpi=200, bbox_inches="tight", pad_inches=0.1,
        facecolor="white",
    )
    plt.close(fig)


def _generate_taper_chart(output_dir: Path, tree_data: pd.DataFrame) -> bool:
    """Generate taper profile chart if taper_data.csv exists. Returns True if generated."""
    taper_path = output_dir / "taper_data.csv"
    if not taper_path.exists():
        return False

    try:
        taper_df = pd.read_csv(taper_path)
    except Exception:
        return False

    if taper_df.empty or "TreeId" not in taper_df.columns:
        return False

    # Columns are: PlotId, TreeId, x_base, y_base, z_base, then height
    # increments as numeric strings (0.0, 0.2, 0.4, ...)
    skip_cols = {"PlotId", "TreeId", "x_base", "y_base", "z_base"}
    height_cols = []
    for c in taper_df.columns:
        if c in skip_cols:
            continue
        try:
            float(c)
            height_cols.append(c)
        except ValueError:
            continue
    if not height_cols:
        return False

    heights = [float(c) for c in height_cols]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_title("Taper Profiles", fontsize=14, color=BRAND_COLORS["dark_forest"], fontweight="bold")
    ax.set_xlabel("Diameter (m)", fontsize=11)
    ax.set_ylabel("Height (m)", fontsize=11)

    # Color cycle from brand palette
    colors = [BRAND_COLORS["dark_forest"], BRAND_COLORS["medium_forest"],
              BRAND_COLORS["medium_green"], BRAND_COLORS["light_mint"],
              "#e67e22", "#3498db", "#9b59b6", "#e74c3c"]

    for i, (_, row) in enumerate(taper_df.iterrows()):
        tree_id = int(row["TreeId"])
        diameters = [float(row[c]) if pd.notna(row[c]) and float(row[c]) > 0 else np.nan for c in height_cols]
        color = colors[i % len(colors)]
        ax.plot(diameters, heights, marker=".", markersize=3, linewidth=1.5,
                color=color, label=f"Tree {tree_id}", alpha=0.8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BRAND_COLORS["light_mint"])
    ax.spines["bottom"].set_color(BRAND_COLORS["light_mint"])

    if len(taper_df) <= 15:
        ax.legend(fontsize=7, loc="upper right", framealpha=0.9)

    fig.savefig(output_dir / "Taper_Profiles.png", dpi=200, bbox_inches="tight", pad_inches=0.1, facecolor="white")
    plt.close(fig)
    return True


def _generate_crown_map(output_dir: Path, plot_summary: pd.DataFrame, tree_data: pd.DataFrame) -> bool:
    """Generate crown projection map. Returns True if generated."""
    if tree_data.empty:
        return False

    required = ["Crown_mean_x", "Crown_mean_y"]
    if not all(c in tree_data.columns for c in required):
        return False

    plot_centre_x = float(plot_summary["Plot Centre X"].iloc[0])
    plot_centre_y = float(plot_summary["Plot Centre Y"].iloc[0])
    plot_radius = float(plot_summary["Plot Radius"].iloc[0])

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title("Crown Projection Map", fontsize=16, color=BRAND_COLORS["dark_forest"], fontweight="bold", pad=12)
    ax.set_xlabel(f"Easting + {plot_centre_x:.2f} (m)", fontsize=11)
    ax.set_ylabel(f"Northing + {plot_centre_y:.2f} (m)", fontsize=11)
    ax.set_aspect("equal")

    if plot_radius > 0:
        ax.set_facecolor("#f5f5f0")
        circle_face = plt.Circle(xy=(0, 0), radius=plot_radius, facecolor="white", edgecolor=None, zorder=1)
        ax.add_patch(circle_face)
        circle_outline = plt.Circle(xy=(0, 0), radius=plot_radius, fill=False, edgecolor=BRAND_COLORS["dark_forest"], linewidth=2, zorder=8)
        ax.add_patch(circle_outline)

    # Color cycle
    colors = [BRAND_COLORS["medium_green"], BRAND_COLORS["medium_forest"],
              "#3498db", "#e67e22", "#9b59b6", "#e74c3c", "#1abc9c", "#f1c40f"]

    for i, (_, row) in enumerate(tree_data.iterrows()):
        cx = float(row["Crown_mean_x"]) - plot_centre_x
        cy = float(row["Crown_mean_y"]) - plot_centre_y

        # Estimate crown radius from Crown_area if available, else from DBH
        if "Crown_area" in tree_data.columns and pd.notna(row.get("Crown_area")) and float(row["Crown_area"]) > 0:
            crown_r = float(np.sqrt(float(row["Crown_area"]) / np.pi))
        elif "DBH" in tree_data.columns and pd.notna(row.get("DBH")):
            crown_r = float(row["DBH"]) * 10  # rough estimate
        else:
            crown_r = 1.0

        color = colors[i % len(colors)]
        crown_circle = plt.Circle(xy=(cx, cy), radius=crown_r, facecolor=color, edgecolor=BRAND_COLORS["dark_forest"], linewidth=0.8, alpha=0.3, zorder=5)
        ax.add_patch(crown_circle)

    # Plot stem positions
    if "x_tree_base" in tree_data.columns:
        x_base = tree_data["x_tree_base"].values - plot_centre_x
        y_base = tree_data["y_tree_base"].values - plot_centre_y
        tree_ids = tree_data["TreeId"].values
        ax.scatter(x_base, y_base, marker="o", s=30, facecolor=BRAND_COLORS["dark_forest"], edgecolor="white", linewidth=0.8, zorder=7)
        for j in range(len(x_base)):
            ax.annotate(str(int(tree_ids[j])), (x_base[j], y_base[j]), textcoords="offset points", xytext=(4, 4), fontsize=7, color=BRAND_COLORS["dark_forest"], zorder=9)

    ax.scatter([0], [0], marker="+", s=80, c=BRAND_COLORS["dark_forest"], linewidth=2, zorder=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(output_dir / "Crown_Projection_Map.png", dpi=200, bbox_inches="tight", pad_inches=0.1, facecolor="white")
    plt.close(fig)
    return True
