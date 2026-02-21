"""Export tree data to GIS-compatible formats.

Supports GeoJSON (pure Python, no extra dependencies), Shapefile (requires
geopandas), and CSV with coordinate columns.

Feature 16: Export to GIS
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


def export_geojson(
    tree_data: pd.DataFrame,
    output_path: str,
    crs: str = "EPSG:0",
) -> str:
    """Export tree data as a GeoJSON FeatureCollection with one Point per tree.

    Pure Python implementation -- no extra dependencies beyond pandas and the
    standard library.

    Args:
        tree_data: DataFrame with at least ``x_tree_base`` and ``y_tree_base``
            columns.  All other columns are written as Feature properties.
        output_path: Destination file path (should end in ``.geojson``).
        crs: Coordinate reference system identifier stored in the
            FeatureCollection metadata (e.g. ``"EPSG:32610"``).

    Returns:
        The *output_path* string, for convenience when chaining calls.
    """
    features: list[dict[str, Any]] = []

    for _, row in tree_data.iterrows():
        x = row.get("x_tree_base")
        y = row.get("y_tree_base")

        # Skip rows without valid coordinates
        if x is None or y is None:
            continue
        try:
            x = float(x)
            y = float(y)
        except (TypeError, ValueError):
            continue
        if math.isnan(x) or math.isnan(y):
            continue

        # Build properties from all non-coordinate columns
        properties: dict[str, Any] = {}
        for col in tree_data.columns:
            if col in ("x_tree_base", "y_tree_base"):
                continue
            val = row[col]
            # Convert numpy/pandas types to native Python for JSON
            if isinstance(val, float) and math.isnan(val):
                properties[col] = None
            elif hasattr(val, "item"):
                # numpy scalar -> Python scalar
                properties[col] = val.item()
            else:
                properties[col] = val

        feature: dict[str, Any] = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [x, y],
            },
            "properties": properties,
        }
        features.append(feature)

    feature_collection: dict[str, Any] = {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {
                "name": crs,
            },
        },
        "features": features,
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(feature_collection, f, indent=2)

    return str(out)


def export_shapefile(
    tree_data: pd.DataFrame,
    output_path: str,
    crs: str = "EPSG:0",
) -> str:
    """Export tree data as a Shapefile using *geopandas*.

    If geopandas is not installed the function raises ``ImportError`` with a
    helpful message.

    Args:
        tree_data: DataFrame with at least ``x_tree_base`` and ``y_tree_base``
            columns.
        output_path: Destination file path (should end in ``.shp``).
        crs: Coordinate reference system identifier (e.g. ``"EPSG:32610"``).

    Returns:
        The *output_path* string.

    Raises:
        ImportError: If geopandas is not available.
    """
    try:
        import geopandas as gpd
        from shapely.geometry import Point
    except ImportError:
        raise ImportError(
            "geopandas (and shapely) are required for Shapefile export. "
            "Install them with:  pip install geopandas"
        )

    df = tree_data.copy()
    geometry = [
        Point(float(row["x_tree_base"]), float(row["y_tree_base"]))
        for _, row in df.iterrows()
    ]
    gdf = gpd.GeoDataFrame(df, geometry=geometry)

    # Drop the raw coordinate columns -- they are now encoded in geometry
    for col in ("x_tree_base", "y_tree_base"):
        if col in gdf.columns:
            gdf = gdf.drop(columns=[col])

    if crs and crs != "EPSG:0":
        gdf = gdf.set_crs(crs)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(str(out))

    return str(out)


def export_csv_with_coords(
    tree_data: pd.DataFrame,
    output_path: str,
) -> str:
    """Export tree data as a CSV with coordinate columns first.

    Reorders the columns so that ``x_tree_base`` and ``y_tree_base`` appear as
    the first two data columns (after ``TreeId`` if present), making the file
    easy to import into any GIS tool.

    Args:
        tree_data: DataFrame with at least ``x_tree_base`` and ``y_tree_base``
            columns.
        output_path: Destination file path (should end in ``.csv``).

    Returns:
        The *output_path* string.
    """
    df = tree_data.copy()

    # Build a preferred column order: TreeId first, then coordinates, then rest
    leading: list[str] = []
    if "TreeId" in df.columns:
        leading.append("TreeId")
    for coord_col in ("x_tree_base", "y_tree_base"):
        if coord_col in df.columns:
            leading.append(coord_col)

    remaining = [c for c in df.columns if c not in leading]
    df = df[leading + remaining]

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(out), index=False)

    return str(out)
