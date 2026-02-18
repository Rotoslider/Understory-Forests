"""Growth Tracking Dashboard — Feature 14.

Displays per-tree growth metrics (DBH, Height) across multiple pipeline
runs, enabling longitudinal monitoring of forest plot dynamics.

Data sources:
    1. Primary: scan all ``run_*/output/tree_data.csv`` files in the project's
       runs directory and collate per-tree measurements across scans.
    2. Optional: if a TreeRegistry JSON exists, use its ``get_growth_data()``
       for richer scan history (dates, positions).

UI layout:
    +------------------------------------------------------------------+
    | Growth Tracking Dashboard                               [X Close] |
    +------------------------------------------------------------------+
    | Tree selector (multi-select list)  |  DBH over time chart         |
    |                                    |  Height over time chart       |
    +------------------------------------+------------------------------+
    | Scan History Table (all runs, dates, summary stats)               |
    +------------------------------------------------------------------+
    | [Export Growth CSV]                                                |
    +------------------------------------------------------------------+
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QGroupBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QFileDialog,
    QMessageBox,
    QAbstractItemView,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

# Brand colors
DARK_FOREST = "#1a4a3a"
MEDIUM_FOREST = "#2d7a5e"
MEDIUM_GREEN = "#4a9e7e"
LIGHT_MINT = "#a8d8c0"
PALE_MINT = "#f0f7f4"

# Extended palette for multi-line charts (up to 12 distinct trees)
_CHART_COLORS = [
    MEDIUM_FOREST,
    MEDIUM_GREEN,
    "#e67e22",  # orange
    "#3498db",  # blue
    "#9b59b6",  # purple
    "#e74c3c",  # red
    "#1abc9c",  # teal
    "#f1c40f",  # yellow
    "#2c3e50",  # dark blue-grey
    "#d35400",  # dark orange
    "#8e44ad",  # dark purple
    "#16a085",  # dark teal
]


def _parse_run_timestamp(run_name: str) -> Optional[datetime]:
    """Extract a datetime from a run folder name like ``run_2024-01-15_10-30-45``.

    Returns None if the name cannot be parsed.
    """
    match = re.search(r"run_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", run_name)
    if match:
        try:
            return datetime.strptime(match.group(1), "%Y-%m-%d_%H-%M-%S")
        except ValueError:
            pass
    return None


def _format_run_label(run_name: str) -> str:
    """Human-readable label for a run folder name."""
    ts = _parse_run_timestamp(run_name)
    if ts:
        return ts.strftime("%Y-%m-%d %H:%M")
    return run_name


class GrowthChartCanvas(FigureCanvasQTAgg):
    """Dual-axis matplotlib canvas for DBH and Height over time."""

    def __init__(self, parent: Optional[QWidget] = None):
        self._fig = Figure(figsize=(7, 8), dpi=100)
        self._fig.set_facecolor(PALE_MINT)
        super().__init__(self._fig)
        self.setParent(parent)

        self._ax_dbh = self._fig.add_subplot(2, 1, 1)
        self._ax_height = self._fig.add_subplot(2, 1, 2)
        self._style_axes()
        self._fig.tight_layout(pad=2.5)

    def _style_axes(self) -> None:
        """Apply brand styling to both axes."""
        for ax, ylabel, title in [
            (self._ax_dbh, "DBH (m)", "DBH Over Time"),
            (self._ax_height, "Height (m)", "Height Over Time"),
        ]:
            ax.set_facecolor("#ffffff")
            ax.set_title(title, fontsize=12, fontweight="bold", color=DARK_FOREST)
            ax.set_ylabel(ylabel, fontsize=10, color=DARK_FOREST)
            ax.set_xlabel("Scan", fontsize=10, color=DARK_FOREST)
            ax.tick_params(colors=DARK_FOREST, labelsize=9)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color(LIGHT_MINT)
            ax.spines["bottom"].set_color(LIGHT_MINT)
            ax.grid(True, alpha=0.3, color=LIGHT_MINT)

    def update_charts(
        self,
        growth_df: pd.DataFrame,
        tree_ids: list[int],
        run_labels: list[str],
    ) -> None:
        """Redraw both charts for the selected tree IDs.

        Args:
            growth_df: DataFrame with columns ``run_label``, ``TreeId``,
                ``DBH``, ``Height`` (one row per tree per run).
            tree_ids: Which trees to plot.
            run_labels: Ordered list of run labels (x-axis tick labels).
        """
        self._ax_dbh.clear()
        self._ax_height.clear()
        self._style_axes()

        if growth_df.empty or not tree_ids:
            self._ax_dbh.text(
                0.5, 0.5, "Select one or more trees",
                ha="center", va="center", fontsize=11, color=MEDIUM_FOREST,
                transform=self._ax_dbh.transAxes,
            )
            self._ax_height.text(
                0.5, 0.5, "Select one or more trees",
                ha="center", va="center", fontsize=11, color=MEDIUM_FOREST,
                transform=self._ax_height.transAxes,
            )
            self.draw_idle()
            return

        for i, tid in enumerate(tree_ids):
            color = _CHART_COLORS[i % len(_CHART_COLORS)]
            tree_rows = growth_df[growth_df["TreeId"] == tid].copy()
            if tree_rows.empty:
                continue

            # Map run_label to x position (index in the ordered run_labels list)
            label_to_x = {label: idx for idx, label in enumerate(run_labels)}
            tree_rows = tree_rows.copy()
            tree_rows["x"] = tree_rows["run_label"].map(label_to_x)
            tree_rows = tree_rows.dropna(subset=["x"]).sort_values("x")

            x_vals = tree_rows["x"].values

            # DBH chart
            dbh_vals = pd.to_numeric(tree_rows["DBH"], errors="coerce")
            valid_dbh = dbh_vals.notna()
            if valid_dbh.any():
                self._ax_dbh.plot(
                    x_vals[valid_dbh], dbh_vals[valid_dbh],
                    marker="o", markersize=5, linewidth=2,
                    color=color, label=f"Tree {tid}",
                )

            # Height chart
            height_vals = pd.to_numeric(tree_rows["Height"], errors="coerce")
            valid_height = height_vals.notna()
            if valid_height.any():
                self._ax_height.plot(
                    x_vals[valid_height], height_vals[valid_height],
                    marker="s", markersize=5, linewidth=2,
                    color=color, label=f"Tree {tid}",
                )

        # Configure x-axis ticks
        for ax in (self._ax_dbh, self._ax_height):
            if run_labels:
                ax.set_xticks(range(len(run_labels)))
                ax.set_xticklabels(run_labels, rotation=30, ha="right", fontsize=8)
            legend = ax.legend(
                fontsize=8, loc="upper left", framealpha=0.9,
                edgecolor=LIGHT_MINT,
            )
            if legend:
                legend.get_frame().set_facecolor(PALE_MINT)

        self._fig.tight_layout(pad=2.5)
        self.draw_idle()


class GrowthPanel(QDialog):
    """Growth Tracking Dashboard dialog.

    Scans all ``run_*/output/tree_data.csv`` files in a project's runs
    directory and presents per-tree longitudinal growth charts and a
    scan history summary table.
    """

    def __init__(
        self,
        project_dir: str | Path,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._project_dir = Path(project_dir).resolve()
        self.setWindowTitle("Growth Tracking Dashboard")
        self.setMinimumSize(1000, 700)
        self.resize(1200, 800)

        # Data containers
        self._growth_df = pd.DataFrame()   # all tree rows across runs
        self._run_labels: list[str] = []   # ordered run labels (oldest first)
        self._all_tree_ids: list[int] = [] # unique tree IDs across all runs

        self._setup_ui()
        self._load_data()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(8)

        # Title
        title = QLabel("Growth Tracking Dashboard")
        title.setStyleSheet(
            f"font-size: 20px; font-weight: bold; color: {DARK_FOREST}; "
            f"padding: 4px 0;"
        )
        root_layout.addWidget(title)

        project_label = QLabel(f"Project: {self._project_dir.name}")
        project_label.setStyleSheet(f"color: {MEDIUM_FOREST}; font-size: 12px;")
        root_layout.addWidget(project_label)

        # Main splitter: tree selector | charts
        main_splitter = QSplitter(Qt.Horizontal)

        # --- Left: tree selector ---
        selector_group = QGroupBox("Select Trees")
        selector_layout = QVBoxLayout(selector_group)

        selector_info = QLabel("Select one or more trees to chart:")
        selector_info.setWordWrap(True)
        selector_info.setStyleSheet("font-size: 11px;")
        selector_layout.addWidget(selector_info)

        self._tree_list = QListWidget()
        self._tree_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._tree_list.setAlternatingRowColors(True)
        self._tree_list.itemSelectionChanged.connect(self._on_selection_changed)
        selector_layout.addWidget(self._tree_list)

        # Select all / none helpers
        btn_row = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self._select_all_trees)
        btn_row.addWidget(select_all_btn)
        clear_btn = QPushButton("Clear Selection")
        clear_btn.clicked.connect(self._tree_list.clearSelection)
        btn_row.addWidget(clear_btn)
        selector_layout.addLayout(btn_row)

        selector_group.setMinimumWidth(200)
        selector_group.setMaximumWidth(300)
        main_splitter.addWidget(selector_group)

        # --- Right: charts ---
        self._chart_canvas = GrowthChartCanvas()
        main_splitter.addWidget(self._chart_canvas)

        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)
        main_splitter.setSizes([220, 780])
        root_layout.addWidget(main_splitter, stretch=3)

        # --- Bottom: scan history table ---
        history_group = QGroupBox("Scan History")
        history_layout = QVBoxLayout(history_group)

        self._history_table = QTableWidget()
        self._history_table.setAlternatingRowColors(True)
        self._history_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._history_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._history_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        self._history_table.horizontalHeader().setStretchLastSection(True)
        self._history_table.setMinimumHeight(150)
        history_layout.addWidget(self._history_table)

        root_layout.addWidget(history_group, stretch=1)

        # --- Export button ---
        export_row = QHBoxLayout()
        export_row.addStretch()
        self._export_btn = QPushButton("Export Growth Data as CSV")
        self._export_btn.setMinimumWidth(220)
        self._export_btn.clicked.connect(self._export_csv)
        export_row.addWidget(self._export_btn)
        export_row.addStretch()
        root_layout.addLayout(export_row)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_data(self) -> None:
        """Scan all runs and build the growth DataFrame."""
        from understory.core.paths import ProjectPaths

        project_paths = ProjectPaths(self._project_dir)
        runs = project_paths.list_runs()  # newest first

        if not runs:
            self._show_no_data("No pipeline runs found in this project.")
            return

        # Process oldest-first for chronological ordering
        runs_oldest_first = list(reversed(runs))

        all_rows: list[pd.DataFrame] = []
        run_labels: list[str] = []
        scan_summary_rows: list[dict] = []

        for run_dir in runs_oldest_first:
            csv_path = ProjectPaths.run_output_dir(run_dir) / "tree_data.csv"
            if not csv_path.exists():
                continue

            try:
                df = pd.read_csv(csv_path)
            except Exception:
                continue

            if df.empty:
                continue

            # Ensure required columns exist
            if "TreeId" not in df.columns:
                continue

            label = _format_run_label(run_dir.name)
            run_labels.append(label)

            df = df.copy()
            df["run_label"] = label
            df["run_dir"] = run_dir.name
            all_rows.append(df)

            # Build summary row for the scan history table
            ts = _parse_run_timestamp(run_dir.name)
            summary: dict = {
                "Run": run_dir.name,
                "Date": ts.strftime("%Y-%m-%d %H:%M:%S") if ts else "Unknown",
                "Trees": int(df["TreeId"].nunique()),
            }
            if "DBH" in df.columns:
                dbh_vals = pd.to_numeric(df["DBH"], errors="coerce").dropna()
                if not dbh_vals.empty:
                    summary["Mean DBH (m)"] = round(float(dbh_vals.mean()), 3)
                    summary["Min DBH (m)"] = round(float(dbh_vals.min()), 3)
                    summary["Max DBH (m)"] = round(float(dbh_vals.max()), 3)
                else:
                    summary["Mean DBH (m)"] = ""
                    summary["Min DBH (m)"] = ""
                    summary["Max DBH (m)"] = ""
            if "Height" in df.columns:
                h_vals = pd.to_numeric(df["Height"], errors="coerce").dropna()
                if not h_vals.empty:
                    summary["Mean Height (m)"] = round(float(h_vals.mean()), 2)
                    summary["Max Height (m)"] = round(float(h_vals.max()), 2)
                else:
                    summary["Mean Height (m)"] = ""
                    summary["Max Height (m)"] = ""
            scan_summary_rows.append(summary)

        if not all_rows:
            self._show_no_data(
                "No tree_data.csv files found in any run.\n"
                "Run the pipeline with measurement enabled first."
            )
            return

        self._growth_df = pd.concat(all_rows, ignore_index=True)
        self._run_labels = run_labels

        # Collect unique tree IDs, sorted numerically
        self._all_tree_ids = sorted(
            self._growth_df["TreeId"]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )

        self._populate_tree_list()
        self._populate_scan_history(scan_summary_rows)

        # Select all trees by default if there are 10 or fewer
        if len(self._all_tree_ids) <= 10:
            self._select_all_trees()
        else:
            # Pre-select the first tree so the chart is not empty
            if self._tree_list.count() > 0:
                self._tree_list.item(0).setSelected(True)

    def _show_no_data(self, message: str) -> None:
        """Display a message when no data is available."""
        self._tree_list.clear()
        self._tree_list.addItem(QListWidgetItem("(No data)"))
        self._tree_list.setEnabled(False)
        self._export_btn.setEnabled(False)

        self._history_table.setRowCount(0)
        self._history_table.setColumnCount(1)
        self._history_table.setHorizontalHeaderLabels(["Info"])
        self._history_table.setRowCount(1)
        self._history_table.setItem(0, 0, QTableWidgetItem(message))

    def _populate_tree_list(self) -> None:
        """Fill the tree selector list widget."""
        self._tree_list.clear()
        for tid in self._all_tree_ids:
            # Show tree ID and number of runs it appears in
            n_runs = self._growth_df[
                self._growth_df["TreeId"] == tid
            ]["run_label"].nunique()
            item = QListWidgetItem(f"Tree {tid}  ({n_runs} scan{'s' if n_runs != 1 else ''})")
            item.setData(Qt.UserRole, tid)
            self._tree_list.addItem(item)

    def _populate_scan_history(self, summary_rows: list[dict]) -> None:
        """Fill the scan history table."""
        if not summary_rows:
            self._history_table.setRowCount(0)
            return

        columns = list(summary_rows[0].keys())
        self._history_table.setColumnCount(len(columns))
        self._history_table.setHorizontalHeaderLabels(columns)
        self._history_table.setRowCount(len(summary_rows))

        for row_idx, row_data in enumerate(summary_rows):
            for col_idx, col_name in enumerate(columns):
                value = row_data.get(col_name, "")
                item = QTableWidgetItem(str(value))
                item.setTextAlignment(Qt.AlignCenter)
                self._history_table.setItem(row_idx, col_idx, item)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    @Slot()
    def _on_selection_changed(self) -> None:
        """Update charts when tree selection changes."""
        selected_ids = self._get_selected_tree_ids()
        self._chart_canvas.update_charts(
            self._growth_df, selected_ids, self._run_labels
        )

    def _get_selected_tree_ids(self) -> list[int]:
        """Return the list of currently selected tree IDs."""
        ids = []
        for item in self._tree_list.selectedItems():
            tid = item.data(Qt.UserRole)
            if tid is not None:
                ids.append(int(tid))
        return ids

    def _select_all_trees(self) -> None:
        """Select every tree in the list."""
        self._tree_list.selectAll()

    @Slot()
    def _export_csv(self) -> None:
        """Export the full growth DataFrame as a CSV file."""
        if self._growth_df.empty:
            QMessageBox.information(
                self, "No Data", "No growth data to export."
            )
            return

        default_name = f"{self._project_dir.name}_growth_data.csv"
        default_path = str(self._project_dir / default_name)
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Export Growth Data",
            default_path,
            "CSV Files (*.csv);;All Files (*)",
        )
        if not filepath:
            return
        if not filepath.endswith(".csv"):
            filepath += ".csv"

        try:
            # Build a tidy export DataFrame
            export_cols = ["run_label", "run_dir", "TreeId"]
            for col in ["PlotId", "x_tree_base", "y_tree_base", "z_tree_base",
                         "DBH", "CCI_at_BH", "Height", "Volume_1", "Volume_2",
                         "Crown_mean_x", "Crown_mean_y", "Crown_top_x",
                         "Crown_top_y", "Crown_top_z",
                         "mean_understory_height_in_5m_radius"]:
                if col in self._growth_df.columns:
                    export_cols.append(col)

            export_df = self._growth_df[
                [c for c in export_cols if c in self._growth_df.columns]
            ].copy()
            export_df = export_df.sort_values(["TreeId", "run_label"])
            export_df.to_csv(filepath, index=False)

            QMessageBox.information(
                self, "Export Complete",
                f"Growth data exported to:\n{filepath}\n\n"
                f"{len(export_df)} rows, "
                f"{export_df['TreeId'].nunique()} trees, "
                f"{export_df['run_label'].nunique()} scans.",
            )
        except Exception as e:
            QMessageBox.critical(
                self, "Export Error", f"Failed to export growth data:\n{e}"
            )
