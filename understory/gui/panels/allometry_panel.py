"""Allometric equations dialog — define, preview, and apply allometric models.

Provides a QDialog for managing a library of allometric equations, previewing
computed values against the current tree DataFrame, and applying selected
equations as new columns.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from PySide6.QtCore import Qt, QModelIndex, Signal
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QTextEdit,
    QPushButton,
    QListWidget,
    QListWidgetItem,
    QTableView,
    QHeaderView,
    QMessageBox,
    QFileDialog,
    QSplitter,
    QWidget,
    QSizePolicy,
)
from PySide6.QtCore import QAbstractTableModel

from understory.core.allometry import (
    AllometricEquation,
    AllometryRegistry,
    FormulaValidationError,
    compute_allometric_columns,
)


# ---------------------------------------------------------------------------
# Read-only table model for the preview
# ---------------------------------------------------------------------------

class _PreviewTableModel(QAbstractTableModel):
    """Minimal read-only pandas table model for the equation preview."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._df: Optional[pd.DataFrame] = None

    def set_dataframe(self, df: Optional[pd.DataFrame]) -> None:
        self.beginResetModel()
        self._df = df
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._df) if self._df is not None else 0

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._df.columns) if self._df is not None else 0

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):
        if role == Qt.DisplayRole and self._df is not None:
            value = self._df.iloc[index.row(), index.column()]
            if isinstance(value, float):
                if np.isnan(value):
                    return ""
                return f"{value:.4f}"
            return str(value)
        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole):
        if role == Qt.DisplayRole and self._df is not None:
            if orientation == Qt.Horizontal:
                return str(self._df.columns[section])
            return str(section + 1)
        return None


# ---------------------------------------------------------------------------
# Main dialog
# ---------------------------------------------------------------------------

class AllometryPanel(QDialog):
    """Dialog for managing and applying allometric equations.

    Signals
    -------
    equations_applied(pd.DataFrame)
        Emitted when the user clicks *Apply*.  The payload is the tree
        DataFrame with the computed allometric columns appended.
    """

    equations_applied = Signal(object)  # pd.DataFrame

    def __init__(
        self,
        tree_data: Optional[pd.DataFrame] = None,
        registry: Optional[AllometryRegistry] = None,
        parent: Optional[QWidget] = None,
        output_dir: Optional[str] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Allometric Equations")
        self.resize(900, 620)

        # Auto-load tree_data from output directory if not provided directly
        if tree_data is None and output_dir:
            tree_data_path = Path(output_dir) / "tree_data.csv"
            if tree_data_path.exists():
                try:
                    tree_data = pd.read_csv(tree_data_path)
                except Exception:
                    pass

        self._tree_data: pd.DataFrame = tree_data if tree_data is not None else pd.DataFrame()
        self._registry: AllometryRegistry = registry if registry is not None else AllometryRegistry()
        self._editing_index: Optional[int] = None  # None = add mode, int = edit mode

        self._setup_ui()
        self._refresh_list()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)

        splitter = QSplitter(Qt.Horizontal)

        # ---- Left: equation list + buttons ----
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        list_group = QGroupBox("Equations")
        list_layout = QVBoxLayout(list_group)

        self._eq_list = QListWidget()
        self._eq_list.currentRowChanged.connect(self._on_selection_changed)
        list_layout.addWidget(self._eq_list)

        btn_row = QHBoxLayout()
        self._add_btn = QPushButton("Add")
        self._add_btn.clicked.connect(self._on_add)
        btn_row.addWidget(self._add_btn)

        self._edit_btn = QPushButton("Edit")
        self._edit_btn.setEnabled(False)
        self._edit_btn.clicked.connect(self._on_edit)
        btn_row.addWidget(self._edit_btn)

        self._remove_btn = QPushButton("Remove")
        self._remove_btn.setEnabled(False)
        self._remove_btn.clicked.connect(self._on_remove)
        btn_row.addWidget(self._remove_btn)
        list_layout.addLayout(btn_row)

        # Import / Export / Reset
        io_row = QHBoxLayout()
        import_btn = QPushButton("Import YAML...")
        import_btn.clicked.connect(self._import_yaml)
        io_row.addWidget(import_btn)

        export_btn = QPushButton("Export YAML...")
        export_btn.clicked.connect(self._export_yaml)
        io_row.addWidget(export_btn)

        reset_btn = QPushButton("Reset Defaults")
        reset_btn.clicked.connect(self._reset_defaults)
        io_row.addWidget(reset_btn)
        list_layout.addLayout(io_row)

        left_layout.addWidget(list_group)

        # ---- Right: editor + preview ----
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Editor group
        editor_group = QGroupBox("Equation Editor")
        editor_form = QFormLayout(editor_group)

        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("e.g. AGB")
        editor_form.addRow("Name:", self._name_edit)

        self._formula_edit = QLineEdit()
        self._formula_edit.setPlaceholderText("e.g. 0.0673 * (DBH * 100)**2.148 * Height**0.7572")
        self._formula_edit.textChanged.connect(self._on_formula_changed)
        editor_form.addRow("Formula:", self._formula_edit)

        self._vars_label = QLabel("")
        self._vars_label.setWordWrap(True)
        self._vars_label.setStyleSheet("color: #666;")
        editor_form.addRow("Variables:", self._vars_label)

        self._desc_edit = QTextEdit()
        self._desc_edit.setMaximumHeight(60)
        self._desc_edit.setPlaceholderText("Description of the equation...")
        editor_form.addRow("Description:", self._desc_edit)

        # Available columns hint
        if not self._tree_data.empty:
            cols_text = ", ".join(self._tree_data.columns)
        else:
            cols_text = "(no tree data loaded)"
        available_label = QLabel(f"Available columns: {cols_text}")
        available_label.setWordWrap(True)
        available_label.setStyleSheet("color: #888; font-size: 11px;")
        editor_form.addRow(available_label)

        # Save / Cancel buttons for editor
        save_row = QHBoxLayout()
        self._save_btn = QPushButton("Save Equation")
        self._save_btn.clicked.connect(self._save_equation)
        save_row.addWidget(self._save_btn)

        self._cancel_edit_btn = QPushButton("Cancel")
        self._cancel_edit_btn.clicked.connect(self._cancel_edit)
        save_row.addWidget(self._cancel_edit_btn)
        editor_form.addRow(save_row)

        right_layout.addWidget(editor_group)

        # Preview group
        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_group)

        self._preview_model = _PreviewTableModel()
        self._preview_table = QTableView()
        self._preview_table.setModel(self._preview_model)
        self._preview_table.setAlternatingRowColors(True)
        self._preview_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        self._preview_table.horizontalHeader().setStretchLastSection(True)
        self._preview_table.horizontalHeader().setMinimumSectionSize(70)
        self._preview_table.setMinimumHeight(160)
        self._preview_table.setSelectionBehavior(QTableView.SelectRows)
        preview_layout.addWidget(self._preview_table)

        preview_btn_row = QHBoxLayout()
        preview_btn = QPushButton("Refresh Preview")
        preview_btn.clicked.connect(self._refresh_preview)
        preview_btn_row.addWidget(preview_btn)

        export_csv_btn = QPushButton("Export CSV...")
        export_csv_btn.clicked.connect(self._export_preview_csv)
        preview_btn_row.addWidget(export_csv_btn)
        preview_layout.addLayout(preview_btn_row)

        right_layout.addWidget(preview_group)

        # Assemble splitter
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        root.addWidget(splitter, 1)

        # ---- Bottom: Apply / Close ----
        bottom_row = QHBoxLayout()
        bottom_row.addStretch()

        self._apply_btn = QPushButton("Apply to Tree Data")
        self._apply_btn.setEnabled(not self._tree_data.empty)
        self._apply_btn.clicked.connect(self._apply_equations)
        bottom_row.addWidget(self._apply_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        bottom_row.addWidget(close_btn)

        root.addLayout(bottom_row)

    # ------------------------------------------------------------------
    # List management
    # ------------------------------------------------------------------

    def _refresh_list(self) -> None:
        """Repopulate the equation list widget from the registry."""
        self._eq_list.clear()
        for eq in self._registry.equations:
            item = QListWidgetItem(eq.name)
            item.setToolTip(eq.description or eq.formula)
            self._eq_list.addItem(item)

    def _on_selection_changed(self, row: int) -> None:
        has_selection = row >= 0
        self._edit_btn.setEnabled(has_selection)
        self._remove_btn.setEnabled(has_selection)

        if has_selection:
            eq = self._registry.equations[row]
            self._populate_editor(eq)
            self._editing_index = None  # Just viewing, not editing

    def _populate_editor(self, eq: AllometricEquation) -> None:
        """Fill the editor fields from an equation (read-only preview)."""
        self._name_edit.setText(eq.name)
        self._formula_edit.setText(eq.formula)
        self._desc_edit.setPlainText(eq.description)
        self._vars_label.setText(", ".join(eq.variables) if eq.variables else "(auto-detected)")

    def _clear_editor(self) -> None:
        self._name_edit.clear()
        self._formula_edit.clear()
        self._desc_edit.clear()
        self._vars_label.clear()
        self._editing_index = None

    # ------------------------------------------------------------------
    # Add / Edit / Remove
    # ------------------------------------------------------------------

    def _on_add(self) -> None:
        """Enter add mode: clear the editor for a new equation."""
        self._clear_editor()
        self._editing_index = None  # signals "add" mode
        self._name_edit.setFocus()

    def _on_edit(self) -> None:
        """Enter edit mode for the currently selected equation."""
        row = self._eq_list.currentRow()
        if row < 0:
            return
        eq = self._registry.equations[row]
        self._populate_editor(eq)
        self._editing_index = row
        self._name_edit.setFocus()

    def _on_remove(self) -> None:
        """Remove the currently selected equation."""
        row = self._eq_list.currentRow()
        if row < 0:
            return
        eq = self._registry.equations[row]
        reply = QMessageBox.question(
            self,
            "Remove Equation",
            f"Remove equation '{eq.name}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self._registry.remove(eq.name)
            self._refresh_list()
            self._clear_editor()

    def _save_equation(self) -> None:
        """Save the editor contents as a new or updated equation."""
        name = self._name_edit.text().strip()
        formula = self._formula_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Validation", "Equation name is required.")
            return
        if not formula:
            QMessageBox.warning(self, "Validation", "Formula is required.")
            return

        # Detect variables from the formula by parsing AST names
        variables = self._detect_variables(formula)

        eq = AllometricEquation(
            name=name,
            formula=formula,
            variables=variables,
            description=self._desc_edit.toPlainText().strip(),
        )

        # If editing, remove the old entry first (handles name changes)
        if self._editing_index is not None:
            old_eq = self._registry.equations[self._editing_index]
            self._registry.remove(old_eq.name)

        self._registry.add(eq)
        self._refresh_list()

        # Select the newly saved equation
        for i in range(self._eq_list.count()):
            if self._eq_list.item(i).text() == name:
                self._eq_list.setCurrentRow(i)
                break

        self._editing_index = None

    def _cancel_edit(self) -> None:
        """Discard editor changes and return to the previously selected item."""
        self._editing_index = None
        row = self._eq_list.currentRow()
        if row >= 0:
            eq = self._registry.equations[row]
            self._populate_editor(eq)
        else:
            self._clear_editor()

    # ------------------------------------------------------------------
    # Formula analysis
    # ------------------------------------------------------------------

    def _on_formula_changed(self, text: str) -> None:
        """Update the variable preview when the formula text changes."""
        if not text.strip():
            self._vars_label.setText("")
            return
        variables = self._detect_variables(text)
        if variables:
            self._vars_label.setText(", ".join(variables))
        else:
            self._vars_label.setText("(none detected)")

    @staticmethod
    def _detect_variables(formula: str) -> list[str]:
        """Extract DataFrame column references from a formula string.

        Returns the sorted list of ``ast.Name`` identifiers that are NOT
        in the whitelisted numpy/math function set.
        """
        from understory.core.allometry import _WHITELISTED_NAMES

        try:
            tree = __import__("ast").parse(formula, mode="eval")
        except SyntaxError:
            return []

        names: set[str] = set()
        for node in __import__("ast").walk(tree):
            if isinstance(node, __import__("ast").Name):
                if node.id not in _WHITELISTED_NAMES:
                    names.add(node.id)
        return sorted(names)

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def _refresh_preview(self) -> None:
        """Evaluate all registered equations and show results in the table."""
        if self._tree_data.empty:
            QMessageBox.information(
                self, "No Data",
                "No tree data is loaded. Load pipeline results first.",
            )
            return

        try:
            result = compute_allometric_columns(
                self._tree_data, self._registry.equations,
            )
        except Exception as exc:
            QMessageBox.warning(
                self, "Evaluation Error",
                f"Could not compute allometric columns:\n{exc}",
            )
            return

        # Build a compact preview DataFrame: TreeId + key columns + computed
        preview_cols = []
        for col in ["TreeId", "DBH", "Height"]:
            if col in result.columns:
                preview_cols.append(col)
        # Append computed columns (those not in original tree_data)
        for eq in self._registry.equations:
            if eq.name in result.columns:
                preview_cols.append(eq.name)
        # Deduplicate while preserving order
        seen: set[str] = set()
        ordered: list[str] = []
        for c in preview_cols:
            if c not in seen:
                ordered.append(c)
                seen.add(c)

        preview_df = result[ordered].copy() if ordered else result.copy()
        self._preview_model.set_dataframe(preview_df)

    def _export_preview_csv(self) -> None:
        """Export the current preview table to a CSV file."""
        df = self._preview_model._df
        if df is None or df.empty:
            QMessageBox.information(
                self, "No Preview",
                "Refresh the preview first to generate data.",
            )
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Preview CSV", "", "CSV Files (*.csv);;All Files (*)",
        )
        if filepath:
            if not filepath.endswith(".csv"):
                filepath += ".csv"
            df.to_csv(filepath, index=False)
            QMessageBox.information(self, "Exported", f"Preview exported to:\n{filepath}")

    # ------------------------------------------------------------------
    # Apply
    # ------------------------------------------------------------------

    def _apply_equations(self) -> None:
        """Compute all equations and emit the result."""
        if self._tree_data.empty:
            QMessageBox.information(
                self, "No Data",
                "No tree data is loaded.",
            )
            return

        try:
            result = compute_allometric_columns(
                self._tree_data, self._registry.equations,
            )
        except Exception as exc:
            QMessageBox.warning(
                self, "Evaluation Error",
                f"Could not compute allometric columns:\n{exc}",
            )
            return

        computed_names = [
            eq.name for eq in self._registry.equations
            if eq.name in result.columns
        ]
        self.equations_applied.emit(result)
        QMessageBox.information(
            self, "Applied",
            f"Added columns: {', '.join(computed_names)}" if computed_names
            else "No new columns were computed.",
        )

    # ------------------------------------------------------------------
    # YAML import / export / reset
    # ------------------------------------------------------------------

    def _import_yaml(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Equations",
            "", "YAML Files (*.yaml *.yml);;All Files (*)",
        )
        if not path:
            return
        try:
            self._registry.load(path)
            self._refresh_list()
        except Exception as exc:
            QMessageBox.critical(
                self, "Import Error", f"Failed to load equations:\n{exc}",
            )

    def _export_yaml(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Equations",
            "", "YAML Files (*.yaml *.yml);;All Files (*)",
        )
        if not path:
            return
        if not path.endswith((".yaml", ".yml")):
            path += ".yaml"
        try:
            self._registry.save(path)
            QMessageBox.information(
                self, "Exported",
                f"Equations saved to:\n{path}",
            )
        except Exception as exc:
            QMessageBox.critical(
                self, "Export Error", f"Failed to save equations:\n{exc}",
            )

    def _reset_defaults(self) -> None:
        reply = QMessageBox.question(
            self,
            "Reset Defaults",
            "Replace all equations with the built-in defaults?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self._registry.reset_defaults()
            self._refresh_list()
            self._clear_editor()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_tree_data(self, tree_data: pd.DataFrame) -> None:
        """Update the tree DataFrame used for preview and apply."""
        self._tree_data = tree_data
        self._apply_btn.setEnabled(not tree_data.empty)

    def get_registry(self) -> AllometryRegistry:
        """Return the current registry (e.g. for external persistence)."""
        return self._registry
