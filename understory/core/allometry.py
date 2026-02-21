"""Allometric equation registry and evaluation engine.

Provides a safe, extensible framework for computing tree-level allometric
estimates (above-ground biomass, carbon, crown volume, etc.) from measured
tree data.  Equations are stored as human-readable formula strings and
evaluated in a restricted namespace that only exposes numpy math functions
and the columns present in the tree DataFrame.

Persistence uses YAML so that users can share equation libraries across
projects.
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class AllometricEquation:
    """A single allometric equation definition.

    Attributes:
        name: Short display name, also used as the output column name.
        formula: Python-style expression string.  May reference any column
            in the input DataFrame by name, plus whitelisted numpy functions.
        variables: Ordered list of column names the formula depends on.
        description: Human-readable explanation of the equation.
    """

    name: str
    formula: str
    variables: list[str] = field(default_factory=list)
    description: str = ""


# ---------------------------------------------------------------------------
# Safe evaluation helpers
# ---------------------------------------------------------------------------

# Numpy / math functions that are safe to expose inside formulas.
_WHITELISTED_NAMES: dict[str, object] = {
    # Basic math
    "abs": np.abs,
    "sqrt": np.sqrt,
    "cbrt": np.cbrt,
    "exp": np.exp,
    "log": np.log,
    "log2": np.log2,
    "log10": np.log10,
    "power": np.power,
    "pow": np.power,
    # Trigonometric
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "arcsin": np.arcsin,
    "arccos": np.arccos,
    "arctan": np.arctan,
    "arctan2": np.arctan2,
    # Rounding
    "ceil": np.ceil,
    "floor": np.floor,
    "round": np.round,
    # Aggregation helpers (per-element results, but useful in subexpressions)
    "maximum": np.maximum,
    "minimum": np.minimum,
    "clip": np.clip,
    # Constants
    "pi": np.pi,
    "e": np.e,
    "nan": np.nan,
    "inf": np.inf,
}

# AST node types that are considered safe for formula evaluation.
_SAFE_NODE_TYPES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.IfExp,
    ast.Call,
    ast.Constant,
    ast.Name,
    ast.Load,
    ast.Attribute,
    ast.Subscript,
    ast.Index,  # Python 3.8 compat
    ast.Slice,
    ast.Starred,
    ast.Tuple,
    ast.List,
    # Operators
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.Not,
    ast.And,
    ast.Or,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.keyword,
)


class FormulaValidationError(Exception):
    """Raised when a formula contains disallowed syntax or unknown names."""


def _validate_ast(node: ast.AST, allowed_names: set[str]) -> None:
    """Recursively validate that every AST node is whitelisted.

    Raises ``FormulaValidationError`` on the first illegal construct.
    """
    if not isinstance(node, _SAFE_NODE_TYPES):
        raise FormulaValidationError(
            f"Disallowed syntax: {type(node).__name__}"
        )
    if isinstance(node, ast.Name) and node.id not in allowed_names:
        raise FormulaValidationError(
            f"Unknown name '{node.id}'. "
            f"Allowed: {sorted(allowed_names)}"
        )
    for child in ast.iter_child_nodes(node):
        _validate_ast(child, allowed_names)


def _safe_eval(
    formula: str,
    namespace: dict[str, object],
    allowed_names: set[str],
) -> object:
    """Parse, validate, and evaluate *formula* in a restricted namespace.

    Parameters
    ----------
    formula:
        A Python expression string.
    namespace:
        The ``dict`` passed as ``globals`` to ``eval``.
    allowed_names:
        The full set of names that may appear as ``ast.Name`` nodes.

    Returns
    -------
    The result of evaluating the expression (typically a ``pd.Series`` or
    scalar).

    Raises
    ------
    FormulaValidationError
        If the formula contains disallowed syntax or references unknown names.
    """
    tree = ast.parse(formula, mode="eval")
    _validate_ast(tree, allowed_names)
    code = compile(tree, "<allometry>", "eval")
    return eval(code, {"__builtins__": {}}, namespace)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def _default_equations() -> list[AllometricEquation]:
    """Built-in allometric equations shipped with Understory."""
    return [
        AllometricEquation(
            name="AGB",
            formula="0.0673 * (DBH * 100)**2.148 * Height**0.7572",
            variables=["DBH", "Height"],
            description=(
                "Generic above-ground biomass (kg) using the pantropical "
                "model of Chave et al. (2014).  DBH in metres is converted "
                "to centimetres inside the formula."
            ),
        ),
        AllometricEquation(
            name="Carbon",
            formula="AGB * 0.47",
            variables=["AGB"],
            description=(
                "Carbon content (kg) estimated as 47% of above-ground "
                "biomass.  Requires the AGB column to be computed first."
            ),
        ),
        AllometricEquation(
            name="CrownVolume",
            formula="(4 / 3) * pi * (sqrt(Crown_area / pi))**2 * (Height * 0.4)",
            variables=["Crown_area", "Height"],
            description=(
                "Approximate crown volume (m^3) modeled as a prolate "
                "ellipsoid.  Crown radius is derived from Crown_area and "
                "crown depth is assumed to be 40% of total tree height."
            ),
        ),
    ]


class AllometryRegistry:
    """Manages a collection of allometric equations with YAML persistence.

    Parameters
    ----------
    yaml_path:
        Optional path to a YAML file.  If it exists, equations are loaded
        from it on construction.  If *None*, only the built-in defaults
        are available.
    """

    def __init__(self, yaml_path: Optional[str | Path] = None):
        self._path: Optional[Path] = Path(yaml_path) if yaml_path else None
        self._equations: list[AllometricEquation] = []

        if self._path and self._path.exists():
            self.load(self._path)
        else:
            self._equations = _default_equations()

    # -- Accessors ----------------------------------------------------------

    @property
    def equations(self) -> list[AllometricEquation]:
        """Return a shallow copy of the current equation list."""
        return list(self._equations)

    def get(self, name: str) -> Optional[AllometricEquation]:
        """Look up an equation by name (case-sensitive)."""
        for eq in self._equations:
            if eq.name == name:
                return eq
        return None

    # -- Mutators -----------------------------------------------------------

    def add(self, equation: AllometricEquation) -> None:
        """Append an equation.  Replaces an existing one with the same name."""
        self._equations = [
            eq for eq in self._equations if eq.name != equation.name
        ]
        self._equations.append(equation)

    def remove(self, name: str) -> bool:
        """Remove the equation with the given name.  Returns *True* if found."""
        before = len(self._equations)
        self._equations = [eq for eq in self._equations if eq.name != name]
        return len(self._equations) < before

    def reset_defaults(self) -> None:
        """Replace the equation list with the built-in defaults."""
        self._equations = _default_equations()

    # -- Persistence --------------------------------------------------------

    def save(self, path: Optional[str | Path] = None) -> None:
        """Serialize equations to YAML.

        Parameters
        ----------
        path:
            File path.  Falls back to the path given at construction time.
        """
        path = Path(path) if path else self._path
        if path is None:
            raise ValueError("No save path specified and no default path set.")
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(eq) for eq in self._equations]
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def load(self, path: Optional[str | Path] = None) -> None:
        """Deserialize equations from YAML, replacing the current list.

        Parameters
        ----------
        path:
            File path.  Falls back to the path given at construction time.
        """
        path = Path(path) if path else self._path
        if path is None or not path.exists():
            raise FileNotFoundError(f"Equation file not found: {path}")
        with open(path) as f:
            data = yaml.safe_load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected a YAML list, got {type(data).__name__}")
        self._equations = [AllometricEquation(**item) for item in data]

    # -- Evaluation ---------------------------------------------------------

    @staticmethod
    def evaluate(
        equation: AllometricEquation,
        tree_data: pd.DataFrame,
    ) -> pd.Series:
        """Evaluate a single equation against a tree DataFrame.

        The formula is parsed and validated to ensure only whitelisted numpy
        math functions and DataFrame column names are referenced.

        Parameters
        ----------
        equation:
            The equation to evaluate.
        tree_data:
            DataFrame whose columns (and any previously computed allometric
            columns) are exposed as variables.

        Returns
        -------
        A ``pd.Series`` with the computed values, indexed like *tree_data*.

        Raises
        ------
        FormulaValidationError
            If the formula contains disallowed constructs.
        KeyError
            If a required column is missing from *tree_data*.
        """
        # Check that all declared variables are present
        missing = [v for v in equation.variables if v not in tree_data.columns]
        if missing:
            raise KeyError(
                f"Equation '{equation.name}' requires columns {missing} "
                f"which are not in the DataFrame. "
                f"Available: {list(tree_data.columns)}"
            )

        # Build restricted namespace: whitelisted functions + column Series
        namespace: dict[str, object] = dict(_WHITELISTED_NAMES)
        for col in tree_data.columns:
            namespace[col] = tree_data[col]

        allowed_names = set(_WHITELISTED_NAMES.keys()) | set(tree_data.columns)

        result = _safe_eval(equation.formula, namespace, allowed_names)

        # Coerce to Series
        if isinstance(result, pd.Series):
            return result.rename(equation.name)
        return pd.Series(result, index=tree_data.index, name=equation.name)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def compute_allometric_columns(
    tree_data: pd.DataFrame,
    equations: list[AllometricEquation],
) -> pd.DataFrame:
    """Evaluate a sequence of equations and append them as new columns.

    Equations are evaluated in order so that later equations may reference
    columns created by earlier ones (e.g. Carbon depends on AGB).

    Parameters
    ----------
    tree_data:
        Input DataFrame with at minimum the columns required by the first
        equation.
    equations:
        Ordered list of equations to compute.

    Returns
    -------
    A *copy* of *tree_data* with new columns appended.
    """
    df = tree_data.copy()
    for eq in equations:
        try:
            df[eq.name] = AllometryRegistry.evaluate(eq, df)
        except (KeyError, FormulaValidationError) as exc:
            # Skip equations whose dependencies are not met and warn.
            import warnings
            warnings.warn(
                f"Skipping equation '{eq.name}': {exc}",
                stacklevel=2,
            )
    return df
