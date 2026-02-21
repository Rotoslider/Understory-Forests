"""Shared fixtures for the Understory test suite."""

import os
import sys
from pathlib import Path

import pytest

# Ensure scripts/ is on sys.path for pipeline imports
SCRIPTS_DIR = str(Path(__file__).resolve().parent.parent / "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_LAS = PROJECT_ROOT / "data" / "train" / "example.las"
MODEL_PATH = PROJECT_ROOT / "model" / "model.pth"


@pytest.fixture
def project_root():
    return PROJECT_ROOT


@pytest.fixture
def example_las():
    """Path to the example LAS file (skip if missing)."""
    if not EXAMPLE_LAS.exists():
        pytest.skip("example.las not found in data/train/")
    return str(EXAMPLE_LAS)


@pytest.fixture
def model_path():
    """Path to model.pth (skip if missing)."""
    if not MODEL_PATH.exists():
        pytest.skip("model.pth not found in model/")
    return str(MODEL_PATH)
