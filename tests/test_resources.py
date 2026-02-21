"""Tests for resources, tooltips, and branding assets."""

from pathlib import Path

import pytest

from understory.gui.tooltips import TOOLTIPS, get_tooltip


class TestTooltips:
    def test_tooltips_dict_not_empty(self):
        assert len(TOOLTIPS) > 0

    def test_key_parameters_have_tooltips(self):
        key_params = [
            "plot_radius",
            "batch_size",
            "slice_thickness",
            "model_filename",
        ]
        for param in key_params:
            assert param in TOOLTIPS, f"Missing tooltip for {param}"
            assert len(TOOLTIPS[param]) > 10

    def test_get_tooltip_returns_string(self):
        tip = get_tooltip("plot_radius")
        assert isinstance(tip, str)
        assert len(tip) > 0

    def test_get_tooltip_unknown_returns_empty(self):
        tip = get_tooltip("nonexistent_param_xyz")
        assert tip == ""


class TestBrandingAssets:
    def test_icon_exists(self):
        icon = Path(__file__).parent.parent / "understory" / "resources" / "icons" / "understory-icon.png"
        assert icon.exists()

    def test_logo_exists(self):
        logo = Path(__file__).parent.parent / "understory" / "resources" / "icons" / "understory-logo.png"
        assert logo.exists()

    def test_stylesheet_exists(self):
        qss = Path(__file__).parent.parent / "understory" / "resources" / "styles" / "understory.qss"
        assert qss.exists()

    def test_report_template_exists(self):
        tmpl = Path(__file__).parent.parent / "understory" / "resources" / "report_template.html"
        assert tmpl.exists()

    def test_stylesheet_not_empty(self):
        qss = Path(__file__).parent.parent / "understory" / "resources" / "styles" / "understory.qss"
        content = qss.read_text()
        assert len(content) > 100
        assert "QMainWindow" in content or "QWidget" in content

    def test_report_template_has_brand_elements(self):
        tmpl = Path(__file__).parent.parent / "understory" / "resources" / "report_template.html"
        content = tmpl.read_text()
        assert "Understory" in content or "understory" in content
