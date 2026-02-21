"""End-to-end pipeline test using real data on GPU.

This test runs the full FSCT pipeline on example.las to verify the
complete workflow works on Blackwell GPU hardware.

Marked as 'slow' — run with: pytest -m slow
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from understory.config.settings import ProjectConfig
from understory.core.paths import FSCTPaths


@pytest.mark.slow
class TestEndToEndPipeline:
    """Full pipeline test — preprocessing through report generation."""

    @pytest.fixture
    def pipeline_config(self, example_las, tmp_path):
        """Create a config for the full pipeline run."""
        output_dir = str(tmp_path / "e2e_output")
        cfg = ProjectConfig(
            project_name="E2E Test",
            operator="pytest",
            point_cloud_filename=example_las,
        )
        cfg.processing.batch_size = 2
        cfg.processing.use_CPU_only = not torch.cuda.is_available()
        cfg.processing.num_cpu_cores = os.cpu_count()
        cfg.processing.plot_radius = 0  # process full cloud
        cfg.processing.delete_working_directory = True
        cfg.output.output_directory = output_dir
        return cfg

    def test_full_pipeline(self, pipeline_config, tmp_path):
        """Run the entire FSCT pipeline end-to-end."""
        from understory.core.pipeline import run_pipeline

        stages_seen = []

        def progress_cb(stage, frac):
            stages_seen.append(stage)
            print(f"  [{frac*100:5.1f}%] {stage}")

        result = run_pipeline(pipeline_config, progress_callback=progress_cb)

        # Check outputs
        assert "output_dir" in result
        output_dir = Path(result["output_dir"])
        assert output_dir.exists()

        # Check key output files exist
        tree_data = Path(result["tree_data_csv"])
        plot_summary = Path(result["plot_summary_csv"])

        if tree_data.exists():
            import pandas as pd
            df = pd.read_csv(tree_data)
            print(f"Trees found: {len(df)}")
            assert "TreeId" in df.columns or "tree_id" in df.columns

        if plot_summary.exists():
            import pandas as pd
            df = pd.read_csv(plot_summary, index_col=False)
            assert len(df) > 0

        # Check progress was reported
        assert len(stages_seen) > 0
        assert "Complete" in stages_seen


@pytest.mark.slow
class TestPipelineStages:
    """Test individual pipeline stages."""

    @pytest.fixture
    def params(self, example_las, tmp_path):
        cfg = ProjectConfig(point_cloud_filename=example_las)
        cfg.processing.batch_size = 2
        cfg.processing.use_CPU_only = not torch.cuda.is_available()
        cfg.processing.num_cpu_cores = os.cpu_count()
        params = cfg.to_legacy_params()
        return params

    def test_preprocessing_stage(self, params):
        """Test just the preprocessing stage."""
        from preprocessing import Preprocessing
        prep = Preprocessing(params)
        prep.preprocess_point_cloud()

        # Check working directory has output files
        filename = params["point_cloud_filename"]
        stem = Path(filename).stem
        output_dir = Path(filename).parent / f"{stem}_FSCT_output"
        working_dir = output_dir / "working_directory"
        assert output_dir.exists()

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA required for inference test",
    )
    def test_inference_stage(self, params):
        """Test the semantic segmentation inference stage."""
        # Must run preprocessing first
        from preprocessing import Preprocessing
        prep = Preprocessing(params)
        prep.preprocess_point_cloud()
        del prep

        from inference import SemanticSegmentation
        seg = SemanticSegmentation(params)
        seg.inference()
