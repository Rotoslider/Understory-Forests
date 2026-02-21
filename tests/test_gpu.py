"""GPU and model tests — verifying CUDA works on Blackwell architecture."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


def requires_cuda(fn):
    """Skip test if CUDA is not available."""
    return pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available",
    )(fn)


class TestCUDABasics:
    @requires_cuda
    def test_cuda_available(self):
        assert torch.cuda.is_available()

    @requires_cuda
    def test_gpu_properties(self):
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        assert len(name) > 0
        assert mem > 0
        print(f"GPU: {name}, VRAM: {mem:.0f} GB")

    @requires_cuda
    def test_cuda_compute(self):
        a = torch.randn(1000, 1000, device="cuda")
        b = torch.randn(1000, 1000, device="cuda")
        c = a @ b
        assert c.shape == (1000, 1000)
        assert c.device.type == "cuda"

    @requires_cuda
    def test_cuda_memory_allocation(self):
        """Allocate a large tensor to verify GPU memory works."""
        # ~400MB tensor
        t = torch.randn(100_000_000, device="cuda")
        assert t.shape[0] == 100_000_000
        del t
        torch.cuda.empty_cache()

    @requires_cuda
    def test_cuda_to_cpu_transfer(self):
        t = torch.randn(100, 100, device="cuda")
        t_cpu = t.cpu()
        assert t_cpu.device.type == "cpu"
        np.testing.assert_allclose(t.cpu().numpy(), t_cpu.numpy())


class TestPyGOnGPU:
    @requires_cuda
    def test_pyg_imports(self):
        import torch_geometric
        from torch_geometric.nn import PointNetConv, fps, radius, global_max_pool
        from torch_geometric.data import Data

    @requires_cuda
    def test_pyg_data_on_gpu(self):
        from torch_geometric.data import Data
        pos = torch.randn(100, 3, device="cuda")
        batch = torch.zeros(100, dtype=torch.long, device="cuda")
        data = Data(pos=pos, batch=batch)
        assert data.pos.device.type == "cuda"

    @requires_cuda
    def test_pyg_fps_on_gpu(self):
        """Test farthest point sampling on GPU."""
        from torch_geometric.nn import fps
        pos = torch.randn(500, 3, device="cuda")
        batch = torch.zeros(500, dtype=torch.long, device="cuda")
        idx = fps(pos, batch, ratio=0.1)
        assert idx.shape[0] == 50  # 500 * 0.1

    @requires_cuda
    def test_pyg_radius_on_gpu(self):
        """Test radius neighbor search on GPU."""
        from torch_geometric.nn import radius
        pos = torch.randn(200, 3, device="cuda")
        batch = torch.zeros(200, dtype=torch.long, device="cuda")
        row, col = radius(pos, pos[:20], r=0.5, batch_x=batch, batch_y=batch[:20])
        assert row.device.type == "cuda"


class TestModelArchitecture:
    @requires_cuda
    def test_model_instantiation(self):
        from model import Net
        model = Net(num_classes=4)
        model = model.cuda()
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0
        print(f"Model parameters: {total_params:,}")

    @requires_cuda
    def test_model_forward_pass(self):
        """Verify forward pass through the PointNet++ model on GPU."""
        from model import Net
        from torch_geometric.data import Data

        model = Net(num_classes=4).cuda().eval()

        # Simulate a small point cloud batch
        num_points = 1000
        pos = torch.randn(num_points, 3, device="cuda")
        batch = torch.zeros(num_points, dtype=torch.long, device="cuda")
        data = Data(pos=pos, x=None, batch=batch)

        with torch.no_grad():
            output = model(data)

        assert output.shape[0] == 1
        assert output.shape[1] == 4  # num_classes
        assert output.shape[2] == num_points

    @requires_cuda
    def test_model_load_weights(self, model_path):
        """Load the actual model weights and verify."""
        from model import Net
        model = Net(num_classes=4)
        state_dict = torch.load(model_path, weights_only=True, map_location="cuda")
        model.load_state_dict(state_dict, strict=False)
        model = model.cuda().eval()
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0

    @requires_cuda
    def test_model_inference_with_real_weights(self, model_path):
        """Run a forward pass with real weights to verify full pipeline."""
        from model import Net
        from torch_geometric.data import Data

        model = Net(num_classes=4)
        state_dict = torch.load(model_path, weights_only=True, map_location="cuda")
        model.load_state_dict(state_dict, strict=False)
        model = model.cuda().eval()

        num_points = 2000
        pos = torch.randn(num_points, 3, device="cuda")
        batch = torch.zeros(num_points, dtype=torch.long, device="cuda")
        data = Data(pos=pos, x=None, batch=batch)

        with torch.no_grad():
            output = model(data)

        # Output should be log-probabilities
        probs = torch.exp(output)
        assert probs.min() >= 0
        assert probs.max() <= 1.01  # allow small numerical error
        # Classes should sum to ~1
        class_sums = probs.sum(dim=1).squeeze()
        np.testing.assert_allclose(class_sums.cpu().numpy(), 1.0, atol=0.01)
