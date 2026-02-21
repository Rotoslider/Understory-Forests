#!/usr/bin/env bash
# Understory — Automated Setup for Ubuntu 24.04
# Installs Understory with PyTorch (CUDA 12.8) and PyG extensions.
set -euo pipefail

VENV_DIR="venv"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"
PYG_FIND_LINKS="https://data.pyg.org/whl/torch-2.10.0+cu128.html"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

# ── 1. Check Python ─────────────────────────────────────────────────
PYTHON=""
for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" &>/dev/null; then
        ver=$("$candidate" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        major="${ver%%.*}"
        minor="${ver##*.}"
        if [[ "$major" -eq 3 && "$minor" -ge 10 && "$minor" -le 12 ]]; then
            PYTHON="$candidate"
            break
        fi
    fi
done
[[ -z "$PYTHON" ]] && error "Python 3.10 – 3.12 is required but not found on PATH."
info "Using $PYTHON ($($PYTHON --version))"

# ── 2. Check NVIDIA driver ──────────────────────────────────────────
if command -v nvidia-smi &>/dev/null; then
    DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    info "NVIDIA driver $DRIVER_VER detected"
else
    warn "nvidia-smi not found — GPU acceleration will not be available."
    warn "Install the NVIDIA driver (570+) and CUDA toolkit 12.8 for GPU support."
fi

# ── 3. Create virtual environment ───────────────────────────────────
if [[ -d "$VENV_DIR" ]]; then
    info "Virtual environment already exists at $VENV_DIR — reusing."
else
    info "Creating virtual environment …"
    "$PYTHON" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
info "Activated venv ($VENV_DIR)"

# ── 4. Upgrade pip ──────────────────────────────────────────────────
info "Upgrading pip …"
pip install --upgrade pip --quiet

# ── 5. Install PyTorch with CUDA 12.8 ──────────────────────────────
if python -c "import torch" &>/dev/null; then
    TORCH_VER=$(python -c "import torch; print(torch.__version__)")
    info "PyTorch $TORCH_VER is already installed — skipping."
else
    info "Installing PyTorch (CUDA 12.8) …"
    pip install torch torchvision torchaudio --index-url "$TORCH_INDEX"
fi

# ── 6. Install Understory in editable mode ──────────────────────────
info "Installing Understory (editable mode) …"
pip install -e .

# ── 7. Install PyTorch Geometric extensions ─────────────────────────
if python -c "import torch_scatter" &>/dev/null; then
    info "PyG extensions already installed — skipping."
else
    info "Installing PyTorch Geometric extensions …"
    pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
        -f "$PYG_FIND_LINKS" --no-build-isolation
fi

# ── 8. Verify installation ──────────────────────────────────────────
info "Verifying installation …"
python -c "
import torch
import PySide6
import understory
print(f'  torch        {torch.__version__}  CUDA {torch.version.cuda or \"N/A\"}')
print(f'  PySide6      {PySide6.__version__}')
print(f'  understory   OK')
"

echo ""
info "Setup complete!"
echo ""
echo "  To launch Understory:"
echo ""
echo "    source $VENV_DIR/bin/activate"
echo "    python -m understory"
echo ""
