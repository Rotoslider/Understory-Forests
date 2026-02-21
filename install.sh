#!/usr/bin/env bash
# Understory — Automated Setup for Ubuntu 24.04
# Installs Understory with PyTorch (CUDA 12.8) and PyG extensions.
set -euo pipefail

VENV_DIR="venv"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

# ── Parse flags ───────────────────────────────────────────────────────
CLEAN=false
for arg in "$@"; do
    case "$arg" in
        --clean) CLEAN=true ;;
        --help|-h)
            echo "Usage: ./install.sh [--clean]"
            echo "  --clean   Remove existing venv and start fresh"
            exit 0
            ;;
        *) error "Unknown option: $arg" ;;
    esac
done

# ── 1. Check / install system dependencies ────────────────────────────
info "Checking system dependencies …"

MISSING_PKGS=()

# Python venv support (e.g. python3.12-venv)
PY_VER=$( (python3.12 --version 2>/dev/null || python3.11 --version 2>/dev/null || python3.10 --version 2>/dev/null || python3 --version 2>/dev/null) | grep -oP '\d+\.\d+' | head -1)
if [[ -n "$PY_VER" ]]; then
    VENV_PKG="python${PY_VER}-venv"
    if ! dpkg -s "$VENV_PKG" &>/dev/null; then
        MISSING_PKGS+=("$VENV_PKG")
    fi
fi

# Build tools (needed for compiled extensions like open3d, hdbscan, etc.)
for pkg in build-essential python3-dev; do
    if ! dpkg -s "$pkg" &>/dev/null; then
        MISSING_PKGS+=("$pkg")
    fi
done

# X11 / OpenGL libraries needed by PySide6 and VTK
for pkg in libgl1 libegl1 libxcb-cursor0 libxkbcommon-x11-0 libxcb-icccm4 libxcb-keysyms1 libxcb-shape0 libxcb-xinerama0; do
    if ! dpkg -s "$pkg" &>/dev/null; then
        MISSING_PKGS+=("$pkg")
    fi
done

if [[ ${#MISSING_PKGS[@]} -gt 0 ]]; then
    info "Installing missing system packages: ${MISSING_PKGS[*]}"
    sudo apt-get update -qq
    sudo apt-get install -y "${MISSING_PKGS[@]}"
else
    info "All system dependencies are present."
fi

# ── 2. Check Python ──────────────────────────────────────────────────
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

# ── 3. Check NVIDIA driver ──────────────────────────────────────────
if command -v nvidia-smi &>/dev/null; then
    DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    info "NVIDIA driver $DRIVER_VER detected"
else
    warn "nvidia-smi not found — GPU acceleration will not be available."
    warn "Install the NVIDIA driver (570+) and CUDA toolkit 12.8 for GPU support."
fi

# ── 4. Create virtual environment ───────────────────────────────────
if [[ "$CLEAN" == true && -d "$VENV_DIR" ]]; then
    info "Removing existing virtual environment (--clean) …"
    rm -rf "$VENV_DIR"
fi

if [[ -d "$VENV_DIR" ]]; then
    # Verify the existing venv is functional
    if "$VENV_DIR/bin/python" -c "import pip" &>/dev/null; then
        info "Virtual environment already exists at $VENV_DIR — reusing."
    else
        warn "Existing virtual environment is broken — recreating …"
        rm -rf "$VENV_DIR"
        "$PYTHON" -m venv "$VENV_DIR"
    fi
else
    info "Creating virtual environment …"
    "$PYTHON" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
info "Activated venv ($VENV_DIR)"

# ── 5. Upgrade pip ──────────────────────────────────────────────────
info "Upgrading pip …"
pip install --upgrade pip --quiet

# ── 6. Install PyTorch with CUDA 12.8 ──────────────────────────────
if python -c "import torch" &>/dev/null; then
    TORCH_VER=$(python -c "import torch; print(torch.__version__)")
    info "PyTorch $TORCH_VER is already installed — skipping."
else
    info "Installing PyTorch (CUDA 12.8) …"
    pip install torch torchvision torchaudio --index-url "$TORCH_INDEX"
fi

# ── 7. Install Understory in editable mode ──────────────────────────
info "Installing Understory (editable mode) …"
pip install -e .

# ── 8. Install PyTorch Geometric extensions ─────────────────────────
# PyG extensions (torch-scatter, torch-cluster, etc.) need pre-built wheels
# that match the exact PyTorch+CUDA version. Wheels are hosted at data.pyg.org
# but often lag behind the latest PyTorch releases.

# Detect installed torch version to construct the correct wheel URL
TORCH_SHORT=$(python -c "import torch; print(torch.__version__.split('+')[0])")
CUDA_TAG=$(python -c "import torch; v=torch.version.cuda; print('+cu'+v.replace('.','') if v else '+cpu')")
PYG_FIND_LINKS="https://data.pyg.org/whl/torch-${TORCH_SHORT}${CUDA_TAG}.html"

_install_pyg_extensions() {
    # Check if pre-built wheels exist for this torch version
    if curl -sf "$PYG_FIND_LINKS" -o /dev/null 2>&1; then
        info "Pre-built PyG wheels available for torch ${TORCH_SHORT}${CUDA_TAG}"
        pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
            -f "$PYG_FIND_LINKS"
    else
        warn "No pre-built PyG wheels for torch ${TORCH_SHORT}${CUDA_TAG} — building from source."
        # Building CUDA extensions from source requires nvcc (CUDA compiler)
        if ! command -v nvcc &>/dev/null; then
            info "Installing nvidia-cuda-toolkit (needed to compile CUDA extensions) …"
            sudo apt-get update -qq
            sudo apt-get install -y nvidia-cuda-toolkit
        fi
        pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
            --no-build-isolation
    fi
}

if python -c "import torch_cluster" &>/dev/null; then
    info "PyG extensions already installed — skipping."
else
    info "Installing PyTorch Geometric extensions …"
    _install_pyg_extensions
fi

# ── 9. Verify CUDA support in PyG extensions ─────────────────────────
if command -v nvidia-smi &>/dev/null; then
    info "Verifying CUDA support in PyG extensions …"
    if python -c "
import torch, torch_cluster
# Test farthest-point-sampling on GPU (used by the PointNet++ model)
pos = torch.randn(16, 3, device='cuda')
batch = torch.zeros(16, dtype=torch.long, device='cuda')
torch_cluster.fps(pos, batch, ratio=0.5)
" 2>/dev/null; then
        info "PyG CUDA support: OK"
    else
        echo ""
        warn "PyG extensions were installed WITHOUT CUDA support!"
        warn "GPU inference will fail. To fix, run:"
        echo ""
        echo "    source $VENV_DIR/bin/activate"
        echo "    pip uninstall -y torch-scatter torch-sparse torch-cluster torch-spline-conv"
        echo "    sudo apt install nvidia-cuda-toolkit"
        echo "    pip install torch-scatter torch-sparse torch-cluster torch-spline-conv --no-build-isolation"
        echo ""
        warn "Alternatively, switch to CPU mode in Process settings (slower but works)."
    fi
fi

# ── 10. Verify installation ──────────────────────────────────────────
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
