#!/bin/bash
# Build the pixi environment ON A LEONARDO LOGIN NODE (has internet).
# Compute nodes have NO internet, so all deps must be installed here first.
#   bash scripts/setup_leonardo.sh
set -e

# 1) install pixi if missing
if ! command -v pixi >/dev/null 2>&1; then
    curl -fsSL https://pixi.sh/install.sh | bash
    export PATH="$HOME/.pixi/bin:$PATH"
fi

cd "$(dirname "$0")/.."
[ -f pixi.toml ] || pixi init .

# 2) core deps (from conda-forge unless --pypi)
pixi add python=3.12 numpy pyyaml tqdm scikit-learn matplotlib

# 3) PyTorch with CUDA for the A100s. The exact spec may need adjusting for
#    Leonardo's CUDA — check the HPC onboarding kit (Ch.6) if this fails:
#      https://ai-at.eu/hpc-onboarding/
pixi add "pytorch-gpu" "cuda-version=12.*" || pixi add --pypi torch

# 4) optional experiment tracking
pixi add --pypi wandb || true

echo
echo "pixi env ready. Quick CPU smoke (login node, <10 min CPU limit):"
echo "  pixi run python src/process_logic/train.py --smoke --device cpu"
echo "Full GPU run:"
echo "  sbatch scripts/run_train.sh"
