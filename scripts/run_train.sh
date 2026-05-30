#!/bin/bash
# Leonardo (CINECA) Slurm job — train the process-logic base LM on 1 A100.
# Submit from the repo root on a login node:  sbatch scripts/run_train.sh
# For 2 GPUs: --gpus-per-task=2 --mem=240GB --cpus-per-task=16 (etc. up to 4).
#SBATCH --job-name=plm-train
#SBATCH --account=euhpc_d30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# pixi lives in ~/.pixi/bin; ensure it's on PATH even in a non-login shell.
export PATH="$HOME/.pixi/bin:$PATH"

# ---- Weights & Biases ----
# Default: OFFLINE during the job. Compute nodes have no direct internet, and the
# proxy is low-bandwidth and restarts ~every 10 min, so live streaming can stall a
# job. Offline writes the run to ./wandb (gitignored); after the job, push it online
# from a LOGIN node (stable internet):   pixi run wandb sync wandb/latest-run
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-silicongpt}"
export WANDB_DIR="$(pwd)"
# For LIVE online logging from the compute node instead, submit with
#   sbatch --export=ALL,WANDB_MODE=online scripts/run_train.sh
# and uncomment the proxy (needed for any compute-node internet):
#   export HTTP_PROXY=http://proxyuser:5dd1d2bd00@10.99.0.1:38425
#   export HTTPS_PROXY=$HTTP_PROXY http_proxy=$HTTP_PROXY https_proxy=$HTTP_PROXY

# Run inside the pixi env (build it on a login node first: bash scripts/setup_leonardo.sh)
PIXI_RUN="pixi run --manifest-path $(pwd)/pixi.toml"

# Pre-flight: fail fast if CUDA isn't visible (wrong node / driver mismatch).
$PIXI_RUN python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available on this node'; print('CUDA OK:', torch.cuda.get_device_name(0), '| torch', torch.__version__)"

$PIXI_RUN python src/process_logic/train.py \
    --config configs/train_v1.yaml \
    --model-config configs/model_v1.yaml \
    --wandb --run-name "v1.1-${SLURM_JOB_ID:-local}" \
    "$@"
