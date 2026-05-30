#!/bin/bash
# Leonardo (CINECA) Slurm job — train the process-logic base LM on 1 A100.
# Submit from the repo root on a login node:  sbatch scripts/run_train.sh
# For 2 GPUs: --gpus-per-task=2 --mem=240GB --cpus-per-task=16 (etc. up to 4).
#SBATCH --job-name=plm-train
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

# Compute nodes have NO internet. Proxy only needed for low-bandwidth (e.g. wandb).
# export HTTP_PROXY=http://proxyuser:5dd1d2bd00@10.99.0.1:38425
# export HTTPS_PROXY=$HTTP_PROXY http_proxy=$HTTP_PROXY https_proxy=$HTTP_PROXY

# Run inside the pixi env (build it on a login node first: bash scripts/setup_leonardo.sh)
PIXI_RUN="pixi run --manifest-path $(pwd)/pixi.toml"

$PIXI_RUN python src/process_logic/train.py \
    --config configs/train_v1.yaml \
    --model-config configs/model_v1.yaml \
    "$@"
