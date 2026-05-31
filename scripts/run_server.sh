#!/bin/bash
#SBATCH --job-name=sgpt-api
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120GB
#SBATCH --time=12:00:00
#SBATCH --output=slurm-api-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
echo "Backend node : $(hostname)"
echo "Port         : ${PORT:-5050}"
echo "Checkpoint   : ${CHECKPOINT_PATH:-(unset)}"
export PORT=${PORT:-5050}
export HOST=0.0.0.0
pixi run python server/app.py
