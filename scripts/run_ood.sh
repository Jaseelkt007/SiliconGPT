#!/bin/bash
# Leonardo (CINECA) Slurm job — OOD generalization experiment (the deciding metric).
# Trains on TWO families (excludes EXCLUDE, default ic), then predicts + scores on the
# full eval files. Read the held-out family's row in the score output as the OOD result,
# and the other two families as in-distribution (for THIS model) — the gap is the ID->OOD drop.
#   sbatch scripts/run_ood.sh                 # excludes ic (matches ood_holdout.csv)
#   sbatch --export=ALL,EXCLUDE=igbt scripts/run_ood.sh
#SBATCH --job-name=plm-ood
#SBATCH --account=euhpc_d30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=slurm-ood-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export PATH="$HOME/.pixi/bin:$PATH"

EXCLUDE="${EXCLUDE:-ic}"
CKPT_DIR="checkpoints/ood_${EXCLUDE}"
OUT_DIR="extras/results/ood_${EXCLUDE}"
mkdir -p "$OUT_DIR"

# W&B offline (sync from a login node afterwards: pixi run wandb sync wandb/latest-run)
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-silicongpt}"
export WANDB_DIR="$(pwd)"

PIXI_RUN="pixi run --manifest-path $(pwd)/pixi.toml"

# Pre-flight: fail fast if CUDA isn't visible.
$PIXI_RUN python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('CUDA OK:', torch.cuda.get_device_name(0))"

# 1) train WITHOUT the held-out family (separate ckpt/log dirs so the ID model is untouched)
$PIXI_RUN python src/process_logic/train.py \
    --config configs/train_v1.yaml \
    --model-config configs/model_v1.yaml \
    --exclude-family "$EXCLUDE" \
    --ckpt-dir "$CKPT_DIR" \
    --out-dir "$OUT_DIR" \
    --wandb --run-name "ood-excl-${EXCLUDE}-${SLURM_JOB_ID:-local}"

# 2) predict on the FULL eval files with the OOD model
$PIXI_RUN python src/process_logic/predict.py --ckpt "$CKPT_DIR/best.pt" \
    --out-dir "$OUT_DIR" \
    --nextstep-input data/eval_nextstep.csv \
    --completion-input data/eval_completion.csv \
    --anomaly-input data/eval_anomaly.csv \
    --calib-file data/val_id.csv

# 3) score per family. The "$EXCLUDE" row = OOD; the other two = in-distribution for this model.
echo "=== OOD scores (held-out family = $EXCLUDE) ==="
$PIXI_RUN python src/process_logic/score.py --pred-dir "$OUT_DIR" --gt-dir data
