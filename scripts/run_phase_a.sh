#!/bin/bash
# Leonardo (CINECA) Slurm job — Phase A: validity metric + LM-only anomaly (run on 1 A100).
# Generation-heavy (900 autoregressive decodes), so run on a GPU node, not the login node.
#   sbatch scripts/run_phase_a.sh
#SBATCH --job-name=plm-phaseA
#SBATCH --account=euhpc_d30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --output=slurm-phaseA-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export PATH="$HOME/.pixi/bin:$PATH"
PIXI_RUN="pixi run --manifest-path $(pwd)/pixi.toml"

# Pre-flight: fail fast if CUDA isn't visible.
$PIXI_RUN python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('CUDA OK:', torch.cuda.get_device_name(0))"

# A1 — validity of generations in 3 regimes (greedy ~1.0; sampled/free reveal RL's headroom)
echo "===== A1: validity (temp 1.0) ====="
$PIXI_RUN python scripts/measure_validity.py --ckpt checkpoints/best.pt --n 300 --temp 1.0

# A2 — LM-only anomaly (the MODEL's own evidence, separate from the deterministic-validator hybrid)
echo "===== A2: LM-only anomaly ====="
$PIXI_RUN python src/process_logic/predict.py --ckpt checkpoints/best.pt \
    --out-dir extras/results/lmonly --no-validator \
    --anomaly-input data/eval_anomaly.csv --calib-file data/val_id.csv
$PIXI_RUN python src/process_logic/score.py --pred-dir extras/results/lmonly --gt-dir data
