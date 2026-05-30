#!/bin/bash
# Leonardo (CINECA) Slurm job — run predictions + local scoring on 1 A100.
# Submit after training:  sbatch scripts/run_eval.sh
#SBATCH --job-name=plm-eval
#SBATCH --account=euhpc_d30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=slurm-eval-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export PATH="$HOME/.pixi/bin:$PATH"
PIXI_RUN="pixi run --manifest-path $(pwd)/pixi.toml"

# 1) produce the three submission CSVs on our local eval inputs
$PIXI_RUN python src/process_logic/predict.py --ckpt checkpoints/best.pt \
    --nextstep-input data/eval_nextstep.csv \
    --completion-input data/eval_completion.csv \
    --anomaly-input data/eval_anomaly.csv \
    --calib-file data/val_id.csv

# 2) local per-family scoring
$PIXI_RUN python src/process_logic/score.py --pred-dir extras/results --gt-dir data

# 3) training curves
$PIXI_RUN python scripts/plot_curves.py
