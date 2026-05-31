#!/bin/bash
# Leonardo Slurm — full inference + scoring from the DEFAULT checkpoint (now the 1.37M model) on 1 A100.
# Produces submission CSVs + the complete validated metric matrix (next-step, completion, anomaly, LM-only
# anomaly, validity). All outputs to $HOME repo (persistent). sbatch from repo root.
#SBATCH --job-name=infer-final
#SBATCH --account=euhpc_d30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --output=slurm-infer-final-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export PATH="$HOME/.pixi/bin:$PATH"
PIXI_RUN="pixi run --manifest-path $(pwd)/pixi.toml"
JOB="${SLURM_JOB_ID:-local}"
CKPT="checkpoints/best.pt"           # the new default (1.37M, 3L/192, RoPE)
OUT="extras/results"                 # deliverable location
echo "=== INFER FINAL | job $JOB | $(date) ==="
$PIXI_RUN python -c "import torch; assert torch.cuda.is_available(); print('CUDA OK:', torch.cuda.get_device_name(0))"
$PIXI_RUN python -c "
import torch,sys; sys.path.insert(0,'src')
from process_logic.model import ProcessLM, ModelConfig
ck=torch.load('$CKPT',map_location='cpu',weights_only=False); m=ProcessLM(ModelConfig(**ck['mcfg']))
print('DEFAULT CKPT: %.3fM  n_layer=%d n_embd=%d pos=%s val_loss=%.4f'%(m.num_params()/1e6,ck['mcfg']['n_layer'],ck['mcfg']['n_embd'],ck['mcfg'].get('pos_encoding'),ck['val_loss']))
"

# 1) submission CSVs (hybrid anomaly) on the eval sets
echo "=== PREDICT (submission CSVs) ==="
$PIXI_RUN python src/process_logic/predict.py --ckpt "$CKPT" --device cuda --out-dir "$OUT" \
    --nextstep-input data/eval_nextstep.csv \
    --completion-input data/eval_completion.csv \
    --anomaly-input data/eval_anomaly.csv \
    --calib-file data/val_id.csv

# 2) full per-family scoring -> file
echo "=== SCORE (next-step / completion / anomaly, per family) ==="
$PIXI_RUN python src/process_logic/score.py --pred-dir "$OUT" --gt-dir data | tee "$OUT/final3m/score_indist.txt"

# 3) LM-only anomaly (honest model evidence)
echo "=== LM-ONLY ANOMALY ==="
$PIXI_RUN python src/process_logic/predict.py --ckpt "$CKPT" --device cuda \
    --out-dir "$OUT/lmonly" --no-validator \
    --anomaly-input data/eval_anomaly.csv --calib-file data/val_id.csv
$PIXI_RUN python src/process_logic/score.py --pred-dir "$OUT/lmonly" --gt-dir data | tee "$OUT/final3m/score_lmonly.txt"

# 4) validity (greedy / sampled / free)
echo "=== VALIDITY ==="
$PIXI_RUN python scripts/measure_validity.py --ckpt "$CKPT" --device cuda --n 300 --temp 1.0 | tee "$OUT/final3m/validity.txt"

echo "=== ROW COUNTS ==="
for f in nextstep completion anomaly; do echo "$f.csv = $(tail -n +2 $OUT/$f.csv | wc -l) rows"; done
echo "=== DONE job $JOB | $(date) ==="
