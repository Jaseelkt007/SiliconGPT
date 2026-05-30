#!/bin/bash
# Leonardo Slurm — FINALIZE the model-improvement section in one job.
# Trains the 3M deliverable (all 3 families) in RoPE and NoPE variants, picks the better by val_loss,
# generates submission CSVs, scores all in-dist metrics + validity, and seed-confirms the 3-fold OOD gain.
# ALL outputs to $HOME repo (persistent), copied as we go. sbatch from repo root.
#SBATCH --job-name=final3m
#SBATCH --account=euhpc_d30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=slurm-final3m-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export PATH="$HOME/.pixi/bin:$PATH"
PIXI_RUN="pixi run --manifest-path $(pwd)/pixi.toml"
JOB="${SLURM_JOB_ID:-local}"
OUT="extras/results/final3m"; mkdir -p "$OUT"
CKD="checkpoints"   # persistent ($HOME repo)
echo "=== FINALIZE 3M | job $JOB | $(date) ==="
$PIXI_RUN python -c "import torch; assert torch.cuda.is_available(); print('CUDA OK:', torch.cuda.get_device_name(0))"

train_one () {  # $1=tag(rope|nope)  $2=model-config
  local tag="$1" mcfg="$2"
  echo "=== TRAIN 3M $tag (all families) ==="
  $PIXI_RUN python src/process_logic/train.py \
      --config configs/train_v1.yaml --model-config "$mcfg" \
      --ckpt-dir "$CKD/final_3m_$tag" --out-dir "$OUT/train_$tag" \
      --run-name "final_3m_$tag" --device cuda
}

# ---- A) deliverable: RoPE + NoPE on all 3 families ----
train_one rope configs/model_3m_rope.yaml
train_one nope configs/model_3m_nope.yaml

# ---- pick the better by best_val (RoPE on tie) ----
$PIXI_RUN python - "$CKD/final_3m_rope/best.pt" "$CKD/final_3m_nope/best.pt" > "$OUT/pick.txt" <<'PY'
import sys, torch
r=torch.load(sys.argv[1],map_location="cpu",weights_only=False)["val_loss"]
n=torch.load(sys.argv[2],map_location="cpu",weights_only=False)["val_loss"]
pick="rope" if r<=n+1e-4 else "nope"   # RoPE default unless NoPE clearly better
print(f"rope_val={r:.4f} nope_val={n:.4f} PICK={pick}")
open("extras/results/final3m/PICK","w").write(pick)
PY
cat "$OUT/pick.txt"
PICK=$(cat "$OUT/PICK")
echo "CHOSEN=$PICK"
FINAL_CKPT="$CKD/final_3m_$PICK/best.pt"

# ---- B) submission CSVs from the chosen deliverable (full eval sets) ----
echo "=== PREDICT (submission CSVs) from $FINAL_CKPT ==="
$PIXI_RUN python src/process_logic/predict.py --ckpt "$FINAL_CKPT" \
    --out-dir "$OUT/submission" \
    --nextstep-input data/eval_nextstep.csv \
    --completion-input data/eval_completion.csv \
    --anomaly-input data/eval_anomaly.csv \
    --calib-file data/val_id.csv
echo "=== SCORE in-distribution (3M deliverable) ==="
$PIXI_RUN python src/process_logic/score.py --pred-dir "$OUT/submission" --gt-dir data | tee "$OUT/score_indist.txt"

# LM-only anomaly (honest model evidence)
$PIXI_RUN python src/process_logic/predict.py --ckpt "$FINAL_CKPT" \
    --out-dir "$OUT/submission_lmonly" --no-validator \
    --anomaly-input data/eval_anomaly.csv --calib-file data/val_id.csv
$PIXI_RUN python src/process_logic/score.py --pred-dir "$OUT/submission_lmonly" --gt-dir data | tee "$OUT/score_lmonly.txt"

# ---- validity (greedy + sampled + free) ----
echo "=== VALIDITY (3M deliverable) ==="
$PIXI_RUN python scripts/measure_validity.py --ckpt "$FINAL_CKPT" --n 300 --temp 1.0 | tee "$OUT/validity.txt"

# ---- C) seed-confirm the 3-fold OOD gain: 3M across seeds 42,43,44 ----
echo "=== SEED-CONFIRM 3-fold OOD (3M $PICK, seeds 42 43 44) ==="
for seed in 42 43 44; do
  cat > "$OUT/ood_spec_s$seed.json" <<JSON
{"id":"final3m_ood_s$seed","title":"3M $PICK 3-fold OOD seed $seed",
 "overrides":{"model.n_layer":3,"model.n_embd":192,"model.pos_encoding":"$PICK","train.seed":$seed},
 "ood":{"folds":["ic","igbt","mosfet"]},"tasks":["nextstep","completion","ood_detect"]}
JSON
  $PIXI_RUN python scripts/run_experiment.py --spec "$OUT/ood_spec_s$seed.json" \
      --tier full --device cuda --out "$OUT/result_ood_s$seed.json" \
      --workdir "$SCRATCH/final3m_ood_s${seed}_$JOB" || echo "seed $seed FAILED (continuing)"
done

echo "=== SEED SUMMARY ==="
$PIXI_RUN python - "$OUT" <<'PY'
import json, os, sys, statistics as st
OUT=sys.argv[1]
vals=[]
for s in (42,43,44):
    p=os.path.join(OUT,f"result_ood_s{s}.json")
    if os.path.exists(p):
        v=json.load(open(p))["metrics"]["ood"]["nextstep"]["top1"]; vals.append(v)
        print(f"seed {s}: OOD top1 = {v:.4f}")
if vals:
    m=sum(vals)/len(vals)
    sd=st.pstdev(vals) if len(vals)>1 else 0.0
    print(f"MEAN OOD top1 = {m:.4f} +/- {sd:.4f}  (n={len(vals)}; baseline 25M = 0.4947)")
    open(os.path.join(OUT,"ood_seed_summary.txt"),"w").write(f"mean={m:.4f} sd={sd:.4f} n={len(vals)} vals={vals}\n")
PY
echo "=== DONE job $JOB | $(date) ==="