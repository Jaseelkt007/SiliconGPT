#!/bin/bash
# Leonardo Slurm — additional config-expressible OOD levers, each a full 3-fold OOD run.
# Staged for the closing GPU window; submit after reviewing the augmentation A/B.
# Levers (each independent, written to its own result JSON):
#   nope    : pos_encoding="nope"            (NoPE — causal-mask-only order; length/compositional OOD)
#   small   : n_layer=4,n_embd=256 (~6M)     (scaling point — does less capacity generalize better?)
#   augnope : cross_family_recomb + nope     (combine the two structural levers)
# ~3 models x ~3 min x 3 levers ~= 30 min on 1 A100. sbatch from repo root.
#SBATCH --job-name=coscilab-lev
#SBATCH --account=euhpc_d30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=slurm-coscilab-lev-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export PATH="$HOME/.pixi/bin:$PATH"
PIXI_RUN="pixi run --manifest-path $(pwd)/pixi.toml"
JOB="${SLURM_JOB_ID:-local}"
RUNDIR="${SCRATCH}/coscilab_lev_${JOB}"; mkdir -p "$RUNDIR"
OUT="extras/results/coscilab"; mkdir -p "$OUT"
echo "=== levers Tier-2 | job $JOB | $(date) ==="
$PIXI_RUN python -c "import torch; assert torch.cuda.is_available(); print('CUDA OK:', torch.cuda.get_device_name(0))"

run_lever () {  # $1=id  $2=overrides-json
  local id="$1" ov="$2"
  cat > "$RUNDIR/${id}_spec.json" <<JSON
{"id":"$id","title":"$id lever 3-fold OOD","overrides":$ov,
 "ood":{"folds":["ic","igbt","mosfet"]},"tasks":["nextstep","completion","ood_detect"]}
JSON
  echo "=== LEVER $id : $ov ==="
  $PIXI_RUN python scripts/run_experiment.py --spec "$RUNDIR/${id}_spec.json" \
      --tier full --device cuda --out "$OUT/result_${id}.json" --workdir "$RUNDIR/${id}_wd" || \
      echo "LEVER $id FAILED (continuing)"
}

run_lever nope    '{"model.pos_encoding":"nope"}'
run_lever small   '{"model.n_layer":4,"model.n_embd":256}'
run_lever augnope '{"data.augmentation":"cross_family_recomb","data.aug_ratio":0.15,"data.aug_cap":6000,"model.pos_encoding":"nope"}'

echo "=== SUMMARY: OOD next-step top1 per lever (vs control 0.4947) ==="
$PIXI_RUN python - "$OUT" <<'PY'
import json, os, sys
OUT = sys.argv[1]
ctrl = json.load(open(os.path.join(OUT,"result_control.json")))["metrics"]["ood"]["nextstep"]["top1"]
print(f"control            OOD top1 = {ctrl}")
for id in ["aug","nope","small","augnope"]:
    p = os.path.join(OUT, f"result_{id}.json")
    if os.path.exists(p):
        r = json.load(open(p))
        o = r["metrics"]["ood"]["nextstep"]["top1"]; i = r["metrics"]["id"]["nextstep"]["top1"]
        print(f"{id:18} OOD top1 = {o}  (ID {i}, dOOD {o-ctrl:+.4f}, verdict {r.get('verdict')})")
PY
echo "=== DONE job $JOB | $(date) ==="