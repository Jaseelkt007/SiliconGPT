#!/bin/bash
# Leonardo Slurm — h2 weight-sharing (Universal-Transformer depth sharing) at the OOD-best ~3M base.
# Tests whether structural capacity-removal (one block reused across depth) stacks with the size
# capacity-removal that already won (scaling). Two arms at n_layer=3,n_embd=192 (~3M):
#   ws_off : baseline ~3M (control for this comparison; ~= result_xsmall)
#   ws_on  : same size + model.weight_share=true
# Each = full 3-fold OOD run_experiment.py. ~3 models x ~3 min x 2 = ~20 min. sbatch from repo root.
#SBATCH --job-name=coscilab-ws
#SBATCH --account=euhpc_d30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=slurm-coscilab-ws-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export PATH="$HOME/.pixi/bin:$PATH"
PIXI_RUN="pixi run --manifest-path $(pwd)/pixi.toml"
JOB="${SLURM_JOB_ID:-local}"
RUNDIR="${SCRATCH}/coscilab_ws_${JOB}"; mkdir -p "$RUNDIR"
OUT="extras/results/coscilab"; mkdir -p "$OUT"
echo "=== h2 weight-share Tier-2 | job $JOB | $(date) ==="
$PIXI_RUN python -c "import torch; assert torch.cuda.is_available(); print('CUDA OK:', torch.cuda.get_device_name(0))"

run_arm () {  # $1=id  $2=overrides
  local id="$1" ov="$2"
  cat > "$RUNDIR/${id}_spec.json" <<JSON
{"id":"$id","title":"$id 3-fold OOD","overrides":$ov,
 "ood":{"folds":["ic","igbt","mosfet"]},"tasks":["nextstep","completion","ood_detect"]}
JSON
  echo "=== ARM $id : $ov ==="
  $PIXI_RUN python scripts/run_experiment.py --spec "$RUNDIR/${id}_spec.json" \
      --tier full --device cuda --out "$OUT/result_${id}.json" --workdir "$RUNDIR/${id}_wd" || \
      echo "ARM $id FAILED (continuing)"
}

run_arm ws3m_off '{"model.n_layer":3,"model.n_embd":192}'
run_arm ws3m_on  '{"model.n_layer":3,"model.n_embd":192,"model.weight_share":true}'

echo "=== h2 SUMMARY (3M base: weight_share off vs on; xsmall ref OOD 0.512) ==="
$PIXI_RUN python - "$OUT" <<'PY'
import json, os, sys
OUT=sys.argv[1]
for id in ["ws3m_off","ws3m_on"]:
    p=os.path.join(OUT,f"result_{id}.json")
    if not os.path.exists(p): print(f"{id}: pending"); continue
    m=json.load(open(p))['metrics']
    print(f"{id:10} ID={m['id']['nextstep']['top1']:.4f} OOD={m['ood']['nextstep']['top1']:.4f} "
          f"OOD5={m['ood']['nextstep']['top5']:.4f} comp={m['ood']['completion']['token_acc']:.4f}")
PY
echo "=== DONE job $JOB | $(date) ==="
