#!/bin/bash
# Leonardo Slurm — round-2 scaling extension (config-only, no code).
# Round 1 found ~6M (n_layer=4,n_embd=256) improves OOD +0.017 over 25M. Map the curve:
# does smaller keep winning, or is ~6M the sweet spot? Each = full 3-fold OOD run_experiment.py.
#   tiny  : n_layer=3,n_embd=128  (~1.5M)
#   xsmall: n_layer=3,n_embd=192  (~3M)
#   mid   : n_layer=6,n_embd=384  (~15M)
# (~6M 'small' and 25M baseline already measured.) ~3 models x ~3 min x 3 = ~30 min. sbatch from repo root.
#SBATCH --job-name=coscilab-scale
#SBATCH --account=euhpc_d30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=slurm-coscilab-scale-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export PATH="$HOME/.pixi/bin:$PATH"
PIXI_RUN="pixi run --manifest-path $(pwd)/pixi.toml"
JOB="${SLURM_JOB_ID:-local}"
RUNDIR="${SCRATCH}/coscilab_scale_${JOB}"; mkdir -p "$RUNDIR"
OUT="extras/results/coscilab"; mkdir -p "$OUT"
echo "=== scaling extension | job $JOB | $(date) ==="
$PIXI_RUN python -c "import torch; assert torch.cuda.is_available(); print('CUDA OK:', torch.cuda.get_device_name(0))"

run_size () {  # $1=id  $2=n_layer  $3=n_embd
  local id="$1" nl="$2" ne="$3"
  cat > "$RUNDIR/${id}_spec.json" <<JSON
{"id":"$id","title":"scaling $id (n_layer=$nl,n_embd=$ne) 3-fold OOD",
 "overrides":{"model.n_layer":$nl,"model.n_embd":$ne},
 "ood":{"folds":["ic","igbt","mosfet"]},"tasks":["nextstep","completion","ood_detect"]}
JSON
  echo "=== SIZE $id : n_layer=$nl n_embd=$ne ==="
  $PIXI_RUN python scripts/run_experiment.py --spec "$RUNDIR/${id}_spec.json" \
      --tier full --device cuda --out "$OUT/result_${id}.json" --workdir "$RUNDIR/${id}_wd" || \
      echo "SIZE $id FAILED (continuing)"
}

run_size tiny   3 128
run_size xsmall 3 192
run_size mid    6 384

echo "=== SCALING SUMMARY: OOD next-step top1 vs size (control 0.4947, small/6M 0.5119) ==="
$PIXI_RUN python - "$OUT" <<'PY'
import json, os, sys
OUT=sys.argv[1]
def params(nl,ne,V=202):
    # rough: tok(V*ne, tied head) + per-layer(attn 4*ne*ne + mlp 3*ne*(8/3*ne ~ rounded))
    per=4*ne*ne + 3*ne*int(((8/3*ne)+7)//8*8); return (V*ne + nl*per)/1e6
rows=[("baseline/25M",8,512,"result_control.json"),("small/6M",4,256,"result_small.json"),
      ("tiny",3,128,"result_tiny.json"),("xsmall",3,192,"result_xsmall.json"),("mid",6,384,"result_mid.json")]
print(f"{'config':14}{'~M':>7}{'ID_top1':>9}{'OOD_top1':>9}{'OOD_top5':>9}{'OOD_comp':>9}")
for name,nl,ne,f in rows:
    p=os.path.join(OUT,f)
    if not os.path.exists(p): print(f"{name:14}{params(nl,ne):7.1f}  (pending)"); continue
    r=json.load(open(p)); m=r['metrics']
    print(f"{name:14}{params(nl,ne):7.1f}{m['id']['nextstep']['top1']:9.4f}{m['ood']['nextstep']['top1']:9.4f}{m['ood']['nextstep']['top5']:9.4f}{m['ood']['completion']['token_acc']:9.4f}")
PY
echo "=== DONE job $JOB | $(date) ==="