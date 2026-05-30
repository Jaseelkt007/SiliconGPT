#!/bin/bash
# Leonardo Slurm — TREATMENT arm only (cross_family_recomb), full 3-fold OOD.
# CONTROL already ran (extras/results/coscilab/result_control.json). This adds the aug arm
# + prints the A/B vs the saved control. ~13 min on 1 A100. sbatch from repo root.
#SBATCH --job-name=coscilab-aug
#SBATCH --account=euhpc_d30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=slurm-coscilab-aug-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export PATH="$HOME/.pixi/bin:$PATH"
PIXI_RUN="pixi run --manifest-path $(pwd)/pixi.toml"
JOB="${SLURM_JOB_ID:-local}"
RUNDIR="${SCRATCH}/coscilab_aug_${JOB}"; mkdir -p "$RUNDIR"
OUT="extras/results/coscilab"; mkdir -p "$OUT"
echo "=== aug-only Tier-2 | job $JOB | $(date) ==="

$PIXI_RUN python -c "import torch; assert torch.cuda.is_available(); print('CUDA OK:', torch.cuda.get_device_name(0))"

cat > "$RUNDIR/aug_spec.json" <<'JSON'
{"id":"aug_xfam","title":"cross_family_recomb augmentation, 3-fold OOD",
 "overrides":{"data.augmentation":"cross_family_recomb","data.aug_ratio":0.15,"data.aug_cap":6000},
 "ood":{"folds":["ic","igbt","mosfet"]},
 "tasks":["nextstep","completion","ood_detect"]}
JSON
echo "=== TREATMENT (cross_family_recomb) — full 3-fold OOD ==="
$PIXI_RUN python scripts/run_experiment.py --spec "$RUNDIR/aug_spec.json" \
    --tier full --device cuda --out "$OUT/result_aug.json" --workdir "$RUNDIR/aug_wd"

echo "=== A/B SUMMARY (control vs cross_family_recomb) ==="
$PIXI_RUN python - "$OUT/result_control.json" "$OUT/result_aug.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1])); a = json.load(open(sys.argv[2]))
def g(r, sp, t, k): return r["metrics"].get(sp, {}).get(t, {}).get(k)
print(f"{'metric':26}{'control':>10}{'aug':>10}{'delta':>10}")
for name, sp, t, k in [("ID next-step top1","id","nextstep","top1"),
                       ("ID next-step top5","id","nextstep","top5"),
                       ("ID completion tok","id","completion","token_acc"),
                       ("OOD next-step top1","ood","nextstep","top1"),
                       ("OOD next-step top5","ood","nextstep","top5"),
                       ("OOD completion tok","ood","completion","token_acc")]:
    cv, av = g(c,sp,t,k), g(a,sp,t,k)
    d = (av-cv) if isinstance(cv,(int,float)) and isinstance(av,(int,float)) else None
    print(f"{name:26}{cv!s:>10}{av!s:>10}{('%+.4f'%d) if d is not None else '-':>10}")
print("per-fold OOD top1 control:", {k:v['nextstep']['top1'] for k,v in c['metrics']['ood_per_fold'].items()})
print("per-fold OOD top1 aug    :", {k:v['nextstep']['top1'] for k,v in a['metrics']['ood_per_fold'].items()})
print("aug verdict:", a.get("verdict"), "| aug vs prevbest OOD:", a.get("vs_baseline",{}).get("prevbest_ood_top1_delta"))
PY
echo "=== DONE job $JOB | $(date) ==="