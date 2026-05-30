#!/bin/bash
# Leonardo (CINECA) Slurm job — co-scientist-lab Tier-2 experiment on 1 A100.
# Runs a clean A/B with the CURRENT code (no pre-knob confound):
#   CONTROL    = baseline config (no augmentation)            -> result_control.json
#   TREATMENT  = + data.augmentation=cross_family_recomb      -> result_aug.json
# Each is a full 3-fold OOD run_experiment.py (1 in-dist model + 3 held-out-family models),
# scoring next-step + completion in-dist AND on each held-out family + ood_detect AUROC.
# ~182s/model on an A100 => ~8 models ~= 25 min total. Submit from repo root on a login node:
#   sbatch scripts/run_experiment.sh
#SBATCH --job-name=coscilab-aug
#SBATCH --account=euhpc_d30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=slurm-coscilab-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export PATH="$HOME/.pixi/bin:$PATH"
PIXI_RUN="pixi run --manifest-path $(pwd)/pixi.toml"

JOB="${SLURM_JOB_ID:-local}"
RUNDIR="${SCRATCH}/coscilab_run_${JOB}"
mkdir -p "$RUNDIR"
OUT="extras/results/coscilab"
mkdir -p "$OUT"
echo "=== co-scientist-lab Tier-2 experiment | job $JOB | $(date) ==="
echo "RUNDIR=$RUNDIR  OUT=$OUT"

# --- preflight: CUDA must be visible (fail fast on a bad node) ---
$PIXI_RUN python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('CUDA OK:', torch.cuda.get_device_name(0), '| torch', torch.__version__)"

# --- GPU smoke: catch runtime bugs on the device BEFORE the full run (cheap) ---
echo "=== GPU SMOKE (tiny config, both arms, 1 fold) ==="
cat > "$RUNDIR/smoke_spec.json" <<'JSON'
{"id":"smoke_aug","title":"gpu smoke - cross_family_recomb",
 "overrides":{"data.augmentation":"cross_family_recomb","data.aug_ratio":0.2},
 "ood":{"folds":["ic"]},"tasks":["nextstep"]}
JSON
$PIXI_RUN python scripts/run_experiment.py --spec "$RUNDIR/smoke_spec.json" \
    --tier smoke --device cuda --out "$RUNDIR/smoke_result.json" \
    --workdir "$RUNDIR/smoke_wd" --max-eval-per-family 120
echo "GPU smoke OK"

# --- CONTROL: full 3-fold OOD, current code, no augmentation ---
echo "=== CONTROL (no augmentation) — full 3-fold OOD ==="
cat > "$RUNDIR/control_spec.json" <<'JSON'
{"id":"control_v2","title":"current-code baseline (no aug), 3-fold OOD",
 "overrides":{},
 "ood":{"folds":["ic","igbt","mosfet"]},
 "tasks":["nextstep","completion","ood_detect"]}
JSON
$PIXI_RUN python scripts/run_experiment.py --spec "$RUNDIR/control_spec.json" \
    --tier full --device cuda --out "$OUT/result_control.json" \
    --workdir "$RUNDIR/control_wd"

# --- TREATMENT: full 3-fold OOD, + cross-family recombination augmentation ---
echo "=== TREATMENT (cross_family_recomb) — full 3-fold OOD ==="
cat > "$RUNDIR/aug_spec.json" <<'JSON'
{"id":"aug_xfam","title":"cross_family_recomb augmentation, 3-fold OOD",
 "overrides":{"data.augmentation":"cross_family_recomb","data.aug_ratio":0.3},
 "ood":{"folds":["ic","igbt","mosfet"]},
 "tasks":["nextstep","completion","ood_detect"]}
JSON
$PIXI_RUN python scripts/run_experiment.py --spec "$RUNDIR/aug_spec.json" \
    --tier full --device cuda --out "$OUT/result_aug.json" \
    --workdir "$RUNDIR/aug_wd"

# --- side-by-side summary ---
echo "=== A/B SUMMARY (control vs cross_family_recomb) ==="
$PIXI_RUN python - "$OUT/result_control.json" "$OUT/result_aug.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1])); a = json.load(open(sys.argv[2]))
def g(r, split, task, k): return r["metrics"].get(split, {}).get(task, {}).get(k)
print(f"{'metric':28} {'control':>10} {'aug':>10} {'delta':>10}")
rows = [("ID next-step top1", "id","nextstep","top1"),
        ("ID next-step top5", "id","nextstep","top5"),
        ("ID completion tok", "id","completion","token_acc"),
        ("OOD next-step top1","ood","nextstep","top1"),
        ("OOD next-step top5","ood","nextstep","top5"),
        ("OOD completion tok","ood","completion","token_acc")]
for name, sp, t, k in rows:
    cv, av = g(c,sp,t,k), g(a,sp,t,k)
    d = (av-cv) if (isinstance(cv,(int,float)) and isinstance(av,(int,float))) else None
    print(f"{name:28} {cv!s:>10} {av!s:>10} {('%+.4f'%d) if d is not None else '—':>10}")
print("control verdict:", c.get("verdict"), "| aug verdict:", a.get("verdict"))
print("control vs prevbest OOD:", c.get("vs_baseline",{}).get("prevbest_ood_top1_delta"),
      "| aug vs prevbest OOD:", a.get("vs_baseline",{}).get("prevbest_ood_top1_delta"))
PY
echo "=== DONE job $JOB | $(date) ==="
echo "Results: $OUT/result_control.json  $OUT/result_aug.json"