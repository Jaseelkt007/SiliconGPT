#!/bin/bash
# ---------------------------------------------------------------------------
# Quick CPU demo: prediction + scoring on a TINY subset of the eval inputs.
# Finishes in seconds on a single CPU (no GPU needed). For live demos / smoke
# checks that the predict -> score pipeline works end-to-end.
#
#   Usage:   bash scripts/run_demo.sh
#
# Outputs -> extras/test_folder/{nextstep,completion,anomaly}.csv + score_demo.txt
# Uses the committed deliverable checkpoint (checkpoints/best.pt, the 1.37M model).
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."          # repo root, regardless of where it's called from

CKPT="checkpoints/best.pt"
OUT="extras/test_folder"
mkdir -p "$OUT"

# --- pick a Python runner ---------------------------------------------------
# pixi (our env, on Leonardo / a clean checkout that ran setup_leonardo.sh),
# else fall back to whatever `python` is on PATH (e.g. after pip install -r requirements.txt).
if [ -x "$HOME/.pixi/bin/pixi" ] || command -v pixi >/dev/null 2>&1; then
    export PATH="$HOME/.pixi/bin:$PATH"
    RUN="pixi run python"
else
    RUN="python"
fi
echo "runner: $RUN"

# --- ensure the tiny demo subsets exist (data/ is gitignored/regenerable) ---
# If absent, carve them from the full eval files. Run build_datasets.py first
# if those are missing too.
need_full() { [ -f "data/eval_$1.csv" ] || { echo "ERROR: data/eval_$1.csv missing — run 'python scripts/build_datasets.py' first." >&2; exit 1; }; }

# NOTE: carving uses pure awk line-limiting (no `head` on a pipe) so it is safe
# under `set -o pipefail` — `head` closing a pipe early would SIGPIPE awk/tail.
if [ ! -f data/eval_nextstep_demo.csv ]; then
    need_full nextstep
    echo "carving data/eval_nextstep_demo.csv (2 per family)"
    awk -F, 'NR==1 {print; next} $2=="mosfet" && ++m<=2; $2=="igbt" && ++i<=2; $2=="ic" && ++c<=2' \
        data/eval_nextstep.csv > data/eval_nextstep_demo.csv
fi
if [ ! -f data/eval_completion_demo.csv ]; then
    need_full completion
    echo "carving data/eval_completion_demo.csv (5 examples)"
    awk 'NR<=6' data/eval_completion.csv > data/eval_completion_demo.csv
fi
if [ ! -f data/eval_anomaly_demo.csv ]; then
    need_full anomaly
    echo "carving data/eval_anomaly_demo.csv (3 valid + 3 invalid)"
    awk -F, 'NR==1 {print; next} $4==1 && ++v<=3; $4==0 && ++n<=3' \
        data/eval_anomaly.csv > data/eval_anomaly_demo.csv
fi

# --- 1) predict on the demo subset (CPU) ------------------------------------
echo "=== PREDICT (demo subset, CPU) ==="
$RUN src/process_logic/predict.py \
    --ckpt "$CKPT" --device cpu --out-dir "$OUT" \
    --nextstep-input   data/eval_nextstep_demo.csv \
    --completion-input data/eval_completion_demo.csv \
    --anomaly-input    data/eval_anomaly_demo.csv \
    --calib-file       data/val_id.csv

# --- 2) score (filter full ground truth down to the demo IDs via --intersect)
echo "=== SCORE (next-step / completion / anomaly, per family) ==="
$RUN src/process_logic/score.py \
    --pred-dir "$OUT" --gt-dir data --intersect | tee "$OUT/score_demo.txt"

echo "=== DONE -> $OUT/{nextstep,completion,anomaly}.csv + score_demo.txt ==="
