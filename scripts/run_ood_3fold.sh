#!/bin/bash
# Submit the 3-fold OOD experiment: hold out each family once (ic, igbt, mosfet).
#   bash scripts/run_ood_3fold.sh                        # baseline (random init)
#   EMB_INIT=emb_init.npz bash scripts/run_ood_3fold.sh  # with description-init
# When all 3 jobs finish:  pixi run python scripts/ood_summary.py
set -euo pipefail
cd "$(dirname "$0")/.."

for fam in ic igbt mosfet; do
    if [ -n "${EMB_INIT:-}" ]; then
        sbatch --export=ALL,EXCLUDE="$fam",EMB_INIT="$EMB_INIT" scripts/run_ood.sh
    else
        sbatch --export=ALL,EXCLUDE="$fam" scripts/run_ood.sh
    fi
done
echo "Submitted 3 OOD jobs (held out: ic, igbt, mosfet). Watch with: squeue --me"
echo "After they finish:  pixi run python scripts/ood_summary.py"
