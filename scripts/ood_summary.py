"""Aggregate the 3-fold OOD experiment into one table + average.

Each fold trained WITHOUT one family and predicted on the full eval set; the held-out
family's row is that fold's OOD result. Run after the 3 jobs finish:
  pixi run python scripts/ood_summary.py
"""
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from process_logic.score import read_rows, score_nextstep  # noqa: E402

GT = ROOT / "data" / "eval_nextstep.csv"
FOLDS = ["ic", "igbt", "mosfet"]


def main():
    gt = read_rows(GT)
    print(f"{'held-out (OOD)':16} {'top1':>7} {'top3':>7} {'top5':>7} {'mrr':>7}")
    got = []
    for fam in FOLDS:
        pred = ROOT / "extras" / "results" / f"ood_{fam}" / "nextstep.csv"
        if not pred.exists():
            print(f"{fam:16} (missing {pred} — run scripts/run_ood_3fold.sh)")
            continue
        m = score_nextstep(read_rows(pred), gt).get(fam)
        if not m:
            print(f"{fam:16} (no '{fam}' rows in ground truth)")
            continue
        got.append(m)
        print(f"{fam:16} {m['top1']:7.3f} {m['top3']:7.3f} {m['top5']:7.3f} {m['mrr']:7.3f}")
    if len(got) > 1:
        print("-" * 48)
        print(f"{'3-fold average':16} "
              f"{statistics.mean(m['top1'] for m in got):7.3f} "
              f"{statistics.mean(m['top3'] for m in got):7.3f} "
              f"{statistics.mean(m['top5'] for m in got):7.3f} "
              f"{statistics.mean(m['mrr'] for m in got):7.3f}")


if __name__ == "__main__":
    main()
