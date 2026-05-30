"""Aggregate the 3-fold OOD experiment into the REPORT.md table (baseline vs description-init).

Baseline runs write to extras/results/ood_<fam>/; description-init runs (EMB_INIT set) write to
extras/results/ood_<fam>_desc/. Each fold trained WITHOUT one family and predicted on the full eval
set, so the held-out family's row is that fold's OOD result.

  pixi run python scripts/ood_summary.py
Then copy the printed markdown table straight into REPORT.md (Task 4 section).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from process_logic.score import read_rows, score_nextstep  # noqa: E402

GT = ROOT / "data" / "eval_nextstep.csv"
FOLDS = ["ic", "igbt", "mosfet"]


def held_out(pred_dir, fam, gt):
    p = ROOT / "extras" / "results" / pred_dir / "nextstep.csv"
    return score_nextstep(read_rows(p), gt).get(fam) if p.exists() else None


def c(m, key):
    return f"{m[key]:.3f}" if m else "_TBD_"


def avg(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def main():
    gt = read_rows(GT)
    rows = []
    for fam in FOLDS:
        base = held_out(f"ood_{fam}", fam, gt)
        desc = held_out(f"ood_{fam}_desc", fam, gt)
        delta = f"{desc['top1'] - base['top1']:+.3f}" if (base and desc) else "_TBD_"
        rows.append((fam, base, desc, delta))

    print("\n| held-out family | baseline top-1 | +desc-init top-1 | Δ | baseline top-5 | +desc-init top-5 |")
    print("|---|---|---|---|---|---|")
    for fam, base, desc, delta in rows:
        print(f"| {fam:14} | {c(base,'top1')} | {c(desc,'top1')} | {delta} | {c(base,'top5')} | {c(desc,'top5')} |")

    b1 = avg([r[1]['top1'] if r[1] else None for r in rows])
    d1 = avg([r[2]['top1'] if r[2] else None for r in rows])
    b5 = avg([r[1]['top5'] if r[1] else None for r in rows])
    d5 = avg([r[2]['top5'] if r[2] else None for r in rows])
    dd = avg([r[2]['top1'] - r[1]['top1'] if (r[1] and r[2]) else None for r in rows])
    fmt = lambda x: f"{x:.3f}" if x is not None else "_TBD_"
    print(f"| **3-fold avg**  | {fmt(b1)} | {fmt(d1)} | {(f'{dd:+.3f}' if dd is not None else '_TBD_')} | {fmt(b5)} | {fmt(d5)} |")

    print("\nper-fold detail (top1 / top3 / top5 / MRR):")
    for fam, base, desc, _ in rows:
        for tag, m in (("baseline", base), ("desc-init", desc)):
            if m:
                print(f"  {fam:7} {tag:9} {m['top1']:.3f} {m['top3']:.3f} {m['top5']:.3f} {m['mrr']:.3f}")


if __name__ == "__main__":
    main()
