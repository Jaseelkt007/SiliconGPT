#!/usr/bin/env python3
"""h7 diagnostic — is validator-guided constrained decoding worth building?

For each OOD next-step example (held-out family), get the model's greedy argmax. Among the
TOP-1 ERRORS, bucket them:
  - recoverable : argmax introduces a NEW grammar violation vs the prefix (masking it would
                  remove a wrong candidate -> the correct, valid step can win) AND the true
                  step is valid (it always is) -> constrained decoding can fix these.
  - valid-but-wrong : argmax is grammar-valid -> masking can't help; needs better learning.
Also reports, for recoverable errors, whether the TRUE step is within the model's top-5
(i.e. reachable once invalid candidates are masked).

High recoverable fraction OOD (and low in-dist) => build the constrained decoder (h7).
Runs on existing checkpoints, CPU-friendly. Reuses load_checkpoint + rank_next_steps + validate_sequence.

  python scripts/diag_constrained.py --ckpt checkpoints/ood_ic/best.pt --family ic
  python scripts/diag_constrained.py --all-folds            # ic/igbt/mosfet + in-dist control
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from process_logic.generate import load_checkpoint, rank_next_steps   # noqa: E402
from process_logic.generation import validate_sequence                # noqa: E402


def read_rows(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def new_violation(prefix, step):
    """True if appending `step` to `prefix` introduces a rule violation not already present.
    Violation fields: rule, description, step_index, step_name (no `.message`)."""
    before = {(v.rule, v.step_index) for v in validate_sequence(prefix)}
    after = {(v.rule, v.step_index) for v in validate_sequence(prefix + [step])}
    return len(after - before) > 0


def diagnose(ckpt, family, device, n_max=None, topk=5):
    model, vocab = load_checkpoint(ckpt, device)
    rows = [r for r in read_rows(ROOT / "data/eval_nextstep.csv") if r["FAMILY"] == family]
    if n_max:
        rows = rows[:n_max]
    n = errors = recoverable = recov_true_invalid = valid_wrong = recov_true_in_top5 = 0
    for r in rows:
        prefix = r["PARTIAL_SEQUENCE"].split("|")
        true = r["TRUE_NEXT_STEP"]
        ranks = rank_next_steps(model, vocab, prefix, k=topk, device=device)
        if not ranks:
            continue
        n += 1
        argmax = ranks[0]
        if argmax == true:
            continue                       # correct top-1, not an error
        errors += 1
        if new_violation(prefix, argmax):  # argmax is grammar-invalid given prefix
            recoverable += 1
            # would the correct step survive masking? it's valid by construction; is it reachable (top-5)?
            if true in ranks:
                recov_true_in_top5 += 1
        else:
            valid_wrong += 1
    return {"family": family, "n": n, "errors": errors,
            "err_rate": round(errors / max(1, n), 4),
            "recoverable": recoverable,
            "recoverable_frac_of_errors": round(recoverable / max(1, errors), 4),
            "recov_true_in_top5": recov_true_in_top5,
            "valid_but_wrong": valid_wrong,
            "valid_wrong_frac_of_errors": round(valid_wrong / max(1, errors), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt"); ap.add_argument("--family")
    ap.add_argument("--all-folds", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-max", type=int, default=None)
    a = ap.parse_args()

    jobs = []
    if a.all_folds:
        for fam in ["ic", "igbt", "mosfet"]:
            jobs.append((ROOT / f"checkpoints/ood_{fam}/best.pt", fam, f"OOD/{fam}"))
        jobs.append((ROOT / "checkpoints/best.pt", "ic", "INDIST-control/ic"))  # full model on ic
    else:
        jobs.append((Path(a.ckpt), a.family, a.family))

    print(f"{'split':22}{'n':>6}{'err%':>8}{'recov%err':>11}{'recov∈top5':>12}{'validwrong%':>13}")
    agg = {"recoverable": 0, "errors": 0}
    for ckpt, fam, label in jobs:
        if not Path(ckpt).exists():
            print(f"{label:22}  (missing {ckpt})"); continue
        d = diagnose(ckpt, fam, a.device, a.n_max)
        print(f"{label:22}{d['n']:6}{d['err_rate']*100:8.1f}{d['recoverable_frac_of_errors']*100:11.1f}"
              f"{d['recov_true_in_top5']:12}{d['valid_wrong_frac_of_errors']*100:13.1f}")
        if label.startswith("OOD"):
            agg["recoverable"] += d["recoverable"]; agg["errors"] += d["errors"]
    if agg["errors"]:
        print(f"\nOOD overall recoverable fraction of errors: "
              f"{agg['recoverable']/agg['errors']*100:.1f}%  "
              f"({agg['recoverable']}/{agg['errors']})")
        print("VERDICT:", "BUILD constrained decoder (high recoverable)" if agg['recoverable']/agg['errors'] > 0.25
              else "SKIP — most OOD errors are grammar-valid-but-wrong (masking won't help)")


if __name__ == "__main__":
    main()
