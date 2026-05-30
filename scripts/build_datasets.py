#!/usr/bin/env python3
"""
Build all training/eval datasets for the process-logic model.

Uses the vendored grammar/generator (src/process_logic/generation.py) which
exposes generate_sequence, generate_dataset, validate_sequence and the rule
constants. Every anomalous (corrupted) sequence is re-checked with
validate_sequence so its IS_VALID / RULE_VIOLATED labels are guaranteed correct.

Outputs (under --out, default data/):
  train_pool.csv      SEQUENCE_ID, FAMILY, SEQUENCE            (valid, balanced)
  val_id.csv          SEQUENCE_ID, FAMILY, SEQUENCE            (in-distribution val)
  ood_holdout.csv     SEQUENCE_ID, FAMILY, SEQUENCE            (one held-out family)
  eval_nextstep.csv   EXAMPLE_ID, FAMILY, PARTIAL_SEQUENCE, TRUE_NEXT_STEP
  eval_completion.csv EXAMPLE_ID, FAMILY, COMPLETION_FRACTION, PARTIAL_SEQUENCE, TRUE_SUFFIX
  eval_anomaly.csv    EXAMPLE_ID, FAMILY, SEQUENCE, IS_VALID, RULE_VIOLATED
  anomaly_train.csv   EXAMPLE_ID, FAMILY, SEQUENCE, IS_VALID, RULE_VIOLATED
"""
from __future__ import annotations
import argparse
import csv
import random
import sys
from pathlib import Path

# import the vendored grammar/generator
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import process_logic.generation as G  # noqa: E402

FAMILIES = ["mosfet", "igbt", "ic"]
RULES = [
    "RULE_DEP_NO_CLEAN", "RULE_METAL_ETCH_NO_LITHO", "RULE_ETCH_NO_MASK",
    "RULE_LITHO_LEVEL_SKIP", "RULE_IMPLANT_NO_MASK", "RULE_CMP_NO_DEP",
    "RULE_PAD_OPEN_BEFORE_DEP", "RULE_TEST_BEFORE_PASSIVATION",
    "RULE_SHIP_BEFORE_TEST", "RULE_BACKSIDE_BEFORE_PASSIVATION",
]
DEVELOP_STEPS = frozenset({"DEVELOP PHOTORESIST", "DEVELOP PAD WINDOW"})
PASSIVATION_DEPOSIT = frozenset({"DEPOSIT PASSIVATION", "DEPOSIT PASSIVATION LAYER"})
CURE = "CURE PASSIVATION"


# --------------------------------------------------------------------------- #
# Generation with dedup shared across splits (prevents train/val/ood overlap)  #
# --------------------------------------------------------------------------- #
def gen_unique(family, n, rng, seen, validate=True):
    out = []
    attempts = 0
    cap = n * 60 + 1000
    while len(out) < n and attempts < cap:
        attempts += 1
        seq = G.generate_sequence(family, rng)
        key = tuple(seq)
        if key in seen:
            continue
        if validate and G.validate_sequence(seq):
            continue
        seen.add(key)
        out.append(seq)
    if len(out) < n:
        print(f"  [WARN] {family}: only {len(out)}/{n} unique sequences", file=sys.stderr)
    return out


# --------------------------------------------------------------------------- #
# Corruption helpers — each returns a mutated copy or None                      #
# --------------------------------------------------------------------------- #
def _move(steps, i, j):
    s = list(steps)
    el = s.pop(i)
    if i < j:
        j -= 1
    s.insert(max(0, j), el)
    return s


def _drop_window(steps, idx, win, targets):
    drop = {k for k in range(max(0, idx - win), idx) if steps[k] in targets}
    if not drop:
        return None
    return [s for k, s in enumerate(steps) if k not in drop]


def _idxs(steps, member):
    return [i for i, s in enumerate(steps) if s in member]


def c_dep_no_clean(steps, rng):
    for d in _shuf(_idxs(steps, G.DEPOSITION_STEPS), rng):
        out = _drop_window(steps, d, 12, G.CLEAN_STEPS)
        if out:
            return out
    return None


def c_etch_no_mask(steps, rng):
    for e in _shuf(_idxs(steps, G.ETCH_STEPS), rng):
        out = _drop_window(steps, e, 12, DEVELOP_STEPS)
        if out:
            return out
    return None


def c_metal_etch_no_litho(steps, rng):
    targets = DEVELOP_STEPS | {s for s in steps if s.startswith("EXPOSE LITHO LEVEL")}
    for m in _shuf(_idxs(steps, G.METAL_ETCH_STEPS), rng):
        out = _drop_window(steps, m, 15, targets)
        if out:
            return out
    return None


def c_implant_no_mask(steps, rng):
    for i in _shuf(_idxs(steps, G.IMPLANT_STEPS), rng):
        out = _drop_window(steps, i, 15, G.IMPLANT_OPENER_STEPS)
        if out:
            return out
    return None


def c_cmp_no_dep(steps, rng):
    for c in _shuf(_idxs(steps, G.CMP_STEPS), rng):
        out = _drop_window(steps, c, 6, G.FILL_STEPS)
        if out:
            return out
    return None


def c_litho_level_skip(steps, rng):
    aligns = [(i, int(s.split("ALIGN MASK LEVEL ")[1]))
              for i, s in enumerate(steps)
              if s.startswith("ALIGN MASK LEVEL ") and s.split("ALIGN MASK LEVEL ")[1].isdigit()]
    if len(aligns) < 2:
        return None
    # relabel two align steps so a higher level appears before a lower one
    (pa, la), (pb, lb) = aligns[0], aligns[1]
    if la == lb:
        return None
    s = list(steps)
    s[pa] = f"ALIGN MASK LEVEL {max(la, lb)}"
    s[pb] = f"ALIGN MASK LEVEL {min(la, lb)}"
    return s


def c_pad_open_before_dep(steps, rng):
    pads = _idxs(steps, G.PAD_WINDOW_STEPS)
    deps = _idxs(steps, PASSIVATION_DEPOSIT)
    if not pads or not deps:
        return None
    return _move(steps, pads[0], deps[0])  # move pad-open before passivation deposit


def c_test_before_passivation(steps, rng):
    if CURE not in steps:
        return None
    cp = steps.index(CURE)
    tests = [i for i in _idxs(steps, G.ELECTRICAL_TEST_STEPS) if i > cp]
    if not tests:
        return None
    return _move(steps, tests[0], cp)  # move a test before CURE PASSIVATION


def c_ship_before_test(steps, rng):
    if "SHIP LOT" not in steps or "WAFER SORT TEST" not in steps:
        return None
    return _move(steps, steps.index("SHIP LOT"), steps.index("WAFER SORT TEST"))


def c_backside_before_passivation(steps, rng):
    if "DEPOSIT BACKSIDE METAL" not in steps or CURE not in steps:
        return None
    return _move(steps, steps.index("DEPOSIT BACKSIDE METAL"), steps.index(CURE))


def _shuf(lst, rng):
    lst = list(lst)
    rng.shuffle(lst)
    return lst


CORRUPTORS = {
    "RULE_DEP_NO_CLEAN": c_dep_no_clean,
    "RULE_ETCH_NO_MASK": c_etch_no_mask,
    "RULE_METAL_ETCH_NO_LITHO": c_metal_etch_no_litho,
    "RULE_IMPLANT_NO_MASK": c_implant_no_mask,
    "RULE_CMP_NO_DEP": c_cmp_no_dep,
    "RULE_LITHO_LEVEL_SKIP": c_litho_level_skip,
    "RULE_PAD_OPEN_BEFORE_DEP": c_pad_open_before_dep,
    "RULE_TEST_BEFORE_PASSIVATION": c_test_before_passivation,
    "RULE_SHIP_BEFORE_TEST": c_ship_before_test,
    "RULE_BACKSIDE_BEFORE_PASSIVATION": c_backside_before_passivation,
}


def corrupt(steps, rule, rng):
    """Return a corrupted sequence that violates `rule` (verified), or None."""
    fn = CORRUPTORS[rule]
    out = fn(steps, rng)
    if out is None:
        return None
    rules_hit = {v.rule for v in G.validate_sequence(out)}
    return out if rule in rules_hit else None


# --------------------------------------------------------------------------- #
# Writers                                                                       #
# --------------------------------------------------------------------------- #
def write_compact(path, fam_seqs):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["SEQUENCE_ID", "FAMILY", "SEQUENCE"])
        for fam, seqs in fam_seqs:
            for i, seq in enumerate(seqs, 1):
                w.writerow([f"{fam}_{i:06d}", fam, "|".join(seq)])


def build_anomaly(pool, n_valid, n_per_rule, rng, tag):
    """pool: list of (family, steps). Returns list of rows."""
    rows = []
    rng.shuffle(pool)
    for fam, steps in pool[:n_valid]:
        rows.append([fam, "|".join(steps), 1, ""])
    counts = {r: 0 for r in RULES}
    i = 0
    guard = 0
    while any(counts[r] < n_per_rule for r in RULES) and guard < len(pool) * 200:
        guard += 1
        rule = RULES[i % len(RULES)]
        i += 1
        if counts[rule] >= n_per_rule:
            continue
        fam, steps = pool[rng.randrange(len(pool))]
        bad = corrupt(steps, rule, rng)
        if bad is None:
            continue
        rows.append([fam, "|".join(bad), 0, rule])
        counts[rule] += 1
    rng.shuffle(rows)
    out = []
    for n, r in enumerate(rows, 1):
        out.append([f"{tag}_{n:06d}", r[0], r[1], r[2], r[3]])
    return out, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-train", type=int, default=20000, help="per family")
    ap.add_argument("--n-val", type=int, default=4000, help="per family")
    ap.add_argument("--ood-family", default="ic")
    ap.add_argument("--n-ood", type=int, default=4000)
    ap.add_argument("--n-eval-pool", type=int, default=400, help="held-out per family for eval files")
    ap.add_argument("--anom-valid", type=int, default=600)
    ap.add_argument("--anom-per-rule", type=int, default=40)
    ap.add_argument("--anom-train-valid", type=int, default=8000)
    ap.add_argument("--anom-train-per-rule", type=int, default=800)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    seen = set()

    # 1) train pool (balanced, all families)
    train = [(fam, gen_unique(fam, args.n_train, rng, seen)) for fam in FAMILIES]
    write_compact(out / "train_pool.csv", train)

    # 2) in-distribution val (held out from train)
    val = [(fam, gen_unique(fam, args.n_val, rng, seen)) for fam in FAMILIES]
    write_compact(out / "val_id.csv", val)

    # 3) OOD holdout (one family, distinct sequences)
    ood = gen_unique(args.ood_family, args.n_ood, rng, seen)
    write_compact(out / "ood_holdout.csv", [(args.ood_family, ood)])

    # 4) eval pool (held out, used to build eval_* with ground truth)
    eval_pool = {fam: gen_unique(fam, args.n_eval_pool, rng, seen) for fam in FAMILIES}

    # eval_nextstep: 3 random cut points per sequence
    with (out / "eval_nextstep.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["EXAMPLE_ID", "FAMILY", "PARTIAL_SEQUENCE", "TRUE_NEXT_STEP"])
        n = 0
        for fam in FAMILIES:
            for seq in eval_pool[fam]:
                for _ in range(3):
                    k = rng.randint(8, len(seq) - 2)
                    n += 1
                    w.writerow([f"ns_{n:06d}", fam, "|".join(seq[:k]), seq[k]])

    # eval_completion: cut at 0.6 and 0.8, 100 seqs/family/cut
    with (out / "eval_completion.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["EXAMPLE_ID", "FAMILY", "COMPLETION_FRACTION", "PARTIAL_SEQUENCE", "TRUE_SUFFIX"])
        n = 0
        for fam in FAMILIES:
            for seq in eval_pool[fam][:100]:
                for frac in (0.6, 0.8):
                    k = int(len(seq) * frac)
                    n += 1
                    w.writerow([f"cp_{n:06d}", fam, frac, "|".join(seq[:k]), "|".join(seq[k:])])

    # eval_anomaly + anomaly_train (from held-out eval pool to avoid leakage)
    flat_pool = [(fam, s) for fam in FAMILIES for s in eval_pool[fam]]
    anom_rows, anom_counts = build_anomaly(flat_pool, args.anom_valid, args.anom_per_rule, rng, "an")
    _write_anom(out / "eval_anomaly.csv", anom_rows)

    train_flat = [(fam, s) for fam, seqs in train for s in seqs]
    atr_rows, atr_counts = build_anomaly(train_flat, args.anom_train_valid, args.anom_train_per_rule, rng, "atr")
    _write_anom(out / "anomaly_train.csv", atr_rows)

    # summary
    print("\n=== SUMMARY ===")
    for fam, seqs in train:
        print(f"  train_pool   {fam:7} {len(seqs):>7}")
    for fam, seqs in val:
        print(f"  val_id       {fam:7} {len(seqs):>7}")
    print(f"  ood_holdout  {args.ood_family:7} {len(ood):>7}")
    print(f"  eval_anomaly rows={len(anom_rows)} per-rule={anom_counts}")
    print(f"  anomaly_train rows={len(atr_rows)} per-rule={atr_counts}")


def _write_anom(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["EXAMPLE_ID", "FAMILY", "SEQUENCE", "IS_VALID", "RULE_VIOLATED"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
