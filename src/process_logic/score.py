"""Local scorer for the three tasks (pure stdlib — no torch, no sklearn).

Joins our prediction CSVs (from predict.py) with the ground-truth eval files and
reports per-family + overall metrics. The organizers' eval_metrics.py is the
official scorer (and adds Block-level Accuracy); this lets us measure progress
before/without it.

  python src/process_logic/score.py --pred-dir extras/results --gt-dir data
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


# ---------- helpers ----------
def read_rows(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def index_by_id(rows):
    return {r["EXAMPLE_ID"]: r for r in rows}


def edit_distance(a, b):
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


def roc_auc(scores, labels):
    """AUC with positive class = 1. Rank-based (Mann-Whitney), handles ties."""
    n = len(scores)
    npos = sum(labels)
    nneg = n - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    order = sorted(range(n), key=lambda i: scores[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    sum_pos = sum(ranks[i] for i in range(n) if labels[i] == 1)
    return (sum_pos - npos * (npos + 1) / 2.0) / (npos * nneg)


# ---------- Task 1: next-step ----------
def score_nextstep(pred_rows, gt_rows):
    P = index_by_id(pred_rows)
    fam = defaultdict(lambda: dict(t1=0, t3=0, t5=0, mrr=0.0, n=0))
    for g in gt_rows:
        ex, true = g["EXAMPLE_ID"], g["TRUE_NEXT_STEP"]
        ranks = [P[ex].get(f"RANK_{i}", "") for i in range(1, 6)] if ex in P else []
        rank = ranks.index(true) + 1 if true in ranks else 0
        for key in (g["FAMILY"], "ALL"):
            d = fam[key]
            d["n"] += 1
            d["t1"] += rank == 1
            d["t3"] += 1 <= rank <= 3
            d["t5"] += 1 <= rank <= 5
            d["mrr"] += (1.0 / rank) if rank else 0.0
    out = {}
    for k, d in fam.items():
        n = max(1, d["n"])
        out[k] = {"top1": d["t1"] / n, "top3": d["t3"] / n,
                  "top5": d["t5"] / n, "mrr": d["mrr"] / n, "n": d["n"]}
    return out


# ---------- Task 2: completion ----------
def score_completion(pred_rows, gt_rows):
    P = index_by_id(pred_rows)
    fam = defaultdict(lambda: dict(em=0, ned=0.0, tok=0.0, n=0))
    for g in gt_rows:
        ex = g["EXAMPLE_ID"]
        true = g["TRUE_SUFFIX"].split("|") if g["TRUE_SUFFIX"] else []
        pred = (P[ex]["PREDICTED_SEQUENCE"].split("|")
                if ex in P and P[ex]["PREDICTED_SEQUENCE"] else [])
        d_em = pred == true
        dist = edit_distance(pred, true)
        ned = dist / max(1, max(len(pred), len(true)))
        match = sum(1 for i in range(min(len(pred), len(true))) if pred[i] == true[i])
        tok = match / max(1, len(true))
        for key in (g["FAMILY"], "ALL"):
            d = fam[key]
            d["n"] += 1
            d["em"] += d_em
            d["ned"] += ned
            d["tok"] += tok
    out = {}
    for k, d in fam.items():
        n = max(1, d["n"])
        out[k] = {"exact_match": d["em"] / n, "norm_edit_dist": d["ned"] / n,
                  "token_acc": d["tok"] / n, "n": d["n"]}
    return out


# ---------- Task 3: anomaly ----------
def score_anomaly(pred_rows, gt_rows):
    """Positive class = anomaly (IS_VALID == 0)."""
    P = index_by_id(pred_rows)
    fam_rows = defaultdict(list)
    for g in gt_rows:
        ex = g["EXAMPLE_ID"]
        if ex not in P:
            continue
        gt_valid = int(g["IS_VALID"])
        pr_valid = int(P[ex]["IS_VALID"])
        score = float(P[ex].get("SCORE", 1.0))            # P(valid)
        anom_label = 1 - gt_valid                          # 1 = anomaly
        anom_score = 1.0 - score                           # higher = more anomalous
        rule_ok = (gt_valid == 0 and pr_valid == 0
                   and P[ex].get("PREDICTED_RULE", "") == g.get("RULE_VIOLATED", ""))
        rec = (gt_valid, pr_valid, anom_label, anom_score, rule_ok)
        fam_rows[g["FAMILY"]].append(rec)
        fam_rows["ALL"].append(rec)

    out = {}
    for k, rows in fam_rows.items():
        tp = sum(1 for gv, pv, *_ in rows if gv == 0 and pv == 0)
        fp = sum(1 for gv, pv, *_ in rows if gv == 1 and pv == 0)
        fn = sum(1 for gv, pv, *_ in rows if gv == 0 and pv == 1)
        tn = sum(1 for gv, pv, *_ in rows if gv == 1 and pv == 1)
        n = len(rows)
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        acc = (tp + tn) / max(1, n)
        auc = roc_auc([r[3] for r in rows], [r[2] for r in rows])
        n_detected = sum(1 for gv, pv, *_ in rows if gv == 0 and pv == 0)
        rule_acc = sum(1 for r in rows if r[4]) / max(1, n_detected)
        out[k] = {"acc": acc, "precision": prec, "recall": rec, "f1": f1,
                  "roc_auc": auc, "rule_attr": rule_acc,
                  "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn}, "n": n}
    return out


# ---------- printing / CLI ----------
def _print(title, metrics, keys):
    print(f"\n== {title} ==")
    for fam in ["ALL", "mosfet", "igbt", "ic"]:
        if fam not in metrics:
            continue
        m = metrics[fam]
        cells = " ".join(f"{k}={m[k]:.3f}" if isinstance(m[k], float) else f"{k}={m[k]}"
                         for k in keys)
        print(f"  {fam:7} n={m['n']:<5} {cells}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", default="extras/results")
    ap.add_argument("--gt-dir", default="data")
    args = ap.parse_args()
    pred, gt = Path(args.pred_dir), Path(args.gt_dir)

    if (pred / "nextstep.csv").exists():
        _print("Task 1 — Next-step",
               score_nextstep(read_rows(pred / "nextstep.csv"), read_rows(gt / "eval_nextstep.csv")),
               ["top1", "top3", "top5", "mrr"])
    if (pred / "completion.csv").exists():
        _print("Task 2 — Completion",
               score_completion(read_rows(pred / "completion.csv"), read_rows(gt / "eval_completion.csv")),
               ["exact_match", "norm_edit_dist", "token_acc"])
    if (pred / "anomaly.csv").exists():
        _print("Task 3 — Anomaly",
               score_anomaly(read_rows(pred / "anomaly.csv"), read_rows(gt / "eval_anomaly.csv")),
               ["acc", "precision", "recall", "f1", "roc_auc", "rule_attr"])


if __name__ == "__main__":
    main()
