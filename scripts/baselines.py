#!/usr/bin/env python3
"""N-gram (Markov) baseline — the trivial floor of the ladder.

Trains an order-3 backoff model on train_pool and emits the SAME nextstep/completion CSVs
(scored by src/process_logic/score.py). No torch, no API — runs in seconds on the FULL eval set.
This makes the "we beat a trigram" claim reproducible (it was ad-hoc before).

  python scripts/baselines.py
  python src/process_logic/score.py --pred-dir extras/results/baseline_ngram --gt-dir data
"""
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from process_logic.dataset import load_compact_csv  # noqa: E402


def train_ngrams(seqs, K=3):
    models = {k: defaultdict(Counter) for k in range(K + 1)}
    for _, s in seqs:
        s2 = ["<BOS>"] + s
        for i in range(1, len(s2)):
            for k in range(K + 1):
                models[k][tuple(s2[max(0, i - k):i])][s2[i]] += 1
    return models


def topk(models, context, k=5):
    for order in sorted(models, reverse=True):           # back off high -> low order
        c = models[order].get(tuple(context[-order:]) if order else ())
        if c:
            return [w for w, _ in c.most_common(k)]
    return []


def main():
    train = load_compact_csv(ROOT / "data" / "train_pool.csv")
    models = train_ngrams(train, K=3)
    out = ROOT / "extras" / "results" / "baseline_ngram"
    out.mkdir(parents=True, exist_ok=True)

    # next-step: top-5 from the backoff model
    rows = list(csv.DictReader(open(ROOT / "data" / "eval_nextstep.csv", encoding="utf-8")))
    with open(out / "nextstep.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["EXAMPLE_ID", "RANK_1", "RANK_2", "RANK_3", "RANK_4", "RANK_5"])
        for r in rows:
            ctx = ["<BOS>"] + r["PARTIAL_SEQUENCE"].split("|")
            ranks = (topk(models, ctx, 5) + [""] * 5)[:5]
            w.writerow([r["EXAMPLE_ID"], *ranks])

    # completion: greedy argmax until SHIP LOT (or 80 steps)
    rows = list(csv.DictReader(open(ROOT / "data" / "eval_completion.csv", encoding="utf-8")))
    with open(out / "completion.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["EXAMPLE_ID", "PREDICTED_SEQUENCE"])
        for r in rows:
            ctx = ["<BOS>"] + r["PARTIAL_SEQUENCE"].split("|")
            gen = []
            for _ in range(80):
                nx = topk(models, ctx, 1)
                if not nx or nx[0] == "<BOS>":
                    break
                gen.append(nx[0]); ctx.append(nx[0])
                if nx[0] == "SHIP LOT":
                    break
            w.writerow([r["EXAMPLE_ID"], "|".join(gen)])

    print(f"wrote {out} (nextstep + completion on the full eval set)")


if __name__ == "__main__":
    main()
