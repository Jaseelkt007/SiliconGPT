#!/usr/bin/env python3
"""Collate the benchmark into extras/results/benchmark.md.

Scores every model on the SAME common held-out examples per task (intersection of the
EXAMPLE_IDs each model predicted), so it's apples-to-apples even when one model ran fewer
samples. Reusable for the improvement loop — add a model dir to MODELS and re-run.

  python scripts/benchmark_table.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from process_logic.score import read_rows, score_nextstep, score_completion, score_anomaly  # noqa: E402

RES = ROOT / "extras" / "results"
# name -> dir under extras/results ("" = root = our V1 submission CSVs)
MODELS = {
    "n-gram (trigram)": "baseline_ngram",
    "Gemini 3.5-flash": "baseline_gemini",
    "ours (V1, 25M)": "",
}
GT = {t: read_rows(ROOT / "data" / f"eval_{t}.csv") for t in ("nextstep", "completion", "anomaly")}
TASKS = [
    ("nextstep", score_nextstep, ["top1", "top3", "top5", "mrr"], "Task 1 — Next-step"),
    ("completion", score_completion, ["exact_match", "norm_edit_dist", "token_acc"], "Task 2 — Completion"),
    ("anomaly", score_anomaly, ["acc", "precision", "recall", "f1", "roc_auc", "rule_attr"], "Task 3 — Anomaly"),
]


def load(d, task):
    p = (RES / d / f"{task}.csv") if d else (RES / f"{task}.csv")
    return read_rows(p) if p.exists() else None


def main():
    out = ["# Benchmark — process-logic model vs. baselines", "",
           "Scored on the **held-out eval set** (`data/eval_*.csv`). For each task every model is",
           "scored on the **same common examples** (intersection of predicted EXAMPLE_IDs), so a",
           "model that ran fewer samples is still compared like-for-like.", ""]
    for task, scorer, cols, header in TASKS:
        preds = {n: load(d, task) for n, d in MODELS.items()}
        idsets = [{r["EXAMPLE_ID"] for r in p} for p in preds.values() if p]
        common = set.intersection(*idsets) if idsets else set()
        gt_sub = [g for g in GT[task] if g["EXAMPLE_ID"] in common]
        out += [f"## {header}  (n={len(gt_sub)} common)", "",
                "| model | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
        for name, p in preds.items():
            m = scorer(p, gt_sub).get("ALL") if p else None
            cells = [f"{m[c]:.3f}" for c in cols] if m else ["—"] * len(cols)
            out.append(f"| {name} | " + " | ".join(cells) + " |")
        out.append("")
    out += [
        "## Learnings", "",
        "- **Next-step is saturated** — n-gram ≈ Gemini ≈ ours (~0.76–0.81 top-1). Not discriminating.",
        "- **Completion ranks ours > n-gram > Gemini** — it rewards fitting *this* generator's path",
        "  (specialization), which our trained model has and a general LLM does not. The LLM's",
        "  completions are still mostly valid, just different.",
        "- **Validity** ~100% for ours, ~0.92 for Gemini under sampling — everyone broadly obeys the rules.",
        "- **OOD (held-out family) is the deciding, still-mostly-untested axis** — the LLM (broad",
        "  knowledge) may generalize better than our specialist; our V1 OOD next-step is ~0.50.",
        "- **Proper baselines for the improvement loop:** previous-best checkpoint (primary target),",
        "  n-gram (floor — we're only ~0.05/~0.12 above it), Gemini (OOD/frontier bar).",
        "- **No baseline model was provided by the committee** — only the generator/validator.",
        "- Kimi K2.6 deferred (Moonshot rate-limiting / thinking-not-disabled → 120–310 s/call).",
        "",
        "> NOTE: the tables use the **25 Gemini-limited common examples** (noisy). More reliable",
        "> full-eval numbers: ours top-1 **0.807** / completion token-acc **0.400**; n-gram **0.761** /",
        "> **0.283** (see REPORT.md). The key takeaway holds: ours ≳ n-gram ≫ Gemini in-distribution,",
        "> our V1 is only marginally above the n-gram (→ improvement must target OOD + long-range).",
    ]
    (RES / "benchmark.md").write_text("\n".join(out), encoding="utf-8")
    print("wrote", RES / "benchmark.md")
    print("\n".join(out))


if __name__ == "__main__":
    main()
