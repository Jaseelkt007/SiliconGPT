#!/usr/bin/env python3
"""Score our model vs the LLM baselines on the SAME common examples (apples-to-apples),
using the official-mirror metrics in score.py. Emits a Markdown table + a JSON the
frontend can consume.

  python scripts/benchmark_compare.py
"""
import sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from process_logic.score import read_rows, score_nextstep, score_completion, score_anomaly  # noqa: E402

RES = ROOT / "extras" / "results"
# display name -> dir under extras/results
MODELS = {
    "Ours (1.37M)":      "final3m/submission",
    "Gemini 3.5-flash":  "baseline_gemini",
    "DeepSeek":          "baseline_featherless-deepseek",
    "Qwen":              "baseline_featherless-qwen",
    "n-gram (trigram)":  "baseline_ngram",
}
GT = {t: read_rows(ROOT / "data" / f"eval_{t}.csv") for t in ("nextstep", "completion", "anomaly")}
TASKS = [
    ("nextstep",   score_nextstep,   ["top1", "top3", "top5", "mrr"]),
    ("completion", score_completion, ["exact_match", "norm_edit_dist", "token_acc"]),
    ("anomaly",    score_anomaly,    ["acc", "precision", "recall", "f1", "roc_auc", "rule_attr"]),
]


def load(d, t):
    p = RES / d / f"{t}.csv"
    return read_rows(p) if p.exists() else None


def main():
    result = {}
    md = ["# Benchmark comparison — our model vs LLM baselines", "",
          "Each model is scored with `src/process_logic/score.py` on the **examples it actually answered**",
          "(the `n` column). Our model + n-gram ran the FULL held-out eval on the A100; the frontier LLMs",
          "were sampled on the first 200 examples (cost). All values are the real measured numbers.", ""]
    for task, scorer, cols in TASKS:
        preds = {n: load(d, task) for n, d in MODELS.items()}
        have = {n: p for n, p in preds.items() if p}
        result[task] = {"models": {}}
        md += [f"## {task.capitalize()}  (each model scored on the examples it answered — see n)", "",
               "| model | n | " + " | ".join(cols) + " |", "|---" * (len(cols) + 2) + "|"]
        for n in MODELS:
            if n not in have:
                md.append(f"| {n} | — | " + " | ".join(["—"] * len(cols)) + " |"); continue
            p = have[n]
            ids = {r["EXAMPLE_ID"] for r in p}
            gt_sub = [g for g in GT[task] if g["EXAMPLE_ID"] in ids]   # own-coverage (intersection) scoring
            m = scorer(p, gt_sub).get("ALL")
            entry = {c: round(m[c], 4) for c in cols}
            entry["n"] = m["n"]
            result[task]["models"][n] = entry
            cells = [f"{m[c]:.4f}" for c in cols]
            md.append(f"| {n} | {m['n']} | " + " | ".join(cells) + " |")
        md.append("")
    (RES / "benchmark_compare.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (RES / "benchmark_compare.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print("wrote", RES / "benchmark_compare.json", "and", RES / "benchmark_compare.md")


if __name__ == "__main__":
    main()
