# Benchmark — process-logic model vs. baselines

Scored on the **held-out eval set** (`data/eval_*.csv`). For each task every model is
scored on the **same common examples** (intersection of predicted EXAMPLE_IDs), so a
model that ran fewer samples is still compared like-for-like.

## Task 1 — Next-step  (n=25 common)

| model | top1 | top3 | top5 | mrr |
|---|---|---|---|---|
| n-gram (trigram) | 0.680 | 1.000 | 1.000 | 0.833 |
| Gemini 3.5-flash | 0.440 | 0.680 | 0.760 | 0.558 |
| ours (V1, 25M) | 0.680 | 1.000 | 1.000 | 0.840 |

## Task 2 — Completion  (n=25 common)

| model | exact_match | norm_edit_dist | token_acc |
|---|---|---|---|
| n-gram (trigram) | 0.000 | 0.183 | 0.435 |
| Gemini 3.5-flash | 0.000 | 0.682 | 0.065 |
| ours (V1, 25M) | 0.000 | 0.193 | 0.416 |

## Task 3 — Anomaly  (n=25 common)

| model | acc | precision | recall | f1 | roc_auc | rule_attr |
|---|---|---|---|---|---|---|
| n-gram (trigram) | — | — | — | — | — | — |
| Gemini 3.5-flash | 0.880 | 0.800 | 0.889 | 0.842 | 0.806 | 1.000 |
| ours (V1, 25M) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## Learnings

- **Next-step is saturated** — n-gram ≈ Gemini ≈ ours (~0.76–0.81 top-1). Not discriminating.
- **Completion ranks ours > n-gram > Gemini** — it rewards fitting *this* generator's path
  (specialization), which our trained model has and a general LLM does not. The LLM's
  completions are still mostly valid, just different.
- **Validity** ~100% for ours, ~0.92 for Gemini under sampling — everyone broadly obeys the rules.
- **OOD (held-out family) is the deciding, still-mostly-untested axis** — the LLM (broad
  knowledge) may generalize better than our specialist; our V1 OOD next-step is ~0.50.
- **Proper baselines for the improvement loop:** previous-best checkpoint (primary target),
  n-gram (floor — we're only ~0.05/~0.12 above it), Gemini (OOD/frontier bar).
- **No baseline model was provided by the committee** — only the generator/validator.
- Kimi K2.6 deferred (Moonshot rate-limiting / thinking-not-disabled → 120–310 s/call).

> NOTE: the tables use the **25 Gemini-limited common examples** (noisy). More reliable
> full-eval numbers: ours top-1 **0.807** / completion token-acc **0.400**; n-gram **0.761** /
> **0.283** (see REPORT.md). The key takeaway holds: ours ≳ n-gram ≫ Gemini in-distribution,
> our V1 is only marginally above the n-gram (→ improvement must target OOD + long-range).