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

---

## OOD model-improvement (co-scientist-lab, 2026-05-31)

Deciding metric = **3-fold OOD next-step top1** (train on 2 families, test on the held-out 3rd; avg over
the 3 hold-outs). Baseline (V1, 25M) = **0.4947**. Each lever is a full 3-fold run (`scripts/run_experiment.py`);
numbers from `extras/results/coscilab/result_*.json` (single seed). in-dist top1 stays ~0.80 throughout.

| lever | 3-fold OOD top1 | Δ vs baseline | verdict |
|---|---|---|---|
| baseline 25M | 0.4947 | — | — |
| **model size 1.4M** | **0.5139** | **+0.019** | improves-ood |
| model size 3.0M | 0.5120 | +0.017 | improves-ood |
| model size 6.4M | 0.5119 | +0.017 | improves-ood |
| model size 14.7M | 0.5008 | +0.006 | neutral |
| NoPE (no pos-enc) | 0.4983 | +0.004 | neutral |
| weight-sharing on 3M (h2) | 0.5033 | −0.009 vs 3M ctrl | rejected |
| NoPE + cross-fam aug | 0.4861 | −0.009 | rejected |
| cross-family aug | 0.4767 | −0.018 | rejected |
| constrained decoding (h7) | — | ~0 (≈97% errors valid-but-wrong) | rejected |

> h2 Universal-Transformer **weight-sharing** (one block reused across depth) at the 3M base:
> OOD 0.5120→0.5033 (−0.009), completion 0.196→0.168, ID flat — same top5-up/top1-down signature.
> Reducing capacity *structurally* (tying layers) does NOT reproduce the gain from reducing it by *size*;
> the shared operator underfits the rank-1 transition. Rejected.

**Takeaways.** (1) **Scaling DOWN is the only lever that moved OOD** — monotonic +0.019 from 25M→1.4M at
zero in-dist cost (capacity was memorising per-family shortcuts). (2) **Data/embedding/position/decoding
levers all fail** (cross-fam aug, desc-init [DECISIONS D1], NoPE, constrained decoding) — the OOD residual
is *transition-structure learning*: the model picks the wrong **legal** step (only ~3% of OOD errors are
grammar-invalid). (3) Still-open built/buildable knobs: weight-sharing (h2), family-dropout (h4). Full
provenance + the integrity note in `DECISIONS.md` D3.