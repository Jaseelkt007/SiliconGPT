# Research overview — OOD generalization of a from-scratch process-sequence transformer

Final synthesis of the co-scientist-lab run (2 rounds + final consolidation). All figures verified from
on-disk results (`extras/results/coscilab/`, `extras/results/final3m/`).

## Goal
Close the in-distribution→OOD gap for a small from-scratch decoder on semiconductor process recipes, where
OOD = next-step prediction on a held-out 4th product family (3-fold proxy). Baseline 25M: in-dist top-1
0.807, 3-fold OOD top-1 0.4947.

## Top direction (selected + shipped): reduce model capacity
**Rationale.** In-distribution is saturated (a trigram nearly ties the 25M model), so surplus capacity is
spent memorising per-family co-occurrence shortcuts that don't transfer. Less capacity ⇒ the model is
forced onto the family-agnostic grammar the unseen family also obeys.
**Evidence.** Monotonic OOD scaling curve 25M→1.4M (0.4947→0.5139), zero in-dist cost. The 3M Pareto point,
seed-confirmed, gives 3-fold OOD **0.5163 ± 0.0017** (+0.022) with in-dist *up* to 0.821.
**Shipped.** 3M (3L/192, NoPE), all families — the deliverable. 8.4× smaller than V1, better everywhere.

## Falsified directions (high-confidence negatives)
- **Embedding placement** (description-init) — −0.018 OOD.
- **Data augmentation** (cross-family recombination, validator-filtered) — −0.018 OOD.
- **Positional encoding as an OOD driver** — NoPE neutral (+0.004); NoPE+aug −0.009. (NoPE won the final
  3M A/B as a mild val tie-break, but it is not what moves OOD.)
- **Inference-time validity constraints** (constrained decoding) — ~97% of OOD errors are valid-but-wrong,
  so masking invalid steps cannot help.
- **Structural capacity reduction** (Universal-Transformer weight-sharing) — −0.009; tying layers ≠
  shrinking size.

## The core finding (mechanistic)
The OOD failure is **not** about validity, embeddings, position, decoding, or weight-tying. Out of
distribution the model still emits a *legal* step ~100% of the time — it chooses the **wrong legal step**.
The residual is genuine **transition-structure learning** for the unseen family's ordering. Capacity
reduction recovers ~+0.02 of the ~0.31 gap; the remainder is a measured, largely-irreducible frontier.

## Suggested next experiments (if the section reopens)
1. **Confirm + extend the curve** with more seeds and intermediate sizes (2M, 4M) to pin the Pareto knee.
2. **family-dropout + UNKNOWN-row conditioning** (h4) — the one round-1 idea never built; could let the
   model use family when given and degrade gracefully when held out. Honest prior: modest, given five
   negatives already.
3. **Transition-structure objectives** that directly target the "wrong legal step" failure (e.g. a
   contrastive loss aligning same-stage states across families, h6) — higher-effort, higher-variance.
4. A **larger, more diverse training distribution** of *recipe structures* (not more of the same family) —
   the only data lever not yet tried, since augmentation by recombination failed.

## Confidence tiers
- **Full-confirmed (measured, multi-seed):** capacity reduction improves OOD.
- **Full-confirmed (measured, single-seed):** the five negatives.
- **Reasoned only:** the next-experiment suggestions above.
