<!--
This is the SEED config for the co-scientist-lab run. At Step 1, the Supervisor copies this into
.coscientist_runs/<slug>/config.md and CONFIRMS it with the user (expert-in-the-loop checkpoint #1).
It is a starting point — the Experiment agent (and Generation/Evolution) may PROPOSE NEW tasks, metrics,
and criteria as the run discovers them (e.g. new measurable capabilities). Refine here, then confirm.
-->

# Research goal
Discover the architecture / data / training / conditioning changes that make our small (~10–50M),
**from-scratch** process-sequence transformer (a) **generalize to an unseen 4th semiconductor product
family (OOD)**, (b) **infer the family (mosfet/igbt/ic) from the step pattern** rather than memorize
per-family shortcuts, and (c) **beat our baselines on the benchmark** — while staying small, from-scratch,
not overfitting, and not regressing in-distribution. The hidden post-submission **OOD metric decides the
Zero One Hack Industrial-AI / Infineon track**, so it is the headline.

Background to read first: `MODEL_IMPROVEMENT_PLAN.md`, `CO_SCIENTIST_LAB_DESIGN.md`,
`extras/results/benchmark.md`, `DECISIONS.md` (desc-init rejected, RL deferred), `V2_RL_PLAN.md` §11,
`CLAUDE.md`. State at start: near the entropy floor in-distribution (a trigram nearly matches us),
~100% valid completions, LM-only anomaly ROC-AUC 0.997, **OOD next-step ~0.50 (the gap)**.

## Objective
Close the in-distribution→OOD gap and add family-inference + OOD-detection, beating **previous-best**
(primary target), staying above the **n-gram floor**, and chasing **Gemini** on OOD — every claim measured
on our benchmark, in-distribution AND OOD.

## Preferences
- **OOD generalization is the deciding axis** — weight it above in-distribution polish (already saturated).
- Prefer **config-expressible** changes (auto-testable) or ideas cheaply turned into a reusable knob.
- Prefer **compositional / structural inductive biases** over raw scale or raw data volume.
- Prefer ideas that also serve **family-inference** and **OOD-detection** (the new capabilities).
- Every claim must be measured: top-1/3/5, MRR; completion EM / norm-edit / token-acc / block-acc;
  anomaly Acc/P/R/F1/ROC-AUC/rule-attr; validity; **family-acc**; **ood-auroc** — in-dist AND OOD.

## Attributes / Criteria      # injected as {preferences}/{idea_attributes} into EVERY agent
- **OOD-generalization potential** — plausibly closes the ID→OOD gap (deciding axis; **highest weight**).
- **Buildability** — implementable + testable in our loop; config-expressible preferred.
- **Empirical status** — unvalidated | smoke-confirmed | full-confirmed. **Measured > simulated > reasoned.**
- **Overfit-resistance** — reduces, not increases, memorization (train≈val; no per-family shortcut).
- **Efficiency / sovereignty** — stays small & from-scratch; trains in minutes on 1 A100.
- **Novelty + grounding** — grounded in the compositional-generalization literature (web-checked).

## Constraints   (hard — must / never)
- **MUST** be from scratch — no big pretrained LLM, no non-sovereign dependency.
- **MUST** beat the n-gram (trigram) floor on every reported metric.
- **MUST NOT** regress in-distribution vs previous-best — **revert** any change that does.
- **MUST** train in minutes on a single A100 (smoke on CPU locally).
- Anomaly / validity **MUST** stay ≥ previous-best (already ~100% valid / ROC-AUC 0.997).

## Stop conditions
- **max_iterations:** 6   (focused run; raise only if no plateau)
- **success:** a **full-confirmed** hypothesis that improves **OOD** next-step (and/or completion) over
  previous-best by a margin that **holds across the 3-fold OOD**, with **no in-dist regression**, PLUS a
  working **family_id** and **ood_detect** capability above baseline. Otherwise stop on top-Elo plateau,
  the iteration cap, or the token/GPU budget.
- **per-round budget:** K Tier-2 (full Leonardo) experiments = set at checkpoint #1; token cap per round set there.
