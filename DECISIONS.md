# Decision Log — what we tried, what we learned, why we pivoted

A running record of the experiments and the evidence-based course changes. (The judges reward
knowing *when not* to apply a technique as much as applying one.)

---

## D1 — Description-init embeddings for OOD → **REJECTED (honest negative)**
**Hypothesis:** the OOD gap (held-out family) is largely "mechanical" — unseen-family tokens keep
random embeddings — so warm-starting them from their text (frozen MiniLM) would recover ~half-to-
two-thirds of the gap.

**Experiment:** 3-fold OOD (hold out ic / igbt / mosfet), baseline vs. `+desc-init`.

**Result:** next-step top-1 **0.495 → 0.477 (−0.018)**; top-5 only +0.018. Hypothesis **refuted.**

**Learning:** warm-starting makes unseen tokens *reachable* (top-5 ↑) but cannot supply the *transition
knowledge* to make them the argmax (top-1 flat/down). **The OOD gap is structural** (the model never
learned the unseen family's ordering), not an embedding-placement problem. No training-data embedding
trick closes it when the held-out family has genuinely novel structure (ic's backside-first / tungsten).

**Pivot:** away from embedding-init for OOD; treat OOD as a measured, largely-irreducible frontier
(the memorize-vs-understand result: mosfet 0.55 > igbt 0.48 > ic 0.45, by structural distinctness).

---

## D2 — RL-with-validator (RFT/GRPO) for generation validity → **NOT PURSUED (no headroom)**
**Hypothesis:** RL with `validate_sequence` as a reward would push generation validity up and give the
rubric's "optimized" rung.

**Evidence (Phase A, `best.pt`, n=300):**
- Validity: greedy **1.000**, sampled (temp 1.0) **1.000**, free-from-`<BOS>` **0.997** (1/300).
- LM-only anomaly (no validator): **ROC-AUC 0.997** overall, **1.000** per family.

**Learning:** the base LM **already generates ~100% valid sequences** and its perplexity **already
separates valid from rule-violating essentially perfectly**. RL targets validity — which is saturated —
and does not help the other axes (top-1 capped by entropy floor; completion already 100% valid; OOD
needs structure RL can't supply). So RL would consume GPU for a near-null result.

**Decision:** **do not run RL.** Get the residual 0.3% validity *for free* via **constrained decoding**
(accuracy-safe, no weight change), and realise the "optimized" rung through *real* cheap optimisations
instead of a hollow RL run:
- **Constrained decoding** → guaranteed 100% validity (and can only help accuracy).
- **Per-family anomaly threshold** → LM-only F1 **0.826 → ~0.97** (the 0.826 is purely a single-global-
  threshold artifact; ic's higher perplexity over-flags. AUC is already 1.000 per family).

**Pivot:** redirect remaining effort to areas with real headroom / judged-but-untouched:
scaling study (size × data), a baseline-vs-trained **demo/dashboard**, and (cheap) constrained decoding +
per-family threshold calibration. (A small RFT remains *optional* purely as an RLVR narrative, but it is
largely redundant with constrained decoding and would not move metrics — documented here so we don't
re-litigate it.)

---

## Positive findings worth highlighting in the submission
- **The model learned real, family-agnostic process logic** — beats a trigram (loss 0.33 vs 0.43),
  sits at the grammar's entropy floor, generates ~100% valid recipes, and separates valid/invalid by
  perplexity at ROC-AUC 0.997. It is genuinely *understanding* the local grammar, not memorising
  (train ≈ val; top-5 = 1.0 on unseen held-out sequences).
- **OOD is the honest frontier** — partial transfer (0.50 vs 0.005 random) with a structural residual
  that no quick lever closed; quantified across 3 folds. This is the "does it learn or memorise?"
  question, answered with data.

---

## D3 — co-scientist-lab model-improvement run (2026-05-30/31)

We built an experimentally-grounded discovery harness (`scripts/run_experiment.py`: config → train
in-dist + per-OOD-fold → score → JSON; `extras/results/coscilab/`) and ran the **co-scientist-lab** skill
(Generation → Reflection → Experiment/GPU → Elo ranking → Meta-review) for two rounds. 9 hypotheses
generated; the config-expressible ones were trained + benchmarked on Leonardo A100s (full 3-fold OOD each).
All numbers below are read from the on-disk result JSONs (verified, single seed unless noted).

### D3.1 — Scale DOWN improves OOD → **CONFIRMED (the run's positive result)**
**Hypothesis:** in-distribution is saturated (a trigram nearly ties the 25M model), so excess capacity is
spent memorising per-family shortcuts that don't transfer to an unseen family. Smaller ⇒ more
family-agnostic ⇒ better OOD.

**Experiment:** identical training at five sizes, each scored on the held-out family (3-fold).

| size | params | 3-fold OOD top1 | in-dist top1 |
|---|---|---|---|
| baseline | 27.6M | 0.4947 | 0.8072 |
| mid | 14.7M | 0.5008 | 0.8089 |
| small | 6.4M | 0.5119 | 0.8044 |
| xsmall | 3.0M | 0.5120 | 0.8111 |
| tiny | 1.4M | **0.5139** | 0.8014 |

**Result:** OOD top1 rises **monotonically** as capacity shrinks 25M→1.4M (**+0.019**), at **zero in-dist
cost** (ID flat ~0.80). No turnover yet at 1.4M. A clean confirmation of the memorise-vs-generalise thesis
on our grammar (cf. Furrer et al. 2022, flat/negative OOD scaling for fine-tuned seq models). Modest in
magnitude but real and consistent across all three folds.

**Decision:** prefer a **small (~1–6M) model** for the OOD-deciding metric. (Confirm across seeds before
treating the exact sweet spot as final.)

### D3.2 — Validator-guided constrained decoding (h7) → **REJECTED**
**Hypothesis:** the model leaks probability onto grammar-invalid steps OOD; masking invalid candidates at
decode time (accuracy-safe — the true step is always valid) would convert top-5 reachability into top-1.

**Experiment (`scripts/diag_constrained.py`, n=200/fold, no retrain):** bucket OOD top-1 errors into
grammar-**invalid** argmax (recoverable by masking) vs grammar-**valid-but-wrong**.

**Result:** only **~3% of OOD top-1 errors are grammar-invalid** (ic 3.1% / igbt 3.6% / mosfet 1.8%;
in-dist 0%). **~97% are valid-but-wrong.** Masking cannot fix a valid-but-wrong pick → no OOD benefit.

**Learning (important reframe):** the model **almost never emits an illegal step** even OOD — it picks the
**wrong legal step**. So the OOD residual is genuine *transition-structure learning*, not a
decoding/validity problem, and not (per D1) embedding placement or (below) data augmentation. This narrows
the frontier sharply.

### D3.3 — Cross-family recombination augmentation → **REJECTED**; NoPE → neutral
- **Cross-family recombination** (GECA-style prefix(A)+suffix(B) splice, validator-filtered): 3-fold OOD
  top1 0.4947→**0.4767 (−0.018)**, same top1-down/top5-up signature as the D1 desc-init negative. With NoPE
  (augnope) 0.486. Augmentation **hurts** OOD here.
- **NoPE** (no positional encoding vs RoPE): 0.4947→0.4983 (**+0.004**, neutral); a safe, free default but
  not a driver.

### D3.4 — Universal-Transformer weight-sharing (h2) → **REJECTED**
**Hypothesis:** reduce capacity *structurally* — reuse one decoder block across all depths — to force a
single shared transition operator (Dehghani 2019; Csordás 2021), stacking with the size-reduction win.

**Experiment:** at the OOD-best ~3M base (n_layer=3, n_embd=192), weight_share off vs on, full 3-fold OOD.

**Result:** OOD top1 0.5120→**0.5033 (−0.009)**, completion 0.196→0.168, ID flat (0.811→0.812). Same
top5-up/top1-down signature.

**Learning:** reducing capacity by *tying layers* does **not** reproduce the gain from reducing it by
*size*. A single block reused 3× underfits the rank-1 transition decision (its OOD top-5 rises but top-1
falls). So the scaling win is about *total free parameters available to memorise*, not about enforcing a
recurrent/iterative computation. Rejected.

**Cumulative:** **five** independent levers touching *data, embeddings, position, decoding, or weight-tying*
all fail to move OOD top1 (desc-init, cross-family aug, NoPE, constrained decoding, weight-sharing). The
**one** lever that moved it was **plain size reduction**. Remaining un-GPU-tested idea from the run:
**family-dropout + UNKNOWN-row** conditioning (h4) — knob not built; deferred. The evidence strongly
suggests the OOD residual is a hard, largely-irreducible *transition-structure* gap, partially mitigated
(~+0.02) by training a smaller model.

**Process note (honesty):** during the run, intermittent scratch-tmpfs corruption produced garbled tool
output; two unverified figures were briefly stated mid-session (a "0.531 @ 3M peak" and an "h7 ~52%
recoverable") and then **retracted** — they never entered git or the results. All D3 numbers above are the
authoritative on-disk values (`git` commit `c16a754`).
