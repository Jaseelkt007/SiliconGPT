# Process-Logic Model — Report (V1 baseline → 3M final deliverable)

*Zero One Hack_01 — Industrial AI / Infineon: "Learning and Benchmarking Process Logic."*

> **Recommended final model = a 1.37M-param from-scratch decoder (3 layers, n_embd=192, RoPE), trained on
> all 3 families** — ≈18× smaller than the 25M V1, matching it in-distribution and slightly better OOD
> (3-fold OOD next-step top-1 **0.5031 ± 0.0069** vs 0.4947 single-seed = **+0.008, ≈1 sd, modest**).
> Selected by the co-scientist-lab improvement loop; **source of truth = `ARCHITECTURE_FINAL.md`**, see
> **"V2 FINAL"** below and `DECISIONS.md` D3. The original V1 baseline writeup follows unchanged.

## TL;DR
We train a **small (~25M-param) decoder-only transformer from scratch** on semiconductor fab
process recipes (ordered step sequences, ~200-token vocabulary, 3 families: mosfet / igbt / ic)
and evaluate on next-step prediction, sequence completion, and anomaly detection. The V1 baseline
**ranks the true next step in the top-5 essentially 100% of the time (top-1 ≈ 0.81, MRR ≈ 0.91)**
and achieves **perfect anomaly detection (F1 = 1.0)** via a validator-hybrid. Free-form sequence
completion is the weak axis (token-acc ≈ 0.40) and is our primary V2 target. The OOD metric (hidden
4th family) — the deciding axis — is addressed by the from-scratch + RoPE + no-family-conditioning
design and will be measured directly via a held-out-family experiment in V2.

## V2 FINAL — the small model (from the co-scientist-lab improvement loop)

> **Full, source-of-truth writeup: `ARCHITECTURE_FINAL.md`.** All numbers below are disk-verified
> (`extras/results/final3m/`). They CORRECT earlier figures in commit 60b3b7d that were fabricated under a
> tmpfs-corruption episode — see the Integrity note in `ARCHITECTURE_FINAL.md`.

**Recommended final model: a 3-layer, n_embd=192 decoder = 1.37M params (≈18× smaller than the 25M V1),
RoPE, trained on all 3 families.** Selected by an experimentally-grounded discovery loop (the private
**co-scientist-lab** skill) that tested 9 hypotheses against the real 3-fold OOD metric. Verdict: **the only
lever that improved OOD was reducing model capacity** — a small, weakly-significant effect; five other
directions were falsified (`DECISIONS.md` D3).

### Final metric matrix — 1.37M RoPE vs 25M V1 (verified from disk)
| metric | 25M V1 | **1.37M (3L/192)** |
|---|---|---|
| params | 25.31M | **1.37M** (≈18× smaller) |
| next-step top-1 / top-5 / MRR (in-dist) | 0.807 / 1.000 / 0.901 | 0.811 / 1.000 / 0.903 |
| completion EM / norm-edit / token-acc | 0.002 / 0.227 / 0.400 | 0.000 / 0.222 / 0.405 |
| anomaly Acc / F1 / ROC-AUC (hybrid) | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 |
| anomaly F1 / ROC-AUC (LM-only) | 0.826 / 0.997 | 0.815 / 0.995 |
| validity greedy / sampled / free | 1.000 / 1.000 / 0.997 | 1.000 / 0.997 / 0.997 |
| ood_detect AUROC | 1.000 | 1.000 |
| **3-fold OOD next-step top-1** | **0.4947** (1 seed) | **0.5031 ± 0.0069** (seeds 42/43/44) |

**OOD honestly seed-confirmed:** seeds 42/43/44 = 0.5120 / 0.5019 / 0.4953 → mean **0.5031, sd 0.0069**, i.e.
**+0.0084 over the 25M baseline — a small gain, about the size of its own scatter (±0.007), not a decisive
win.** (The 25M baseline is single-seed, so its variance is unknown — the fair comparison would seed both.)
In-distribution: **no cost** (top1 0.811 ≥ 0.807; val_loss 0.329 ≈ V1). The defensible claim is *"18× smaller,
matches V1 in-distribution, slightly better OOD."*

### Positional encoding
RoPE and NoPE **tied on validation** (both val_loss 0.3289) → keep **RoPE** as the default. NoPE is not an
OOD driver (it was neutral, +0.004, in the ablation).

### Why smaller helps (mechanism)
In-distribution is **saturated** — a trigram nearly ties the 25M model. Surplus capacity is spent
**memorising per-family co-occurrence shortcuts** that don't transfer to an unseen family; a small model is
pushed toward the **family-agnostic grammar** the unseen family also obeys. Scaling curve (single seed, OOD
top-1): 25.3M 0.4947 → 10.7M 0.5008 → 3.2M 0.5119 → 1.37M 0.5120 → 0.62M 0.5139. We pick **1.37M** (not the
0.62M end) because 0.62M slightly regresses in-distribution. (Size labels here are the TRUE param counts;
earlier "3M/6M/15M" labels were ~2× too high.)

### Five principled negatives (each rules out a tempting direction)
1. **Description-init embeddings** → −0.018 OOD (D1).
2. **Cross-family recombination augmentation** → −0.018 OOD.
3. **NoPE alone** neutral (+0.004); **NoPE + augmentation** −0.009.
4. **Validator-guided constrained decoding** → rejected: only **~3% of OOD top-1 errors are grammar-invalid**
   (≈97% valid-but-wrong), so masking can't help.
5. **Universal-Transformer weight-sharing** → −0.009 OOD: tying layers ≠ shrinking size.

### The diagnosis (the honest frontier)
The OOD residual is a **hard transition-structure gap**: out-of-distribution the model almost never emits an
*illegal* step (~97% of its errors are grammar-*valid-but-wrong*) — it picks the **wrong legal step**. Not
fixable by data, embeddings, positional encoding, decoding, or weight-tying; it is about learning the unseen
family's *ordering*. Capacity reduction recovers only a small slice (~+0.008 seed-confirmed); the rest is a
measured, largely-irreducible frontier — the "does it learn or memorise?" question, answered with data.

### Reproduce the small model
```bash
sbatch scripts/run_final_3m.sh   # trains 3M RoPE+NoPE on all families, picks better by val,
                                 # writes submission CSVs, scores all metrics, seed-confirms 3-fold OOD
```
Chosen checkpoint: `checkpoints/final_3m_rope/best.pt` (RoPE; the `final_3m_nope/` variant tied on val).
Submission CSVs: `extras/results/{nextstep,completion,anomaly}.csv` (from the 1.37M RoPE model; 25M V1
copies preserved under `extras/results/submission_v1_25m/`). Source of truth: **`ARCHITECTURE_FINAL.md`**.
Full loop log: `LOOP_LOG.md`; decisions + provenance: `DECISIONS.md` D3; per-run results:
`extras/results/final3m/` and `extras/results/coscilab/`. NOTE: `configs/model_v1.yaml` is still the 25M
default — adopting the small model as the project default is a deliberate next-session step.

---

## Problem
Three scored tasks (+ a hidden OOD generalization task scored post-submission):
1. **Next-step prediction** — partial sequence → next step (Top-1/3/5, MRR)
2. **Sequence completion** — 60%/80% prefix → suffix (Exact Match, Norm. Edit Distance, Token Acc)
3. **Anomaly detection** — valid vs. rule-violating (Acc, P, R, F1, ROC-AUC, Rule Attribution)
4. **OOD generalization** — unseen 4th product family (drop ID→OOD; the deciding metric)

## Approach
- **Model:** decoder-only transformer, **25.31M params** (n_layer=8, n_head=8, n_embd=512,
  block_size=256), modern components: **RMSNorm, RoPE, SwiGLU, weight-tied head**, bf16 autocast.
  Trained **from scratch** — deliberately *not* a large pretrained LLM, whose memorization capacity
  would hurt the OOD metric.
- **Tokenizer:** custom vocabulary, **one process step = one token** (202 tokens = 4 specials + 198
  steps). No BPE (it would shred the step=position structure).
- **No explicit family conditioning** in V1 (avoids one-hotting the family, which would hurt OOD).
- **Training:** next-token cross-entropy, AdamW (betas 0.9/0.95, wd 0.1 on 2-D params), cosine LR
  6e-4→6e-5 with 100-step warmup, grad-clip 1.0, batch 64, up to 4000 iters with **early stopping**
  (patience 8) and **best-on-validation checkpointing**. Validation uses a **fixed, family-balanced**
  eval set (`build_eval_batches`) so the metric is stable and unbiased. W&B logging is flag-gated
  (`--wandb`, offline on compute nodes); grad-norm is logged each eval.
- **Anomaly = hybrid:** deterministic `validate_sequence` for the decision + exact rule attribution,
  with LM perplexity as the continuous score. (See honesty note in Results.)
- **Data:** deterministic generator. 60K train (20K/family), 12K in-distribution val. All sequences
  validator-clean; no train/val/ood overlap.

## Results (local scorer, per family + overall)
Checkpoint: best-on-val at **iter 4000, val_loss 0.3288** (V1.1 clean-eval run). Eval inputs:
`data/eval_*.csv`.

### Task 1 — Next-step prediction
| family | n | top1 | top3 | top5 | MRR |
|---|---|---|---|---|---|
| **ALL** | 3600 | **0.807** | 0.997 | 1.000 | 0.901 |
| mosfet | 1200 | 0.812 | 0.996 | 1.000 | 0.904 |
| igbt | 1200 | 0.821 | 0.998 | 1.000 | 0.909 |
| ic | 1200 | 0.789 | 0.996 | 0.999 | 0.891 |

### Task 2 — Sequence completion
| family | n | exact_match | norm_edit_dist ↓ | token_acc |
|---|---|---|---|---|
| **ALL** | 600 | **0.002** | 0.227 | **0.400** |
| mosfet | 200 | 0.005 | 0.168 | 0.459 |
| igbt | 200 | 0.000 | 0.233 | 0.463 |
| ic | 200 | 0.000 | 0.279 | 0.277 |

### Task 3 — Anomaly detection (hybrid: LM + deterministic validator)
| family | n | acc | precision | recall | F1 | ROC-AUC | rule_attr |
|---|---|---|---|---|---|---|---|
| **ALL** | 1000 | 1.000 | 1.000 | 1.000 | **1.000** | 1.000 | 0.910 |
| mosfet | 341 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.894 |
| igbt | 358 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.933 |
| ic | 301 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.900 |

**LM-only anomaly (the model's *own* evidence — perplexity threshold, no validator).** Reported beside
the hybrid so the perfect hybrid F1 is honestly attributed to the deterministic checker, and the LM's
intrinsic anomaly ability is shown separately. Threshold calibrated on `val_id` (one global cut).
| family | n | acc | precision | recall | F1 | ROC-AUC | rule_attr |
|---|---|---|---|---|---|---|---|
| **ALL** | 1000 | 0.831 | 0.703 | 1.000 | 0.826 | **0.997** | n/a |
| mosfet | 341 | 0.974 | 0.940 | 1.000 | 0.969 | 1.000 | n/a |
| igbt | 358 | 0.972 | 0.937 | 1.000 | 0.968 | 1.000 | n/a |
| ic | 301 | 0.502 | 0.423 | 1.000 | 0.595 | 1.000 | n/a |

The LM **ranks** valid vs. rule-violating sequences essentially perfectly on its own — **ROC-AUC 0.997
overall and 1.000 within every family** — strong evidence it learned process logic, not just the validator
doing the work. The lower F1 (0.826) is purely a *single-threshold* effect: ic recipes have systematically
higher perplexity (most-distinct family), so one global cut over-flags valid ic sequences (ic precision
0.423 at recall 1.000) even though ic's ranking is perfect (AUC 1.000). A per-family threshold removes
this; rule attribution is n/a for LM-only (that is the validator's role in the hybrid).

### Validity (process-validity ≠ accuracy)
Fraction of *generated* sequences passing all 10 rules (`scripts/measure_validity.py`, `best.pt`, n=300).
A high-accuracy model can still emit rule-breaking sequences, so this is a separate, first-class axis.
| regime | valid_frac | notes |
|---|---|---|
| greedy completion | **1.000** | prefix → greedy |
| sampled completion (temp 1.0) | **1.000** | prefix → temperature sampling |
| free generation (temp 1.0) | **0.997** | full recipe from `<BOS>`; 1/300 broke RULE_IMPLANT_NO_MASK |

The base LM is already near-perfectly valid even under sampling — so RL's validity *headroom is tiny*
(~0.3%, only in free generation). This reframes Phase B honestly: **constrained decoding** can *guarantee*
the remaining 0.3% (accuracy-safe, weights unchanged), while RL's value is the "optimized" rung / RLVR
demonstration rather than a large validity gain.

Loss/accuracy curves: `extras/results/curves.png`. Raw log: `extras/results/train_log.csv`.
Submission CSVs: `extras/results/{nextstep,completion,anomaly}.csv`. LM-only anomaly:
`extras/results/lmonly/anomaly.csv`.

### Task 4 — OOD generalization (the deciding metric)
We train an identical model on **two families only** (mosfet + igbt; `--exclude-family ic`,
40K train / 8K val, best-on-val 0.3087) and evaluate on the **held-out `ic` family** it has never
seen. Vocab covers all three families, so `ic`-specific tokens have embeddings but were never trained
— exactly the "unseen 4th family" setup. Same architecture, isolated checkpoint
(`checkpoints/ood_ic/`); the mosfet/igbt rows act as a health control.

**Next-step on `ic` — ID model (saw ic) vs. OOD model (never saw ic):**
| metric | ID | OOD | drop |
|---|---|---|---|
| top1 | 0.789 | **0.451** | −0.338 |
| top3 | 0.996 | 0.608 | −0.388 |
| top5 | 0.999 | 0.623 | −0.376 |
| MRR  | 0.891 | 0.530 | −0.361 |

**Completion on `ic`:** token-acc 0.277 → **0.169** (−0.108); norm-edit-dist 0.279 → 0.488 (worse).

**Health control — the OOD model on the families it *did* train on:** mosfet top1 **0.823**, igbt
top1 **0.819** — equal to (slightly above) the full model. So the `ic` collapse is a genuine
*generalization* gap, not an undertrained/broken model. Anomaly stays F1 = 1.0 (deterministic
validator, family-independent).

**Read:** there is real but partial transfer — a model that never saw an `ic` recipe still reaches
top1 0.451 / top5 0.623 on `ic` (vs. ~0.005 random), so it learned family-agnostic process logic
(clean→deposit→litho→etch ordering). But the large top1 drop (0.79→0.45) shows it leans on
family-specific token co-occurrences unavailable for an unseen family. **This is the baseline the V2
OOD levers must beat.** We hypothesised the biggest cause was unseen tokens keeping random embeddings,
and tested description-init to fix it — but it did not help top-1 (see the 3-fold result below), which
relocates the bottleneck from *embedding placement* to *missing transition/ordering structure*.

**Gap decomposition (measured on the 1,200 `ic` next-step examples):** 29 of `ic`'s 130 tokens are
never trained when `ic` is held out. **22.3%** of `ic` next-step targets are such unseen tokens (the
OOD model's output row stays random → it *cannot* emit them), and **100%** of prefixes contain ≥1
unseen token (context corruption). Implied OOD top-1 *on shared/trained targets* ≈ 0.451 / 0.777 ≈
**0.58** (vs. ID ~0.79). We *hypothesised* the unseen-token portion was a "mechanical" component that
embedding warm-start would recover — but the description-init experiment below **refutes** this: giving
those tokens sensible vectors raised top-5 recall yet did not improve top-1. So even the unseen-token
gap is really about *missing transition structure*, not embedding placement — the whole −0.34 drop
behaves as a structural generalization residual.

**Description-init result — 3-fold OOD, next-step on the held-out family.** Baseline = random init;
`+desc-init` = embedding warm-started from each step's **name** via a frozen MiniLM encoder
(`scripts/build_emb_init.py`; no description CSVs were available on the server, so name-only). Each
fold trains without one family and is scored on the held-out family. Reproduce: `bash
scripts/run_ood_3fold.sh` then `EMB_INIT=emb_init.npz bash scripts/run_ood_3fold.sh`, then
`pixi run python scripts/ood_summary.py`.

| held-out family | baseline top-1 | +desc-init top-1 | Δ | baseline top-5 | +desc-init top-5 |
|---|---|---|---|---|---|
| ic              | 0.451 | 0.439 | −0.012 | 0.623 | 0.647 |
| igbt            | 0.484 | 0.470 | −0.014 | 0.674 | 0.679 |
| mosfet          | 0.549 | 0.521 | −0.028 | 0.727 | 0.748 |
| **3-fold avg**  | **0.495** | **0.477** | **−0.018** | **0.674** | **0.692** |

**Result (honest negative): name-based description-init did NOT close the OOD gap.** It slightly
*lowered* top-1 (−0.018 avg) while modestly *raising* top-5 (+0.018) and top-3. The
"recovers half-to-two-thirds of the gap" hypothesis is **rejected**. The top-1/top-5 split is the
tell: warm-starting unseen-family tokens from their names makes them *reachable* (they enter the top-5
candidate set → recall up) but does not fix the rank-1 decision. Why:
- **Embedding placement ≠ transition knowledge.** The model still never observed the *transitions*
  into/out of an unseen token; a good vector makes it a candidate, not the argmax.
- **Name-semantic similarity ≠ process-logic adjacency.** A generic text encoder over step *names*
  clusters by wording, which only partially matches which steps are grammar-interchangeable. (Descriptions,
  which we didn't have, might help — untested.)
- It re-initialises the *whole* embedding matrix from text space, mildly perturbing trained tokens too.

Methodology note: mosfet is the easiest held-out fold (0.549), ic the hardest (0.451), confirming ic is
the most structurally distinct family. The baseline `ic` fold reproduced the earlier number (0.451)
exactly — a clean cross-check. **Takeaway for V2:** the OOD residual is mostly *structural* (novel step
*ordering*), which embedding init cannot supply; the more promising levers are those that inject
transition/ordering structure — cross-family recombination augmentation, or RL with the validator as
reward — not better token embeddings.

## What worked
- **Clean, fast, monotonic convergence.** With the fixed family-balanced eval set, val loss descends
  smoothly **0.339 → 0.329** over 4000 iters (no oscillation); top-5 next-step accuracy is **1.000**
  and MRR **0.901** — the model learned the recipe grammar's local structure well.
- **Per-family consistency** on Tasks 1 & 3 (no family collapses), a good sign for transfer.
- **Anomaly hybrid** delivers perfect detection with exact rule attribution at 0.91.
- **Trustworthy instrumentation.** A full health test (`tests/test_health.py`, 25/25) verifies causal
  attention (no future leakage), RoPE relative-position sensitivity, RMSNorm, weight-tying,
  dataloader padding + ground-truth shift alignment, per-parameter gradient flow, and weight updates.
- **Reproducible & portable.** Fixed seed (42); deterministic data; pixi env pinned. Resolved the
  Leonardo RHEL8 `GLIBCXX` issue for the PyPI torch wheel via `LD_LIBRARY_PATH=$CONDA_PREFIX/lib`
  (in `pixi.toml [activation.env]`). Verified CUDA on the A100 (`torch 2.10.0+cu128`).

## What didn't / honest caveats
- **Top-1 (0.81) is below the naive 90–95% expectation — but this is grammar branching, not a bug.**
  top-3 = 0.997 and top-5 = 1.000 mean the true step is almost always in the top 3; at many cut
  points several next-steps are *equally valid* by the grammar, so a single-label top-1 is capped.
  (V2: quantify validator-valid next-steps per cut point to confirm.)
- **Completion is the weak axis** (token-acc 0.40, exact-match ≈ 0). Two causes: greedy decoding
  compounds error over long suffixes, and the model likely emits *a* valid ordering that differs from
  the *single* reference suffix. `ic` (our OOD-proxy family) is weakest (0.277). V2 diagnostic: check
  whether generated completions are validator-valid even when they don't match the reference.
- **Task 3's perfect scores come from the *deterministic validator*, not the LM.** This is by design,
  but we must report the **LM-only** (perplexity) anomaly numbers separately as the honest evidence the
  model itself learned the logic — *not yet run* (`predict.py --no-validator`).
- **The earlier "overfitting tail" was a measurement artifact.** The first run's val_loss appeared to
  oscillate 0.31↔0.37; that was a *family-blocked* validation sampler scoring a different biased slice
  each eval, not real instability. With the fixed balanced eval set, train ≈ val and the curve is
  monotonic — so early stopping (patience 8) is purely a **compute saver**, not overfitting protection.
  The grammar saturates by ~iter 500; `max_iters` is now 4000 (down from 20000).

## Next steps (V2)
1. **OOD experiment (deciding metric) — DONE (see Task 4).** 3-fold baseline avg top-1 **0.495**.
   Description-init **tested and rejected** for top-1 (−0.018; +0.018 on top-5). The gap is structural.
2. **Inject ordering structure (the promising OOD levers, given the desc-init result):**
   **cross-family recombination augmentation** (splice validator-valid sub-sequences across families to
   teach family-agnostic ordering) and **RL with the validator as reward** (rejection-sampling FT →
   GRPO) — both add transition knowledge that embedding init cannot.
3. **Description-init follow-ups (lower priority):** retry with the kit's step *descriptions* (not just
   names) if obtainable; or warm-start ONLY unseen-family tokens (leave trained tokens untouched) to
   avoid perturbing them — isolates whether the small top-1 drop is the whole-matrix re-init.
4. **LM-only anomaly pass** for honest model-evidence (perplexity ROC-AUC), reported alongside hybrid.
5. **Completion fix:** constrained/structured decoding (mask grammar-invalid next-steps); investigate
   block-level accuracy vs. the reference.
6. **Scaling study:** sweep model {1M,5M,15M,50M} × data {1K…100K/family} (early stopping already in).

## Reproduce
```bash
bash scripts/setup_leonardo.sh                 # login node: build pixi env
sbatch scripts/run_train.sh                    # A100: train (best.pt on val)
sbatch scripts/run_eval.sh                     # A100: predict + score + plot
```
Seed 42 throughout. Data is deterministic (`python scripts/build_datasets.py --seed 42`).
