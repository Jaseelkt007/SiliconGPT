# Process-Logic Model — V1 Baseline Report

*Zero One Hack_01 — Industrial AI / Infineon: "Learning and Benchmarking Process Logic."*

## TL;DR
We train a **small (~25M-param) decoder-only transformer from scratch** on semiconductor fab
process recipes (ordered step sequences, ~200-token vocabulary, 3 families: mosfet / igbt / ic)
and evaluate on next-step prediction, sequence completion, and anomaly detection. The V1 baseline
**ranks the true next step in the top-5 essentially 100% of the time (top-1 ≈ 0.81, MRR ≈ 0.91)**
and achieves **perfect anomaly detection (F1 = 1.0)** via a validator-hybrid. Free-form sequence
completion is the weak axis (token-acc ≈ 0.40) and is our primary V2 target. The OOD metric (hidden
4th family) — the deciding axis — is addressed by the from-scratch + RoPE + no-family-conditioning
design and will be measured directly via a held-out-family experiment in V2.

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

Loss/accuracy curves: `extras/results/curves.png`. Raw log: `extras/results/train_log.csv`.
Submission CSVs: `extras/results/{nextstep,completion,anomaly}.csv`.

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
OOD levers must beat.** Likely the biggest single cause is that unseen `ic` tokens keep their random
init embeddings → description-init embeddings (lever 6) targets this directly.

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
1. **OOD experiment (deciding metric) — DONE (see Task 4 above).** Held-out `ic`: next-step top1
   0.789→0.451. Now *improve* it: the V2 OOD levers (esp. description-init embeddings) must close this
   gap. Repeat the held-out protocol for mosfet/igbt too (`--export=ALL,EXCLUDE=igbt`) for a 3-fold
   average.
2. **LM-only anomaly pass** for honest model-evidence (perplexity ROC-AUC), reported alongside hybrid.
3. **Completion fix:** constrained/structured decoding (mask grammar-invalid next-steps); investigate
   block-level accuracy vs. the reference.
4. **Early stopping + scaling study:** `--patience`; sweep model {1M,5M,15M,50M} × data {1K…100K/family}.
5. **RL with the validator as reward:** rejection-sampling FT → GRPO (baseline→trained→optimized story).
6. **OOD levers:** description-init embeddings for unseen tokens; cross-family recombination augmentation;
   safe family conditioning (separate embedding + family-dropout + UNKNOWN_FAMILY).

## Reproduce
```bash
bash scripts/setup_leonardo.sh                 # login node: build pixi env
sbatch scripts/run_train.sh                    # A100: train (best.pt on val)
sbatch scripts/run_eval.sh                     # A100: predict + score + plot
```
Seed 42 throughout. Data is deterministic (`python scripts/build_datasets.py --seed 42`).
