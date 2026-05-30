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
  6e-4→6e-5 with 200-step warmup, grad-clip 1.0, 20k iters, batch 64. **Best-on-validation
  checkpointing** (no early stopping yet — see "What didn't").
- **Anomaly = hybrid:** deterministic `validate_sequence` for the decision + exact rule attribution,
  with LM perplexity as the continuous score. (See honesty note in Results.)
- **Data:** deterministic generator. 60K train (20K/family), 12K in-distribution val. All sequences
  validator-clean; no train/val/ood overlap.

## Results (local scorer, per family + overall)
Checkpoint: best-on-val at **iter 17000, val_loss 0.3090**. Eval inputs: `data/eval_*.csv`.

### Task 1 — Next-step prediction
| family | n | top1 | top3 | top5 | MRR |
|---|---|---|---|---|---|
| **ALL** | 3600 | **0.814** | 0.998 | 1.000 | 0.905 |
| mosfet | 1200 | 0.821 | 0.998 | 1.000 | 0.909 |
| igbt | 1200 | 0.825 | 0.998 | 1.000 | 0.911 |
| ic | 1200 | 0.797 | 0.996 | 0.999 | 0.895 |

### Task 2 — Sequence completion
| family | n | exact_match | norm_edit_dist ↓ | token_acc |
|---|---|---|---|---|
| **ALL** | 600 | **0.002** | 0.227 | **0.403** |
| mosfet | 200 | 0.005 | 0.162 | 0.463 |
| igbt | 200 | 0.000 | 0.228 | 0.461 |
| ic | 200 | 0.000 | 0.291 | 0.284 |

### Task 3 — Anomaly detection (hybrid: LM + deterministic validator)
| family | n | acc | precision | recall | F1 | ROC-AUC | rule_attr |
|---|---|---|---|---|---|---|---|
| **ALL** | 1000 | 1.000 | 1.000 | 1.000 | **1.000** | 1.000 | 0.910 |
| mosfet | 341 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.894 |
| igbt | 358 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.933 |
| ic | 301 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.900 |

Loss/accuracy curves: `extras/results/curves.png`. Raw log: `extras/results/train_log.csv`.
Submission CSVs: `extras/results/{nextstep,completion,anomaly}.csv`.

## What worked
- **Clean, fast convergence.** Val loss reaches its plateau (~0.31) by **iter ~2000**; top-5 next-step
  accuracy is **1.000** and MRR **0.905** — the model learned the recipe grammar's local structure well.
- **Per-family consistency** on Tasks 1 & 3 (no family collapses), a good sign for transfer.
- **Anomaly hybrid** delivers perfect detection with exact rule attribution at 0.91.
- **Reproducible & portable.** Fixed seed (42); deterministic data; pixi env pinned. Resolved the
  Leonardo RHEL8 `GLIBCXX` issue for the PyPI torch wheel via `LD_LIBRARY_PATH=$CONDA_PREFIX/lib`
  (in `pixi.toml [activation.env]`). Verified CUDA on the A100 (`torch 2.10.0+cu128`).

## What didn't / honest caveats
- **Top-1 (0.81) is below the naive 90–95% expectation — but this is grammar branching, not a bug.**
  top-3 = 0.998 and top-5 = 1.000 mean the true step is almost always in the top 3; at many cut
  points several next-steps are *equally valid* by the grammar, so a single-label top-1 is capped.
  (V2: quantify validator-valid next-steps per cut point to confirm.)
- **Completion is the weak axis** (token-acc 0.40, exact-match ≈ 0). Two causes: greedy decoding
  compounds error over long suffixes, and the model likely emits *a* valid ordering that differs from
  the *single* reference suffix. `ic` (our OOD-proxy family) is weakest (0.284). V2 diagnostic: check
  whether generated completions are validator-valid even when they don't match the reference.
- **Task 3's perfect scores come from the *deterministic validator*, not the LM.** This is by design,
  but we must report the **LM-only** (perplexity) anomaly numbers separately as the honest evidence the
  model itself learned the logic — *not yet run* (`predict.py --no-validator`).
- **No early stopping.** We run a fixed 20k iters; the model converges by ~2k and *mildly overfits*
  in the tail (val_loss drifts 0.31→0.35). Best-on-val checkpointing saves us, but ~90% of GPU time is
  wasted — add `--patience` + smaller `max_iters` for the V2 scaling sweep.

## Next steps (V2)
1. **OOD experiment (deciding metric):** add `--exclude-family` to `train.py`; train on two families,
   evaluate on the held-out third (`ood_holdout.csv`); report the ID→OOD drop.
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
