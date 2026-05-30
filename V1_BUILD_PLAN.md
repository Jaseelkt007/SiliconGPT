# Process-Logic Model — V1 Build Plan

> Industrial AI / Infineon track — "Learning & Benchmarking Process Logic".
> This is the working reference plan. V1 = build the whole pipeline + base training + baseline numbers.
> Training & RL plans are included as the roadmap (clearly phased). No Codex — we write the code directly.

---

## 0. Strategy & phasing

Principle: **make it work end-to-end first, then make it good.** A complete, submittable pipeline with honest baseline numbers beats a half-finished "perfect" model.

| Stage | What | Output |
|---|---|---|
| **V1 (this plan)** | Full code + base LM training + local eval + the 3 submission files | Working artifact + baseline scores |
| **V1.5 (optimization)** | Two RL stages on top of the base model | Higher validity / completion scores |
| **V2 (deferred levers)** | Scaling study, description-init, family conditioning, anomaly classifier, OOD augmentation, demo | Stretch goals + winning margin |

The three scored tasks: **(1) next-step prediction, (2) sequence completion (60/80%), (3) anomaly detection.** Hidden **(4) OOD** = unseen 4th family, scored post-submission.

---

## 1. Dataset (already created)

Compact format (one full sequence per row, steps joined by `|`) except eval files. All under `data/`.

| File | Purpose | Columns |
|---|---|---|
| `train_pool.csv` | Base LM training (balanced, ~100K/family) | `SEQUENCE_ID, FAMILY, SEQUENCE` |
| `val_id.csv` | In-distribution validation | `SEQUENCE_ID, FAMILY, SEQUENCE` |
| `ood_holdout.csv` | **OOD proxy** — one whole family held out | `SEQUENCE_ID, FAMILY, SEQUENCE` |
| `eval_nextstep.csv` | Local Task-1 scoring | `EXAMPLE_ID, FAMILY, PARTIAL_SEQUENCE, TRUE_NEXT_STEP` |
| `eval_completion.csv` | Local Task-2 scoring | `EXAMPLE_ID, FAMILY, COMPLETION_FRACTION, PARTIAL_SEQUENCE, TRUE_SUFFIX` |
| `eval_anomaly.csv` | Local Task-3 scoring | `EXAMPLE_ID, FAMILY, SEQUENCE, IS_VALID, RULE_VIOLATED` |
| `anomaly_train.csv` | Train the anomaly classifier (V2) | same as eval_anomaly, larger balanced mix |

The `eval_*` files mirror the organizers' input format + ground-truth columns so we can score locally with `eval_metrics.py`. **OOD trick:** train on two families, test on `ood_holdout` to measure generalization before submission.

---

## 2. Repo structure

```
process-logic/
├── README.md                 # setup + run instructions (judged)
├── REPORT.md                 # technical write-up (judged)
├── LICENSE                   # MIT (required)
├── requirements.txt          # pinned deps
├── configs/
│   ├── data.yaml             # sizes, families, paths
│   ├── model_v1.yaml         # arch params (size = one number to change)
│   └── train_v1.yaml         # lr, batch, epochs, logging
├── data/                     # the dataset files from §1 (gitignored)
├── src/process_logic/
│   ├── vocab.py              # step<->id, special tokens, save/load vocab.json
│   ├── dataset.py            # Dataset/DataLoader: tokenize, pad, mask, batch
│   ├── model.py              # decoder transformer (modern components)
│   ├── train.py              # train loop: checkpoint, resume, eval, logging
│   ├── generate.py           # decoding: next-step top-5 + autoregressive completion
│   ├── anomaly.py            # LM-perplexity score + deterministic-validator hybrid
│   ├── predict.py            # run model on eval inputs -> submission CSVs
│   ├── grammar.py            # vendored validate_sequence (source of truth)
│   └── rl.py                 # V1.5: rejection-sampling FT + GRPO reward hooks
├── scripts/
│   ├── build_datasets.py     # data builder (already written)
│   ├── run_train.sh          # local --smoke run + SLURM launch for cluster
│   ├── run_rl.sh             # V1.5 RL launch
│   └── make_submission.py    # writes the 3 files into extras/results/
├── eval/eval_metrics.py      # organizers' scorer (dropped in at event start)
├── extras/results/           # submission outputs + score reports + loss curves
└── tests/
    ├── test_units.py         # vocab/dataset/model unit tests
    └── test_smoke.py         # tiny end-to-end sanity run (no GPU)
```

---

## 3. Components & responsibilities

| Module | Does | Key detail |
|---|---|---|
| `grammar.py` | The 10-rule validator | Vendored from `generate_sequences.py`; reused by anomaly **and** RL reward |
| `vocab.py` | Fixed vocab (~204) + `PAD/BOS/EOS/UNK` → 256 | Fit once from `train_pool`, saved to `vocab.json` (frozen) |
| `dataset.py` | CSV → padded token batches | Masks `PAD` from loss & attention; builds next-token targets |
| `model.py` | Decoder-only transformer | v1 ≈ 10–25M; size is **one config line** (for the scaling study) |
| `train.py` | Train + checkpoint + eval | Next-token CE; AdamW + cosine + warmup; bf16; eval val loss + top-1/3/5; resume |
| `generate.py` | Inference for Tasks 1 & 2 | Task 1: top-5 logits at last position. Task 2: autoregressive decode to `EOS` |
| `anomaly.py` | Inference for Task 3 | Hybrid: validator + LM perplexity; reports both model-only and hybrid scores |
| `predict.py` | Eval inputs → submission CSVs | Reads checkpoint + vocab; organizers' exact format |
| `rl.py` | V1.5 optimization | Rejection-sampling FT + GRPO using the validator as reward |

---

## 4. Build order (dependency-ordered)

```
Phase 0  Scaffold: repo, requirements, LICENSE, vendor grammar.py, drop in eval_metrics.py
Phase 1  vocab.py        -> vocab.json
Phase 2  dataset.py      -> loads train_pool, yields padded batches      [unit-test shapes]
Phase 3  model.py        -> forward pass on a dummy batch                 [check param count]
Phase 4  train.py        -> smoke-train 50 steps on 500 seqs (CPU)        <- END-TO-END GREEN
Phase 5  generate.py     -> next-step top-5 + completion decode
Phase 6  anomaly.py      -> perplexity + validator hybrid
Phase 7  predict.py + make_submission.py -> the 3 CSVs
Phase 8  eval_metrics.py on local eval_* -> scores + per-family + loss-curve plot
Phase 9  run_train.sh (SLURM) + README/REPORT -> full run on the A100 server
```

Phase 4 is the milestone: once the smoke run is green, the whole loop works.

---

## 5. How the pieces connect (data flow)

```
build_datasets.py -> data/*.csv
        |
        +- train_pool.csv -> vocab.py -> vocab.json
        |                        |
        +-------------------> dataset.py --> train.py(model.py) --> checkpoint.pt
                                                                        |
 eval_nextstep / eval_completion / eval_anomaly -> predict.py <---------+
   (+ vocab.json, grammar.py) -> generate.py / anomaly.py               |
                                        |                                |
                                        v                                |
                         extras/results/{nextstep,completion,anomaly}.csv
                                        |
                                        v
                              eval/eval_metrics.py -> scores + loss-curve plot
```

---

## 6. Model spec (V1)

Decoder-only transformer with modern components (not vanilla 2022 nanoGPT):

| Param | V1 value | Note |
|---|---|---|
| Params | ~10–25M | one config line; scaling study sweeps this in V2 |
| Layers | 8 | |
| d_model | 512 | |
| Heads | 8 | head dim 64 |
| FFN | SwiGLU, ~2048 | gated FFN |
| Norm | RMSNorm, pre-norm | |
| Positional | RoPE | (V2 ablate: randomized / NoPE for OOD) |
| Context | 256 | covers 155-step sequences + specials |
| Embedding | learned from scratch, weight-tied to output | (V2: description-init) |
| Dropout | 0.1 | |
| Bias | none | |
| Attention | Flash-Attention | A100 |

Toolkit options: `x-transformers` (toggle all of the above) or a small Llama-style HF config.

---

## 7. Training plan (base LM)

**Objective:** next-token cross-entropy, `ignore_index=PAD`, optional label smoothing 0.1.

**Data handling:** train_pool tokenized via vocab; pad to batch max; train on **all 3 families mixed and balanced** (no family bias). `val_id` for in-distribution validation; never train on `ood_holdout`.

**Optimizer / schedule:**
- AdamW, betas (0.9, 0.95), weight_decay 0.1, grad_clip 1.0
- Peak LR 3e-4–1e-3, cosine decay to 10%, warmup ~150 steps
- bf16 + `torch.compile`; batch 32–64; context 256
- Train until val loss plateaus; **early stop on val** (this grammar is low-entropy — watch for memorization)

**Eval cadence:** every N steps log val loss + next-step **top-1/3/5** on a val subset. Checkpoint best-on-val; keep last for resume.

**Regularization:** dropout 0.1, weight decay, label smoothing, early stopping.

**Logging (judged):** train/val loss curves + metrics over time (W&B, or CSV → matplotlib). Save plots to `extras/results/`.

**Expected:** next-step top-1 ~90–95% (cleaner than business-process logs). Below ~85% => bug.

---

## 8. Inference → submission generation (the 3 tasks)

- **Task 1 (next-step):** forward the partial sequence, take logits at the last position, mask special/PAD tokens, output **top-5** → `nextstep.csv` (`EXAMPLE_ID, RANK_1..5`). Top-k + MRR scored.
- **Task 2 (completion):** autoregressive decode from the prefix until `EOS`/`SHIP LOT` or a length cap. **Greedy for v1** (deterministic, reproducible); beam = V2. Output **only steps after the cut** → `completion.csv` (`EXAMPLE_ID, PREDICTED_SEQUENCE` pipe-joined). Optionally constrain decoding to never emit special tokens mid-sequence.
- **Task 3 (anomaly):** report **two results** for honesty + best score:
  1. **Model-only** (the "did it learn?" story): per-token surprisal / sequence perplexity from the LM; threshold calibrated on a valid held-out set; localize anomalous step by max-surprisal. Catches local rules well; weaker on global rules.
  2. **Hybrid** (best practical score): also run `validate_sequence`; if it fires → `IS_VALID=0`, `RULE_VIOLATED=`returned rule (this nails global rules + exact attribution). LM gives the `SCORE` for AUC.
  → `anomaly.csv` (`EXAMPLE_ID, IS_VALID, SCORE, PREDICTED_RULE`). Be transparent in REPORT that the validator is deterministic; the model-only result is the learning evidence.

---

## 9. Test & validation plan

**Unit tests (`tests/test_units.py`):**
- vocab encode→decode round-trip is identity; special-token ids stable
- dataset batch shapes correct; PAD masked from loss & attention; targets shifted by one
- model forward output shape `[B, T, vocab]`; logged param count matches config
- generation emits only valid token ids; stops at `EOS`

**Smoke test (`tests/test_smoke.py`) — the gate before any GPU run:**
- end-to-end on 500 sequences, 50 steps, CPU → produces all 3 CSVs without error

**Data validation:**
- every `train_pool` sequence passes `validate_sequence` (assert 0 violations)
- `eval_*` files have the right columns; `eval_anomaly` rule labels verified by the validator
- **no overlap** between train / val / ood (hash sequences); OOD family absent from train

**Sanity baselines (must be beaten — catches silent bugs):**
- random, unigram, bigram/trigram next-step accuracy as floors; the transformer must beat them
- **overfit test:** train on 50 sequences, confirm train loss → ~0 (proves learning works)

**Eval-harness correctness:**
- feed a trivially-perfect prediction file to `eval_metrics.py`, confirm ~100% (validates our output format matches the scorer)

**Reporting:** per-family + per-cut breakdowns; log baseline metrics so every later change is measured against them (regression tracking).

---

## 10. RL plan (V1.5 — two kinds, post-baseline)

Prereq: a trained base/SFT model + the validator as reward + a set of "prompts" (partial sequences to complete). Both use `grammar.validate_sequence` as the verifier.

### Kind 1 — Rejection-Sampling Fine-Tuning (RFT / STaR) — do this first
Simple, stable, no RL infra; ideal because our verifier is exact + free.
```
LOOP (1–3 iterations):
  1. sample N completions per prompt (temp ~1.0)
  2. validate_sequence each -> keep only VALID completions
  3. SFT on (prompt + valid completion), mixed with original valid data (avoid forgetting)
  4. eval: valid% of generations, top-5, MRR, completion metrics
```

### Kind 2 — GRPO (verifier reward) — when RFT plateaus
- Start **from the RFT/SFT checkpoint** (never raw base).
- Generate G completions/prompt; reward = validator. **Partial credit** `1 - violations/10` if pure binary stalls.
- TRL `GRPOTrainer`: `num_generations=8`, `loss_type=dr_grpo`, lr ~5e-6, `beta` ~0.01 (small KL).
- Monitor `entropy` (collapse) and `frac_reward_zero_std` (no signal → balance prompt difficulty).

**What RL improves:** completion validity (Task 2), sharper next-step distribution, and the **baseline → trained → optimized** rubric story. Measure all 3 task metrics before/after.

---

## 11. Deferred levers (V2 — the winning margin)

- **Scaling study** (model size × data volume) — explicit stretch goal; run the grid in parallel on the A100s; plot curves.
- **Description-init embeddings** (frozen-auxiliary, strong open text encoder) — big OOD lever for unseen tokens.
- **Family conditioning** = small separate embedding table + `UNKNOWN_FAMILY` row, additive (every position) + **family-dropout** (~15%) so it helps ID without breaking OOD.
- **Anomaly classifier + rule-attribution head** (multi-task) for stronger F1/AUC + attribution.
- **Cross-family recombination augmentation** (GECA-style, uniform sampling) for OOD.
- **Big-LLM memorization baseline** (1B + LoRA) — a *foil* that shows "big memorizes, small generalizes."
- **Positional-encoding ablation** (RoPE vs randomized vs NoPE) on the OOD proxy.
- **Demo dashboard** — baseline-vs-trained side-by-side, loss curves, confusion matrix, scaling plots.

---

## 12. Definition of done (V1)

1. `--smoke` loop runs green end-to-end (no GPU).
2. Full training run on the server → checkpoint + loss curve.
3. `predict.py` emits the 3 submission files in the correct format.
4. `eval_metrics.py` reports scores for all 3 tasks **with per-family breakdown**.
5. Transformer beats the random/unigram/bigram baselines.
6. Baseline numbers recorded in `REPORT.md`.

→ A complete, submittable artifact + a measured baseline to improve against.

---

## 13. Local-dev → GPU-server workflow

- **Locally:** write all code + a `--smoke` flag (tiny model, few hundred sequences, 50 steps, CPU) to verify the full loop **before** burning GPU time.
- **Server (Leonardo A100s):** `run_train.sh` = SLURM script with the real config; checkpoints to scratch; `--resume` support; loss curves logged.
- `requirements.txt` pinned so the cluster env matches local.
