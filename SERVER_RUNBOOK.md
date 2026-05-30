# Server Runbook & Context — Process-Logic Model (Leonardo / CINECA)

> **Read this top-to-bottom before acting.** This is the full context for the **server-side
> Claude Code session**. The V1 code is written and committed but its torch parts were only
> *syntax-checked* on a torchless laptop — your job is to take it from "syntax-verified" to a
> **trained model with real numbers**, produce the three submission files, then iterate (V2).
> Also read `CLAUDE.md`, `V1_BUILD_PLAN.md`, and `README.md` in this repo.

---

## 0. TL;DR — what to do on the server (in order)
1. `git pull`; build the **pixi env on a LOGIN node** (compute nodes have no internet).
2. **Regenerate the data** (deterministic) + build the vocab — `data/` is gitignored and NOT pushed.
3. Run the **`--smoke` test** — the first real execution of the torch code = the runtime gate.
4. `sbatch` a full training run on 1× A100 (~10–25M-param model; trains in minutes–~1h).
5. `predict.py` → 3 submission CSVs; `score.py` → baseline numbers (per-family).
6. Plot curves, fill `REPORT.md`. Then iterate: OOD experiment, scaling study, RL — brainstorm first.

---

## 0b. V1.1 instrumentation (already applied — verify with the health test)
After the first training run we added (see git log):
- **Stable validation:** `dataset.build_eval_batches` builds a FIXED, family-balanced eval set — fixes the earlier family-blocked, noisy metric (top-1 oscillating 0.78↔0.82 was a *sampling artifact*, not instability).
- **Early stopping** (`patience`) + `max_iters` lowered to 4000 (grammar saturates by ~iter 500). NOTE: there was **no overfitting** (train≈val); this only stops wasting compute.
- **W&B logging** — flag-gated (`wandb: true` in config; on compute nodes use `WANDB_MODE=offline`, sync from a login node). Grad-norm now logged too.
- **OOD experiment:** `train.py --exclude-family ic` trains/vals WITHOUT a family for the held-out OOD test.
- **Health test** `tests/test_health.py` — verifies **causal attention (no future leakage)**, RoPE, RMSNorm, weight-tying, dataloader shapes/padding/ground-truth alignment, **per-parameter gradient flow**, and weight updates. **Run it before trusting any training run.**

Confirmed-correct facts (baked into the health test): attention is **causal** via `F.scaled_dot_product_attention(..., is_causal=True)` (PyTorch fused/FlashAttention; RoPE/RMSNorm/SwiGLU hand-written, no HF/x-transformers). Loss = next-token CE with internal shift + `ignore_index=-100`. Right-padding ⇒ no manual padding mask needed.

## 1. Mission & objective
**Hackathon:** Zero One Hack_01 — **Industrial AI / Infineon** track, "Learning & Benchmarking Process Logic." Compute = Leonardo (CINECA) A100s. **Deadline: submit by Sunday 10:00** (Tally form).

**The problem:** semiconductor fab "recipes" are ordered sequences of ~115–155 process steps from a ~200-token vocabulary, across 3 product families (MOSFET / IGBT / IC). We train sequence model(s) to learn the *grammar* of these recipes. Scored tasks:

| # | Task | Input | Metrics |
|---|---|---|---|
| 1 | **Next-step prediction** | partial sequence | Top-1/3/5 Accuracy, MRR |
| 2 | **Sequence completion** | partial (60%/80%) | Exact Match, Normalized Edit Distance, Token Acc, Block-level Acc |
| 3 | **Anomaly detection** | full sequence | Binary Acc, Precision, Recall, F1, Confusion, ROC-AUC, Rule Attribution Acc |
| 4 | **OOD generalization** (hidden) | unseen 4th family | performance drop ID→OOD (organizers compute post-submission) |

**The deciding metric is #4 (OOD).** The judges repeatedly ask: *does the model learn transferable process logic, or just memorize?* **Win = strong OOD generalization + a clean baseline→trained→optimized story + reproducibility + honest evaluation.** They explicitly penalize "basic LLM wrappers — there must be real engineering underneath."

**Deliverables (in the repo):** `nextstep.csv`, `completion.csv`, `anomaly.csv` in `extras/results/`; training checkpoints + loss curves; `score.py`/`eval_metrics.py` results with per-family breakdown; `REPORT.md`; a baseline-vs-trained demo. Public MIT repo (the user handles the GitHub/license/push).

---

## 2. Repo state (V1 — code-complete)
```
src/process_logic/
  generation.py  vendored grammar + validate_sequence (the 10 rules)  [authoritative]
  vocab.py       tokenizer: 202 tokens (4 specials + 198 steps)        [tested ✅]
  dataset.py     CSV load + numpy collate + torch DataLoader           [tested ✅]
  model.py       decoder: RMSNorm, RoPE, SwiGLU, weight-tied           [syntax only ⏳]
  train.py       training loop + --smoke                               [syntax only ⏳]
  generate.py    rank_next_steps + complete_sequence                   [syntax only ⏳]
  anomaly.py     LM-perplexity + validator hybrid                      [syntax only ⏳]
  predict.py     eval inputs -> the 3 submission CSVs                  [syntax only ⏳]
  score.py       LOCAL scorer (all metrics, per-family)                [tested ✅]
scripts/         build_datasets.py, run_train.sh (Slurm), setup_leonardo.sh, plot_curves.py
configs/         model_v1.yaml, train_v1.yaml
tests/           test_{vocab,dataset,model,generate,score}.py  (model/generate are torch-gated)
data/            gitignored — REGENERATE on the server
vocab.json       committed (regenerable)
```
`⏳` = written to convention but **never executed** (no torch locally). The `--smoke` test is their first real run — expect to fix small runtime issues there; that is normal and fast.

---

## 3. Core design decisions (and WHY — do not undo without reason)
- **Small/medium decoder transformer trained FROM SCRATCH** (~10–25M params; modern: RMSNorm, RoPE, SwiGLU, weight-tied head). **NOT a big pretrained LLM.** Rationale: the "language" is tiny and low-entropy (~200 tokens, simple grammar); a 1B+ model has ~1000× more memorization capacity than the data has information → it memorizes, which *hurts* the OOD metric. Scaling/compositional-generalization literature shows bigger models don't help OOD and often hurt. Also no pretrained semiconductor model fits our modality (they're text/image models).
- **Custom vocabulary, one step = one token** (no BPE — BPE shreds step strings and destroys the step=position structure). `<PAD>=0 <BOS>=1 <EOS>=2 <UNK>=3`, then 198 steps.
- **Right-padding + causal attention** ⇒ no key-padding mask needed (real tokens never attend to future pads; pad labels are -100).
- **Anomaly = hybrid**: deterministic `validate_sequence` for the decision + exact rule attribution (nails the long-range "global" rules), with the LM's perplexity as the continuous SCORE for ROC-AUC. ALSO report a **model-only** result (LM surprisal) — that's the evidence the model *learned* logic (be transparent in REPORT that the validator is deterministic).
- **Use big compute for BREADTH** (scaling study, many seeds, RL runs) — not one giant model. Up to 4 A100s/node, 1 node/team.

---

## 4. Data (regenerate on the server — it is deterministic)
`python scripts/build_datasets.py --seed 42` reproduces it byte-for-byte (~30s). Compact format = one sequence per row, steps joined by `|`.

| File | Rows | Columns | Use |
|---|---|---|---|
| `train_pool.csv` | 60K (20K/family) | SEQUENCE_ID, FAMILY, SEQUENCE | training |
| `val_id.csv` | 12K | same | in-distribution val + anomaly threshold calibration |
| `ood_holdout.csv` | 4K (ic) | same | **OOD proxy** (train on the other 2, test here) |
| `eval_nextstep.csv` | ~1.8K | EXAMPLE_ID, FAMILY, PARTIAL_SEQUENCE, TRUE_NEXT_STEP | local Task 1 scoring |
| `eval_completion.csv` | 600 | EXAMPLE_ID, FAMILY, COMPLETION_FRACTION, PARTIAL_SEQUENCE, TRUE_SUFFIX | local Task 2 scoring |
| `eval_anomaly.csv` | 1000 (600 valid+400 invalid, 40/rule) | EXAMPLE_ID, FAMILY, SEQUENCE, IS_VALID, RULE_VIOLATED | local Task 3 scoring |
| `anomaly_train.csv` | 16K (8K+8K, 800/rule) | same | (V2) anomaly classifier training |

Generation is fast (~5,500 seq/s). To scale for the data-volume study: `--n-train 100000` etc. **Put data + checkpoints under `$SCRATCH`** ($HOME is 50GB; SCRATCH is large, auto-deleted after 40 days).

---

## 5. Model & training
**Config** (`configs/model_v1.yaml`): n_layer=8, n_head=8, n_embd=512, block_size=256 (covers longest seq), dropout=0.1, SwiGLU, RoPE, tie_weights. ≈ a ~25M model; change `n_layer`/`n_embd` for the scaling study.

**Objective:** next-token cross-entropy, `ignore_index=-100`, internal one-token shift. AdamW (betas 0.9/0.95, wd 0.1 on 2-D params only), cosine LR (peak 6e-4 → 6e-5) + 200-step warmup, grad-clip 1.0, bf16 autocast on CUDA, `eval_interval=500`, checkpoint best on val to `checkpoints/best.pt` (stores model + config + vocab). Logs to `extras/results/train_log.csv`.

**Expected (sanity):** next-step Top-1 should reach **~90–95%** (cleaner than business-process logs). **If below ~85%, something is wrong** (check the smoke first). Beat the trivial baselines (random/unigram/bigram) — if not, there's a bug.

---

## 6. Metrics & evaluation
**Local scoring (use now, before the organizers' script arrives):**
```
python src/process_logic/score.py --pred-dir extras/results --gt-dir data
```
Computes Task1 (Top-1/3/5, MRR), Task2 (ExactMatch, NormEditDist, TokenAcc), Task3 (Acc, P, R, F1, ROC-AUC, RuleAttribution) — **per family + overall**. Validated: perfect predictions → all metrics 1.0. (Block-level Accuracy is only in the organizers' `eval_metrics.py`.)

**Submission output formats** (predict.py already emits these):
- Task1 `nextstep.csv`: `EXAMPLE_ID, RANK_1..RANK_5`
- Task2 `completion.csv`: `EXAMPLE_ID, PREDICTED_SEQUENCE` (steps AFTER the cut only, `|`-joined)
- Task3 `anomaly.csv`: `EXAMPLE_ID, IS_VALID, SCORE, PREDICTED_RULE`

**At event start** the organizers distribute `eval_input_valid.csv` (`EXAMPLE_ID, FAMILY, COMPLETION_FRACTION, PARTIAL_SEQUENCE`; 600 rows) and `eval_input_anomaly.csv` (`EXAMPLE_ID, FAMILY, SEQUENCE`; 987 rows), plus the official `eval_metrics.py` (drop into `eval/`). For the **real submission**, point predict at those:
```
python src/process_logic/predict.py --ckpt checkpoints/best.pt \
    --nextstep-input <eval_input_valid.csv> \
    --completion-input <eval_input_valid.csv> \
    --anomaly-input <eval_input_anomaly.csv> \
    --calib-file data/val_id.csv
```
(`predict.py` reads `PARTIAL_SEQUENCE` for tasks 1&2 and `SEQUENCE` for task 3 — column-compatible with both their files and ours.)

**OOD experiment (the deciding axis):** train a model on **only two families**, evaluate on the held-out third (`ood_holdout.csv`), and report the metric drop ID→OOD. ⚠️ `train.py` does not yet support filtering families — add a small `--exclude-family` option (filter `train_ex`/`val_ex` by the FAMILY field) before running this. This is the single most important experiment for winning.

---

## 7. Leonardo HPC specifics (from the AI:AT onboarding deck)
- **SSH (no 2FA for the hackathon):** `ssh <user>@login01-ext.leonardo.cineca.it` (also login02/05/07).
- **Env = pixi** (https://pixi.sh). Build it **on a login node** (internet there): `bash scripts/setup_leonardo.sh`. **Compute nodes have NO internet.** ⚠️ The exact PyTorch+CUDA spec may need adjusting for Leonardo — if `pixi add pytorch-gpu cuda-version=12.*` fails, check the onboarding kit Ch.6, or use `module load` + pip, or a Singularity/Apptainer container (`singularity exec --nv ...`).
- **Storage:** `$HOME` 50GB · `$SCRATCH` large (auto-deleted after 40 days — **use for data + checkpoints**) · `$PUBLIC` 50GB · do **not** use `$FAST`/`$WORK`.
- **Login-node CPU limit = 10 min.** For longer interactive work: `srun --partition=lrd_all_serial --time 04:00:00 --gres=tmpfs:100G --mem=16G --pty bash`.
- **Compute-node internet** (only for low-bandwidth, e.g. wandb): export the proxy in the Slurm script — `export HTTP_PROXY=http://proxyuser:5dd1d2bd00@10.99.0.1:38425` (+ HTTPS_PROXY/http_proxy/https_proxy). Restarts ~every 10 min. **Download large files from a login node.** Default: keep wandb off, we log to CSV.
- **Slurm job (1 GPU)** — `scripts/run_train.sh` already encodes this:
  ```
  #SBATCH --partition=boost_usr_prod
  #SBATCH --reservation=s_tra_ncc        # hackathon (only 1 node/team)
  #SBATCH --nodes=1 --ntasks-per-node=1
  #SBATCH --gpus-per-task=1              # up to 4 on Leonardo
  #SBATCH --mem=120GB                    # 120GB * gpus-per-task
  #SBATCH --cpus-per-task=8              # 8 * gpus-per-task
  #SBATCH --time=02:00:00                # up to 24:00:00
  ```
  For 2/4 GPUs scale mem (240/480GB) and cpus (16/32). The run command inside is `pixi run python src/process_logic/train.py ...`.
- **Useful Slurm:** `sbatch scripts/run_train.sh` · `squeue --me` · `tail -c +0 -f slurm-<id>.out` · `scancel <id>` · shell into a running job: `srun --overlap --pty --jobid=<id> bash`.
- **Onboarding kit:** https://ai-at.eu/hpc-onboarding/ (Ch.5 first steps on Leonardo, Ch.6 software).

---

## 8. THE RUNBOOK — exact commands, in order
```bash
# --- on a LOGIN NODE (has internet) ---
git clone <repo-url> && cd process-logic        # or: git pull
bash scripts/setup_leonardo.sh                  # builds pixi env (adjust torch/CUDA if needed)

# regenerate data + vocab (deterministic; data/ is gitignored)
pixi run python scripts/build_datasets.py --seed 42
pixi run python src/process_logic/vocab.py

# run the local (non-torch) tests to confirm data integrity
pixi run python tests/test_vocab.py && pixi run python tests/test_dataset.py && pixi run python tests/test_score.py

# SMOKE TEST = the runtime gate (CPU, <10 min so OK on a login node).
# If this fails, fix the runtime bug it surfaces before anything else.
pixi run python src/process_logic/train.py --smoke --device cpu
# also: pixi run python tests/test_model.py && pixi run python tests/test_generate.py
# FULL HEALTH CHECK (causality / RoPE / RMSNorm / grad-flow / padding / GT alignment / weight-update):
pixi run python tests/test_health.py                       # architecture (fresh model)
# (after a run) inspect the trained model too — synonym-embedding cosine etc.:
pixi run python tests/test_health.py --ckpt checkpoints/best.pt

# --- full training on a GPU node ---
sbatch scripts/run_train.sh                     # watch: squeue --me ; tail -f slurm-<id>.out

# --- after training: predictions + scores ---
pixi run python src/process_logic/predict.py --ckpt checkpoints/best.pt \
    --nextstep-input data/eval_nextstep.csv \
    --completion-input data/eval_completion.csv \
    --anomaly-input data/eval_anomaly.csv \
    --calib-file data/val_id.csv
pixi run python src/process_logic/score.py --pred-dir extras/results --gt-dir data
pixi run python scripts/plot_curves.py          # -> extras/results/curves.png
```
Tip: put the repo + data + checkpoints on `$SCRATCH`. Point `train.py`/`predict.py` at it via `--data-dir` / config if you keep data outside the repo tree.

---

## 9. Pending work & V2 roadmap (brainstorm before building)
**Immediate (finish V1):** smoke-verify → train → baseline numbers → fill `REPORT.md` + `curves.png`. Add `--exclude-family` to `train.py` for the **OOD experiment** (train on 2 families, test on the 3rd).

**V2 levers (ranked by expected payoff for the OOD-deciding score):**
1. **OOD-first training** — avoid absolute positional encoding (we already use RoPE; ablate randomized/NoPE); **never one-hot the family ID** (kills OOD). Cross-family recombination augmentation (GECA-style, uniform sampling). Description-init embeddings (frozen, from a strong open text encoder) so unseen 4th-family tokens get sensible vectors.
2. **Scaling study** (explicit stretch goal) — train sizes {≈1M,5M,15M,50M} × data {1K,10K,50K,100K/family}, multiple seeds, in parallel across the 4 A100s; plot loss/accuracy vs params & data.
3. **RL with the validator as reward** (baseline→trained→**optimized** story): (a) **rejection-sampling fine-tuning** first (sample completions → keep validator-passing → SFT → repeat; simple & stable), then (b) **GRPO** (TRL `GRPOTrainer`, start from SFT ckpt, partial-credit reward `1−violations/10`).
4. **Anomaly**: a discriminative classifier + rule-attribution head trained on `anomaly_train.csv` for stronger F1/AUC; explicit "A-before-B" features for the global rules.
5. **Family conditioning** done safely: a small separate embedding + `UNKNOWN_FAMILY` row + **family-dropout** (~15%) so it helps in-distribution without breaking OOD.
6. **Big-LLM foil** (optional): one LoRA fine-tune of a ~1B model showing it memorizes (near-zero train loss, collapses on the held-out family) vs. our small model generalizing — a great rubric narrative.
7. **Demo** — baseline-vs-trained side-by-side + loss curves + confusion matrix + scaling plots (Streamlit/Gradio or a notebook).

---

## 10. Gotchas / risks
- **torch+CUDA install** on Leonardo may need adjustment (pixi `pytorch-gpu` + cuda-version, or module load, or container). First confirm `pixi run python -c "import torch; print(torch.cuda.is_available())"` is True on a GPU node.
- **Compute nodes have no internet** — install everything on a login node first; set the proxy only for wandb.
- **Don't commit `train_pool.csv`** (142MB > GitHub's 100MB limit) — regenerate instead.
- **block_size=256** covers the longest sequence (~155 steps + BOS/EOS); if you generate longer variants, raise it.
- The model/inference code is unrun — treat the first smoke as a debugging step, not a formality.
- Keep everything reproducible (fixed `--seed`); the rubric weighs reproducibility heavily.

---

## 11. Submission checklist (the user handles GitHub/push)
- [ ] `nextstep.csv`, `completion.csv`, `anomaly.csv` in `extras/results/` (on the organizers' eval inputs)
- [ ] checkpoint(s) + `train_log.csv` + `curves.png`
- [ ] `score.py` / `eval_metrics.py` results with per-family breakdown, in `REPORT.md`
- [ ] baseline-vs-trained demo (side-by-side outputs on identical inputs)
- [ ] `REPORT.md` filled (TL;DR, problem, approach, results, what worked/didn't, next steps)
- [ ] repo runs from a clean checkout; `requirements.txt`/pixi present; no secrets
