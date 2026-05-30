# CLAUDE.md — Process-Logic Model

Project context for Claude Code. Read this first each session.

## What this is
Submission for the **Zero One Hack_01 — Industrial AI / Infineon** track: *"Learning and Benchmarking Process Logic."* We train sequence model(s) on semiconductor fab **process recipes** (ordered step sequences) and are scored on:
1. **Next-step prediction** (Top-1/3/5, MRR)
2. **Sequence completion** (60%/80% prefix → suffix; Exact Match, Norm. Edit Distance, Token/Block Acc)
3. **Anomaly detection** (valid vs. rule-violating; F1, ROC-AUC, Rule Attribution)
4. **OOD generalization** (hidden 4th product family; scored post-submission — the deciding metric)

Goal: **win.** Honest, reproducible, real engineering (judges penalize "LLM wrappers").

## Core decisions (locked)
- **Small/medium decoder transformer trained FROM SCRATCH** (~10–25M, modern: RMSNorm, SwiGLU, RoPE, weight-tied). NOT a big pretrained LLM — it would memorize and hurt the OOD metric; no pretrained semiconductor model fits our modality (they're text/image models).
- **Custom vocabulary** (~200 step tokens + PAD/BOS/EOS/UNK, padded to 256). One step = one token. Embeddings learned from scratch (V2: initialize from step descriptions via a frozen text encoder — strong OOD lever for unseen tokens).
- **Family** (mosfet/igbt/ic) is given in the eval input. V1 uses no explicit family conditioning; V2 ablates additive family embedding + family-dropout (so it helps in-distribution without breaking OOD).
- **Anomaly**: hybrid of LM-perplexity (the "did it learn?" evidence) + the deterministic `validate_sequence` (best score + exact rule attribution). Report both honestly.
- **Optimization roadmap**: base LM → SFT → RL with the validator as reward (rejection-sampling FT first, then GRPO).
- We write all code directly (no Codex). Develop locally, train on the Leonardo A100 cluster.

## Full plan
`V1_BUILD_PLAN.md` (in this repo and at `/mnt/d/bounded/V1_BUILD_PLAN.md`) — repo structure, component build order, training plan, test plan, two-stage RL plan, V2 levers.

## Repo layout
```
src/process_logic/
  generation.py   # vendored official grammar+generator (generate_sequence, validate_sequence, rule constants)
  vocab.py        # (todo) tokenizer
  dataset.py      # (todo) DataLoader
  model.py        # (todo) decoder transformer
  train.py        # (todo) training loop
  generate.py     # (todo) next-step + completion decoding
  anomaly.py      # (todo) perplexity + validator hybrid
  predict.py      # (todo) eval inputs -> submission CSVs
scripts/
  build_datasets.py   # DONE — generates everything under data/
tests/                # (todo) unit + smoke tests
data/                 # generated (gitignored)
eval/eval_metrics.py  # organizers' scorer (drop in at event start)
extras/results/       # submission outputs + plots
```

## Data (already generated — `python scripts/build_datasets.py`)
Compact format = one sequence per row, steps joined by `|`.
- `train_pool.csv` (60K: 20K/family) `SEQUENCE_ID,FAMILY,SEQUENCE`
- `val_id.csv` (12K) — in-distribution validation
- `ood_holdout.csv` (4K, ic) — OOD proxy (train on the other two, test here)
- `eval_nextstep.csv` `EXAMPLE_ID,FAMILY,PARTIAL_SEQUENCE,TRUE_NEXT_STEP`
- `eval_completion.csv` `EXAMPLE_ID,FAMILY,COMPLETION_FRACTION,PARTIAL_SEQUENCE,TRUE_SUFFIX`
- `eval_anomaly.csv` (600 valid + 400 invalid, 40/rule) `EXAMPLE_ID,FAMILY,SEQUENCE,IS_VALID,RULE_VIOLATED`
- `anomaly_train.csv` (8K valid + 8K invalid, 800/rule) — for the anomaly classifier
Verified: no train/val/ood overlap; all train sequences pass `validate_sequence`; anomaly labels match the validator. Regenerate larger on the server via CLI args (`--n-train 100000`, etc.).

## Conventions
- Python 3.12, PyTorch. Configs in `configs/*.yaml`. Fixed seeds for reproducibility.
- The 10 rules / grammar are authoritative in `src/process_logic/generation.py` and `tracks/industrial-infineon/training_data/generation_rules.md`.
- Always keep a `--smoke` path (tiny model/data, CPU) that runs the full loop before any GPU run.

## Server (Leonardo / CINECA — EuroHPC, A100 64GB)
- SSH (no 2FA for the hackathon): `ssh <user>@login01-ext.leonardo.cineca.it` (also login02/05/07).
- Env: **pixi** (https://pixi.sh). Build it ON A LOGIN NODE (internet there) — `bash scripts/setup_leonardo.sh`. **Compute nodes have NO internet.**
- Storage: put data + checkpoints under **`$SCRATCH`** ($HOME is 50GB; SCRATCH is large but auto-deleted after 40 days). Don't use $FAST/$WORK.
- Slurm (`scripts/run_train.sh`): partition `boost_usr_prod`, `--reservation=s_tra_ncc` (hackathon = 1 node/team), `--gpus-per-task` 1–4 with `mem=120GB×gpus`, `cpus=8×gpus`, `--time` ≤ 24:00:00.
- Commands: `sbatch scripts/run_train.sh` → `squeue --me` → `tail -f slurm-<id>.out` → `scancel <id>`. Shell into a running job: `srun --overlap --pty --jobid=<id> bash`.
- CPU smoke on the server: a login node (10-min CPU limit) or `srun --partition=lrd_all_serial --time 04:00:00 --gres=tmpfs:100G --mem=16G --pty bash`, then `pixi run python src/process_logic/train.py --smoke --device cpu`.
- Compute-node internet (wandb only): export HTTP(S)_PROXY=http://proxyuser:5dd1d2bd00@10.99.0.1:38425 (low-bandwidth; restarts ~10 min). Default: wandb off, we log to `extras/results/train_log.csv`.
- Onboarding kit: https://ai-at.eu/hpc-onboarding/ (Ch.5 first steps, Ch.6 software).

## V2 status & benchmark (as of 2026-05-30)
V1 complete + validated. V2 exploration (see `DECISIONS.md`, `V2_RL_PLAN.md`, `extras/results/benchmark.md`):
- **Validity is a first-class metric** (`validity.py`, `scripts/measure_validity.py`): greedy/sampled completion ~100% valid; free-gen 0.997.
- **LM-only anomaly ROC-AUC 0.997** → the model genuinely learned process logic (not just the validator-hybrid).
- **Description-init for OOD → REJECTED** (honest negative; 3-fold top-1 0.495→0.477). OOD gap is *structural*, not embedding placement.
- **RL (RFT/GRPO) → DEFERRED** — Phase A showed ~no validity headroom (already ~100% valid).
- **Baselines (committee provided NONE — only the generator/validator):** n-gram floor (`scripts/baselines.py`) + LLM frontier (`scripts/llm_baseline.py`: Gemini/GPT/Kimi via `.env`, thinking-off; Kimi deferred — rate-limited). Collate with `scripts/benchmark_table.py`.
- **Benchmark result:** ours ≳ n-gram ≫ Gemini **in-distribution**; next-step is saturated; **V1 is only marginally above a trigram in-distribution**; **OOD is the deciding, mostly-untested axis** (the LLM may generalize better).

## NEXT SECTION — model improvement (new session)
Make the small from-scratch model decisively better where a trigram/LLM can't:
- **Generalize to OOD** (deciding metric; V1 OOD next-step ~0.50) and **infer the family from context** (don't memorize per-family patterns).
- **Targets:** beat previous-best checkpoint (primary), stay above n-gram (floor), chase Gemini on OOD (frontier bar).
- **Levers** (V2_RL_PLAN §11 / DECISIONS): cross-family recombination augmentation, family conditioning + dropout, positional-encoding ablations, scaling study; cheap wins = constrained decoding (B0) + per-family anomaly threshold (C1).
- **Benchmark every iteration** with `scripts/benchmark_table.py` (+ the OOD run).
- The LLM baseline env is built locally: `.venv` + keys in `.env` (gitignored). `.venv/bin/python scripts/llm_baseline.py --provider gemini ...`.
