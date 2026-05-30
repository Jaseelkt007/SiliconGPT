# Process-Logic Model

**Industrial AI / Infineon track — Zero One Hack_01.** Learn the grammar of semiconductor
fab process recipes (ordered step sequences) and benchmark **next-step prediction**,
**sequence completion**, and **anomaly detection** — plus OOD generalization to an unseen
product family.

A small, modern decoder transformer (RMSNorm · RoPE · SwiGLU) trained **from scratch** on
synthetic, grammar-generated process sequences. No giant pretrained LLM, no API wrapper.

## Status (V1)
- [x] Deterministic data generation + validation — `scripts/build_datasets.py`
- [x] Tokenizer `vocab.py`, data loaders `dataset.py`
- [x] Model `model.py`, training `train.py` (`--smoke`)
- [x] Inference + submission files — `generate.py`, `anomaly.py`, `predict.py`
- [ ] Full training run + scores on Leonardo
- [ ] V2 — scaling study, RL (rejection-sampling → GRPO), description-init embeddings, OOD augmentation

## Quickstart
```bash
# 1. (re)generate the dataset — deterministic, ~30s  -> data/
python scripts/build_datasets.py --seed 42

# 2. build the vocab artifact -> vocab.json
python src/process_logic/vocab.py

# 3. smoke-test the whole loop (needs torch; tiny model, CPU)
python src/process_logic/train.py --smoke --device cpu

# 4. train (GPU)
python src/process_logic/train.py --config configs/train_v1.yaml

# 5. produce the three submission files -> extras/results/
python src/process_logic/predict.py --ckpt checkpoints/best.pt \
    --nextstep-input data/eval_nextstep.csv \
    --completion-input data/eval_completion.csv \
    --anomaly-input data/eval_anomaly.csv \
    --calib-file data/val_id.csv
```

## Tests
```bash
python tests/test_vocab.py      # tokenizer
python tests/test_dataset.py    # batching
python tests/test_model.py      # model (needs torch)
python tests/test_generate.py   # inference (needs torch)
```

## Repo
`src/process_logic/` — `vocab` · `dataset` · `model` · `train` · `generate` · `anomaly` · `predict` · `generation` (vendored grammar+validator).
`scripts/` — `build_datasets.py`, `run_train.sh` (Leonardo Slurm), `setup_leonardo.sh` (pixi env).
See **`CLAUDE.md`** (project context + Leonardo/Slurm) and **`V1_BUILD_PLAN.md`** (full plan).

## Data
`data/` is **gitignored and regenerable** — `python scripts/build_datasets.py --seed 42` reproduces
it byte-for-byte. `train_pool.csv` is ~142 MB (above GitHub's 100 MB file limit), so regenerate it on
the server instead of committing it.
