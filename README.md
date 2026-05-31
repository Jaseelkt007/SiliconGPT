# SiliconGPT — Learning & Benchmarking Process Logic

**Zero One Hack_01 · Industrial AI / Infineon track.**

A small, modern decoder transformer (**RMSNorm · RoPE · SwiGLU**) trained **from scratch** on
semiconductor-fab process recipes (ordered step sequences). No giant pretrained LLM, no API
wrapper — a sovereign, reproducible stack. We then used a **measurement-grounded multi-agent
discovery loop** (an adaptation of Google's AI Co-Scientist, extended with a GPU **Experiment
agent**) to search the architecture space, and found that a **1.37M-parameter** model
generalizes *better* out-of-distribution than the 25M V1 — at no in-distribution cost.

**Tasks (scored by the organizers' `eval_metrics.py`):** next-step prediction · sequence
completion · anomaly detection · (post-submission) OOD generalization to a hidden 4th family.

> **Headline:** the 1.37M model beats Gemini 3.5-flash / GPT-5 / DeepSeek / Qwen on all three
> tasks while being ~1000× smaller, and is the first lever to move the deciding OOD metric
> (+0.008, 3-seed mean). Full numbers + the discovery story: **[`REPORT.md`](REPORT.md)**.

---

## Run it from a clean checkout

Needs **Python 3.12** and a GPU for training (CPU works for the smoke test + inference). The
dataset and checkpoints are **gitignored and regenerable** — the steps below reproduce them.

```bash
# 1. Install (one manifest covers training, inference, and the demo backend)
pip install -r requirements.txt
#    (on the Leonardo cluster we use pixi: bash scripts/setup_leonardo.sh)

# 2. Regenerate the synthetic dataset — deterministic, ~30 s  -> data/
python scripts/build_datasets.py --seed 42

# 3. (optional) smoke-test the whole loop on CPU in seconds
python src/process_logic/train.py --smoke --device cpu

# 4. Train the 1.37M deliverable (3 layers · d=192 · RoPE) on all families — minutes on 1 GPU
python src/process_logic/train.py \
    --config configs/train_v1.yaml --model-config configs/model_3m_rope.yaml \
    --ckpt-dir checkpoints/final_3m_rope --device cuda
#    -> checkpoints/final_3m_rope/best.pt
```

> **The trained deliverable is already committed** at `checkpoints/best.pt` (5.2 MB — the 1.37M
> RoPE model), so you can **skip step 4 and run inference directly**. Step 4 reproduces the
> *byte-identical* checkpoint at `checkpoints/final_3m_rope/best.pt`. (The dataset and the larger
> experiment checkpoints stay gitignored/regenerable.)

### Reproduce the official submission (the judges' path)

Run the trained model on the organizers' eval inputs to produce the three submission files,
then score with their `eval_metrics.py`:

```bash
python src/process_logic/predict.py --ckpt checkpoints/best.pt \
    --out-dir extras/results \
    --nextstep-input   eval_input_valid.csv \
    --completion-input eval_input_valid.csv \
    --anomaly-input    eval_input_anomaly.csv \
    --calib-file       data/val_id.csv
#    -> extras/results/{nextstep,completion,anomaly}.csv   (organizer submission format)

python eval/eval_metrics.py        # official scorer (drop in at event start)
```

Formats follow the spec exactly (`generation_rules.md §5`): inputs are pipe-separated
`PARTIAL_SEQUENCE` / `SEQUENCE`; outputs are `EXAMPLE_ID,RANK_1..RANK_5`,
`EXAMPLE_ID,PREDICTED_SEQUENCE`, and `EXAMPLE_ID,IS_VALID,SCORE,PREDICTED_RULE`. Our own held-out
eval CSVs (`data/eval_*.csv`) are drop-in compatible if the organizer files aren't to hand.

### Quick CPU demo (no GPU, finishes in seconds)

To confirm the whole **predict → score** pipeline works on your machine, run it on a tiny
subset of the eval inputs. **No GPU required** — it runs on a single CPU.

**One command (recommended):**

```bash
bash scripts/run_demo.sh
```

It uses the committed checkpoint (`checkpoints/best.pt`), carves a tiny subset
(`data/eval_*_demo.csv` — 6 next-step / 5 completion / 6 anomaly examples) from the full eval
files if it isn't there yet, predicts on it, and scores it. Outputs land in
**`extras/test_folder/`** (`nextstep.csv`, `completion.csv`, `anomaly.csv`, `score_demo.txt`).
The script auto-detects the env: it uses **pixi** if present, otherwise the `python` on your
`PATH` (e.g. after `pip install -r requirements.txt`).

> Needs `data/` to exist — run `python scripts/build_datasets.py` first on a clean checkout
> (the dataset is gitignored/regenerable). The script tells you if a file is missing.

**Manual (the exact two steps the script runs)** — useful if you want to point at your own
files or run inside an explicitly activated environment:

```bash
# activate the env (pick one):
export PATH="$HOME/.pixi/bin:$PATH"      # our env -> prefix the commands below with `pixi run`
# or:  pip install -r requirements.txt   # plain venv -> run `python ...` directly

# 1) predict on the tiny subset, on CPU
pixi run python src/process_logic/predict.py \
    --ckpt checkpoints/best.pt --device cpu --out-dir extras/test_folder \
    --nextstep-input   data/eval_nextstep_demo.csv \
    --completion-input data/eval_completion_demo.csv \
    --anomaly-input    data/eval_anomaly_demo.csv \
    --calib-file       data/val_id.csv

# 2) score (--intersect filters the full ground truth down to the subset's IDs)
pixi run python src/process_logic/score.py \
    --pred-dir extras/test_folder --gt-dir data --intersect
```

Swap `--device cpu` for `--device cuda` and the `*_demo.csv` inputs for the full `eval_*.csv`
(or the organizer files) to run the real thing.

---

## Live demo (optional UI)

- **Live demo:** https://silicon-oracle-suite.lovable.app/  *(works while the backend tunnel is up)*
- **Frontend source:** https://github.com/Unais2003/silicongpt-intelligence-front  (React · TanStack Start · MIT)

A Flask backend exposes the model; the frontend visualizes it. Both call the **same** model — the
backend is the real engine, the UI is the demo.

```bash
python server/app.py        # serves http://localhost:5050 (defaults to checkpoints/best.pt; override with CHECKPOINT_PATH)
```

- **Single-sequence:** load a held-out validation example → next-step (top-1/3/5) · complete
  (greedy or sampled) · validate (10 rules) · anomaly.
- **Batch:** "Run validation set" (no upload) or drop an `eval_*` CSV → full metrics + per-family.

Frontend + screenshots: see **[`REPORT.md`](REPORT.md)** (Results) and `extras/results/`.

---

## Tests

```bash
python tests/test_vocab.py      # tokenizer (no torch)
python tests/test_dataset.py    # batching  (no torch)
python tests/test_score.py      # metrics   (no torch)
python tests/test_model.py      # model     (needs torch)
python tests/test_generate.py   # inference (needs torch)
```

## Repo layout

```
src/process_logic/   vocab · dataset · model · train · generate · anomaly · predict · score
                     generation (vendored official grammar + validate_sequence)
scripts/             build_datasets.py · run_*.sh (Leonardo Slurm) · benchmark_*.py
server/              Flask inference API (app.py, inference.py)
configs/             model_3m_rope.yaml (deliverable) · model_v1.yaml (25M) · train_v1.yaml
data/                generated (gitignored) — regenerate with build_datasets.py
extras/results/      submission CSVs + benchmark outputs + the discovery run record
eval/                drop the organizers' eval_metrics.py here at event start
```

See **[`REPORT.md`](REPORT.md)** (the jury report), `DECISIONS.md`, and `CLAUDE.md`
(project context + Leonardo/Slurm runbook).

> **Honesty note:** the final 1.37M deliverable checkpoint **is committed** (`checkpoints/best.pt`,
> 5.2 MB) so judges can test without retraining; the dataset and the larger experiment checkpoints
> (25M V1, OOD runs) are *not* committed (gitignored) — they regenerate deterministically with the
> commands above. The model trains from scratch and uses
> no external API at inference. Block-level Accuracy shown in the UI is our 5-step-window
> interpretation; the authoritative number is from the organizers' `eval_metrics.py`.
