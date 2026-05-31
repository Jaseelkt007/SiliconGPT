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

### Reproduce the official submission (the judges' path)

Run the trained model on the organizers' eval inputs to produce the three submission files,
then score with their `eval_metrics.py`:

```bash
python src/process_logic/predict.py --ckpt checkpoints/final_3m_rope/best.pt \
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

---

## Live demo (optional UI)

- **Live demo:** https://silicon-oracle-suite.lovable.app/  *(works while the backend tunnel is up)*
- **Frontend source:** https://github.com/Unais2003/silicongpt-intelligence-front  (React · TanStack Start · MIT)

A Flask backend exposes the model; the frontend visualizes it. Both call the **same** model — the
backend is the real engine, the UI is the demo.

```bash
python server/app.py        # serves http://localhost:5050 (set CHECKPOINT_PATH to your .pt)
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

> **Honesty note:** checkpoints and the dataset are *not* committed (gitignored) — they
> regenerate deterministically with the commands above. The model trains from scratch and uses
> no external API at inference. Block-level Accuracy shown in the UI is our 5-step-window
> interpretation; the authoritative number is from the organizers' `eval_metrics.py`.
