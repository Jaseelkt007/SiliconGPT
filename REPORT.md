# Process-Logic Model — Report

> Fill in once we have numbers from the Leonardo run. Sections per the submission template.

## TL;DR
_What we built, who it's for, what it achieved (2–3 sentences)._

## Problem
Learn the grammar of semiconductor fab process recipes (ordered step sequences) and
benchmark next-step prediction, sequence completion, and anomaly detection — with a focus
on **generalizing to an unseen 4th product family (OOD)** rather than memorizing.

## Approach
- Small **decoder transformer trained from scratch** (RMSNorm · RoPE · SwiGLU), custom 202-token
  vocabulary (one step = one token). _Why: tiny low-entropy language; a big pretrained LLM would
  memorize and hurt OOD._
- Deterministic synthetic data from the process grammar (`scripts/build_datasets.py`); a held-out
  family as an **OOD proxy**.
- Anomaly detection = **LM perplexity + deterministic validator hybrid** (model-only result reported
  separately as evidence of learned logic).
- _(V2)_ Scaling study, RL with the validator as reward (rejection-sampling → GRPO), description-init
  embeddings, cross-family augmentation.

## How to run it
See `README.md`. Train: `sbatch scripts/run_train.sh`. Predict: `python src/process_logic/predict.py ...`.
Score locally: `python src/process_logic/score.py --pred-dir extras/results --gt-dir data`.

## Results
_Headline metrics + baseline vs trained + per-family breakdown. Paste `score.py` / `eval_metrics.py`
output here. Include `extras/results/curves.png`._

| Task | Metric | Baseline | Trained |
|---|---|---|---|
| 1 Next-step | Top-1 / Top-5 / MRR | | |
| 2 Completion | ExactMatch / NormEditDist / TokenAcc | | |
| 3 Anomaly | F1 / ROC-AUC / RuleAttr | | |
| 4 OOD (proxy) | drop ID→OOD | | |

## What worked / What didn't
_Honest engineering notes._

## What we'd do with another 36 hours
_Concrete next steps (scaling curve, RL, OOD augmentation, demo dashboard)._

## Credits & dependencies
PyTorch, numpy, pyyaml, matplotlib. Grammar/validator vendored from the track kit. AI coding assistant: Claude Code.
