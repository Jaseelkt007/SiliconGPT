# bounded — Industrial AI (Infineon)

## Team

- **Mohammed Jaseel Kunnathodika** — AI / ML (model, training, inference)
- **Muhammed Unais Perinchikkal** — Full-stack (backend & frontend)

**Team name:** bounded · **Repository:** SiliconGPT
**Track:** Industrial AI (Infineon) — *Learning and Benchmarking Process Logic*
**Live demo:** https://silicon-oracle-suite.lovable.app/ · **Frontend repo:** https://github.com/Unais2003/silicongpt-intelligence-front

---

## TL;DR

We trained a small decoder transformer **from scratch** to learn semiconductor-fab process
grammar, then built a **measurement-grounded multi-agent discovery loop** (AI Co-Scientist +
a GPU **Experiment agent**) that searched the architecture space and discovered that a
**1.37M-parameter** model generalizes *better* out-of-distribution than the 25M baseline — at no
in-distribution cost. It beats Gemini 3.5-flash / GPT-5 / DeepSeek / Qwen on all three tasks while
being ~1000× smaller.

---

## Problem

Industrial process recipes are long, ordered step sequences whose meaning depends on order and
process logic. The question is whether a model *memorizes patterns* or learns a *transferable*
understanding — measured as generalization to an **unseen 4th product family** (OOD), the deciding
metric. In-distribution next-step is near-saturated (a trigram nearly ties a trained model), so the
real, open problem is **OOD generalization** on a small, sovereign, from-scratch model — not an LLM wrapper.

---

## Approach

- **Small decoder from scratch** (RMSNorm · RoPE · SwiGLU · 202-token vocab, one step = one token) on
  synthetic, grammar-generated sequences. No pretrained LLM, no inference-time API.
- **Anomaly = hybrid:** deterministic rule validator (exact rule attribution) + LM perplexity
  (evidence the model *learned* the logic, not just the checker).
- **A multi-agent discovery loop to *improve* the model** (not build it): 1 Supervisor + 6 Co-Scientist
  specialists + our **Experiment agent**, which actually **trains + benchmarks** each hypothesis on the
  A100s up a 3-tier ladder (debate → smoke → full 3-fold OOD). The Elo tournament ranks on *measured* OOD.
- **Runs on the Leonardo cluster** (pixi env, Slurm); the model is tiny and converges in minutes.

Architecture diagram + the full discovery run: `extras/results/coscilab/` and the frontend's
discovery page.

---

## How to run it

Exact commands in **[`README.md`](README.md)**. The judges' path, in short:

```bash
pip install -r requirements.txt
python scripts/build_datasets.py --seed 42
python src/process_logic/train.py --config configs/train_v1.yaml \
    --model-config configs/model_3m_rope.yaml --ckpt-dir checkpoints/final_3m_rope --device cuda
python src/process_logic/predict.py --ckpt checkpoints/final_3m_rope/best.pt --out-dir extras/results \
    --nextstep-input eval_input_valid.csv --completion-input eval_input_valid.csv \
    --anomaly-input eval_input_anomaly.csv --calib-file data/val_id.csv
python eval/eval_metrics.py          # official scorer
```

Needs Leonardo/GPU for training (minutes); inference + scoring run on CPU. Dataset + checkpoints are
gitignored and regenerate deterministically.

---

## Results

**Final model: 1.37M (3 layers · d=192 · RoPE), trained on all three families.** Numbers verified
from `extras/results/` (our scorer; the *authoritative* numbers come from the organizers'
`eval_metrics.py` on their eval files).

**Baseline vs final** (the required before/after):
| Metric | V1 (25M) | **Final (1.37M)** |
|---|---|---|
| Next-step Top-1 (in-dist) | 0.807 | **0.811** |
| Top-5 / MRR | 1.000 / 0.901 | 1.000 / 0.903 |
| Completion token-acc / NED | 0.400 / 0.227 | **0.405 / 0.222** |
| Anomaly F1 / ROC-AUC | 1.000 / 1.000 | **1.000 / 1.000** |
| **OOD next-step Top-1** | 0.4947 | **0.5031** (3-seed mean, **+0.008**) |
| Params | 25M | **1.37M (≈18× smaller)** |

**Head-to-head vs baselines** (next-step Top-1; LLMs sampled on 200, ours/n-gram full eval):
| Model | Top-1 | Completion token-acc | Anomaly F1 |
|---|---|---|---|
| **SiliconGPT (1.37M)** | **0.811** | **0.405** | **1.000** |
| n-gram (trigram) | 0.761 | 0.283 | — |
| Gemini 3.5-flash | 0.555 | 0.076 | 0.910 |
| GPT-5 | 0.525 | — | — |
| DeepSeek V3 | 0.480 | 0.056 | 0.603 |
| Qwen | 0.415 | 0.024 | 0.690 |

Raw outputs in `extras/results/` (`benchmark_compare.{md,json}`, the submission CSVs, the
co-scientist run record). Validity of generated completions: ~99–100%.

---

## What worked

- **Capacity *reduction* as the OOD lever.** Shrinking 25M → ~1.37M *raised* held-out-family OOD
  with no in-distribution loss — the first lever to move the deciding metric. Smaller models can't
  afford the non-transferable per-family shortcuts a 25M model memorizes.
- **The Experiment agent.** Grounding the multi-agent search in *real* train-and-benchmark runs (not
  argument) turned architecture search into measured, falsifiable evidence — and produced honest negatives.
- **Hybrid anomaly.** Validator + LM perplexity gives perfect rule attribution *and* shows the model
  genuinely learned process logic (LM-only ROC-AUC ≈ 0.995).

---

## What didn't work

Five levers were tried and **rejected** with measured evidence (each rules out a tempting direction):
- **Description-init embeddings** (−0.018 OOD), **cross-family augmentation** (−0.018),
  **NoPE + augmentation** (−0.009; NoPE alone neutral), **weight-sharing** (−0.009).
- **Constrained decoding**: a no-retrain diagnostic showed **~97% of OOD errors are valid-but-wrong**
  (the model picks a *legal* but incorrect step), so masking can't help. This pins the residual as a
  **transition-structure** gap, not a decoding/data/embedding problem.

---

## What you'd do with another 36 hours

- Extend the scaling curve below 1.37M and add stronger regularization at the small size to push the
  capacity-removal lever further.
- Add a family-inference auxiliary head + an OOD-detector (flag unseen-family inputs).
- Run the organizers' `eval_metrics.py` on the official eval files to lock the headline numbers, and
  multi-seed every lever, not just the winner.

---

## Track-specific deliverables (Industrial AI / Infineon)

- [x] Eval submission files in `extras/results/`: `nextstep.csv`, `completion.csv`, `anomaly.csv` (organizer format)
- [x] Training artifacts: checkpoints + `extras/results/*/train_log.csv` (loss curves)
- [x] Baseline (25M) vs trained/optimized (1.37M) comparison made explicit (above + `extras/results/`)
- [x] Demo: baseline-vs-trained, single-step + batch, in the frontend + backend
- [ ] Scores from the official `eval_metrics.py` — run on the organizers' eval files at submission

---

## Credits & dependencies

- **Open-source libraries:** PyTorch (≥2.2), NumPy, PyYAML, tqdm, Flask + flask-cors (demo backend),
  scikit-learn / matplotlib (plots), wandb (optional). Frontend: React 19, TanStack Start, Vite, Tailwind v4, recharts.
- **Pre-trained models used:** none (trained from scratch). Frontier LLMs (Gemini/GPT-5/DeepSeek/Qwen)
  used only as *benchmark baselines*, not in the product.
- **External APIs:** none at inference. LLM-baseline scoring used provider APIs offline.
- **AI coding assistant:** Claude Code (during the hackathon).
- **Datasets:** organizer-provided process grammar + our synthetic sequences (`build_datasets.py`).

---

## A note on honesty

Checkpoints and the dataset are gitignored but regenerate deterministically. The OOD gain is **modest
(+0.008, 3-seed mean)** — real and reproducible, not a large jump; the bigger wins are efficiency
(~18× smaller) and the rigorous, measured discovery process with five honest negatives. The UI's
Block-level Accuracy is our 5-step-window interpretation; the official figure comes from
`eval_metrics.py`. Anomaly uses the deterministic validator (part of our deployed system), reported
alongside the LM-only number.

---

*Submitted by team **bounded** (project: SiliconGPT) for Zero One Hack_01, 2026-05-31.*
