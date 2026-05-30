# Phase A + B Plan — Finish V1 (validity) → Optimize (RL + constrained decoding)

> Goal of this stage: make **process-validity** a first-class, reported axis, then **optimize** the
> model so it generates valid sequences — *without regressing accuracy* — and produce the rubric's
> "baseline → trained → **optimized**" third rung. Everything logged to **Weights & Biases** in real time.

## Honest framing (what we are and aren't improving)
- Validity ≠ accuracy. A high-accuracy model can still emit a rule-breaking sequence — so validity is a
  separate, important axis we have not yet *reported*. (We measured 100% greedy-completion validity once;
  we now make it a tracked metric, including under sampling / free generation, where it may be < 100%.)
- The base LM is already near the entropy floor in-distribution (top-1 capped by genuine grammar
  branching), so RL will **not** materially move top-1 / exact-match. RL's payoff is: (a) robust validity
  under diverse generation, (b) the "optimized" rung, (c) showcasing RL-from-verifiable-rewards (RLVR)
  with our deterministic validator. **Success = validity → ~100% with no accuracy regression**, not a
  leaderboard jump.

## Two mechanisms for "valid generation" (we use both)
| Mechanism | Weights changed? | Guarantee | Role |
|---|---|---|---|
| **Constrained decoding** | No | **Hard 100%** (mask invalid next tokens) | cheap, accuracy-safe; do first |
| **RL (RFT → GRPO)** | **Yes** | Soft (validity baked into weights) | the "optimized" model + demo + RLVR story |
Constrained decoding can never hurt accuracy (the true next step is always valid, so masking invalid
tokens only removes wrong options). RL bakes a soft prior into the weights for *unconstrained* generation.

---

## PHASE A — finish V1 (validity + honest anomaly)

### A1. Validity as a first-class metric
- **New:** `src/process_logic/validity.py` — `sequence_validity(steps) -> (is_valid, violated_rules)` (thin
  wrapper over `generation.validate_sequence`), and `batch_validity(list_of_step_lists) -> {valid_frac,
  per_rule_counts}`.
- **New:** `scripts/measure_validity.py` — loads `best.pt`, measures the **% of generations that pass all
  10 rules** in three regimes:
  1. **greedy completion** (prefix→greedy) — confirm ≈ 1.00
  2. **sampled completion** (temp 0.8 / 1.0 / 1.2) — expect < 1.00; this is RL's headroom
  3. **free generation** (from `<BOS>`, sampled) — full-recipe validity
  Reports `valid_frac` + the breakdown of which rules break.
- **Test:** `tests/test_validity.py` — valid reference sequences → 1.0; a known-corrupted one → flagged.
- **Report:** add a "Validity" subsection to `REPORT.md` with the three numbers.
- *Improves:* nothing numeric yet — it **exposes** the validity axis and quantifies RL's headroom.

### A2. LM-only anomaly pass (honest model evidence)
- Run `predict.py --no-validator --anomaly-input data/eval_anomaly.csv --calib-file data/val_id.csv`
  → `anomaly_lmonly.csv`; score it.
- **Report:** show perplexity-based F1 / ROC-AUC **next to** the hybrid (so the validator-perfect 1.0 is
  clearly attributed to the deterministic checker, and the *model's own* anomaly ability is reported).
- *Improves:* honesty/credibility of the anomaly result.

---

## PHASE B — optimize (constrained decoding + RL)

### B0. Constrained decoding (decode-time validity — cheap, accuracy-safe; do first)
- **`generate.py`:** add `valid_next_mask(prefix_steps, vocab) -> bool[V]` and a `--constrained` path in
  `rank_next_steps` / `complete_sequence` that sets invalid candidates to `-inf` before argmax/top-k.
  - Local rules (clean-before-deposit, mask-before-etch, CMP-after-deposit, implant-after-mask,
    metal-etch-after-litho) are checkable **incrementally** in the trailing window.
  - Global ordering rules (ship-after-test, test/backside/pad after passivation, litho-level order) are
    enforced as generation-time guards (don't emit SHIP LOT before WAFER SORT TEST, etc.).
- **Measure:** validity (→ should be 1.00 even under sampling) and the **scored completion metrics**
  (token-acc / edit-dist should improve or hold — never worse).
- *Improves:* validity → 100% guaranteed; potentially Task-1 top-k and Task-2 completion (accuracy-safe).

### B1. Rejection-Sampling Fine-Tuning (RFT) — the simple, stable RL
- **New:** `src/process_logic/rl.py` (RFT loop) + `scripts/run_rl.sh` (Slurm).
- Loop (1–3 iterations), starting from `best.pt`:
  1. Build prompts = partial sequences (60/80% cuts from train, balanced by family).
  2. Sample **N=16–64** completions per prompt (temp ~1.0).
  3. Keep only **validator-valid** completions (`validate_sequence` == clean).
  4. **SFT** on (prefix + valid completion), **mixed with original valid data** (avoid forgetting).
  5. Re-evaluate; repeat.
- **WandB (on):** log `valid_frac`, train loss, and the **3 task metrics + val loss each round**
  (regression guard).
- *Improves:* free/sampled-generation validity baked into the weights.

### B2. GRPO (only if RFT plateaus)
- **TRL `GRPOTrainer`**, starting from the RFT checkpoint:
  - reward = `1.0 if valid else 1 - violations/10` (partial credit via `validate_sequence`).
  - `num_generations=8`, `loss_type=dr_grpo`, `beta≈0.01` (KL keeps it near the accurate base), `lr≈5e-6`.
  - Monitor `entropy` (collapse) and `frac_reward_zero_std` (no-signal prompts) in WandB.
- *Improves:* validity beyond RFT; the strongest "optimized" result.

### B3. Optimized deliverable
- Save `checkpoints/optimized/best.pt`; produce the 3 submission CSVs with it.
- **Report:** a **base vs. optimized** table: `valid_frac` (greedy/sampled/free) ▲ and the 3 task metrics
  (must be ≈ unchanged or ▲ — the regression guard). This is the "baseline → trained → optimized" rung.

---

## What WandB tracks (real-time)
`valid_frac` (greedy / sampled / free), `val_loss`, `top1/3/5`, `MRR`, completion `token_acc`/`edit_dist`,
`grad_norm`; for GRPO also `reward`, `entropy`, `frac_reward_zero_std`. Enable with `--wandb`
(`WANDB_MODE=offline` on compute nodes, then `wandb sync` from a login node; or online via the proxy).

## Regression guard (ensure accuracy doesn't drop)
Every RL round re-scores Task-1/2 on `data/eval_*.csv`. **If top-1 drops > 0.5%**, stop and raise KL
(GRPO) or add more original data (RFT). Constrained decoding is the accuracy-safe fallback regardless.

## Risks
- *Reward hacking / mode collapse* (model emits one trivial valid sequence) → KL penalty + entropy
  monitor + keep prompt diversity.
- *Validity already ~100% greedy* → RFT/GRPO headroom may be small; the real win shows under **sampling**
  and **free generation** (which is why A1 measures those first).
- *Constrained-decoding global rules* are fiddly → start with local-rule masking + end-of-sequence guard.

## Definition of done
1. `validity.py` + `measure_validity.py` + report rows (greedy/sampled/free).
2. LM-only anomaly numbers reported beside hybrid.
3. Constrained decoding: validity = 1.00, completion metrics ≥ baseline.
4. RFT (and GRPO if used): `optimized/best.pt` + base-vs-optimized table, accuracy not regressed.
5. All runs in WandB; `REPORT.md` updated; clean commits.

## V1 status note
V1 (base LM + 3 tasks + OOD baseline + honest report) is complete; early stopping (patience 8) correctly
ran the full 4000 iters because val_loss kept improving monotonically (0.339→0.329), never plateauing.
A full V2 retrain (optionally larger / longer / with these levers) is a later step.
