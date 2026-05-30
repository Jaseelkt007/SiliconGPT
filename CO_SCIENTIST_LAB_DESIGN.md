# co-scientist-lab — experimentally-grounded discovery skill (design)

> The **updated skill** for the model-improvement section. It keeps the full co-scientist debate loop
> (Generation → Reflection → Proximity → Ranking/Elo → Evolution → Meta-review) and adds an **empirical
> grounding layer**: hypotheses get *measured*, not just argued. Debate stays the value; experiments make
> the debate cite real OOD numbers. Read with `MODEL_IMPROVEMENT_PLAN.md` + `CLAUDE.md`.

## 0. Decisions locked (2026-05-30)
- **Compute path = Hybrid.** Tier-1 smoke runs locally inside the loop (fast directional signal); Tier-2
  full runs on Leonardo. The loop emits a prioritized experiment queue; you `sbatch` it; results ingest
  back for the next round.
- **Packaging = fork into a project skill.** New skill at `process-logic/.claude/skills/co-scientist-lab/`
  (copy of `~/.claude/skills/co-scientist/` + the experiment stage + our criteria preset). The general
  skill stays untouched.
- **Autonomy = mostly autonomous.** The local round runs end-to-end without approval pauses and surfaces
  the final ranked shortlist + meta-review. The only human step is running the Leonardo queue (async Slurm
  is the batch boundary, not a gate). Bounded by a per-round budget.

## 1. The tiered validation ladder (the core idea)
Don't train every hypothesis — async Slurm would stall the loop. Each hypothesis carries **whatever
evidence tier it has earned**, and Ranking weights **measured > simulated > reasoned**:
- **Tier 0 — reasoning** (free, instant): normal debate. Every hypothesis.
- **Tier 1 — smoke** (cheap, local CPU, tiny config, ~150 iters): directional signal. Shortlisted ideas.
- **Tier 2 — full** (Leonardo, real in-dist + OOD + family numbers): finalists only.
Meta-review always records the **confidence tier** of each conclusion.

## 2. The autonomous loop + the Leonardo batch boundary
```
LOCAL ROUND (autonomous, no pauses):
  Generation  -> hypotheses; each config-expressible one carries an EXPERIMENT SPEC (sec 4)
                 + new-capability ideas (family-ID, OOD-detect) as first-class hypotheses
  Reflection  -> correctness + novelty (web) + TRIAGE: auto-testable (config) vs needs-code
  Experiment  -> [NEW] run_experiment.py Tier-1 smoke on top auto-testable candidates -> metrics to memory
  Proximity   -> cluster / dedupe
  Ranking     -> Elo debate; a smoke-confirmed OOD gain outranks speculation; empirical results seed bracket
  Evolution   -> combine winners; propose follow-up ablations
  Meta-review -> synthesize args + measurements; mark confirmed vs speculative; emit the Tier-2 QUEUE
        |
        v  prioritized experiment queue (specs) + ranked shortlist
LEONARDO (batch, you run):  sbatch the queue -> result JSON per spec
        |
        v  elo.py ingest  -> next autonomous round uses real OOD numbers
```

## 3. What the skill gets access to
- **NEW `scripts/run_experiment.py`** — the bridge (sec 4).
- Existing: `train.py`, `generate.py`/`predict.py` (inference), `score.py`, `benchmark_table.py`, the OOD
  3-fold run, `data/`, baselines (`baseline_ngram`, `baseline_gemini`).
- Memory: a new `experiments/` dir in the run folder (one result record per tested hypothesis).

## 4. `run_experiment.py` — spec in, result out
**Experiment spec** (Generation emits this; the harness consumes it):
```json
{
  "id": "hyp_017",
  "title": "family-conditioning + 15% family-dropout",
  "tier": "smoke",                         // or "full"
  "base": "configs/base.yaml",
  "overrides": { "model.family_conditioning": true, "model.family_dropout": 0.15,
                 "data.augmentation": "cross_family_recomb", "pos_encoding": "rope",
                 "train.max_iters": 200 },
  "ood": { "exclude_family": "ic", "folds": 3 },
  "tasks": ["nextstep","completion","anomaly","family_id","ood_detect"],
  "needs_code": false                       // true => not auto-runnable; implement first
}
```
**Result record** (harness writes to `experiments/<id>.json`):
```json
{
  "id": "hyp_017", "tier": "smoke", "status": "done", "commit": "abc123",
  "train": { "final_val_loss": 0.33, "iters": 200, "wall_s": 240 },
  "metrics": {
    "id":  { "nextstep": {"top1":0.81,"top5":1.0,"mrr":0.88}, "completion": {...}, "anomaly": {...} },
    "ood": { "nextstep": {...}, "completion": {...} },
    "family_id": { "acc": 0.0 }, "ood_detect": { "auroc": 0.0 }
  },
  "vs_baseline": { "ngram_top1_delta": 0.03, "prevbest_ood_top1_delta": 0.05 },
  "verdict": "improves-ood"                 // improves-ood | regresses-id | neutral
}
```
Ranking reads `verdict` + the deltas; Meta-review reads the whole record.

## 5. Realism: the config surface bounds what's auto-testable
`run_experiment.py` is only as powerful as the **knob surface** of `train.py`/`model.py`. V1 has RoPE fixed,
no family conditioning, no augmentation switch — so in round 1 **few ideas are config-expressible**. Two
consequences:
- **First build step = widen the config surface** ("knob inventory"): expose `pos_encoding`,
  `family_conditioning`+`family_dropout`, `data.augmentation`, objective flags, size. Each knob added makes
  a whole family of hypotheses auto-testable.
- **`needs_code` triage** (Reflection): a hypothesis needing a *new* knob is implemented once (adds the
  knob, one commit), then it + its variants become config-expressible for later rounds.

## 6. New capabilities = new scored tasks (your "lots of new things")
`run_experiment.py` must be extensible to score new tasks, not just the 3:
- **family_id** — infer mosfet/igbt/ic from a prefix (auxiliary head or frozen probe). Metric: accuracy,
  + OOD variant (does it degrade gracefully on the unseen family?).
- **ood_detect** — flag unseen-family input via perplexity / unseen-token rate / max-prob. Metric: AUROC
  (held-out family as positive class).
- Add others as Generation proposes them (calibration, rule-violation localization, …).

## 7. Memory additions (vs the base skill)
- `experiments/<id>.json` — one result record per tested hypothesis (sec 4).
- `tournament.json` / `hypotheses.json` — extend each hypothesis with `empirical_status`
  (`unvalidated` | `smoke` | `full`) and a pointer to its result record.
- `meta_review.md` — now separates **empirically confirmed** directions from **still-speculative** ones,
  and lists the **next Tier-2 queue**.
- Mutate JSON only via `scripts/elo.py` (add/match/setstatus/ingest/proximity/rank) + a new
  `ingest-experiment` action.

## 8. Criteria rubric (final, for `config.md`)
1. **OOD-generalization potential** — closes the ID→OOD gap (deciding axis).
2. **Buildability** — implementable + testable in our loop.
3. **Empirical status** — unvalidated / smoke-confirmed / full-confirmed (Ranking weights this).
4. **Overfit-resistance** — reduces, not increases, memorization.
5. **Efficiency / sovereignty** — stays small & from-scratch (NOT "use a big LLM").
6. **Novelty + grounding** — web-checked vs compositional-generalization literature.
- **Hard constraints:** from-scratch · beat the n-gram floor · NO in-dist regression · trains in minutes
  on 1 A100.

## 9. Methodology guard: validate smoke -> full correlation once
Smoke (tiny, 150 iters) gives *directional* signal; its OOD ranking may not perfectly match full runs.
**Before trusting smoke for ranking, run ~3 hypotheses both ways and check rank-correlation.** If smoke
mis-ranks, raise the smoke config (more iters / dim) until it tracks. Document the chosen smoke fidelity.

## 10. Build checklist for the next session (ordered)
0. **Already drafted** (this session): `co-scientist-lab/agents/experiment.md` (the new agent) +
   `co-scientist-lab/PROJECT_CONFIG.md` (seed `config.md` — goal + criteria, confirm at checkpoint #1).
1. **Complete the fork** → copy the base skill's remaining files into `co-scientist-lab/` (`SKILL.md`, the 6
   existing agents, `references/`, `scripts/elo.py`+`coscientist_workflow.js`); add the **`ingest-experiment`**
   action to `elo.py` (mirrors `ingest`; attaches the result record + `empirical_status` to a hypothesis);
   wire the empirical hooks — Reflection (config-expressible/needs-code triage handoff to Experiment),
   Ranking (weight measured > reasoned), Meta-review (separate confirmed vs speculative + emit the Tier-2 queue).
2. **Widen the config surface** in `model.py`/`train.py` (pos_encoding, family_conditioning+dropout,
   data.augmentation, objective, size) — the knob inventory.
3. **Build `scripts/run_experiment.py`** (spec → train → inference → score id+ood+family_id+ood_detect →
   JSON), with `--tier smoke|full` and `--exclude-family`.
4. **Add the new eval tasks** (family_id, ood_detect) to `score.py` + `benchmark_table.py`.
5. **Smoke→full correlation check** (sec 9).
6. **Run co-scientist-lab** autonomously (goal + criteria from `MODEL_IMPROVEMENT_PLAN.md` §3) → shortlist
   + Tier-2 queue.
7. **`sbatch` the queue on Leonardo** → ingest results → next round.
8. Update `benchmark.md`, `DECISIONS.md`, `REPORT.md`; keep winners, revert regressions (one commit each).

## 11. Budget & guardrails
- Cap **K Tier-2 experiments / round** + **max rounds** (set at session start).
- One commit per experiment (clean revert); benchmark is the final judge.
- Hard constraints in the rubric keep Generation from proposing big-LLM / non-sovereign ideas.
- Token budget for the autonomous loop set per round.

## 12. Model allocation — which Claude runs each agent (decided 2026-05-30: TIERED)
Quality scales with rounds (the paper's core finding); Opus is slower, so put the best model where reasoning
**decides the outcome** and a fast model on the **mechanical** agents → more rounds inside the deadline. NB:
the **agent LLM ≠ the trained model** — the GPU does the training; the agent LLM only orchestrates/reasons.

**Session:** launch the server session on **Opus 4.8** → the Supervisor and any *inherited* subagent default
to it. Run with **extended thinking ON**. Reasoning *effort* is **session-level + prompt-elicited**, NOT a
per-subagent dial (the `Agent` tool exposes `model`, not effort) — the rich agent prompts do the work.

**Per-agent** (the Supervisor passes `model=` on each `Agent` call; values: `opus`/`sonnet`/`haiku`):
| Agent / mode | Model |
|---|---|
| Supervisor (main loop) | **opus** (4.8, session) |
| Generation | **opus** |
| Reflection — full + deep-verification | **opus** |
| Reflection — quick initial filter | sonnet |
| Ranking — multi-turn debate (top pairs) | **opus** |
| Ranking — single-turn (low-stakes pairs) | sonnet |
| Evolution | **opus** |
| Meta-review | **opus** |
| Experiment | sonnet (escalate ambiguous triage / new-task design to the Supervisor) |
| Proximity | haiku |

**Implement:** in the lab's `SKILL.md` orchestration section, set `model=` per the table on every `Agent`
spawn (mode-dependent for Reflection and Ranking). Part of the wiring/build step (§10 / §1 of the checklist).
