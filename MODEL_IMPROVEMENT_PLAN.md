# Model-Improvement Plan (V2) — co-scientist-driven, OOD-focused

> **Purpose:** continue the model-improvement discussion in a new chat. Read this together with
> `CLAUDE.md`, `extras/results/benchmark.md`, `DECISIONS.md`, `V2_RL_PLAN.md`. The plan: use the
> **co-scientist** multi-agent skill to generate → debate → tournament-rank → evolve improvement
> hypotheses, and **empirically validate the config-expressible ones by actually training + benchmarking**
> (training converges in ~minutes, so we test, not just argue). co-scientist proposes; our benchmark disposes.

---

## 1. Where we are (state at handoff)
- **Model:** 25M from-scratch decoder (RMSNorm/RoPE/SwiGLU), custom 202-token vocab. Near the entropy
  floor in-distribution; **100% valid completions**; LM-only anomaly ROC-AUC **0.997**.
- **Benchmark** (`benchmark.md`, held-out, common examples): ours ≳ n-gram ≫ Gemini **in-distribution** —
  but **our V1 is only marginally above a trigram in-distribution** (next-step ~tied; completion 0.40 vs 0.28).
- **OOD is the deciding, mostly-open axis:** V1 OOD next-step ~**0.50** (3-fold avg 0.495); a large gap.
- **Negatives already recorded** (`DECISIONS.md`): description-init for OOD **rejected**; RL **deferred**
  (no validity headroom).
- **Proper baselines:** previous-best checkpoint (primary target) · n-gram (floor) · Gemini 3.5-flash
  (OOD/frontier bar). **No baseline was provided by the committee** — only the generator/validator.

## 2. What we're improving — the matrix
| Task | Metrics | Beat | Goal dimension |
|---|---|---|---|
| Next-step | Top-1/3/5, MRR | prev-best · n-gram · Gemini | in-dist · **OOD** |
| Completion | Exact-match, Norm-edit-dist, Token-acc, Block-acc | prev-best · n-gram · Gemini | in-dist · **OOD** · validity |
| Anomaly | Acc/P/R/F1/ROC-AUC (LM-only **and** hybrid), Rule-attr | prev-best · Gemini | reasoning · OOD |
| **Validity** | % rule-valid (greedy/sampled/free) | prev-best | reliability |
| **Family-detection** *(new)* | accuracy of inferring mosfet/igbt/ic from a prefix | — | "which transistor?" |
| **OOD** *(deciding)* | **ID→OOD drop** of every metric above | prev-best · Gemini | the headline target |
| Efficiency | params · latency · train-compute | — | sovereignty |

Score on the full `eval_*` (3600/600/987) **+** the committee's 3,000 `*_variants.csv` (extra held-out
cross-check) **+** the 3-fold OOD. **Family-detection and OOD are the new headline rows.**

## 3. The co-scientist approach (how we use the skill)
Skill at `~/.claude/skills/co-scientist/` — Supervisor orchestrates **Generation → Reflection → Proximity
→ Ranking (Elo tournament/debate) → Evolution → Meta-review**, file-based memory, 2 expert checkpoints,
quality scales with rounds. Retarget by swapping the **criteria rubric** + the goal.

**Draft research goal (paste into the skill):**
> *"Discover the architecture / data / training / conditioning changes that make our small (~10–50M)
> from-scratch process-sequence transformer (a) generalize to an unseen 4th product family (OOD),
> (b) infer the family (mosfet/igbt/ic) from context rather than memorize per-family patterns, and
> (c) beat our baselines on the benchmark — while staying small, from-scratch, not overfitting, and not
> regressing in-distribution. Background: benchmark.md, DECISIONS.md (desc-init negative, RL deferred),
> V2_RL_PLAN.md §11."*

**Draft criteria rubric (discriminating + hard constraints):**
1. **OOD-generalization potential** — plausibly closes the ID→OOD gap (the deciding axis).
2. **Buildability** — implementable + testable in our loop on the A100s.
3. **Benchmark-measurable / falsifiable** — validated by `benchmark_table.py` + the OOD run.
4. **Overfit-resistance** — reduces, not increases, memorization.
5. **Efficiency / sovereignty** — stays small & from-scratch (NOT "use a big LLM").
6. **Novelty + grounding** — web-checked vs the compositional-generalization literature.
- **Hard constraints (must/never):** from-scratch · must beat the n-gram floor · must NOT regress
  in-distribution · small enough to train in minutes on 1 A100.

**Loop:** Generation seeds ideas (see §6) → Reflection kills flawed/non-novel → Ranking debates pairwise
(Elo) → Evolution combines top ideas → Meta-review synthesizes. **Expert checkpoint #1** (confirm the
config) before round 1; **#2** (pick winners) at the end. Run a *focused* few-round pass (not a marathon).

## 4. Experimental co-scientist — give it training access (LOCKED → see `CO_SCIENTIST_LAB_DESIGN.md`)
**Decided 2026-05-30:** we fork the skill into `process-logic/.claude/skills/co-scientist-lab/`, run it
**mostly autonomously** with a **tiered validation ladder** (reasoning → local smoke → Leonardo full), and
ingest Leonardo results back each round. Full design + build checklist in **`CO_SCIENTIST_LAB_DESIGN.md`**.
Summary below.

Because training is fast, the loop is **empirically validated**, not debate-only:
- **Build `scripts/run_experiment.py`** — input: a config (model size, positional-encoding, family-conditioning
  on/off + dropout, augmentation toggle, objective, `--exclude-family` for OOD). It **trains** (early-stop,
  small `max_iters`), **predicts**, and **benchmarks in-dist + 3-fold OOD**, returning the **metric matrix as
  JSON**. (We already have config-driven `train.py`, `score.py`, `benchmark_table.py` — this stitches them.)
- **Wire it into the loop:** config-expressible hypotheses get **auto-trained + auto-benchmarked**; the
  **real metrics feed the Elo ranking + meta-review** (the tournament becomes empirical). Non-config ideas
  (need real code) → human implements, then benchmark.
- **Guardrails:** only config-expressible ideas auto-run · cap to **top-K hypotheses/round** · each run
  bounded (early-stop, small max_iters) · **one commit per experiment** (clean revert) · the **benchmark is
  the judge** — keep winners, **revert any regression** (esp. in-distribution / vs n-gram).

## 5. Data strategy (does "more data" help?)
- **In-distribution: no** — we're at the entropy floor; more of the same = diminishing returns.
- **For OOD: diversity, not volume** — cross-family recombination + uniform sampling over the variation
  axes; data that breaks family-global shortcuts and forces compositional generalization.
- **Include the committee's 3,000 `*_variants.csv` + reference files** — same distribution (little
  statistically) but good for completeness/credibility and as a fixed held-out cross-check.
- **Data + architecture co-evolve** — extra signal only helps if the inductive bias can exploit structure.
  Treat data strategy as part of the co-scientist hypothesis space.

## 6. Seed hypothesis space (for Generation — co-scientist will extend/debate)
- **Family-conditioning** = small separate embedding + `UNKNOWN_FAMILY` row + family-dropout (~15%) so it
  helps in-dist without breaking OOD.
- **Positional-encoding ablation** — RoPE vs **NoPE** vs randomized positions (length/OOD generalization).
- **Compositional inductive bias** — QK-norm + orthogonal-V loss (Csordás), Universal-Transformer / weight
  tying for systematic generalization.
- **Auxiliary family-inference head** (predict mosfet/igbt/ic from the prefix) → the "which transistor?" goal.
- **OOD detector** — flag unseen-family input via perplexity / unseen-token rate / low max-prob.
- **Cross-family recombination augmentation** (GECA-style, uniform sampling).
- **Constrained decoding** (B0) + **per-family anomaly threshold** (C1) — cheap, accuracy-safe wins.
- **Scaling study** — size × data; let the curve decide small-vs-larger.
- **Contrastive / objective changes** for family-invariant representations.

## 7. Explicit sub-goals
- **Family detection:** infer mosfet/igbt/ic from a partial sequence (auxiliary head or implicit).
- **OOD detection:** recognize when input is from an unseen family and handle gracefully.
- **Close the ID→OOD gap** (the deciding metric) — the headline.

## 8. How to start the next session
1. Read `CLAUDE.md` + this doc + `benchmark.md` + `DECISIONS.md`.
2. (If giving training access) build `scripts/run_experiment.py` (config → train → benchmark → JSON).
3. Invoke **co-scientist** with the goal + criteria (§3) + the grounding pack; confirm config (checkpoint #1).
4. Run a focused few-round pass → ranked shortlist + meta-review.
5. Checkpoint #2: pick top 2–3 → implement / auto-test via `run_experiment.py` → benchmark → keep/revert.
6. Iterate; update `benchmark.md`, `DECISIONS.md`, `REPORT.md` as we learn.

## 9. Open decisions to confirm at session start
- **Training access for co-scientist?** (recommended **yes** — auto-test config-expressible hypotheses.)
- **Model-size direction** (smaller vs larger) — let the scaling study + co-scientist decide, not a guess.
- **Compute / round budget** for the experiment loop.
- **Scope of co-scientist run** (number of rounds, fan-out) given test-time-compute cost.
