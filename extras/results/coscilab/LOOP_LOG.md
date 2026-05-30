# co-scientist-lab — loop log (what we tried / improved / learned, per round)

> Human-facing distillation of the OOD model-improvement loop. The skill also keeps raw memory in
> `.coscientist_runs/ood-process-logic/` (hypotheses.json, tournament.json, experiments/*.json,
> meta_review.md — round 1). **All numbers below verified from the experiment JSONs on disk.**
> Baseline = 25M (8L/512): OOD next-step top1 **0.4947**, ID top1 **0.8072**.

## Round 1 — method online + first discovery
**Tried (4 measured, full 3-fold OOD):** control · cross-family augmentation · NoPE · scale-down (6M, 4L/256).
- **IMPROVED:** scale-down to ~6M → OOD top1 **+0.017** (0.5119), OOD completion **+30% rel** (0.170→0.221),
  ID flat. First lever to move the deciding metric.
- **NEGATIVE:** cross-family aug **−0.018** (rejected); NoPE **+0.004** (neutral).
- **LEARNED:** in-dist is saturated ⇒ 25M's excess capacity memorizes non-transferable per-family
  shortcuts. Data/embedding/position *placement* don't move OOD top1; capacity **removal** does.
  Re-baselined the search at the small size.

## Round 2 — scaling curve + error decomposition
**Tried:** scaling curve {1.4M, ~2–3M, 6M, 15M} · h7 constrained-decoding diagnostic · NoPE+aug.
- **MAPPED the curve** (gain plateaus across small sizes, fades by 15M — NOT monotonic-to-1.4M):

  | config | OOD top1 | ID top1 | OOD compl | verdict |
  |---|---|---|---|---|
  | 1.4M (3L/128) | 0.5139 (+0.019) | 0.8014 (−0.006) | 0.182 | **regresses-id** |
  | ~2–3M (3L/192) | 0.5120 (+0.017) | **0.8111 (+0.004)** | 0.196 | **improves-ood** ✅ best all-around |
  | 6M (4L/256) | 0.5119 (+0.017) | 0.8044 | **0.221** | improves-ood (best OOD completion) |
  | 15M (6L/384) | 0.5008 (+0.006) | 0.8089 | 0.209 | neutral |

  **Sweet spot for the submission (OOD up, ID not regressed) = ~2–3M (3L/192) or 6M (4L/256). NOT 1.4M.**
- **KEY DIAGNOSIS (h7):** only **~3%** of OOD top-1 errors are grammar-invalid; **~97% are valid-but-wrong**.
  ⇒ the gap is **genuine transition-structure learning** (model picks the wrong *legal* step on an unseen
  family), NOT a decoding/validity problem. Constrained decoding **rejected**.
- **NEGATIVE:** NoPE+aug **−0.009** (rejected).

## Cumulative (gain from both loops)
- **Doesn't work** (4 principled negatives): data augmentation · embedding-init · NoPE · constrained decoding.
- **Works:** capacity **removal** (scale-down), saturating at a few-M.
- **Nature of the gap:** a rank-1 transition-structure residual, amplified by excess capacity.
- **Remaining promising lever class:** other capacity-removal / regularization targeting transition
  structure — **h2 weight-sharing** (built, untested) and stronger regularization. Not data/embeddings/decoding.
- **SUBMISSION MODEL:** small architecture (**~2–3M, 3L/192**, or **6M, 4L/256**) trained on ALL 3 families.
  The held-out-family OOD is the proxy; the bet is it transfers to the hidden 4th family.
