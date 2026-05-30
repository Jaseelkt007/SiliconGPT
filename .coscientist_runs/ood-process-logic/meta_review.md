# Meta-review — Round 1

## Tournament ranking (Elo, after round-1 matches)
| rank | id | elo | status | verdict | lever |
|---|---|---|---|---|---|
| 1 | **h8** | 1261 | **full** | **improves-ood** | scale DOWN to ~6M |
| 2 | h7 | 1231 | unvalidated | — | validator-guided constrained decoding |
| 3 | h4 | 1217 | unvalidated | — | family-dropout + UNKNOWN row |
| 4 | h2 | 1216 | unvalidated | — | Universal-Transformer weight sharing |
| 5 | h1 | 1201 | full | neutral | NoPE (+0.004) |
| 6 | aug_xfam | 1185 | full | neutral(reject) | cross-family aug (−0.018) |
| 7 | h9 | 1185 | unvalidated | — | test-time training |
| 8 | h5 | 1184 | unvalidated | — | adversarial family head |
| 9 | control_v2 | 1184 | full | neutral | baseline |
| 10 | h3 | 1168 | unvalidated | — | QK-norm |
| 11 | h6 | 1167 | unvalidated | — | stage-contrastive |

## Empirically CONFIRMED (measured, full 3-fold OOD)
- **h8 — scaling DOWN works (the round's discovery).** ~6M (n_layer=4,n_embd=256) vs 25M: OOD top1 **0.4947→0.5119 (+0.017)**, OOD top5 +0.017, **OOD completion token-acc 0.170→0.221 (+30% rel)**, all 3 folds up, ID flat (val_loss 0.329). Confirms memorization-vs-structure: in-dist saturated ⇒ excess capacity stored non-transferable per-family shortcuts. **This is now the base config for every other lever.**
- **h1 — NoPE neutral** (+0.004 OOD, +0.006 ID). Safe, free, slightly positive; keep as default, not a driver.
- **aug_xfam — rejected** (−0.018 OOD), top5-up/top1-down signature.
- **Cross-cutting (3 measured negatives + 1 win):** data/embedding/position levers (desc-init, aug, NoPE) don't move OOD top1; the one thing that did was **removing capacity**. The OOD failure is a rank-1 transition residual + a memorization-capacity effect.

## Still SPECULATIVE (top reasoned, ranked to measure next)
- **h7 (constrained decoding)** — highest-ceiling, accuracy-SAFE, no-retrain; directly converts the persistent top5-up/top1-down signature into top-1. Cheap diagnostic first (no code): fraction of OOD top-1 errors that are grammar-INVALID (recoverable).
- **h4 (family-dropout+UNKNOWN)** — smallest objective change; built-in dose-response falsifier (p=0 should hurt, p=1=baseline).
- **h2 (weight-sharing)** — attacks per-layer memorization structurally; synergistic with h8 (both reduce effective capacity).

## Next Tier-2 queue (K=4, run on the ~6M base where applicable)
1. **h8-extend — push scaling further:** add ~2-3M (n_layer=3,n_embd=192) and ~15M (n_layer=6,n_embd=384) points → is OOD monotonic-decreasing in size, or is ~6M the sweet spot? CONFIG-EXPRESSIBLE, zero code. **Highest value, cheapest.**
2. **h7-diagnostic — constrained-decoding recoverability probe** (no retrain): on the ~6M OOD checkpoints, bucket OOD top-1 errors into grammar-invalid (recoverable) vs valid-but-wrong. Gates whether to build full constrained decoding. Tiny code.
3. **h4 — family-dropout+UNKNOWN at ~6M** (needs ~35-45 line knob; one commit): dose-response p∈{0,0.15,0.3,1.0}.
4. **h2 — weight-sharing at ~6M** (needs ~25-35 line knob): does structural capacity-removal stack with size capacity-removal?

## needs-code blocking measurement (build one knob each, then auto-testable)
h7 (decode mask, generate.py), h4 (family embedding+dropout, model.py+dataset.py), h2 (shared block, model.py), h6/h5 (aux losses, larger). Build h7-diagnostic + h4 + h2 knobs next; they unlock the queue.

## Guidance appended to all agents next round
- Re-baseline everything at **~6M**, not 25M — the capacity finding changes the reference point.
- Prefer **accuracy-safe / inference-time** levers (h7, h9) and **capacity-reduction** (h8, h2) — where measured signal is.
- Treat any new data-augmentation / embedding-init idea as low-prior (3 negatives).
- Always report top1 AND top5: the top5-up/top1-down gap is the diagnostic of a rank-1 residual that inference-time levers can still capture.

---

# ROUND 2 + FINAL — closure (synthesis of both rounds)

## Empirically confirmed
- **Capacity reduction improves OOD (the result).** Scaling curve 25M 0.4947 → 15M 0.5008 → 6M 0.5119 →
  3M 0.5120 → 1.4M 0.5139, monotonic, ID flat ~0.80. **Deliverable 3M NoPE seed-confirmed OOD 0.5163 ±
  0.0017** (seeds 42/43/44) = +0.022, in-dist *up* to 0.821, 8.4× smaller. **full-confirmed, multi-seed.**

## Rejected this round (measured, single-seed)
- **h7 constrained decoding** — only ~3% of OOD errors grammar-invalid (97% valid-but-wrong) → masking
  can't help. Reframes the gap as transition-structure, not decoding.
- **h2 weight-sharing** — −0.009 at 3M; tying layers ≠ shrinking size.

## Final scoreboard
1 win (capacity ↓) / 5 principled negatives (desc-init, cross-fam aug, NoPE-as-driver, constrained
decoding, weight-sharing). The OOD residual is a hard transition-structure gap: the model picks the wrong
*legal* step on the unseen family.

## Tier-2 queue → EMPTY (run concluded)
The deciding lever is shipped. The one untested round-1 idea (h4 family-dropout+UNKNOWN, needs-code) is
deferred with a modest honest prior given five negatives. Future directions in `research_overview.md`.

## Integrity
OOD wins are seed-confirmed (sd 0.0017). Two mid-session unverified figures (a "0.531 @ 3M peak", an
"h7 ~52% recoverable") were caused by login-node tmpfs corruption, caught and **retracted**; they never
entered git/results. Every number here is re-read from a committed JSON on persistent storage.
