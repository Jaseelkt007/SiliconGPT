# Round 1 — Reflection (triage + filter)

All 9 hypotheses PASS (none discarded). Empirical triage + priority for the K=4 Tier-2 queue.

| id | summary | triage | score | novelty | priority | biggest risk |
|----|---------|--------|-------|---------|----------|--------------|
| h1 | NoPE (pos_encoding=nope) | **config-expressible** | 8 | 5 | 9 | gain may be noise; NoPE degrades at long ctx |
| h2 | Universal-Transformer weight sharing | needs-code (~25-35 ln) | 7 | 7 | 7 | matched-n_layer cuts params → may regress saturated in-dist |
| h3 | QK-norm + attn temp | needs-code (~12-18 ln) | 6 | 6 | 6 | qk-norm mainly stability; may be a no-op on OOD |
| h4 | Family-dropout + UNKNOWN row | needs-code (~35-45 ln) | 8 | 6 | 9 | UNKNOWN row averages seen families → partial shortcut may remain |
| h5 | Adversarial family head (GRL) | needs-code (~55-75 ln) | 6 | 8 | 6 | GRL unstable on tiny model/minutes; pooled invariance ≠ token-local |
| h6 | Stage-aligned SupCon | needs-code (~65-95 ln) | 7 | 8 | 7 | HINGES on a clean stage map; generation.py sets overlap/incomplete |
| h7 | Validator-guided constrained decoding | needs-code (~60-100 ln) | **9** | 6 | **9** | only helps if OOD argmax errors are grammar-INVALID (diagnostic first) |
| h8 | Model-size sweep | **config-expressible** | 6 | 4 | 5 | likely within seed noise; indirect (not the rank-1 failure) |
| h9 | Test-time training on prefix | needs-code (~80-120 ln) | 7 | 8 | 6 | K inner steps can overfit/destabilize rank-1; per-example latency |

## Triage outcomes
- **Config-expressible NOW (auto-testable):** h1 (DONE — measured, neutral +0.004), h8 (size sweep; ~6M arm already in the levers job).
- **Needs-code (one-commit knob, then testable):** h2, h3 (cheapest, both in model.py Attention/Block), h4 (smallest objective change), h5, h6, h7, h9.
- **Accuracy-safe / no-retrain standouts:** h7 (constrained decoding — can only hold/improve top1) and h9 (TTT) directly exploit the measured top5-up/top1-down signature. h7 has a cheap diagnostic to run FIRST (no code): what fraction of OOD top-1 errors are grammar-INVALID (recoverable).

## Cross-cutting note for Ranking/Meta-review
Two measured levers so far (aug −0.018, NoPE +0.004) both fail to move OOD top1 → the rank-1 transition residual is real and resistant to training-side levers tried. This RAISES the priority of **inference-side, accuracy-safe** levers (h7, h9) and the **representation-invariance** levers that attack the shortcut directly (h4, h5, h6), and LOWERS pure-capacity h8 / stability-only h3.
