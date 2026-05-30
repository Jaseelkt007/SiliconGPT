"""Process-validity metric — is a generated sequence rule-valid?

Distinct from accuracy: a high-accuracy model can still emit a rule-breaking sequence.
This wraps the deterministic `validate_sequence` (the 10 rules) into a first-class metric
we can report (greedy / sampled / free-generation validity). Pure stdlib — no torch.
"""
from __future__ import annotations

from collections import Counter

from process_logic import generation as G


def sequence_validity(steps):
    """Return (is_valid: bool, violated_rules: list[str])."""
    viol = G.validate_sequence(steps)
    return (len(viol) == 0, [v.rule for v in viol])


def batch_validity(seqs):
    """seqs: list of step-lists. Return {n, valid_frac, per_rule} where per_rule counts how
    many sequences violated each rule (a sequence with multiple violations counts each once)."""
    n = len(seqs)
    valid = 0
    per_rule = Counter()
    for s in seqs:
        ok, rules = sequence_validity(s)
        if ok:
            valid += 1
        else:
            for r in set(rules):
                per_rule[r] += 1
    return {"n": n, "valid_frac": valid / max(1, n), "per_rule": dict(per_rule)}
