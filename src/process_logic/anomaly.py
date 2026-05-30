"""Anomaly detection: LM perplexity + deterministic validator hybrid.

We report two flavours:
  - model-only (LM surprisal)  -> evidence the model *learned* process logic
  - hybrid (LM + validate_sequence) -> best score + exact rule attribution
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from process_logic import generation as G   # validate_sequence  # noqa: E402


@torch.no_grad()
def sequence_nll(model, vocab, steps, device="cpu"):
    """Mean per-token negative log-likelihood of a full sequence (lower = more normal)."""
    ids = vocab.encode(steps, add_bos=True, add_eos=True)
    ids = ids[:model.cfg.block_size]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    logits, _ = model(x)
    nll = F.cross_entropy(logits[0, :-1], torch.tensor(ids[1:], device=device), reduction="mean")
    return float(nll)


@torch.no_grad()
def calibrate_threshold(model, vocab, valid_step_lists, device="cpu", percentile=95.0):
    """Pick an NLL threshold at the given percentile over known-valid sequences."""
    scores = sorted(sequence_nll(model, vocab, s, device) for s in valid_step_lists)
    if not scores:
        return None
    idx = min(len(scores) - 1, int(len(scores) * percentile / 100.0))
    return scores[idx]


@torch.no_grad()
def score_anomaly(model, vocab, steps, device="cpu", threshold=None, use_validator=True):
    """Return (is_valid:int, score:float = P(valid) in [0,1], predicted_rule:str)."""
    nll = sequence_nll(model, vocab, steps, device)
    if threshold is not None:
        score = 1.0 / (1.0 + math.exp((nll - threshold) * 4.0))  # high nll -> low P(valid)
    else:
        score = math.exp(-nll)

    if use_validator:
        viol = G.validate_sequence(steps)
        if viol:
            return 0, min(score, 0.05), viol[0].rule
        return 1, max(score, 0.95), ""

    is_valid = 1 if (threshold is None or nll <= threshold) else 0
    return is_valid, score, ""
