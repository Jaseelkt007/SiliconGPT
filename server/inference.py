"""Thread-safe inference wrapper around the trained ProcessLM checkpoint.

Loads one model + vocab at process start, then exposes pure functions that the
Flask routes call. All torch calls go through `_LOCK` so concurrent requests
serialize on a single CPU/GPU.
"""
from __future__ import annotations

import math
import os
import sys
import threading
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from process_logic.vocab import Vocab, SPECIALS                          # noqa: E402
from process_logic.model import ProcessLM, ModelConfig                   # noqa: E402
from process_logic.generate import (                                     # noqa: E402
    load_checkpoint,
    rank_next_steps,
    complete_sequence,
)
from process_logic.anomaly import (                                      # noqa: E402
    sequence_nll,
    calibrate_threshold,
    score_anomaly,
)
from process_logic.dataset import load_compact_csv                       # noqa: E402
from process_logic import generation as G                                # noqa: E402

_LOCK = threading.Lock()


class _State:
    model: Optional[ProcessLM] = None
    vocab: Optional[Vocab] = None
    device: str = "cpu"
    ckpt_path: Optional[str] = None
    threshold: Optional[float] = None
    load_error: Optional[str] = None


STATE = _State()


def init(ckpt_path: Optional[str] = None,
         device: Optional[str] = None,
         calib_path: Optional[str] = None) -> None:
    """Load the checkpoint once at startup. Safe to call again to reload."""
    ckpt_path = ckpt_path or os.environ.get("CHECKPOINT_PATH") or str(ROOT / "checkpoints" / "best.pt")
    device = device or os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    calib_path = calib_path or os.environ.get("CALIB_PATH") or str(ROOT / "data" / "val_id.csv")

    STATE.ckpt_path = ckpt_path
    STATE.device = device

    if not Path(ckpt_path).exists():
        STATE.load_error = f"checkpoint not found at {ckpt_path}"
        return

    try:
        model, vocab = load_checkpoint(ckpt_path, device)
        STATE.model = model
        STATE.vocab = vocab
        STATE.load_error = None
    except Exception as e:                                   # noqa: BLE001
        STATE.load_error = f"failed to load checkpoint: {e!r}"
        return

    if Path(calib_path).exists():
        try:
            valids = [s for _, s in load_compact_csv(calib_path)][:1000]
            STATE.threshold = calibrate_threshold(model, vocab, valids, device)
        except Exception:                                    # noqa: BLE001
            STATE.threshold = None


def ready() -> bool:
    return STATE.model is not None and STATE.vocab is not None


def require_model():
    if not ready():
        raise RuntimeError(STATE.load_error or "model not loaded")
    return STATE.model, STATE.vocab, STATE.device


# ---------- thin wrappers around the existing inference helpers ---------- #

def predict_topk(partial: list[str], k: int = 5) -> list[dict]:
    """Top-k next-step tokens with calibrated probabilities (softmax over the
    last-token logits, specials masked, then re-normalised over the top-k)."""
    model, vocab, device = require_model()
    with _LOCK, torch.no_grad():
        ids = vocab.encode(partial, add_bos=True, add_eos=False)
        ids = ids[-model.cfg.block_size:]
        x = torch.tensor([ids], dtype=torch.long, device=device)
        logits, _ = model(x)
        last = logits[0, -1].clone()
        for sid in (vocab.pad_id, vocab.bos_id, vocab.eos_id, vocab.unk_id):
            last[sid] = float("-inf")
        probs = torch.softmax(last, dim=-1)
        topv, topi = torch.topk(probs, k=min(k, probs.numel()))
        return [{"token": vocab.itos[int(i)], "prob": float(v)}
                for v, i in zip(topv.tolist(), topi.tolist())]


def complete(partial: list[str], max_new: int = 220,
             greedy: bool = True, temperature: float = 1.0) -> list[str]:
    model, vocab, device = require_model()
    with _LOCK:
        return complete_sequence(model, vocab, partial,
                                 max_new=max_new, device=device,
                                 greedy=greedy, temperature=temperature)


def sample_random(prefix: list[str] | None = None,
                  max_new: int = 220,
                  temperature: float = 1.0) -> list[str]:
    """Unconditional sampling from BOS (+ optional prefix). V1 model has no
    family conditioning, so this generates a plausible recipe from scratch."""
    return complete(prefix or [], max_new=max_new,
                    greedy=False, temperature=max(0.05, temperature))


def validate(sequence: list[str]) -> list[dict]:
    """Deterministic rule check — no model required."""
    viols = G.validate_sequence(sequence)
    return [{"rule": v.rule,
             "description": v.description,
             "step_index": v.step_index,
             "step_name": v.step_name} for v in viols]


def anomaly(sequence: list[str], use_validator: bool = True) -> dict:
    model, vocab, device = require_model()
    with _LOCK:
        nll = sequence_nll(model, vocab, sequence, device=device)
        thr = STATE.threshold
        if thr is not None:
            score = 1.0 / (1.0 + math.exp((nll - thr) * 4.0))
        else:
            score = math.exp(-nll)
        lm_is_valid = 1 if (thr is None or nll <= thr) else 0
        result = {
            "nll": nll,
            "threshold": thr,
            "lm_only": {"is_valid": lm_is_valid, "score": score},
        }
        if use_validator:
            viols = G.validate_sequence(sequence)
            if viols:
                result["is_valid"] = 0
                result["score"] = min(score, 0.05)
                result["predicted_rule"] = viols[0].rule
                result["violations"] = [{"rule": v.rule,
                                         "description": v.description,
                                         "step_index": v.step_index,
                                         "step_name": v.step_name} for v in viols]
            else:
                result["is_valid"] = 1
                result["score"] = max(score, 0.95)
                result["predicted_rule"] = ""
                result["violations"] = []
        else:
            result["is_valid"] = lm_is_valid
            result["score"] = score
            result["predicted_rule"] = ""
            result["violations"] = []
        return result


def vocab_tokens() -> list[str]:
    _, vocab, _ = require_model()
    return list(vocab.itos)
