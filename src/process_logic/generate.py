"""Inference helpers: load a checkpoint, rank next steps, complete a sequence."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from process_logic.vocab import Vocab, SPECIALS          # noqa: E402
from process_logic.model import ProcessLM, ModelConfig   # noqa: E402


def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    vocab = Vocab(ck["vocab"])
    model = ProcessLM(ModelConfig(**ck["mcfg"])).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, vocab


def _special_ids(vocab):
    return [vocab.stoi[t] for t in SPECIALS]


@torch.no_grad()
def rank_next_steps(model, vocab, partial_steps, k=5, device="cpu"):
    """Return the top-k most likely next *step* strings (special tokens masked out)."""
    ids = vocab.encode(partial_steps, add_bos=True, add_eos=False)
    ids = ids[-model.cfg.block_size:]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    logits, _ = model(x)
    last = logits[0, -1].clone()
    for sid in _special_ids(vocab):
        last[sid] = float("-inf")
    topk = torch.topk(last, min(k, last.numel())).indices.tolist()
    return [vocab.itos[i] for i in topk]


@torch.no_grad()
def complete_sequence(model, vocab, partial_steps, max_new=220, device="cpu",
                      greedy=True, temperature=1.0):
    """Autoregressively predict the steps AFTER the prefix, until <EOS> or max_new."""
    ids = vocab.encode(partial_steps, add_bos=True, add_eos=False)
    block = model.cfg.block_size
    out = []
    for _ in range(max_new):
        x = torch.tensor([ids[-block:]], dtype=torch.long, device=device)
        logits, _ = model(x)
        last = logits[0, -1].clone()
        for t in (vocab.pad_id, vocab.bos_id, vocab.unk_id):
            last[t] = float("-inf")
        if greedy:
            nxt = int(torch.argmax(last))
        else:
            probs = torch.softmax(last / max(1e-6, temperature), dim=-1)
            nxt = int(torch.multinomial(probs, 1))
        if nxt == vocab.eos_id:
            break
        out.append(vocab.itos[nxt])
        ids.append(nxt)
    return out
