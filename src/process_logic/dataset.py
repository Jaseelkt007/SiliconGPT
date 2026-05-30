"""Dataset + batching for next-token training on process sequences.

Core logic (CSV loading + numpy collation + eval-batch construction) is torch-free
and unit-testable. The torch DataLoader wrapper imports torch lazily, so this module
imports fine on machines without torch (we train on the GPU server).
"""
from __future__ import annotations

import csv
import random as _random
from pathlib import Path

import numpy as np

IGNORE_INDEX = -100


def load_compact_csv(path, seq_col="SEQUENCE"):
    """Read a compact CSV (SEQUENCE_ID, FAMILY, SEQUENCE) -> list[(family, [steps])]."""
    out = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fam = row.get("FAMILY", "")
            seq = row[seq_col].split("|") if row.get(seq_col) else []
            out.append((fam, seq))
    return out


def collate_ids(batch_ids, pad_id, ignore_index=IGNORE_INDEX):
    """Pad a batch of token-id lists (right padding).

    Labels = input_ids with PAD positions set to ignore_index. The one-token
    shift for next-token loss is done inside the model (HF GPT-2 convention).

    Returns numpy arrays: input_ids, attention_mask, labels  (all [B, T] int64).
    """
    B = len(batch_ids)
    T = max((len(x) for x in batch_ids), default=1)
    input_ids = np.full((B, T), pad_id, dtype=np.int64)
    attention_mask = np.zeros((B, T), dtype=np.int64)
    labels = np.full((B, T), ignore_index, dtype=np.int64)
    for i, ids in enumerate(batch_ids):
        L = len(ids)
        input_ids[i, :L] = ids
        attention_mask[i, :L] = 1
        labels[i, :L] = ids
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _round_robin_by_family(examples, seed):
    """Interleave examples across families so any prefix is family-balanced."""
    rng = _random.Random(seed)
    by_fam = {}
    for e in examples:
        by_fam.setdefault(e[0], []).append(e)
    for v in by_fam.values():
        rng.shuffle(v)
    its = [iter(v) for v in by_fam.values()]
    merged, alive = [], True
    while alive:
        alive = False
        for it in its:
            nxt = next(it, None)
            if nxt is not None:
                merged.append(nxt)
                alive = True
    return merged


def cross_family_recomb(examples, n_aug, seed=42, max_attempts_factor=6, attempt_ceiling=120000):
    """Cross-family recombination augmentation (GECA-style; an OOD lever).

    Splice the prefix of a sequence from family A with the suffix of one from a DIFFERENT
    family B, then KEEP ONLY validator-valid results — so we never train on a rule-breaking
    sequence. A shared-step junction heuristic (cut B at a step equal to A's junction step)
    makes the local transition plausible and raises yield. Survivors are genuinely
    family-agnostic orderings — the structure the OOD gap lacks.

    Operates on whatever families are present in `examples` (so under --exclude-family it
    automatically only recombines the trained families — never introduces held-out tokens).
    Returns (list[(family, steps)], attempts). Family label = the prefix's family (only used
    for balanced batching; irrelevant to the LM loss when family-conditioning is off).
    """
    from process_logic.generation import validate_sequence
    rng = _random.Random(seed)
    by_fam = {}
    for fam, steps in examples:
        by_fam.setdefault(fam, []).append(steps)
    fams = sorted(by_fam)
    if len(fams) < 2 or n_aug <= 0:
        return [], 0
    out, seen, attempts = [], set(), 0
    # Yield collapses as `seen` fills (limited valid-recomb diversity from junction-matching),
    # so cap attempts tightly + an absolute ceiling — give up fast rather than grind to a huge
    # n_aug. Returning fewer-but-valid recombinations is fine; the caller caps n_aug too.
    max_attempts = min(attempt_ceiling, max(2000, n_aug * max_attempts_factor))
    while len(out) < n_aug and attempts < max_attempts:
        attempts += 1
        fa, fb = rng.sample(fams, 2)
        A, B = rng.choice(by_fam[fa]), rng.choice(by_fam[fb])
        if len(A) < 4 or len(B) < 4:
            continue
        cut_a = rng.randint(2, len(A) - 2)
        junction = A[cut_a]
        bmatches = [i for i, s in enumerate(B) if s == junction]
        cut_b = rng.choice(bmatches) if (bmatches and rng.random() < 0.8) else rng.randint(2, len(B) - 2)
        cand = A[:cut_a] + B[cut_b:]
        if not (4 <= len(cand) <= 250):
            continue
        key = tuple(cand)
        if key in seen:
            continue
        if len(validate_sequence(cand)) == 0:   # empty violation list == valid
            seen.add(key)
            out.append((fa, cand))
    return out, attempts


def build_eval_batches(examples, vocab, batch_size, n_batches=None, seed=0,
                       balanced=True, add_bos=True, add_eos=True):
    """Build a FIXED list of collated numpy batches for stable, unbiased validation.

    Without this, evaluating a family-blocked val set with a cycling loader produces a
    family-biased (noisy) metric. We interleave families and freeze the batch list so
    every eval scores the exact same, balanced sample.
    """
    ex = _round_robin_by_family(examples, seed) if balanced else list(examples)
    if not balanced:
        _random.Random(seed).shuffle(ex)
    enc = [vocab.encode(s, add_bos=add_bos, add_eos=add_eos) for _, s in ex]
    batches = []
    for i in range(0, len(enc), batch_size):
        batches.append(collate_ids(enc[i:i + batch_size], vocab.pad_id))
        if n_batches and len(batches) >= n_batches:
            break
    return batches


def make_dataloader(examples, vocab, batch_size=64, shuffle=True,
                    num_workers=0, add_bos=True, add_eos=True):
    """Build a torch DataLoader. Imports torch lazily (only needed at train time)."""
    import torch
    from torch.utils.data import Dataset, DataLoader

    pad_id = vocab.pad_id

    class _DS(Dataset):
        def __len__(self):
            return len(examples)

        def __getitem__(self, i):
            _, steps = examples[i]
            return vocab.encode(steps, add_bos=add_bos, add_eos=add_eos)

    def _collate(batch):
        arr = collate_ids(batch, pad_id)
        return {k: torch.from_numpy(v) for k, v in arr.items()}

    return DataLoader(_DS(), batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, collate_fn=_collate)
