"""Vocabulary / tokenizer for process-step sequences.

Each manufacturing step is ONE token. The vocabulary is the union of all step
strings found in the data, plus four special tokens. Pure stdlib (no torch).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

PAD, BOS, EOS, UNK = "<PAD>", "<BOS>", "<EOS>", "<UNK>"
SPECIALS = [PAD, BOS, EOS, UNK]


class Vocab:
    """Bidirectional step<->id map with special tokens at fixed ids 0..3."""

    def __init__(self, tokens):
        assert list(tokens[:4]) == SPECIALS, "tokens must start with SPECIALS"
        self.itos = list(tokens)
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.pad_id = self.stoi[PAD]
        self.bos_id = self.stoi[BOS]
        self.eos_id = self.stoi[EOS]
        self.unk_id = self.stoi[UNK]

    def __len__(self):
        return len(self.itos)

    @property
    def size(self):
        return len(self.itos)

    def encode(self, steps, add_bos=True, add_eos=True):
        ids = [self.bos_id] if add_bos else []
        ids.extend(self.stoi.get(s, self.unk_id) for s in steps)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids, strip_specials=True):
        out = []
        for i in ids:
            tok = self.itos[i] if 0 <= i < len(self.itos) else UNK
            if strip_specials and tok in SPECIALS:
                continue
            out.append(tok)
        return out

    def save(self, path):
        Path(path).write_text(json.dumps({"tokens": self.itos}), encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["tokens"])

    @classmethod
    def build_from_csvs(cls, csv_paths, seq_col="SEQUENCE"):
        """Build a vocab by scanning the SEQUENCE column of compact CSVs."""
        steps = set()
        for p in csv_paths:
            with open(p, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    seq = row.get(seq_col, "")
                    if seq:
                        steps.update(seq.split("|"))
        return cls(SPECIALS + sorted(steps))


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    data = root / "data"
    v = Vocab.build_from_csvs([
        data / "train_pool.csv", data / "val_id.csv", data / "ood_holdout.csv"
    ])
    out = root / "vocab.json"
    v.save(out)
    print(f"vocab size = {len(v)} (4 specials + {len(v) - 4} steps) -> {out}")
    print("first 8:", v.itos[:8])
