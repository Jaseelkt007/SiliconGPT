"""Unit tests for dataset.py (run: python3 tests/test_dataset.py)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from process_logic.vocab import Vocab  # noqa: E402
from process_logic.dataset import (  # noqa: E402
    load_compact_csv, collate_ids, build_eval_batches, _round_robin_by_family, IGNORE_INDEX,
)

DATA = Path(__file__).resolve().parents[1] / "data"


def test_load_compact():
    ex = load_compact_csv(DATA / "train_pool.csv")
    assert len(ex) == 60000, len(ex)
    fam, steps = ex[0]
    assert fam in {"mosfet", "igbt", "ic"}
    assert steps[0] == "RECEIVE WAFER LOT" and steps[-1] == "SHIP LOT"
    assert 100 < len(steps) < 160


def test_collate_shapes_and_padding():
    v = Vocab.build_from_csvs([DATA / "train_pool.csv"])
    seqs = [
        v.encode(["RECEIVE WAFER LOT", "SHIP LOT"]),               # len 4
        v.encode(["RECEIVE WAFER LOT", "LOT IDENTIFICATION", "SHIP LOT"]),  # len 5
    ]
    T = max(len(s) for s in seqs)
    b = collate_ids(seqs, v.pad_id)
    assert b["input_ids"].shape == (2, T)
    L0 = len(seqs[0])
    # shorter sequence padded at the end
    assert (b["input_ids"][0, L0:] == v.pad_id).all()
    assert (b["attention_mask"][0, :L0] == 1).all()
    assert (b["attention_mask"][0, L0:] == 0).all()
    # labels mirror input on real tokens, ignore_index on pad
    assert (b["labels"][0, :L0] == np.array(seqs[0])).all()
    assert (b["labels"][0, L0:] == IGNORE_INDEX).all()
    # full-length row has no padding
    assert (b["attention_mask"][1] == 1).all()


def test_build_eval_batches_stable_and_balanced():
    v = Vocab.build_from_csvs([DATA / "train_pool.csv"])
    val = load_compact_csv(DATA / "val_id.csv")
    b1 = build_eval_batches(val, v, batch_size=64, n_batches=10, seed=42)
    b2 = build_eval_batches(val, v, batch_size=64, n_batches=10, seed=42)
    assert len(b1) == 10
    assert b1[0]["input_ids"].shape[0] == 64
    # deterministic for a fixed seed (so every eval scores the same sample)
    assert (b1[0]["input_ids"] == b2[0]["input_ids"]).all()
    # family-balanced: round-robin interleaving spans all families in the first few items
    # (this is the fix for the family-blocked, biased validation metric).
    merged = _round_robin_by_family(val, seed=42)
    n_fams = len(set(f for f, _ in val))
    assert len(set(f for f, _ in merged[:n_fams])) == n_fams


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
    print("ALL DATASET TESTS PASSED")
