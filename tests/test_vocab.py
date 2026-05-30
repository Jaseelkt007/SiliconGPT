"""Unit tests for vocab.py (run: python3 tests/test_vocab.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from process_logic.vocab import Vocab, SPECIALS  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"


def _build():
    # train_pool contains all 3 families, so it covers the full step vocabulary
    return Vocab.build_from_csvs([DATA / "train_pool.csv"])


def test_size_and_specials():
    v = _build()
    assert v.itos[:4] == SPECIALS
    assert (v.pad_id, v.bos_id, v.eos_id, v.unk_id) == (0, 1, 2, 3)
    assert 190 <= len(v) <= 230, f"unexpected vocab size {len(v)}"


def test_roundtrip():
    v = _build()
    steps = ["RECEIVE WAFER LOT", "LOT IDENTIFICATION", "SHIP LOT"]
    ids = v.encode(steps)
    assert ids[0] == v.bos_id and ids[-1] == v.eos_id
    assert len(ids) == len(steps) + 2
    assert v.decode(ids) == steps


def test_unk():
    v = _build()
    ids = v.encode(["A STEP THAT DOES NOT EXIST"], add_bos=False, add_eos=False)
    assert ids == [v.unk_id]


def test_save_load():
    v = _build()
    p = DATA / "_vocab_test.json"
    v.save(p)
    v2 = Vocab.load(p)
    assert v2.itos == v.itos
    assert v2.stoi == v.stoi
    p.unlink()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
    print("ALL VOCAB TESTS PASSED")
