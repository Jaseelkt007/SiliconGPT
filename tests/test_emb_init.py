"""Unit tests for emb_init.build_embedding_init (pure numpy; runs locally)."""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from process_logic.vocab import Vocab, SPECIALS  # noqa: E402
from process_logic.emb_init import build_embedding_init  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"


def _fake_npz(tokens, enc_dim=16, seed=1):
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((len(tokens), enc_dim)).astype("float32")
    p = Path(tempfile.mkdtemp()) / "emb.npz"
    np.savez(p, tokens=np.array(tokens, dtype=object), vectors=vecs)
    return p


def test_shape_scale_and_alignment():
    vocab = Vocab.build_from_csvs([DATA / "train_pool.csv"])
    steps = [t for t in vocab.itos if t not in SPECIALS]
    npz = _fake_npz(steps, enc_dim=16)
    n_embd = 128
    m, n_sem = build_embedding_init(vocab, npz, n_embd, seed=0)
    assert m.shape == (len(vocab), n_embd)
    assert m.dtype == np.float32
    assert n_sem == len(steps)                       # every step had a vector
    assert 0.01 < m.std() < 0.04                     # scaled near 0.02


def test_deterministic_and_semantic_rows_differ_from_specials():
    vocab = Vocab.build_from_csvs([DATA / "train_pool.csv"])
    steps = [t for t in vocab.itos if t not in SPECIALS]
    npz = _fake_npz(steps, enc_dim=16)
    m1, _ = build_embedding_init(vocab, npz, 64, seed=0)
    m2, _ = build_embedding_init(vocab, npz, 64, seed=0)
    assert np.allclose(m1, m2)                        # deterministic for a fixed seed
    # specials (rows 0..3) were random; a step row (>=4) should generally differ from them
    assert not np.allclose(m1[0], m1[4])


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
    print("ALL EMB_INIT TESTS PASSED")
