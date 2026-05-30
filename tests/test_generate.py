"""Inference smoke tests (torch-gated; run on the server)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import torch  # noqa: F401
except ImportError:
    print("torch not installed -> skipping generate tests (run on the server)")
    sys.exit(0)

from process_logic.vocab import Vocab, SPECIALS          # noqa: E402
from process_logic.model import ProcessLM, ModelConfig   # noqa: E402
from process_logic.generate import rank_next_steps, complete_sequence  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"


def _model_vocab():
    vocab = Vocab.build_from_csvs([DATA / "train_pool.csv"])
    model = ProcessLM(ModelConfig(vocab_size=len(vocab), n_layer=2, n_head=2,
                                  n_embd=64, block_size=256, dropout=0.0))
    model.eval()
    return model, vocab


def test_rank_next_steps():
    model, vocab = _model_vocab()
    ranks = rank_next_steps(model, vocab, ["RECEIVE WAFER LOT", "LOT IDENTIFICATION"], k=5)
    assert len(ranks) == 5
    assert all(r in vocab.stoi for r in ranks)
    assert all(r not in SPECIALS for r in ranks)


def test_complete_sequence():
    model, vocab = _model_vocab()
    comp = complete_sequence(model, vocab, ["RECEIVE WAFER LOT"], max_new=10)
    assert isinstance(comp, list) and len(comp) <= 10
    assert all(s not in SPECIALS for s in comp)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
    print("ALL GENERATE TESTS PASSED")
