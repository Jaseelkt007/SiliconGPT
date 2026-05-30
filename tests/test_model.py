"""Model tests (run: python3 tests/test_model.py).

torch-gated: skips cleanly if torch is not installed (i.e. on a torchless laptop).
Run these on the server where torch is available.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import torch  # noqa: F401
except ImportError:
    print("torch not installed -> skipping model tests (run on the server)")
    sys.exit(0)

from process_logic.model import ProcessLM, ModelConfig  # noqa: E402


def _tiny():
    return ProcessLM(ModelConfig(vocab_size=50, n_layer=2, n_head=2,
                                 n_embd=32, block_size=64, dropout=0.0))


def test_forward_shapes_and_loss():
    m = _tiny()
    ids = torch.randint(0, 50, (4, 20))
    labels = ids.clone()
    logits, loss = m(ids, labels=labels)
    assert logits.shape == (4, 20, 50)
    assert loss is not None and torch.isfinite(loss)
    assert m.num_params() > 0


def test_weight_tying():
    m = _tiny()
    assert m.head.weight.data_ptr() == m.tok.weight.data_ptr()


def test_overfits_one_batch():
    torch.manual_seed(0)
    m = _tiny()
    ids = torch.randint(0, 50, (8, 24))
    labels = ids.clone()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    first = None
    for _ in range(60):
        _, loss = m(ids, labels=labels)
        if first is None:
            first = loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < first, f"did not learn ({first:.3f} -> {loss.item():.3f})"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
    print("ALL MODEL TESTS PASSED")
