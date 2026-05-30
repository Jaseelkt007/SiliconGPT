"""Unit tests for validity.py (pure stdlib; runs locally)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from process_logic.validity import sequence_validity, batch_validity  # noqa: E402
from process_logic.dataset import load_compact_csv  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"


def _a_valid_sequence():
    return load_compact_csv(DATA / "train_pool.csv")[0][1]  # (family, steps) -> steps


def test_valid_sequence_passes():
    ok, rules = sequence_validity(_a_valid_sequence())
    assert ok and rules == []


def test_corrupted_sequence_flagged():
    steps = list(_a_valid_sequence())
    # move SHIP LOT before WAFER SORT TEST -> RULE_SHIP_BEFORE_TEST
    steps = [s for s in steps if s != "SHIP LOT"]
    steps.insert(steps.index("WAFER SORT TEST"), "SHIP LOT")
    ok, rules = sequence_validity(steps)
    assert not ok and "RULE_SHIP_BEFORE_TEST" in rules


def test_batch_validity_fraction():
    good = _a_valid_sequence()
    bad = [s for s in good if s != "SHIP LOT"]
    bad.insert(bad.index("WAFER SORT TEST"), "SHIP LOT")
    m = batch_validity([good, bad, good])
    assert m["n"] == 3
    assert abs(m["valid_frac"] - 2 / 3) < 1e-9
    assert m["per_rule"].get("RULE_SHIP_BEFORE_TEST") == 1


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
    print("ALL VALIDITY TESTS PASSED")
