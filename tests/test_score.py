"""Validate the local scorer with synthetic PERFECT predictions -> perfect metrics."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from process_logic.score import (  # noqa: E402
    read_rows, score_nextstep, score_completion, score_anomaly, edit_distance, roc_auc,
)

DATA = Path(__file__).resolve().parents[1] / "data"


def test_edit_distance_and_auc():
    assert edit_distance(["a", "b", "c"], ["a", "b", "c"]) == 0
    assert edit_distance(["a", "b"], ["a", "x"]) == 1
    assert abs(roc_auc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) - 1.0) < 1e-9


def test_perfect_nextstep():
    gt = read_rows(DATA / "eval_nextstep.csv")
    preds = [{"EXAMPLE_ID": g["EXAMPLE_ID"], "RANK_1": g["TRUE_NEXT_STEP"],
              "RANK_2": "", "RANK_3": "", "RANK_4": "", "RANK_5": ""} for g in gt]
    m = score_nextstep(preds, gt)["ALL"]
    assert m["top1"] == 1.0 and m["top5"] == 1.0 and abs(m["mrr"] - 1.0) < 1e-9


def test_perfect_completion():
    gt = read_rows(DATA / "eval_completion.csv")
    preds = [{"EXAMPLE_ID": g["EXAMPLE_ID"], "PREDICTED_SEQUENCE": g["TRUE_SUFFIX"]} for g in gt]
    m = score_completion(preds, gt)["ALL"]
    assert m["exact_match"] == 1.0 and m["norm_edit_dist"] == 0.0 and m["token_acc"] == 1.0


def test_perfect_anomaly():
    gt = read_rows(DATA / "eval_anomaly.csv")
    preds = [{"EXAMPLE_ID": g["EXAMPLE_ID"], "IS_VALID": g["IS_VALID"],
              "SCORE": "1.0" if g["IS_VALID"] == "1" else "0.0",
              "PREDICTED_RULE": g["RULE_VIOLATED"]} for g in gt]
    m = score_anomaly(preds, gt)["ALL"]
    assert m["f1"] == 1.0 and m["acc"] == 1.0 and abs(m["roc_auc"] - 1.0) < 1e-9
    assert m["rule_attr"] == 1.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
    print("ALL SCORE TESTS PASSED")
