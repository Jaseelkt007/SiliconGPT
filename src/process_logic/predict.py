"""Produce the three submission files from eval inputs + a trained checkpoint.

Organizers' files (at event start):
  eval_input_valid.csv   -> pass to BOTH --nextstep-input and --completion-input
  eval_input_anomaly.csv -> pass to --anomaly-input
Our local eval_* files (same input columns) also work for self-evaluation.

Example:
  python src/process_logic/predict.py --ckpt checkpoints/best.pt \
      --nextstep-input data/eval_nextstep.csv \
      --completion-input data/eval_completion.csv \
      --anomaly-input data/eval_anomaly.csv \
      --calib-file data/val_id.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from process_logic.generate import load_checkpoint, rank_next_steps, complete_sequence  # noqa: E402
from process_logic.anomaly import score_anomaly, calibrate_threshold                    # noqa: E402
from process_logic.dataset import load_compact_csv                                      # noqa: E402


def _rows(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def task_nextstep(model, vocab, in_path, out_path, device):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["EXAMPLE_ID", "RANK_1", "RANK_2", "RANK_3", "RANK_4", "RANK_5"])
        for r in _rows(in_path):
            ranks = rank_next_steps(model, vocab, r["PARTIAL_SEQUENCE"].split("|"), k=5, device=device)
            ranks = (ranks + [""] * 5)[:5]
            w.writerow([r["EXAMPLE_ID"], *ranks])


def task_completion(model, vocab, in_path, out_path, device):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["EXAMPLE_ID", "PREDICTED_SEQUENCE"])
        for r in _rows(in_path):
            comp = complete_sequence(model, vocab, r["PARTIAL_SEQUENCE"].split("|"), device=device)
            w.writerow([r["EXAMPLE_ID"], "|".join(comp)])


def task_anomaly(model, vocab, in_path, out_path, device, threshold, use_validator):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["EXAMPLE_ID", "IS_VALID", "SCORE", "PREDICTED_RULE"])
        for r in _rows(in_path):
            isv, score, rule = score_anomaly(model, vocab, r["SEQUENCE"].split("|"),
                                             device=device, threshold=threshold,
                                             use_validator=use_validator)
            w.writerow([r["EXAMPLE_ID"], isv, f"{score:.4f}", rule])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints/best.pt"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-dir", default=str(ROOT / "extras/results"))
    ap.add_argument("--nextstep-input")
    ap.add_argument("--completion-input")
    ap.add_argument("--anomaly-input")
    ap.add_argument("--calib-file", help="compact CSV of valid sequences to calibrate the anomaly threshold")
    ap.add_argument("--no-validator", action="store_true", help="LM-only anomaly (no rule checker)")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab = load_checkpoint(args.ckpt, device)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.nextstep_input:
        task_nextstep(model, vocab, args.nextstep_input, out / "nextstep.csv", device)
        print("wrote", out / "nextstep.csv")
    if args.completion_input:
        task_completion(model, vocab, args.completion_input, out / "completion.csv", device)
        print("wrote", out / "completion.csv")
    if args.anomaly_input:
        threshold = None
        if args.calib_file:
            valids = [s for _, s in load_compact_csv(args.calib_file)][:1000]
            threshold = calibrate_threshold(model, vocab, valids, device)
        task_anomaly(model, vocab, args.anomaly_input, out / "anomaly.csv",
                     device, threshold, not args.no_validator)
        print("wrote", out / "anomaly.csv")


if __name__ == "__main__":
    main()
