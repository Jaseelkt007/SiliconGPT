"""Measure process-validity of a trained model's generations (run on the server, with torch).

Three regimes (this is RL's headroom — greedy is usually ~1.0; sampled/free reveal the gap):
  1. greedy completion   — complete eval prompts greedily, validate(prefix+completion)
  2. sampled completion   — same, with temperature sampling
  3. free generation      — generate full recipes from <BOS>, validate

  pixi run python scripts/measure_validity.py --ckpt checkpoints/best.pt --n 300 --temp 1.0
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch  # noqa: E402,F401
from process_logic.generate import load_checkpoint, complete_sequence  # noqa: E402
from process_logic.validity import batch_validity  # noqa: E402


def _prompts(n):
    rows = list(csv.DictReader(open(ROOT / "data" / "eval_completion.csv", encoding="utf-8")))
    return [r["PARTIAL_SEQUENCE"].split("|") for r in rows][:n]


def _report(title, seqs):
    m = batch_validity(seqs)
    print(f"\n[{title}] n={m['n']}  valid_frac={m['valid_frac']:.3f}")
    if m["per_rule"]:
        top = sorted(m["per_rule"].items(), key=lambda x: -x[1])[:5]
        print("   most-violated rules:", top)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints/best.pt"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--temp", type=float, default=1.0)
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab = load_checkpoint(args.ckpt, device)
    prompts = _prompts(args.n)

    greedy = [p + complete_sequence(model, vocab, p, device=device, greedy=True) for p in prompts]
    _report("greedy completion", greedy)

    sampled = [p + complete_sequence(model, vocab, p, device=device, greedy=False, temperature=args.temp)
               for p in prompts]
    _report(f"sampled completion (temp={args.temp})", sampled)

    free = [complete_sequence(model, vocab, [], device=device, greedy=False,
                              temperature=args.temp, max_new=250) for _ in range(args.n)]
    _report(f"free generation (temp={args.temp})", free)


if __name__ == "__main__":
    main()
