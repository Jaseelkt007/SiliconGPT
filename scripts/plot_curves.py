"""Plot training curves from extras/results/train_log.csv -> extras/results/curves.png."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="extras/results/train_log.csv")
    ap.add_argument("--out", default="extras/results/curves.png")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.log)))
    it = [int(r["iter"]) for r in rows]
    tr = [float(r["train_loss"]) for r in rows]
    vl = [float(r["val_loss"]) for r in rows]
    t1 = [float(r["top1"]) for r in rows]
    t5 = [float(r["top5"]) for r in rows]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(it, tr, label="train"); ax[0].plot(it, vl, label="val")
    ax[0].set_title("loss"); ax[0].set_xlabel("iter"); ax[0].legend()
    ax[1].plot(it, t1, label="top-1"); ax[1].plot(it, t5, label="top-5")
    ax[1].set_title("next-step accuracy"); ax[1].set_xlabel("iter"); ax[1].legend()
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
