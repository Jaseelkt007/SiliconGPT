"""Build emb_init.npz from step names (+ optional descriptions) with a frozen encoder.

RUN ON A LOGIN NODE (needs internet to download the encoder once).
  pixi run python scripts/build_emb_init.py --vocab vocab.json --out emb_init.npz
  # richer: enrich with the kit's step descriptions
  pixi run python scripts/build_emb_init.py --descriptions \
      /path/to/training_data/MOSFET_Longdescr.csv /path/to/IGBT_Longdescr.csv /path/to/IC_Longdescr.csv

Training reads the resulting npz (no encoder/internet needed on compute nodes):
  sbatch scripts/run_train.sh --emb-init emb_init.npz
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from process_logic.vocab import Vocab, SPECIALS  # noqa: E402


def read_descriptions(paths):
    desc = {}
    for p in paths:
        with open(p, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                step = (row.get("STEP") or "").strip().strip('"')
                d = (row.get("DESCRIPTION") or "").strip().strip('"')
                if step and d:
                    desc.setdefault(step, d)
    return desc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", default=str(ROOT / "vocab.json"))
    ap.add_argument("--out", default=str(ROOT / "emb_init.npz"))
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--descriptions", nargs="*", default=[],
                    help="optional STEP,DESCRIPTION CSVs to enrich the text")
    args = ap.parse_args()

    vocab = Vocab.load(args.vocab)
    steps = [t for t in vocab.itos if t not in SPECIALS]
    desc = read_descriptions(args.descriptions) if args.descriptions else {}
    # use the step NAME (robust to unseen families); enrich with description when available
    texts = [f"{t}. {desc[t]}" if t in desc else t for t in steps]

    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(args.model)
    vecs = np.asarray(enc.encode(texts, show_progress_bar=True), dtype=np.float32)

    np.savez(args.out, tokens=np.array(steps, dtype=object), vectors=vecs)
    print(f"saved {len(steps)} token embeddings ({vecs.shape[1]}-dim, "
          f"{len(desc)} enriched with descriptions) -> {args.out}")


if __name__ == "__main__":
    main()
