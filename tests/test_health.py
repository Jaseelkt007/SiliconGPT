"""Full transformer + data health check (run on the server, with torch).

Verifies EVERY part of the stack and prints a health report:
  vocab · dataloader shapes/padding · ground-truth alignment · forward/loss ·
  CAUSALITY (no future leakage) · RoPE positional info · RMSNorm · weight-tying ·
  per-parameter gradient flow · weight-update · NaN/Inf · (optional) trained-checkpoint
  synonym-embedding cosine.

  python tests/test_health.py                       # fresh-model architecture checks
  python tests/test_health.py --ckpt checkpoints/best.pt   # also inspect the trained model
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    print("torch not installed -> run this on the server.")
    sys.exit(0)

from process_logic.vocab import Vocab, SPECIALS                       # noqa: E402
from process_logic.model import ProcessLM, ModelConfig                # noqa: E402
from process_logic.dataset import load_compact_csv, collate_ids       # noqa: E402

DATA = ROOT / "data"
_results = []


def check(name, ok, detail=""):
    _results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))


def section(t):
    print(f"\n=== {t} ===")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()
    torch.manual_seed(0)

    vocab = (Vocab.load(ROOT / "vocab.json") if (ROOT / "vocab.json").exists()
             else Vocab.build_from_csvs([DATA / "train_pool.csv"]))
    V = len(vocab)
    cfg = ModelConfig(vocab_size=V, n_layer=2, n_head=4, n_embd=128, block_size=256, dropout=0.0)
    model = ProcessLM(cfg).eval()

    # ---------------------------------------------------------------- VOCAB
    section("Vocabulary")
    check("specials at ids 0-3", vocab.itos[:4] == SPECIALS)
    check("size sane (~202)", 190 <= V <= 230, f"V={V}")
    rt = ["RECEIVE WAFER LOT", "SHIP LOT"]
    check("encode/decode round-trip", vocab.decode(vocab.encode(rt)) == rt)

    # ---------------------------------------------------- DATALOADER + GT ALIGN
    section("Dataloader: shapes / padding / ground-truth alignment")
    ex = load_compact_csv(DATA / "train_pool.csv")[:6]
    enc = [vocab.encode(s) for _, s in ex]
    b = collate_ids(enc, vocab.pad_id)
    Tb = max(len(e) for e in enc)
    check("input_ids shape [B,T]", b["input_ids"].shape == (6, Tb), str(b["input_ids"].shape))
    check("int64 dtype", b["input_ids"].dtype.name == "int64")
    # attention_mask marks exactly the non-pad positions
    mask_ok = ((b["attention_mask"] == 1) == (b["input_ids"] != vocab.pad_id)).all()
    check("attention_mask == non-pad positions", bool(mask_ok))
    # labels mirror inputs on real tokens, -100 on pad
    lab_ok = True
    for i, e in enumerate(enc):
        L = len(e)
        lab_ok &= (b["labels"][i, :L] == b["input_ids"][i, :L]).all() and (b["labels"][i, L:] == -100).all()
    check("labels == input (real) / -100 (pad)", bool(lab_ok))
    # next-token alignment: the model trains pos t -> token t+1, i.e. labels[1:] == input[1:]
    row0 = enc[0]
    check("seq starts <BOS> ends <EOS>", row0[0] == vocab.bos_id and row0[-1] == vocab.eos_id)
    check("next-token target = shifted input", b["labels"][0, 1:len(row0)].tolist() == row0[1:])

    # --------------------------------------------------------- FORWARD + LOSS
    section("Forward pass + loss")
    ids = torch.from_numpy(b["input_ids"])
    labels = torch.from_numpy(b["labels"])
    logits, loss = model(ids, labels=labels)
    check("logits shape [B,T,V]", tuple(logits.shape) == (6, Tb, V))
    check("logits finite", bool(torch.isfinite(logits).all()))
    import math
    check("fresh-model loss ~ ln(V)", abs(loss.item() - math.log(V)) < 0.6,
          f"loss={loss.item():.3f} ln(V)={math.log(V):.3f}")

    # ---------------------------------------------------- CAUSALITY (critical)
    section("Causal attention — NO future leakage (the key correctness test)")
    x = torch.randint(0, V, (1, 16))
    l1, _ = model(x)
    p = 8
    x2 = x.clone(); x2[0, p] = (int(x[0, p]) + 7) % V       # change a FUTURE token
    l2, _ = model(x2)
    past_unchanged = torch.allclose(l1[0, :p], l2[0, :p], atol=1e-4)
    future_changed = not torch.allclose(l1[0, p], l2[0, p], atol=1e-4)
    check("logits at positions < p UNCHANGED when token p changes (causal)", bool(past_unchanged))
    check("logits at position p DO change (sanity)", bool(future_changed))
    if not past_unchanged:
        print("    >>> WARNING: future tokens affect past outputs — attention is NOT causal!")

    # -------------------------------------------------------- RoPE / position
    section("Positional encoding (RoPE)")
    check("rope cache shape [block, head_dim]",
          tuple(model.rope_cos.shape) == (cfg.block_size, cfg.n_embd // cfg.n_head))
    # RoPE is applied to q,k only (values carry no position), so identical tokens at every
    # position give identical outputs WITH OR WITHOUT RoPE — that is not a valid probe.
    # Valid probes: (a) the cache varies with position, and (b) reordering the context while
    # keeping the SAME final query token changes the final logits (relative-position sensitivity;
    # plain causal attention without positional encoding is order-invariant over the context).
    check("rope cache varies with position",
          not torch.allclose(model.rope_cos[0], model.rope_cos[5], atol=1e-6))
    c1 = torch.tensor([[6, 7, 8, 5]]); c2 = torch.tensor([[8, 7, 6, 5]])  # same final token (5), context reordered
    o1, _ = model(c1); o2, _ = model(c2)
    check("reordering context changes final logits (RoPE relative-position active)",
          not torch.allclose(o1[0, -1], o2[0, -1], atol=1e-5))

    # -------------------------------------------------------------- RMSNorm
    section("RMSNorm")
    r = torch.randn(4, 16, cfg.n_embd) * 5.0
    out = model.norm(r)
    rms = out.pow(2).mean(-1).sqrt().mean().item()          # weight init = 1
    check("output RMS ~= 1", abs(rms - 1.0) < 0.15, f"rms={rms:.3f}")

    # ------------------------------------------------------------ weight tying
    section("Weight tying (head == embedding)")
    check("head.weight shares storage with tok.weight",
          model.head.weight.data_ptr() == model.tok.weight.data_ptr())

    # --------------------------------------------------------- parameter health
    section("Parameter health (NaN/Inf, zeros)")
    bad = [n for n, q in model.named_parameters() if not torch.isfinite(q).all()]
    check("no NaN/Inf in any parameter", not bad, str(bad))
    zeros = [n for n, q in model.named_parameters() if float(q.abs().sum()) == 0.0]
    check("no all-zero parameter tensors", not zeros, str(zeros))

    # --------------------------------------------------- GRADIENT FLOW (all weights)
    section("Gradient flow — EVERY weight must receive a gradient")
    model.train()
    logits, loss = model(ids, labels=labels)
    model.zero_grad(set_to_none=True)
    loss.backward()
    missing = [n for n, q in model.named_parameters() if q.grad is None]
    zer_grad = [n for n, q in model.named_parameters() if q.grad is not None and float(q.grad.norm()) == 0.0]
    check("all parameters have a gradient", not missing, str(missing))
    check("no parameter has exactly-zero gradient", not zer_grad, str(zer_grad))
    gtot = sum(float(q.grad.norm()) ** 2 for _, q in model.named_parameters() if q.grad is not None) ** 0.5
    print(f"    global grad norm = {gtot:.3f}; per-module grad norms:")
    for n, q in model.named_parameters():
        if q.grad is not None and ("blocks.0." in n or n in ("tok.weight", "norm.weight")):
            print(f"      {n:32s} gradnorm={float(q.grad.norm()):.4f}")

    # ----------------------------------------------- WEIGHT-UPDATE (does it learn?)
    section("Weight-update test (loss must drop; weights must move)")
    m2 = ProcessLM(cfg)
    before = {n: q.detach().clone() for n, q in m2.named_parameters()}
    opt = torch.optim.AdamW(m2.parameters(), lr=1e-3)
    first = None
    for _ in range(25):
        _, l = m2(ids, labels=labels)
        if first is None:
            first = l.item()
        opt.zero_grad(); l.backward(); opt.step()
    check("loss decreases over 25 steps", l.item() < first, f"{first:.3f} -> {l.item():.3f}")
    moved = all(float((q - before[n]).norm()) > 0 for n, q in m2.named_parameters())
    check("every parameter changed after stepping", moved)

    # ------------------------------------------- trained checkpoint (optional)
    if args.ckpt and Path(args.ckpt).exists():
        section("Trained checkpoint — did embeddings learn structure?")
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        tv = Vocab(ck["vocab"])
        tm = ProcessLM(ModelConfig(**ck["mcfg"])); tm.load_state_dict(ck["model"]); tm.eval()
        emb = tm.tok.weight.detach()

        def cos(a, b):
            ia, ib = tv.stoi.get(a), tv.stoi.get(b)
            if ia is None or ib is None:
                return None
            return float(F.cosine_similarity(emb[ia], emb[ib], dim=0))

        syn = [("STRIP RESIST", "STRIP PHOTORESIST"), ("RCA CLEAN 1", "WET CLEAN RCA1"),
               ("WET CLEAN RCA2", "RCA CLEAN 2")]
        syn_cos = [c for c in (cos(a, b) for a, b in syn) if c is not None]
        torch.manual_seed(1)
        rand_pairs = torch.randint(4, len(tv), (200, 2))
        rand_cos = [float(F.cosine_similarity(emb[i], emb[j], dim=0))
                    for i, j in rand_pairs if int(i) != int(j)]
        s_mean = sum(syn_cos) / max(1, len(syn_cos))
        r_mean = sum(rand_cos) / max(1, len(rand_cos))
        print(f"    synonym-pair cos={s_mean:.3f}  random-pair cos={r_mean:.3f}")
        check("synonyms more similar than random pairs (learned semantics)", s_mean > r_mean + 0.05)
        check("checkpoint val_loss recorded", "val_loss" in ck, f"val_loss={ck.get('val_loss')}")

    # ----------------------------------------------------------------- summary
    section("SUMMARY")
    passed, total = sum(_results), len(_results)
    print(f"  {passed}/{total} checks passed")
    if passed != total:
        print("  >>> SOME CHECKS FAILED — investigate before trusting training results.")
        sys.exit(1)
    print("  ALL HEALTH CHECKS PASSED ✅")


if __name__ == "__main__":
    main()
