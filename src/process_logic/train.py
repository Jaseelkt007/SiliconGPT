"""Train the process-logic model (base LM, next-token prediction).

Server:  pixi run python src/process_logic/train.py --config configs/train_v1.yaml
Smoke:   python src/process_logic/train.py --smoke --device cpu   (needs torch)

V1.1 changes: stable family-balanced validation, early stopping (patience),
W&B logging (flag-gated), and gradient-norm logging.
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import yaml
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from process_logic.vocab import Vocab                                  # noqa: E402
from process_logic.dataset import load_compact_csv, make_dataloader, build_eval_batches  # noqa: E402
from process_logic.model import ProcessLM, ModelConfig                 # noqa: E402


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def get_lr(it, cfg):
    if it < cfg["warmup_iters"]:
        return cfg["lr"] * (it + 1) / cfg["warmup_iters"]
    if it > cfg["max_iters"]:
        return cfg["min_lr"]
    ratio = (it - cfg["warmup_iters"]) / max(1, cfg["max_iters"] - cfg["warmup_iters"])
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return cfg["min_lr"] + coeff * (cfg["lr"] - cfg["min_lr"])


def cycle(loader):
    while True:
        for b in loader:
            yield b


@torch.no_grad()
def evaluate(model, eval_batches, device):
    """Evaluate on a FIXED, family-balanced list of numpy batches (stable metric)."""
    model.eval()
    losses, top1, top3, top5, tot = [], 0, 0, 0, 0
    for b in eval_batches:
        ids = torch.from_numpy(b["input_ids"]).to(device)
        labels = torch.from_numpy(b["labels"]).to(device)
        logits, loss = model(ids, labels=labels)
        losses.append(loss.item())
        shift_logits, shift_labels = logits[:, :-1], labels[:, 1:]
        mask = shift_labels != -100
        topk = shift_logits.topk(5, dim=-1).indices
        correct = (topk == shift_labels.unsqueeze(-1)) & mask.unsqueeze(-1)
        tot += int(mask.sum().item())
        top1 += int(correct[..., :1].any(-1).sum().item())
        top3 += int(correct[..., :3].any(-1).sum().item())
        top5 += int(correct[..., :5].any(-1).sum().item())
    model.train()
    n = max(1, tot)
    return sum(losses) / len(losses), top1 / n, top3 / n, top5 / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/train_v1.yaml"))
    ap.add_argument("--model-config", default=str(ROOT / "configs/model_v1.yaml"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--exclude-family", default=None,
                    help="train/val WITHOUT this family (mosfet|igbt|ic) for the OOD experiment")
    ap.add_argument("--run-name", default="v1")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    mcfg = load_yaml(args.model_config)
    if args.data_dir:
        cfg["data_dir"] = args.data_dir
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.smoke:
        mcfg.update(n_layer=2, n_head=2, n_embd=64, block_size=256, dropout=0.0)
        cfg.update(max_iters=80, eval_interval=20, eval_iters=4, warmup_iters=5,
                   batch_size=16, log_interval=20, patience=1000, wandb=False)

    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])

    def resolve(p):
        p = Path(p)
        return p if p.is_absolute() else ROOT / p

    data_dir = resolve(cfg["data_dir"])
    vocab = Vocab.load(resolve(cfg["vocab_file"]))
    mcfg["vocab_size"] = len(vocab)

    train_ex = load_compact_csv(data_dir / cfg["train_file"])
    val_ex = load_compact_csv(data_dir / cfg["val_file"])
    if args.exclude_family:
        train_ex = [e for e in train_ex if e[0] != args.exclude_family]
        val_ex = [e for e in val_ex if e[0] != args.exclude_family]
        print(f"OOD mode: excluded '{args.exclude_family}' -> train={len(train_ex)} val={len(val_ex)}")
    if args.smoke:
        train_ex, val_ex = train_ex[:500], val_ex[:200]

    train_iter = cycle(make_dataloader(train_ex, vocab, batch_size=cfg["batch_size"], shuffle=True))
    eval_batches = build_eval_batches(val_ex, vocab, batch_size=cfg["batch_size"],
                                      n_batches=cfg["eval_iters"], seed=cfg["seed"], balanced=True)

    model = ProcessLM(ModelConfig(**mcfg)).to(device)
    print(f"params={model.num_params()/1e6:.2f}M vocab={len(vocab)} device={device} "
          f"train={len(train_ex)} val={len(val_ex)} eval_batches={len(eval_batches)}")

    decay = [p for p in model.parameters() if p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.dim() < 2]
    optim = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg["weight_decay"]},
         {"params": nodecay, "weight_decay": 0.0}],
        lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]))

    # ---- optional Weights & Biases ----
    use_wandb = bool(cfg.get("wandb", False))
    wb = None
    if use_wandb:
        import wandb
        wb = wandb.init(project=cfg.get("wandb_project", "silicongpt"),
                        name=args.run_name, config={**cfg, **mcfg})

    use_amp = (device == "cuda")
    ckpt_dir = resolve(cfg["ckpt_dir"]); ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir = resolve(cfg["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(out_dir / "train_log.csv", "w", newline="")
    log_w = csv.writer(log_f)
    log_w.writerow(["iter", "train_loss", "val_loss", "top1", "top3", "top5", "lr", "grad_norm"])

    best_val, patience, no_improve = float("inf"), cfg.get("patience", 8), 0
    first_loss, loss_val, gnorm = None, float("nan"), 0.0
    t0 = time.time()
    for it in range(cfg["max_iters"] + 1):
        lr = get_lr(it, cfg)
        for g in optim.param_groups:
            g["lr"] = lr
        batch = next(train_iter)
        ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        if use_amp:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(ids, labels=labels)
        else:
            _, loss = model(ids, labels=labels)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"]))
        optim.step()
        loss_val = loss.item()
        if first_loss is None:
            first_loss = loss_val
        if it % cfg["log_interval"] == 0:
            print(f"iter {it:6d} | loss {loss_val:.4f} | grad {gnorm:.3f} | lr {lr:.2e} | {time.time()-t0:.1f}s")
        if it % cfg["eval_interval"] == 0:
            vl, t1, t3, t5 = evaluate(model, eval_batches, device)
            print(f"  [eval] iter {it} val_loss {vl:.4f} top1 {t1:.3f} top3 {t3:.3f} top5 {t5:.3f}")
            log_w.writerow([it, loss_val, vl, t1, t3, t5, lr, gnorm]); log_f.flush()
            if wb:
                wb.log({"iter": it, "train_loss": loss_val, "val_loss": vl, "top1": t1,
                        "top3": t3, "top5": t5, "lr": lr, "grad_norm": gnorm})
            if vl < best_val - 1e-4:
                best_val, no_improve = vl, 0
                torch.save({"model": model.state_dict(), "mcfg": mcfg,
                            "vocab": vocab.itos, "iter": it, "val_loss": vl},
                           ckpt_dir / "best.pt")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"early stop at iter {it} (no val improvement for {patience} evals; best {best_val:.4f})")
                    break
    log_f.close()
    if wb:
        wb.summary["best_val"] = best_val
        wb.finish()
    print(f"done. best_val={best_val:.4f} ckpt={ckpt_dir / 'best.pt'}")

    if args.smoke:
        assert loss_val < first_loss, f"SMOKE FAIL: loss did not drop ({first_loss:.3f} -> {loss_val:.3f})"
        print(f"SMOKE OK: loss {first_loss:.3f} -> {loss_val:.3f}")


if __name__ == "__main__":
    main()
