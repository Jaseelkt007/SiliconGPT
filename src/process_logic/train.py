"""Train the process-logic model (base LM, next-token prediction).

Run on the GPU server:  pixi run python src/process_logic/train.py --config configs/train_v1.yaml
Local CPU smoke test:    python src/process_logic/train.py --smoke --device cpu   (needs torch)
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
from process_logic.vocab import Vocab            # noqa: E402
from process_logic.dataset import load_compact_csv, make_dataloader  # noqa: E402
from process_logic.model import ProcessLM, ModelConfig               # noqa: E402


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
def evaluate(model, val_iter, device, eval_iters):
    model.eval()
    losses, top1, top3, top5, tot = [], 0, 0, 0, 0
    for _ in range(eval_iters):
        batch = next(val_iter)
        ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        logits, loss = model(ids, labels=labels)
        losses.append(loss.item())
        shift_logits = logits[:, :-1]
        shift_labels = labels[:, 1:]
        mask = shift_labels != -100
        k = min(5, shift_logits.size(-1))
        topk = shift_logits.topk(k, dim=-1).indices            # [B,T-1,k]
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
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    mcfg = load_yaml(args.model_config)
    if args.data_dir:
        cfg["data_dir"] = args.data_dir
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.smoke:
        mcfg.update(n_layer=2, n_head=2, n_embd=64, block_size=256, dropout=0.0)
        cfg.update(max_iters=50, eval_interval=25, eval_iters=5,
                   warmup_iters=5, batch_size=16, log_interval=10)

    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])

    def resolve(p):
        p = Path(p)
        return p if p.is_absolute() else ROOT / p

    data_dir = resolve(cfg["data_dir"])
    vocab = Vocab.load(resolve(cfg["vocab_file"]))
    mcfg["vocab_size"] = len(vocab)

    train_ex = load_compact_csv(data_dir / cfg["train_file"])
    val_ex = load_compact_csv(data_dir / cfg["val_file"])
    if args.smoke:
        train_ex, val_ex = train_ex[:500], val_ex[:100]

    train_iter = cycle(make_dataloader(train_ex, vocab, batch_size=cfg["batch_size"], shuffle=True))
    val_iter = cycle(make_dataloader(val_ex, vocab, batch_size=cfg["batch_size"], shuffle=False))

    model = ProcessLM(ModelConfig(**mcfg)).to(device)
    print(f"params={model.num_params()/1e6:.2f}M vocab={len(vocab)} device={device} "
          f"train={len(train_ex)} val={len(val_ex)}")

    decay = [p for p in model.parameters() if p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.dim() < 2]
    optim = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg["weight_decay"]},
         {"params": nodecay, "weight_decay": 0.0}],
        lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]))

    use_amp = (device == "cuda")
    ckpt_dir = resolve(cfg["ckpt_dir"]); ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir = resolve(cfg["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(out_dir / "train_log.csv", "w", newline="")
    log_w = csv.writer(log_f)
    log_w.writerow(["iter", "train_loss", "val_loss", "top1", "top3", "top5", "lr"])

    best_val, first_loss, t0 = float("inf"), None, time.time()
    loss_val = float("nan")
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        optim.step()
        loss_val = loss.item()
        if first_loss is None:
            first_loss = loss_val
        if it % cfg["log_interval"] == 0:
            print(f"iter {it:6d} | loss {loss_val:.4f} | lr {lr:.2e} | {time.time()-t0:.1f}s")
        if it % cfg["eval_interval"] == 0:
            vl, t1, t3, t5 = evaluate(model, val_iter, device, cfg["eval_iters"])
            print(f"  [eval] iter {it} val_loss {vl:.4f} top1 {t1:.3f} top3 {t3:.3f} top5 {t5:.3f}")
            log_w.writerow([it, loss_val, vl, t1, t3, t5, lr]); log_f.flush()
            if vl < best_val:
                best_val = vl
                torch.save({"model": model.state_dict(), "mcfg": mcfg,
                            "vocab": vocab.itos, "iter": it, "val_loss": vl},
                           ckpt_dir / "best.pt")
    log_f.close()
    print(f"done. best_val={best_val:.4f} ckpt={ckpt_dir / 'best.pt'}")

    if args.smoke:
        assert loss_val < first_loss, f"SMOKE FAIL: loss did not drop ({first_loss:.3f} -> {loss_val:.3f})"
        print(f"SMOKE OK: loss {first_loss:.3f} -> {loss_val:.3f}")


if __name__ == "__main__":
    main()
