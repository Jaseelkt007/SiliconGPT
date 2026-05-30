#!/usr/bin/env python3
"""run_experiment.py — the co-scientist-lab bridge: spec -> train -> score -> JSON.

Stitches the existing scripts (train.py, predict.py, score.py) into one call that a
hypothesis's *experiment spec* drives. It TRAINS an in-distribution model and one model
per OOD fold (held-out family), SCORES next-step / completion / anomaly in-distribution
AND on each held-out family, computes ood_detect (and family_id when available), then
writes a RESULT RECORD (schema: CO_SCIENTIST_LAB_DESIGN.md §4) that elo.py
ingest-experiment attaches to the hypothesis.

Two entry modes:
  # from an experiment spec (the lab's Experiment agent emits this):
  python scripts/run_experiment.py --spec spec.json --tier smoke --out result.json
  # proof / ad-hoc (reuse existing checkpoints, no training — validates the scoring path):
  python scripts/run_experiment.py --reuse-ckpts --tier full --out proof.json

Spec schema (subset of §4 actually consumed):
  { "id","title","tier":"smoke|full",
    "base_train":"configs/train_v1.yaml","base_model":"configs/model_v1.yaml",
    "overrides": { "model.n_embd":256, "train.max_iters":1000, "model.pos_encoding":"nope",
                   "model.family_conditioning":true, "data.augmentation":"cross_family_recomb" },
    "ood": { "folds": ["ic"] },            # int N -> first N of [mosfet,igbt,ic]; or explicit list
    "tasks": ["nextstep","completion","anomaly","ood_detect","family_id"] }

Design notes:
- TRAIN is run as a subprocess of train.py (isolates global state between folds, and stays
  robust to any new config knob — the knob just appears in the merged YAML).
- PREDICT is run as a subprocess of predict.py; SCORE imports score.py's pure functions.
- "smoke" tier = the *hypothesis's own config* at reduced scale (small model + few iters), NOT
  train.py's fixed tiny --smoke model, so the directional signal reflects the actual idea.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import process_logic.score as S  # pure-stdlib scorers

FAMILIES = ["mosfet", "igbt", "ic"]
PY = sys.executable  # the pixi-env python that's running us


# ---------------------------------------------------------------- config plumbing
def load_yaml(p):
    with open(p) as f:
        return yaml.safe_load(f)


def apply_overrides(train_cfg, model_cfg, overrides):
    """Route dotted override keys to the right config dict.

    'model.X' -> model_cfg ; 'train.X'/'data.X' -> train_cfg ; bare known model knobs -> model_cfg.
    Data/augmentation knobs live in the TRAIN config (train.py reads them)."""
    MODEL_BARE = {"pos_encoding", "family_conditioning", "family_dropout", "tie_weights",
                  "n_layer", "n_head", "n_embd", "block_size", "dropout", "mlp_ratio",
                  "rope_base", "objective", "aux_family_head", "weight_share"}
    for k, v in (overrides or {}).items():
        if k.startswith("model."):
            model_cfg[k[len("model."):]] = v
        elif k.startswith("train."):
            train_cfg[k[len("train."):]] = v
        elif k.startswith("data."):
            train_cfg[k] = v                       # keep 'data.augmentation' verbatim for train.py
        elif k in MODEL_BARE:
            model_cfg[k] = v
        else:
            train_cfg[k] = v
    return train_cfg, model_cfg


def smoke_scale(train_cfg, model_cfg, overrides):
    """Reduce a config for a cheap, directional CPU smoke (fidelity calibrated in task-6).

    Keeps the hypothesis's structural knobs (pos_encoding, conditioning, augmentation) but
    shrinks size + iters. A model dim explicitly pinned by the spec's overrides is respected;
    otherwise it is shrunk to the smoke default."""
    pinned = set(k.split(".")[-1] for k in (overrides or {}))
    sm_model = {"n_layer": 4, "n_embd": 128, "n_head": 4, "dropout": 0.0}
    sm_train = {"max_iters": 400, "eval_interval": 50, "eval_iters": 10,
                "warmup_iters": 20, "batch_size": 32, "log_interval": 100, "patience": 1000}
    for k, v in sm_model.items():
        if k not in pinned:
            model_cfg[k] = v
    for k, v in sm_train.items():
        if k not in pinned:
            train_cfg[k] = v
    return train_cfg, model_cfg


# ---------------------------------------------------------------- train / predict (subprocess)
def write_tmp_yaml(d, path):
    with open(path, "w") as f:
        yaml.safe_dump(d, f)


def train_model(train_cfg, model_cfg, exclude_family, ckpt_dir, out_dir, device, tmp):
    """Run train.py as a subprocess with merged configs. Returns (best_val, wall_s)."""
    tcfg = dict(train_cfg)
    tcfg["ckpt_dir"] = str(ckpt_dir)
    tcfg["out_dir"] = str(out_dir)
    tcfg["wandb"] = False
    tp = tmp / "train.yaml"
    mp = tmp / "model.yaml"
    write_tmp_yaml(tcfg, tp)
    write_tmp_yaml(model_cfg, mp)
    cmd = [PY, str(ROOT / "src/process_logic/train.py"),
           "--config", str(tp), "--model-config", str(mp),
           "--ckpt-dir", str(ckpt_dir), "--out-dir", str(out_dir), "--device", device]
    if exclude_family:
        cmd += ["--exclude-family", exclude_family]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.time() - t0
    if r.returncode != 0:
        sys.stderr.write(r.stdout + "\n" + r.stderr + "\n")
        raise RuntimeError(f"train.py failed (exclude={exclude_family}) rc={r.returncode}")
    best_val = float("nan")
    for line in r.stdout.splitlines():
        if line.startswith("done. best_val="):
            best_val = float(line.split("best_val=")[1].split()[0])
    return best_val, wall


def run_predict(ckpt, ns_in, comp_in, anom_in, calib, out_dir, device, tasks):
    cmd = [PY, str(ROOT / "src/process_logic/predict.py"),
           "--ckpt", str(ckpt), "--out-dir", str(out_dir), "--device", device]
    if "nextstep" in tasks and ns_in:
        cmd += ["--nextstep-input", str(ns_in)]
    if "completion" in tasks and comp_in:
        cmd += ["--completion-input", str(comp_in)]
    if "anomaly" in tasks and anom_in:
        cmd += ["--anomaly-input", str(anom_in), "--calib-file", str(calib)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout + "\n" + r.stderr + "\n")
        raise RuntimeError(f"predict.py failed rc={r.returncode}")


# ---------------------------------------------------------------- eval-CSV filtering
def filter_eval(src, dst, family=None, max_per_family=None):
    """Write a copy of an eval CSV keeping only `family` rows and/or capping count per family."""
    rows = S.read_rows(src)
    if family:
        rows = [r for r in rows if r.get("FAMILY") == family]
    if max_per_family:
        seen = {}
        kept = []
        for r in rows:
            f = r.get("FAMILY", "ALL")
            seen[f] = seen.get(f, 0) + 1
            if seen[f] <= max_per_family:
                kept.append(r)
        rows = kept
    if not rows:
        return None
    with open(dst, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    return dst


# ---------------------------------------------------------------- scoring
def _gt_on_predicted(pred_rows, gt_rows):
    """Restrict ground-truth rows to the EXAMPLE_IDs actually predicted.

    Critical when eval is capped (max_eval_per_family) or family-filtered: the scorers
    iterate over ALL gt rows and count a missing prediction as a miss, which would deflate
    every metric. We score on the intersection (the same 'common examples' convention the
    benchmark uses), so capped/full runs are comparable like-for-like."""
    have = {r["EXAMPLE_ID"] for r in pred_rows}
    return [g for g in gt_rows if g["EXAMPLE_ID"] in have]


def score_split(pred_dir, gt_dir, family_key, tasks):
    """Score whatever prediction CSVs exist in pred_dir against ground truth; return the
    metrics for `family_key` ('ALL' for in-dist, the held-out family for OOD)."""
    out = {}
    pred_dir, gt_dir = Path(pred_dir), Path(gt_dir)
    if "nextstep" in tasks and (pred_dir / "nextstep.csv").exists():
        pr = S.read_rows(pred_dir / "nextstep.csv")
        m = S.score_nextstep(pr, _gt_on_predicted(pr, S.read_rows(gt_dir / "eval_nextstep.csv")))
        if family_key in m:
            out["nextstep"] = {k: round(v, 4) for k, v in m[family_key].items() if k != "n"}
            out["nextstep"]["n"] = m[family_key]["n"]
    if "completion" in tasks and (pred_dir / "completion.csv").exists():
        pr = S.read_rows(pred_dir / "completion.csv")
        m = S.score_completion(pr, _gt_on_predicted(pr, S.read_rows(gt_dir / "eval_completion.csv")))
        if family_key in m:
            out["completion"] = {k: round(v, 4) for k, v in m[family_key].items() if k != "n"}
            out["completion"]["n"] = m[family_key]["n"]
    if "anomaly" in tasks and (pred_dir / "anomaly.csv").exists():
        pr = S.read_rows(pred_dir / "anomaly.csv")
        m = S.score_anomaly(pr, _gt_on_predicted(pr, S.read_rows(gt_dir / "eval_anomaly.csv")))
        if family_key in m:
            out["anomaly"] = {k: (round(v, 4) if isinstance(v, float) else v)
                              for k, v in m[family_key].items() if k not in ("n", "confusion")}
            out["anomaly"]["n"] = m[family_key]["n"]
    return out


def ood_detect_auroc(ckpt, held_family, data_dir, device, n=300):
    """AUROC for flagging the unseen family as OOD via mean per-token NLL.
    Positive class = held-out family (should have HIGHER nll under a model that never saw it)."""
    from process_logic.generate import load_checkpoint
    from process_logic.anomaly import sequence_nll
    model, vocab = load_checkpoint(ckpt, device)
    val = S.read_rows(Path(data_dir) / "val_id.csv")
    pos = [r["SEQUENCE"].split("|") for r in val if r["FAMILY"] == held_family][:n]
    neg = [r["SEQUENCE"].split("|") for r in val if r["FAMILY"] != held_family][:n]
    if not pos or not neg:
        return None
    scores = [sequence_nll(model, vocab, s, device) for s in pos] + \
             [sequence_nll(model, vocab, s, device) for s in neg]
    labels = [1] * len(pos) + [0] * len(neg)
    return round(S.roc_auc(scores, labels), 4)


# ---------------------------------------------------------------- baselines (from on-disk artifacts)
def compute_baselines(gt_dir):
    """Read on-disk baseline/prev-best prediction CSVs and score them, so vs_baseline deltas
    are computed against real artifacts, not hard-coded constants."""
    res = ROOT / "extras/results"
    gt = Path(gt_dir)
    b = {"ngram_top1": None, "ngram_token_acc": None,
         "prevbest_id_top1": None, "prevbest_id_token_acc": None,
         "prevbest_ood_top1": None}
    try:
        m = S.score_nextstep(S.read_rows(res / "baseline_ngram/nextstep.csv"),
                            S.read_rows(gt / "eval_nextstep.csv"))
        b["ngram_top1"] = round(m["ALL"]["top1"], 4)
    except Exception:
        pass
    try:
        m = S.score_completion(S.read_rows(res / "baseline_ngram/completion.csv"),
                             S.read_rows(gt / "eval_completion.csv"))
        b["ngram_token_acc"] = round(m["ALL"]["token_acc"], 4)
    except Exception:
        pass
    try:
        m = S.score_nextstep(S.read_rows(res / "nextstep.csv"),
                            S.read_rows(gt / "eval_nextstep.csv"))
        b["prevbest_id_top1"] = round(m["ALL"]["top1"], 4)
    except Exception:
        pass
    try:
        m = S.score_completion(S.read_rows(res / "completion.csv"),
                             S.read_rows(gt / "eval_completion.csv"))
        b["prevbest_id_token_acc"] = round(m["ALL"]["token_acc"], 4)
    except Exception:
        pass
    # prev-best OOD: 3-fold held-out-family next-step top-1, averaged
    fold_top1 = []
    for fam in FAMILIES:
        try:
            m = S.score_nextstep(S.read_rows(res / f"ood_{fam}/nextstep.csv"),
                                S.read_rows(gt / "eval_nextstep.csv"))
            if fam in m:
                fold_top1.append(m[fam]["top1"])
        except Exception:
            pass
    if fold_top1:
        b["prevbest_ood_top1"] = round(sum(fold_top1) / len(fold_top1), 4)
    return b


def make_verdict(id_m, ood_m, base, margin=0.01, id_eps=0.005):
    """improves-ood | regresses-id | neutral."""
    id_top1 = id_m.get("nextstep", {}).get("top1")
    ood_top1 = ood_m.get("nextstep", {}).get("top1") if ood_m else None
    pb_id = base.get("prevbest_id_top1")
    pb_ood = base.get("prevbest_ood_top1")
    if id_top1 is not None and pb_id is not None and id_top1 < pb_id - id_eps:
        return "regresses-id"
    if ood_top1 is not None and pb_ood is not None and ood_top1 > pb_ood + margin:
        return "improves-ood"
    return "neutral"


# ---------------------------------------------------------------- main pipeline
def resolve_folds(ood_spec):
    if not ood_spec:
        return ["ic"]
    folds = ood_spec.get("folds", 1)
    if isinstance(folds, int):
        return FAMILIES[:folds] if folds <= len(FAMILIES) else FAMILIES
    return [f for f in folds if f in FAMILIES]


def run(spec, tier, device, reuse_ckpts, out_path, workdir):
    gt_dir = ROOT / "data"
    tasks = spec.get("tasks", ["nextstep", "completion", "anomaly", "ood_detect"])
    folds = resolve_folds(spec.get("ood"))
    max_eval = spec.get("max_eval_per_family", 120 if tier == "smoke" else None)
    base_train = ROOT / spec.get("base_train", "configs/train_v1.yaml")
    base_model = ROOT / spec.get("base_model", "configs/model_v1.yaml")
    overrides = spec.get("overrides", {})
    train_cfg = load_yaml(base_train)
    model_cfg = load_yaml(base_model)
    apply_overrides(train_cfg, model_cfg, overrides)
    if tier == "smoke":
        smoke_scale(train_cfg, model_cfg, overrides)

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    rec = {"id": spec.get("id", "adhoc"), "title": spec.get("title", ""),
           "tier": tier, "status": "running", "device": device,
           "config": {"overrides": overrides, "folds": folds, "max_eval_per_family": max_eval},
           "train": {}, "metrics": {"id": {}, "ood": {}, "ood_per_fold": {}}}

    # ---- in-distribution model ----
    ns = filter_eval(gt_dir / "eval_nextstep.csv", workdir / "ns_id.csv", None, max_eval)
    comp = filter_eval(gt_dir / "eval_completion.csv", workdir / "comp_id.csv", None, max_eval)
    anom = filter_eval(gt_dir / "eval_anomaly.csv", workdir / "anom_id.csv", None, max_eval)
    calib = gt_dir / "val_id.csv"

    if reuse_ckpts:
        id_ckpt = ROOT / "checkpoints/best.pt"
        rec["train"]["id"] = {"reused": str(id_ckpt)}
    else:
        ck = workdir / "ckpt_id"; od = workdir / "out_id"
        with tempfile.TemporaryDirectory() as tmp:
            bv, wall = train_model(train_cfg, model_cfg, None, ck, od, device, Path(tmp))
        id_ckpt = ck / "best.pt"
        rec["train"]["id"] = {"final_val_loss": bv, "wall_s": round(wall, 1)}

    pid = workdir / "pred_id"
    run_predict(id_ckpt, ns, comp, anom, calib, pid, device, tasks)
    rec["metrics"]["id"] = score_split(pid, gt_dir, "ALL", tasks)

    # ---- OOD folds ----
    ood_accum = {}
    ood_auroc = []
    for fam in folds:
        if reuse_ckpts:
            f_ckpt = ROOT / f"checkpoints/ood_{fam}/best.pt"
            rec["train"].setdefault("ood", {})[fam] = {"reused": str(f_ckpt)}
        else:
            ck = workdir / f"ckpt_ood_{fam}"; od = workdir / f"out_ood_{fam}"
            with tempfile.TemporaryDirectory() as tmp:
                bv, wall = train_model(train_cfg, model_cfg, fam, ck, od, device, Path(tmp))
            f_ckpt = ck / "best.pt"
            rec["train"].setdefault("ood", {})[fam] = {"final_val_loss": bv, "wall_s": round(wall, 1)}

        ns_f = filter_eval(gt_dir / "eval_nextstep.csv", workdir / f"ns_{fam}.csv", fam, max_eval)
        comp_f = filter_eval(gt_dir / "eval_completion.csv", workdir / f"comp_{fam}.csv", fam, max_eval)
        pf = workdir / f"pred_ood_{fam}"
        run_predict(f_ckpt, ns_f, comp_f, None, calib, pf, device,
                    [t for t in tasks if t in ("nextstep", "completion")])
        fm = score_split(pf, gt_dir, fam, tasks)
        rec["metrics"]["ood_per_fold"][fam] = fm
        for task, md in fm.items():
            ood_accum.setdefault(task, []).append(md)
        if "ood_detect" in tasks:
            au = ood_detect_auroc(f_ckpt, fam, gt_dir, device)
            if au is not None:
                ood_auroc.append(au)

    # average OOD across folds
    for task, lst in ood_accum.items():
        keys = [k for k in lst[0] if k != "n"]
        rec["metrics"]["ood"][task] = {k: round(sum(d[k] for d in lst) / len(lst), 4) for k in keys}
        rec["metrics"]["ood"][task]["n_folds"] = len(lst)
    if ood_auroc:
        rec["metrics"]["ood_detect"] = {"auroc": round(sum(ood_auroc) / len(ood_auroc), 4),
                                        "per_fold": ood_auroc}

    # ---- baselines + verdict ----
    base = compute_baselines(gt_dir)
    rec["baselines"] = base
    id_top1 = rec["metrics"]["id"].get("nextstep", {}).get("top1")
    ood_top1 = rec["metrics"]["ood"].get("nextstep", {}).get("top1")
    rec["vs_baseline"] = {
        "ngram_top1_delta": (round(id_top1 - base["ngram_top1"], 4)
                             if id_top1 is not None and base["ngram_top1"] is not None else None),
        "prevbest_id_top1_delta": (round(id_top1 - base["prevbest_id_top1"], 4)
                                   if id_top1 is not None and base["prevbest_id_top1"] is not None else None),
        "prevbest_ood_top1_delta": (round(ood_top1 - base["prevbest_ood_top1"], 4)
                                    if ood_top1 is not None and base["prevbest_ood_top1"] is not None else None),
    }
    rec["verdict"] = make_verdict(rec["metrics"]["id"], rec["metrics"]["ood"], base)
    rec["status"] = "done"

    with open(out_path, "w") as f:
        json.dump(rec, f, indent=2)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", help="experiment spec JSON")
    ap.add_argument("--tier", choices=["smoke", "full"], default="smoke")
    ap.add_argument("--out", required=True, help="result-record JSON output path")
    ap.add_argument("--device", default=None)
    ap.add_argument("--reuse-ckpts", action="store_true",
                    help="skip training; score checkpoints/best.pt + checkpoints/ood_<fam>/best.pt "
                         "(validates the scoring pipeline against known numbers)")
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--folds", help="comma list e.g. ic,igbt,mosfet (overrides spec)")
    ap.add_argument("--max-eval-per-family", type=int, default=None)
    args = ap.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    spec = json.load(open(args.spec)) if args.spec else {}
    spec.setdefault("id", "adhoc")
    if args.folds:
        spec["ood"] = {"folds": args.folds.split(",")}
    if args.max_eval_per_family is not None:
        spec["max_eval_per_family"] = args.max_eval_per_family

    workdir = args.workdir or tempfile.mkdtemp(prefix="run_exp_")
    rec = run(spec, args.tier, device, args.reuse_ckpts, args.out, workdir)

    # console summary
    idm = rec["metrics"]["id"]; oodm = rec["metrics"]["ood"]
    print(f"\n=== {rec['id']} ({rec['tier']}) verdict={rec['verdict']} ===")
    if idm.get("nextstep"):
        print(f"  ID  next-step top1={idm['nextstep'].get('top1')} top5={idm['nextstep'].get('top5')} "
              f"mrr={idm['nextstep'].get('mrr')}")
    if idm.get("completion"):
        print(f"  ID  completion token_acc={idm['completion'].get('token_acc')}")
    if oodm.get("nextstep"):
        print(f"  OOD next-step top1={oodm['nextstep'].get('top1')} ({oodm['nextstep'].get('n_folds')} folds)")
    if rec["metrics"].get("ood_detect"):
        print(f"  OOD-detect auroc={rec['metrics']['ood_detect']['auroc']}")
    print(f"  vs_baseline {rec['vs_baseline']}")
    print(f"  result -> {args.out}")


if __name__ == "__main__":
    main()
