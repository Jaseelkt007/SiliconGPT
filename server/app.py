"""Flask backend for the SiliconGPT Process Intelligence Lab UI.

Run locally:
    pip install -r server/requirements.txt
    export CHECKPOINT_PATH=/path/to/best.pt        # optional; defaults to checkpoints/best.pt
    python server/app.py                           # serves on http://localhost:5050

Endpoints (all JSON unless noted):
    GET  /api/health
    GET  /api/vocab
    GET  /api/rules
    POST /api/predict/nextstep   {partial_sequence, k}
    POST /api/predict/complete   {partial_sequence, max_new, greedy, temperature}
    POST /api/generate           {prefix?, max_new, temperature}
    POST /api/validate           {sequence}
    POST /api/anomaly            {sequence, use_validator}
    POST /api/eval/nextstep      multipart file=<csv>
    POST /api/eval/completion    multipart file=<csv>
    POST /api/eval/anomaly       multipart file=<csv>
    POST /api/eval/ood           multipart file=<csv>, task=nextstep|completion|anomaly
"""
from __future__ import annotations

import csv
import io
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import server.inference as INF                           # noqa: E402
from process_logic.score import (                        # noqa: E402
    score_nextstep,
    score_completion,
    score_anomaly as score_anomaly_metrics,
)
from process_logic import generation as G                # noqa: E402


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app,
         resources={r"/api/*": {"origins": "*"}},
         supports_credentials=False,
         expose_headers=["Content-Type"])

    INF.init()

    # ---------- helpers ---------- #
    def _split(s: str) -> list[str]:
        return [t for t in (s or "").split("|") if t]

    def _need_model():
        if not INF.ready():
            return jsonify({"error": INF.STATE.load_error or "model not loaded",
                            "ckpt_path": INF.STATE.ckpt_path}), 503
        return None

    # ---------- health / meta ---------- #
    @app.get("/api/health")
    def health():
        return jsonify({
            "ok": INF.ready(),
            "ckpt_path": INF.STATE.ckpt_path,
            "device": INF.STATE.device,
            "vocab_size": len(INF.STATE.vocab) if INF.STATE.vocab else 0,
            "threshold": INF.STATE.threshold,
            "load_error": INF.STATE.load_error,
            "families": ["mosfet", "igbt", "ic"],
        })

    @app.get("/api/vocab")
    def vocab():
        if (err := _need_model()):
            return err
        return jsonify({"tokens": INF.vocab_tokens()})

    @app.get("/api/rules")
    def rules():
        rs = [
            ("RULE_DEP_NO_CLEAN", "Every deposition step must have a clean step within the prior 12."),
            ("RULE_METAL_ETCH_NO_LITHO", "Metal etch requires EXPOSE LITHO + DEVELOP within prior 15 steps."),
            ("RULE_ETCH_NO_MASK", "Etch must be preceded by an aligned/exposed mask."),
            ("RULE_LITHO_LEVEL_SKIP", "Lithography levels must appear in order; no skipping."),
            ("RULE_IMPLANT_NO_MASK", "Ion implant requires a recent mask."),
            ("RULE_CMP_NO_DEP", "CMP must be preceded by a deposition/fill step."),
            ("RULE_PAD_OPEN_BEFORE_DEP", "Pad-open requires prior passivation deposition."),
            ("RULE_TEST_BEFORE_PASSIVATION", "Electrical test must come after passivation, not before."),
            ("RULE_SHIP_BEFORE_TEST", "Ship/dice steps must occur after electrical test."),
            ("RULE_BACKSIDE_BEFORE_PASSIVATION", "Backside processing must follow front-side passivation."),
        ]
        return jsonify({"rules": [{"id": r, "description": d} for r, d in rs]})

    # ---------- interactive inference ---------- #
    @app.post("/api/predict/nextstep")
    def predict_nextstep():
        if (err := _need_model()):
            return err
        body = request.get_json(force=True, silent=True) or {}
        partial = body.get("partial_sequence", [])
        if isinstance(partial, str):
            partial = _split(partial)
        k = int(body.get("k", 5))
        t0 = time.time()
        preds = INF.predict_topk(partial, k=k)
        return jsonify({"predictions": preds, "latency_ms": int((time.time() - t0) * 1000)})

    @app.post("/api/predict/complete")
    def predict_complete():
        if (err := _need_model()):
            return err
        body = request.get_json(force=True, silent=True) or {}
        partial = body.get("partial_sequence", [])
        if isinstance(partial, str):
            partial = _split(partial)
        max_new = int(body.get("max_new", 220))
        greedy = bool(body.get("greedy", True))
        temperature = float(body.get("temperature", 1.0))
        t0 = time.time()
        gen = INF.complete(partial, max_new=max_new, greedy=greedy, temperature=temperature)
        return jsonify({
            "prefix": partial,
            "generated": gen,
            "full": partial + gen,
            "latency_ms": int((time.time() - t0) * 1000),
        })

    @app.post("/api/generate")
    def generate():
        """Unconditional random sample (V1 model has no family conditioning)."""
        if (err := _need_model()):
            return err
        body = request.get_json(force=True, silent=True) or {}
        prefix = body.get("prefix") or []
        if isinstance(prefix, str):
            prefix = _split(prefix)
        max_new = int(body.get("max_new", 220))
        temperature = float(body.get("temperature", 1.0))
        t0 = time.time()
        gen = INF.sample_random(prefix=prefix, max_new=max_new, temperature=temperature)
        full = prefix + gen
        viols = G.validate_sequence(full)
        return jsonify({
            "prefix": prefix,
            "generated": gen,
            "full": full,
            "is_valid": int(len(viols) == 0),
            "violations": [{"rule": v.rule, "description": v.description,
                            "step_index": v.step_index, "step_name": v.step_name}
                           for v in viols],
            "latency_ms": int((time.time() - t0) * 1000),
        })

    @app.post("/api/validate")
    def validate():
        body = request.get_json(force=True, silent=True) or {}
        seq = body.get("sequence", [])
        if isinstance(seq, str):
            seq = _split(seq)
        viols = INF.validate(seq)
        return jsonify({
            "is_valid": int(len(viols) == 0),
            "violations": viols,
            "n_steps": len(seq),
        })

    @app.post("/api/anomaly")
    def anomaly():
        if (err := _need_model()):
            return err
        body = request.get_json(force=True, silent=True) or {}
        seq = body.get("sequence", [])
        if isinstance(seq, str):
            seq = _split(seq)
        use_validator = bool(body.get("use_validator", True))
        return jsonify(INF.anomaly(seq, use_validator=use_validator))

    # ---------- batch CSV eval ---------- #
    def _read_uploaded_csv() -> list[dict]:
        if "file" not in request.files:
            raise ValueError("missing 'file' upload")
        f = request.files["file"]
        text = f.read().decode("utf-8", errors="replace")
        return list(csv.DictReader(io.StringIO(text)))

    def _read_server_csv(path: Path) -> list[dict]:
        with open(path, encoding="utf-8") as fh:
            return list(csv.DictReader(fh))

    # Per-task processors — shared by the uploaded-CSV routes AND the built-in
    # validation-set route, so both behave identically.
    def _proc_nextstep(rows):
        out, pred_rows = [], []
        for r in rows:
            partial = _split(r.get("PARTIAL_SEQUENCE", ""))
            preds = INF.predict_topk(partial, k=5)
            tokens = [p["token"] for p in preds]
            ex_id = r.get("EXAMPLE_ID", str(len(out)))
            pred_rows.append({"EXAMPLE_ID": ex_id,
                              **{f"RANK_{i+1}": (tokens[i] if i < len(tokens) else "")
                                 for i in range(5)}})
            out.append({"example_id": ex_id, "family": r.get("FAMILY", ""),
                        "partial_sequence": partial, "predictions": preds,
                        "true_next_step": r.get("TRUE_NEXT_STEP")})
        metrics = (score_nextstep(pred_rows, rows)
                   if any(r.get("TRUE_NEXT_STEP") for r in rows) else None)
        return {"rows": out, "metrics": metrics, "n": len(out)}

    def _proc_completion(rows):
        out, pred_rows = [], []
        valid = defaultdict(lambda: [0, 0])   # family -> [n_valid, n_total]
        for r in rows:
            partial = _split(r.get("PARTIAL_SEQUENCE", ""))
            gen = INF.complete(partial, max_new=240, greedy=True)
            ex_id = r.get("EXAMPLE_ID", str(len(out)))
            pred_rows.append({"EXAMPLE_ID": ex_id, "PREDICTED_SEQUENCE": "|".join(gen)})
            fam = r.get("FAMILY", "")
            ok = 0 if INF.validate(partial + gen) else 1   # full generated recipe rule-valid?
            for key in (fam, "ALL"):
                valid[key][0] += ok
                valid[key][1] += 1
            out.append({"example_id": ex_id, "family": fam,
                        "completion_fraction": r.get("COMPLETION_FRACTION"),
                        "partial_sequence": partial, "predicted": gen,
                        "true_suffix": _split(r.get("TRUE_SUFFIX", "")) if r.get("TRUE_SUFFIX") else None})
        metrics = (score_completion(pred_rows, rows)
                   if any(r.get("TRUE_SUFFIX") for r in rows) else {})
        for key, (v, t) in valid.items():        # validity needs no ground truth
            metrics.setdefault(key, {"n": t})
            metrics[key]["validity"] = v / max(1, t)
        return {"rows": out, "metrics": metrics or None, "n": len(out)}

    def _proc_anomaly(rows):
        out, pred_rows = [], []
        for r in rows:
            seq = _split(r.get("SEQUENCE", ""))
            res = INF.anomaly(seq, use_validator=True)
            ex_id = r.get("EXAMPLE_ID", str(len(out)))
            pred_rows.append({"EXAMPLE_ID": ex_id, "IS_VALID": res["is_valid"],
                              "SCORE": res["score"], "PREDICTED_RULE": res["predicted_rule"]})
            out.append({"example_id": ex_id, "family": r.get("FAMILY", ""), "sequence": seq,
                        "is_valid": res["is_valid"], "score": res["score"],
                        "predicted_rule": res["predicted_rule"], "nll": res["nll"],
                        "true_is_valid": int(r["IS_VALID"]) if r.get("IS_VALID", "") != "" else None,
                        "true_rule": r.get("RULE_VIOLATED")})
        metrics = (score_anomaly_metrics(pred_rows, rows)
                   if any(r.get("IS_VALID", "") != "" for r in rows) else None)
        return {"rows": out, "metrics": metrics, "n": len(out)}

    _PROC = {"nextstep": _proc_nextstep, "completion": _proc_completion, "anomaly": _proc_anomaly}

    @app.post("/api/eval/nextstep")
    def eval_nextstep():
        if (err := _need_model()):
            return err
        return jsonify(_proc_nextstep(_read_uploaded_csv()))

    @app.post("/api/eval/completion")
    def eval_completion():
        if (err := _need_model()):
            return err
        return jsonify(_proc_completion(_read_uploaded_csv()))

    @app.post("/api/eval/anomaly")
    def eval_anomaly_route():
        if (err := _need_model()):
            return err
        return jsonify(_proc_anomaly(_read_uploaded_csv()))

    # ---------- built-in held-out data (no upload needed) ---------- #
    @app.get("/api/sample")
    def sample():
        """A held-out (validation-split) example for a family — so the UI can run
        real inference with no upload. Source: data/val_id.csv (in-dist, held-out)."""
        if (err := _need_model()):
            return err
        fam = (request.args.get("family") or "mosfet").lower()
        path = ROOT / "data" / "val_id.csv"
        if not path.exists():
            return jsonify({"error": "validation data not found on server"}), 404
        rows = [r for r in _read_server_csv(path) if r.get("FAMILY", "").lower() == fam]
        if not rows:
            return jsonify({"error": f"no held-out examples for family '{fam}'"}), 404
        i = request.args.get("i")
        n = (int(i) % len(rows)) if (i and i.isdigit()) else 0
        r = rows[n]
        return jsonify({"family": fam, "example_id": r.get("SEQUENCE_ID", f"{fam}-{n}"),
                        "steps": _split(r.get("SEQUENCE", "")), "index": n,
                        "n_available": len(rows), "split": "val_id · held-out · in-distribution"})

    @app.post("/api/eval/builtin")
    def eval_builtin():
        """Run the server's OWN held-out eval set for a task — judges click one
        button, no file needed. ?task=nextstep|completion|anomaly  &limit=N (optional)."""
        if (err := _need_model()):
            return err
        task = (request.args.get("task") or request.form.get("task") or "nextstep").lower()
        if task not in _PROC:
            return jsonify({"error": f"unknown task '{task}'"}), 400
        path = ROOT / "data" / f"eval_{task}.csv"
        if not path.exists():
            return jsonify({"error": f"no built-in eval set for task '{task}'"}), 404
        rows = _read_server_csv(path)
        total = len(rows)
        lim = request.args.get("limit") or request.form.get("limit")
        if lim and str(lim).isdigit():
            rows = rows[: int(lim)]
        res = _PROC[task](rows)
        res["builtin"] = {"task": task, "split": "held-out eval set",
                          "scored": len(rows), "total": total}
        return jsonify(res)

    @app.post("/api/eval/ood")
    def eval_ood():
        """Same as the per-task eval routes but returns per-family breakdown
        explicitly. The CSV must have a FAMILY column; `task` form-field picks
        which scorer to use."""
        if (err := _need_model()):
            return err
        task = (request.form.get("task") or "nextstep").lower()
        rows = _read_uploaded_csv()
        if task == "nextstep":
            pred_rows = []
            for r in rows:
                partial = _split(r.get("PARTIAL_SEQUENCE", ""))
                preds = INF.predict_topk(partial, k=5)
                toks = [p["token"] for p in preds]
                pred_rows.append({"EXAMPLE_ID": r.get("EXAMPLE_ID", ""),
                                  **{f"RANK_{i+1}": (toks[i] if i < len(toks) else "")
                                     for i in range(5)}})
            metrics = score_nextstep(pred_rows, rows)
        elif task == "completion":
            pred_rows = []
            for r in rows:
                partial = _split(r.get("PARTIAL_SEQUENCE", ""))
                gen = INF.complete(partial, max_new=240, greedy=True)
                pred_rows.append({"EXAMPLE_ID": r.get("EXAMPLE_ID", ""),
                                  "PREDICTED_SEQUENCE": "|".join(gen)})
            metrics = score_completion(pred_rows, rows)
        elif task == "anomaly":
            pred_rows = []
            for r in rows:
                seq = _split(r.get("SEQUENCE", ""))
                res = INF.anomaly(seq, use_validator=True)
                pred_rows.append({"EXAMPLE_ID": r.get("EXAMPLE_ID", ""),
                                  "IS_VALID": res["is_valid"],
                                  "SCORE": res["score"],
                                  "PREDICTED_RULE": res["predicted_rule"]})
            metrics = score_anomaly_metrics(pred_rows, rows)
        else:
            return jsonify({"error": f"unknown task '{task}'"}), 400

        fam_counts = defaultdict(int)
        for r in rows:
            fam_counts[r.get("FAMILY", "?")] += 1
        return jsonify({"task": task, "metrics": metrics,
                        "family_counts": dict(fam_counts), "n": len(rows)})

    @app.errorhandler(ValueError)
    def _ve(e):
        return jsonify({"error": str(e)}), 400

    return app


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    host = os.environ.get("HOST", "0.0.0.0")
    create_app().run(host=host, port=port, threaded=True, debug=False)
