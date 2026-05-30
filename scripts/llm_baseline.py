#!/usr/bin/env python3
"""Frontier-LLM baseline — the bar our from-scratch model must beat.

Runs GPT / Gemini / Claude / Kimi (Moonshot) on the SAME eval files, with the SAME system prompt
(augmented with the exact vocabulary + a real few-shot example + an anomaly task spec, so the
comparison is fair), produces the SAME submission CSVs (nextstep / completion / anomaly), and
records validity + latency. Score it with the SAME src/process_logic/score.py for an apples-to-apples table.

RUN WHERE YOU HAVE INTERNET + API KEYS (your laptop or a Leonardo login node — NOT a compute node).
Set keys: OPENAI_API_KEY / GEMINI_API_KEY / ANTHROPIC_API_KEY / MOONSHOT_API_KEY.

  # verify the pipeline locally (no API key):
  python scripts/llm_baseline.py --provider mock --task nextstep --limit 2
  # one real example to confirm your key/model:
  python scripts/llm_baseline.py --provider openai --model gpt-4o --task nextstep --limit 1
  # a real subsampled run, then score:
  python scripts/llm_baseline.py --provider gemini --model gemini-2.5-pro --task all --limit 200
  python src/process_logic/score.py --pred-dir extras/results/baseline_gemini --gt-dir data

NOTE: model IDs change often — pass --model with each provider's CURRENT top model.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from process_logic.vocab import Vocab, SPECIALS          # noqa: E402
from process_logic import generation as G                 # noqa: E402
from process_logic.dataset import load_compact_csv        # noqa: E402

# load API keys from process-logic/.env if present (NEVER commit .env)
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

RULE_IDS = [
    "RULE_DEP_NO_CLEAN", "RULE_METAL_ETCH_NO_LITHO", "RULE_ETCH_NO_MASK",
    "RULE_LITHO_LEVEL_SKIP", "RULE_IMPLANT_NO_MASK", "RULE_CMP_NO_DEP",
    "RULE_PAD_OPEN_BEFORE_DEP", "RULE_TEST_BEFORE_PASSIVATION",
    "RULE_SHIP_BEFORE_TEST", "RULE_BACKSIDE_BEFORE_PASSIVATION",
]

# Defaults reflect May 2026 docs — verify in each dashboard; override with --model.
# "no_think" = how to DISABLE reasoning per provider (applied unless --thinking is set).
PROVIDERS = {
    "openai":    {"base_url": None, "key": "OPENAI_API_KEY", "model": "gpt-5.5", "kind": "openai",
                  "no_think": {"kwargs": {"reasoning_effort": "none"}}},
    "gemini":    {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
                  "key": "GEMINI_API_KEY", "model": "gemini-3.5-flash", "kind": "openai",
                  # gemini-3.5-flash cannot FULLY disable thinking; "minimal" is the floor.
                  # Use gemini-2.5-flash + thinking_budget=0 if you need true zero.
                  "no_think": {"extra_body": {"google": {"thinking_config": {"thinking_level": "minimal"}}}}},
    "kimi":      {"base_url": "https://api.moonshot.ai/v1", "key": "MOONSHOT_API_KEY",
                  "model": "kimi-k2.6", "kind": "openai",
                  # Kimi K2.6 thinks BY DEFAULT — must disable explicitly.
                  "no_think": {"extra_body": {"thinking": {"type": "disabled"}}}},
    "anthropic": {"base_url": None, "key": "ANTHROPIC_API_KEY", "model": "claude-sonnet-4-6",
                  "kind": "anthropic", "no_think": {}},  # extended thinking off by default (omit param)
    "mock":      {"base_url": None, "key": None, "model": "mock", "kind": "mock", "no_think": {}},
}

ANOMALY_SPEC = """
# TASK: ANOMALY
Input Format
TASK=ANOMALY
SEQUENCE: <full pipe-separated sequence>
Output Format
IS_VALID: <1 if the sequence obeys ALL rules, else 0>
CONFIDENCE: <0.0-1.0 probability the sequence is VALID>
RULE: <if invalid, the single violated rule id from the list below; else leave empty>
Valid rule ids: """ + ", ".join(RULE_IDS) + """
Requirements: output only those three lines, no explanation.
"""


# --------------------------------------------------------------------------- #
# vocab mapping (LLM wording -> our exact step tokens) — testable, no API
# --------------------------------------------------------------------------- #
def _lev(a, b):
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def nearest_vocab(token, steps, steps_upper):
    t = token.strip().strip('"').strip().upper()
    if not t:
        return ""
    if t in steps_upper:
        return steps_upper[t]
    # nearest by edit distance (handles minor LLM wording drift)
    best, bestd = t, 10 ** 9
    for s in steps:
        d = _lev(t, s.upper())
        if d < bestd:
            best, bestd = s, d
    return best


def load_steps():
    v = Vocab.load(ROOT / "vocab.json")
    return [t for t in v.itos if t not in SPECIALS]


# --------------------------------------------------------------------------- #
# few-shot built from REAL data (exact vocab, real structure) — fair + clean
# --------------------------------------------------------------------------- #
def build_fewshot(task):
    seq = load_compact_csv(ROOT / "data" / "train_pool.csv")[0][1]  # a real recipe
    if task == "nextstep":
        k = 22
        ranks = [seq[k]] + [s for s in seq[k + 1:k + 5]]
        body = "\n".join(f"RANK_{i+1}: {s}" for i, s in enumerate(ranks[:5]))
        return f"EXAMPLE\nTASK=NEXT_STEP\nPARTIAL_SEQUENCE: {'|'.join(seq[:k])}\n{body}\n"
    if task == "completion":
        k = 18
        suf = seq[k:k + 10]
        return ("EXAMPLE\nTASK=COMPLETE_SEQUENCE\nPARTIAL_SEQUENCE: "
                f"{'|'.join(seq[:k])}\n" + "\n".join(suf) + "\n")
    # anomaly: one valid, one corrupted (ship before test)
    bad = [s for s in seq if s != "SHIP LOT"]
    bad.insert(bad.index("WAFER SORT TEST"), "SHIP LOT")
    return ("EXAMPLE 1 (valid)\nTASK=ANOMALY\nSEQUENCE: " + "|".join(seq) +
            "\nIS_VALID: 1\nCONFIDENCE: 0.97\nRULE:\n\n"
            "EXAMPLE 2 (invalid)\nTASK=ANOMALY\nSEQUENCE: " + "|".join(bad) +
            "\nIS_VALID: 0\nCONFIDENCE: 0.04\nRULE: RULE_SHIP_BEFORE_TEST\n")


def build_system(task, steps):
    base = (ROOT / "baselines" / "system_prompt.txt").read_text(encoding="utf-8")
    return (base + "\n\n# VALID STEP VOCABULARY (use ONLY these exact names)\n"
            + ", ".join(steps) + "\n" + ANOMALY_SPEC
            + "\n# WORKED EXAMPLE\n" + build_fewshot(task))


def build_user(task, row):
    if task == "anomaly":
        return f"TASK=ANOMALY\nSEQUENCE: {row['SEQUENCE']}"
    tag = "NEXT_STEP" if task == "nextstep" else "COMPLETE_SEQUENCE"
    return f"TASK={tag}\nPARTIAL_SEQUENCE: {row['PARTIAL_SEQUENCE']}"


# --------------------------------------------------------------------------- #
# providers
# --------------------------------------------------------------------------- #
def _mock(task):
    s = load_steps()
    if task == "nextstep":
        return "\n".join(f"RANK_{i+1}: {s[i]}" for i in range(5))
    if task == "completion":
        return "\n".join(s[10:18])
    return "IS_VALID: 1\nCONFIDENCE: 0.95\nRULE:"


def query(provider, model, system, user, temperature, task, thinking=False):
    p = PROVIDERS[provider]
    if p["kind"] == "mock":
        return _mock(task)
    nt = {} if thinking else p.get("no_think", {})        # disable reasoning unless --thinking
    if p["kind"] == "openai":
        from openai import OpenAI
        client = OpenAI(base_url=p["base_url"], api_key=os.environ[p["key"]])
        kwargs = dict(nt.get("kwargs", {}))               # e.g. reasoning_effort=none (OpenAI)
        eb = nt.get("extra_body")                          # e.g. gemini thinking_level / kimi thinking
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            r = client.chat.completions.create(model=model, temperature=temperature,
                                               messages=msgs, extra_body=eb, **kwargs)
        except Exception as e:                             # some reasoning models reject non-default temp
            if "temperature" in str(e).lower():
                r = client.chat.completions.create(model=model, messages=msgs, extra_body=eb, **kwargs)
            else:
                raise
        return r.choices[0].message.content
    if p["kind"] == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ[p["key"]])
        extra = dict(nt.get("kwargs", {}))                 # opus-4-8: add effort="low" here if used
        r = client.messages.create(model=model, max_tokens=2048, temperature=temperature,
                                   system=system, messages=[{"role": "user", "content": user}], **extra)
        return r.content[0].text
    raise ValueError(provider)


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #
def parse_nextstep(text, steps, steps_upper):
    out = []
    for line in text.splitlines():
        line = line.strip()
        if ":" in line and line.upper().startswith("RANK"):
            out.append(nearest_vocab(line.split(":", 1)[1], steps, steps_upper))
        elif line and not line.startswith("#"):
            out.append(nearest_vocab(line, steps, steps_upper))
    return (out + [""] * 5)[:5]


def parse_completion(text, steps, steps_upper):
    return [nearest_vocab(l, steps, steps_upper) for l in
            (x.strip() for x in text.splitlines()) if l and ":" not in l]


def parse_anomaly(text):
    is_valid, conf, rule = 1, None, ""
    for line in text.splitlines():
        u = line.upper()
        if u.startswith("IS_VALID"):
            is_valid = 0 if "0" in line.split(":", 1)[1] else 1
        elif u.startswith("CONFIDENCE"):
            try:
                conf = float(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif u.startswith("RULE"):
            r = line.split(":", 1)[1].strip()
            rule = r if r in RULE_IDS else r
    score = conf if conf is not None else (0.9 if is_valid else 0.1)
    return is_valid, round(score, 4), (rule if is_valid == 0 else "")


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def run_task(task, provider, model, limit, out_dir, temperature, thinking=False):
    steps = load_steps()
    steps_upper = {s.upper(): s for s in steps}
    system = build_system(task, steps)
    in_file = {"nextstep": "eval_nextstep.csv", "completion": "eval_completion.csv",
               "anomaly": "eval_anomaly.csv"}[task]
    rows = list(csv.DictReader(open(ROOT / "data" / in_file, encoding="utf-8")))[:limit]
    out_dir.mkdir(parents=True, exist_ok=True)

    latencies, results, completions = [], [], []
    for row in rows:
        t0 = time.time()
        text = query(provider, model, system, build_user(task, row), temperature, task, thinking)
        latencies.append(time.time() - t0)
        if task == "nextstep":
            ranks = parse_nextstep(text, steps, steps_upper)
            results.append([row["EXAMPLE_ID"], *ranks])
        elif task == "completion":
            comp = parse_completion(text, steps, steps_upper)
            results.append([row["EXAMPLE_ID"], "|".join(comp)])
            completions.append(row["PARTIAL_SEQUENCE"].split("|") + comp)
        else:
            iv, sc, rule = parse_anomaly(text)
            results.append([row["EXAMPLE_ID"], iv, f"{sc:.4f}", rule])

    headers = {"nextstep": ["EXAMPLE_ID", "RANK_1", "RANK_2", "RANK_3", "RANK_4", "RANK_5"],
               "completion": ["EXAMPLE_ID", "PREDICTED_SEQUENCE"],
               "anomaly": ["EXAMPLE_ID", "IS_VALID", "SCORE", "PREDICTED_RULE"]}[task]
    outname = {"nextstep": "nextstep.csv", "completion": "completion.csv", "anomaly": "anomaly.csv"}[task]
    with open(out_dir / outname, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(headers); w.writerows(results)

    avg_ms = 1000 * sum(latencies) / max(1, len(latencies))
    print(f"[{task}] {provider}/{model}: {len(results)} examples, {avg_ms:.0f} ms/call avg -> {out_dir/outname}")
    if completions:
        vf = sum(1 for c in completions if not G.validate_sequence(c)) / len(completions)
        print(f"   completion validity (validate_sequence): {vf:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True, choices=list(PROVIDERS))
    ap.add_argument("--model", default=None)
    ap.add_argument("--task", default="all", choices=["nextstep", "completion", "anomaly", "all"])
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--thinking", action="store_true",
                    help="enable reasoning/thinking mode (default OFF for cost/speed)")
    args = ap.parse_args()

    model = args.model or PROVIDERS[args.provider]["model"]
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "extras" / "results" / f"baseline_{args.provider}"
    tasks = ["nextstep", "completion", "anomaly"] if args.task == "all" else [args.task]
    for t in tasks:
        run_task(t, args.provider, model, args.limit, out_dir, args.temperature, args.thinking)


if __name__ == "__main__":
    main()
