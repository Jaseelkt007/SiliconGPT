# Lovable Prompt — Wire SiliconGPT Process Lab to the real backend

Paste the contents of the fenced block below into Lovable. Everything outside
the block is for you (the human) — context, not for the model.

---

## Context (read before pasting)

- The backend already exists locally and runs at **`http://localhost:5050`**.
- We currently render mock predictions in `ProcessLab.tsx` (the `predictTop5`
  function uses a seeded RNG). We will replace the mock paths with real HTTP
  calls to the backend.
- The backend's step vocabulary uses **spaces** in token strings
  (e.g. `RECEIVE WAFER`, `RCA CLEAN`, `DEPOSIT OXIDE`, `EXPOSE LITHO LEVEL 1`).
  The current demo dataset uses underscored fake tokens like `RECEIVE_WAFER`.
  Demo data must be updated to real tokens or the model will treat every step
  as `<UNK>`.
- Five existing task tabs (`predict`, `complete`, `validate`, `anomaly`, `ood`)
  must each map to a real endpoint, **plus** add a new sixth tab `batch` for
  full CSV evaluation with metrics.

---

## Prompt to paste into Lovable

```
You're updating the SiliconGPT Process Intelligence Lab (TanStack Start +
React 19 + Tailwind 4 + framer-motion). Right now `ProcessLab.tsx` shows
fake predictions from a seeded RNG. Wire it to a real Python/Flask backend
that runs at http://localhost:5050. Keep the existing visual design — only
change data sources and add the new pieces listed at the bottom.

============================================================================
1) Add an env-driven API client
============================================================================

Create `src/lib/api.ts` with this exact content:

----------------------------------------------------------------------------
const BASE = (import.meta as any).env?.VITE_BACKEND_URL ?? "http://localhost:5050";

async function jpost<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${path} ${r.status}: ${await r.text()}`);
  return r.json() as Promise<T>;
}

async function jget<T>(path: string): Promise<T> {
  const r = await fetch(`${BASE}${path}`);
  if (!r.ok) throw new Error(`${path} ${r.status}`);
  return r.json() as Promise<T>;
}

async function fpost<T>(path: string, form: FormData): Promise<T> {
  const r = await fetch(`${BASE}${path}`, { method: "POST", body: form });
  if (!r.ok) throw new Error(`${path} ${r.status}: ${await r.text()}`);
  return r.json() as Promise<T>;
}

export type Health = {
  ok: boolean;
  ckpt_path: string | null;
  device: string;
  vocab_size: number;
  threshold: number | null;
  load_error: string | null;
  families: string[];
};

export type Prediction = { token: string; prob: number };

export const api = {
  health: () => jget<Health>("/api/health"),
  vocab:  () => jget<{ tokens: string[] }>("/api/vocab"),
  rules:  () => jget<{ rules: { id: string; description: string }[] }>("/api/rules"),

  predictNextStep: (partial: string[], k = 5) =>
    jpost<{ predictions: Prediction[]; latency_ms: number }>(
      "/api/predict/nextstep", { partial_sequence: partial, k }),

  complete: (partial: string[], opts: { max_new?: number; greedy?: boolean; temperature?: number } = {}) =>
    jpost<{ prefix: string[]; generated: string[]; full: string[]; latency_ms: number }>(
      "/api/predict/complete",
      { partial_sequence: partial, max_new: opts.max_new ?? 220,
        greedy: opts.greedy ?? true, temperature: opts.temperature ?? 1.0 }),

  generate: (opts: { prefix?: string[]; max_new?: number; temperature?: number } = {}) =>
    jpost<{ prefix: string[]; generated: string[]; full: string[];
            is_valid: number; violations: Violation[]; latency_ms: number }>(
      "/api/generate", { prefix: opts.prefix ?? [], max_new: opts.max_new ?? 220,
                          temperature: opts.temperature ?? 1.0 }),

  validate: (sequence: string[]) =>
    jpost<{ is_valid: number; violations: Violation[]; n_steps: number }>(
      "/api/validate", { sequence }),

  anomaly: (sequence: string[], use_validator = true) =>
    jpost<AnomalyResult>("/api/anomaly", { sequence, use_validator }),

  evalNextStep:   (file: File) => { const f = new FormData(); f.append("file", file);
    return fpost<NextStepEval>("/api/eval/nextstep", f); },
  evalCompletion: (file: File) => { const f = new FormData(); f.append("file", file);
    return fpost<CompletionEval>("/api/eval/completion", f); },
  evalAnomaly:    (file: File) => { const f = new FormData(); f.append("file", file);
    return fpost<AnomalyEval>("/api/eval/anomaly", f); },
  evalOOD:        (file: File, task: "nextstep" | "completion" | "anomaly") => {
    const f = new FormData(); f.append("file", file); f.append("task", task);
    return fpost<OODEval>("/api/eval/ood", f); },
};

export type Violation = {
  rule: string; description: string; step_index: number; step_name: string;
};

export type AnomalyResult = {
  is_valid: number; score: number; nll: number; threshold: number | null;
  predicted_rule: string; violations: Violation[];
  lm_only: { is_valid: number; score: number };
};

export type NextStepMetrics  = Record<string, { top1:number; top3:number; top5:number; mrr:number; n:number }>;
export type CompletionMetrics = Record<string, { exact_match:number; norm_edit_dist:number; token_acc:number; n:number }>;
export type AnomalyMetrics   = Record<string, { acc:number; precision:number; recall:number; f1:number; roc_auc:number; rule_attr:number; confusion:{tp:number;fp:number;fn:number;tn:number}; n:number }>;

export type NextStepEval = {
  rows: { example_id: string; family: string; partial_sequence: string[];
          predictions: Prediction[]; true_next_step?: string }[];
  metrics: NextStepMetrics | null; n: number;
};
export type CompletionEval = {
  rows: { example_id: string; family: string; partial_sequence: string[];
          predicted: string[]; true_suffix: string[] | null;
          completion_fraction?: string }[];
  metrics: CompletionMetrics | null; n: number;
};
export type AnomalyEval = {
  rows: { example_id: string; family: string; sequence: string[];
          is_valid: number; score: number; predicted_rule: string; nll: number;
          true_is_valid: number | null; true_rule?: string }[];
  metrics: AnomalyMetrics | null; n: number;
};
export type OODEval = {
  task: string;
  metrics: NextStepMetrics | CompletionMetrics | AnomalyMetrics;
  family_counts: Record<string, number>; n: number;
};
----------------------------------------------------------------------------

Also add to the project root a new file `.env.local` with:

    VITE_BACKEND_URL=http://localhost:5050

============================================================================
2) Replace mock predictions in ProcessLab.tsx
============================================================================

In `src/components/dashboard/ProcessLab.tsx`:

(a) DELETE the `seededRand`, `predictTop5`, and `ALL_STEPS` constants. The
    model is the source of truth now; no more client-side heuristics.

(b) Replace the **DEMO datasets** with the real vocab tokens used by the
    backend. The backend's vocab uses SPACES not underscores. Use these
    representative recipes (these are real tokens from the trained vocab):

      MOSFET demo:
        ["RECEIVE WAFER", "RCA CLEAN", "GROW THERMAL OXIDE", "DEPOSIT POLYSILICON",
         "PHOTORESIST COAT", "ALIGN MASK LEVEL 1", "EXPOSE LITHO LEVEL 1",
         "DEVELOP PHOTORESIST", "ETCH POLYSILICON", "STRIP PHOTORESIST",
         "IMPLANT N+", "ANNEAL DOPANTS", "DEPOSIT ILD", "CMP ILD",
         "PHOTORESIST COAT", "ALIGN MASK LEVEL 2", "EXPOSE LITHO LEVEL 2",
         "DEVELOP PHOTORESIST", "ETCH CONTACT", "DEPOSIT METAL 1",
         "ETCH METAL 1", "DEPOSIT PASSIVATION", "ELECTRICAL TEST"]

      IGBT demo:
        ["RECEIVE WAFER", "RCA CLEAN", "DEPOSIT EPI LAYER", "GROW FIELD OXIDE",
         "PHOTORESIST COAT", "ALIGN MASK LEVEL 1", "EXPOSE LITHO LEVEL 1",
         "DEVELOP PHOTORESIST", "IMPLANT P-BASE", "DRIVE IN", "GROW GATE OXIDE",
         "DEPOSIT POLYSILICON", "ETCH POLYSILICON", "DEPOSIT METAL 1",
         "BACKSIDE GRIND", "DEPOSIT BACKSIDE METAL", "DEPOSIT PASSIVATION",
         "ELECTRICAL TEST"]

      IC demo:
        ["RECEIVE WAFER", "RCA CLEAN", "ETCH STI", "DEPOSIT STI OXIDE", "CMP OXIDE",
         "IMPLANT WELL", "GROW GATE OXIDE", "DEPOSIT POLYSILICON",
         "PHOTORESIST COAT", "ALIGN MASK LEVEL 1", "EXPOSE LITHO LEVEL 1",
         "DEVELOP PHOTORESIST", "ETCH POLYSILICON", "IMPLANT LDD",
         "DEPOSIT SPACER", "IMPLANT S/D", "ANNEAL DOPANTS", "DEPOSIT ILD",
         "ETCH CONTACT", "DEPOSIT METAL 1", "ETCH METAL 1",
         "DEPOSIT PASSIVATION", "ELECTRICAL TEST"]

      (Keep the OOD demo entry as-is; it's only used in the OOD info card.)

(c) PredictTab: replace the `useEffect` that calls `predictTop5(...)` with:

      useEffect(() => {
        setRunning(true); setResults([]);
        api.predictNextStep(prefix, 5)
          .then(r => setResults(r.predictions))
          .catch(e => setError(String(e)))
          .finally(() => setRunning(false));
      }, [cursor, steps.join("|")]);

    Drop the artificial 420 ms `setTimeout`. Keep the same render — it already
    uses `{token, prob}` shape.

(d) CompleteTab: replace the local autoregressive `tick()` loop with one
    backend call:

      const r = await api.complete(prefix, { greedy: true });
      setOut(r.generated);

    Keep the existing animation by staggering reveal client-side: after the
    response arrives, push tokens into `out` one-by-one every ~180 ms so the
    "streaming" feel is preserved.

(e) ValidateTab: REMOVE the hardcoded `VAL_RULES` heuristics. Call:

      api.validate(steps).then(r => setResults(r))

    Render `r.violations[]` (each has `rule`, `description`, `step_index`,
    `step_name`). Show all 10 rule IDs in the side panel by first fetching
    `api.rules()` once at component mount and rendering pass/fail per rule
    based on whether that rule id appears in `r.violations`.

(f) AnomalyTab: REPLACE the per-step heatmap heuristics with a real call:

      api.anomaly(steps, true).then(r => setAnomaly(r));

    The backend returns ONE anomaly verdict per sequence (not per-step). Change
    the layout:
      - Big verdict card: `r.is_valid ? "NORMAL" : "ANOMALY"`,
        `(r.score*100).toFixed(0)%` confidence
      - Numeric stats: NLL = `r.nll.toFixed(3)`, threshold = `r.threshold`,
        LM-only verdict = `r.lm_only.is_valid`
      - Violation list: render `r.violations[]` exactly like ValidateTab does,
        and highlight `step_index` in the sequence chips.

(g) OODTab: replace the hardcoded `perFamily` numbers with a small uploader:

      <input type="file" accept=".csv" onChange={e => {
        const f = e.target.files?.[0]; if (!f) return;
        setLoading(true);
        api.evalOOD(f, task).then(setOod).finally(() => setLoading(false));
      }} />
      <select value={task} onChange={e => setTask(e.target.value as any)}>
        <option value="nextstep">Next-step</option>
        <option value="completion">Completion</option>
        <option value="anomaly">Anomaly</option>
      </select>

    Render a per-family metric table from `oodResult.metrics` (object keyed
    by family name, with "ALL" row pinned at top). Columns depend on the task:
      nextstep:   top1, top3, top5, mrr
      completion: exact_match, norm_edit_dist, token_acc
      anomaly:    acc, precision, recall, f1, roc_auc, rule_attr

============================================================================
3) Add ONE new option in the Import panel: "Generate random"
============================================================================

Add a fourth mode tab to the Import panel next to demo/paste/upload, called
"Random". When selected:

  - Show two controls:
      • temperature slider (0.4 to 1.4, default 0.9)
      • optional "starting prefix" textarea (same parser as Paste)
  - A button "▶ Sample New Recipe" calls:
      api.generate({ prefix, temperature })
  - On success: set the active dataset to
      { id: "RANDOM", family: "Model Sample",
        node: `T=${temperature.toFixed(2)}`,
        description: `${r.full.length}-step random sample · ${r.is_valid ? "VALID" : "INVALID"}` ,
        steps: r.full }
    and setSteps(r.full).

============================================================================
4) Add a NEW tab "BATCH EVAL" (sixth tab in TabBar)
============================================================================

Insert a new TabId `"batch"` with code `T6`, label `"Batch Eval (CSV)"`.
Body:

  - Three sub-modes selected by a segmented control: Next-step / Completion / Anomaly
  - File input (.csv) — when a file is picked, call the matching endpoint
      (api.evalNextStep / api.evalCompletion / api.evalAnomaly)
  - While loading: show a progress bar + the current row count
  - After completion: render in this order
      (i)   A METRICS card (per-family table, "ALL" row first) — same column
            layout as the OOD tab above.
      (ii)  A latency stat (total + per-row avg using `Date.now()` deltas)
      (iii) An expandable "Per-row results" table (paginated 50 rows):
              - Next-step: EXAMPLE_ID | FAMILY | predictions (top-5 chips,
                  highlight the one matching TRUE_NEXT_STEP if present)
              - Completion: EXAMPLE_ID | FAMILY | predicted (joined w/ ›) |
                  true_suffix (if present) | NED (computed client-side)
              - Anomaly: EXAMPLE_ID | FAMILY | is_valid (✓/✗) | score |
                  predicted_rule | true_is_valid (if present)
  - Also expose a "Download predictions.csv" button that converts `rows`
    into the same format `predict.py` writes (so users can run the
    organizer's official scorer offline).

CSV column expectations (show in a small help tooltip):
  Next-step   : EXAMPLE_ID, FAMILY, PARTIAL_SEQUENCE        (+ optional TRUE_NEXT_STEP)
  Completion  : EXAMPLE_ID, FAMILY, PARTIAL_SEQUENCE        (+ optional TRUE_SUFFIX, COMPLETION_FRACTION)
  Anomaly     : EXAMPLE_ID, FAMILY, SEQUENCE                (+ optional IS_VALID, RULE_VIOLATED)

`PARTIAL_SEQUENCE` / `SEQUENCE` cells are `|`-separated step tokens.

============================================================================
5) Top-bar health banner
============================================================================

In `ProcessLab.tsx`'s header strip (the one that currently shows
"§ Lab · Primary Workstation"), add a health pill that polls `api.health()`
on mount. Render:
  - green dot + "READY · {h.device} · vocab {h.vocab_size}" when h.ok
  - amber dot + "NO CHECKPOINT — {h.load_error}" when !h.ok
Use the existing StatusDot component (color "success" / "warning").

Replace the hardcoded checkpoint card text ("sgpt-v041-ep142") with the
basename of `h.ckpt_path`.

============================================================================
6) Misc
============================================================================

- Add `vite` env type declaration if it doesn't already exist:
    src/vite-env.d.ts  -> /// <reference types="vite/client" />
- All new buttons should follow the existing utility classes
  (`font-mono text-xs uppercase tracking-widest`, etc).
- No new dependencies needed — `fetch` and `FormData` are built in.

Do not touch anything outside `src/components/dashboard/ProcessLab.tsx`,
`src/lib/api.ts` (new), `.env.local` (new), and the small env.d.ts
addition. Do not change other dashboard pages.
```

---

## After Lovable finishes — local checklist (for the human)

1. **Backend running?**
   ```bash
   cd /Users/unais/SiliconGPT
   pip install -r server/requirements.txt
   export CHECKPOINT_PATH=/abs/path/to/best.pt
   python server/app.py
   ```
2. **Frontend running?**
   ```bash
   cd /Users/unais/silicongpt-intelligence
   bun install && bun dev   # or pnpm/npm
   ```
3. Open the lab page (`/lab`). Top-bar pill should be green; if amber, the
   message tells you exactly what to fix (usually `CHECKPOINT_PATH`).
4. Try each tab. The "Random" import mode and the "Batch Eval" tab are new.
