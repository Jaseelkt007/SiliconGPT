# SiliconGPT Backend (Flask)

Local inference server wrapping the trained `ProcessLM` checkpoint for the
Lovable frontend (`/Users/unais/silicongpt-intelligence`).

## Quick start

```bash
# from repo root
pip install -r server/requirements.txt

# point at a trained checkpoint (download from Leonardo first)
export CHECKPOINT_PATH=/absolute/path/to/best.pt
# optional: pin a device
export DEVICE=cpu          # or cuda / mps
# optional: a compact CSV of known-valid sequences (used to calibrate
# the anomaly NLL threshold once at startup)
export CALIB_PATH=$(pwd)/data/val_id.csv

python server/app.py       # serves on http://0.0.0.0:5050
```

If the checkpoint isn't found the server still starts; inference routes return
`503 {error, ckpt_path}` so the frontend can show a clear banner. `/api/health`,
`/api/rules`, and `/api/validate` (rule-only, deterministic) keep working
without a model.

## Endpoints

| Method | Path | Body |
|---|---|---|
| GET  | `/api/health` | — |
| GET  | `/api/vocab` | — |
| GET  | `/api/rules` | — |
| POST | `/api/predict/nextstep` | `{partial_sequence: string[]\|string, k?: number}` |
| POST | `/api/predict/complete` | `{partial_sequence, max_new?, greedy?, temperature?}` |
| POST | `/api/generate` | `{prefix?: string[], max_new?, temperature?}` |
| POST | `/api/validate` | `{sequence: string[]\|string}` |
| POST | `/api/anomaly` | `{sequence, use_validator?}` |
| POST | `/api/eval/nextstep` | multipart `file=<csv>` (cols: `EXAMPLE_ID,FAMILY,PARTIAL_SEQUENCE[,TRUE_NEXT_STEP]`) |
| POST | `/api/eval/completion` | multipart CSV (`…,PARTIAL_SEQUENCE[,TRUE_SUFFIX]`) |
| POST | `/api/eval/anomaly` | multipart CSV (`…,SEQUENCE[,IS_VALID,RULE_VIOLATED]`) |
| POST | `/api/eval/ood` | multipart `file=<csv>`, form `task=nextstep\|completion\|anomaly` |

Sequences are arrays of step strings (or a single `|`-joined string for
convenience). Step strings must match tokens in `vocab.json` (or they map to
`<UNK>` and the model will struggle).

## Sanity check

```bash
curl -s http://localhost:5050/api/health | jq
curl -s -X POST http://localhost:5050/api/validate \
     -H 'Content-Type: application/json' \
     -d '{"sequence":["RECEIVE WAFER","RCA CLEAN","DEPOSIT OXIDE"]}' | jq
```
