# AGENTS.md

## Cursor Cloud specific instructions

SiliconGPT is a Python 3.12 ML project (a small from-scratch transformer for semiconductor
process recipes). The only long-running service is the Flask inference backend; everything else is
batch scripts. There is no database and no external API is needed for inference. The frontend is in
a separate repo (not in this workspace), so end-to-end testing here is done against the backend API.

Standard commands live in `README.md` ("Run it from a clean checkout", "Tests") and `server/app.py`
(endpoint list). Notes below are the non-obvious bits.

### Python environment
- Dependencies are installed into a virtualenv at `.venv` (created by the startup update script).
  Activate it before running anything: `source .venv/bin/activate`.
- This matters because the repo scripts (`run_demo.sh`, `predict.py`, etc.) invoke `python`, and the
  bare system interpreter is only available as `python3` — the venv puts `python` on `PATH`.
- Creating the venv requires the system package `python3.12-venv` (installed during environment
  setup; persisted in the snapshot). It is not reinstalled by the update script.
- Torch is the CPU-usable CUDA wheel; `torch.cuda.is_available()` is `False` here. Use `--device cpu`
  / `DEVICE=cpu` for all inference, tests, and the smoke train.

### Data must be regenerated (not in the update script)
- `data/` is gitignored and deterministically regenerable. It is NOT recreated by the update script.
- Run `python scripts/build_datasets.py --seed 42` (~10s) once per fresh checkout before anything
  that reads `data/`: `scripts/run_demo.sh`, `predict.py`/`score.py`, and the backend's
  `/api/sample` and `/api/eval/builtin` endpoints.

### GOTCHA: smoke train overwrites the committed deliverable checkpoint
- `python src/process_logic/train.py --smoke` writes a tiny 0.11M model to `checkpoints/best.pt`,
  clobbering the committed 5.2MB deliverable, and also touches `extras/results/train_log.csv`.
- After running any smoke/train that targets the default `checkpoints/` dir, restore them:
  `git checkout -- checkpoints/best.pt extras/results/train_log.csv`.

### Running things
- Backend: `DEVICE=cpu python server/app.py` serves on port `5050` (loads `checkpoints/best.pt`).
  Confirm readiness via `GET /api/health` (`"ok": true`, `"load_error": null`).
- Demo pipeline (predict -> score on a tiny subset): `bash scripts/run_demo.sh`.
- Tests are plain scripts (no pytest): run each `python tests/test_<name>.py`. `test_model`,
  `test_generate`, `test_health`, `test_emb_init` need torch; the rest do not.
- No lint tooling is configured in this repo.
