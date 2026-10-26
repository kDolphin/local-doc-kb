# Template: noon incremental index refresh (generic)

Use this as an agent or cron prompt on the **build host**. Adjust paths to your install.

## Goal

Incremental refresh of the local document KB indexes for configured roots, then mirror sqlite to the query host. Do **not** wipe vector/OCR tables. Run `scripts/incremental_build.sh`; keep providers local unless the user opted into remote.

## Steps

1. Confirm `config.json` `roots[]` are mounted and readable.
2. Run incremental builders in order (once ported):
   - `build_fts.py` (incremental)
   - `build_vectors.py` (new/changed docs only)
   - `build_desc.py` (local LLM endpoint only)
   - `build_desc_vectors.py`
   - `build_media_vectors_eg2.py`
   - `build_ocr.py` (skip known-corrupt files per config; no `--rebuild` wipe)
3. Mirror `index/*.sqlite3` to the query host.
4. Smoke: `python scripts/ask_hybrid_v2.py 'smoke test keyword' -n 3`
5. Log counts (FTS rows, vector rows, OCR rows) to `logs/`.

## Every Nth workday (optional full supplement)

Same pipeline but scan for gaps; still **no** table wipe. Prefer additive backfill.

## Out of scope

- Writing into document roots
- Paying cloud embedding/LLM APIs without explicit user approval
- Pushing indexes or private docs to a public remote
