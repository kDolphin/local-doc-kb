# Scripts

## Query path

| Script | Notes |
|--------|-------|
| `ask_hybrid_v2.py` | Config-driven paths; RRF over the channels `features` enables; prints an evidence pack (description + excerpt around the query) |
| `ask_hybrid.py` | Thin wrapper → v2 |
| `lib_config.py` | `load_config` / `index_path` / roots / denylist |
| `lib_embed.py` | pack embedding BLOB, vectors schema, device pick |

## Build / sync

Deploy steps are in `AGENTS.md`. The build command that runbook calls is `scripts/incremental_build.sh`. It follows `features` and does not wipe indexes. Builders also exit 0 with `SKIP:` when their feature is off, so a direct call does not start a model the user did not enable.

All builders read `config.json` (or `--config` / `$LOCAL_DOC_KB_CONFIG`). Paths, roots, denylist, model names, and index filenames come from config.

| Script | Role |
|--------|------|
| `build_fts.py` | Extract text from `roots[]` → `index/fts.sqlite3` |
| `build_vectors.py` | Body bge-base → `index/vectors.sqlite3` |
| `build_desc.py` | Local LLM one-liners → `index/desc.sqlite3` |
| `build_desc_vectors.py` | Desc embeddings → `index/desc_vectors.sqlite3` |
| `build_media_vectors_eg2.py` | EG2 media → `index/media_vectors_eg2.sqlite3` |
| `build_ocr.py` | Vision CLI / Tesseract → `index/ocr.sqlite3` |
| `llm_lifecycle.sh <provider> start\|stop\|status` | LM Studio / Ollama / openai_compatible lifecycle (`lmstudio_desc_lifecycle.sh` = back-compat wrapper) |
| `lib_providers.py` | LLM + embedding providers, remote gate, index_meta |
| `doctor.py` | PASS/WARN/FAIL environment + index checks |
| `bootstrap.sh` | arm64 macOS venv + tiered deps |
| `incremental_build.sh` | all enabled build steps, in order, no wipes |
| `eval_queries.py` | hit@k / MRR from queries.jsonl |
| `smoke_test.sh` | offline fixture E2E (`--with-mock`, `--with-vectors`) |
| `sync_mirror.sh` | rsync `*.sqlite3` build → query (`SYNC_SRC`/`SYNC_DEST`/`BUILD_HOST` or `hosts.*`) |
| `restore_query_media_env.sh` | pillow + torchvision + EG2 cache checks |

Optional EG2 **text** vector builders (`build_vectors_eg2.py`, etc.) are out of scope for this scaffold.

## Mac-only / local-tool notes

- `.doc` extract via `textutil` (macOS); antiword/libreoffice fallback if present.
- OCR: `tools/macos-vision-ocr/build.sh` builds the Vision helper. If that binary is missing, tesseract is used.
- EG2 media + bge embeddings use MPS; CPU works but is slow.
- LM Studio (`lms`) for `build_desc.py` — model id from config / `$LMSTUDIO_DESC_MODEL`.
