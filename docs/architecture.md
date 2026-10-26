# Architecture

## Roles

| Role | Responsibility |
|------|----------------|
| **Build host** | Reads configured document roots; runs FTS / vector / desc / media / OCR builders; holds canonical `index/*.sqlite3`. |
| **Query host** | Mirrors sqlite files; runs `ask_hybrid_v2.py` for Q&A. May be the same machine. |

Source documents stay **readonly** from the Q&A path. Never write answers or agent edits back into the indexed roots.

## Indexes

| File | Contents |
|------|----------|
| `fts.sqlite3` | `docs` + FTS5 over relpath/body |
| `vectors.sqlite3` | Body embeddings (default bge-base, dim 768) |
| `desc.sqlite3` | Local-LLM one-line Chinese/English descriptions + keywords |
| `desc_vectors.sqlite3` | Embeddings of those descriptions |
| `media_vectors_eg2.sqlite3` | EG2 vectors for images and scanned PDF pages |
| `ocr.sqlite3` | OCR text + FTS5 for digit/ID-style queries |

Relative paths (`relpath`) are stored under each configured root so citations stay portable.

## Query fusion

`ask_hybrid_v2` runs the channels that `features` enables, fused with **RRF** (K=60). FTS is always on. `config.example.json` enables body vectors and leaves descriptions, media, and OCR off. Flags: `--no-media`, `--no-ocr`, `--no-desc`, `--no-vec`, `--media`, `--ocr`, `--bge-small-all`, `--eg2-body`, `--eg2-all`.

Each printed hit is an evidence pack: relpath, channel ranks, the precomputed description and keywords when present, and `excerpt:` (a short window that covers the query, not a prefix of the file). If the body excerpt misses the query and OCR is on, an `ocr:` line is added. `--json` emits the same fields as one object. How the answering agent uses the pack is in `skills/local-doc-kb/SKILL.md`.

## Config

`config.json` (from `config.example.json`) sets:

- `roots[]` — directories to index (指定目录; not locked to any cloud product)
- `denylist_path_prefixes[]`
- `index.*` — sqlite paths
- `embeddings.*` — model names
- `features` — which channels build and which channels ask runs
- `build.desc_lang` — `doc`, `zh`, or `en` for file descriptions

## Privacy

- Default embeddings and the file-description LLM run on the build host. A remote provider is refused unless `allow_remote` is true and the command passes `--i-understand-remote`.
- Query host needs model weights for vector/media channels (local HF cache or mirrored weights).
- Do not commit `index/`, `raw/`, or real `config.json` with private roots.
