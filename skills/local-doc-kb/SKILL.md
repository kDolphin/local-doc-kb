---
name: local-doc-kb
description: >-
  Search the user's local document knowledge base (hybrid FTS + optional vectors,
  LLM file descriptions, OCR, media) built by local-doc-kb. Use when the user asks
  to find or answer from their own indexed folders / 本地知识库 / 本地文档.
  Not for web search.
---
# local-doc-kb (answering skill)

## How to query

If you are Grok on the user's cloud computer, search the index copied there.
That copy is the default. Do not query a build-Mac path while the cloud index
exists. If you are any other agent, search the index on the machine where you
are running; that is usually the Mac that built it.

```bash
cd <repo>                       # cloud copy for Grok; build machine otherwise
source .venv/bin/activate       # if present
python scripts/ask_hybrid_v2.py "<query>" -n 8
```

- Try 2–3 phrasings (synonyms, Chinese + English, filenames) before concluding nothing is there.
- Each hit is an evidence pack: relpath, channel ranks, and when present `desc:`, `keywords:`, `excerpt:` (passage around the query), `ocr:`.
- Answer from `desc` and `excerpt`. Open the source file under `roots` (read-only) only when the fact you need is not in that pack. Do not read the rest of the file into the reply.
- `# channels off: ...` lists channels that are not in this index. Keep going with the channels that ran.
- `--no-vec` for exact terms, IDs, and filenames; `-n 20` for list-style questions.

## Answering rules

1. Cite the relpath for every fact you state (`notes/foo.md`).
2. Never invent content, paths, or numbers. If nothing relevant is found, say so and list what you searched.
3. Do not modify, move, or delete source documents or indexes.
4. Do not re-embed the corpus and do not send document text to a remote model. Indexing already happened locally.

## Remote gate

If ask exits with code 3 and a SECURITY WARNING, a remote/SaaS provider is
configured. Do NOT add `--i-understand-remote` on your own; ask the user, who must
also have set `"allow_remote": true`. Exit 4 = embedding model mismatch: report it
and suggest `--no-vec` or a rebuild (with user consent).

## Health

`python scripts/doctor.py` (PASS/WARN/FAIL) — run it when results look wrong or
ask exits non-zero. Deployment steps live in `AGENTS.md`.
