#!/usr/bin/env python3
"""Embed FTS docs into vectors.sqlite3 (local SentenceTransformers).

Chunking (v1): one snippet per relpath — cleaned text, first MAX_CHARS
characters, with relpath prepended. Embeddings are float32 LE BLOBs, L2-normalized.
Runs locally via providers.embedding.
"""
from __future__ import annotations

import argparse
import re
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_config import embedding_setting, index_path, load_config  # noqa: E402
from lib_embed import ensure_vectors_schema, pack_embedding  # noqa: E402
from lib_providers import (  # noqa: E402
    Embedder, add_remote_flag, check_index_meta, emb_conf, enforce_remote_gate, write_index_meta,
)

# Re-export for build_desc_vectors compatibility
ensure_schema = ensure_vectors_schema
DEFAULT_MODEL = "BAAI/bge-base-zh-v1.5"
MAX_CHARS = 800
COMMIT_EVERY = 50
TZ = ZoneInfo("Asia/Shanghai")


def clean_text(text: str) -> str:
    """Strip docx/XML dump noise from FTS text; collapse whitespace."""
    if not text:
        return ""
    head = text[:800]
    if head.lstrip().startswith("<?xml") or "<w:document" in head or "<w:p" in head[:200]:
        text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def snippet_for(relpath: str, text: str, max_chars: int) -> str:
    body = clean_text(text)[:max_chars]
    return f"{relpath}\n{body}".strip()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Embed FTS docs into vectors.sqlite3 (local GPU/CPU)"
    )
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--fts", type=Path, default=None)
    ap.add_argument("--db", type=Path, default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--limit", type=int, default=0, help="max new docs (0=all)")
    ap.add_argument("--max-chars", type=int, default=MAX_CHARS)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--commit-every", type=int, default=COMMIT_EVERY)
    add_remote_flag(ap)
    ap.add_argument("--device", default="auto", help="auto|cpu|mps|cuda")
    ap.add_argument(
        "--rebuild",
        action="store_true",
        help="drop existing rows for this model before embedding",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    from lib_providers import features as _features
    if not _features(cfg)["vectors"]:
        print("SKIP: features.vectors is false in config", flush=True)
        return 0
    fts_db = args.fts or index_path(cfg, "fts", "fts.sqlite3")
    vec_db = args.db or index_path(cfg, "vectors", "vectors.sqlite3")
    econf = emb_conf(cfg)
    if args.model:
        econf["model"] = args.model
    enforce_remote_gate(cfg, [econf], ack=args.i_understand_remote)

    if not fts_db.exists():
        print(f"FTS db missing: {fts_db}", file=sys.stderr)
        return 2

    vec_db.parent.mkdir(parents=True, exist_ok=True)

    print(f"loading embedding provider {econf['type']} model {econf['model']} …", flush=True)
    t0 = time.time()
    model = Embedder(econf, device=args.device)
    model_name = model.model_id
    dim = model.dim
    print(f"model ready id={model_name} dim={dim} provider={econf['type']} "
          f"load_s={time.time()-t0:.1f}", flush=True)

    fts = sqlite3.connect(f"file:{fts_db}?mode=ro", uri=True)
    vec = sqlite3.connect(vec_db)
    ensure_vectors_schema(vec)
    mm = check_index_meta(vec, model_name, dim)
    if mm and not args.rebuild:
        print(f"ERROR: embedding model mismatch: {mm}.\n  Switching embedding model "
              "requires --rebuild (or a separate --db). Refusing to mix vector spaces.",
              file=sys.stderr)
        return 4
    if args.rebuild:
        vec.execute("DELETE FROM vectors WHERE model<>?", (model_name,))
    write_index_meta(vec, model=model_name, dim=dim, provider=econf["type"])

    if args.rebuild:
        n = vec.execute("DELETE FROM vectors WHERE model=?", (model_name,)).rowcount
        vec.commit()
        print(f"rebuild: deleted {n} rows for model", flush=True)

    done = {
        r[0]
        for r in vec.execute("SELECT relpath FROM vectors WHERE model=?", (model_name,))
    }
    print(f"already embedded for this model: {len(done)}", flush=True)

    rows = fts.execute(
        """SELECT relpath, text, chars FROM docs
           WHERE chars > 0
           ORDER BY relpath"""
    ).fetchall()
    todo = [(rp, text, chars) for rp, text, chars in rows if rp not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"to embed: {len(todo)} (of {len(rows)} with_text)", flush=True)

    n_ok = n_fail = 0
    t_embed = time.time()
    pending_commit = 0
    for i in range(0, len(todo), args.batch_size):
        batch = todo[i : i + args.batch_size]
        texts = []
        metas = []
        for rp, text, chars in batch:
            snip = snippet_for(rp, text or "", args.max_chars)
            if len(snip) < 2:
                n_fail += 1
                continue
            texts.append(snip)
            metas.append((rp, chars if chars else len(snip)))
        if not texts:
            continue
        try:
            embs = model.encode(texts, batch_size=min(args.batch_size, len(texts)))
        except Exception as e:
            print(f"batch fail at {i}: {e}", file=sys.stderr, flush=True)
            n_fail += len(texts)
            continue
        now = datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S Asia/Shanghai")
        for (rp, chars), emb in zip(metas, embs):
            try:
                vec.execute(
                    """INSERT OR REPLACE INTO vectors
                       (relpath, model, dim, chars, embedded_at, embedding)
                       VALUES (?,?,?,?,?,?)""",
                    (rp, model_name, dim, int(chars), now, pack_embedding(emb)),
                )
                n_ok += 1
                pending_commit += 1
            except Exception as e:
                print(f"insert fail {rp}: {e}", file=sys.stderr, flush=True)
                n_fail += 1
        if pending_commit >= args.commit_every:
            vec.commit()
            pending_commit = 0
        if (i // args.batch_size) % 20 == 0 or i + args.batch_size >= len(todo):
            rate = n_ok / max(time.time() - t_embed, 0.01)
            print(
                f"progress embedded={n_ok} fail={n_fail} "
                f"rate={rate:.1f}/s elapsed={time.time()-t_embed:.0f}s",
                flush=True,
            )

    vec.commit()
    total = vec.execute(
        "SELECT COUNT(*) FROM vectors WHERE model=?", (model_name,)
    ).fetchone()[0]
    fts.close()
    vec.close()
    wall = time.time() - t0
    print(
        f"done ok={n_ok} fail={n_fail} skipped_already={len(done)} "
        f"total_for_model={total} dim={dim} model={model_name} "
        f"db={vec_db} wall_s={wall:.1f}",
        flush=True,
    )
    return 0 if n_ok > 0 or n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
