#!/usr/bin/env python3
"""Embed descs rows into desc_vectors.sqlite3 (same schema as body vectors).

Text format: "{relpath}\\n{desc}\\n{keywords}". Model from config embeddings.body_model
(default BAAI/bge-base-zh-v1.5). Resume-safe. Local only.
"""
from __future__ import annotations

import argparse
import json
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

DEFAULT_MODEL = "BAAI/bge-base-zh-v1.5"
COMMIT_EVERY = 50
TZ = ZoneInfo("Asia/Shanghai")


def snip(relpath: str, desc: str, keywords: str) -> str:
    return f"{relpath}\n{desc or ''}\n{keywords}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Embed file descriptions into desc_vectors.sqlite3")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--desc", type=Path, default=None)
    ap.add_argument("--db", type=Path, default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--commit-every", type=int, default=COMMIT_EVERY)
    add_remote_flag(ap)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    from lib_providers import features as _features
    if not _features(cfg)["desc"]:
        print("SKIP: features.desc is false in config", flush=True)
        return 0
    desc_db = args.desc or index_path(cfg, "desc", "desc.sqlite3")
    vec_db = args.db or index_path(cfg, "desc_vectors", "desc_vectors.sqlite3")
    econf = emb_conf(cfg)
    if args.model:
        econf["model"] = args.model
    enforce_remote_gate(cfg, [econf], ack=args.i_understand_remote)

    if not desc_db.exists():
        print(f"desc db missing: {desc_db}", file=sys.stderr)
        return 2
    vec_db.parent.mkdir(parents=True, exist_ok=True)

    print(f"loading embedding provider {econf['type']} model {econf['model']} …", flush=True)
    t0 = time.time()
    model = Embedder(econf, device=args.device)
    model_name = model.model_id
    dim = model.dim
    print(f"model ready id={model_name} dim={dim} load_s={time.time()-t0:.1f}", flush=True)

    src = sqlite3.connect(f"file:{desc_db}?mode=ro", uri=True)
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

    raw = src.execute("SELECT relpath, desc, keywords_json FROM descs ORDER BY relpath").fetchall()
    todo = []
    for rp, d, kj in raw:
        if rp in done:
            continue
        try:
            kw = " ".join(json.loads(kj or "[]"))
        except Exception:
            kw = kj or ""
        s = snip(rp, d or "", kw)
        if len(s.strip()) < 2:
            continue
        todo.append((rp, s, len(s)))
    if args.limit:
        todo = todo[: args.limit]
    print(f"to embed: {len(todo)} (of {len(raw)} descs)", flush=True)

    n_ok = n_fail = 0
    t_embed = time.time()
    pending = 0
    for i in range(0, len(todo), args.batch_size):
        batch = todo[i : i + args.batch_size]
        texts = [t for _, t, _ in batch]
        metas = [(rp, chars) for rp, _, chars in batch]
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
                pending += 1
            except Exception as e:
                print(f"insert fail {rp}: {e}", file=sys.stderr, flush=True)
                n_fail += 1
        if pending >= args.commit_every:
            vec.commit()
            pending = 0
        if (i // args.batch_size) % 20 == 0 or i + args.batch_size >= len(todo):
            rate = n_ok / max(time.time() - t_embed, 0.01)
            left = len(todo) - n_ok - n_fail
            eta = left / max(rate, 0.01)
            print(
                f"progress embedded={n_ok}/{len(todo)} fail={n_fail} "
                f"rate={rate:.1f}/s eta={eta:.0f}s elapsed={time.time()-t_embed:.0f}s",
                flush=True,
            )

    vec.commit()
    total = vec.execute(
        "SELECT COUNT(*) FROM vectors WHERE model=?", (model_name,)
    ).fetchone()[0]
    src.close()
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
