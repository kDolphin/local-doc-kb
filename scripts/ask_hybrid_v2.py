#!/usr/bin/env python3
"""Hybrid query: keyword + optional vectors, descriptions, media, and OCR, fused by RRF.

Paths come from `config.json` (or `<repo>/index/`). Channels follow `features`:
FTS is always on; vectors, desc, media, and OCR run only when that feature is on
and its index exists. `config.example.json` enables vectors only.

Each hit is an evidence pack for the answering agent: relpath, the precomputed
one-line description when one exists, and a short excerpt around the query
(not the start of the file). `--json` prints one object on stdout.

Chinese queries rarely match FTS5's default tokenizer as a whole phrase, so the
keyword channel falls back to LIKE on the full query and then to bigram hits on
relpath (filename/folder), which is where many table-type docs are findable.

Default embedding model is `providers.embedding` (example: BAAI/bge-base-zh-v1.5).
"""

from __future__ import annotations
import argparse, json, re, sqlite3, struct, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_config import PROJECT_ROOT as ROOT, index_path as _index_path, load_config as _load_config  # noqa: E402
from lib_providers import (  # noqa: E402
    Embedder, add_remote_flag, check_index_meta, emb_conf, enforce_remote_gate, features as _features,
)

_CFG = _load_config()
_EMB = (_CFG.get("embeddings") or {})
FTS_DB = _index_path(_CFG, "fts", "fts.sqlite3")
VEC_DB = _index_path(_CFG, "vectors", "vectors.sqlite3")
VEC_EG2_DB = _index_path(_CFG, "vectors_eg2", "vectors_eg2.sqlite3")
VEC_BGE_BASE_DB = _index_path(_CFG, "vectors_bge_base", "vectors_bge_base.sqlite3")
VEC_BGE_SMALL_DB = _index_path(_CFG, "vectors_bge_small", "vectors_bge_small.sqlite3")
DESC_DB = _index_path(_CFG, "desc", "desc.sqlite3")
DESC_VEC_DB = _index_path(_CFG, "desc_vectors", "desc_vectors.sqlite3")
DESC_VEC_EG2_DB = _index_path(_CFG, "desc_vectors_eg2", "desc_vectors_eg2.sqlite3")
DESC_VEC_BGE_BASE_DB = _index_path(_CFG, "desc_vectors_bge_base", "desc_vectors_bge_base.sqlite3")
DESC_VEC_BGE_SMALL_DB = _index_path(_CFG, "desc_vectors_bge_small", "desc_vectors_bge_small.sqlite3")
MEDIA_VEC_DB = _index_path(_CFG, "media_vectors_eg2", "media_vectors_eg2.sqlite3")
OCR_DB = _index_path(_CFG, "ocr", "ocr.sqlite3")
DEFAULT_MODEL = _EMB.get("body_model") or _EMB.get("model") or "BAAI/bge-base-zh-v1.5"
_PROV_EMB = ((_CFG.get("providers") or {}).get("embedding") or {}).get("model") or ""
BGE_BASE_MODEL = _EMB.get("bge_base_model") or _PROV_EMB or "BAAI/bge-base-zh-v1.5"
BGE_SMALL_MODEL = _EMB.get("bge_small_model") or "BAAI/bge-small-zh-v1.5"
EG2_MODEL = _EMB.get("eg2_model") or "google/embeddinggemma-2"
K = 60  # RRF constant
# tuned 2026-10-22 (see eval_v2.sh); override via flags
DESC_WEIGHT = 1.0
DESC_TOPK = 40
DESC_BIGRAM = 1
DESC_BIGRAM_WEIGHT = 1.0
MEDIA_WEIGHT = 1.0
MEDIA_TOPK = 40
OCR_WEIGHT = 1.0
OCR_TOPK = 40
EXCERPT_CHARS = 420
STOP = {"清单", "列表", "文件", "文档", "资料", "表格", "相关", "哪些", "什么", "一下"}


def query_terms(q: str) -> list[str]:
    """Search terms used to center an excerpt. Longer terms first, duplicates dropped."""
    found: list[str] = []
    for t in re.findall(r"[\u4e00-\u9fff]+|[A-Za-z0-9]+", q or ""):
        if re.match(r"[\u4e00-\u9fff]", t):
            core = t
            for s in STOP:
                core = core.replace(s, "")
            if len(core) >= 2:
                found.append(core)
            found.extend(core[i:i + 2] for i in range(max(len(core) - 1, 0)))
        elif len(t) >= 3:
            found.append(t)
    seen: set[str] = set()
    out: list[str] = []
    for t in sorted(found, key=len, reverse=True):
        key = t.lower()
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out


def best_excerpt(text: str, query: str, limit: int = EXCERPT_CHARS) -> str:
    """Passage that covers the most query terms, capped so the agent need not open the file."""
    flat = re.sub(r"\s+", " ", text or "").strip()
    if not flat:
        return ""
    if len(flat) <= limit:
        return flat
    low = flat.lower()
    terms = query_terms(query)
    locs: list[int] = []
    for t in terms:
        tl = t.lower()
        start = 0
        while True:
            i = low.find(tl, start)
            if i < 0:
                break
            locs.append(i)
            start = i + max(len(tl), 1)
            if len(locs) > 80:
                break
    if not locs:
        snip = flat[:limit].strip()
        return snip + "…"
    distinct = list(dict.fromkeys(t.lower() for t in terms))
    best_start = 0
    best_score = (-1, 0)
    for start in sorted({max(0, i - 48) for i in locs}):
        window = low[start:start + limit]
        covered = sum(1 for t in distinct if t in window)
        score = (covered, -start)
        if score > best_score:
            best_score = score
            best_start = start
    if best_start > 0:
        sp = flat.find(" ", best_start, min(len(flat), best_start + 40))
        if sp != -1:
            best_start = sp + 1
    end = min(len(flat), best_start + limit)
    if end < len(flat):
        sp = flat.rfind(" ", best_start, end)
        if sp > best_start + limit // 2:
            end = sp
    snip = flat[best_start:end].strip()
    if best_start > 0:
        snip = "…" + snip
    if end < len(flat):
        snip = snip + "…"
    return snip


def excerpt_covers(text: str, query: str) -> bool:
    low = (text or "").lower()
    terms = query_terms(query)
    if not terms:
        return bool(low)
    return any(t.lower() in low for t in terms)


def load_desc_rows(desc_db, relpaths: list[str]) -> dict[str, tuple[str, list]]:
    if not relpaths or desc_db is None or not Path(desc_db).exists():
        return {}
    conn = sqlite3.connect(f"file:{desc_db}?mode=ro", uri=True)
    qmarks = ",".join("?" * len(relpaths))
    out: dict[str, tuple[str, list]] = {}
    for rp, desc, kw in conn.execute(
        f"SELECT relpath, desc, keywords_json FROM descs WHERE relpath IN ({qmarks})",
        relpaths,
    ):
        try:
            words = json.loads(kw or "[]")
        except json.JSONDecodeError:
            words = []
        if not isinstance(words, list):
            words = []
        out[rp] = (desc or "", [str(w) for w in words if w])
    conn.close()
    return out


def ocr_blob(ocr_db, relpath: str, cap: int = 8000) -> str:
    if ocr_db is None or not Path(ocr_db).exists():
        return ""
    conn = sqlite3.connect(f"file:{ocr_db}?mode=ro", uri=True)
    parts: list[str] = []
    n = 0
    try:
        rows = conn.execute(
            "SELECT text FROM ocr WHERE parent_relpath=? ORDER BY locator",
            (relpath,),
        )
    except sqlite3.Error:
        conn.close()
        return ""
    for (text,) in rows:
        if not text:
            continue
        parts.append(text)
        n += len(text)
        if n >= cap:
            break
    conn.close()
    return "\n".join(parts)[:cap]


def keyword_hits(conn, q: str, limit: int = 40, desc_db=None) -> list[str]:
    hits: list[str] = []
    seen = set()

    def add(rows):
        for (rp,) in rows:
            if rp not in seen:
                seen.add(rp); hits.append(rp)

    try:
        add(conn.execute(
            """SELECT d.relpath FROM docs_fts JOIN docs d ON d.rowid=docs_fts.rowid
               WHERE docs_fts MATCH ? ORDER BY rank LIMIT ?""", (q, limit)))
    except sqlite3.Error:
        pass
    # full-phrase LIKE: filename first, then body
    like = f"%{q}%"
    add(conn.execute("SELECT relpath FROM docs WHERE relpath LIKE ? LIMIT ?", (like, limit)))
    if len(hits) < limit:
        add(conn.execute("SELECT relpath FROM docs WHERE text LIKE ? LIMIT ?", (like, limit)))
    if desc_db is not None and desc_db.exists():
        dc = sqlite3.connect(f"file:{desc_db}?mode=ro", uri=True)
        add(dc.execute("SELECT relpath FROM descs WHERE desc LIKE ? OR keywords_json LIKE ? LIMIT ?",
                       (like, like, limit)))
        dc.close()
    # CJK bigram scoring on relpath (cheap; catches 二三级企业 for 二三级企业清单)
    terms = [t for t in re.findall(r"[\u4e00-\u9fff]+|[A-Za-z0-9]+", q)]
    grams = set()
    for t in terms:
        if re.match(r"[\u4e00-\u9fff]", t):
            core = t
            for s in STOP:
                core = core.replace(s, "")
            grams.update(core[i:i+2] for i in range(len(core) - 1))
        elif len(t) >= 2:
            grams.add(t)
    if grams:
        score: dict[str, int] = {}
        for g in grams:
            for (rp,) in conn.execute("SELECT relpath FROM docs WHERE relpath LIKE ?", (f"%{g}%",)):
                score[rp] = score.get(rp, 0) + 1
        need = max(2, int(len(grams) * 0.6))
        ranked = sorted((rp for rp, s in score.items() if s >= need),
                        key=lambda rp: (-score[rp], len(rp)))
        add((rp,) for rp in ranked[:limit])
        # CJK bigrams in body text (unicode61 FTS does not segment CJK runs)
        cjk = [g for g in grams if re.match(r"[\u4e00-\u9fff]", g)]
        if cjk and len(hits) < limit:
            bscore: dict[str, int] = {}
            for g in cjk:
                for (rp,) in conn.execute("SELECT relpath FROM docs WHERE text LIKE ? LIMIT 500", (f"%{g}%",)):
                    bscore[rp] = bscore.get(rp, 0) + 1
            bneed = max(1, int(len(cjk) * 0.6))
            add((rp,) for rp in sorted((r for r, v in bscore.items() if v >= bneed),
                                       key=lambda r: -bscore[r])[:limit])
    return hits[:limit * 2]


def desc_bigram_hits(desc_db, q: str, limit: int = 40) -> list[str]:
    """CJK-bigram scoring on desc text + keywords (separate channel)."""
    if desc_db is None or not Path(desc_db).exists():
        return []
    grams = set()
    for t in re.findall(r"[\u4e00-\u9fff]+|[A-Za-z0-9]+", q):
        if re.match(r"[\u4e00-\u9fff]", t):
            core = t.replace("的", "")
            for s in STOP:
                core = core.replace(s, "")
            grams.update(core[i:i+2] for i in range(len(core) - 1))
        elif len(t) >= 2:
            grams.add(t.lower())
    if not grams:
        return []
    dc = sqlite3.connect(f"file:{desc_db}?mode=ro", uri=True)
    score = {}
    for rp, d, k in dc.execute("SELECT relpath, desc, keywords_json FROM descs"):
        txt = ((d or "") + " " + (k or "")).lower()
        sc = sum(1 for g in grams if g in txt)
        if sc:
            score[rp] = sc
    dc.close()
    need = max(2, int(len(grams) * 0.6))
    ranked = sorted((rp for rp, sc in score.items() if sc >= need), key=lambda rp: (-score[rp], len(rp)))
    return ranked[:limit]



def ocr_hits(ocr_db, q: str, limit: int = OCR_TOPK) -> list[str]:
    """FTS5 MATCH + phrase LIKE on ocr text; collapse to parent_relpath (best rank).

    Returns unique parents ranked by OCR relevance. Empty if DB missing.
    """
    if ocr_db is None or not Path(ocr_db).exists():
        return []
    q = (q or "").strip()
    if not q:
        return []
    conn = sqlite3.connect(f"file:{ocr_db}?mode=ro", uri=True)
    best: dict[str, float] = {}  # parent -> lower is better (FTS rank) or insertion order key
    order_keys: dict[str, float] = {}
    try:
        tokens = re.findall(r"[\u4e00-\u9fff]+|[A-Za-z0-9]{2,}", q)
        match_q = " ".join(tokens) if tokens else q
        try:
            rows = conn.execute(
                """SELECT o.parent_relpath, rank
                   FROM ocr_fts
                   JOIN ocr o ON o.id = ocr_fts.rowid
                   WHERE ocr_fts MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                (match_q, limit * 5),
            ).fetchall()
            for i, (parent, rank) in enumerate(rows):
                key = float(rank) if rank is not None else float(i)
                if parent not in best or key < best[parent]:
                    best[parent] = key
                    order_keys[parent] = key
        except sqlite3.Error:
            pass
        like = f"%{q}%"
        # phrase LIKE — fill parents not yet seen; also boost digit/ID queries
        for i, (parent,) in enumerate(
            conn.execute(
                """SELECT parent_relpath FROM ocr
                   WHERE text LIKE ? OR relpath LIKE ?
                   LIMIT ?""",
                (like, like, limit * 5),
            )
        ):
            # LIKE ranks after FTS: use large base + index
            key = 1_000_000.0 + i
            if parent not in best or key < best[parent]:
                best[parent] = key
                order_keys[parent] = key
        # CJK bigram / alnum token scoring on OCR text (cheap catch-up)
        grams = set()
        for t in tokens:
            if re.match(r"[\u4e00-\u9fff]", t):
                core = t
                for s in STOP:
                    core = core.replace(s, "")
                grams.update(core[i : i + 2] for i in range(len(core) - 1))
            elif len(t) >= 2:
                grams.add(t.lower())
        if grams:
            score: dict[str, int] = {}
            # scan is OK for ~8k rows
            for parent, text in conn.execute(
                "SELECT parent_relpath, text FROM ocr"
            ):
                txt = (text or "").lower()
                sc = sum(1 for g in grams if g in txt)
                if sc:
                    score[parent] = max(score.get(parent, 0), sc)
            need = max(1, int(len(grams) * 0.5))
            for parent, sc in score.items():
                if sc < need:
                    continue
                # higher bigram score → lower key
                key = 500_000.0 - sc * 10
                if parent not in best or key < best[parent]:
                    best[parent] = key
                    order_keys[parent] = key
    finally:
        conn.close()
    ranked = sorted(order_keys, key=lambda p: order_keys[p])
    return ranked[:limit]


def fuse(lists_weights) -> list[str]:
    rrf: dict[str, float] = {}
    for lst, w in lists_weights:
        for r, rp in enumerate(lst):
            rrf[rp] = rrf.get(rp, 0.0) + w / (K + r + 1)
    return sorted(rrf, key=rrf.get, reverse=True)


def _device() -> str:
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_body_model(model_name: str, prompt_name: str | None):
    """Load body encoder. EG2 uses text-only float32 + optional SearchQuery."""
    from sentence_transformers import SentenceTransformer
    import torch
    dev = _device()
    is_eg2 = model_name == EG2_MODEL or "embeddinggemma" in model_name.lower()
    if is_eg2:
        st_kwargs = {
            "config_kwargs": {"vision_config": None, "audio_config": None},
            "model_kwargs": {"torch_dtype": torch.float32},
            "device": dev,
        }
        return SentenceTransformer(model_name, **st_kwargs), (prompt_name or "SearchQuery")
    return SentenceTransformer(model_name, device=dev), prompt_name


def encode_query(model, query: str, prompt_name: str | None):
    kwargs = dict(normalize_embeddings=True, show_progress_bar=False)
    if prompt_name:
        kwargs["prompt_name"] = prompt_name
    return model.encode([query], **kwargs)[0]


def score_vectors(db_path: Path, model_name: str, qe) -> dict[str, float]:
    scores: dict[str, float] = {}
    if not db_path.exists():
        return scores
    vec = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    for rp, d, blob in vec.execute(
            "SELECT relpath, dim, embedding FROM vectors WHERE model=?", (model_name,)):
        if len(blob) == 4 * d:
            scores[rp] = sum(a * b for a, b in zip(qe, struct.unpack(f"<{d}f", blob)))
    vec.close()
    return scores


def embedding_mismatch(db_path: Path, model: str, dim: int | None = None) -> str | None:
    """Mismatch message for a vector or media index, or None if it agrees or is absent."""
    if not db_path.exists():
        return None
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return check_index_meta(conn, model, dim)
    except sqlite3.Error:
        return None
    finally:
        conn.close()


def score_media_parents(db_path: Path, qe, topk: int = MEDIA_TOPK, model: str | None = None) -> list[str]:
    """Brute-force cosine over media_vectors_eg2; take topK hits; group by parent_relpath.

    Collapse strategy: max score within parent among the topK media rows.
    Returns unique parent paths ranked by that max score. Empty if DB missing.
    """
    if not db_path.exists():
        return []
    scored: list[tuple[float, str]] = []
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        sql = "SELECT parent_relpath, dim, embedding FROM media"
        params: tuple = ()
        if model:
            sql += " WHERE model=?"
            params = (model,)
        for parent, d, blob in conn.execute(sql, params):
            if len(blob) != 4 * d:
                continue
            vec = struct.unpack(f"<{d}f", blob)
            score = sum(a * b for a, b in zip(qe, vec))
            scored.append((score, parent))
    finally:
        conn.close()
    scored.sort(reverse=True)
    best: dict[str, float] = {}
    order: list[str] = []
    for score, parent in scored[:topk]:
        if parent not in best:
            best[parent] = score
            order.append(parent)
        elif score > best[parent]:
            best[parent] = score
    # re-rank parents by max score (stable among ties via first-seen)
    return sorted(order, key=lambda p: best[p], reverse=True)


def _classify_eg2_media_load_error(exc: BaseException) -> str:
    """Human-readable hint: missing Pillow vs torchvision vs weights vs other."""
    msg = f"{type(exc).__name__}: {exc}"
    low = msg.lower()
    missing_pil = (
        isinstance(exc, ImportError)
        and ("pil" in low or "pillow" in low)
    ) or ("pillow" in low and ("no module" in low or "install" in low or "required" in low))
    missing_tv = (
        "torchvision" in low
        or "videoprocessor" in low
        or "sentence-transformers[image]" in low
        or "embeddinggemma2videoprocessor" in low
    )
    missing_weights = (
        "offline" in low
        or "cannot find" in low and "embeddinggemma" in low
        or "not found" in low and ("safetensors" in low or "embeddinggemma" in low)
        or "repository not found" in low
        or "local_files_only" in low
        or "hf_hub_offline" in low
    )
    hints: list[str] = []
    if missing_pil:
        hints.append(
            "missing Pillow — install pillow in this project's .venv "
            "(pip install pillow)"
        )
    if missing_tv:
        hints.append(
            "missing/incompatible torchvision (EG2 processor) — install matching "
            "torchvision into this project's .venv (or pip install -U "
            "'sentence-transformers[image]'); see docs/deploy.md"
        )
    if missing_weights and not hints:
        hints.append(
            "EG2 weights missing/unreadable under ~/.cache/huggingface/hub/ "
            "(mirror Mac cache for google/embeddinggemma-2)"
        )
    if not hints:
        # opaque processor ImportError often is Pillow even when message is vague
        if isinstance(exc, ImportError):
            hints.append(
                "ImportError during EG2/processor load — check pillow + torchvision "
                "in this project's .venv (see docs/deploy.md)"
            )
        else:
            hints.append("see docs/deploy.md for media query deps")
    return f"{msg} | hint: {'; '.join(hints)}"


def load_eg2_for_media(model_name: str | None = None, full_multimodal: bool = False):
    """Load EG2 for media query encode. Prefer text-only; full if requested.

    Catches PIL/ImportError (and processor failures that mention torchvision)
    and re-raises with a clear install/mirror hint so the media channel line
    is not an opaque processors error.
    """
    try:
        from sentence_transformers import SentenceTransformer
        import torch
    except ImportError as e:
        raise RuntimeError(_classify_eg2_media_load_error(e)) from e
    # Fail fast with a clear message if Pillow is absent (processor import
    # often buries this). Text-only EG2 still pulls the multimodal processor.
    try:
        import PIL  # noqa: F401
    except ImportError as e:
        raise RuntimeError(_classify_eg2_media_load_error(e)) from e
    dev = _device()
    kwargs = {
        "model_kwargs": {"torch_dtype": torch.float32},
        "device": dev,
    }
    if not full_multimodal:
        kwargs["config_kwargs"] = {"vision_config": None, "audio_config": None}
    try:
        return SentenceTransformer(model_name or EG2_MODEL, **kwargs)
    except Exception as e:
        # Text-only path already requested; if processor still demands image
        # extras / weights, surface a classified hint (caller prints
        # "# media channel unavailable: ...").
        raise RuntimeError(_classify_eg2_media_load_error(e)) from e



def main() -> int:
    ap = argparse.ArgumentParser(description="local-doc-kb hybrid search (RRF)")
    ap.add_argument("query")
    ap.add_argument("--config", type=Path, default=None,
                    help="path to config.json (default: <repo>/config.json or $LOCAL_DOC_KB_CONFIG)")
    ap.add_argument("-n", type=int, default=8)
    ap.add_argument("--json", action="store_true",
                    help="print one JSON object on stdout (diagnostics go to stderr)")
    ap.add_argument("--fts", type=Path, default=FTS_DB)
    # legacy body flags (kept for identical default CLI)
    ap.add_argument("--db", type=Path, default=None,
                    help="body vector db (alias of --body-db)")
    ap.add_argument("--model", default=None,
                    help="body model (alias of --body-model)")
    ap.add_argument("--body-db", type=Path, default=None,
                    help=f"body vector sqlite (default {VEC_DB.name})")
    ap.add_argument("--body-model", default=None,
                    help=f"body embedding model (default {DEFAULT_MODEL})")
    ap.add_argument("--body-prompt", default=None,
                    help='encode prompt_name for body query (e.g. SearchQuery for EG2; empty=plain)')
    ap.add_argument("--eg2-body", action="store_true",
                    help="convenience: body=vectors_eg2 + google/embeddinggemma-2 + SearchQuery")
    ap.add_argument("--eg2-all", "--eg2-full", dest="eg2_all", action="store_true",
                    help="full EG2 swap: body+desc vectors_eg2/desc_vectors_eg2; one EG2 SearchQuery encode")
    ap.add_argument("--bge-base", action="store_true",
                    help="no-op: body and description already use the configured embedding model")
    ap.add_argument("--bge-base-all", action="store_true",
                    help="no-op: body and description already use the configured embedding model")
    ap.add_argument("--bge-small", action="store_true",
                    help="rollback body only: vectors_bge_small + bge-small; desc stays default base")
    ap.add_argument("--bge-small-all", action="store_true",
                    help="rollback dual small: vectors_bge_small + desc_vectors_bge_small")
    ap.add_argument("--desc-db", type=Path, default=DESC_DB)
    ap.add_argument("--desc-vec-db", type=Path, default=DESC_VEC_DB)
    ap.add_argument("--no-desc", action="store_true", help="disable desc channel (= v1)")
    ap.add_argument("--rank-of", action="append", default=[], help="print fused rank of relpath containing this substring")
    ap.add_argument("--no-vec", action="store_true", help="keyword channel only")
    ap.add_argument("--desc-weight", type=float, default=DESC_WEIGHT, help="RRF weight of desc-vector channel")
    ap.add_argument("--desc-topk", type=int, default=DESC_TOPK)
    ap.add_argument("--desc-bigram", type=int, choices=[0, 1], default=DESC_BIGRAM, help="desc bigram keyword channel")
    ap.add_argument("--desc-bigram-weight", type=float, default=DESC_BIGRAM_WEIGHT)
    ap.add_argument("--media", dest="media", action="store_true",
                    help="force the media channel on (default: features.media, off in config.example.json)")
    ap.add_argument("--no-media", dest="media", action="store_false",
                    help="disable the media channel (media_vectors_eg2)")
    ap.add_argument("--media-db", type=Path, default=MEDIA_VEC_DB)
    ap.add_argument("--media-weight", type=float, default=MEDIA_WEIGHT)
    ap.add_argument("--media-topk", type=int, default=MEDIA_TOPK)
    ap.add_argument("--media-full-model", action="store_true",
                    help="load full multimodal EG2 for query (default: text-only; matches builder space)")
    ap.add_argument("--ocr", dest="ocr", action="store_true",
                    help="force the OCR channel on (default: features.ocr, off in config.example.json)")
    ap.add_argument("--no-ocr", dest="ocr", action="store_false",
                    help="disable the OCR channel (ocr.sqlite3)")
    ap.add_argument("--ocr-db", type=Path, default=OCR_DB)
    ap.add_argument("--ocr-weight", type=float, default=OCR_WEIGHT)
    ap.add_argument("--ocr-topk", type=int, default=OCR_TOPK)
    add_remote_flag(ap)
    ap.add_argument("--strict-model", action="store_true",
                    help="exit 4 (instead of disabling vectors) on embedding model/dim mismatch")
    ap.set_defaults(media=None, ocr=None)
    args = ap.parse_args()
    _cfg_now = _load_config(args.config) if args.config is not None else _CFG
    _feat = _features(_cfg_now)
    if args.media is None:
        args.media = _feat["media"]
    if args.ocr is None:
        args.ocr = _feat["ocr"]
    if not _feat["vectors"]:
        args.no_vec = True
    if not _feat["desc"]:
        args.no_desc = True
    _econf = emb_conf(_cfg_now)

    # optional late config: rebind path args from --config without touching module globals
    if args.config is not None:
        cfg = _load_config(args.config)
        emb = (cfg.get("embeddings") or {})
        args.fts = _index_path(cfg, "fts", "fts.sqlite3")
        args.desc_db = _index_path(cfg, "desc", "desc.sqlite3")
        args.desc_vec_db = _index_path(cfg, "desc_vectors", "desc_vectors.sqlite3")
        args.media_db = _index_path(cfg, "media_vectors_eg2", "media_vectors_eg2.sqlite3")
        args.ocr_db = _index_path(cfg, "ocr", "ocr.sqlite3")
        # stash for body-arm resolution below
        args._cfg_body_db = _index_path(cfg, "vectors", "vectors.sqlite3")
        args._cfg_vec_eg2 = _index_path(cfg, "vectors_eg2", "vectors_eg2.sqlite3")
        args._cfg_vec_bge_small = _index_path(cfg, "vectors_bge_small", "vectors_bge_small.sqlite3")
        args._cfg_desc_vec_eg2 = _index_path(cfg, "desc_vectors_eg2", "desc_vectors_eg2.sqlite3")
        args._cfg_desc_vec_bge_small = _index_path(cfg, "desc_vectors_bge_small", "desc_vectors_bge_small.sqlite3")
        args._cfg_default_model = emb.get("body_model") or emb.get("model") or DEFAULT_MODEL
        args._cfg_bge_small = emb.get("bge_small_model") or BGE_SMALL_MODEL
        args._cfg_eg2 = emb.get("eg2_model") or EG2_MODEL
    else:
        args._cfg_body_db = VEC_DB
        args._cfg_vec_eg2 = VEC_EG2_DB
        args._cfg_vec_bge_small = VEC_BGE_SMALL_DB
        args._cfg_desc_vec_eg2 = DESC_VEC_EG2_DB
        args._cfg_desc_vec_bge_small = DESC_VEC_BGE_SMALL_DB
        args._cfg_default_model = DEFAULT_MODEL
        args._cfg_bge_small = BGE_SMALL_MODEL
        args._cfg_eg2 = EG2_MODEL

    if not args.fts.exists():
        print(
            "ERROR: FTS index not found at\n"
            f"  {args.fts}\n"
            "Configure document roots in config.json, build or mirror indexes into "
            "index/, then retry. See docs/deploy.md and config.example.json.",
            file=sys.stderr,
        )
        return 2

    # resolve body/desc arms
    # precedence: --eg2-all > --eg2-body > --bge-small-all > --bge-small > explicit > defaults(base)
    # --bge-base / --bge-base-all are no-ops (default already dual base)
    body_db = args._cfg_body_db
    body_model = args._cfg_default_model
    body_prompt: str | None = None
    desc_model = args._cfg_default_model
    desc_prompt: str | None = None
    def note(msg: str) -> None:
        print(msg, file=sys.stderr if args.json else sys.stdout)

    if args.bge_base or args.bge_base_all:
        note("# note: --bge-base/--bge-base-all are no-ops; body and description already use the configured embedding model")
    # keep args.desc_vec_db as-is unless full-swap / rollback overrides
    if args.eg2_all:
        body_db = args._cfg_vec_eg2
        body_model = args._cfg_eg2
        body_prompt = "SearchQuery"
        desc_model = args._cfg_eg2
        desc_prompt = "SearchQuery"
        if "desc_vectors" in args.desc_vec_db.name and "eg2" not in args.desc_vec_db.name:
            args.desc_vec_db = args._cfg_desc_vec_eg2
    elif args.eg2_body:
        body_db = args._cfg_vec_eg2
        body_model = args._cfg_eg2
        body_prompt = "SearchQuery"
    elif getattr(args, "bge_small_all", False):
        body_db = args._cfg_vec_bge_small
        body_model = args._cfg_bge_small
        body_prompt = None
        desc_model = args._cfg_bge_small
        desc_prompt = None
        if "desc_vectors" in args.desc_vec_db.name and "bge_small" not in args.desc_vec_db.name:
            args.desc_vec_db = args._cfg_desc_vec_bge_small
    elif getattr(args, "bge_small", False):
        body_db = args._cfg_vec_bge_small
        body_model = args._cfg_bge_small
        body_prompt = None
    if args.db is not None:
        body_db = args.db
    if args.body_db is not None:
        body_db = args.body_db
    if args.model is not None:
        body_model = args.model
    if args.body_model is not None:
        body_model = args.body_model
    if args.body_prompt is not None:
        body_prompt = args.body_prompt or None  # empty string → plain

    off: list[str] = []
    if not args.no_vec and not body_db.exists():
        off.append(f"vectors(missing {body_db.name})")
        args.no_vec = True
    elif args.no_vec:
        off.append("vectors(disabled)")
    if args.no_desc:
        off.append("desc(disabled)")
    elif not args.desc_db.exists():
        off.append(f"desc(missing {args.desc_db.name})")
    if not args.media:
        off.append("media(disabled)")
    if not args.ocr:
        off.append("ocr(disabled)")
    channels_off = off
    note("# channels off: " + (", ".join(channels_off) if channels_off else "none") + "  (fts always on)")
    use_provider = (body_model == args._cfg_default_model and body_prompt is None
                    and desc_model == body_model)
    if not args.no_vec and use_provider:
        enforce_remote_gate(_cfg_now, [_econf], ack=args.i_understand_remote)

    fts = sqlite3.connect(f"file:{args.fts}?mode=ro", uri=True)
    kw = keyword_hits(fts, args.query.strip(), desc_db=None if args.no_desc else args.desc_db)

    vtop: list[str] = []
    vscore: dict[str, float] = {}
    dtop: list[str] = []
    dscore: dict[str, float] = {}
    if not args.no_vec:
        try:
          if use_provider:
            emb = Embedder(_econf)
            body_model = desc_model = emb.model_id
            q_body = emb.encode([args.query], query=True)[0]
            for dbp in (body_db, args.desc_vec_db):
                if not dbp.exists():
                    continue
                if dbp == args.desc_vec_db and args.no_desc:
                    continue
                _c = sqlite3.connect(f"file:{dbp}?mode=ro", uri=True)
                mm = check_index_meta(_c, emb.model_id, len(q_body))
                _c.close()
                if mm:
                    print(f"# WARNING embedding mismatch in {dbp.name}: {mm}. "
                          "Query embeddings must use the index model; rebuild vectors or fix config.",
                          file=sys.stderr)
                    if args.strict_model:
                        return 4
                    raise RuntimeError("model mismatch; vector channel disabled")
            vscore = score_vectors(body_db, body_model, q_body)
            vtop = sorted(vscore, key=vscore.get, reverse=True)[:40]
            if not args.no_desc and args.desc_vec_db.exists():
                dscore = score_vectors(args.desc_vec_db, desc_model, q_body)
                dtop = sorted(dscore, key=dscore.get, reverse=True)[:args.desc_topk]
          else:
            for dbp, mid in ((body_db, body_model), (args.desc_vec_db, desc_model)):
                if dbp == args.desc_vec_db and (args.no_desc or not dbp.exists()):
                    continue
                mm = embedding_mismatch(dbp, mid, None)
                if mm:
                    print(f"# WARNING embedding mismatch in {dbp.name}: {mm}. "
                          "Query embeddings must use the index model; rebuild vectors or fix config.",
                          file=sys.stderr)
                    if args.strict_model:
                        return 4
                    raise RuntimeError("model mismatch; vector channel disabled")
            body_st, body_prompt_resolved = load_body_model(body_model, body_prompt)
            q_body = encode_query(body_st, args.query, body_prompt_resolved)
            mm = embedding_mismatch(body_db, body_model, len(q_body))
            if mm:
                print(f"# WARNING embedding mismatch in {body_db.name}: {mm}. "
                      "Query embeddings must use the index model; rebuild vectors or fix config.",
                      file=sys.stderr)
                if args.strict_model:
                    return 4
                raise RuntimeError("model mismatch; vector channel disabled")
            vscore = score_vectors(body_db, body_model, q_body)
            vtop = sorted(vscore, key=vscore.get, reverse=True)[:40]

            if not args.no_desc and args.desc_vec_db.exists():
                # Same model (+ same prompt family): reuse one query encode for both arms
                if body_model == desc_model and body_prompt_resolved == desc_prompt:
                    q_desc = q_body
                elif body_model == desc_model and not body_prompt_resolved and not desc_prompt:
                    # default bge path: one encode for body+desc
                    q_desc = q_body
                else:
                    # different model (e.g. --eg2-body with desc still bge)
                    if "embeddinggemma" in desc_model.lower():
                        desc_st, desc_prompt_resolved = load_body_model(desc_model, desc_prompt or "SearchQuery")
                        q_desc = encode_query(desc_st, args.query, desc_prompt_resolved)
                    else:
                        from sentence_transformers import SentenceTransformer
                        desc_st = SentenceTransformer(desc_model, device=_device())
                        q_desc = encode_query(desc_st, args.query, None)
                mm = embedding_mismatch(args.desc_vec_db, desc_model, len(q_desc))
                if mm:
                    print(f"# WARNING embedding mismatch in {args.desc_vec_db.name}: {mm}. "
                          "Query embeddings must use the index model; rebuild vectors or fix config.",
                          file=sys.stderr)
                    if args.strict_model:
                        return 4
                    raise RuntimeError("model mismatch; vector channel disabled")
                dscore = score_vectors(args.desc_vec_db, desc_model, q_desc)
                dtop = sorted(dscore, key=dscore.get, reverse=True)[:args.desc_topk]
        except Exception as e:  # vectors optional
            note(f"# vector channel unavailable: {e}")

    db = [] if (args.no_desc or not args.desc_bigram) else desc_bigram_hits(args.desc_db, args.query.strip())
    mtop: list[str] = []
    if args.media:
        try:
            media_model = args._cfg_eg2
            if not args.media_db.exists():
                note(f"# media channel unavailable: missing {args.media_db.name}")
            else:
                mm = embedding_mismatch(args.media_db, media_model, None)
                if mm:
                    print(f"# WARNING embedding mismatch in {args.media_db.name}: {mm}. "
                          "Rebuild media vectors or fix embeddings.eg2_model.",
                          file=sys.stderr)
                    if args.strict_model:
                        return 4
                    raise RuntimeError("model mismatch; media channel disabled")
                _mc = sqlite3.connect(f"file:{args.media_db}?mode=ro", uri=True)
                try:
                    n_media = _mc.execute(
                        "SELECT COUNT(*) FROM media WHERE model=?", (media_model,)).fetchone()[0]
                finally:
                    _mc.close()
                if n_media == 0:
                    note(f"# media channel unavailable: no rows for {media_model}")
                else:
                    # Text-only EG2 shares the 768-d space with the full multimodal builder.
                    media_st = load_eg2_for_media(media_model, full_multimodal=args.media_full_model)
                    q_media = encode_query(media_st, args.query, "SearchQuery")
                    mm = embedding_mismatch(args.media_db, media_model, len(q_media))
                    if mm:
                        print(f"# WARNING embedding mismatch in {args.media_db.name}: {mm}. "
                              "Rebuild media vectors or fix embeddings.eg2_model.",
                              file=sys.stderr)
                        if args.strict_model:
                            return 4
                        raise RuntimeError("model mismatch; media channel disabled")
                    mtop = score_media_parents(args.media_db, q_media, topk=args.media_topk, model=media_model)
        except Exception as e:
            # load_eg2_for_media already wraps with Pillow/torchvision/weights hints
            detail = str(e)
            note(f"# media channel unavailable: {detail}")
            mtop = []
    otop: list[str] = []
    if args.ocr:
        try:
            if not args.ocr_db.exists():
                note(f"# ocr channel unavailable: missing {args.ocr_db}")
            else:
                otop = ocr_hits(args.ocr_db, args.query.strip(), limit=args.ocr_topk)
        except Exception as e:
            note(f"# ocr channel unavailable: {e}")
            otop = []
    channels = [(kw, 1.0), (vtop, 1.0), (dtop, args.desc_weight), (db, args.desc_bigram_weight)]
    if args.media:
        channels.append((mtop, args.media_weight))
    if args.ocr:
        channels.append((otop, args.ocr_weight))
    full = fuse(channels)
    for sub in args.rank_of:
        r = next((i + 1 for i, rp in enumerate(dtop) if sub in rp), None)
        r2 = next((i + 1 for i, rp in enumerate(sorted(dscore, key=dscore.get, reverse=True)) if sub in rp), None)
        note(f"#DESCRANK {sub}\ttop{r}\traw{r2}\tkw{next((i+1 for i,rp in enumerate(kw) if sub in rp),None)}\tvec{next((i+1 for i,rp in enumerate(vtop) if sub in rp),None)}\tbigram{next((i+1 for i,rp in enumerate(db) if sub in rp),None)}")
    for sub in args.rank_of:
        r = next((i + 1 for i, rp in enumerate(full) if sub in rp), None)
        extra = ""
        if args.media:
            mrnk = next((i + 1 for i, rp in enumerate(mtop) if sub in rp), None)
            extra += f"\tmedia{mrnk}"
        if args.ocr:
            ornk = next((i + 1 for i, rp in enumerate(otop) if sub in rp), None)
            extra += f"\tocr{ornk}"
        note(f"#RANK {sub}\t{r}{extra}")
    merged = full[: args.n]
    kwr = {rp: i + 1 for i, rp in enumerate(kw)}
    vr = {rp: i + 1 for i, rp in enumerate(vtop)}
    dr = {rp: i + 1 for i, rp in enumerate(dtop)}
    body_tag = f"body={body_model}" + (f"/{body_prompt}" if body_prompt else "")
    desc_tag = f"desc={desc_model}" + (f"/{desc_prompt}" if desc_prompt else "")
    mr = {rp: i + 1 for i, rp in enumerate(mtop)} if args.media else {}
    orr = {rp: i + 1 for i, rp in enumerate(otop)} if args.ocr else {}
    media_note = f" + media {len(mtop)}" if args.media else ""
    ocr_note = f" + ocr {len(otop)}" if args.ocr else ""
    summary = (
        f"# hybrid hits for {args.query!r} (keyword {len(kw)} + vector {len(vtop)} + desc {len(dtop)}"
        f"{media_note}{ocr_note}, RRF; {body_tag}; {desc_tag})"
    )
    note(summary)
    desc_rows = load_desc_rows(None if args.no_desc else args.desc_db, merged)
    hits_out = []
    for i, rp in enumerate(merged, 1):
        row = fts.execute("SELECT chars, text FROM docs WHERE relpath=?", (rp,)).fetchone()
        chars, text = row if row else (0, "")
        if text and text.lstrip().startswith("<?xml"):
            text = re.sub(r"<[^>]+>", " ", text)
        snip = best_excerpt(text or "", args.query)
        ocr_snip = ""
        if args.ocr and not excerpt_covers(snip, args.query):
            ocr_snip = best_excerpt(ocr_blob(args.ocr_db, rp), args.query, limit=280)
        desc, keywords = desc_rows.get(rp, ("", []))
        channels = {
            "kw": kwr.get(rp),
            "vec": vr.get(rp),
            "desc": dr.get(rp),
        }
        if args.media:
            channels["media"] = mr.get(rp)
        if args.ocr:
            channels["ocr"] = orr.get(rp)
        hits_out.append({
            "rank": i,
            "relpath": rp,
            "chars": chars,
            "channels": channels,
            "desc": desc,
            "keywords": keywords[:8],
            "excerpt": snip,
            "ocr_excerpt": ocr_snip,
        })
        if args.json:
            continue
        media_bit = f" media#{mr.get(rp, '-')}" if args.media else ""
        ocr_bit = f" ocr#{orr.get(rp, '-')}" if args.ocr else ""
        print(
            f"\n## {i}. {rp}  (chars={chars} kw#{kwr.get(rp, '-')} vec#{vr.get(rp, '-')}"
            f" desc#{dr.get(rp, '-')}{media_bit}{ocr_bit})"
        )
        if desc:
            print("desc: " + re.sub(r"\s+", " ", desc).strip())
        if keywords:
            print("keywords: " + ", ".join(keywords[:8]))
        if snip:
            print("excerpt: " + snip)
        if ocr_snip:
            print("ocr: " + ocr_snip)
    fts.close()
    if args.json:
        print(json.dumps({
            "query": args.query,
            "channels_off": channels_off,
            "hits": hits_out,
        }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
