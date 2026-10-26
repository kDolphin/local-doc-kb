#!/usr/bin/env python3
"""Build EG2 multimodal media vectors (images + scanned PDF pages).

Output: index/media_vectors_eg2.sqlite3 (config.index.media_vectors_eg2).
Does NOT touch body/desc vector tables. Local only.

Scope:
- Image extensions under configured document roots
- Scanned/empty PDFs from FTS (chars < pdf_char_thresh), pages via pdftoppm
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_config import (  # noqa: E402
    DEFAULT_IMG_EXTS,
    PROJECT_ROOT,
    build_section,
    denylist_prefixes,
    embedding_setting,
    exclude_globs,
    index_path,
    is_path_denied,
    iter_files_under,
    load_config,
    resolve_doc_scan_roots,
)
from lib_embed import pack_embedding, pick_device  # noqa: E402
from lib_providers import check_index_meta, features, write_index_meta  # noqa: E402

TZ = ZoneInfo("Asia/Shanghai")
PDF_CHAR_THRESH = 50
PDF_PAGE_CAP = 30
PDF_DPI = 150
COMMIT_EVERY = 25
DEFAULT_MODEL = "google/embeddinggemma-2"


def setup_log(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("build_media_eg2")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def file_fingerprint(path: Path) -> str:
    st = path.stat()
    return f"sz{st.st_size}_mt{int(st.st_mtime_ns)}"


def page_fingerprint(path: Path, page: int) -> str:
    return f"{file_fingerprint(path)}_p{page}"


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS media (
            relpath TEXT NOT NULL,
            parent_relpath TEXT NOT NULL,
            fingerprint TEXT NOT NULL,
            modality TEXT NOT NULL,
            locator TEXT NOT NULL DEFAULT '',
            dim INTEGER NOT NULL,
            model TEXT NOT NULL,
            embedding BLOB NOT NULL,
            embedded_at TEXT NOT NULL,
            PRIMARY KEY (relpath, locator)
        )"""
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_media_parent ON media(parent_relpath)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_media_modality ON media(modality)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_media_fp ON media(fingerprint)")
    conn.commit()


def collect_scanned_pdfs(
    fts_db: Path,
    roots: list[Path],
    log: logging.Logger,
    *,
    thresh: int,
    prefixes: list[str],
    globs: list[str],
) -> list[tuple[str, Path]]:
    if not fts_db.exists():
        log.error("FTS missing: %s", fts_db)
        return []
    conn = sqlite3.connect(f"file:{fts_db}?mode=ro", uri=True)
    rows = conn.execute(
        """SELECT relpath FROM docs
           WHERE lower(relpath) LIKE '%.pdf'
             AND coalesce(chars, 0) < ?
           ORDER BY relpath""",
        (thresh,),
    ).fetchall()
    conn.close()
    # Map relpath → absolute via roots
    out: list[tuple[str, Path]] = []
    missing = 0
    multi = len(roots) > 1
    for (rel,) in rows:
        rel = rel.replace("\\", "/")
        if is_path_denied(rel, prefixes, globs):
            continue
        found = None
        for root in roots:
            # With multi-root, relpath is "basename/..."
            if multi and rel.startswith(root.name + "/"):
                cand = root / rel[len(root.name) + 1 :]
            else:
                cand = root / rel
            if cand.is_file():
                found = cand
                break
        if found is None:
            missing += 1
            continue
        out.append((rel, found))
    log.info(
        "collected scanned/empty PDFs: %d (fts_lt%d=%d missing_on_disk=%d)",
        len(out),
        thresh,
        len(rows),
        missing,
    )
    return out


def pdf_page_count(path: Path) -> int:
    try:
        r = subprocess.run(
            ["pdfinfo", str(path)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if r.returncode == 0:
            for line in r.stdout.splitlines():
                if line.startswith("Pages:"):
                    return int(line.split(":", 1)[1].strip())
    except Exception:
        pass
    return -1


def render_pdf_pages(
    path: Path, out_dir: Path, max_pages: int, dpi: int
) -> list[tuple[int, Path]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / "page"
    cmd = [
        "pdftoppm",
        "-png",
        "-r",
        str(dpi),
        "-f",
        "1",
        "-l",
        str(max_pages),
        str(path),
        str(prefix),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        raise RuntimeError(f"pdftoppm rc={r.returncode}: {(r.stderr or r.stdout)[:300]}")
    pages: list[tuple[int, Path]] = []
    for p in sorted(out_dir.glob("page-*.png")):
        try:
            num = int(p.stem.split("-", 1)[1])
        except Exception:
            continue
        pages.append((num, p))
    pages.sort(key=lambda x: x[0])
    return pages


def load_model(model_name: str, device: str, log: logging.Logger):
    import torch
    from sentence_transformers import SentenceTransformer

    log.info("loading full multimodal EG2 float32 on %s …", device)
    t0 = time.time()
    model = SentenceTransformer(
        model_name,
        device=device,
        model_kwargs={"torch_dtype": torch.float32},
    )
    get_dim = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
    dim = int(get_dim())
    log.info("model ready dim=%d load_s=%.1f", dim, time.time() - t0)
    return model, dim


def open_rgb_image(path: Path, max_side: int = 4096):
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    im = Image.open(path)
    try:
        try:
            im.draft("RGB", (max_side, max_side))
        except Exception:
            pass
        im = im.convert("RGB")
        w, h = im.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            im = im.resize(
                (max(1, int(w * scale)), max(1, int(h * scale))),
                Image.Resampling.LANCZOS,
            )
        return im
    except Exception:
        im.close()
        raise


def encode_image(model, pil_img, device: str):
    import torch

    emb = model.encode(
        [{"image": pil_img}],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]
    if device == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    return emb


def existing_fps(conn: sqlite3.Connection) -> dict[tuple[str, str], str]:
    return {
        (rp, loc): fp
        for rp, loc, fp in conn.execute("SELECT relpath, locator, fingerprint FROM media")
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="EG2 media vectors: images + scanned PDF pages under config roots"
    )
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--root", type=Path, action="append", default=None)
    ap.add_argument("--fts", type=Path, default=None)
    ap.add_argument("--db", type=Path, default=None)
    ap.add_argument("--log", type=Path, default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit-images", type=int, default=0, help="0=all")
    ap.add_argument("--limit-pdfs", type=int, default=0, help="0=all PDFs")
    ap.add_argument("--pdf-page-cap", type=int, default=None)
    ap.add_argument("--pdf-dpi", type=int, default=None)
    ap.add_argument("--pdf-char-thresh", type=int, default=None)
    ap.add_argument("--images-only", action="store_true")
    ap.add_argument("--pdfs-only", action="store_true")
    ap.add_argument("--rebuild", action="store_true", help="wipe media table first")
    ap.add_argument("--commit-every", type=int, default=COMMIT_EVERY)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if not features(cfg)["media"]:
        print("SKIP: features.media is false in config", flush=True)
        return 0
    b = build_section(cfg)
    roots = resolve_doc_scan_roots(cfg, args.root)
    fts_db = args.fts or index_path(cfg, "fts", "fts.sqlite3")
    out_db = args.db or index_path(cfg, "media_vectors_eg2", "media_vectors_eg2.sqlite3")
    log_path = args.log or (PROJECT_ROOT / "logs" / "build_media_vectors_eg2.log")
    model_name = args.model or embedding_setting(cfg, "eg2_model", default=DEFAULT_MODEL)
    page_cap = args.pdf_page_cap if args.pdf_page_cap is not None else int(b.get("pdf_page_cap") or PDF_PAGE_CAP)
    dpi = args.pdf_dpi if args.pdf_dpi is not None else int(b.get("pdf_dpi") or PDF_DPI)
    thresh = (
        args.pdf_char_thresh
        if args.pdf_char_thresh is not None
        else int(b.get("pdf_char_thresh") or PDF_CHAR_THRESH)
    )
    img_exts = set(DEFAULT_IMG_EXTS)
    if b.get("media_image_exts"):
        img_exts = {
            (e if str(e).startswith(".") else f".{e}").lower() for e in b["media_image_exts"]
        }
    prefixes = denylist_prefixes(cfg)
    globs = exclude_globs(cfg)

    log = setup_log(log_path)
    log.info("=== build_media_vectors_eg2 start ===")
    log.info("roots=%s fts=%s db=%s page_cap=%d dpi=%d", roots, fts_db, out_db, page_cap, dpi)

    if not roots:
        log.error("No document roots. Set roots[] in config.json or pass --root.")
        return 2
    for r in roots:
        if not r.is_dir():
            log.error("Document root missing: %s", r)
            return 2

    import torch  # noqa: F401

    device = pick_device(args.device)
    out_db.parent.mkdir(parents=True, exist_ok=True)
    model, dim = load_model(model_name, device, log)

    conn = sqlite3.connect(out_db)
    ensure_schema(conn)
    mm = check_index_meta(conn, model_name, dim)
    if mm and not args.rebuild:
        log.error("embedding model mismatch: %s", mm)
        print(
            f"ERROR: embedding model mismatch: {mm}.\n"
            "  Switching the media model requires --rebuild. Refusing to mix vector spaces.",
            file=sys.stderr,
        )
        return 4
    write_index_meta(conn, model=model_name, dim=dim, provider="sentence_transformers")
    if args.rebuild:
        n = conn.execute("DELETE FROM media").rowcount
        conn.commit()
        log.info("rebuild: deleted %d rows", n)

    have = existing_fps(conn)
    log.info("already in DB: %d rows", len(have))

    failures: list[str] = []
    n_ok = n_skip = n_fail = 0
    t0 = time.time()
    pending = 0

    def now_fmt() -> str:
        return datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S Asia/Shanghai")

    if not args.pdfs_only:
        images = iter_files_under(roots, exts=img_exts, prefixes=prefixes, globs=globs)
        if args.limit_images:
            images = images[: args.limit_images]
        log.info("embedding images: %d", len(images))
        for i, (rel, path) in enumerate(images):
            loc = ""
            try:
                fp = file_fingerprint(path)
                if have.get((rel, loc)) == fp:
                    n_skip += 1
                    continue
                im = open_rgb_image(path)
                try:
                    emb = encode_image(model, im, device)
                finally:
                    try:
                        im.close()
                    except Exception:
                        pass
                conn.execute(
                    """INSERT OR REPLACE INTO media
                       (relpath, parent_relpath, fingerprint, modality, locator,
                        dim, model, embedding, embedded_at)
                       VALUES (?,?,?,?,?,?,?,?,?)""",
                    (rel, rel, fp, "image", loc, dim, model_name, pack_embedding(emb), now_fmt()),
                )
                n_ok += 1
                pending += 1
                have[(rel, loc)] = fp
            except Exception as e:
                n_fail += 1
                msg = f"image FAIL {rel}: {e}"
                failures.append(msg)
                log.warning(msg)
                if device == "mps":
                    try:
                        torch.mps.empty_cache()
                    except Exception:
                        pass
            if pending >= args.commit_every:
                conn.commit()
                pending = 0
            if (i + 1) % 50 == 0 or i + 1 == len(images):
                elapsed = time.time() - t0
                done = n_ok + n_fail
                rate = done / max(elapsed, 0.01)
                left = len(images) - (i + 1)
                log.info(
                    "img progress %d/%d ok=%d skip=%d fail=%d rate=%.2f/s eta=%.0fs",
                    i + 1,
                    len(images),
                    n_ok,
                    n_skip,
                    n_fail,
                    rate,
                    left / max(rate, 0.01),
                )
        conn.commit()
        pending = 0

    if not args.images_only:
        pdfs = collect_scanned_pdfs(
            fts_db, roots, log, thresh=thresh, prefixes=prefixes, globs=globs
        )
        if args.limit_pdfs:
            pdfs = pdfs[: args.limit_pdfs]
        log.info("embedding PDF pages from %d PDFs (cap=%d)", len(pdfs), page_cap)
        tmp_root = Path(tempfile.mkdtemp(prefix="eg2mm_pdf_"))
        try:
            for pi, (rel, path) in enumerate(pdfs):
                try:
                    n_pages = pdf_page_count(path)
                    cap = page_cap
                    if n_pages > 0:
                        cap = min(cap, n_pages)
                    work = tmp_root / f"pdf_{pi}"
                    if work.exists():
                        shutil.rmtree(work, ignore_errors=True)
                    pages = render_pdf_pages(path, work, cap, dpi)
                    if not pages:
                        raise RuntimeError("pdftoppm produced 0 pages")
                    for page_num, png in pages:
                        loc = f"page:{page_num}"
                        try:
                            fp = page_fingerprint(path, page_num)
                            if have.get((rel, loc)) == fp:
                                n_skip += 1
                                continue
                            im = open_rgb_image(png)
                            try:
                                emb = encode_image(model, im, device)
                            finally:
                                try:
                                    im.close()
                                except Exception:
                                    pass
                            conn.execute(
                                """INSERT OR REPLACE INTO media
                                   (relpath, parent_relpath, fingerprint, modality,
                                    locator, dim, model, embedding, embedded_at)
                                   VALUES (?,?,?,?,?,?,?,?,?)""",
                                (
                                    rel,
                                    rel,
                                    fp,
                                    "pdfpage",
                                    loc,
                                    dim,
                                    model_name,
                                    pack_embedding(emb),
                                    now_fmt(),
                                ),
                            )
                            n_ok += 1
                            pending += 1
                            have[(rel, loc)] = fp
                        except Exception as e:
                            n_fail += 1
                            msg = f"pdfpage FAIL {rel} {loc}: {e}"
                            failures.append(msg)
                            log.warning(msg)
                            if device == "mps":
                                try:
                                    torch.mps.empty_cache()
                                except Exception:
                                    pass
                    shutil.rmtree(work, ignore_errors=True)
                except Exception as e:
                    n_fail += 1
                    msg = f"pdf FAIL {rel}: {e}"
                    failures.append(msg)
                    log.warning(msg)
                if pending >= args.commit_every:
                    conn.commit()
                    pending = 0
                if (pi + 1) % 10 == 0 or pi + 1 == len(pdfs):
                    elapsed = time.time() - t0
                    log.info(
                        "pdf progress %d/%d ok=%d skip=%d fail=%d elapsed=%.0fs",
                        pi + 1,
                        len(pdfs),
                        n_ok,
                        n_skip,
                        n_fail,
                        elapsed,
                    )
            conn.commit()
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

    by_mod = dict(conn.execute("SELECT modality, COUNT(*) FROM media GROUP BY modality").fetchall())
    total = conn.execute("SELECT COUNT(*) FROM media").fetchone()[0]
    conn.close()
    wall = time.time() - t0
    log.info(
        "DONE ok=%d skip=%d fail=%d total_rows=%d by_modality=%s wall_s=%.1f db=%s",
        n_ok,
        n_skip,
        n_fail,
        total,
        by_mod,
        wall,
        out_db,
    )
    if failures:
        fail_path = log_path.with_suffix(".failures.txt")
        fail_path.write_text("\n".join(failures) + "\n", encoding="utf-8")
        log.info("failures written: %s (%d)", fail_path, len(failures))
    return 0 if n_fail == 0 or n_ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
