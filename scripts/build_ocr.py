#!/usr/bin/env python3
"""Build OCR index for images + scanned PDF pages under configured roots.

Output: index/ocr.sqlite3. Does NOT touch FTS / vectors / desc / media tables.
Engines: Apple Vision via configurable vision CLI (primary); tesseract
chi_sim+eng fallback when Vision returns empty or short text. Local only.
"""
from __future__ import annotations

import argparse
import logging
import os
import re
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
    exclude_globs,
    index_path,
    is_path_denied,
    iter_files_under,
    load_config,
    resolve_doc_scan_roots,
    resolve_path,
)
from lib_providers import features  # noqa: E402

TZ = ZoneInfo("Asia/Shanghai")
PDF_CHAR_THRESH = 50
PDF_PAGE_CAP = 30
PDF_DPI = 200
COMMIT_EVERY = 25
PROGRESS_EVERY = 25
VISION_MIN_CHARS = 20


def setup_log(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("build_ocr")
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


def now_fmt() -> str:
    return datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S Asia/Shanghai")


def file_fingerprint(path: Path) -> str:
    st = path.stat()
    return f"sz{st.st_size}_mt{int(st.st_mtime_ns)}"


def page_fingerprint(path: Path, page: int) -> str:
    return f"{file_fingerprint(path)}_p{page}"


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS ocr (
            id INTEGER PRIMARY KEY,
            relpath TEXT NOT NULL,
            parent_relpath TEXT NOT NULL,
            fingerprint TEXT NOT NULL,
            modality TEXT NOT NULL,
            locator TEXT NOT NULL DEFAULT '',
            engine TEXT NOT NULL,
            text TEXT NOT NULL,
            chars INTEGER NOT NULL,
            ocr_at TEXT NOT NULL,
            UNIQUE(relpath, locator)
        )"""
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ocr_parent ON ocr(parent_relpath)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ocr_fp ON ocr(fingerprint)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ocr_modality ON ocr(modality)")
    conn.execute(
        """CREATE VIRTUAL TABLE IF NOT EXISTS ocr_fts USING fts5(
            text,
            content='ocr',
            content_rowid='id',
            tokenize='unicode61'
        )"""
    )
    conn.commit()


def upsert_ocr(
    conn: sqlite3.Connection,
    *,
    relpath: str,
    parent: str,
    fp: str,
    modality: str,
    locator: str,
    engine: str,
    text: str,
) -> None:
    chars = len(text or "")
    row = conn.execute(
        "SELECT id FROM ocr WHERE relpath=? AND locator=?",
        (relpath, locator),
    ).fetchone()
    if row:
        rid = row[0]
        conn.execute("DELETE FROM ocr_fts WHERE rowid=?", (rid,))
        conn.execute(
            """UPDATE ocr SET parent_relpath=?, fingerprint=?, modality=?,
               engine=?, text=?, chars=?, ocr_at=? WHERE id=?""",
            (parent, fp, modality, engine, text or "", chars, now_fmt(), rid),
        )
        conn.execute("INSERT INTO ocr_fts(rowid, text) VALUES (?, ?)", (rid, text or ""))
    else:
        cur = conn.execute(
            """INSERT INTO ocr
               (relpath, parent_relpath, fingerprint, modality, locator,
                engine, text, chars, ocr_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (relpath, parent, fp, modality, locator, engine, text or "", chars, now_fmt()),
        )
        rid = cur.lastrowid
        conn.execute("INSERT INTO ocr_fts(rowid, text) VALUES (?, ?)", (rid, text or ""))


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
    out: list[tuple[str, Path]] = []
    missing = 0
    multi = len(roots) > 1
    for (rel,) in rows:
        rel = rel.replace("\\", "/")
        if is_path_denied(rel, prefixes, globs):
            continue
        found = None
        for root in roots:
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
            ["pdfinfo", str(path)], capture_output=True, text=True, timeout=60
        )
        if r.returncode == 0:
            for line in r.stdout.splitlines():
                if line.startswith("Pages:"):
                    return int(line.split(":", 1)[1].strip())
    except Exception:
        pass
    return -1


def strip_vision_headers(raw: str) -> str:
    lines = []
    for line in raw.splitlines():
        if line.startswith("=====") and line.endswith("====="):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def parse_vision_pdf_pages(raw: str) -> dict[int, str]:
    pages: dict[int, str] = {}
    cur: int | None = None
    buf: list[str] = []
    hdr = re.compile(r"^===== page (\d+) chars=\d+ sec=[\d.]+ =====\s*$")
    for line in raw.splitlines():
        m = hdr.match(line)
        if m:
            if cur is not None:
                pages[cur] = "\n".join(buf).strip()
            cur = int(m.group(1))
            buf = []
            continue
        if line.startswith("=====") and line.endswith("====="):
            continue
        if cur is not None:
            buf.append(line)
    if cur is not None:
        pages[cur] = "\n".join(buf).strip()
    return pages


def run_vision_image(vision_bin: Path, path: Path, timeout: int = 120) -> str:
    r = subprocess.run(
        [str(vision_bin), str(path)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if r.returncode != 0 and not (r.stdout or "").strip():
        raise RuntimeError(f"visionocr rc={r.returncode}: {(r.stderr or '')[:300]}")
    return strip_vision_headers(r.stdout or "")


def run_vision_pdf_pages(
    vision_bin: Path, path: Path, max_pages: int, dpi: int, timeout: int = 600
) -> dict[int, str]:
    r = subprocess.run(
        [
            str(vision_bin),
            "--pdf-max",
            str(max_pages),
            "--pdf-dpi",
            str(dpi),
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if r.returncode != 0 and not (r.stdout or "").strip():
        raise RuntimeError(f"visionocr pdf rc={r.returncode}: {(r.stderr or '')[:300]}")
    return parse_vision_pdf_pages(r.stdout or "")


def run_vision_pdf_page(
    vision_bin: Path, path: Path, page: int, dpi: int, timeout: int = 180
) -> str:
    r = subprocess.run(
        [
            str(vision_bin),
            "--pdf-page",
            str(page),
            "--pdf-dpi",
            str(dpi),
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if r.returncode != 0 and not (r.stdout or "").strip():
        raise RuntimeError(
            f"visionocr pdf-page rc={r.returncode}: {(r.stderr or '')[:300]}"
        )
    return strip_vision_headers(r.stdout or "")


def render_pdf_pages_pdftoppm(
    path: Path,
    out_dir: Path,
    max_pages: int,
    dpi: int,
    first_page: int = 1,
    timeout: int = 60,
) -> list[tuple[int, Path]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / "page"
    first = max(1, int(first_page))
    last = max(first, int(max_pages))
    cmd = [
        "pdftoppm",
        "-png",
        "-r",
        str(dpi),
        "-f",
        str(first),
        "-l",
        str(last),
        str(path),
        str(prefix),
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, 15)
        except ProcessLookupError:
            pass
        try:
            stdout, stderr = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, 9)
            except ProcessLookupError:
                pass
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            stdout, stderr = proc.communicate()
        raise RuntimeError(
            f"pdftoppm timeout={timeout}s killed pg pages {first}-{last}: "
            f"{(stderr or stdout or '')[:200]}"
        )
    if proc.returncode != 0:
        raise RuntimeError(
            f"pdftoppm rc={proc.returncode}: {(stderr or stdout or '')[:300]}"
        )
    pages: list[tuple[int, Path]] = []
    for png in sorted(out_dir.glob("page-*.png")):
        try:
            num = int(png.stem.split("-", 1)[1])
        except Exception:
            continue
        pages.append((num, png))
    pages.sort(key=lambda x: x[0])
    return pages


def run_tesseract(path: Path, timeout: int = 120) -> str:
    work = path
    tmp: Path | None = None
    try:
        ext = path.suffix.lower()
        if ext in {".heic", ".webp", ".gif", ".tif", ".tiff"}:
            tmp = Path(tempfile.mkstemp(suffix=".png")[1])
            # Prefer sips on macOS; fall back to Pillow
            r = subprocess.run(
                ["sips", "-s", "format", "png", str(path), "--out", str(tmp)],
                capture_output=True,
                timeout=60,
                check=False,
            )
            if r.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
                try:
                    from PIL import Image

                    Image.open(path).convert("RGB").save(tmp)
                except Exception:
                    pass
            if tmp.exists() and tmp.stat().st_size > 0:
                work = tmp
        try:
            from PIL import Image

            im = Image.open(work)
            if im.mode in ("RGBA", "LA", "P"):
                rgb = Image.new("RGB", im.size, (255, 255, 255))
                if im.mode == "P":
                    im = im.convert("RGBA")
                if im.mode in ("RGBA", "LA"):
                    rgb.paste(im, mask=im.split()[-1])
                else:
                    rgb.paste(im)
                if tmp is None:
                    tmp = Path(tempfile.mkstemp(suffix=".png")[1])
                rgb.save(tmp)
                work = tmp
                im.close()
            else:
                im.close()
        except Exception:
            pass
        r = subprocess.run(
            ["tesseract", str(work), "stdout", "-l", "chi_sim+eng"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if r.returncode != 0 and not (r.stdout or "").strip():
            raise RuntimeError(f"tesseract rc={r.returncode}: {(r.stderr or '')[:200]}")
        return (r.stdout or "").strip()
    finally:
        if tmp is not None:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass


def ocr_with_fallback(
    vision_bin: Path | None,
    path: Path,
    vision_text: str | None = None,
    *,
    tesseract_only: bool = False,
) -> tuple[str, str]:
    """Return (text, engine). engine in {vision, tesseract, vision+tesseract, empty}."""
    if tesseract_only or vision_bin is None:
        tt = run_tesseract(path).strip()
        return (tt, "tesseract") if tt else ("", "empty")

    if vision_text is None:
        vision_text = run_vision_image(vision_bin, path)
    vt = (vision_text or "").strip()
    if len(vt) >= VISION_MIN_CHARS:
        return vt, "vision"
    try:
        tt = run_tesseract(path).strip()
    except Exception as e:
        if vt:
            return vt, "vision"
        raise RuntimeError(f"tess fallback failed: {e}") from e
    if len(tt) > len(vt):
        if vt:
            return tt, "vision+tesseract"
        return tt, "tesseract"
    if vt:
        return vt, "vision"
    if tt:
        return tt, "tesseract"
    return "", "empty"


def should_skip_relpath(rel: str, substrs: list[str]) -> bool:
    if not substrs:
        return False
    rp = rel.replace("\\", "/")
    return any(s in rp for s in substrs)


def existing_fps(conn: sqlite3.Connection) -> dict[tuple[str, str], str]:
    return {
        (rp, loc): fp
        for rp, loc, fp in conn.execute(
            "SELECT relpath, locator, fingerprint FROM ocr"
        )
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="OCR index: images + scanned PDF pages under config roots"
    )
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--root", type=Path, action="append", default=None)
    ap.add_argument("--fts-db", type=Path, default=None)
    ap.add_argument("--out-db", type=Path, default=None)
    ap.add_argument(
        "--vision-bin",
        type=Path,
        default=None,
        help="path to Vision OCR CLI (config build.vision_cli)",
    )
    ap.add_argument("--log", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--limit-images", type=int, default=0)
    ap.add_argument("--limit-pdfs", type=int, default=0)
    ap.add_argument("--pdf-page-cap", type=int, default=None)
    ap.add_argument("--pdf-dpi", type=int, default=None)
    ap.add_argument("--pdf-char-thresh", type=int, default=None)
    ap.add_argument("--images-only", action="store_true")
    ap.add_argument("--pdfs-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--commit-every", type=int, default=COMMIT_EVERY)
    ap.add_argument(
        "--pdftoppm-fallback",
        action="store_true",
        help="if Vision PDF path fails, render via pdftoppm then Vision/Tess",
    )
    ap.add_argument(
        "--tesseract-only",
        action="store_true",
        help="skip Vision CLI; use tesseract only (Linux / no macOS Vision)",
    )
    ap.add_argument(
        "--skip-relpath-substr",
        action="append",
        default=None,
        help="skip relpaths containing this substring (repeatable; also comma-list)",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    if not features(cfg)["ocr"]:
        print("SKIP: features.ocr is false in config", flush=True)
        return 0
    b = build_section(cfg)
    roots = resolve_doc_scan_roots(cfg, args.root)
    fts_db = args.fts_db or index_path(cfg, "fts", "fts.sqlite3")
    out_db = args.out_db or index_path(cfg, "ocr", "ocr.sqlite3")
    log_path = args.log or (PROJECT_ROOT / "logs" / "build_ocr.log")
    page_cap = (
        args.pdf_page_cap
        if args.pdf_page_cap is not None
        else int(b.get("pdf_page_cap") or PDF_PAGE_CAP)
    )
    dpi = args.pdf_dpi if args.pdf_dpi is not None else int(b.get("pdf_dpi") or PDF_DPI)
    thresh = (
        args.pdf_char_thresh
        if args.pdf_char_thresh is not None
        else int(b.get("pdf_char_thresh") or PDF_CHAR_THRESH)
    )
    img_exts = set(DEFAULT_IMG_EXTS)
    if b.get("media_image_exts"):
        img_exts = {
            (e if str(e).startswith(".") else f".{e}").lower()
            for e in b["media_image_exts"]
        }
    prefixes = denylist_prefixes(cfg)
    globs = exclude_globs(cfg)

    vision_bin: Path | None = None
    if not args.tesseract_only:
        vb = args.vision_bin or resolve_path(b.get("vision_cli") or "")
        if vb and vb.is_file():
            vision_bin = vb
        elif shutil.which("visionocr"):
            vision_bin = Path(shutil.which("visionocr"))
        else:
            # Prefer tesseract if Vision missing
            if shutil.which("tesseract"):
                args.tesseract_only = True
                print(
                    "# note: vision CLI not found; falling back to --tesseract-only",
                    file=sys.stderr,
                )
            else:
                print(
                    "ERROR: neither vision CLI nor tesseract found.\n"
                    "  Set build.vision_cli in config.json (macOS Vision helper),\n"
                    "  or install tesseract and pass --tesseract-only.\n"
                    "  See docs/deploy.md.",
                    file=sys.stderr,
                )
                return 2

    skip_subs: list[str] = []
    for item in args.skip_relpath_substr or []:
        for part in str(item).split(","):
            part = part.strip()
            if part:
                skip_subs.append(part)
    # Optional skips from config
    for s in b.get("ocr_skip_relpath_substr") or []:
        if s and s not in skip_subs:
            skip_subs.append(str(s))
    args.skip_relpath_substr = skip_subs

    if args.limit:
        if not args.limit_images:
            args.limit_images = args.limit
        if not args.limit_pdfs and not args.images_only:
            args.limit_pdfs = args.limit

    log = setup_log(log_path)
    log.info("=== build_ocr start ===")
    log.info(
        "roots=%s fts=%s out=%s vision=%s tesseract_only=%s page_cap=%d dpi=%d",
        roots,
        fts_db,
        out_db,
        vision_bin,
        args.tesseract_only,
        page_cap,
        dpi,
    )

    if not roots:
        log.error("No document roots. Set roots[] in config.json or pass --root.")
        return 2
    for r in roots:
        if not r.is_dir():
            log.error("Document root missing: %s", r)
            return 2

    out_db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(out_db)
    ensure_schema(conn)
    if args.rebuild:
        n = conn.execute("DELETE FROM ocr").rowcount
        try:
            conn.execute("INSERT INTO ocr_fts(ocr_fts) VALUES('delete-all')")
        except sqlite3.Error:
            conn.execute("DELETE FROM ocr_fts")
        conn.commit()
        log.info("rebuild: deleted %d rows", n)

    have = {} if args.force else existing_fps(conn)
    log.info("already in DB: %d rows (force=%s)", len(have) if not args.force else 0, args.force)

    failures: list[str] = []
    n_ok = n_skip = n_fail = 0
    n_tess = 0
    t0 = time.time()
    pending = 0

    if not args.pdfs_only:
        images = iter_files_under(roots, exts=img_exts, prefixes=prefixes, globs=globs)
        if args.skip_relpath_substr:
            before = len(images)
            images = [
                (r, p)
                for r, p in images
                if not should_skip_relpath(r, args.skip_relpath_substr)
            ]
            skipped_n = before - len(images)
            if skipped_n:
                log.info("skip-relpath-substr dropped %d images", skipped_n)
        if args.limit_images:
            images = images[: args.limit_images]
        log.info("OCR images: %d", len(images))
        for i, (rel, path) in enumerate(images):
            loc = ""
            try:
                fp = file_fingerprint(path)
                if not args.force and have.get((rel, loc)) == fp:
                    n_skip += 1
                    continue
                text, engine = ocr_with_fallback(
                    vision_bin, path, tesseract_only=args.tesseract_only
                )
                if "tesseract" in engine:
                    n_tess += 1
                upsert_ocr(
                    conn,
                    relpath=rel,
                    parent=rel,
                    fp=fp,
                    modality="image",
                    locator=loc,
                    engine=engine,
                    text=text,
                )
                n_ok += 1
                pending += 1
                have[(rel, loc)] = fp
            except Exception as e:
                n_fail += 1
                msg = f"image FAIL {rel}: {e}"
                failures.append(msg)
                log.warning(msg)
            if pending >= args.commit_every:
                conn.commit()
                pending = 0
            if (i + 1) % PROGRESS_EVERY == 0 or i + 1 == len(images):
                elapsed = time.time() - t0
                done = n_ok + n_fail
                rate = done / max(elapsed, 0.01)
                left = len(images) - (i + 1)
                log.info(
                    "img progress %d/%d ok=%d skip=%d fail=%d tess=%d rate=%.2f/s eta=%.0fs",
                    i + 1,
                    len(images),
                    n_ok,
                    n_skip,
                    n_fail,
                    n_tess,
                    rate,
                    left / max(rate, 0.01),
                )
        conn.commit()
        pending = 0

    if not args.images_only:
        pdfs = collect_scanned_pdfs(
            fts_db, roots, log, thresh=thresh, prefixes=prefixes, globs=globs
        )
        if args.skip_relpath_substr:
            before = len(pdfs)
            pdfs = [
                (r, p)
                for r, p in pdfs
                if not should_skip_relpath(r, args.skip_relpath_substr)
            ]
            if before - len(pdfs):
                log.info("skip-relpath-substr dropped %d PDFs", before - len(pdfs))
        if args.limit_pdfs:
            pdfs = pdfs[: args.limit_pdfs]
        log.info("OCR PDF pages from %d PDFs (cap=%d dpi=%d)", len(pdfs), page_cap, dpi)
        tmp_root = Path(tempfile.mkdtemp(prefix="ocr_pdf_"))
        try:
            for pi, (rel, path) in enumerate(pdfs):
                try:
                    n_pages = pdf_page_count(path)
                    cap = page_cap
                    if n_pages > 0:
                        cap = min(cap, n_pages)
                    if not args.force and n_pages > 0:
                        all_cached = True
                        for pg in range(1, cap + 1):
                            loc = f"page:{pg}"
                            fp = page_fingerprint(path, pg)
                            if have.get((rel, loc)) != fp:
                                all_cached = False
                                break
                        if all_cached:
                            n_skip += cap
                            continue

                    pages_text: dict[int, str] = {}
                    vision_pdf_ok = False
                    if not args.tesseract_only and vision_bin is not None:
                        try:
                            pages_text = run_vision_pdf_pages(
                                vision_bin, path, cap, dpi
                            )
                            vision_pdf_ok = True
                        except Exception as e:
                            log.warning("vision PDF batch fail %s: %s", rel, e)
                            if not args.pdftoppm_fallback:
                                for pg in range(1, cap + 1):
                                    loc = f"page:{pg}"
                                    fp = page_fingerprint(path, pg)
                                    if not args.force and have.get((rel, loc)) == fp:
                                        n_skip += 1
                                        continue
                                    try:
                                        vt = run_vision_pdf_page(
                                            vision_bin, path, pg, dpi
                                        )
                                        pages_text[pg] = vt
                                    except Exception as e2:
                                        n_fail += 1
                                        msg = f"pdfpage FAIL {rel} page:{pg}: {e2}"
                                        failures.append(msg)
                                        log.warning(msg)
                                vision_pdf_ok = bool(pages_text)

                    if (not vision_pdf_ok and args.pdftoppm_fallback) or (
                        args.tesseract_only and not vision_pdf_ok
                    ):
                        work = tmp_root / f"pdf_{pi}"
                        if work.exists():
                            shutil.rmtree(work, ignore_errors=True)
                        rendered = render_pdf_pages_pdftoppm(path, work, cap, dpi)
                        for page_num, png in rendered:
                            loc = f"page:{page_num}"
                            fp = page_fingerprint(path, page_num)
                            if not args.force and have.get((rel, loc)) == fp:
                                n_skip += 1
                                continue
                            try:
                                text, engine = ocr_with_fallback(
                                    vision_bin,
                                    png,
                                    tesseract_only=args.tesseract_only,
                                )
                                if "tesseract" in engine:
                                    n_tess += 1
                                upsert_ocr(
                                    conn,
                                    relpath=rel,
                                    parent=rel,
                                    fp=fp,
                                    modality="pdfpage",
                                    locator=loc,
                                    engine=engine,
                                    text=text,
                                )
                                n_ok += 1
                                pending += 1
                                have[(rel, loc)] = fp
                            except Exception as e:
                                n_fail += 1
                                msg = f"pdfpage FAIL {rel} {loc}: {e}"
                                failures.append(msg)
                                log.warning(msg)
                        shutil.rmtree(work, ignore_errors=True)
                    else:
                        page_nums = sorted(pages_text.keys()) or list(range(1, cap + 1))
                        for page_num in page_nums:
                            if page_num > cap:
                                continue
                            loc = f"page:{page_num}"
                            fp = page_fingerprint(path, page_num)
                            if not args.force and have.get((rel, loc)) == fp:
                                n_skip += 1
                                continue
                            vt = pages_text.get(page_num, "")
                            try:
                                if len(vt.strip()) >= VISION_MIN_CHARS:
                                    text, engine = vt.strip(), "vision"
                                else:
                                    work = tmp_root / f"pdf_{pi}_p{page_num}"
                                    if work.exists():
                                        shutil.rmtree(work, ignore_errors=True)
                                    png = None
                                    try:
                                        rendered = render_pdf_pages_pdftoppm(
                                            path,
                                            work,
                                            page_num,
                                            dpi,
                                            first_page=page_num,
                                            timeout=60,
                                        )
                                        for pn, pp in rendered:
                                            if pn == page_num:
                                                png = pp
                                                break
                                        if png is None and rendered:
                                            png = rendered[0][1]
                                    except Exception as render_err:
                                        log.warning(
                                            "pdftoppm single-page skip %s page:%d: %s",
                                            rel,
                                            page_num,
                                            render_err,
                                        )
                                        png = None
                                    if png is not None:
                                        try:
                                            text, engine = ocr_with_fallback(
                                                vision_bin,
                                                png,
                                                vision_text=vt,
                                                tesseract_only=args.tesseract_only,
                                            )
                                            if "tesseract" in engine:
                                                n_tess += 1
                                        except Exception as tess_err:
                                            log.warning(
                                                "tess fallback skip %s page:%d: %s",
                                                rel,
                                                page_num,
                                                tess_err,
                                            )
                                            text, engine = (
                                                vt.strip(),
                                                "vision" if vt.strip() else "empty",
                                            )
                                    else:
                                        text, engine = (
                                            vt.strip(),
                                            "vision" if vt.strip() else "empty",
                                        )
                                    shutil.rmtree(work, ignore_errors=True)
                                upsert_ocr(
                                    conn,
                                    relpath=rel,
                                    parent=rel,
                                    fp=fp,
                                    modality="pdfpage",
                                    locator=loc,
                                    engine=engine,
                                    text=text,
                                )
                                n_ok += 1
                                pending += 1
                                have[(rel, loc)] = fp
                            except Exception as e:
                                n_fail += 1
                                msg = f"pdfpage FAIL {rel} {loc}: {e}"
                                failures.append(msg)
                                log.warning(msg)
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
                        "pdf progress %d/%d ok=%d skip=%d fail=%d tess=%d elapsed=%.0fs",
                        pi + 1,
                        len(pdfs),
                        n_ok,
                        n_skip,
                        n_fail,
                        n_tess,
                        elapsed,
                    )
            conn.commit()
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

    by_mod = dict(
        conn.execute("SELECT modality, COUNT(*) FROM ocr GROUP BY modality").fetchall()
    )
    by_eng = dict(
        conn.execute("SELECT engine, COUNT(*) FROM ocr GROUP BY engine").fetchall()
    )
    total = conn.execute("SELECT COUNT(*) FROM ocr").fetchone()[0]
    conn.close()
    wall = time.time() - t0
    log.info(
        "DONE ok=%d skip=%d fail=%d tess_fallback=%d total_rows=%d "
        "by_modality=%s by_engine=%s wall_s=%.1f db=%s",
        n_ok,
        n_skip,
        n_fail,
        n_tess,
        total,
        by_mod,
        by_eng,
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
