#!/usr/bin/env python3
"""Build SQLite FTS5 index from configured document roots.

Paths, allow-list extensions, and denylist come from config.json
(see config.example.json). Stdlib + optional pdftotext / textutil / xlrd.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import re
import sqlite3
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_config import (  # noqa: E402
    allow_extensions,
    denylist_prefixes,
    exclude_globs,
    index_path,
    is_path_denied,
    iter_files_under,
    load_config,
    resolve_doc_scan_roots,
)

TZ = ZoneInfo("Asia/Shanghai")
_NS_RE = re.compile(r"\{[^}]+\}")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _local(tag: str) -> str:
    return _NS_RE.sub("", tag)


def _xml_texts(data: bytes, local_names: set[str]) -> list[str]:
    out: list[str] = []
    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        raw = data.decode("utf-8", errors="replace")
        return [html.unescape(m) for m in re.findall(r"<t[^>]*>(.*?)</t>", raw, flags=re.S)]
    for el in root.iter():
        if _local(el.tag) in local_names and el.text:
            out.append(html.unescape(el.text))
    return out


def _extract_xlsx(p: Path) -> str:
    parts: list[str] = []
    try:
        with zipfile.ZipFile(p) as z:
            names = set(z.namelist())
            if "xl/workbook.xml" in names:
                parts.extend(_xml_texts(z.read("xl/workbook.xml"), {"name"}))
            if "xl/sharedStrings.xml" in names:
                parts.extend(_xml_texts(z.read("xl/sharedStrings.xml"), {"t"}))
            for name in sorted(
                n for n in names if n.startswith("xl/worksheets/sheet") and n.endswith(".xml")
            ):
                data = z.read(name)
                parts.extend(_xml_texts(data, {"t"}))
                try:
                    root = ET.fromstring(data)
                    for el in root.iter():
                        if _local(el.tag) != "c":
                            continue
                        if el.attrib.get("t") == "s":
                            continue
                        for child in el:
                            if _local(child.tag) == "v" and child.text:
                                parts.append(child.text)
                except ET.ParseError:
                    pass
    except Exception:
        return ""
    return "\n".join(x.strip() for x in parts if x and x.strip())


def _extract_pptx(p: Path) -> str:
    parts: list[str] = []
    try:
        with zipfile.ZipFile(p) as z:
            slides = sorted(
                n for n in z.namelist()
                if n.startswith("ppt/slides/slide") and n.endswith(".xml")
            )
            for name in slides:
                parts.extend(_xml_texts(z.read(name), {"t"}))
    except Exception:
        return ""
    return "\n".join(x.strip() for x in parts if x and x.strip())


def _extract_doc(p: Path) -> str:
    """Legacy .doc: macOS textutil -> antiword -> libreoffice --headless; '' if none."""
    import shutil
    import tempfile
    cmds = []
    if shutil.which("textutil"):
        cmds.append(["textutil", "-convert", "txt", "-stdout", str(p)])
    if shutil.which("antiword"):
        cmds.append(["antiword", str(p)])
    for argv in cmds:
        try:
            out = subprocess.check_output(argv, stderr=subprocess.DEVNULL, timeout=120)
            txt = out.decode("utf-8", errors="replace")
            if txt.strip():
                return txt
        except Exception:
            pass
    lo = shutil.which("soffice") or shutil.which("libreoffice")
    if lo:
        try:
            with tempfile.TemporaryDirectory() as td:
                subprocess.run([lo, "--headless", "--convert-to", "txt:Text", "--outdir", td, str(p)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180)
                outs = list(Path(td).glob("*.txt"))
                if outs:
                    return outs[0].read_text(encoding="utf-8", errors="replace")
        except Exception:
            pass
    return ""


def _extract_xlsx(p: Path) -> str:
    parts: list[str] = []
    try:
        with zipfile.ZipFile(p) as z:
            names = set(z.namelist())
            if "xl/workbook.xml" in names:
                parts.extend(_xml_texts(z.read("xl/workbook.xml"), {"name"}))
            if "xl/sharedStrings.xml" in names:
                parts.extend(_xml_texts(z.read("xl/sharedStrings.xml"), {"t"}))
            for name in sorted(
                n for n in names if n.startswith("xl/worksheets/sheet") and n.endswith(".xml")
            ):
                data = z.read(name)
                parts.extend(_xml_texts(data, {"t"}))
                try:
                    root = ET.fromstring(data)
                    for el in root.iter():
                        if _local(el.tag) != "c":
                            continue
                        if el.attrib.get("t") == "s":
                            continue
                        for child in el:
                            if _local(child.tag) == "v" and child.text:
                                parts.append(child.text)
                except ET.ParseError:
                    pass
    except Exception:
        return ""
    return "\n".join(x.strip() for x in parts if x and x.strip())


def _extract_pptx(p: Path) -> str:
    parts: list[str] = []
    try:
        with zipfile.ZipFile(p) as z:
            slides = sorted(
                n for n in z.namelist()
                if n.startswith("ppt/slides/slide") and n.endswith(".xml")
            )
            for name in slides:
                parts.extend(_xml_texts(z.read(name), {"t"}))
    except Exception:
        return ""
    return "\n".join(x.strip() for x in parts if x and x.strip())


def _extract_doc(p: Path) -> str:
    """Legacy .doc via macOS textutil; empty where unavailable."""
    try:
        out = subprocess.check_output(
            ["textutil", "-convert", "txt", "-stdout", str(p)],
            stderr=subprocess.DEVNULL,
            timeout=120,
        )
        return out.decode("utf-8", errors="replace")
    except Exception:
        return ""


def _extract_xls(p: Path) -> str:
    """Legacy .xls via xlrd (optional); empty when missing or unreadable."""
    try:
        import xlrd  # type: ignore

        book = xlrd.open_workbook(str(p), on_demand=True)
        parts: list[str] = []
        for sh in book.sheets():
            parts.append(sh.name)
            for r in range(sh.nrows):
                cells = [str(v).strip() for v in sh.row_values(r) if str(v).strip()]
                if cells:
                    parts.append(" ".join(cells))
        return "\n".join(parts)
    except Exception:
        return ""


def extract(p: Path) -> str:
    ext = p.suffix.lower()
    if ext in {".md", ".txt", ".csv", ".html", ".htm", ".json", ".log", ".rtf"}:
        return p.read_text(encoding="utf-8", errors="replace")
    if ext == ".pdf":
        try:
            out = subprocess.check_output(
                ["pdftotext", "-layout", str(p), "-"],
                stderr=subprocess.DEVNULL,
            )
            return out.decode("utf-8", errors="replace")
        except Exception:
            return ""
    if ext == ".docx":
        try:
            with zipfile.ZipFile(p) as z:
                xml = z.read("word/document.xml")
            return "\n".join(_xml_texts(xml, {"t"}))
        except Exception:
            return ""
    if ext == ".xlsx":
        return _extract_xlsx(p)
    if ext == ".pptx":
        return _extract_pptx(p)
    if ext == ".doc":
        return _extract_doc(p)
    if ext == ".xls":
        return _extract_xls(p)
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build FTS5 index from config roots (or --root / raw_dir)"
    )
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument(
        "--root",
        type=Path,
        action="append",
        default=None,
        help="document root (repeatable; overrides config.roots)",
    )
    ap.add_argument("--db", type=Path, default=None, help="output fts sqlite")
    ap.add_argument("--limit", type=int, default=0, help="max new/changed files (0=all)")
    ap.add_argument("--force", action="store_true", help="re-extract even when sha256 unchanged")
    ap.add_argument(
        "--force-ext",
        default="",
        help="comma list of extensions to re-extract regardless of sha256, e.g. .doc,.xls",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    roots = resolve_doc_scan_roots(cfg, args.root)
    if not roots:
        print(
            "ERROR: no document roots. Set roots[] in config.json or pass --root PATH.\n"
            "See config.example.json and docs/deploy.md.",
            file=sys.stderr,
        )
        return 2

    db = args.db or index_path(cfg, "fts", "fts.sqlite3")
    allow = allow_extensions(cfg)
    prefixes = denylist_prefixes(cfg)
    globs = exclude_globs(cfg)

    force_ext = {
        e.strip().lower() if e.strip().startswith(".") else "." + e.strip().lower()
        for e in args.force_ext.split(",")
        if e.strip()
    }

    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS docs (
        path TEXT PRIMARY KEY,
        relpath TEXT,
        sha256 TEXT,
        size INTEGER,
        mtime REAL,
        extracted_at TEXT,
        chars INTEGER,
        text TEXT
    )"""
    )
    conn.execute(
        """CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(
        relpath, text, content='docs', content_rowid='rowid'
    )"""
    )
    conn.executescript(
        """
    CREATE TRIGGER IF NOT EXISTS docs_ai AFTER INSERT ON docs BEGIN
      INSERT INTO docs_fts(rowid, relpath, text) VALUES (new.rowid, new.relpath, new.text);
    END;
    CREATE TRIGGER IF NOT EXISTS docs_ad AFTER DELETE ON docs BEGIN
      INSERT INTO docs_fts(docs_fts, rowid, relpath, text) VALUES('delete', old.rowid, old.relpath, old.text);
    END;
    CREATE TRIGGER IF NOT EXISTS docs_au AFTER UPDATE ON docs BEGIN
      INSERT INTO docs_fts(docs_fts, rowid, relpath, text) VALUES('delete', old.rowid, old.relpath, old.text);
      INSERT INTO docs_fts(rowid, relpath, text) VALUES (new.rowid, new.relpath, new.text);
    END;
    """
    )

    existing = {r[0]: r[1] for r in conn.execute("SELECT path, sha256 FROM docs")}
    files = iter_files_under(roots, exts=allow, prefixes=prefixes, globs=globs)

    # Purge rows whose relpath now matches denylist/globs
    for (path_key, sha) in list(existing.items()):
        row = conn.execute("SELECT relpath FROM docs WHERE path=?", (path_key,)).fetchone()
        if row and is_path_denied(row[0], prefixes, globs):
            conn.execute("DELETE FROM docs WHERE path=?", (path_key,))

    n_new = n_skip = n_empty = 0
    for rel, p in files:
        if args.limit and n_new >= args.limit:
            break
        key = str(p.resolve())
        digest = sha256(p)
        if not args.force and p.suffix.lower() not in force_ext and existing.get(key) == digest:
            n_skip += 1
            continue
        text = extract(p)
        if not text.strip():
            n_empty += 1
        now = datetime.now(TZ).strftime("%Y-%m-%d %H:%M Asia/Shanghai")
        conn.execute(
            """INSERT INTO docs(path, relpath, sha256, size, mtime, extracted_at, chars, text)
               VALUES(?,?,?,?,?,?,?,?)
               ON CONFLICT(path) DO UPDATE SET
                 relpath=excluded.relpath,
                 sha256=excluded.sha256, size=excluded.size, mtime=excluded.mtime,
                 extracted_at=excluded.extracted_at, chars=excluded.chars, text=excluded.text
            """,
            (key, rel, digest, p.stat().st_size, p.stat().st_mtime, now, len(text), text),
        )
        n_new += 1

    conn.commit()
    total = conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
    with_text = conn.execute("SELECT COUNT(*) FROM docs WHERE chars>0").fetchone()[0]
    conn.close()
    print(
        f"indexed new/changed={n_new} skipped_unchanged={n_skip} empty_extract={n_empty} "
        f"total={total} with_text={with_text} roots={len(roots)} db={db}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
