#!/usr/bin/env python3
"""Shared config loading and path helpers for local-doc-kb (ask + build)."""
from __future__ import annotations

import json
import os
import os
import sys
from fnmatch import fnmatch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ALLOW_EXTENSIONS = {
    ".md", ".txt", ".csv", ".html", ".htm", ".json", ".log",
    ".pdf", ".docx", ".doc", ".xlsx", ".xls", ".pptx", ".rtf",
}

DEFAULT_IMG_EXTS = {
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff", ".heic", ".webp",
}


def load_config(path: Path | None = None) -> dict:
    """Load config.json; missing file → empty dict (callers use defaults)."""
    cfg_path = path or Path(
        os.environ.get("LOCAL_DOC_KB_CONFIG", PROJECT_ROOT / "config.json")
    )
    if not cfg_path.exists():
        return {}
    try:
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"# warning: could not parse config {cfg_path}: {e}", file=sys.stderr)
        return {}


def resolve_path(raw: str | Path | None, *, base: Path | None = None) -> Path | None:
    """Resolve absolute or project-relative path. None/empty → None."""
    if raw is None or raw == "":
        return None
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p
    return (base or PROJECT_ROOT) / p


def index_path(cfg: dict, key: str, default_name: str) -> Path:
    """Resolve an index DB path from config.index.* or default under index/."""
    idx = cfg.get("index") or {}
    raw = idx.get(key)
    if not raw:
        return PROJECT_ROOT / "index" / default_name
    p = Path(raw).expanduser()
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def embedding_setting(cfg: dict, *keys: str, default: str = "") -> str:
    emb = cfg.get("embeddings") or {}
    for k in keys:
        v = emb.get(k)
        if v:
            return str(v)
    return default


def build_section(cfg: dict) -> dict:
    return cfg.get("build") or {}


def hosts_section(cfg: dict) -> dict:
    return cfg.get("hosts") or {}


def allow_extensions(cfg: dict) -> set[str]:
    raw = cfg.get("allow_extensions")
    if not raw:
        return set(DEFAULT_ALLOW_EXTENSIONS)
    out: set[str] = set()
    for e in raw:
        e = str(e).strip().lower()
        if not e:
            continue
        out.add(e if e.startswith(".") else f".{e}")
    return out or set(DEFAULT_ALLOW_EXTENSIONS)


def denylist_prefixes(cfg: dict) -> list[str]:
    """Path-segment prefixes to skip (e.g. '/.git' → skip any '.git' component)."""
    raw = cfg.get("denylist_path_prefixes") or []
    out: list[str] = []
    for p in raw:
        s = str(p).strip().replace("\\", "/")
        if not s:
            continue
        # Normalize to bare segment names where possible
        s = s.strip("/")
        if s:
            out.append(s)
    return out


def exclude_globs(cfg: dict) -> list[str]:
    b = build_section(cfg)
    return [str(g) for g in (b.get("exclude_path_globs") or []) if g]


def document_roots(cfg: dict) -> list[Path]:
    """Configured document roots (absolute paths). Empty if unset."""
    roots: list[Path] = []
    for r in cfg.get("roots") or []:
        p = resolve_path(r)
        if p is not None:
            roots.append(p)
    return roots


def raw_dir(cfg: dict) -> Path:
    b = build_section(cfg)
    return resolve_path(b.get("raw_dir") or "raw/") or (PROJECT_ROOT / "raw")


def is_path_denied(relpath: str, prefixes: list[str], globs: list[str] | None = None) -> bool:
    """True if relpath should be skipped by denylist segments or exclude globs."""
    rp = relpath.replace("\\", "/").lstrip("./")
    parts = [p for p in rp.split("/") if p]
    for pref in prefixes:
        pref_clean = pref.strip("/")
        if not pref_clean:
            continue
        # Match as a path prefix or as any path component
        if rp == pref_clean or rp.startswith(pref_clean + "/"):
            return True
        if pref_clean in parts:
            return True
    for g in globs or []:
        if fnmatch(rp, g) or fnmatch("/" + rp, g):
            return True
    return False


def relpath_for(root: Path, file_path: Path, *, multi_root: bool, roots: list[Path]) -> str:
    """Stable relative path for indexing; basename-prefix when multiple roots."""
    try:
        rel = file_path.relative_to(root).as_posix()
    except ValueError:
        rel = file_path.as_posix()
    if multi_root and len(roots) > 1:
        label = root.name or "root"
        return f"{label}/{rel}"
    return rel


def iter_files_under(
    roots: list[Path],
    *,
    exts: set[str] | None = None,
    prefixes: list[str] | None = None,
    globs: list[str] | None = None,
) -> list[tuple[str, Path]]:
    """Walk roots; return sorted (relpath, absolute Path) for matching files."""
    prefixes = prefixes or []
    globs = globs or []
    multi = len(roots) > 1
    out: list[tuple[str, Path]] = []
    for root in roots:
        if not root.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            rel_dir = Path(dirpath).relative_to(root).as_posix()
            if rel_dir == ".":
                rel_dir = ""
            # Prune denied directories
            keep_dirs = []
            for d in dirnames:
                cand = f"{rel_dir}/{d}" if rel_dir else d
                if multi:
                    cand = f"{root.name}/{cand}"
                if is_path_denied(cand, prefixes, globs):
                    continue
                keep_dirs.append(d)
            dirnames[:] = keep_dirs
            for name in filenames:
                p = Path(dirpath) / name
                if exts is not None and p.suffix.lower() not in exts:
                    continue
                rel = relpath_for(root, p, multi_root=multi, roots=roots)
                if is_path_denied(rel, prefixes, globs):
                    continue
                out.append((rel, p))
    out.sort(key=lambda x: x[0])
    return out


def require_config_or_warn(cfg: dict, *, need_roots: bool = False) -> int | None:
    """Return exit code 2 with message if config/roots missing when needed; else None."""
    if need_roots and not document_roots(cfg):
        raw = raw_dir(cfg)
        if raw.is_dir() and any(raw.iterdir()):
            return None
        print(
            "ERROR: no document roots configured.\n"
            "  Copy config.example.json → config.json and set roots[] to your\n"
            "  document directories (or place files under build.raw_dir).\n"
            "  See docs/deploy.md.",
            file=sys.stderr,
        )
        return 2
    return None


def resolve_doc_scan_roots(cfg: dict, cli_roots: list[Path] | None = None) -> list[Path]:
    """CLI roots override config.roots; else config.roots; else raw_dir if it exists."""
    if cli_roots:
        return [p.expanduser().resolve() for p in cli_roots]
    roots = [p.resolve() for p in document_roots(cfg)]
    if roots:
        return roots
    raw = raw_dir(cfg)
    if raw.is_dir():
        return [raw.resolve()]
    return []
