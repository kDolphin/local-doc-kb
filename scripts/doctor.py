#!/usr/bin/env python3
"""local-doc-kb doctor: PASS/WARN/FAIL checks + next-step hints.

Target platform: Apple Silicon (arm64) macOS. On other platforms the platform
checks report FAIL/WARN, but FTS-only checks still run (useful for CI).

Usage: doctor.py [--config PATH] [--no-network] [--json]
Exit: 0 if no FAIL, 1 otherwise.  Never sends document content anywhere.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_config import PROJECT_ROOT, document_roots, index_path, load_config  # noqa: E402
from lib_providers import (  # noqa: E402
    emb_conf, features, is_remote, llm_conf, provider_health, read_index_meta,
)

rows: list[tuple[str, str, str, str]] = []


def add(status, name, detail, hint=""):
    rows.append((status, name, detail, hint))


def have_mod(m):
    return importlib.util.find_spec(m) is not None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--no-network", action="store_true", help="skip provider reachability")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    cfg_path = a.config or PROJECT_ROOT / "config.json"
    cfg = load_config(a.config)
    feat = features(cfg)
    is_mac = sys.platform == "darwin"

    # --- platform
    mach = platform.machine()
    if is_mac and mach == "arm64":
        add("PASS", "platform", f"macOS {platform.mac_ver()[0]} arm64")
    else:
        add("FAIL" if not feat_only_fts(feat) else "WARN", "platform",
            f"{sys.platform}/{mach}", "Supported target is Apple Silicon macOS; FTS-only may still work")
    if is_mac:
        add("PASS" if shutil.which("brew") else "FAIL", "homebrew", shutil.which("brew") or "missing",
            "install from https://brew.sh")

    # --- python / venv
    v = sys.version_info
    add("PASS" if v >= (3, 10) else "FAIL", "python", platform.python_version(), "need Python >= 3.10")
    in_venv = sys.prefix != sys.base_prefix
    add("PASS" if in_venv else "WARN", "venv", sys.prefix, "run scripts/bootstrap.sh then source .venv/bin/activate")
    try:
        sqlite3.connect(":memory:").execute("CREATE VIRTUAL TABLE t USING fts5(x)")
        add("PASS", "sqlite fts5", sqlite3.sqlite_version)
    except Exception as e:
        add("FAIL", "sqlite fts5", str(e), "use a Python build with FTS5 (python.org or Homebrew)")

    # --- config
    if cfg_path.exists() and cfg:
        add("PASS", "config", str(cfg_path))
    else:
        add("FAIL", "config", f"missing/invalid {cfg_path}", "cp config.example.json config.json and edit roots[]")
    roots = document_roots(cfg)
    bad = [str(r) for r in roots if not r.is_dir()]
    if not roots:
        add("FAIL", "roots", "none configured", "ask the user which folders to index, set roots[]")
    elif bad:
        add("FAIL", "roots", f"not a directory: {bad}")
    else:
        add("PASS", "roots", f"{len(roots)} root(s)")
    add("PASS", "features", ", ".join(f"{k}={'on' if v_ else 'off'}" for k, v_ in feat.items()))

    # --- packages
    need = []
    if feat["vectors"] or feat["desc"]:
        if emb_conf(cfg)["type"] == "sentence_transformers":
            need += ["sentence_transformers", "torch"]
    if feat["media"]:
        need += ["torch", "torchvision", "PIL"]
    for m in dict.fromkeys(need):
        add("PASS" if have_mod(m) else "FAIL", f"pkg {m}", "importable" if have_mod(m) else "missing",
            "scripts/bootstrap.sh --tier build (or --tier media)")
    add("PASS" if have_mod("xlrd") else "WARN", "pkg xlrd", "for legacy .xls", "pip install xlrd")
    if "torch" in need and have_mod("torch"):
        import torch
        mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        add("PASS" if mps else "WARN", "torch MPS", f"torch {torch.__version__} mps={mps}",
            "embeddings will run on CPU (slow) without MPS")

    # --- system tools
    tools = [("pdftotext", True, "brew install poppler"), ("pdftoppm", feat["ocr"], "brew install poppler")]
    if is_mac:
        tools.append(("textutil", True, "built into macOS (.doc extraction)"))
    for t, required, hint in tools:
        p = shutil.which(t)
        add("PASS" if p else ("FAIL" if required else "WARN"), f"tool {t}", p or "missing", hint)
    if feat["ocr"]:
        vcli = (cfg.get("build") or {}).get("vision_cli") or ""
        vp = (PROJECT_ROOT / vcli) if vcli and not Path(vcli).is_absolute() else Path(vcli or "/nonexistent")
        tess = shutil.which("tesseract")
        add("PASS" if vp.exists() else "WARN", "ocr vision_cli", str(vp), "optional Apple Vision helper; else tesseract")
        if tess:
            langs = subprocess.run([tess, "--list-langs"], capture_output=True, text=True).stdout
            ok = "chi_sim" in langs
            add("PASS" if ok else "WARN", "tesseract chi_sim", "present" if ok else "missing",
                "brew install tesseract-lang")
        elif not vp.exists():
            add("FAIL", "ocr backend", "neither Vision CLI nor tesseract", "brew install tesseract tesseract-lang")

    # --- providers + remote gate
    provs = []
    if feat["vectors"] or feat["desc"]:
        provs.append(("embedding", emb_conf(cfg)))
    if feat["desc"]:
        provs.append(("llm", llm_conf(cfg)))
    for role, p in provs:
        if is_remote(p):
            ok = bool(cfg.get("allow_remote"))
            add("WARN" if ok else "FAIL", f"remote {role}",
                f"{p['type']} {p.get('base_url')} — DOCUMENT CONTENT LEAVES THIS MACHINE",
                "needs allow_remote:true + --i-understand-remote; confirm with user")
        if a.no_network and p["type"] != "sentence_transformers":
            add("WARN", f"provider {role}", f"{p['type']} (network check skipped)")
            continue
        ok, msg = provider_health(p)
        add("PASS" if ok else "FAIL", f"provider {role}", f"{p['type']}:{p['model']} {msg}",
            "scripts/llm_lifecycle.sh <provider> start" if role == "llm" else "")

    # --- disk
    du = shutil.disk_usage(PROJECT_ROOT)
    free_gb = du.free / 1e9
    add("PASS" if free_gb > 20 else ("WARN" if free_gb > 5 else "FAIL"), "disk free", f"{free_gb:.1f} GB",
        "index ≈ 1-2x extracted text; models 0.1-20 GB")

    # --- indexes
    fts = index_path(cfg, "fts", "fts.sqlite3")
    if fts.exists():
        n = sqlite3.connect(f"file:{fts}?mode=ro", uri=True).execute("SELECT COUNT(*) FROM docs").fetchone()[0]
        add("PASS", "index fts", f"{n} docs")
    else:
        add("WARN", "index fts", "not built yet", "python3 scripts/build_fts.py --limit 50 (smoke)")
    if feat["vectors"] or feat["desc"]:
        want = emb_conf(cfg)
        want_id = want["model"] if want["type"] == "sentence_transformers" else f"{want['type']}:{want['model']}"
        for key, name in (("vectors", "vectors.sqlite3"), ("desc_vectors", "desc_vectors.sqlite3")):
            p = index_path(cfg, key, name)
            if not p.exists():
                add("WARN", f"index {key}", "not built")
                continue
            meta = read_index_meta(sqlite3.connect(f"file:{p}?mode=ro", uri=True))
            got = meta.get("embedding_model")
            if not got:
                add("WARN", f"index {key}", "no index_meta (legacy); rebuild recommended")
            elif got != want_id:
                add("FAIL", f"index {key}", f"built with {got} dim={meta.get('embedding_dim')}, config wants {want_id}",
                    "rebuild vectors with --rebuild (ask user) or revert config")
            else:
                add("PASS", f"index {key}", f"{got} dim={meta.get('embedding_dim')}")

    if a.json:
        print(json.dumps([dict(zip(("status", "check", "detail", "hint"), r)) for r in rows], ensure_ascii=False, indent=1))
    else:
        w = max(len(r[1]) for r in rows)
        for s, n, d, h in rows:
            print(f"{s:<5} {n:<{w}}  {d}" + (f"\n{'':6}{'':<{w}}  -> {h}" if h and s != "PASS" else ""))
        nf = sum(r[0] == "FAIL" for r in rows)
        nw = sum(r[0] == "WARN" for r in rows)
        print(f"\nsummary: {len(rows)-nf-nw} PASS, {nw} WARN, {nf} FAIL")
        if nf:
            print("next: fix FAIL rows top to bottom, then re-run doctor (see AGENTS.md).")
    return 1 if any(r[0] == "FAIL" for r in rows) else 0


def feat_only_fts(feat):
    return not any(feat[k] for k in ("vectors", "desc", "media", "ocr"))


if __name__ == "__main__":
    raise SystemExit(main())
