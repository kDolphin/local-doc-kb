#!/usr/bin/env bash
# Bootstrap local-doc-kb on Apple Silicon macOS (arm64). Idempotent.
#
# Usage: scripts/bootstrap.sh [--tier fts|query|build|media|ocr|full] [--brew] [--dry-run]
#                             [--dev-unsupported]
#   --tier   fts   : venv only (stdlib FTS build + ask)          [default: build]
#            query : + sentence-transformers/torch (MPS)
#            build : query + xlrd   (+ poppler via --brew)
#            media : build + pillow/torchvision
#            ocr   : build + tesseract/tesseract-lang via --brew (Vision CLI optional)
#            full  : media + ocr
#   --brew   actually run `brew install` for missing system tools (otherwise print commands)
#   --dry-run  print what would happen
#   --dev-unsupported  allow non-arm64-macOS hosts (CI/testing only; unsupported)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TIER=build; BREW=0; DRY=0; DEV=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tier) TIER="$2"; shift 2 ;;
    --brew) BREW=1; shift ;;
    --dry-run) DRY=1; shift ;;
    --dev-unsupported) DEV=1; shift ;;
    -h|--help) sed -n 2,16p "$0"; exit 0 ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
done
run() { echo "+ $*"; [[ $DRY == 1 ]] || "$@"; }

OS="$(uname -s)"; ARCH="$(uname -m)"
if [[ "$OS" != "Darwin" || "$ARCH" != "arm64" ]]; then
  if [[ $DEV == 1 ]]; then
    echo "WARNING: $OS/$ARCH is unsupported (target: Apple Silicon macOS). Continuing (--dev-unsupported)."
  else
    echo "ERROR: local-doc-kb supports Apple Silicon macOS (arm64) only; got $OS/$ARCH." >&2
    echo "       Use --dev-unsupported for CI/testing (FTS-only is expected to work)." >&2
    exit 1
  fi
fi

PY="${PYTHON:-python3}"
"$PY" -c 'import sys; assert sys.version_info>=(3,10), sys.version' || { echo "need Python >= 3.10 (brew install python@3.12)" >&2; exit 1; }
[[ -d "$ROOT/.venv" ]] || run "$PY" -m venv "$ROOT/.venv"
VPY="$ROOT/.venv/bin/python"
[[ $DRY == 1 && ! -x "$VPY" ]] && VPY="$PY"
run "$VPY" -m pip install -q --upgrade pip

case "$TIER" in
  fts) REQ="" ;;
  query) REQ=requirements-query.txt ;;
  build|ocr) REQ=requirements-build.txt ;;
  media|full) REQ=requirements-media.txt ;;
  *) echo "bad tier $TIER" >&2; exit 2 ;;
esac
if [[ -n "$REQ" ]]; then
  # Apple Silicon: default PyPI torch wheels include MPS. Nothing special needed.
  run "$VPY" -m pip install -r "$ROOT/$REQ"
  if [[ $DRY == 0 && "$OS" == "Darwin" ]]; then
    "$VPY" -c 'import torch; print("torch", torch.__version__, "mps", torch.backends.mps.is_available())' || true
  fi
fi

# System tools (Homebrew)
want=(poppler)
[[ "$TIER" == ocr || "$TIER" == full ]] && want+=(tesseract tesseract-lang)
missing=()
command -v pdftotext >/dev/null || missing+=(poppler)
if [[ " ${want[*]} " == *" tesseract "* ]]; then
  command -v tesseract >/dev/null || missing+=(tesseract tesseract-lang)
  if command -v tesseract >/dev/null && ! tesseract --list-langs 2>/dev/null | grep -qx chi_sim; then missing+=(tesseract-lang); fi
fi
if [[ "$OS" == "Darwin" ]]; then
  command -v textutil >/dev/null && echo "textutil: present (.doc extraction)" || echo "WARN: textutil missing?!"
  command -v brew >/dev/null || echo "WARN: Homebrew missing — install from https://brew.sh"
fi
if [[ ${#missing[@]} -gt 0 ]]; then
  if [[ $BREW == 1 && "$OS" == "Darwin" ]]; then run brew install "${missing[@]}"
  else echo "missing system tools; run:  brew install ${missing[*]}"; fi
fi
if [[ "$TIER" == ocr || "$TIER" == full ]]; then
  echo "OCR: Apple Vision CLI is primary (set build.vision_cli to a helper binary you build/trust);"
  echo "     tesseract (+chi_sim) is the fallback. See docs/operations.md."
fi
echo
echo "next: source .venv/bin/activate && python scripts/doctor.py"
