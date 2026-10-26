#!/usr/bin/env bash
# Restore / verify query-host media env (pillow, torchvision, EG2 HF cache).
# Usage:
#   restore_query_media_env.sh             # install deps + check + optional smoke
#   restore_query_media_env.sh --check     # check only
#   restore_query_media_env.sh --unpack-eg2 /path/to/eg2-hf-bundle.tgz
#   restore_query_media_env.sh --smoke     # force smoke test (needs indexes)
# Prefer mirroring weights from the build host over re-downloading.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/.venv"
PY="$VENV/bin/python"
HUB="${HF_HOME:-$HOME/.cache/huggingface}/hub"
EG2_MODEL_DIR="${EG2_MODEL_DIR:-models--google--embeddinggemma-2}"
EG2_SNAP_GLOB="$HUB/${EG2_MODEL_DIR}/snapshots/*"
MEDIA_DB="$ROOT/index/media_vectors_eg2.sqlite3"
# Override TORCH_INDEX for CUDA/CPU wheels as needed (empty = default PyPI)
TORCH_INDEX="${TORCH_INDEX:-}"

CHECK_ONLY=0
SMOKE=0
UNPACK=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --check) CHECK_ONLY=1; shift ;;
    --smoke) SMOKE=1; shift ;;
    --unpack-eg2) UNPACK="$2"; shift 2 ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

die() { echo "FAIL: $*" >&2; exit 1; }
ok() { echo "OK: $*"; }

[[ -x "$PY" ]] || die "missing venv python: $PY (create with: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt)"

if [[ -n "$UNPACK" ]]; then
  [[ -f "$UNPACK" ]] || die "bundle not found: $UNPACK"
  mkdir -p "$HUB"
  tmp=$(mktemp -d)
  tar xzf "$UNPACK" -C "$tmp"
  if [[ -d "$tmp/${EG2_MODEL_DIR}" ]]; then
    rsync -a "$tmp/${EG2_MODEL_DIR}/" "$HUB/${EG2_MODEL_DIR}/"
  fi
  if [[ -d "$tmp/blobs" ]]; then
    mkdir -p "$HUB/blobs"
    rsync -a "$tmp/blobs/" "$HUB/blobs/"
  fi
  ok "unpacked EG2 bundle into $HUB"
  rm -rf "$tmp"
fi

ensure_pip() {
  if ! "$PY" -m pip --version >/dev/null 2>&1; then
    "$PY" -m ensurepip --upgrade
  fi
}

install_deps() {
  ensure_pip
  "$PY" -m pip install -q 'pillow>=10'
  if [[ -n "$TORCH_INDEX" ]]; then
    "$PY" -m pip install -q 'torchvision' --index-url "$TORCH_INDEX"
  else
    "$PY" -m pip install -q 'torchvision' || true
  fi
}

check_py() {
  "$PY" - <<'PY'
import importlib, sys
def need(m):
    try:
        mod = importlib.import_module(m)
        ver = getattr(mod, "__version__", "?")
        print(f"OK: {m} {ver}")
    except Exception as e:
        print(f"FAIL: {m}: {e}")
        sys.exit(1)
need("PIL")
need("torch")
need("torchvision")
PY
}

check_media_db() {
  if [[ ! -f "$MEDIA_DB" ]]; then
    echo "WARN: missing media db: $MEDIA_DB (mirror indexes first)"
    return 0
  fi
  n=$("$PY" - <<PY
import sqlite3
c=sqlite3.connect("$MEDIA_DB")
print(c.execute("select count(*) from media").fetchone()[0])
PY
)
  [[ "$n" -gt 0 ]] || die "media table empty"
  ok "media_vectors_eg2 rows=$n"
}

check_eg2() {
  shopt -s nullglob
  snaps=($EG2_SNAP_GLOB)
  [[ ${#snaps[@]} -gt 0 ]] || die "no EG2 snapshot under $HUB/${EG2_MODEL_DIR} (mirror from build host)"
  snap="${snaps[0]}"
  [[ -e "$snap/model.safetensors" ]] || die "missing model.safetensors in $snap"
  sz=$("$PY" - <<PY
from pathlib import Path
p=Path("$snap/model.safetensors").resolve()
print(p.stat().st_size if p.exists() else 0)
PY
)
  [[ "$sz" -gt 100000000 ]] || die "model.safetensors too small ($sz); remirror from build host"
  pool="$snap/1_Pooling/config.json"
  [[ -f "$pool" ]] || die "missing $pool (Pooling embedding_dimension)"
  grep -q embedding_dimension "$pool" || die "Pooling config lacks embedding_dimension"
  [[ -f "$snap/2_Normalize/config.json" ]] || die "missing 2_Normalize/config.json"
  ok "EG2 snapshot $snap (model bytes=$sz, pooling ok)"
}

smoke() {
  local q="${SMOKE_QUERY:-test query}"
  HF_HUB_OFFLINE=1 "$PY" "$ROOT/scripts/ask_hybrid_v2.py" "$q" -n 3 2>&1 \
    | tee /tmp/local-doc-kb-media-smoke.out | head -30
  if grep -q 'media channel unavailable' /tmp/local-doc-kb-media-smoke.out; then
    die "smoke still unavailable — see /tmp/local-doc-kb-media-smoke.out"
  fi
  if ! grep -qE 'media [1-9]' /tmp/local-doc-kb-media-smoke.out; then
    die "smoke header missing media N>0 (set SMOKE_QUERY or ensure media index)"
  fi
  ok "smoke media channel up"
}

echo "== local-doc-kb query media env =="
if [[ "$CHECK_ONLY" -eq 0 ]]; then
  install_deps
fi
check_py
check_media_db
check_eg2
if [[ "$SMOKE" -eq 1 ]]; then
  smoke
elif [[ "$CHECK_ONLY" -eq 0 ]]; then
  smoke || true
fi
echo "DONE. See docs/deploy.md for media env notes."
