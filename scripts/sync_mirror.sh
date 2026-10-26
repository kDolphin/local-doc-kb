#!/usr/bin/env bash
# Mirror index/*.sqlite3 from build host path → query host path.
#
# Config / env (any combination):
#   LOCAL_DOC_KB_CONFIG   path to config.json (default: <repo>/config.json)
#   SYNC_SRC              source dir containing *.sqlite3  (or hosts.build_index_dir)
#   SYNC_DEST             destination index dir            (or hosts.query_index_dir)
#   BUILD_HOST            optional user@host for remote rsync source
#   SYNC_RSYNC_OPTS       extra rsync flags (default: -avz --progress)
#
# Examples:
#   SYNC_SRC=/path/on/build/index SYNC_DEST=./index ./scripts/sync_mirror.sh
#   BUILD_HOST=me@build-box SYNC_SRC=/home/me/local-doc-kb/index \
#     SYNC_DEST=/workspace/local-doc-kb/index ./scripts/sync_mirror.sh
#   ./scripts/sync_mirror.sh --dry-run
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CFG="${LOCAL_DOC_KB_CONFIG:-$ROOT/config.json}"
DRY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    -h|--help)
      sed -n '2,18p' "$0"
      exit 0
      ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# Resolve SRC/DEST from env, else config.json hosts.*, else repo index/
read_cfg() {
  local key="$1"
  python3 - "$CFG" "$key" <<'PY'
import json, sys
from pathlib import Path
cfg_path, key = sys.argv[1], sys.argv[2]
p = Path(cfg_path)
if not p.exists():
    print("")
    raise SystemExit(0)
try:
    cfg = json.loads(p.read_text(encoding="utf-8"))
except Exception:
    print("")
    raise SystemExit(0)
hosts = cfg.get("hosts") or {}
val = hosts.get(key) or ""
print(val)
PY
}

SRC="${SYNC_SRC:-$(read_cfg build_index_dir)}"
DEST="${SYNC_DEST:-$(read_cfg query_index_dir)}"
HOST="${BUILD_HOST:-$(read_cfg build_host)}"
OPTS="${SYNC_RSYNC_OPTS:--avz --progress}"

# Defaults
if [[ -z "$SRC" ]]; then
  SRC="$ROOT/index"
fi
if [[ -z "$DEST" ]]; then
  DEST="$ROOT/index"
fi
# Relative DEST → under repo
if [[ "$DEST" != /* ]]; then
  DEST="$ROOT/$DEST"
fi
if [[ "$SRC" != /* && -z "$HOST" ]]; then
  SRC="$ROOT/$SRC"
fi

mkdir -p "$DEST"

if [[ -n "$HOST" ]]; then
  REMOTE="${HOST}:${SRC}/"
else
  REMOTE="${SRC}/"
fi

echo "mirror: ${REMOTE}*.sqlite3  →  ${DEST}/"
if [[ "$DRY" -eq 1 ]]; then
  echo "(dry-run) rsync $OPTS --include='*.sqlite3' --include='*.sqlite3-*' --exclude='*' \"$REMOTE\" \"$DEST/\""
  rsync $OPTS --dry-run --include='*.sqlite3' --include='*.sqlite3-*' --exclude='*' "$REMOTE" "$DEST/" || true
  exit 0
fi

rsync $OPTS --include='*.sqlite3' --include='*.sqlite3-*' --exclude='*' "$REMOTE" "$DEST/"
echo "DONE dest=$DEST"
ls -lh "$DEST"/*.sqlite3 2>/dev/null || echo "(no sqlite3 yet in dest)"
