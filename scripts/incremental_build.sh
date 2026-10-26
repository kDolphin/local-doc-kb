#!/usr/bin/env bash
# Incremental build in dependency order; each step is resume-safe and skips
# disabled features. Never wipes indexes (no --rebuild/--force here).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
PY="${PYTHON:-python3}"; [[ -x .venv/bin/python ]] && PY=.venv/bin/python
FEAT="$($PY -c 'import sys; sys.path.insert(0,"scripts"); from lib_config import load_config; from lib_providers import features, llm_conf; f=features(load_config()); print(" ".join(k for k,v in f.items() if v), llm_conf(load_config())["type"])')"
LLM_TYPE="${FEAT##* }"
echo "$(date '+%F %T') incremental start features: ${FEAT% *}"
$PY scripts/doctor.py --no-network >/dev/null || { echo "doctor FAIL; aborting"; $PY scripts/doctor.py --no-network; exit 1; }
$PY scripts/build_fts.py
[[ " $FEAT " == *" ocr "* ]] && $PY scripts/build_ocr.py
if [[ " $FEAT " == *" desc "* ]]; then
  scripts/llm_lifecycle.sh "$LLM_TYPE" start
  trap 'scripts/llm_lifecycle.sh "$LLM_TYPE" stop' EXIT
  $PY scripts/build_desc.py "$@"
fi
[[ " $FEAT " == *" vectors "* ]] && $PY scripts/build_vectors.py "$@"
[[ " $FEAT " == *" desc "* ]] && $PY scripts/build_desc_vectors.py "$@"
[[ " $FEAT " == *" media "* ]] && $PY scripts/build_media_vectors_eg2.py
echo "$(date '+%F %T') incremental done"
