#!/usr/bin/env bash
# Generic local-LLM lifecycle for build_desc.py.
# Usage: llm_lifecycle.sh <lmstudio|ollama|openai_compatible|saas> start|stop|status
# Model/base_url come from config providers.llm (or env LLM_MODEL / LLM_BASE_URL).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CFG="${LOCAL_DOC_KB_CONFIG:-$ROOT/config.json}"
PROVIDER="${1:-}"; CMD="${2:-}"
[[ -n "$PROVIDER" && -n "$CMD" ]] || { echo "usage: $0 <lmstudio|ollama|openai_compatible|saas> start|stop|status" >&2; exit 2; }

read -r CFG_MODEL CFG_BASE < <(python3 - "$CFG" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[0]).resolve().parent))
try:
    cfg = json.loads(Path(sys.argv[1]).read_text())
except Exception:
    cfg = {}
p = (cfg.get("providers") or {}).get("llm") or {}
b = cfg.get("build") or {}
print(p.get("model") or b.get("desc_model") or "-", p.get("base_url") or b.get("llm_desc_endpoint") or "-")
PY
)
MODEL="${LLM_MODEL:-${LMSTUDIO_DESC_MODEL:-$CFG_MODEL}}"
BASE="${LLM_BASE_URL:-$CFG_BASE}"

health() { curl -fsS -m 5 "$1" >/dev/null 2>&1; }

case "$PROVIDER" in
  lmstudio)
    export PATH="${HOME}/.lmstudio/bin:${PATH}"
    PORT="${LMSTUDIO_PORT:-1234}"
    case "$CMD" in
      start)
        command -v lms >/dev/null || { echo "lms CLI not found (install LM Studio, run it once)" >&2; exit 1; }
        lms server start -p "$PORT" >/dev/null
        for _ in $(seq 1 30); do health "http://127.0.0.1:${PORT}/v1/models" && break; sleep 1; done
        [[ "$MODEL" != "-" ]] && lms load "$MODEL" -y
        lms ps ;;
      stop) lms unload --all >/dev/null 2>&1 || true; lms server stop >/dev/null 2>&1 || true ;;
      status) lms status || true; lms ps || true ;;
      *) exit 2 ;;
    esac ;;
  ollama)
    B="${BASE:-http://127.0.0.1:11434}"; [[ "$B" == "-" ]] && B="http://127.0.0.1:11434"; B="${B%/v1}"
    case "$CMD" in
      start)
        command -v ollama >/dev/null || { echo "ollama not found: brew install ollama" >&2; exit 1; }
        if ! health "$B/api/tags"; then
          echo "starting 'ollama serve' in background (log: logs/ollama.log)"
          mkdir -p "$ROOT/logs"; nohup ollama serve >"$ROOT/logs/ollama.log" 2>&1 &
          for _ in $(seq 1 30); do health "$B/api/tags" && break; sleep 1; done
        fi
        if [[ "$MODEL" != "-" ]] && ! ollama list | awk '{print $1}' | grep -qx "$MODEL"; then
          echo "model $MODEL not present. Pulling (large download — the agent must have user consent)."
          [[ "${LLM_ALLOW_PULL:-0}" == "1" ]] || { echo "set LLM_ALLOW_PULL=1 to allow 'ollama pull $MODEL'" >&2; exit 1; }
          ollama pull "$MODEL"
        fi
        echo "ollama ready: $MODEL" ;;
      stop) [[ "$MODEL" != "-" ]] && ollama stop "$MODEL" 2>/dev/null || true ;;
      status) health "$B/api/tags" && ollama ps || echo "ollama not reachable at $B" ;;
      *) exit 2 ;;
    esac ;;
  openai_compatible|saas)
    case "$CMD" in
      start|status)
        [[ "$BASE" == "-" ]] && { echo "providers.llm.base_url not set" >&2; exit 1; }
        if health "${BASE%/}/models"; then echo "reachable: $BASE"; else echo "NOT reachable: $BASE" >&2; exit 1; fi ;;
      stop) echo "nothing to stop for $PROVIDER (externally managed)" ;;
      *) exit 2 ;;
    esac ;;
  *) echo "unknown provider $PROVIDER" >&2; exit 2 ;;
esac
