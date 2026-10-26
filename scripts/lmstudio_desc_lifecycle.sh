#!/usr/bin/env bash
# Back-compat wrapper: lmstudio_desc_lifecycle.sh start|stop|status
exec "$(dirname "$0")/llm_lifecycle.sh" lmstudio "${1:-}"
