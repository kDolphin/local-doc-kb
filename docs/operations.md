# Operations (Apple Silicon macOS)

## Incremental vs full rebuild
- All builders are incremental: FTS skips unchanged sha256; vectors/desc skip done rows.
- `scripts/incremental_build.sh` runs the enabled steps in order without wiping.
- Full rebuild (`--force`, `--rebuild`, deleting `index/*.sqlite3`) only with user consent;
  back up `index/` first (`cp -c` clones on APFS are cheap).

## Scheduling
- launchd template: `templates/com.local-doc-kb.incremental.plist` (wraps `caffeinate -i`,
  low-priority IO). Install only with consent:
  `launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.local-doc-kb.incremental.plist`;
  remove with `launchctl bootout gui/$(id -u)/com.local-doc-kb.incremental`.
- cron also works (`crontab -e`: `30 12 * * * cd REPO && ./scripts/incremental_build.sh >> logs/inc.log 2>&1`).
- Agent routine: `templates/routine-noon-incremental.md`.

## Build host vs query host
Build on the Mac that has the documents and memory.

For Grok, also copy `index/*.sqlite3` to the user's Grok cloud computer
(`scripts/sync_mirror.sh`, or `hosts.*` in config) and search that copy. That
is the default, not an optional extra. Prefer the cloud index whenever it
exists. For any other agent, query on the build Mac unless the user asks for
a second machine.

The query host must use the **same embedding model** (doctor checks `index_meta`).
The mirror is as sensitive as the source. Do not commit it or publish it.

## Privacy
- Source roots are only read. Use `denylist_path_prefixes` for sensitive dirs
  (HR, finance, `.ssh`, password-manager exports); denied rows are purged on next FTS build.
- `index/` contains full extracted text: keep it out of git (already in `.gitignore`),
  out of cloud-synced folders unless intended, and on an encrypted disk (FileVault).
- Remote providers are gated (`allow_remote` + `--i-understand-remote`).

## Resource limits
- `build_desc.py --conc N` (default 3) — lower on 16 GB.
- `--batch-size` for vectors; `--limit` for trial runs.
- Prevent sleep: `caffeinate -i <cmd>`. Prefer AC power (`pmset -g batt`); the LLM step
  drains battery fast.

## Evaluation
Create `queries.jsonl`, one per line: `{"query": "...", "expected": ["path/substring"]}`.
`python scripts/eval_queries.py queries.jsonl -k 10 [--no-vec ...]` prints hit@1/3/k and MRR.
Compare before/after changing models or weights. Fixture example: `tests/fixtures/queries.jsonl`.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `ERROR: FTS index not found` | not built / wrong config | `build_fts.py`; check `LOCAL_DOC_KB_CONFIG` |
| exit 3 + SECURITY WARNING | remote provider | keep local, or consent + `allow_remote` + flag |
| exit 4 / `embedding mismatch` | config model ≠ index model | revert config or rebuild vectors (consent) |
| `vector 0` hits | vectors not built / feature off | `build_vectors.py`; see `# channels off` |
| PDFs empty | no poppler / scanned PDF | `brew install poppler`; enable OCR |
| `.doc` empty | textutil failed | convert manually; check file isn't encrypted |
| Chinese OCR poor | missing chi_sim | `brew install tesseract-lang` or Vision CLI |
| torch mps=False | x86 Python under Rosetta | use arm64 Python (`file $(which python3)`) |
| LLM step fails | server down / model missing | `llm_lifecycle.sh <type> status`, `ollama pull` (consent) |

## Upgrading / migration
- `git pull` (if the user tracks upstream), re-run `bootstrap.sh`, then `doctor.py`.
- Vector DBs without `index_meta` (older versions) get a WARN; they still work if the
  model matches; next `build_vectors.py` writes the metadata.
- Changing embedding model → rebuild vectors (consent). Changing LLM → old descs stay
  (keyed by model); re-run `build_desc.py` for new descs.

## Uninstall
1. Remove schedule (launchctl bootout / crontab entry).
2. Remove the skill from the agent's skills dir.
3. `rm -rf .venv index/*.sqlite3 logs/` (🛑 confirm — indexes are slow to rebuild).
4. Optional: `ollama rm <model>`, LM Studio model delete, `~/.cache/huggingface` entries.
5. Delete the repo folder. Source documents are untouched.
