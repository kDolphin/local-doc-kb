# AGENTS.md — self-deploy runbook for coding / bot agents

You are an agent asked to deploy **local-doc-kb** and wire it into an answering
agent. The user names one or more local folders. You index them read-only.
Build-time work (text extract, embeddings, local-LLM file descriptions, OCR,
image vectors) is what makes answers cheap: the answering agent later calls
`scripts/ask_hybrid_v2.py` and gets a short evidence pack (path, description,
excerpt around the query) instead of embedding the corpus or reading whole files.

Follow these steps **in order**. Run `scripts/doctor.py` first and after every
step; do not move on while it shows a FAIL row that the step was meant to fix.
This file is the only deploy procedure. If another doc disagrees with it, follow
this file.

**Supported platform: Apple Silicon (M-series) macOS only.** On anything else,
stop and tell the user (FTS-only may work for testing via
`scripts/bootstrap.sh --dev-unsupported`, but it is unsupported).

> 🛑 = **ask the user before doing this.** Never assume consent for these.

## 0. Ground rules

- Source folders are **read-only**. Never modify, move or delete user documents.
- Indexes contain document text: treat `index/` as sensitive. Never commit it or put it on a public host. Copying it to the user's own Grok cloud computer is the default in step 9. For any other agent, copy only with consent.
- Never wipe/rebuild an index (`--rebuild`, `--force`, `rm index/*`) without consent. 🛑
- Never put API keys in `config.json`; use env vars (`api_key_env`) and a git-ignored `.env`.
- Remote/SaaS providers are refused by the tools unless explicitly opted in (section 4). 🛑
- Large model downloads (> ~1 GB) need consent. 🛑

## 1. Detect OS and hardware

```bash
uname -sm                                   # must be: Darwin arm64
sysctl -n machdep.cpu.brand_string          # Apple M1/M2/M3/M4...
sysctl -n hw.memsize | awk '{print $1/2^30 " GB unified memory"}'
df -h .                                     # free disk
command -v brew || echo "no Homebrew"
```

| Result | Action |
|--------|--------|
| not `Darwin arm64` | stop; report unsupported platform |
| no Homebrew | 🛑 ask user to install from https://brew.sh (or allow you to) |
| < 16 GB memory | minimal profile, no local LLM descriptions |
| 16 GB | standard (vectors + desc); LLM ≤ 4B–8B (Q4). minimal only if the user declines desc |
| 32 GB and up | standard, then ask about scans and images. full only if the user says yes. LLM 7B–14B at 32 GB, up to ~30B at ≥ 64 GB |
| < 10 GB free disk | warn user; avoid model downloads |

## 2. Install dependencies

```bash
scripts/bootstrap.sh --tier build --dry-run     # show the plan
scripts/bootstrap.sh --tier build               # venv + pip (torch w/ MPS)
source .venv/bin/activate
```

Tiers: `fts` (stdlib only) · `query` · `build` · `media` · `ocr` · `full`.
System tools are installed only with `--brew` (🛑 ask first): poppler (PDF),
tesseract + tesseract-lang (OCR fallback). `textutil` (`.doc`) ships with macOS.
Apple Vision OCR is the primary OCR backend. The source is
`tools/macos-vision-ocr/`; build it with `tools/macos-vision-ocr/build.sh`
(compiles `visionocr` and self-tests). `config.example.json` already points
`build.vision_cli` at that binary. If the build fails, `build_ocr.py` falls
back to tesseract. Do not block the rest of the deploy on Vision.

## 3. Pick a profile and features

| Profile | features | Needs | Good for |
|---------|----------|-------|----------|
| **fts-only** | all false | nothing | trial, very low RAM |
| **minimal** | `vectors` | small embedding (bge-small, ~100 MB) | 8–16 GB Macs |
| **standard** | `vectors`, `desc` | bge-base + local 7B LLM | most users |
| **full** | `vectors`, `desc`, `media`, `ocr` | + EG2 media model, OCR | 32 GB+ when the user wants scans and images |

Query has six channels. FTS is always on. The other five follow `features`.
Say this list to the user when you confirm the profile:

| # | Channel | Turns on when |
|---|---------|----------------|
| 1 | Keyword (FTS) | always |
| 2 | Body vectors | `features.vectors` |
| 3 | Description vectors | `features.desc` |
| 4 | Description keywords (bigram) | `features.desc` (same switch as 3) |
| 5 | Image and scanned-page vectors | `features.media` |
| 6 | OCR text | `features.ocr` |

`ask_hybrid_v2.py` prints `# channels off: ...` for anything not built. Search
continues with the rest. Missing `features` keys mean the same as
`config.example.json`: vectors on, desc/media/ocr off.
🛑 Confirm the profile with the user (it determines downloads and build time).
On 16 GB and up, propose **standard** (`features.desc: true` plus vectors) and
set it unless the user declines. Descriptions are local build-time work; they
are the line the answering agent reads instead of the whole file. Keep
`desc: false` only under 16 GB, or when the user says no.
Then ask whether those folders contain scans or images the user wants
searchable. Channels 5 and 6 are that capability. Ask even when the user never
mentions scans. On 32 GB and up, if the user says yes, propose **full**
(`media` and `ocr` true) and turn them on. Under 32 GB, say that those two
channels expect 32 GB and leave them off unless the user accepts that cost.
If the user says no, leave both false. Do not turn **full** on because the
machine is large.
`config.example.json` stays at vectors only so a copied file does not start an
LLM, OCR, or the media model by itself.

🛑 Ask what language the folders are mostly in: Chinese, English, or mixed.
Do this even when the user never mentions language. Then set the embedding
model from [docs/models.md](docs/models.md) before any vector build:

| Answer | `providers.embedding.model` | `build.desc_lang` |
|--------|-----------------------------|-------------------|
| mostly Chinese | `BAAI/bge-base-zh-v1.5` (small: `BAAI/bge-small-zh-v1.5`) | `zh` |
| mostly English | `BAAI/bge-base-en-v1.5` (small: `BAAI/bge-small-en-v1.5`) | `en` |
| mixed | `BAAI/bge-m3` | `doc` |

On an 8–16 GB Mac use the small model from that row. Write the same id into
`embeddings.body_model` and `embeddings.bge_base_model`, and the matching small
id into `embeddings.bge_small_model`. The names in `config.example.json` are a
Chinese example, not the choice. Do not leave `bge-*-zh` in place unless the
user said the corpus is Chinese.

## 4. Pick model providers

Decision table — embeddings (`providers.embedding.type`):

| Situation | type | Notes |
|-----------|------|-------|
| default | `sentence_transformers` | in-process, MPS; see docs/models.md |
| Ollama already used | `ollama` | `ollama pull bge-m3` etc. |
| llama.cpp/vLLM/LocalAI/Jan server on localhost | `openai_compatible` | `/v1/embeddings` |
| user insists on cloud | `saas` | 🛑 remote gate (below) |

LLM for file descriptions (`providers.llm.type`), only if `features.desc`:

| Situation | type | lifecycle |
|-----------|------|-----------|
| LM Studio installed | `lmstudio` | `scripts/llm_lifecycle.sh lmstudio start/stop` (lms server+load/unload) |
| Ollama installed / preferred | `ollama` | `llm_lifecycle.sh ollama start` (checks `ollama serve`; pulls only with `LLM_ALLOW_PULL=1` 🛑) |
| other local server | `openai_compatible` | health-check of `base_url` only |
| cloud API | `saas` | 🛑 remote gate |

The model id in `config.example.json` (`qwen2.5:7b-instruct` on Ollama) is only
an example. Do not keep it unless that is the model the user already runs or
the one they just agreed to download.

If `features.desc` is on:

1. See what is already installed. `lms ls` (or the LM Studio app) and
   `ollama list`. Prefer a model already on the machine. If both exist, ask
   which app they use.
2. Write that exact model id into `providers.llm.model`, `providers.llm.type`
   (`lmstudio` or `ollama`), `build.desc_model`, and `embeddings.desc_model`.
   Gemma, Qwen, and Llama are all valid. Match the id the app shows.
3. If nothing is installed, suggest one model from
   [docs/models.md](docs/models.md) for their memory and corpus language, name
   both a Qwen and a Gemma option when both fit, and download only after they
   pick one. 🛑

**Remote gate.** If any provider is `saas` or has a non-localhost `base_url`,
document content (and queries) leaves the machine. Before enabling, 🛑 ask the
user to confirm: data classification, contract/compliance/DPA, API-key handling
(env var only), and cost. Only then set `"allow_remote": true` **and** run tools
with `--i-understand-remote` (or `LOCAL_DOC_KB_I_UNDERSTAND_REMOTE=1`).
Otherwise the tools print a warning and exit with code 3.

**Index/query consistency.** Query embeddings must use the same model as the
index. Vector DBs store `index_meta` (model id, dim, provider). Builders refuse
to mix models (exit 4); `ask` warns and disables vectors (or exits 4 with
`--strict-model`); `doctor` shows FAIL. Switching models = rebuild vectors 🛑.

## 5. Write config.json

🛑 Ask the user **which folders to index** and which sub-folders are sensitive.

```bash
cp config.example.json config.json         # or templates/config.minimal.json
```

Edit: `roots[]` (absolute paths), `denylist_path_prefixes` (add sensitive dirs,
e.g. `/private`, `/HR`, `/.ssh`), `allow_extensions`, `features`, `providers`.
Keep `allow_remote: false` unless section 4 consent was given.

## 6. Run doctor

```bash
python scripts/doctor.py            # PASS/WARN/FAIL table + hints; exit 1 on FAIL
python scripts/doctor.py --json     # machine-readable
```

Fix FAILs top to bottom. WARN rows are acceptable if they concern disabled features.

## 7. Smoke build

```bash
scripts/smoke_test.sh               # offline fixture corpus, FTS + gates; must print SMOKE OK
scripts/smoke_test.sh --with-mock   # also provider pipeline against a local mock server
python scripts/build_fts.py --limit 50
python scripts/ask_hybrid_v2.py "a term you know is in those files" -n 5
```

If `features.vectors`: `python scripts/build_vectors.py --limit 50` (first run may
download the embedding model 🛑 if large), then ask again and check `vector N` > 0.
A good hit prints `excerpt:` with the passage around the query, and `desc:` /
`keywords:` once descriptions are built. That pack is what the answering agent
uses. `--json` prints the same pack as one object.

## 8. Full build

Order (each step is incremental and resume-safe):

```bash
caffeinate -i python scripts/build_fts.py
python scripts/build_ocr.py                      # if features.ocr
scripts/llm_lifecycle.sh <llm-type> start        # if features.desc
python scripts/build_desc.py ; scripts/llm_lifecycle.sh <llm-type> stop
python scripts/build_vectors.py
python scripts/build_desc_vectors.py             # if features.desc
python scripts/build_media_vectors_eg2.py        # if features.media
python scripts/doctor.py
```

Or simply `scripts/incremental_build.sh`. Tell the user the expected duration
(roughly: FTS minutes; vectors ~50–300 docs/s on MPS; LLM desc 5–30 files/min).
Prefer running on AC power. Optional quality check: `scripts/eval_queries.py
queries.jsonl` (see docs/operations.md).

## 9. Install the skill into the answering agent

See `skills/local-doc-kb/INSTALL.md`. Copy or symlink `skills/local-doc-kb/`
into the agent's skills directory, or paste `SKILL.md` into its instructions.
Then verify by asking the agent a question whose answer is in the corpus and
checking it cites real paths from the evidence pack. The skill tells it to open
a source file only when the excerpt does not contain the fact.

The answering agent runs `scripts/ask_hybrid_v2.py` beside `index/`.

- **Grok.** Default: the Mac builds the index, `scripts/sync_mirror.sh` copies
  `index/*.sqlite3` to the user's Grok cloud computer, and search uses that
  copy. Prefer the cloud index over the Mac index whenever the copy exists.
  If `hosts.build_host` and `hosts.query_index_dir` are empty, ask once for
  the cloud repo path and write them into config. Do not hardcode a path.
  Tell the user this copy is the default; skip it only if they refuse.
  After every later build, refresh the cloud copy. The cloud machine needs
  the same embedding model as the index. Install the skill on that cloud
  computer, since that is where Grok answers.
- **Any other agent.** Same machine as the build is the default. Copy the
  index only if the user says that agent runs elsewhere. 🛑 Confirm first.

The query script and the index format are the same for every agent. What
changes is the skill directory and the absolute repo path on the machine
that answers. This copy is rsync of the index, not a GitHub upload.

## 10. Optional: schedule incremental builds

🛑 Ask the user whether to schedule, when, and on which power state.
Template: `templates/com.local-doc-kb.incremental.plist` (launchd) running
`scripts/incremental_build.sh`; or an agent routine (`templates/routine-noon-incremental.md`).
For Grok, the scheduled build should end with `scripts/sync_mirror.sh` so the
cloud index stays the one search uses. Other agents mirror only when the user
asked for a second machine (docs/operations.md).

## 11. Report back to the user

Summarize: profile, corpus language, embedding model, description model (or
none), providers (local vs remote), roots + denylist, index sizes (`doctor`),
smoke/eval results, schedule (if any), and how to uninstall.

## Exit codes

| code | meaning |
|------|---------|
| 0 | ok |
| 1 | generic failure / doctor FAIL |
| 2 | missing config/roots/index |
| 3 | refused: remote provider without opt-in |
| 4 | refused: embedding model/dim mismatch with index |
