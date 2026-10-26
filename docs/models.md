# Model choice (Apple Silicon macOS)

The deploying agent asks two things before the first build: whether the folders
are mostly Chinese, English, or mixed, and which local instruct model is already
installed. Use the tables below. Names in `config.example.json` are a Chinese
corpus plus one Ollama example, not the default for every user.

All numbers are rough guidance for unified memory; check each model card yourself.

## Embeddings (`providers.embedding`)

| Corpus language | Default | Alt (smaller) | Alt (multilingual / bigger) | Dim |
|-----------------|---------|---------------|-----------------------------|-----|
| Chinese | `BAAI/bge-base-zh-v1.5` | `BAAI/bge-small-zh-v1.5` (512) | `BAAI/bge-m3` (1024) | 768 |
| English | `BAAI/bge-base-en-v1.5` | `BAAI/bge-small-en-v1.5` (384) | `bge-m3` | 768 |
| Mixed / multilingual | `BAAI/bge-m3` | `intfloat/multilingual-e5-small` (384) | `intfloat/multilingual-e5-large` (1024) | 1024 |

- Sizes: small ≈ 100 MB, base ≈ 400 MB, m3 / e5-large ≈ 2.2 GB on disk.
- Licenses (verify on the model card before redistribution): BGE models — MIT;
  multilingual-e5 — MIT; EmbeddingGemma — Gemma terms of use.
- e5 models expect `query: ` / `passage: ` prefixes; bge-*-v1.5 works without.
- **Switching the embedding model requires rebuilding vectors** (`build_vectors.py
  --rebuild`, `build_desc_vectors.py --rebuild`). The index stores model id + dim
  and tools refuse to mix them.

## LLM for file descriptions (`providers.llm`)

Only needed when `features.desc` is true. A small instruct model is enough.

| Unified memory | Suggested (Q4 quant) | Approx RAM |
|----------------|----------------------|------------|
| 8 GB | skip desc, or `qwen2.5:3b-instruct` / `gemma` 2–4B | 3–4 GB |
| 16 GB | `qwen2.5:7b-instruct`, `gemma` ~4B–9B | 5–7 GB |
| 32 GB | `qwen2.5:14b-instruct`, `gemma` ~12B | 9–12 GB |
| ≥ 64 GB | ~30B MoE / dense models | 18–25 GB |

Leave ≥ 6 GB for macOS + the embedding model. Run the LLM via Ollama or LM Studio
(MLX/GGUF backends both use the GPU). Licenses: Qwen2.5 mostly Apache-2.0 (some
sizes differ), Gemma under Gemma terms — check before commercial use.

`build.desc_lang`: `zh` when the corpus is mostly Chinese, `en` when it is
mostly English, `doc` when it is mixed (same language as each file).
Descriptions are extra searchable text plus the one line ask prints to the
answering agent. The description model is whatever instruct model the user
already runs locally. Qwen and Gemma are both fine. Do not download a second
model just because the example file names one.

## Media / OCR

- Media vectors: EmbeddingGemma-style multimodal model (`embeddings.eg2_model`),
  needs `requirements-media.txt`; ≥ 32 GB recommended.
- OCR: build the Apple Vision helper with `tools/macos-vision-ocr/build.sh`
  (`build.vision_cli` in the example config). tesseract + `chi_sim`
  (`brew install tesseract tesseract-lang`) is the fallback.
