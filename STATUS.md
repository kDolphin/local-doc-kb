# Status

Public release of local-doc-kb. Supported platform: Apple Silicon (M-series) macOS.

- Six search channels. The example configuration enables body-vector search only. File descriptions, photo search, and OCR stay off until the user asks for them.
- Setup asks whether the files are mostly Chinese, English, or mixed, and selects the embedding model from that answer. Descriptions use a local instruct model the user already runs. The Qwen name in the example config is an example, not a requirement.
- Grok builds the index on the Mac, copies it to the user's Grok cloud computer, and searches that copy. Other agents search on the build Mac unless the user asks for a different machine.
- The Apple Vision OCR helper is source in `tools/macos-vision-ocr/`. `build.sh` compiles it. Without that binary, OCR uses tesseract.
- Indexes, `config.json`, `.env`, and model weights are not in this repository.

Limits:

- The photo and scanned-page channel uses EmbeddingGemma on this machine. Those weights download only when that channel is turned on.
- Linux can run the offline smoke test. It is not a supported install.
