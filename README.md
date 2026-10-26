# local-doc-kb

A knowledge-base index for the files already on your Mac. You name the folders. The project reads them in place and builds a local index, so an answering agent can search that index instead of opening your documents one by one.

## What you get

- **Your files stay where they are.** PDFs, Office files, text, images, and scans are indexed in place. Nothing in those folders is edited, moved, or uploaded as part of the index build.
- **Multimodal when you ask for it.** Keyword search over text is always on. You can add meaning search, a one-line description of each file, OCR for scans, and search across photos and scanned pages. A picture and a PDF can show up in the same answer.
- **Very little agent quota.** Reading files, OCR, embeddings, and descriptions run once on your Mac. A question returns a short evidence card: the file path, a one-line description when one was built, and the passage that matches. The agent cites that card. It does not re-read your library or paste whole files into the model.
- **Private unless you say otherwise.** The build runs on your machine. A remote API is refused unless you turn that on yourself.

## Where answers run

Build the index on the Apple Silicon Mac that holds the files.

With Grok, the usual setup is to copy that index to your Grok cloud computer and let Grok search the copy there. That is the default, not an extra option. The Mac still does the building. The cloud copy is what Grok reads when you ask a question. The index contains text from your files, so it stays on machines you control. It is not uploaded to GitHub.

With any other agent, questions run on the same Mac that built the index. A copy to another computer happens only if you want that agent to live somewhere else.

Any of these agents can answer, as long as it can run `scripts/ask_hybrid_v2.py` next to the index it should use. What you set per agent is where its instruction file is installed, and the path of the repo on that computer. See [skills/local-doc-kb/INSTALL.md](skills/local-doc-kb/INSTALL.md).

## Six search channels

| # | Channel | On when |
|---|---------|---------|
| 1 | Keyword search | always |
| 2 | Meaning search over the body | `features.vectors` |
| 3 | Meaning search over the one-line descriptions | `features.desc` |
| 4 | Keyword search over those descriptions | `features.desc` (same switch as 3) |
| 5 | Photos and scanned pages | `features.media` |
| 6 | Text read from scans (OCR) | `features.ocr` |

The example configuration turns on meaning search over the body and leaves descriptions, photos, and OCR off. Copying it does not start a local language model or download the image model. Channels you leave off are skipped. The others still answer.

Setup asks whether your files are mostly Chinese, English, or mixed, and uses an embedding model for that language. File descriptions use a local model you already have, such as Gemma or Qwen in LM Studio or Ollama, instead of a single required model.

## Platform and privacy

Apple Silicon (M-series) macOS. Linux is only for a small offline test (`scripts/bootstrap.sh --dev-unsupported`).

Source folders are read-only. Do not commit `index/` or put it in a public store. API keys stay in environment variables named by `api_key_env`, never in `config.json`. A remote or non-localhost provider sends document text off the machine. The tools exit with code 3 unless `"allow_remote": true` and `--i-understand-remote`.

The person or agent who sets this up follows [AGENTS.md](AGENTS.md). That file is the setup procedure. It includes the questions to ask you before anything large is downloaded.

## Reference

[AGENTS.md](AGENTS.md) · [docs/models.md](docs/models.md) · [docs/operations.md](docs/operations.md) · [docs/architecture.md](docs/architecture.md) · [skills/local-doc-kb/SKILL.md](skills/local-doc-kb/SKILL.md)

MIT License.

---

# local-doc-kb

给已经在你 Mac 上的文件做的知识库索引。你指定文件夹，项目就地读取并在本机建成索引。之后回答问题的 agent 查的是这份索引，不必把每份文件再打开一遍。

## 你会得到什么

- **文件留在原地。** PDF、Office、文本、图片和扫描件在原处被索引。构建索引不会改写、移动这些文件，也不会把它们当成索引的一部分上传走。
- **需要时可以多模态。** 文本关键词检索始终开启。你还可以加上语义检索、每个文件的一行描述、扫描件 OCR，以及对照片和扫描页的检索。一张图片和一份 PDF 可以出现在同一次回答里。
- **agent 配额用得很少。** 读文件、OCR、向量和描述都在你的 Mac 上做一次。提问时 agent 只收到一张短证据卡：文件路径、建过的话还有一行描述、以及命中的那段原文。它引用这张卡，不会把整个资料库再读一遍，也不会把整份文件贴进模型。
- **默认留在本机。** 构建在你自己的机器上完成。远程接口默认拒绝，除非你自己打开。

## 在哪里回答

在存放文件的 Apple Silicon Mac 上建索引。

用 Grok 时，通常会把索引拷到你的 Grok 云电脑上，由 Grok 优先搜索那一份。这是默认做法，不是附加选项。Mac 仍然负责构建。你提问时，Grok 读的是云电脑上的那份索引。索引里有文件正文，所以只放在你自己控制的机器上，不会传到 GitHub。

用其他 agent 时，在建索引的那台 Mac 上提问。只有你希望那个 agent 跑在别的电脑上时，才另做一份拷贝。

这些 agent 只要能在该用的那份索引旁边运行 `scripts/ask_hybrid_v2.py`，就可以回答。换一种 agent 时，要设置的是说明文件的安装位置，以及那台电脑上仓库的路径。见 [skills/local-doc-kb/INSTALL.md](skills/local-doc-kb/INSTALL.md)。

## 六路检索

| # | 通道 | 何时开启 |
|---|------|----------|
| 1 | 关键词 | 始终 |
| 2 | 正文语义 | `features.vectors` |
| 3 | 一行描述的语义 | `features.desc` |
| 4 | 这些描述的关键词 | `features.desc`（与第 3 路同一个开关） |
| 5 | 照片和扫描页 | `features.media` |
| 6 | 从扫描件读出的文字（OCR） | `features.ocr` |

示例配置只开正文语义，描述、照片和 OCR 默认关闭。复制它不会启动本地语言模型，也不会下载图像模型。关掉的通道会跳过，其余通道继续回答。

安装时会问这些文件主要是中文、英文，还是两种都有，并按回答选择嵌入模型。文件描述用你本机已经有的模型，例如 LM Studio 或 Ollama 里的 Gemma、Qwen，不规定必须用某一个。

## 平台与数据

仅支持 Apple Silicon（M 系列）macOS。Linux 只用于一次小型离线测试（`scripts/bootstrap.sh --dev-unsupported`）。

源目录只读。不要把 `index/` 提交进仓库，也不要放进公开存储。API 密钥放在 `api_key_env` 指定的环境变量里，不要写进 `config.json`。远程或非本机地址会把文档文本送出这台机器。除非同时设置 `"allow_remote": true` 和 `--i-understand-remote`，工具以退出码 3 拒绝。

负责安装的人或 agent 按 [AGENTS.md](AGENTS.md) 操作。那是安装步骤，里面包含下载大模型之前必须先问你的问题。

## 参考

[AGENTS.md](AGENTS.md) · [docs/models.md](docs/models.md) · [docs/operations.md](docs/operations.md) · [docs/architecture.md](docs/architecture.md) · [skills/local-doc-kb/SKILL.md](skills/local-doc-kb/SKILL.md)

MIT 许可证。
