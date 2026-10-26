#!/usr/bin/env python3
"""LLM file-description builder via providers.llm (lmstudio|ollama|openai_compatible|saas).

Writes index/desc.sqlite3 table descs. Resumable: skips (relpath, sha256) already
done for the model. Remote/SaaS providers are refused unless explicitly opted in.

Usage:
  build_desc.py --ext .pptx,.doc,.xls [--sample-ext .docx:50,.pdf:50 --seed 42]
                [--conc 3] [--limit N] [--dry-run]
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import re
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_config import build_section, index_path, load_config  # noqa: E402
from lib_providers import LLM, add_remote_flag, enforce_remote_gate, llm_conf  # noqa: E402

# build.desc_lang: zh = Chinese catalog, en = English catalog, doc = same language as the file.
DESC_LANGS = ("zh", "en", "doc")

COMMON_ZH = (
    '只输出JSON：{"desc": "1-2句中文", "keywords": ["5-10个可能被搜索的词，包括同义说法"]}。'
    "内容中出现的具体品牌、城市、单位名称、项目名、年份、数量请写入描述或关键词；"
    "不要编造内容中没有的信息；不要输出其他内容。"
)
COMMON_EN = (
    'Output JSON only: {"desc": "1-2 sentences", "keywords": ["5-10 terms a person might search, including synonyms"]}. '
    "Copy concrete brands, places, organizations, project names, years, and quantities into the description or keywords. "
    "Do not invent facts that are not in the excerpt. No other text."
)
COMMON_DOC = (
    'Output JSON only: {"desc": "1-2 sentences in the same language as the excerpt", '
    '"keywords": ["5-10 search terms in that same language, including synonyms"]}. '
    "Copy concrete brands, places, organizations, project names, years, and quantities into the description or keywords. "
    "Do not invent facts that are not in the excerpt. No other text."
)
PROMPTS_ZH = {
    "sheet": (
        "你是文档目录编目员。根据给出的Excel文件路径与内容片段，用中文描述这张表。"
        "说明表的类型（如企业名单/预算表/人员名单/申请表模板/配置表/账单）、范围、关键列；"
        "如果表中主要是实体名称（公司/企业/人员），关键词必须包含“名单”“清单”等说法。",
        "内容片段（首行通常为工作表名，之后为单元格文本）",
    ),
    "slides": (
        "你是文档目录编目员。根据给出的PPT演示文稿路径与幻灯片文字，用中文描述这份演示文稿。"
        "说明其用途类型（如方案汇报/产品介绍/培训课件/项目总结）、面向对象、主要议题与结构。",
        "幻灯片文字片段",
    ),
    "doc": (
        "你是文档目录编目员。根据给出的Word文档路径与正文片段，用中文描述这份文档。"
        "说明文档类型（如合同/协议/通知/方案/报告/制度/申请/会议纪要）、涉及单位或项目、主要内容。",
        "正文片段",
    ),
    "pdf": (
        "你是文档目录编目员。根据给出的PDF文件路径与提取文本片段，用中文描述这份文件。"
        "说明文件类型（如合同/发票/报告/说明书/方案/证书/扫描件）、涉及单位或产品、主要内容。",
        "提取文本片段",
    ),
}
PROMPTS_EN = {
    "sheet": (
        "You catalog documents. From the spreadsheet path and cell excerpt, describe the sheet: "
        "its kind (roster, budget, application, config, invoice, or similar), scope, and important columns. "
        "If it is mostly names of organizations or people, the keywords must include \"list\" or \"roster\".",
        "Excerpt (first line is usually the sheet name, then cell text)",
    ),
    "slides": (
        "You catalog documents. From the presentation path and slide text, describe the deck: "
        "its purpose (proposal, product intro, training, project summary, or similar), audience, and main topics.",
        "Slide text excerpt",
    ),
    "doc": (
        "You catalog documents. From the document path and body excerpt, describe the document: "
        "its type (contract, notice, plan, report, policy, minutes, or similar), the organizations or projects involved, and the main point.",
        "Body excerpt",
    ),
    "pdf": (
        "You catalog documents. From the PDF path and extracted text, describe the file: "
        "its type (contract, invoice, report, manual, certificate, scan, or similar), the organizations or products involved, and the main point.",
        "Extracted text excerpt",
    ),
}
KIND = {
    ".xlsx": "sheet",
    ".xls": "sheet",
    ".csv": "sheet",
    ".pptx": "slides",
    ".ppt": "slides",
    ".docx": "doc",
    ".doc": "doc",
    ".rtf": "doc",
    ".txt": "doc",
    ".md": "doc",
    ".pdf": "pdf",
}


def norm_ext(e: str) -> str:
    e = e.strip().lower()
    return e if e.startswith(".") else "." + e


def desc_prompts(lang: str) -> tuple[dict, str]:
    if lang == "zh":
        return PROMPTS_ZH, COMMON_ZH
    if lang == "en":
        return PROMPTS_EN, COMMON_EN
    return PROMPTS_EN, COMMON_DOC


def main() -> int:
    ap = argparse.ArgumentParser(description="Build LLM file descriptions into desc.sqlite3")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--fts", type=Path, default=None)
    ap.add_argument("--db", type=Path, default=None)
    ap.add_argument("--model", default=None, help="LM Studio / OpenAI-compatible model id")
    ap.add_argument("--endpoint", default=None, help="e.g. http://127.0.0.1:1234/v1")
    ap.add_argument("--ext", default="", help="comma list: all files with text for these extensions")
    ap.add_argument(
        "--sample-ext",
        default="",
        help="comma list ext:N random sample, e.g. .docx:50,.pdf:50",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample-list", default="", help="write chosen sample relpaths here (tsv)")
    ap.add_argument("--conc", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--snip", type=int, default=1500)
    ap.add_argument("--dry-run", action="store_true")
    add_remote_flag(ap)
    a = ap.parse_args()

    cfg = load_config(a.config)
    from lib_providers import features as _features
    if not _features(cfg)["desc"]:
        print("SKIP: features.desc is false in config", flush=True)
        return 0
    b = build_section(cfg)
    lang = str(b.get("desc_lang") or "doc").lower()
    if lang not in DESC_LANGS:
        print(f"ERROR: build.desc_lang must be one of {DESC_LANGS}", file=sys.stderr)
        return 2
    prompts, common = desc_prompts(lang)
    fts_db = a.fts or index_path(cfg, "fts", "fts.sqlite3")
    out_db = a.db or index_path(cfg, "desc", "desc.sqlite3")
    lconf = llm_conf(cfg)
    if a.model or os.environ.get("LMSTUDIO_DESC_MODEL"):
        lconf["model"] = a.model or os.environ["LMSTUDIO_DESC_MODEL"]
    if a.endpoint:
        lconf["base_url"] = a.endpoint
    enforce_remote_gate(cfg, [lconf], ack=a.i_understand_remote)
    llm = LLM(lconf)
    model = lconf["model"]

    # Default --ext from config.build.desc_types when neither --ext nor --sample-ext given
    if not a.ext and not a.sample_ext:
        types = b.get("desc_types") or []
        if types:
            a.ext = ",".join(
                (t if str(t).startswith(".") else f".{t}") for t in types
            )

    if not fts_db.exists():
        print(f"FTS db missing: {fts_db}", file=sys.stderr)
        return 2

    out_db.parent.mkdir(parents=True, exist_ok=True)
    src = sqlite3.connect(f"file:{fts_db}?mode=ro", uri=True)
    out = sqlite3.connect(out_db, check_same_thread=False)
    lock = threading.Lock()
    out.execute(
        """CREATE TABLE IF NOT EXISTS descs(
            relpath TEXT PRIMARY KEY,
            sha256 TEXT,
            model TEXT,
            desc TEXT,
            keywords_json TEXT,
            created_at TEXT
        )"""
    )
    done = {
        (r[0], r[1])
        for r in out.execute("SELECT relpath, sha256 FROM descs WHERE model=?", (model,))
    }

    def rows_for(ext: str):
        return [
            r
            for r in src.execute(
                "SELECT relpath, sha256, chars FROM docs WHERE lower(relpath) LIKE ? AND chars>0 ORDER BY relpath",
                ("%" + ext,),
            )
        ]

    sel = []
    for e in filter(None, a.ext.split(",")):
        sel += rows_for(norm_ext(e))
    sample = []
    for spec in filter(None, a.sample_ext.split(",")):
        e, n = spec.split(":")
        e = norm_ext(e)
        pool = rows_for(e)
        rnd = random.Random(f"{a.seed}{e}")
        pick = rnd.sample(pool, min(int(n), len(pool)))
        sample += [(e, r[0]) for r in pick]
        sel += pick
    if a.sample_list and sample:
        with open(a.sample_list, "w", encoding="utf-8") as f:
            f.write(
                f"# seed={a.seed} model={model} generated="
                f"{datetime.datetime.now().isoformat(timespec='seconds')}\n"
            )
            for e, rp in sample:
                f.write(f"{e}\t{rp}\n")

    seen: set[str] = set()
    rows = []
    for r in sel:
        if r[0] in seen or (r[0], r[1]) in done:
            continue
        seen.add(r[0])
        rows.append(r)
    if a.limit:
        rows = rows[: a.limit]
    by: dict[str, int] = {}
    for r in rows:
        e = os.path.splitext(r[0])[1].lower()
        by[e] = by.get(e, 0) + 1
    print(time.strftime("%F %T"), "todo", len(rows), by, "conc", a.conc, "model", model, flush=True)
    if a.dry_run:
        return 0
    if not rows:
        print("nothing to do", flush=True)
        return 0

    def call(rp, chars, text):
        kind = KIND.get(os.path.splitext(rp)[1].lower(), "doc")
        sysmsg, label = prompts[kind]
        t = re.sub(r"[ \t]{2,}", " ", re.sub(r"\n\s*\n+", "\n", text.strip()))[: a.snip]
        c = llm.chat([
            {"role": "system", "content": sysmsg + common},
            {"role": "user", "content": f"Path: {rp}\nCharacters: {chars}\n{label}:\n{t}"},
        ])
        p = json.loads(re.search(r"\{.*\}", c, re.S).group(0))
        assert isinstance(p.get("desc"), str) and isinstance(p.get("keywords"), list)
        return p

    def work(row):
        rp, sha, chars = row
        text = (
            sqlite3.connect(f"file:{fts_db}?mode=ro", uri=True)
            .execute("SELECT text FROM docs WHERE relpath=?", (rp,))
            .fetchone()[0]
        )
        p = err = None
        for _ in range(2):
            try:
                p = call(rp, chars, text)
                break
            except Exception as e:
                err = e
        if p:
            with lock:
                out.execute(
                    "INSERT OR REPLACE INTO descs VALUES(?,?,?,?,?,?)",
                    (
                        rp,
                        sha,
                        model,
                        p["desc"],
                        json.dumps(p["keywords"], ensure_ascii=False),
                        datetime.datetime.now().isoformat(timespec="seconds"),
                    ),
                )
                out.commit()
            return True
        print("FAIL", rp, repr(err)[:200], flush=True)
        return False

    T = time.time()
    n = ok = 0
    with ThreadPoolExecutor(a.conc) as ex:
        for good in ex.map(work, rows):
            n += 1
            ok += good
            if n % 25 == 0 or n == len(rows):
                el = time.time() - T
                rate = n / el * 60 if el else 0
                eta = (len(rows) - n) / (n / el) / 60 if el and n else 0
                print(
                    time.strftime("%F %T"),
                    f"{n}/{len(rows)} ok={ok} {rate:.1f}/min eta={eta:.0f}min",
                    flush=True,
                )
    print("DONE", time.strftime("%F %T"), n, ok, flush=True)
    return 0 if ok > 0 or n == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
