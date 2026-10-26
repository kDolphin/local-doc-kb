#!/usr/bin/env python3
"""Model provider abstraction for local-doc-kb (LLM descriptions + embeddings).

Config (config.json):

  "allow_remote": false,
  "providers": {
    "llm":       {"type": "lmstudio|ollama|openai_compatible|saas",
                  "base_url": "http://127.0.0.1:1234/v1", "model": "...",
                  "api_key_env": "OPENAI_API_KEY"},
    "embedding": {"type": "sentence_transformers|ollama|openai_compatible|saas",
                  "model": "BAAI/bge-base-zh-v1.5", "base_url": "...",
                  "api_key_env": "...", "device": "auto"}
  }

API keys are NEVER read from config.json — only from the env var named by
`api_key_env` (load a local, git-ignored .env yourself if you want).

Remote gate: a provider is "remote" if type == "saas" or its base_url host is
not loopback. Remote use requires BOTH `"allow_remote": true` in config AND
`--i-understand-remote` (or env LOCAL_DOC_KB_I_UNDERSTAND_REMOTE=1).
"""
from __future__ import annotations

import json
import math
import os
import sys
import urllib.error
import urllib.request
from urllib.parse import urlparse

LLM_TYPES = ("lmstudio", "ollama", "openai_compatible", "saas")
EMB_TYPES = ("sentence_transformers", "ollama", "openai_compatible", "saas")

DEFAULT_BASE = {
    "lmstudio": "http://127.0.0.1:1234/v1",
    "ollama": "http://127.0.0.1:11434",
    "openai_compatible": "http://127.0.0.1:8080/v1",
    "saas": "https://api.openai.com/v1",
}
DEFAULT_LLM_MODEL = ""  # set providers.llm.model from the model already installed
DEFAULT_EMB_MODEL = ""  # set providers.embedding.model after asking corpus language
LOOPBACK = {"localhost", "127.0.0.1", "::1", "0.0.0.0", ""}
REMOTE_ENV = "LOCAL_DOC_KB_I_UNDERSTAND_REMOTE"

REMOTE_WARNING = """\
=====================================================================
 SECURITY WARNING: remote / SaaS model provider configured
---------------------------------------------------------------------
 Provider: {what}
 Document CONTENT (text snippets, paths, and/or queries) will be sent
 OFF THIS MACHINE to: {where}
 Before continuing, confirm:
   * data classification allows sending these documents to a third party
   * your contract / compliance / DPA permits this processor
   * the API key is in an env var (never in config.json / git)
   * you accept the per-token cost and rate limits
 To proceed you need BOTH:  "allow_remote": true  in config.json
 AND  --i-understand-remote  (or env {env}=1).
=====================================================================
"""


# ---------------------------------------------------------------- config
def llm_conf(cfg: dict) -> dict:
    """Resolve LLM provider config, with back-compat for build.llm_desc_endpoint."""
    p = dict(((cfg.get("providers") or {}).get("llm")) or {})
    b = cfg.get("build") or {}
    emb = cfg.get("embeddings") or {}
    p.setdefault("type", "lmstudio" if b.get("llm_desc_endpoint") or not p else "openai_compatible")
    if p["type"] not in LLM_TYPES:
        raise SystemExit(f"ERROR: providers.llm.type must be one of {LLM_TYPES}")
    p.setdefault("base_url", b.get("llm_desc_endpoint") or DEFAULT_BASE[p["type"]])
    p["model"] = p.get("model") or b.get("desc_model") or emb.get("desc_model") or DEFAULT_LLM_MODEL
    if not p["model"]:
        raise SystemExit(
            "ERROR: providers.llm.model is empty. Use the instruct model already "
            "installed in LM Studio or Ollama (Gemma, Qwen, Llama, or another). "
            "See docs/models.md."
        )
    return p


def emb_conf(cfg: dict) -> dict:
    p = dict(((cfg.get("providers") or {}).get("embedding")) or {})
    emb = cfg.get("embeddings") or {}
    p.setdefault("type", "sentence_transformers")
    if p["type"] not in EMB_TYPES:
        raise SystemExit(f"ERROR: providers.embedding.type must be one of {EMB_TYPES}")
    p["model"] = p.get("model") or emb.get("body_model") or emb.get("model") or DEFAULT_EMB_MODEL
    if not p["model"]:
        raise SystemExit(
            "ERROR: providers.embedding.model is empty. Ask whether the corpus is "
            "mostly Chinese, English, or mixed, then set a model from docs/models.md."
        )
    if p["type"] != "sentence_transformers":
        p.setdefault("base_url", DEFAULT_BASE[p["type"]])
    return p


def features(cfg: dict) -> dict:
    f = dict(cfg.get("features") or {})
    out = {"fts": True}
    # Missing keys match config.example.json: vectors on, the rest off.
    for k, d in (("vectors", True), ("desc", False), ("media", False), ("ocr", False)):
        out[k] = bool(f.get(k, d))
    return out


# ---------------------------------------------------------------- remote gate
def is_remote(p: dict) -> bool:
    if p.get("type") == "saas":
        return True
    if p.get("type") == "sentence_transformers":
        return False
    host = (urlparse(str(p.get("base_url") or "")).hostname or "").lower()
    return host not in LOOPBACK


def remote_ack(argv_flag: bool = False) -> bool:
    return argv_flag or os.environ.get(REMOTE_ENV, "") in ("1", "true", "yes")


def enforce_remote_gate(cfg: dict, providers: list[dict], *, ack: bool) -> None:
    """Print warning + exit(3) unless remote use is explicitly opted into."""
    remote = [p for p in providers if p and is_remote(p)]
    if not remote:
        return
    for p in remote:
        print(REMOTE_WARNING.format(
            what=f"{p.get('type')} model={p.get('model')}",
            where=p.get("base_url") or "(SaaS default)", env=REMOTE_ENV), file=sys.stderr)
    if not cfg.get("allow_remote"):
        print("REFUSING: config allow_remote is not true.", file=sys.stderr)
        raise SystemExit(3)
    if not remote_ack(ack):
        print(f"REFUSING: pass --i-understand-remote or set {REMOTE_ENV}=1.", file=sys.stderr)
        raise SystemExit(3)
    print("# remote provider explicitly allowed; proceeding.", file=sys.stderr)


def add_remote_flag(ap) -> None:
    ap.add_argument("--i-understand-remote", dest="i_understand_remote", action="store_true",
                    help="acknowledge that a remote/SaaS provider will receive document content")


# ---------------------------------------------------------------- http
def _api_key(p: dict) -> str | None:
    env = p.get("api_key_env")
    if not env:
        return None
    v = os.environ.get(env)
    if not v and p.get("type") == "saas":
        raise SystemExit(f"ERROR: env var {env} (providers.*.api_key_env) is not set")
    return v


def _http_json(url: str, body: dict | None, key: str | None = None, timeout: float = 300):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data, headers, method="POST" if data else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _v1(base: str) -> str:
    b = base.rstrip("/")
    return b if b.endswith("/v1") else b + "/v1"


def _norm(v):
    v = [float(x) for x in v]
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


# ---------------------------------------------------------------- LLM
class LLM:
    def __init__(self, conf: dict):
        self.conf = conf
        self.type = conf["type"]
        self.model = conf["model"]
        self.base = str(conf["base_url"]).rstrip("/")
        self.key = _api_key(conf)

    def chat(self, messages: list[dict], *, temperature=0.2, max_tokens=250) -> str:
        if self.type == "ollama":
            r = _http_json(self.base.removesuffix("/v1") + "/api/chat", {
                "model": self.model, "messages": messages, "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens}})
            return (r.get("message") or {}).get("content") or ""
        base = self.base
        url = base if base.endswith("/chat/completions") else _v1(base) + "/chat/completions"
        body = {"model": self.model, "messages": messages,
                "temperature": temperature, "max_tokens": max_tokens}
        if self.type == "lmstudio":
            body["reasoning_effort"] = "none"
        r = _http_json(url, body, self.key)
        return r["choices"][0]["message"]["content"] or ""

    def health(self) -> tuple[bool, str]:
        return provider_health(self.conf)


# ---------------------------------------------------------------- embeddings
class Embedder:
    """encode(texts) -> list[list[float]] (L2-normalized). Exposes model_id + dim."""

    def __init__(self, conf: dict, device: str = "auto"):
        self.conf = conf
        self.type = conf["type"]
        self.model_name = conf["model"]
        self._st = None
        self._dim = conf.get("dim")
        if self.type == "sentence_transformers":
            from sentence_transformers import SentenceTransformer
            from lib_embed import pick_device
            self._st = SentenceTransformer(self.model_name,
                                           device=pick_device(conf.get("device") or device))
            f = getattr(self._st, "get_embedding_dimension", None) or self._st.get_sentence_embedding_dimension
            self._dim = int(f())
        else:
            self.base = str(conf["base_url"]).rstrip("/")
            self.key = _api_key(conf)

    @property
    def model_id(self) -> str:
        """Stored in index metadata; provider-qualified for non-ST providers."""
        if self.type == "sentence_transformers":
            return self.model_name
        return f"{self.type}:{self.model_name}"

    @property
    def dim(self) -> int:
        if not self._dim:
            self._dim = len(self.encode(["dim probe"])[0])
        return int(self._dim)

    def encode(self, texts: list[str], batch_size: int = 32, query: bool = False):
        if self._st is not None:
            kw = dict(batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
            return [list(map(float, v)) for v in self._st.encode(texts, **kw)]
        out = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            if self.type == "ollama":
                r = _http_json(self.base.removesuffix("/v1") + "/api/embed",
                               {"model": self.model_name, "input": chunk})
                out += [_norm(v) for v in r["embeddings"]]
            else:
                r = _http_json(_v1(self.base) + "/embeddings",
                               {"model": self.model_name, "input": chunk}, self.key)
                data = sorted(r["data"], key=lambda d: d.get("index", 0))
                out += [_norm(d["embedding"]) for d in data]
        if out and not self._dim:
            self._dim = len(out[0])
        return out


# ---------------------------------------------------------------- health
def provider_health(p: dict, timeout: float = 5) -> tuple[bool, str]:
    """Reachability + model availability (best effort). Never sends document text."""
    t = p.get("type")
    if t == "sentence_transformers":
        try:
            import sentence_transformers  # noqa: F401
        except Exception as e:
            return False, f"sentence-transformers not importable: {e}"
        try:
            from huggingface_hub import try_to_load_from_cache
            hit = try_to_load_from_cache(p["model"], "config.json")
            cached = isinstance(hit, str)
        except Exception:
            cached = False
        return True, f"in-process; model {'cached' if cached else 'NOT cached (will download on first use)'}"
    base = str(p.get("base_url") or "").rstrip("/")
    try:
        if t == "ollama":
            r = _http_json(base.removesuffix("/v1") + "/api/tags", None, timeout=timeout)
            names = [m.get("name", "") for m in r.get("models", [])]
            ok = any(n == p["model"] or n.split(":")[0] == p["model"] for n in names)
            return ok, ("model present" if ok else f"model missing: run `ollama pull {p['model']}`")
        key = None
        if p.get("api_key_env"):
            key = os.environ.get(p["api_key_env"])
        r = _http_json(_v1(base) + "/models", None, key, timeout=timeout)
        ids = [m.get("id") for m in r.get("data", [])]
        if not ids:
            return True, "reachable (no model list)"
        ok = p["model"] in ids
        return ok, ("model listed" if ok else f"model {p['model']} not in {ids[:5]}")
    except (urllib.error.URLError, OSError, ValueError) as e:
        return False, f"unreachable {base}: {e}"


# ---------------------------------------------------------------- index metadata
def write_index_meta(conn, *, model: str, dim: int, provider: str) -> None:
    conn.execute("CREATE TABLE IF NOT EXISTS index_meta (key TEXT PRIMARY KEY, value TEXT)")
    for k, v in (("embedding_model", model), ("embedding_dim", str(dim)),
                 ("embedding_provider", provider)):
        conn.execute("INSERT OR REPLACE INTO index_meta VALUES (?,?)", (k, v))
    conn.commit()


def read_index_meta(conn) -> dict:
    try:
        return dict(conn.execute("SELECT key, value FROM index_meta"))
    except Exception:
        return {}


def check_index_meta(conn, model: str, dim: int | None = None) -> str | None:
    """Return a mismatch message, or None if consistent/unknown-but-matching rows."""
    meta = read_index_meta(conn)
    if not meta:
        rows: list[str] = []
        if _has(conn, "vectors"):
            rows = [r[0] for r in conn.execute("SELECT DISTINCT model FROM vectors")]
        elif _has(conn, "media"):
            rows = [r[0] for r in conn.execute("SELECT DISTINCT model FROM media")]
        if rows and model not in rows:
            return f"index has models {rows}, query model is {model}"
        return None
    if meta.get("embedding_model") != model:
        return f"index built with {meta.get('embedding_model')}, query model is {model}"
    if dim and meta.get("embedding_dim") and int(meta["embedding_dim"]) != int(dim):
        return f"index dim {meta['embedding_dim']} != query dim {dim}"
    return None


def _has(conn, table: str) -> bool:
    return bool(conn.execute("SELECT 1 FROM sqlite_master WHERE name=?", (table,)).fetchone())
