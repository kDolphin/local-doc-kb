#!/usr/bin/env python3
"""Tiny offline OpenAI-compatible stub (loopback only) for smoke tests.

Serves /v1/models, /v1/embeddings (deterministic hashed bag-of-words, dim 64)
and /v1/chat/completions (returns a JSON desc built from the prompt). It lets the
smoke test exercise the openai_compatible LLM + embedding providers end to end
without downloading any model. NOT a real model.
"""
import hashlib, json, re, sys
from http.server import BaseHTTPRequestHandler, HTTPServer

DIM = 64


def embed(t):
    v = [0.0] * DIM
    for w in re.findall(r"\w", t.lower()) + re.findall(r"[a-z0-9]+", t.lower()):
        v[int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1.0
    return v


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        self._send({"data": [{"id": "mock-embed"}, {"id": "mock-llm"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path.endswith("/embeddings"):
            inp = body["input"] if isinstance(body["input"], list) else [body["input"]]
            self._send({"data": [{"index": i, "embedding": embed(t)} for i, t in enumerate(inp)]})
        else:
            user = body["messages"][-1]["content"]
            words = re.findall(r"[A-Za-z]{5,}", user)[:6]
            self._send({"choices": [{"message": {"content": json.dumps(
                {"desc": "mock description: " + " ".join(words), "keywords": words})}}]})


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 18080
    HTTPServer(("127.0.0.1", port), H).serve_forever()
