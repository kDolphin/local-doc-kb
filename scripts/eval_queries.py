#!/usr/bin/env python3
"""Evaluate retrieval: queries.jsonl lines {"query": "...", "expected": ["path substr", ...]}.

Runs ask_hybrid_v2.py per query, reports hit@1/3/k and MRR (first expected match).
Usage: eval_queries.py queries.jsonl [-k 10] [--config C] [-- extra ask args]
"""
import argparse, json, re, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("queries", type=Path)
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--config", type=Path)
    a, extra = ap.parse_known_args()
    extra = [x for x in extra if x != "--"]
    qs = [json.loads(l) for l in a.queries.read_text(encoding="utf-8").splitlines() if l.strip()]
    ks = sorted({1, 3, a.k})
    hits = {k: 0 for k in ks}
    mrr = 0.0
    for q in qs:
        cmd = [sys.executable, str(HERE / "ask_hybrid_v2.py"), q["query"], "-n", str(a.k)] + extra
        if a.config:
            cmd += ["--config", str(a.config)]
        out = subprocess.run(cmd, capture_output=True, text=True).stdout
        paths = re.findall(r"^## \d+\. (.+?)  \(", out, flags=re.M)
        rank = next((i + 1 for i, p in enumerate(paths) if any(e in p for e in q["expected"])), None)
        for k in ks:
            hits[k] += bool(rank and rank <= k)
        mrr += 1.0 / rank if rank else 0.0
        print(f"{'OK ' if rank else 'MISS'} rank={rank or '-':<3} {q['query']}")
    n = max(len(qs), 1)
    print("\n" + "  ".join(f"hit@{k}={hits[k]/n:.2f}" for k in ks) + f"  MRR={mrr/n:.3f}  n={len(qs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
