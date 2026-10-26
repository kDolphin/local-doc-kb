#!/usr/bin/env python3
"""Shared embedding helpers (pack BLOBs, schema, device pick)."""
from __future__ import annotations

import sqlite3
import struct


def pack_embedding(vec) -> bytes:
    return struct.pack(f"<{len(vec)}f", *[float(x) for x in vec])


def ensure_vectors_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS vectors (
            relpath TEXT PRIMARY KEY,
            model TEXT NOT NULL,
            dim INTEGER NOT NULL,
            chars INTEGER NOT NULL,
            embedded_at TEXT NOT NULL,
            embedding BLOB NOT NULL
        )"""
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_vectors_model ON vectors(model)")
    conn.commit()


def pick_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    try:
        import torch
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"
