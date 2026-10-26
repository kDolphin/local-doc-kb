#!/usr/bin/env python3
"""Backward-compatible entry: delegates to ask_hybrid_v2."""
import runpy
from pathlib import Path
runpy.run_path(str(Path(__file__).resolve().parent / "ask_hybrid_v2.py"), run_name="__main__")
