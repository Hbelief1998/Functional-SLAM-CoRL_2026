from __future__ import annotations

# Lightweight prompt loader with simple in-memory caching to avoid per-frame file I/O.

import functools
from pathlib import Path
from typing import Dict

_PROMPT_CACHE: Dict[str, str] = {}
_PROMPT_DIR = Path(__file__).resolve().parent / "prompts"


def _read_prompt(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_prompt(name: str) -> str:
    """Load a prompt file by name from the prompts directory with caching."""
    if name in _PROMPT_CACHE:
        return _PROMPT_CACHE[name]
    path = _PROMPT_DIR / name
    content = _read_prompt(path)
    _PROMPT_CACHE[name] = content
    return content
