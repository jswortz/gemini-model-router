"""Optional `.env` injection.

Loaded once at CLI / REPL entry. We deliberately don't take a `python-dotenv`
dependency — the file format we support is the obvious subset (`KEY=value`,
optional `#` comments, optional surrounding quotes). Existing environment
variables always win, so an explicit `export FOO=bar` in the user's shell
overrides whatever `.env` says.
"""

from __future__ import annotations

import os
from pathlib import Path


def _parse(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        out[key] = value
    return out


def _candidate_paths(cfg_path: Path | None) -> list[Path]:
    here = [Path.cwd() / ".env"]
    if cfg_path is not None:
        here.append(cfg_path.parent / ".env")
        here.append(cfg_path.parent.parent / ".env")
    # repo root (two parents up from this file: src/router/env.py)
    here.append(Path(__file__).resolve().parents[2] / ".env")
    seen: set[Path] = set()
    out: list[Path] = []
    for p in here:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        out.append(rp)
    return out


def load_dotenv(cfg_path: Path | None = None) -> Path | None:
    """Load the first `.env` we find. Returns the path used, or None.

    Precedence: CWD, then sibling-of-config, then config-dir's parent,
    then this repo's root. Existing env vars are never clobbered.
    """
    for path in _candidate_paths(cfg_path):
        if not path.exists():
            continue
        try:
            text = path.read_text()
        except OSError:
            continue
        for key, value in _parse(text).items():
            os.environ.setdefault(key, value)
        return path
    return None
