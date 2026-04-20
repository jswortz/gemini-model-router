"""Shared validate-and-write path for router.yaml + anchors.yaml.

Both the web UI (`router-config`) and the CLI (`router config ...`) must use
the same edit pipeline so backups, validation, and write semantics never
diverge.

Conventions:
- Dotted paths walk dicts by key. Lists of dicts that all have a "name" field
  are treated as name-keyed mappings, so `backends.gemma4.cost_in_per_1m`
  works without index arithmetic. Numeric indices also work as a fallback.
- `set_path` parses the raw value as YAML so ints, floats, bools, lists, and
  unquoted strings all coerce correctly.
- Writes are atomic: write to a sibling tempfile, then `os.replace`.
- Backups are timestamped (`<file>.bak-<utc>`) and live next to the file.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from router.config_loader import RouterConfig, load_anchors

# ---------------- file helpers ----------------


def backup(path: Path) -> Path | None:
    """Copy `path` to a timestamped sibling. Returns the backup path, or None
    if the source didn't exist."""
    if not path.exists():
        return None
    now = datetime.now(UTC)
    ts = now.strftime("%Y%m%dT%H%M%S") + f"{now.microsecond // 1000:03d}Z"
    bak = path.with_suffix(path.suffix + f".bak-{ts}")
    shutil.copy2(path, bak)
    return bak


def read_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text())


def write_yaml_atomic(path: Path, payload: Any) -> None:
    text = yaml.safe_dump(payload, sort_keys=False, default_flow_style=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise


def list_backups(path: Path) -> list[Path]:
    """Return existing backup files for `path`, newest first."""
    parent = path.parent
    if not parent.exists():
        return []
    pattern = f"{path.name}.bak-*"
    return sorted(parent.glob(pattern), reverse=True)


# ---------------- dotted-path navigation ----------------


def _step(node: Any, key: str) -> tuple[Any, Any]:
    """Return (parent_container, accessor) for the next descent step.

    accessor is the key/index to use for both get and set on the container.
    Raises KeyError with a readable message if `key` doesn't resolve.
    """
    if isinstance(node, dict):
        if key not in node:
            raise KeyError(f"key {key!r} not found in object")
        return node, key
    if isinstance(node, list):
        # Try numeric index first.
        if key.lstrip("-").isdigit():
            idx = int(key)
            if not -len(node) <= idx < len(node):
                raise KeyError(f"index {idx} out of range for list of length {len(node)}")
            return node, idx
        # Then try name-keyed lookup: list of dicts each with a "name" field.
        for i, item in enumerate(node):
            if isinstance(item, dict) and item.get("name") == key:
                return node, i
        raise KeyError(f"no list item with name={key!r}")
    raise KeyError(f"cannot descend into {type(node).__name__} with key {key!r}")


def get_path(data: Any, dotted: str) -> Any:
    if not dotted:
        return data
    node = data
    for part in dotted.split("."):
        container, accessor = _step(node, part)
        node = container[accessor]
    return node


def set_path(data: Any, dotted: str, value: Any) -> None:
    """Mutate `data` in place: assign `value` at `dotted`."""
    if not dotted:
        raise ValueError("empty path")
    parts = dotted.split(".")
    parent_path, leaf = parts[:-1], parts[-1]
    node = data
    for part in parent_path:
        container, accessor = _step(node, part)
        node = container[accessor]
    container, accessor = _step(node, leaf)
    container[accessor] = value


def parse_value(raw: str) -> Any:
    """Parse a CLI/REPL value as YAML so 0.5, true, [a, b] all work."""
    return yaml.safe_load(raw)


# ---------------- high-level config edits ----------------


def apply_config_set(cfg_path: Path, dotted: str, raw_value: str) -> tuple[Path | None, Any]:
    """Read router.yaml, set dotted=raw_value, validate via RouterConfig,
    backup, and write atomically. Returns (backup_path, parsed_value)."""
    data = read_yaml(cfg_path)
    if not isinstance(data, dict):
        raise ValueError(f"{cfg_path} is not a YAML mapping")
    parsed = parse_value(raw_value)
    set_path(data, dotted, parsed)
    # Re-validate the *whole* config — same model the loader uses, so a save
    # that the writer accepts will always be loadable.
    RouterConfig.model_validate(data)
    bak = backup(cfg_path)
    write_yaml_atomic(cfg_path, data)
    return bak, parsed


def apply_full_config(cfg_path: Path, payload: dict) -> Path | None:
    """Write a full router.yaml after validating through RouterConfig.
    Used by the web UI's PUT /api/config handler."""
    validated = RouterConfig.model_validate(payload)
    if not validated.backends:
        raise ValueError("config must define at least one backend")
    bak = backup(cfg_path)
    write_yaml_atomic(cfg_path, validated.model_dump(exclude_none=True))
    return bak


def apply_full_anchors(anchors_path: Path, payload: dict[str, list[str]]) -> Path | None:
    """Write a full anchors.yaml after validating shape."""
    if not payload:
        raise ValueError("anchors payload is empty")
    for label, exemplars in payload.items():
        if not isinstance(exemplars, list) or not all(isinstance(e, str) for e in exemplars):
            raise ValueError(f"anchors[{label}] must be list[str]")
    bak = backup(anchors_path)
    write_yaml_atomic(anchors_path, payload)
    return bak


def add_anchor(anchors_path: Path, backend: str, exemplar: str) -> Path | None:
    data = load_anchors(anchors_path)
    data.setdefault(backend, []).append(exemplar)
    return apply_full_anchors(anchors_path, data)


def remove_anchor(anchors_path: Path, backend: str, index: int) -> tuple[Path | None, str]:
    data = load_anchors(anchors_path)
    if backend not in data:
        raise KeyError(f"no anchors for backend {backend!r}")
    items = data[backend]
    if not -len(items) <= index < len(items):
        raise IndexError(f"index {index} out of range for {backend!r} (len={len(items)})")
    removed = items.pop(index)
    bak = apply_full_anchors(anchors_path, data)
    return bak, removed


def restore_backup(bak_path: Path, target_path: Path) -> None:
    """Copy a `.bak-...` file back over the live config."""
    if not bak_path.exists():
        raise FileNotFoundError(bak_path)
    shutil.copy2(bak_path, target_path)
