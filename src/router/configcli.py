"""`router config ...` subcommand tree.

Validated edits go through `router.config_writer`; the same pipeline the web
UI uses, so backups, schema validation, and atomic write semantics are
identical across both surfaces.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from router import config_writer as cw
from router.config_loader import RouterConfig, load_anchors, load_config


def _resolve_anchors_path(cfg_path: Path) -> Path:
    cfg = load_config(cfg_path)
    candidate = Path(cfg.classifier.anchors_file)
    if not candidate.is_absolute():
        # anchors_file is relative; try CWD then sibling-of-cfg.
        if candidate.exists():
            return candidate.resolve()
        return (cfg_path.parent / candidate.name).resolve()
    return candidate


def _print_yaml(payload, file=sys.stdout) -> None:
    yaml.safe_dump(payload, file, sort_keys=False, default_flow_style=False)


# ---------------- subcommand handlers ----------------


def cmd_show(args, cfg_path: Path) -> int:
    _print_yaml(yaml.safe_load(cfg_path.read_text()))
    return 0


def cmd_get(args, cfg_path: Path) -> int:
    data = yaml.safe_load(cfg_path.read_text())
    try:
        value = cw.get_path(data, args.path or "")
    except KeyError as e:
        sys.stderr.write(f"error: {e}\n")
        return 2
    if isinstance(value, (dict, list)):
        _print_yaml(value)
    else:
        sys.stdout.write(f"{value}\n")
    return 0


def cmd_set(args, cfg_path: Path) -> int:
    try:
        bak, parsed = cw.apply_config_set(cfg_path, args.path, args.value)
    except KeyError as e:
        sys.stderr.write(f"error: {e}\n")
        return 2
    except Exception as e:
        sys.stderr.write(f"error: invalid config edit: {e}\n")
        return 2
    if bak:
        sys.stderr.write(f"backed up {cfg_path.name} -> {bak.name}\n")
    sys.stdout.write(f"set {args.path} = {parsed!r}\n")
    return 0


def cmd_edit(args, cfg_path: Path) -> int:
    editor = os.environ.get("EDITOR") or os.environ.get("VISUAL") or "vi"
    # Edit a copy, validate, then commit.
    import tempfile

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=cfg_path.suffix, delete=False, dir=cfg_path.parent
    ) as fh:
        fh.write(cfg_path.read_text())
        scratch = Path(fh.name)
    try:
        result = subprocess.call([editor, str(scratch)])
        if result != 0:
            sys.stderr.write(f"editor exited {result}; not saving\n")
            return result
        try:
            payload = yaml.safe_load(scratch.read_text())
            RouterConfig.model_validate(payload)
        except Exception as e:
            sys.stderr.write(f"error: edited file failed validation: {e}\n")
            sys.stderr.write(f"your edits remain at {scratch}\n")
            return 2
        bak = cw.backup(cfg_path)
        cw.write_yaml_atomic(cfg_path, payload)
        if bak:
            sys.stderr.write(f"backed up {cfg_path.name} -> {bak.name}\n")
        sys.stdout.write(f"saved {cfg_path}\n")
        return 0
    finally:
        scratch.unlink(missing_ok=True)


def cmd_rebuild_anchors(args, cfg_path: Path) -> int:
    from router.classifier.embed_anchors import EmbedAnchorsClassifier

    cfg = load_config(cfg_path)
    clf = EmbedAnchorsClassifier(cfg.classifier)
    clf.rebuild()
    sys.stdout.write("anchor centroids rebuilt\n")
    return 0


def cmd_backups_list(args, cfg_path: Path) -> int:
    baks = cw.list_backups(cfg_path)
    if not baks:
        sys.stdout.write("(no backups)\n")
        return 0
    for b in baks:
        sys.stdout.write(f"{b.name}\n")
    return 0


def cmd_backups_restore(args, cfg_path: Path) -> int:
    bak_path = cfg_path.parent / args.name
    if not bak_path.exists():
        sys.stderr.write(f"error: backup not found: {bak_path}\n")
        return 2
    # Validate the backup loads cleanly before clobbering.
    try:
        RouterConfig.model_validate(yaml.safe_load(bak_path.read_text()))
    except Exception as e:
        sys.stderr.write(f"error: backup failed to validate: {e}\n")
        return 2
    pre = cw.backup(cfg_path)
    cw.restore_backup(bak_path, cfg_path)
    if pre:
        sys.stderr.write(f"backed up current {cfg_path.name} -> {pre.name}\n")
    sys.stdout.write(f"restored {cfg_path.name} from {args.name}\n")
    return 0


def cmd_anchor_list(args, cfg_path: Path) -> int:
    anchors = load_anchors(_resolve_anchors_path(cfg_path))
    if args.backend:
        items = anchors.get(args.backend, [])
        for i, a in enumerate(items):
            sys.stdout.write(f"  [{i}] {a}\n")
        return 0
    for label, items in anchors.items():
        sys.stdout.write(f"{label}: ({len(items)})\n")
        for i, a in enumerate(items):
            sys.stdout.write(f"  [{i}] {a}\n")
    return 0


def cmd_anchor_add(args, cfg_path: Path) -> int:
    anchors_path = _resolve_anchors_path(cfg_path)
    bak = cw.add_anchor(anchors_path, args.backend, args.exemplar)
    if bak:
        sys.stderr.write(f"backed up {anchors_path.name} -> {bak.name}\n")
    sys.stdout.write(f"added anchor to {args.backend}\n")
    if args.rebuild:
        return cmd_rebuild_anchors(args, cfg_path)
    sys.stderr.write("(run `router config rebuild-anchors` to refresh the centroid cache)\n")
    return 0


def cmd_anchor_remove(args, cfg_path: Path) -> int:
    anchors_path = _resolve_anchors_path(cfg_path)
    try:
        bak, removed = cw.remove_anchor(anchors_path, args.backend, args.index)
    except (KeyError, IndexError) as e:
        sys.stderr.write(f"error: {e}\n")
        return 2
    if bak:
        sys.stderr.write(f"backed up {anchors_path.name} -> {bak.name}\n")
    sys.stdout.write(f"removed anchor [{args.index}] from {args.backend}: {removed!r}\n")
    if args.rebuild:
        return cmd_rebuild_anchors(args, cfg_path)
    return 0


# ---------------- argparse wiring ----------------


def build_config_parser() -> argparse.ArgumentParser:
    """Standalone parser for `router config <subcmd>` (argv minus the leading
    `config` word). Lives separately from the top-level one-shot parser so the
    one-shot `prompt` positional doesn't get eaten by subparser dispatch."""
    cfg_p = argparse.ArgumentParser(prog="router config")
    cfg_p.add_argument("--config", default=None, help="Path to router.yaml.")
    sub = cfg_p.add_subparsers(dest="config_cmd", required=True)

    p_show = sub.add_parser("show", help="Pretty-print the full router.yaml.")
    p_show.set_defaults(func=cmd_show)

    p_get = sub.add_parser("get", help="Read a value at a dotted path.")
    p_get.add_argument("path", nargs="?", default="", help="e.g. policy.weights.cost")
    p_get.set_defaults(func=cmd_get)

    p_set = sub.add_parser("set", help="Write a value at a dotted path (validated).")
    p_set.add_argument("path", help="e.g. policy.weights.cost")
    p_set.add_argument("value", help="YAML-parsed: 0.5, true, '[a, b]', etc.")
    p_set.set_defaults(func=cmd_set)

    p_edit = sub.add_parser("edit", help="Open router.yaml in $EDITOR; validate before saving.")
    p_edit.set_defaults(func=cmd_edit)

    p_rb = sub.add_parser("rebuild-anchors", help="Re-embed anchor centroids.")
    p_rb.set_defaults(func=cmd_rebuild_anchors)

    p_bak = sub.add_parser("backups", help="Manage timestamped router.yaml backups.")
    bak_sub = p_bak.add_subparsers(dest="backups_cmd", required=True)
    p_bak_list = bak_sub.add_parser("list", help="List backups, newest first.")
    p_bak_list.set_defaults(func=cmd_backups_list)
    p_bak_restore = bak_sub.add_parser("restore", help="Restore a named backup.")
    p_bak_restore.add_argument("name", help="Backup filename (just the basename).")
    p_bak_restore.set_defaults(func=cmd_backups_restore)

    p_anc = sub.add_parser("anchor", help="Manage anchors.yaml exemplars.")
    anc_sub = p_anc.add_subparsers(dest="anchor_cmd", required=True)
    p_anc_list = anc_sub.add_parser("list", help="List anchors for one or all backends.")
    p_anc_list.add_argument("backend", nargs="?", default=None)
    p_anc_list.set_defaults(func=cmd_anchor_list)
    p_anc_add = anc_sub.add_parser("add", help="Append an exemplar.")
    p_anc_add.add_argument("backend")
    p_anc_add.add_argument("exemplar")
    p_anc_add.add_argument(
        "--rebuild", action="store_true", help="Re-embed centroids after adding."
    )
    p_anc_add.set_defaults(func=cmd_anchor_add)
    p_anc_rm = anc_sub.add_parser("remove", help="Remove an exemplar by index.")
    p_anc_rm.add_argument("backend")
    p_anc_rm.add_argument("index", type=int)
    p_anc_rm.add_argument(
        "--rebuild", action="store_true", help="Re-embed centroids after removing."
    )
    p_anc_rm.set_defaults(func=cmd_anchor_remove)


def dispatch(args: argparse.Namespace) -> int:
    from router.orchestrator import find_default_config

    cfg_path = Path(args.config).resolve() if args.config else find_default_config().resolve()
    if not cfg_path.exists():
        sys.stderr.write(f"error: config not found at {cfg_path}\n")
        return 2
    return args.func(args, cfg_path)


# Re-export helpers used by REPL slash commands so it doesn't need to know
# about argparse internals.
def repl_get(cfg_path: Path, dotted: str):
    data = yaml.safe_load(cfg_path.read_text())
    return cw.get_path(data, dotted)


def repl_set(cfg_path: Path, dotted: str, raw_value: str) -> tuple[Path | None, object]:
    return cw.apply_config_set(cfg_path, dotted, raw_value)


def repl_show(cfg_path: Path) -> str:
    return yaml.safe_dump(
        yaml.safe_load(cfg_path.read_text()),
        sort_keys=False,
        default_flow_style=False,
    )


# Make `shutil` reachable from tests that monkeypatch its parts.
__all__ = [
    "build_config_parser",
    "dispatch",
    "repl_get",
    "repl_set",
    "repl_show",
    "shutil",
]
