from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import yaml

from router.classifier.embed_anchors import EmbedAnchorsClassifier
from router.config_loader import expand, load_config
from router.orchestrator import find_default_config

# gemini-dreams writes to ~/.gemini/dream_metrics.db; rows for our agent live in
# session_analysis with session_id LIKE 'router:%'. Schema based on the docs.
DEFAULT_DB = "~/.gemini/dream_metrics.db"
SCHEMA_TABLE = "session_analysis"


def _db_path() -> Path:
    return expand(DEFAULT_DB)


def _ensure_db() -> sqlite3.Connection | None:
    p = _db_path()
    if not p.exists():
        return None
    return sqlite3.connect(p)


def cmd_list(args: argparse.Namespace) -> int:
    conn = _ensure_db()
    if conn is None:
        sys.stderr.write(
            f"no dream_metrics.db at {_db_path()} — run `dream run` first.\n"
        )
        return 1
    try:
        cur = conn.execute(
            f"SELECT id, session_id, created_at, status FROM {SCHEMA_TABLE} "
            "WHERE agent_name = ? ORDER BY id DESC LIMIT 50",
            ("router",),
        )
        rows = cur.fetchall()
    except sqlite3.OperationalError as e:
        sys.stderr.write(f"db error (schema mismatch?): {e}\n")
        return 1
    if not rows:
        sys.stdout.write("(no proposals)\n")
        return 0
    for r in rows:
        sys.stdout.write(f"#{r[0]}  session={r[1][:16]}…  at={r[2]}  status={r[3]}\n")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    conn = _ensure_db()
    if conn is None:
        sys.stderr.write("no dream_metrics.db.\n")
        return 1
    cur = conn.execute(
        f"SELECT id, session_id, epiphany, skill_updates, created_at FROM {SCHEMA_TABLE} WHERE id = ?",
        (int(args.id),),
    )
    row = cur.fetchone()
    if not row:
        sys.stderr.write(f"no proposal with id {args.id}\n")
        return 1
    sys.stdout.write(f"#{row[0]}  session={row[1]}  at={row[4]}\n\n")
    sys.stdout.write("EPIPHANY\n--------\n" + (row[2] or "") + "\n\n")
    sys.stdout.write("SKILL UPDATES\n-------------\n" + (row[3] or "") + "\n")
    return 0


def _load_anchors_yaml(path: Path) -> dict[str, list[str]]:
    return yaml.safe_load(path.read_text()) or {}


def _save_anchors_yaml(path: Path, anchors: dict[str, list[str]]) -> None:
    path.write_text(yaml.safe_dump(anchors, sort_keys=True))


def cmd_approve(args: argparse.Namespace) -> int:
    """Append-an-exemplar style approval. The dream proposal text is parsed for a
    simple convention:  `move-to <backend>: "<exemplar prompt>"` (one per line).
    """
    conn = _ensure_db()
    if conn is None:
        sys.stderr.write("no dream_metrics.db.\n")
        return 1
    cur = conn.execute(
        f"SELECT skill_updates FROM {SCHEMA_TABLE} WHERE id = ?", (int(args.id),)
    )
    row = cur.fetchone()
    if not row:
        sys.stderr.write(f"no proposal {args.id}\n")
        return 1
    raw = row[0] or ""

    cfg = load_config(find_default_config())
    anchors_path = Path(cfg.classifier.anchors_file)
    anchors = _load_anchors_yaml(anchors_path)

    added: list[tuple[str, str]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line.lower().startswith("move-to "):
            continue
        try:
            head, exemplar = line.split(":", 1)
        except ValueError:
            continue
        backend = head[len("move-to "):].strip()
        exemplar = exemplar.strip().strip('"').strip("'")
        if backend in anchors and exemplar and exemplar not in anchors[backend]:
            anchors[backend].append(exemplar)
            added.append((backend, exemplar))

    if not added:
        sys.stdout.write(
            "no `move-to <backend>: \"<prompt>\"` directives found in proposal.\n"
        )
        return 0

    _save_anchors_yaml(anchors_path, anchors)
    conn.execute(
        f"UPDATE {SCHEMA_TABLE} SET status = 'Approved' WHERE id = ?", (int(args.id),)
    )
    conn.commit()
    for backend, exemplar in added:
        sys.stdout.write(f"+ {backend}: {exemplar}\n")
    sys.stdout.write(f"approved #{args.id}; run `router-eval rebuild-anchors` to refresh.\n")
    return 0


def cmd_rebuild(args: argparse.Namespace) -> int:
    cfg = load_config(find_default_config())
    clf = EmbedAnchorsClassifier(cfg.classifier)
    clf.rebuild()
    sys.stdout.write("centroids rebuilt.\n")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(prog="router-eval")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list").set_defaults(func=cmd_list)
    sp = sub.add_parser("show"); sp.add_argument("id"); sp.set_defaults(func=cmd_show)
    sp = sub.add_parser("approve"); sp.add_argument("id"); sp.set_defaults(func=cmd_approve)
    sub.add_parser("rebuild-anchors").set_defaults(func=cmd_rebuild)
    args = p.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
