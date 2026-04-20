"""Unit tests for the REPL slash-command parser and the multi-line heuristic.

The async chat loop itself isn't unit-tested (it owns terminal I/O and would
require pty plumbing); instead we test the pure functions that decide what to
do with each line.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from router import configcli
from router.config_loader import load_config

# ---------------- repl pure helpers ----------------


def test_unclosed_fence_detects_odd_count():
    from router.repl import _has_unclosed_fence

    assert _has_unclosed_fence("```python\nfoo")
    assert not _has_unclosed_fence("```\nfoo\n```")


def test_wants_continuation_on_trailing_backslash():
    from router.repl import _wants_continuation

    assert _wants_continuation("hello \\")
    assert not _wants_continuation("hello")


def test_wants_continuation_on_open_fence():
    from router.repl import _wants_continuation

    assert _wants_continuation("```py\nx = 1")


def test_wants_continuation_empty_is_false():
    from router.repl import _wants_continuation

    assert not _wants_continuation("")


# ---------------- repl_get / repl_set round-trip ----------------


_BASE_CFG = {
    "version": 1,
    "policy": {
        "weights": {"quality": 1.0, "cost": 0.4, "latency": 0.3},
        "fallback_backend": "gemma4",
        "capability_bonuses": {"local_short": 0.5, "agentic_tool": 0.4},
    },
    "backends": [
        {
            "name": "gemma4",
            "kind": "vllm",
            "capabilities": ["stream", "local"],
            "endpoint": "http://localhost:8000/v1",
            "model": "google/gemma-4-it",
            "max_context": 8192,
        },
    ],
}


@pytest.fixture
def cfg_file(tmp_path: Path) -> Path:
    p = tmp_path / "router.yaml"
    p.write_text(yaml.safe_dump(_BASE_CFG, sort_keys=False))
    return p


def test_repl_set_round_trips_through_load_config(cfg_file: Path):
    bak, parsed = configcli.repl_set(cfg_file, "policy.weights.cost", "0.7")
    assert parsed == 0.7
    assert bak is not None and bak.exists()
    cfg = load_config(cfg_file)
    assert cfg.policy.weights.cost == 0.7


def test_repl_get_returns_scalar(cfg_file: Path):
    assert configcli.repl_get(cfg_file, "policy.weights.cost") == 0.4


def test_repl_get_returns_subtree(cfg_file: Path):
    sub = configcli.repl_get(cfg_file, "policy.weights")
    assert sub == {"quality": 1.0, "cost": 0.4, "latency": 0.3}


def test_repl_get_missing_key_raises(cfg_file: Path):
    with pytest.raises(KeyError):
        configcli.repl_get(cfg_file, "policy.does_not_exist")


def test_repl_set_invalid_value_does_not_write(cfg_file: Path):
    original = cfg_file.read_text()
    with pytest.raises(ValidationError):
        configcli.repl_set(cfg_file, "policy.weights.cost", "[not, a, number]")
    assert cfg_file.read_text() == original


def test_repl_show_dumps_loadable_yaml(cfg_file: Path):
    text = configcli.repl_show(cfg_file)
    parsed = yaml.safe_load(text)
    assert parsed["policy"]["weights"]["cost"] == 0.4


# ---------------- session log enumeration ----------------


def test_read_recent_sessions_parses_jsonl(tmp_path: Path):
    from router.repl import _read_recent_sessions

    log = tmp_path / "history.jsonl"
    log.write_text(
        '{"session_id":"a","prompt":"hello world","router":{"turn":1}}\n'
        '{"session_id":"a","prompt":"second","router":{"turn":2}}\n'
        '{"session_id":"b","prompt":"other","router":{"turn":1}}\n'
    )
    rows = _read_recent_sessions(log, limit=10)
    by_sid = {sid: (turns, preview) for sid, turns, preview in rows}
    assert by_sid["a"] == (2, "second")
    assert by_sid["b"] == (1, "other")


def test_read_recent_sessions_handles_missing_file(tmp_path: Path):
    from router.repl import _read_recent_sessions

    assert _read_recent_sessions(tmp_path / "nope.jsonl", limit=5) == []


def test_read_recent_sessions_skips_malformed_lines(tmp_path: Path):
    from router.repl import _read_recent_sessions

    log = tmp_path / "history.jsonl"
    log.write_text('{"session_id":"a","prompt":"ok","router":{"turn":1}}\nthis is not json\n\n')
    rows = _read_recent_sessions(log, limit=5)
    assert len(rows) == 1
    assert rows[0][0] == "a"


# ---------------- entry-point selection ----------------


def test_should_enter_chat_with_no_prompt_and_tty(monkeypatch):
    from router.cli import _should_enter_chat

    class A:
        cmd = None
        prompt = None

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    assert _should_enter_chat(A()) is True


def test_should_enter_chat_skips_when_prompt_given(monkeypatch):
    from router.cli import _should_enter_chat

    class A:
        cmd = None
        prompt = "hello"

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    assert _should_enter_chat(A()) is False


def test_should_enter_chat_skips_when_config_subcommand(monkeypatch):
    from router.cli import _should_enter_chat

    class A:
        cmd = "config"
        prompt = None

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    assert _should_enter_chat(A()) is False


def test_should_enter_chat_skips_when_stdin_piped(monkeypatch):
    from router.cli import _should_enter_chat

    class A:
        cmd = None
        prompt = None

    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert _should_enter_chat(A()) is False
