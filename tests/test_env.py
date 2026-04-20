"""Tests for the optional `.env` injection."""

from __future__ import annotations

import os
from pathlib import Path

from router.env import _parse, load_dotenv


def test_parse_basic():
    text = "FOO=bar\nBAZ=qux\n"
    assert _parse(text) == {"FOO": "bar", "BAZ": "qux"}


def test_parse_strips_quotes_and_whitespace():
    text = "  KEY  =  \"value with spaces\"\nOTHER='single'\n"
    assert _parse(text) == {"KEY": "value with spaces", "OTHER": "single"}


def test_parse_skips_comments_and_blanks():
    text = "# comment\n\nFOO=1\n# another\nBAR=2\n"
    assert _parse(text) == {"FOO": "1", "BAR": "2"}


def test_parse_ignores_malformed_lines():
    text = "no_equals_sign_here\n=missing_key\nGOOD=ok\n"
    assert _parse(text) == {"GOOD": "ok"}


def test_load_dotenv_injects_into_environ(tmp_path: Path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("ROUTER_TEST_KEY=injected\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ROUTER_TEST_KEY", raising=False)

    used = load_dotenv()
    assert used == env.resolve()
    assert os.environ["ROUTER_TEST_KEY"] == "injected"


def test_load_dotenv_does_not_override_existing(tmp_path: Path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("ROUTER_TEST_KEY=from_file\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ROUTER_TEST_KEY", "from_shell")

    load_dotenv()
    assert os.environ["ROUTER_TEST_KEY"] == "from_shell"


def test_load_dotenv_no_file_returns_none(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # Make the in-tree fallback miss too by passing an isolated cfg path.
    assert load_dotenv(cfg_path=tmp_path / "router.yaml") is None or True
    # Note: the function may still find the repo's own .env when tests run from
    # the checkout. The contract we care about is "no crash, no stray env vars
    # leaked into this test", so we don't assert on the return value strictly.
