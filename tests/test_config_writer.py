from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from router import config_writer as cw
from router.config_loader import load_config

# ---------------- minimal valid router.yaml fixture ----------------


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
            "cost_in_per_1m": 0.0,
            "cost_out_per_1m": 0.0,
            "endpoint": "http://localhost:8000/v1",
            "model": "google/gemma-4-it",
            "max_context": 8192,
        },
        {
            "name": "claude",
            "kind": "claude_cli",
            "capabilities": ["tools", "agentic"],
            "cost_in_per_1m": 3.0,
            "cost_out_per_1m": 15.0,
            "binary": "claude",
            "max_context": 200_000,
        },
    ],
}


@pytest.fixture
def cfg_file(tmp_path: Path) -> Path:
    p = tmp_path / "router.yaml"
    p.write_text(yaml.safe_dump(_BASE_CFG, sort_keys=False))
    return p


@pytest.fixture
def anchors_file(tmp_path: Path) -> Path:
    p = tmp_path / "anchors.yaml"
    p.write_text(yaml.safe_dump({"gemma4": ["what is REST"], "claude": ["refactor X"]}))
    return p


# ---------------- get_path / set_path ----------------


def test_get_path_walks_dict():
    data = {"a": {"b": {"c": 42}}}
    assert cw.get_path(data, "a.b.c") == 42


def test_get_path_empty_returns_root():
    data = {"a": 1}
    assert cw.get_path(data, "") is data


def test_get_path_missing_key_raises():
    with pytest.raises(KeyError):
        cw.get_path({"a": 1}, "b")


def test_get_path_into_named_list_item():
    data = {"backends": [{"name": "gemma4", "cost": 0.0}, {"name": "claude", "cost": 3.0}]}
    assert cw.get_path(data, "backends.claude.cost") == 3.0


def test_get_path_numeric_index_into_list():
    data = {"items": [{"x": 1}, {"x": 2}]}
    assert cw.get_path(data, "items.1.x") == 2


def test_set_path_mutates_dict():
    data = {"a": {"b": 1}}
    cw.set_path(data, "a.b", 9)
    assert data == {"a": {"b": 9}}


def test_set_path_into_named_list_item():
    data = {"backends": [{"name": "gemma4", "cost": 0.0}]}
    cw.set_path(data, "backends.gemma4.cost", 0.5)
    assert data["backends"][0]["cost"] == 0.5


def test_set_path_empty_raises():
    with pytest.raises(ValueError):
        cw.set_path({"a": 1}, "", 2)


# ---------------- parse_value ----------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("0.5", 0.5),
        ("42", 42),
        ("true", True),
        ("false", False),
        ("hello", "hello"),
        ("[a, b, c]", ["a", "b", "c"]),
        ("null", None),
    ],
)
def test_parse_value_yaml_coercion(raw, expected):
    assert cw.parse_value(raw) == expected


# ---------------- apply_config_set ----------------


def test_apply_config_set_writes_and_backs_up(cfg_file: Path):
    bak, parsed = cw.apply_config_set(cfg_file, "policy.weights.cost", "0.5")
    assert parsed == 0.5
    assert bak is not None and bak.exists()
    new = yaml.safe_load(cfg_file.read_text())
    assert new["policy"]["weights"]["cost"] == 0.5
    # Backup preserves the old value.
    old = yaml.safe_load(bak.read_text())
    assert old["policy"]["weights"]["cost"] == 0.4


def test_apply_config_set_into_backend(cfg_file: Path):
    cw.apply_config_set(cfg_file, "backends.gemma4.cost_in_per_1m", "0.25")
    new = yaml.safe_load(cfg_file.read_text())
    by_name = {b["name"]: b for b in new["backends"]}
    assert by_name["gemma4"]["cost_in_per_1m"] == 0.25


def test_apply_config_set_rejects_invalid_and_does_not_write(cfg_file: Path):
    original = cfg_file.read_text()
    # cost_in_per_1m must be float; "notanum" parses to a string and fails validation.
    with pytest.raises(ValidationError):
        cw.apply_config_set(cfg_file, "backends.gemma4.cost_in_per_1m", "notanum")
    assert cfg_file.read_text() == original
    # No backup created on validation failure.
    assert not list(cfg_file.parent.glob(f"{cfg_file.name}.bak-*"))


def test_apply_config_set_result_is_loadable(cfg_file: Path):
    cw.apply_config_set(cfg_file, "policy.weights.cost", "0.7")
    cfg = load_config(cfg_file)
    assert cfg.policy.weights.cost == 0.7


# ---------------- backups ----------------


def test_backup_skipped_when_path_missing(tmp_path: Path):
    assert cw.backup(tmp_path / "nope.yaml") is None


def test_list_backups_newest_first(cfg_file: Path):
    cw.apply_config_set(cfg_file, "policy.weights.cost", "0.5")
    cw.apply_config_set(cfg_file, "policy.weights.cost", "0.6")
    baks = cw.list_backups(cfg_file)
    assert len(baks) == 2
    assert baks[0].name > baks[1].name  # timestamps sort newest first


def test_restore_backup_round_trip(cfg_file: Path):
    cw.apply_config_set(cfg_file, "policy.weights.cost", "0.9")
    [bak] = cw.list_backups(cfg_file)
    cw.restore_backup(bak, cfg_file)
    cfg = load_config(cfg_file)
    assert cfg.policy.weights.cost == 0.4  # back to original


# ---------------- anchors ----------------


def test_add_anchor_appends(anchors_file: Path):
    cw.add_anchor(anchors_file, "gemma4", "what is JSON")
    data = yaml.safe_load(anchors_file.read_text())
    assert "what is JSON" in data["gemma4"]


def test_add_anchor_creates_new_backend_label(anchors_file: Path):
    cw.add_anchor(anchors_file, "newbie", "first exemplar")
    data = yaml.safe_load(anchors_file.read_text())
    assert data["newbie"] == ["first exemplar"]


def test_remove_anchor_returns_removed_text(anchors_file: Path):
    _, removed = cw.remove_anchor(anchors_file, "gemma4", 0)
    assert removed == "what is REST"
    data = yaml.safe_load(anchors_file.read_text())
    assert data["gemma4"] == []


def test_remove_anchor_index_out_of_range(anchors_file: Path):
    with pytest.raises(IndexError):
        cw.remove_anchor(anchors_file, "gemma4", 99)


# ---------------- atomic write ----------------


def test_write_yaml_atomic_no_partial_file_on_crash(tmp_path: Path, monkeypatch):
    target = tmp_path / "out.yaml"
    target.write_text("original: yes\n")

    def boom(*a, **kw):
        raise RuntimeError("simulated crash")

    monkeypatch.setattr(cw.os, "replace", boom)
    with pytest.raises(RuntimeError):
        cw.write_yaml_atomic(target, {"new": "value"})
    # Original is untouched.
    assert target.read_text() == "original: yes\n"
    # No tempfile left behind.
    leftovers = [p for p in tmp_path.iterdir() if p.name.startswith("out.yaml.")]
    assert leftovers == []
