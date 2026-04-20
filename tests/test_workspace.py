"""Tests for the sandbox workspace selection."""

from __future__ import annotations

from pathlib import Path

import pytest

from router.config_loader import SandboxCfg
from router.sandbox.workspace import (
    DockerWorkspace,
    cwd_workspace,
    ephemeral_workspace,
    workspace_for,
)


def test_ephemeral_workspace_creates_and_cleans_tempdir():
    held: Path | None = None
    with ephemeral_workspace() as ws:
        held = ws
        assert ws.exists() and ws.is_dir()
        assert "router-ws-" in ws.name
    assert held is not None and not held.exists()


def test_cwd_workspace_yields_actual_cwd(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with cwd_workspace() as ws:
        assert ws == tmp_path.resolve()
    # No cleanup — the user's cwd is not ours to delete.
    assert tmp_path.exists()


def test_workspace_for_tempdir_default():
    cfg = SandboxCfg()  # mode=tempdir, copy_cwd=False
    with workspace_for(cfg) as ws:
        assert "router-ws-" in ws.name


def test_workspace_for_cwd_passthrough(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = SandboxCfg(mode="cwd")
    with workspace_for(cfg) as ws:
        assert ws == tmp_path.resolve()


def test_workspace_for_cwd_ignores_copy_cwd_flag(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # copy_cwd is meaningless in cwd mode — must not crash or copy anything.
    cfg = SandboxCfg(mode="cwd", copy_cwd=True)
    with workspace_for(cfg) as ws:
        assert ws == tmp_path.resolve()
        # No router-ws-* sibling tempdir got created.
        assert not list(tmp_path.glob("router-ws-*"))


def test_workspace_for_docker_raises_not_implemented():
    cfg = SandboxCfg(mode="docker")
    cm = workspace_for(cfg)
    assert isinstance(cm, DockerWorkspace)
    with pytest.raises(NotImplementedError), cm:
        pass


def test_sandbox_cfg_rejects_unknown_mode():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        SandboxCfg(mode="quantum")
