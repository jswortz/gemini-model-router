from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path

# Auth env vars we forward into the subprocess (everything else is dropped).
_AUTH_PASSTHROUGH = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "GOOGLE_CLOUD_PROJECT",
    "VERTEX_PROJECT",
    "VERTEX_LOCATION",
)
_BASE_PASSTHROUGH = ("PATH", "HOME", "LANG", "LC_ALL", "TERM", "SHELL", "USER")


def scrubbed_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env: dict[str, str] = {}
    for k in _BASE_PASSTHROUGH + _AUTH_PASSTHROUGH:
        v = os.environ.get(k)
        if v is not None:
            env[k] = v
    if extra:
        env.update(extra)
    return env


@contextmanager
def ephemeral_workspace(
    seed_files: dict[str, bytes] | None = None,
    copy_cwd: bool = False,
) -> Iterator[Path]:
    """Per-request temp dir; auto-cleaned on exit. Adopts GoogleCloudPlatform/scion's
    'isolated workspace per agent invocation' pattern, scaled down to a single user."""
    d = Path(tempfile.mkdtemp(prefix="router-ws-"))
    try:
        if copy_cwd:
            for entry in Path.cwd().iterdir():
                if entry.name in {".git", "node_modules", ".venv", "__pycache__"}:
                    continue
                target = d / entry.name
                if entry.is_dir():
                    shutil.copytree(entry, target, ignore=shutil.ignore_patterns(".git"))
                else:
                    shutil.copy2(entry, target)
        for rel, blob in (seed_files or {}).items():
            p = d / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(blob)
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


@contextmanager
def cwd_workspace() -> Iterator[Path]:
    """Pass-through workspace: backend sub-CLIs run in the user's actual CWD.

    No copy, no cleanup. Anything the backend writes lands in the user's
    real filesystem — that's the whole point (questions like "what's in my
    current repo?" need this), but it means the user is responsible for
    `git status`-level review afterwards.
    """
    yield Path.cwd().resolve()


def workspace_for(cfg) -> AbstractContextManager[Path]:
    """Pick the right workspace context manager for a `SandboxCfg`.

    `cfg` is `router.config_loader.SandboxCfg` — typed loosely to avoid an
    import cycle between sandbox/ and config_loader.
    """
    mode = getattr(cfg, "mode", "tempdir")
    if mode == "cwd":
        return cwd_workspace()
    if mode == "docker":
        return DockerWorkspace()  # raises NotImplementedError on __enter__
    return ephemeral_workspace(copy_cwd=getattr(cfg, "copy_cwd", False))


class DockerWorkspace:
    """Phase-2 stub: a containerized workspace per request, after the
    GoogleCloudPlatform/scion sandbox harness pattern. Not implemented in v1."""

    def __init__(self, image: str = "ubuntu:24.04"):
        self.image = image

    def __enter__(self) -> Path:
        raise NotImplementedError(
            "DockerWorkspace is a Phase-2 stub. See GoogleCloudPlatform/scion for the "
            "containerized sandbox pattern this will adopt."
        )

    def __exit__(self, *exc) -> None:
        return None
