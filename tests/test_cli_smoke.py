"""End-to-end CLI smoke tests.

These run the installed `router` console script against live backends
(Gemini / Claude / vLLM), so they require working credentials and network
access. Skipped by default — opt in with `pytest -m smoke`.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

pytestmark = pytest.mark.smoke


def _have_router() -> bool:
    return shutil.which("router") is not None


@pytest.fixture(scope="module")
def router_cli():
    if not _have_router():
        pytest.skip("`router` CLI not on PATH (install with `uv tool install -e .`)")
    return shutil.which("router")


def _run(cli: str, *args: str, timeout: int = 180) -> subprocess.CompletedProcess:
    """Run `router <args>` from outside the repo so we also smoke-test that
    relative paths in router.yaml resolve regardless of cwd."""
    return subprocess.run(
        [cli, *args],
        cwd="/tmp",
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_hello_world(router_cli):
    """The simplest possible turn: `router "hello world"` returns text."""
    proc = _run(router_cli, "hello world", "--why")
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}"
    # Body goes to stdout; --why metadata goes to stderr.
    assert proc.stdout.strip(), "expected non-empty response on stdout"
    assert "[router] chosen=" in proc.stderr, "expected --why metadata on stderr"


def test_inflation_analysis_routes_to_capable_backend(router_cli):
    """A long-form analysis prompt should route to a backend with sufficient
    quality (gemini or claude) — not local gemma4 — and produce a substantive
    answer that mentions the relevant historical anchors."""
    prompt = (
        "Provide a detailed analysis of economic inflation in the United States "
        "in the early 1980s, including the role of Volcker's monetary policy."
    )
    proc = _run(router_cli, prompt, "--why")
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}"

    body = proc.stdout.strip()
    assert len(body) > 500, f"expected a substantive analysis, got {len(body)} chars"

    body_lower = body.lower()
    # Sanity-check the model actually answered the question — it should mention
    # at least two of these anchors.
    anchors = ["volcker", "inflation", "federal reserve", "interest rate", "1980"]
    hits = sum(1 for a in anchors if a in body_lower)
    assert hits >= 2, f"expected ≥2 topical anchors, found {hits}; body starts: {body[:200]!r}"

    # The router should have picked a non-local backend for this length / topic.
    assert "[router] chosen=" in proc.stderr
    chosen_line = next(
        (line for line in proc.stderr.splitlines() if "[router] chosen=" in line),
        "",
    )
    assert "chosen=gemma4" not in chosen_line, (
        f"expected gemini/claude for a long analysis prompt, got: {chosen_line}"
    )
