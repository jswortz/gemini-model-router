from __future__ import annotations

import asyncio

from router.backends.base import BackendResponse, Usage
from router.config_loader import (
    BackendCfg,
    LoggingCfg,
    PolicyCfg,
    RouterConfig,
    SandboxCfg,
)
from router.orchestrator import Orchestrator
from router.session import SessionState, SessionStore

# ---------- SessionState / SessionStore ----------


def test_session_state_lock_and_unlock():
    s = SessionState(session_id="abc")
    assert s.locked_backend is None
    s.lock("claude")
    assert s.locked_backend == "claude"
    s.unlock()
    assert s.locked_backend is None


def test_session_state_remembers_vendor_session_ids_per_backend():
    s = SessionState(session_id="abc")
    s.remember_vendor_session("claude", "claude-sess-1")
    s.remember_vendor_session("gemini", "gemini-sess-1")
    assert s.vendor_session_for("claude") == "claude-sess-1"
    assert s.vendor_session_for("gemini") == "gemini-sess-1"
    assert s.vendor_session_for("gemma4") is None


def test_session_state_ignores_empty_vendor_id():
    s = SessionState(session_id="abc")
    s.remember_vendor_session("claude", "")
    assert s.vendor_session_for("claude") is None


def test_session_store_returns_same_state_per_id():
    store = SessionStore()
    a1 = store.get("a")
    a2 = store.get("a")
    b = store.get("b")
    assert a1 is a2
    assert a1 is not b


def test_session_store_unlock_returns_false_when_no_lock():
    store = SessionStore()
    store.get("a")
    assert store.unlock("a") is False
    store.get("a").lock("claude")
    assert store.unlock("a") is True
    assert store.get("a").locked_backend is None


# ---------- Orchestrator affinity wiring ----------


def _backend_cfgs() -> list[BackendCfg]:
    return [
        BackendCfg(
            name="gemma4",
            kind="vllm",
            endpoint="http://x",
            model="m",
            capabilities=["stream", "local"],
            cost_in_per_1m=0.0,
            cost_out_per_1m=0.0,
            expected_latency_ms_per_1k_out=80,
            max_context=8192,
        ),
        BackendCfg(
            name="gemini",
            kind="gemini_cli",
            binary="gemini",
            capabilities=["tools", "long_ctx"],
            cost_in_per_1m=0.30,
            cost_out_per_1m=2.50,
            expected_latency_ms_per_1k_out=350,
            max_context=1_000_000,
        ),
        BackendCfg(
            name="claude",
            kind="claude_cli",
            binary="claude",
            resume_flag="--resume",
            capabilities=["tools", "agentic", "long_ctx"],
            cost_in_per_1m=3.0,
            cost_out_per_1m=15.0,
            expected_latency_ms_per_1k_out=600,
            max_context=200_000,
        ),
    ]


class FakeClassifier:
    def __init__(self, default: dict[str, float]):
        self.default = default
        self.calls = 0

    def classify(self, prompt: str) -> dict[str, float]:
        self.calls += 1
        return dict(self.default)


class FakeBackend:
    def __init__(self, name: str, capabilities: set[str], vendor_session_id: str = "vsid-1"):
        self.name = name
        self.capabilities = capabilities
        self.invocations: list[dict] = []
        self._vsid = vendor_session_id

    async def invoke(self, prompt, workspace, stream_to=None, *, vendor_session_id=None):
        self.invocations.append({"prompt": prompt, "vendor_session_id": vendor_session_id})
        return BackendResponse(
            text=f"reply:{prompt}",
            usage=Usage(input_tokens=10, output_tokens=10, cost_usd=0.0),
            latency_ms=1.0,
            success=True,
            vendor_session_id=self._vsid,
        )

    async def health(self) -> bool:
        return True


class FakeLogger:
    def __init__(self):
        self.records: list[dict] = []

    def write(self, **kw):
        self.records.append(kw)


def _make_orch(
    quality: dict[str, float],
) -> tuple[Orchestrator, dict[str, FakeBackend], FakeClassifier, FakeLogger]:
    cfg = RouterConfig(
        sandbox=SandboxCfg(),
        policy=PolicyCfg(),
        logging=LoggingCfg(),
        backends=_backend_cfgs(),
    )
    orch = Orchestrator.__new__(Orchestrator)
    orch.cfg = cfg
    backends = {
        "gemma4": FakeBackend("gemma4", {"stream", "local"}, vendor_session_id=""),
        "gemini": FakeBackend("gemini", {"tools", "long_ctx"}, vendor_session_id="g-1"),
        "claude": FakeBackend("claude", {"tools", "agentic", "long_ctx"}, vendor_session_id="c-1"),
    }
    classifier = FakeClassifier(quality)
    logger = FakeLogger()
    orch.classifier = classifier
    orch.backends = backends
    orch.logger = logger
    orch.sessions = SessionStore()
    return orch, backends, classifier, logger


def _run(coro):
    return asyncio.run(coro)


def test_first_turn_classifies_and_locks_session():
    orch, backends, classifier, _ = _make_orch(
        quality={"gemma4": 0.1, "gemini": 0.2, "claude": 0.7}
    )
    r = _run(orch.route("refactor src/foo.py to use async", session_id="s1", stream=False))
    assert r.chosen_backend == "claude"
    assert classifier.calls == 1
    assert orch.sessions.get("s1").locked_backend == "claude"
    # vendor session id captured for next turn
    assert orch.sessions.get("s1").vendor_session_for("claude") == "c-1"


def test_second_turn_skips_classifier_and_passes_resume_id():
    orch, backends, classifier, logger = _make_orch(
        quality={"gemma4": 0.1, "gemini": 0.2, "claude": 0.7}
    )
    _run(orch.route("first", session_id="s1", stream=False))
    classifier.calls = 0  # reset

    r = _run(orch.route("follow-up", session_id="s1", stream=False))

    assert r.chosen_backend == "claude"
    assert r.affinity_locked is True
    # classifier still ran in shadow
    assert classifier.calls == 1
    assert r.shadow_choice is not None
    # vendor session id was forwarded to the backend
    last_call = backends["claude"].invocations[-1]
    assert last_call["vendor_session_id"] == "c-1"
    # log captured the affinity + shadow info
    assert logger.records[-1]["router_block"]["affinity_locked"] is True
    assert logger.records[-1]["router_block"]["shadow_choice"] == "claude"


def test_no_affinity_flag_re_classifies_each_turn():
    orch, backends, classifier, _ = _make_orch(
        quality={"gemma4": 0.1, "gemini": 0.2, "claude": 0.7}
    )
    _run(orch.route("first", session_id="s1", stream=False, no_affinity=True))
    # Lock should NOT have been set when no_affinity is on
    assert orch.sessions.get("s1").locked_backend is None
    classifier.calls = 0
    _run(orch.route("second", session_id="s1", stream=False, no_affinity=True))
    # Classifier ran again normally (not in shadow path)
    assert classifier.calls == 1


def test_unlock_releases_session_lock():
    orch, _, classifier, _ = _make_orch(quality={"gemma4": 0.1, "gemini": 0.2, "claude": 0.7})
    _run(orch.route("first", session_id="s1", stream=False))
    assert orch.sessions.get("s1").locked_backend == "claude"

    assert orch.unlock_session("s1") is True
    assert orch.sessions.get("s1").locked_backend is None

    classifier.calls = 0
    r = _run(orch.route("second after unlock", session_id="s1", stream=False))
    # Classifier ran in the live path (not shadow), and session re-locked.
    assert classifier.calls == 1
    assert r.affinity_locked is False
    assert orch.sessions.get("s1").locked_backend == r.chosen_backend


def test_force_flag_bypasses_affinity():
    orch, backends, classifier, _ = _make_orch(
        quality={"gemma4": 0.1, "gemini": 0.2, "claude": 0.7}
    )
    _run(orch.route("first", session_id="s1", stream=False))
    classifier.calls = 0
    r = _run(orch.route("second", session_id="s1", stream=False, force="gemini"))
    assert r.chosen_backend == "gemini"
    # force_flag is a hard override; classifier doesn't run.
    assert classifier.calls == 0


def test_claude_backend_passes_resume_flag(tmp_path):
    """Unit-level: claude_cli backend builds argv with --resume <id> when given."""
    from router.backends.claude_cli import ClaudeCLIBackend

    cfg = BackendCfg(
        name="claude",
        kind="claude_cli",
        binary="claude",
        resume_flag="--resume",
        extra_args=["--output-format", "json"],
        capabilities=["tools"],
    )
    b = ClaudeCLIBackend(cfg)
    # We don't actually run the subprocess; just verify argv assembly logic by
    # patching create_subprocess_exec.
    captured: dict = {}

    async def fake_create(*argv, **kw):
        captured["argv"] = argv
        raise FileNotFoundError("not running for real")

    import router.backends.claude_cli as mod

    orig = mod.asyncio.create_subprocess_exec
    mod.asyncio.create_subprocess_exec = fake_create
    try:
        resp = asyncio.run(b.invoke("hi", tmp_path, vendor_session_id="vsid-99"))
    finally:
        mod.asyncio.create_subprocess_exec = orig

    assert resp.success is False
    assert "--resume" in captured["argv"]
    idx = captured["argv"].index("--resume")
    assert captured["argv"][idx + 1] == "vsid-99"
