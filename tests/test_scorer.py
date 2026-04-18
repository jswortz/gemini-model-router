from router.config_loader import BackendCfg, PolicyCfg, PolicyWeights
from router.features.extractor import extract
from router.policy import rules, scorer


def _backends():
    return [
        BackendCfg(
            name="gemma4", kind="vllm", endpoint="http://x", model="m",
            capabilities=["stream", "local"],
            cost_in_per_1m=0.0, cost_out_per_1m=0.0,
            expected_latency_ms_per_1k_out=80, max_context=8192,
        ),
        BackendCfg(
            name="gemini", kind="gemini_cli", binary="gemini",
            capabilities=["tools", "long_ctx"],
            cost_in_per_1m=0.30, cost_out_per_1m=2.50,
            expected_latency_ms_per_1k_out=350, max_context=1_000_000,
        ),
        BackendCfg(
            name="claude", kind="claude_cli", binary="claude",
            capabilities=["tools", "agentic", "long_ctx"],
            cost_in_per_1m=3.0, cost_out_per_1m=15.0,
            expected_latency_ms_per_1k_out=600, max_context=200_000,
        ),
    ]


def test_force_flag_short_circuits_to_chosen():
    f = extract("hello")
    o = rules.apply(f, _backends(), PolicyCfg(), force="claude")
    assert o.chosen == "claude"
    assert o.reason == "force_flag"


def test_sensitive_prompt_forced_local():
    f = extract("token AKIAABCDEFGHIJKLMNOP do something with it")
    o = rules.apply(f, _backends(), PolicyCfg())
    assert o.chosen == "gemma4"
    assert o.reason == "sensitive"


def test_unhealthy_vllm_drops_gemma_from_candidates():
    f = extract("explain RAG briefly")
    o = rules.apply(
        f, _backends(), PolicyCfg(),
        healthy={"gemma4": False, "gemini": True, "claude": True},
    )
    assert "gemma4" not in o.candidate_set
    assert o.chosen is None or o.chosen != "gemma4"


def test_local_bonus_pushes_short_qa_to_gemma():
    backends = _backends()
    f = extract("what is REST")
    quality = {"gemma4": 0.34, "gemini": 0.33, "claude": 0.33}
    d = scorer.score(
        f, quality, backends, PolicyCfg(),
        candidate_set=["gemma4", "gemini", "claude"],
    )
    assert d.chosen == "gemma4"


def test_agentic_bonus_pushes_tool_required_to_claude():
    backends = _backends()
    f = extract("refactor src/auth/login.py to use async/await")
    # Quality vector intentionally indistinct so the capability_bonus must drive the choice.
    quality = {"gemma4": 0.33, "gemini": 0.33, "claude": 0.34}
    d = scorer.score(
        f, quality, backends, PolicyCfg(),
        candidate_set=["gemma4", "gemini", "claude"],
    )
    assert d.chosen == "claude"


def test_tie_break_falls_back_when_within_margin():
    backends = _backends()
    f = extract("ambiguous prompt here")
    quality = {"gemma4": 0.34, "gemini": 0.34, "claude": 0.34}
    d = scorer.score(
        f, quality, backends,
        PolicyCfg(weights=PolicyWeights(quality=1.0, cost=0.0, latency=0.0),
                  confidence_margin=0.5, fallback_backend="gemma4"),
        candidate_set=["gemma4", "gemini", "claude"],
    )
    assert d.fallback_used is True
    assert d.chosen == "gemma4"
