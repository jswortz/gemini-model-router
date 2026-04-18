from __future__ import annotations

from dataclasses import dataclass

from router.config_loader import BackendCfg, PolicyCfg
from router.features.extractor import PromptFeatures


@dataclass
class Decision:
    chosen: str
    confidence: float
    scores: dict[str, float]
    fallback_used: bool


def _normalized_cost(b: BackendCfg, n_in: int, expected_out: int) -> float:
    # Express cost as a fraction of $0.10 ceiling so it's roughly in [0, 1].
    cost = (n_in * b.cost_in_per_1m + expected_out * b.cost_out_per_1m) / 1e6
    return min(cost / 0.10, 1.0)


def _normalized_latency(b: BackendCfg, expected_out: int) -> float:
    # 0..1 where 1 = ~3 seconds expected.
    ms = (expected_out / 1000.0) * b.expected_latency_ms_per_1k_out
    return min(ms / 3000.0, 1.0)


def _capability_bonus(b: BackendCfg, f: PromptFeatures, policy: PolicyCfg) -> float:
    bonus = 0.0
    caps = set(b.capabilities)
    cb = policy.capability_bonuses
    if "local" in caps and f.n_tokens_est < cb.local_short_token_threshold \
            and f.code_fence_count == 0 and not f.tool_required:
        bonus += cb.local_short
    if "agentic" in caps and f.tool_required:
        bonus += cb.agentic_tool
    if "long_ctx" in caps and f.n_tokens_est > cb.long_ctx_token_threshold:
        bonus += cb.long_ctx
    if f.has_url and "tools" in caps:
        bonus += cb.tools_url
    return bonus


def score(
    features: PromptFeatures,
    quality_fit: dict[str, float],
    backends: list[BackendCfg],
    policy: PolicyCfg,
    *,
    candidate_set: list[str],
    sticky_backend: str | None = None,
) -> Decision:
    expected_out = min(2 * features.n_tokens_est, 1024)
    by_name = {b.name: b for b in backends}

    raw: dict[str, float] = {}
    for name in candidate_set:
        b = by_name[name]
        s = (
            policy.weights.quality * quality_fit.get(name, 0.0)
            - policy.weights.cost * _normalized_cost(b, features.n_tokens_est, expected_out)
            - policy.weights.latency * _normalized_latency(b, expected_out)
            + _capability_bonus(b, features, policy)
        )
        if sticky_backend == name:
            s += policy.sticky_bonus
        raw[name] = round(s, 4)

    # rank
    ranked = sorted(raw.items(), key=lambda kv: kv[1], reverse=True)
    top, top_score = ranked[0]
    fallback_used = False

    if len(ranked) >= 2:
        runner_up_score = ranked[1][1]
        if (top_score - runner_up_score) < policy.confidence_margin:
            if policy.fallback_backend in raw:
                top = policy.fallback_backend
                top_score = raw[policy.fallback_backend]
                fallback_used = True

    confidence = top_score
    return Decision(
        chosen=top,
        confidence=round(confidence, 4),
        scores=raw,
        fallback_used=fallback_used,
    )
