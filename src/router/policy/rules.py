from __future__ import annotations

from dataclasses import dataclass

from router.config_loader import BackendCfg, PolicyCfg
from router.features.extractor import PromptFeatures


@dataclass
class HardOverride:
    chosen: str | None              # if set, dispatch directly
    reason: str | None              # human-readable label for logs
    candidate_set: list[str]        # backends still in the running


def _estimate_cost(b: BackendCfg, n_in: int, expected_out: int) -> float:
    return (n_in * b.cost_in_per_1m + expected_out * b.cost_out_per_1m) / 1e6


def apply(
    features: PromptFeatures,
    backends: list[BackendCfg],
    policy: PolicyCfg,
    *,
    force: str | None = None,
    healthy: dict[str, bool] | None = None,
) -> HardOverride:
    names = [b.name for b in backends]

    if force:
        if force not in names:
            raise ValueError(f"--force {force!r} not in backends {names}")
        return HardOverride(chosen=force, reason="force_flag", candidate_set=[force])

    if features.sensitive:
        local = [b.name for b in backends if "local" in b.capabilities]
        chosen = local[0] if local else policy.fallback_backend
        return HardOverride(chosen=chosen, reason="sensitive", candidate_set=[chosen])

    health = healthy or {n: True for n in names}
    candidates = [n for n in names if health.get(n, True)]

    # cost ceiling drops candidates we cannot afford in expectation
    expected_out = min(2 * features.n_tokens_est, 1024)
    affordable = []
    for b in backends:
        if b.name not in candidates:
            continue
        cost = _estimate_cost(b, features.n_tokens_est, expected_out)
        if cost <= policy.cost_ceiling_usd_per_request:
            affordable.append(b.name)
    if affordable:
        candidates = affordable

    # context window
    ctx_ok = []
    for b in backends:
        if b.name in candidates and features.n_tokens_est <= b.max_context:
            ctx_ok.append(b.name)
    if ctx_ok:
        candidates = ctx_ok

    if not candidates:
        return HardOverride(
            chosen=policy.fallback_backend,
            reason="no_candidates",
            candidate_set=[policy.fallback_backend],
        )

    if len(candidates) == 1:
        return HardOverride(chosen=candidates[0], reason="only_candidate", candidate_set=candidates)

    return HardOverride(chosen=None, reason=None, candidate_set=candidates)
