from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class SandboxCfg(BaseModel):
    mode: Literal["tempdir", "docker"] = "tempdir"
    copy_cwd: bool = False


class ClassifierCfg(BaseModel):
    type: Literal["embed_anchors", "mmbert"] = "embed_anchors"
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    anchors_file: str = "config/anchors.yaml"
    cache_dir: str = "~/.router/cache"
    softmax_temp: float = 0.07


class PolicyWeights(BaseModel):
    quality: float = 1.0
    cost: float = 0.4
    latency: float = 0.3


class CapabilityBonuses(BaseModel):
    """Tunable per-capability score adjustments applied in policy.scorer.

    These previously lived as hardcoded constants in `_capability_bonus`. Lifted
    to config so weight tuning can actually move the router (the experiment
    showed weight changes were dominated by the +0.5 local-short bonus).
    """

    local_short: float = (
        0.5  # backend has "local" cap, prompt is short, no fences, not tool-required
    )
    agentic_tool: float = 0.4  # backend has "agentic" cap and prompt is tool_required
    long_ctx: float = 0.3  # backend has "long_ctx" cap and prompt > 8k tokens
    tools_url: float = 0.1  # backend has "tools" cap and prompt has a URL
    local_short_token_threshold: int = 200  # boundary for the local_short bonus
    long_ctx_token_threshold: int = 8000


class PolicyCfg(BaseModel):
    weights: PolicyWeights = Field(default_factory=PolicyWeights)
    confidence_margin: float = 0.05
    fallback_backend: str = "gemma4"
    cost_ceiling_usd_per_request: float = 0.10
    sticky_bonus: float = 0.05
    capability_bonuses: CapabilityBonuses = Field(default_factory=CapabilityBonuses)


class LoggingCfg(BaseModel):
    path: str = "~/.router/sessions/router_history.jsonl"
    redact_prompts: bool = False


class BackendCfg(BaseModel):
    name: str
    kind: Literal["vllm", "gemini_cli", "claude_cli"]
    capabilities: list[str] = Field(default_factory=list)
    cost_in_per_1m: float = 0.0
    cost_out_per_1m: float = 0.0
    expected_latency_ms_per_1k_out: float = 200.0
    max_context: int = 8192

    # vllm-specific
    endpoint: str | None = None
    model: str | None = None

    # cli-specific
    binary: str | None = None
    extra_args: list[str] = Field(default_factory=list)
    # If set, the backend will pass [resume_flag, vendor_session_id] to the CLI
    # on subsequent turns of the same router session, so the vendor preserves
    # prompt cache + tool registry. Leave None if the CLI has no resume concept.
    resume_flag: str | None = None


class RouterConfig(BaseModel):
    version: int = 1
    sandbox: SandboxCfg = Field(default_factory=SandboxCfg)
    classifier: ClassifierCfg = Field(default_factory=ClassifierCfg)
    policy: PolicyCfg = Field(default_factory=PolicyCfg)
    logging: LoggingCfg = Field(default_factory=LoggingCfg)
    backends: list[BackendCfg]


def load_config(path: str | Path) -> RouterConfig:
    cfg_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(cfg_path.read_text())
    cfg = RouterConfig.model_validate(raw)
    # Resolve a relative `anchors_file` against the config dir (and its parent,
    # for the historical `config/anchors.yaml` layout) so the router works
    # regardless of the cwd it was launched from.
    cfg.classifier.anchors_file = str(_resolve_relative(cfg.classifier.anchors_file, cfg_path))
    return cfg


def _resolve_relative(p: str, cfg_path: Path) -> Path:
    candidate = Path(p).expanduser()
    if candidate.is_absolute():
        return candidate
    for base in (cfg_path.parent, cfg_path.parent.parent):
        resolved = (base / candidate).resolve()
        if resolved.exists():
            return resolved
    return candidate.resolve()


def load_anchors(path: str | Path) -> dict[str, list[str]]:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"anchors file {path} must be a mapping of backend->list[str]")
    return {k: list(v) for k, v in raw.items()}


def expand(p: str) -> Path:
    return Path(p).expanduser()
