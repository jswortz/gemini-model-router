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


class PolicyCfg(BaseModel):
    weights: PolicyWeights = Field(default_factory=PolicyWeights)
    confidence_margin: float = 0.05
    fallback_backend: str = "gemma4"
    cost_ceiling_usd_per_request: float = 0.10
    sticky_bonus: float = 0.05


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
    raw = yaml.safe_load(Path(path).read_text())
    return RouterConfig.model_validate(raw)


def load_anchors(path: str | Path) -> dict[str, list[str]]:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"anchors file {path} must be a mapping of backend->list[str]")
    return {k: list(v) for k, v in raw.items()}


def expand(p: str) -> Path:
    return Path(p).expanduser()
