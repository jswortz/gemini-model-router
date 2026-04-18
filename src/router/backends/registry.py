from __future__ import annotations

from router.backends.base import Backend
from router.backends.claude_cli import ClaudeCLIBackend
from router.backends.gemini_cli import GeminiCLIBackend
from router.backends.vllm import VLLMBackend
from router.config_loader import BackendCfg


def build(cfgs: list[BackendCfg]) -> dict[str, Backend]:
    out: dict[str, Backend] = {}
    for c in cfgs:
        if c.kind == "vllm":
            out[c.name] = VLLMBackend(c)
        elif c.kind == "gemini_cli":
            out[c.name] = GeminiCLIBackend(c)
        elif c.kind == "claude_cli":
            out[c.name] = ClaudeCLIBackend(c)
        else:
            raise ValueError(f"unknown backend kind: {c.kind}")
    return out
