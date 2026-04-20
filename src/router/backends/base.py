from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from router.config_loader import BackendCfg


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class BackendResponse:
    text: str
    usage: Usage
    latency_ms: float
    raw: dict = field(default_factory=dict)
    success: bool = True
    error: str | None = None
    vendor_session_id: str | None = None


class Backend(Protocol):
    name: str
    capabilities: set[str]

    async def invoke(
        self,
        prompt: str,
        workspace: Path,
        stream_to: Callable[[str], None] | None = None,
        *,
        vendor_session_id: str | None = None,
    ) -> BackendResponse: ...

    async def health(self) -> bool: ...


def compute_cost(cfg: BackendCfg, in_tok: int, out_tok: int) -> float:
    return round((in_tok * cfg.cost_in_per_1m + out_tok * cfg.cost_out_per_1m) / 1e6, 6)


StreamCallback = Callable[[str], None] | Callable[[str], Awaitable[None]]
