from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

import httpx

from router.backends.base import Backend, BackendResponse, Usage, compute_cost
from router.config_loader import BackendCfg


class VLLMBackend(Backend):
    def __init__(self, cfg: BackendCfg):
        if cfg.endpoint is None or cfg.model is None:
            raise ValueError("vllm backend requires endpoint and model")
        self.cfg = cfg
        self.name = cfg.name
        self.capabilities = set(cfg.capabilities)
        self._base = cfg.endpoint.rstrip("/")

    async def invoke(
        self,
        prompt: str,
        workspace: Path,
        stream_to: Callable[[str], None] | None = None,
        *,
        vendor_session_id: str | None = None,
    ) -> BackendResponse:
        # vLLM has no native session concept; vendor_session_id is intentionally
        # ignored. Multi-turn state for the local backend lives in the router's
        # SessionState (turn-prefix prepending will land with the cascade work).
        del vendor_session_id
        url = f"{self._base}/chat/completions"
        body = {
            "model": self.cfg.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        chunks: list[str] = []
        usage = Usage()
        t0 = time.perf_counter()
        try:
            async with (
                httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=5.0)) as client,
                client.stream("POST", url, json=body) as resp,
            ):
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        evt = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    # streamed token deltas
                    for choice in evt.get("choices", []) or []:
                        delta = choice.get("delta", {}).get("content")
                        if delta:
                            chunks.append(delta)
                            if stream_to:
                                stream_to(delta)
                    # final usage chunk (vLLM emits when include_usage=true)
                    u = evt.get("usage")
                    if u:
                        usage = Usage(
                            input_tokens=int(u.get("prompt_tokens", 0)),
                            output_tokens=int(u.get("completion_tokens", 0)),
                        )
        except httpx.HTTPError as e:
            return BackendResponse(
                text="",
                usage=usage,
                latency_ms=(time.perf_counter() - t0) * 1000,
                raw={"error": str(e)},
                success=False,
                error=f"vllm transport error: {e}",
            )

        usage.cost_usd = compute_cost(self.cfg, usage.input_tokens, usage.output_tokens)
        text = "".join(chunks)
        return BackendResponse(
            text=text,
            usage=usage,
            latency_ms=(time.perf_counter() - t0) * 1000,
            raw={"model": self.cfg.model},
            success=True,
        )

    async def health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(0.5)) as client:
                r = await client.get(f"{self._base}/models")
                return r.status_code == 200
        except httpx.HTTPError:
            return False
