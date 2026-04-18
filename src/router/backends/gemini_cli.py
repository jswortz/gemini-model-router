from __future__ import annotations

import asyncio
import json
import shutil
import time
from pathlib import Path
from typing import Callable

from router.backends.base import Backend, BackendResponse, Usage, compute_cost
from router.config_loader import BackendCfg
from router.sandbox.workspace import scrubbed_env


def _truncate(d: dict, limit: int = 4000) -> dict:
    s = json.dumps(d)
    if len(s) <= limit:
        return d
    return {"_truncated": True, "preview": s[:limit]}


class GeminiCLIBackend(Backend):
    def __init__(self, cfg: BackendCfg):
        if cfg.binary is None:
            raise ValueError("gemini_cli backend requires binary")
        self.cfg = cfg
        self.name = cfg.name
        self.capabilities = set(cfg.capabilities)
        self._binary = cfg.binary

    async def invoke(
        self,
        prompt: str,
        workspace: Path,
        stream_to: Callable[[str], None] | None = None,
        *,
        vendor_session_id: str | None = None,
    ) -> BackendResponse:
        # v1: no streaming for CLI backends; spinner is rendered by the caller.
        resume_args: list[str] = []
        if vendor_session_id and self.cfg.resume_flag:
            resume_args = [self.cfg.resume_flag, vendor_session_id]
        argv = [self._binary, *resume_args, "-p", prompt, *self.cfg.extra_args]
        t0 = time.perf_counter()
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                cwd=str(workspace),
                env=scrubbed_env(),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
        except FileNotFoundError:
            return BackendResponse(
                text="",
                usage=Usage(),
                latency_ms=(time.perf_counter() - t0) * 1000,
                raw={"error": "binary_not_found"},
                success=False,
                error=f"{self._binary} not found on PATH",
            )

        latency_ms = (time.perf_counter() - t0) * 1000
        if proc.returncode != 0:
            return BackendResponse(
                text="",
                usage=Usage(),
                latency_ms=latency_ms,
                raw={"stderr": stderr.decode(errors="replace")[:2000]},
                success=False,
                error=f"gemini exit={proc.returncode}",
            )

        try:
            data = json.loads(stdout.decode(errors="replace"))
        except json.JSONDecodeError:
            text = stdout.decode(errors="replace")
            return BackendResponse(
                text=text,
                usage=Usage(),
                latency_ms=latency_ms,
                raw={"non_json_stdout": True},
                success=True,
            )

        text = (
            data.get("response")
            or data.get("text")
            or data.get("output")
            or ""
        )
        # Gemini CLI puts metrics under `stats.tokens` or `usage_metadata` depending on version.
        in_tok, out_tok = 0, 0
        stats = data.get("stats", {}) or {}
        tokens = stats.get("tokens", {}) if isinstance(stats, dict) else {}
        if isinstance(tokens, dict):
            in_tok = int(tokens.get("input", 0) or tokens.get("prompt", 0) or 0)
            out_tok = int(tokens.get("output", 0) or tokens.get("response", 0) or 0)
        if not (in_tok or out_tok):
            um = data.get("usage_metadata", {})
            in_tok = int(um.get("prompt_token_count", 0))
            out_tok = int(um.get("candidates_token_count", 0))
        usage = Usage(
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=compute_cost(self.cfg, in_tok, out_tok),
        )

        if stream_to and text:
            stream_to(text)

        sid_out = data.get("session_id") or data.get("sessionId")
        return BackendResponse(
            text=text,
            usage=usage,
            latency_ms=latency_ms,
            raw=_truncate(data),
            success=True,
            vendor_session_id=str(sid_out) if sid_out else None,
        )

    async def health(self) -> bool:
        return shutil.which(self._binary) is not None
