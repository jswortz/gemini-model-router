from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from router.config_loader import LoggingCfg, expand


def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.") + f"{datetime.now(timezone.utc).microsecond // 1000:03d}Z"


def _redact(s: str) -> str:
    return "sha256:" + hashlib.sha256(s.encode()).hexdigest()[:16]


class JsonlWriter:
    """Append-only JSONL writer in the gemini-dreams session shape so that
    `dream run` can analyze our logs with no glue beyond a one-line config edit."""

    def __init__(self, cfg: LoggingCfg):
        self.cfg = cfg
        self.path = expand(cfg.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        *,
        session_id: str,
        prompt: str,
        response_text: str,
        latency_ms: float,
        router_block: dict[str, Any],
        usage_block: dict[str, Any],
        backend_meta: dict[str, Any],
        workspace: str,
        success: bool,
        error: str | None = None,
        user_followup_hint: str | None = None,
    ) -> None:
        record = {
            "timestamp": _iso_now(),
            "session_id": session_id,
            "agent_name": "router",
            "cli_type": "router",
            "prompt": _redact(prompt) if self.cfg.redact_prompts else prompt,
            "prompt_response": response_text,
            "latency_ms": round(latency_ms, 2),
            "skills": {},
            "router": router_block,
            "usage": usage_block,
            "backend_meta": backend_meta,
            "workspace": workspace,
            "success": success,
            "error": error,
            "user_followup_hint": user_followup_hint,
        }
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
