from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import ulid

from router.backends.base import Backend, BackendResponse
from router.backends.registry import build as build_backends
from router.config_loader import RouterConfig
from router.features.extractor import extract
from router.logging.jsonl_writer import JsonlWriter
from router.policy import rules, scorer
from router.sandbox.workspace import ephemeral_workspace
from router.session import SessionStore


@dataclass
class RouteResult:
    response: BackendResponse
    chosen_backend: str
    decision_scores: dict[str, float]
    confidence: float
    fallback_used: bool
    hard_override: str | None
    affinity_locked: bool = False
    shadow_choice: str | None = None


class Orchestrator:
    def __init__(self, config: RouterConfig):
        # Imported lazily so callers that inject their own classifier (tests,
        # offline shims) don't pay the numpy / sentence-transformers cost.
        from router.classifier.embed_anchors import EmbedAnchorsClassifier

        self.cfg = config
        self.classifier = EmbedAnchorsClassifier(config.classifier)
        self.backends: dict[str, Backend] = build_backends(config.backends)
        self.logger = JsonlWriter(config.logging)
        self.sessions = SessionStore()

    async def _health_snapshot(self, candidates: list[str]) -> dict[str, bool]:
        names = [n for n in candidates if n in self.backends]
        results = await asyncio.gather(
            *(self.backends[n].health() for n in names), return_exceptions=True
        )
        return {n: (r is True) for n, r in zip(names, results)}

    def _classify_and_score(
        self,
        prompt: str,
        features,
        override,
        sticky_backend: str | None,
    ):
        quality = self.classifier.classify(prompt)
        return scorer.score(
            features,
            quality,
            self.cfg.backends,
            self.cfg.policy,
            candidate_set=override.candidate_set,
            sticky_backend=sticky_backend,
        )

    async def route(
        self,
        prompt: str,
        *,
        session_id: str | None = None,
        force: str | None = None,
        stream: bool = True,
        no_affinity: bool = False,
    ) -> RouteResult:
        sid = session_id or str(ulid.new())
        state = self.sessions.get(sid)
        features = extract(prompt)

        all_names = [b.name for b in self.cfg.backends]
        health = await self._health_snapshot(all_names)

        override = rules.apply(
            features,
            self.cfg.backends,
            self.cfg.policy,
            force=force,
            healthy=health,
        )

        affinity_locked = False
        shadow_choice: str | None = None

        if override.chosen is not None:
            chosen_name = override.chosen
            scores = {chosen_name: 1.0}
            confidence = 1.0
            fallback_used = False
            hard_override = override.reason
        elif (
            not no_affinity
            and state.locked_backend is not None
            and state.locked_backend in override.candidate_set
        ):
            # Affinity path: stick with the locked backend, but still run the
            # classifier in shadow so we can train the v2 classifier on the
            # data without disturbing the conversation.
            chosen_name = state.locked_backend
            shadow = self._classify_and_score(
                prompt, features, override, sticky_backend=state.locked_backend
            )
            shadow_choice = shadow.chosen
            scores = shadow.scores
            confidence = shadow.scores.get(chosen_name, shadow.confidence)
            fallback_used = False
            hard_override = None
            affinity_locked = True
        else:
            decision = self._classify_and_score(
                prompt, features, override, sticky_backend=state.locked_backend
            )
            chosen_name = decision.chosen
            scores = decision.scores
            confidence = decision.confidence
            fallback_used = decision.fallback_used
            hard_override = None

        backend = self.backends[chosen_name]

        followup_hint = None
        if state.last_prompt and state.last_prompt.strip() == prompt.strip():
            followup_hint = "exact_re_ask"

        stream_to: Callable[[str], None] | None = None
        if stream and "stream" in backend.capabilities:
            stream_to = lambda chunk: (sys.stdout.write(chunk), sys.stdout.flush())  # noqa: E731

        vendor_sid = state.vendor_session_for(chosen_name)
        with ephemeral_workspace(copy_cwd=self.cfg.sandbox.copy_cwd) as ws:
            resp = await backend.invoke(
                prompt, ws, stream_to=stream_to, vendor_session_id=vendor_sid
            )
            workspace_str = str(ws)

        # update session state
        state.turn_count += 1
        state.last_prompt = prompt
        if resp.success:
            # Lock to the first backend that successfully serves this session.
            if state.locked_backend is None and not no_affinity:
                state.lock(chosen_name)
            if resp.vendor_session_id:
                state.remember_vendor_session(chosen_name, resp.vendor_session_id)

        backend_cfg = next(b for b in self.cfg.backends if b.name == chosen_name)
        self.logger.write(
            session_id=sid,
            prompt=prompt,
            response_text=resp.text,
            latency_ms=resp.latency_ms,
            router_block={
                "chosen_backend": chosen_name,
                "confidence": confidence,
                "scores": scores,
                "features": features.to_dict(),
                "hard_override": hard_override,
                "fallback_used": fallback_used,
                "affinity_locked": affinity_locked,
                "shadow_choice": shadow_choice,
                "vendor_session_id": resp.vendor_session_id,
                "turn": state.turn_count,
            },
            usage_block={
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
                "cost_usd": resp.usage.cost_usd,
            },
            backend_meta={
                "kind": backend_cfg.kind,
                "model": backend_cfg.model or backend_cfg.binary,
                "raw_pruned": resp.raw,
            },
            workspace=workspace_str,
            success=resp.success,
            error=resp.error,
            user_followup_hint=followup_hint,
        )

        return RouteResult(
            response=resp,
            chosen_backend=chosen_name,
            decision_scores=scores,
            confidence=confidence,
            fallback_used=fallback_used,
            hard_override=hard_override,
            affinity_locked=affinity_locked,
            shadow_choice=shadow_choice,
        )

    def unlock_session(self, session_id: str) -> bool:
        return self.sessions.unlock(session_id)


def find_default_config() -> Path:
    candidates = [
        Path("config/router.yaml"),
        Path(__file__).resolve().parents[2] / "config" / "router.yaml",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("config/router.yaml not found in CWD or repo root")
