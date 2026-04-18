"""Replay 24h of gemini_dreams prompts through several router configs.

Pulls user turns from the BigQuery `dream_raw_logs` table and feeds them through
the same `features → rules → classifier → scorer` pipeline the live router uses,
*without* invoking any backend. For each (config, prompt) we record the chosen
backend and the projected cost/latency from the per-backend pricing in
`config/router.yaml`. Output is a long-format CSV consumed by `analyze.py`.

Why simulate instead of live-invoke: a Cartesian product of N prompts × C configs
would otherwise burn real Gemini/Claude API spend. Routing decisions are
deterministic given inputs, so simulation gives identical chosen-backend results
to live runs.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from router.config_loader import RouterConfig  # noqa: E402
from router.features.extractor import extract  # noqa: E402
from router.policy.rules import apply as apply_rules  # noqa: E402
from router.policy.scorer import score  # noqa: E402


# ---------- config variants (overrides on policy block) ----------

CONFIG_VARIANTS: dict[str, dict] = {
    "default":       {"weights": {"quality": 1.0, "cost": 0.4, "latency": 0.3}, "sticky_bonus": 0.05},
    "cost_heavy":    {"weights": {"quality": 1.0, "cost": 2.0, "latency": 0.3}, "sticky_bonus": 0.05},
    "latency_heavy": {"weights": {"quality": 1.0, "cost": 0.4, "latency": 2.0}, "sticky_bonus": 0.05},
    "quality_heavy": {"weights": {"quality": 2.0, "cost": 0.4, "latency": 0.3}, "sticky_bonus": 0.05},
    "no_affinity":   {"weights": {"quality": 1.0, "cost": 0.4, "latency": 0.3}, "sticky_bonus": 0.0},
}


def load_base_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def make_config(base: dict, overrides: dict) -> RouterConfig:
    cfg = copy.deepcopy(base)
    cfg["policy"].update({k: v for k, v in overrides.items() if k != "weights"})
    cfg["policy"]["weights"].update(overrides.get("weights", {}))
    return RouterConfig.model_validate(cfg)


# ---------- data loading ----------

def load_turns(csv_path: Path) -> list[dict]:
    """Expand session-level rows into per-user-turn rows."""
    turns: list[dict] = []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            try:
                payload = json.loads(row["log_content"])
            except json.JSONDecodeError:
                continue
            sess_latency = payload.get("latency_ms")
            for i, m in enumerate(payload.get("messages", [])):
                if m.get("role") != "user":
                    continue
                content = m.get("content", "")
                if not content:
                    continue
                turns.append({
                    "agent": row["agent_name"],
                    "session_id": row["session_id"],
                    "turn_idx": i,
                    "row_ts": row["timestamp"],
                    "prompt": content,
                    "observed_latency_ms": sess_latency,
                })
    return turns


# ---------- classifier (lazy) ----------

class _Cls:
    def __init__(self, cfg):
        from router.classifier.embed_anchors import EmbedAnchorsClassifier
        self._impl = EmbedAnchorsClassifier(cfg.classifier)

    def quality_fit(self, prompt: str, candidates: list[str]) -> dict[str, float]:
        # Returns softmax over all configured anchors; restrict to candidates.
        scores = self._impl.classify(prompt)
        return {n: scores.get(n, 0.0) for n in candidates}


# ---------- per-prompt simulation ----------

def project_cost_usd(backend, n_in: int, n_out: int) -> float:
    return (n_in * backend.cost_in_per_1m + n_out * backend.cost_out_per_1m) / 1e6


def project_latency_ms(backend, n_out: int) -> float:
    return (n_out / 1000.0) * backend.expected_latency_ms_per_1k_out


def simulate_one(cfg: RouterConfig, prompt: str, sticky: str | None, classifier: _Cls) -> dict:
    feats = extract(prompt)
    expected_out = min(2 * feats.n_tokens_est, 1024)

    override = apply_rules(feats, cfg.backends, cfg.policy)
    if override.chosen is not None:
        chosen = override.chosen
        scores = {chosen: 1.0}
        confidence = 1.0
        fallback = False
        path = override.reason
    else:
        qf = classifier.quality_fit(prompt, override.candidate_set)
        decision = score(
            features=feats,
            quality_fit=qf,
            backends=cfg.backends,
            policy=cfg.policy,
            candidate_set=override.candidate_set,
            sticky_backend=sticky,
        )
        chosen = decision.chosen
        scores = decision.scores
        confidence = decision.confidence
        fallback = decision.fallback_used
        path = "scored"

    backend = next(b for b in cfg.backends if b.name == chosen)
    return {
        "chosen": chosen,
        "confidence": confidence,
        "fallback": fallback,
        "path": path,
        "n_tokens_est": feats.n_tokens_est,
        "expected_out_tokens": expected_out,
        "projected_cost_usd": project_cost_usd(backend, feats.n_tokens_est, expected_out),
        "projected_latency_ms": project_latency_ms(backend, expected_out),
        "tool_required": feats.tool_required,
        "has_url": feats.has_url,
        "code_fence_count": feats.code_fence_count,
        "scores_json": json.dumps(scores, sort_keys=True),
    }


def simulate_all(turns: list[dict], variants: dict[str, dict], base_cfg: dict, classifier: _Cls) -> list[dict]:
    """For each variant × each turn, run the pipeline. Sticky tracked per session."""
    rows: list[dict] = []
    for vname, overrides in variants.items():
        cfg = make_config(base_cfg, overrides)
        sticky_by_session: dict[str, str | None] = {}
        for t in turns:
            sticky = sticky_by_session.get(t["session_id"])
            sim = simulate_one(cfg, t["prompt"], sticky, classifier)
            sticky_by_session[t["session_id"]] = sim["chosen"]
            rows.append({
                "variant": vname,
                **{k: t[k] for k in ("agent", "session_id", "turn_idx", "row_ts", "observed_latency_ms")},
                "prompt_len": len(t["prompt"]),
                **sim,
            })
        sys.stderr.write(f"[replay] {vname}: {len(turns)} turns simulated\n")
    return rows


# ---------- I/O ----------

def write_csv(rows: list[dict], out: Path) -> None:
    if not rows:
        out.write_text("")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/tmp/dream_raw_logs.csv",
                    help="CSV exported from `wortz-project-352116.gemini_dreams.dream_raw_logs`")
    ap.add_argument("--config", default=str(ROOT / "config" / "router.yaml"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "experiments" / "assets" / "replay.csv"))
    args = ap.parse_args()

    turns = load_turns(Path(args.input))
    sys.stderr.write(f"[replay] loaded {len(turns)} user turns "
                     f"from {len(set(t['session_id'] for t in turns))} sessions\n")

    base = load_base_config(Path(args.config))
    # Build the classifier once; it's shared across variants since the model + anchors don't change.
    cfg0 = make_config(base, CONFIG_VARIANTS["default"])
    classifier = _Cls(cfg0)

    rows = simulate_all(turns, CONFIG_VARIANTS, base, classifier)
    write_csv(rows, Path(args.out))
    sys.stderr.write(f"[replay] wrote {len(rows)} rows -> {args.out}\n")


if __name__ == "__main__":
    main()
