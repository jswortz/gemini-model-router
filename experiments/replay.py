"""Replay gemini-dreams BQ logs through the router decision logic under
multiple configs, and report cost/latency/quality with bootstrap CIs.

No backends are invoked. This is pure decision-replay: features → classifier
→ scorer. We then *estimate* cost/latency from the chosen backend's pricing
and the prompt's token estimate, and use the chosen-backend `quality_fit`
(from the MiniLM classifier) as a quality proxy.

Usage:
    python experiments/replay.py --raw experiments/raw_logs.json \
        --augment 1000 --bootstrap 2000 --out experiments/results
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from router.config_loader import RouterConfig, load_config, load_anchors  # noqa: E402
from router.features.extractor import extract  # noqa: E402
from router.policy import scorer as scorer_mod  # noqa: E402
from router.policy import rules as rules_mod  # noqa: E402
from router.classifier.embed_anchors import EmbedAnchorsClassifier  # noqa: E402


# ---------- data loading ----------

def parse_bq_rows(path: Path) -> list[dict]:
    """Returns a list of {prompt, agent_name, session_id, latency_ms, cli_type}."""
    rows = json.loads(path.read_text())
    out = []
    for row in rows:
        try:
            log = json.loads(row["log_content"])
        except (json.JSONDecodeError, TypeError):
            continue
        cli_type = log.get("cli_type") or row.get("agent_name")
        latency_ms = log.get("latency_ms")
        sid = row.get("session_id", "")
        for msg in log.get("messages", []):
            if msg.get("role") == "user" and msg.get("content"):
                out.append({
                    "prompt": msg["content"],
                    "session_id": sid,
                    "agent_name": row.get("agent_name"),
                    "cli_type": cli_type,
                    "latency_ms": latency_ms,
                })
    return out


# ---------- synthetic augmentation ----------

# Templates designed so the *empirical distribution* over (length, agent_keywords,
# code_ratio, has_path_ref, has_url) roughly matches what we see in the real
# gemini-dreams workload — short Q&A heavy, occasional refactor, occasional URL.
_SYNTHETIC_TEMPLATES = [
    # short Q&A
    "what is {topic}",
    "explain {topic} in one paragraph",
    "summarize {topic}",
    "give me the {topic} command",
    "how do I {action}",
    "what does {topic} mean",
    "list the steps to {action}",
    # mid coding
    "fix the bug in {path}",
    "refactor {path} to use {pattern}",
    "write a unit test for {path}",
    "implement a {action} function",
    # agentic / tool-required
    "trace through this stack trace and find the root cause:\n```\n{stacktrace}\n```",
    "grep the repo for {topic} and summarize the matches",
    "audit {path} for {topic} issues",
    # long-context / web
    "summarize https://example.com/{slug}",
    "search for the latest {topic} release notes",
    # gcloud / cloud-run heavy (matches your real data)
    "how do I deploy {topic} to cloud run",
    "what region should I use for {topic}",
    "give me the gcloud command to {action}",
]
_TOPICS = ["REST", "OAuth", "kubernetes", "BigQuery", "Vertex AI", "gemma 4",
           "vLLM", "FastAPI", "asyncio", "the singleton pattern", "MiniLM",
           "session affinity", "prompt caching", "MCP", "GKE", "Cloud Run",
           "Pub/Sub", "Cloud Storage", "Spanner", "the V8 isolate"]
_ACTIONS = ["deploy", "rollback", "set memory limits", "configure autoscaling",
            "create a service account", "grant IAM roles", "tail logs",
            "trigger a build", "schedule a job", "set up monitoring"]
_PATHS = ["src/auth/login.py", "lib/handlers/users.ts", "cmd/server/main.go",
          "internal/db/migrations.sql", "frontend/components/Nav.tsx",
          "scripts/bootstrap.sh", "config/router.yaml"]
_PATTERNS = ["async/await", "context managers", "dependency injection",
             "the repository pattern", "structured logging"]
_STACKTRACE = ("Traceback (most recent call last):\n"
               "  File \"x.py\", line 42, in handler\n"
               "    return svc.do(req)\n"
               "AttributeError: 'NoneType' object has no attribute 'do'")
_SLUGS = ["k8s-release-notes", "gemma-4-launch", "vertex-ai-changes",
          "agent-engine-overview"]


def synthesize(n: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        tpl = rng.choice(_SYNTHETIC_TEMPLATES)
        prompt = tpl.format(
            topic=rng.choice(_TOPICS),
            action=rng.choice(_ACTIONS),
            path=rng.choice(_PATHS),
            pattern=rng.choice(_PATTERNS),
            stacktrace=_STACKTRACE,
            slug=rng.choice(_SLUGS),
        )
        out.append({
            "prompt": prompt,
            "session_id": f"synth-{i}",
            "agent_name": "synthetic",
            "cli_type": "synthetic",
            "latency_ms": None,
        })
    return out


# ---------- config variants ----------

def _clone_with(cfg: RouterConfig, *, w_q=None, w_c=None, w_l=None,
                fallback=None, bonuses: dict | None = None) -> RouterConfig:
    new = copy.deepcopy(cfg)
    if w_q is not None: new.policy.weights.quality = w_q
    if w_c is not None: new.policy.weights.cost = w_c
    if w_l is not None: new.policy.weights.latency = w_l
    if fallback is not None: new.policy.fallback_backend = fallback
    if bonuses is not None:
        for k, v in bonuses.items():
            setattr(new.policy.capability_bonuses, k, v)
    return new


def build_variants(base: RouterConfig) -> dict[str, RouterConfig]:
    no_bonuses = {"local_short": 0.0, "agentic_tool": 0.0,
                  "long_ctx": 0.0, "tools_url": 0.0}
    softer_bonuses = {"local_short": 0.15, "agentic_tool": 0.15,
                      "long_ctx": 0.15, "tools_url": 0.05}
    return {
        "baseline":          _clone_with(base),
        "cost_tilted":       _clone_with(base, w_q=1.0, w_c=1.0, w_l=0.2),
        "quality_tilted":    _clone_with(base, w_q=1.5, w_c=0.1, w_l=0.1),
        "latency_tilted":    _clone_with(base, w_q=1.0, w_c=0.2, w_l=1.0),
        "no_bonuses":        _clone_with(base, bonuses=no_bonuses),
        "softer_bonuses":    _clone_with(base, bonuses=softer_bonuses),
        "cost_no_bonuses":   _clone_with(base, w_q=1.0, w_c=1.0, w_l=0.2, bonuses=no_bonuses),
        "qual_no_bonuses":   _clone_with(base, w_q=1.5, w_c=0.1, w_l=0.1, bonuses=no_bonuses),
    }


# ---------- replay ----------

def _expected_out(n_in: int) -> int:
    return min(2 * n_in, 1024)


def _est_cost_usd(b, n_in: int) -> float:
    out = _expected_out(n_in)
    return (n_in * b.cost_in_per_1m + out * b.cost_out_per_1m) / 1e6


def _est_latency_ms(b, n_in: int) -> float:
    out = _expected_out(n_in)
    return (out / 1000.0) * b.expected_latency_ms_per_1k_out


def replay_one(prompt: str, cfg: RouterConfig, classifier,
               health: dict[str, bool], force_backend: str | None = None) -> dict:
    features = extract(prompt)
    by_name = {b.name: b for b in cfg.backends}

    if force_backend is not None:
        chosen = force_backend
        quality_fit = classifier.classify(prompt)
        scores = {chosen: float(quality_fit.get(chosen, 0.0))}
    else:
        override = rules_mod.apply(features, cfg.backends, cfg.policy,
                                   force=None, healthy=health)
        if override.chosen is not None:
            chosen = override.chosen
            scores = {chosen: 1.0}
            quality_fit = classifier.classify(prompt)
        else:
            quality_fit = classifier.classify(prompt)
            decision = scorer_mod.score(
                features, quality_fit, cfg.backends, cfg.policy,
                candidate_set=override.candidate_set, sticky_backend=None,
            )
            chosen = decision.chosen
            scores = decision.scores

    b = by_name[chosen]
    return {
        "chosen": chosen,
        "cost_usd": _est_cost_usd(b, features.n_tokens_est),
        "latency_ms": _est_latency_ms(b, features.n_tokens_est),
        "quality_fit": float(quality_fit.get(chosen, 0.0)),
        "n_tokens_est": features.n_tokens_est,
    }


# ---------- bootstrap ----------

def bootstrap_ci(values: list[float], n_boot: int, seed: int = 0,
                 ci: float = 0.95) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return (math.nan, math.nan, math.nan)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    n = len(arr)
    for i in range(n_boot):
        sample = arr[rng.integers(0, n, n)]
        means[i] = sample.mean()
    lo = float(np.quantile(means, (1 - ci) / 2))
    hi = float(np.quantile(means, 1 - (1 - ci) / 2))
    return (float(arr.mean()), lo, hi)


def welch_t(a: list[float], b: list[float]) -> tuple[float, float]:
    """Returns (t-stat, approximate two-sided p) using Welch's formula and
    a normal approximation (good enough for n>=30 and a proxy here)."""
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if len(a_arr) < 2 or len(b_arr) < 2:
        return (math.nan, math.nan)
    ma, mb = a_arr.mean(), b_arr.mean()
    va, vb = a_arr.var(ddof=1), b_arr.var(ddof=1)
    se = math.sqrt(va / len(a_arr) + vb / len(b_arr))
    if se == 0:
        return (math.nan, math.nan)
    t = (ma - mb) / se
    # normal approx for p-value
    from math import erf, sqrt
    p = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(t) / sqrt(2.0))))
    return (t, p)


# ---------- main ----------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path, default=ROOT / "experiments/raw_logs.json")
    p.add_argument("--config", type=Path, default=ROOT / "config/router.yaml")
    p.add_argument("--augment", type=int, default=1000)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--out", type=Path, default=ROOT / "experiments/results")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print("[1/5] loading real prompts from BQ dump…", flush=True)
    real = parse_bq_rows(args.raw)
    print(f"      real prompts: {len(real)}")

    print(f"[2/5] synthesizing {args.augment} prompts to reach significance…", flush=True)
    synth = synthesize(args.augment)
    all_prompts = real + synth
    print(f"      total prompts: {len(all_prompts)} ({len(real)} real, {len(synth)} synthetic)")

    print("[3/5] loading router config + warming MiniLM classifier…", flush=True)
    base_cfg = load_config(args.config)
    classifier = EmbedAnchorsClassifier(base_cfg.classifier)
    # Force model load + warmup
    _ = classifier.classify("warmup")
    health = {b.name: True for b in base_cfg.backends}

    variants = build_variants(base_cfg)
    naive_baselines = ["gemma4", "gemini", "claude"]

    print("[4/5] replaying prompts through every config…", flush=True)
    results: dict[str, list[dict]] = {}
    t0 = time.time()

    for name, cfg in variants.items():
        rows = []
        for rec in all_prompts:
            r = replay_one(rec["prompt"], cfg, classifier, health)
            r["source"] = "real" if rec["session_id"].startswith(("synth-",)) is False else "synth"
            r["source"] = "synth" if rec["session_id"].startswith("synth-") else "real"
            rows.append(r)
        results[name] = rows
        print(f"      router_{name}: done")

    for forced in naive_baselines:
        rows = []
        for rec in all_prompts:
            r = replay_one(rec["prompt"], base_cfg, classifier, health,
                           force_backend=forced)
            r["source"] = "synth" if rec["session_id"].startswith("synth-") else "real"
            rows.append(r)
        results[f"all_{forced}"] = rows
        print(f"      all_{forced}: done")

    print(f"      replay wall time: {time.time() - t0:.1f}s")

    print("[5/5] computing bootstrap CIs and writing report…", flush=True)
    report = build_report(results, args.bootstrap)
    (args.out / "results.json").write_text(json.dumps(results, default=str, indent=1))
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print_report(report)
    print(f"\nartifacts:\n  {args.out / 'results.json'}\n  {args.out / 'report.json'}")


def build_report(results: dict[str, list[dict]], n_boot: int) -> dict:
    report: dict = {"variants": {}, "comparisons": {}}
    for variant, rows in results.items():
        v = {}
        for subset_name, subset in [
            ("all", rows),
            ("real_only", [r for r in rows if r["source"] == "real"]),
        ]:
            costs = [r["cost_usd"] for r in subset]
            lats = [r["latency_ms"] for r in subset]
            qfits = [r["quality_fit"] for r in subset]
            mix: dict[str, int] = {}
            for r in subset:
                mix[r["chosen"]] = mix.get(r["chosen"], 0) + 1
            v[subset_name] = {
                "n": len(subset),
                "cost_usd_per_prompt": bootstrap_ci(costs, n_boot, seed=1),
                "latency_ms_per_prompt": bootstrap_ci(lats, n_boot, seed=2),
                "quality_fit_mean": bootstrap_ci(qfits, n_boot, seed=3),
                "total_cost_usd": sum(costs),
                "backend_mix": mix,
            }
        report["variants"][variant] = v

    base = "baseline"
    if base in results:
        for other in ["all_claude", "all_gemini", "all_gemma4",
                      "cost_tilted", "quality_tilted", "latency_tilted",
                      "no_bonuses", "softer_bonuses",
                      "cost_no_bonuses", "qual_no_bonuses"]:
            if other not in results:
                continue
            a = [r["cost_usd"] for r in results[base]]
            b = [r["cost_usd"] for r in results[other]]
            t, pv = welch_t(a, b)
            report["comparisons"][f"baseline_vs_{other}_cost"] = {
                "mean_baseline": float(np.mean(a)),
                "mean_other": float(np.mean(b)),
                "delta_per_prompt": float(np.mean(a) - np.mean(b)),
                "delta_pct_vs_other": (
                    100.0 * (np.mean(a) - np.mean(b)) / max(np.mean(b), 1e-12)
                ),
                "welch_t": t,
                "approx_p": pv,
            }
    return report


def print_report(report: dict) -> None:
    print()
    print("=" * 78)
    print("CONFIG VARIANT COMPARISON  (bootstrap 95% CIs over per-prompt means)")
    print("=" * 78)
    header = f"{'variant':<18} {'n':>5} {'cost/req $':>22} {'latency ms':>22} {'qfit':>16}"
    print(header)
    print("-" * len(header))
    for variant, v in report["variants"].items():
        a = v["all"]
        c = a["cost_usd_per_prompt"]
        l = a["latency_ms_per_prompt"]
        q = a["quality_fit_mean"]
        print(f"{variant:<18} {a['n']:>5d} "
              f"{c[0]:>9.5f} [{c[1]:.5f},{c[2]:.5f}]  "
              f"{l[0]:>7.1f} [{l[1]:.1f},{l[2]:.1f}]  "
              f"{q[0]:>5.3f} [{q[1]:.3f},{q[2]:.3f}]")
    print()
    print("BACKEND MIX (% of prompts routed to each backend, ALL data)")
    print("-" * 78)
    print(f"{'variant':<18} {'gemma4':>10} {'gemini':>10} {'claude':>10}")
    for variant, v in report["variants"].items():
        m = v["all"]["backend_mix"]
        n = v["all"]["n"] or 1
        g = 100.0 * m.get("gemma4", 0) / n
        gm = 100.0 * m.get("gemini", 0) / n
        cl = 100.0 * m.get("claude", 0) / n
        print(f"{variant:<18} {g:>9.1f}% {gm:>9.1f}% {cl:>9.1f}%")
    print()
    print("REAL-DATA-ONLY SUBSET  (n is small, CIs will be wider)")
    print("-" * 78)
    print(f"{'variant':<18} {'n':>5} {'cost/req $':>22} {'latency ms':>22}")
    for variant, v in report["variants"].items():
        a = v["real_only"]
        c = a["cost_usd_per_prompt"]
        l = a["latency_ms_per_prompt"]
        print(f"{variant:<18} {a['n']:>5d} "
              f"{c[0]:>9.5f} [{c[1]:.5f},{c[2]:.5f}]  "
              f"{l[0]:>7.1f} [{l[1]:.1f},{l[2]:.1f}]")
    print()
    print("BASELINE vs OTHER  (Welch's t on cost; negative delta = baseline cheaper)")
    print("-" * 78)
    print(f"{'comparison':<40} {'Δ$/req':>10} {'Δ%':>8} {'t':>8} {'~p':>10}")
    for k, v in report["comparisons"].items():
        print(f"{k:<40} {v['delta_per_prompt']:>+10.5f} "
              f"{v['delta_pct_vs_other']:>+7.1f}% {v['welch_t']:>+8.2f} {v['approx_p']:>10.4f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
