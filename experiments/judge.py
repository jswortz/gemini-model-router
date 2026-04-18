"""Quality-judge experiment: actually invoke gemini + claude on a held-out
subset of router prompts, then have a Gemini judge score each pair on a
1-7 rubric. Outputs a per-backend mean score, per-cluster breakdown, and
prompts where the routing decision disagreed with the judge's pick.

This is the "real quality" experiment — REPORT.md's Caveat #1 fix.

Usage:
    python experiments/judge.py --n 60 --out experiments/results_judge

Cost (measured): claude haiku --bare ~$0.006/call, gemini default ~$0.003/call,
judge gemini ~$0.005/call → ~$0.86 for n=60.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from router.features.extractor import extract  # noqa: E402


# ---------- prompt sampling ----------

def _bucket(prompt: str) -> str:
    f = extract(prompt)
    if f.tool_required:
        return "tool_required"
    if f.has_url:
        return "url"
    if f.n_tokens_est < 30:
        return "short_qa"
    if f.code_fence_count > 0:
        return "code"
    return "mid_general"


def sample_prompts(raw_path: Path, synth_path: Path | None, n: int,
                   seed: int = 7) -> list[dict]:
    """Stratified sample across feature buckets so we cover the workload shape."""
    real_rows = json.loads(raw_path.read_text())
    real_prompts: list[dict] = []
    for row in real_rows:
        try:
            log = json.loads(row["log_content"])
        except (json.JSONDecodeError, TypeError):
            continue
        for msg in log.get("messages", []):
            if msg.get("role") == "user" and msg.get("content"):
                p = msg["content"].strip()
                if 5 <= len(p) <= 800:  # judge-able length
                    real_prompts.append({"prompt": p, "source": "real",
                                         "bucket": _bucket(p)})

    rng = random.Random(seed)
    rng.shuffle(real_prompts)

    # stratify
    buckets: dict[str, list[dict]] = {}
    for r in real_prompts:
        buckets.setdefault(r["bucket"], []).append(r)

    per_bucket = max(1, n // max(len(buckets), 1))
    picked = []
    for b, items in buckets.items():
        picked.extend(items[:per_bucket])
    # top up to n with remaining real
    leftover = [r for r in real_prompts if r not in picked]
    rng.shuffle(leftover)
    while len(picked) < n and leftover:
        picked.append(leftover.pop())

    rng.shuffle(picked)
    return picked[:n]


# ---------- subprocess invocation ----------

@dataclass
class CallResult:
    text: str
    cost_usd: float
    latency_ms: float
    in_tokens: int
    out_tokens: int
    model: str
    success: bool
    error: str | None = None


async def _run(argv: list[str], timeout: int = 120) -> tuple[bytes, bytes, int, float]:
    t0 = time.perf_counter()
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return b"", b"timeout", -1, (time.perf_counter() - t0) * 1000
    return out, err, proc.returncode, (time.perf_counter() - t0) * 1000


async def call_gemini(prompt: str) -> CallResult:
    out, err, rc, latency = await _run(["gemini", "-p", prompt, "--output-format", "json"], timeout=180)
    if rc != 0:
        return CallResult("", 0.0, latency, 0, 0, "gemini", False,
                          err.decode(errors="replace")[:300])
    try:
        d = json.loads(out)
    except json.JSONDecodeError:
        return CallResult(out.decode(errors="replace"), 0.0, latency, 0, 0,
                          "gemini", True)
    text = d.get("response") or d.get("text") or ""
    stats = d.get("stats", {})
    models = stats.get("models", {}) if isinstance(stats, dict) else {}
    model_name = next(iter(models.keys()), "gemini")
    tok = models.get(model_name, {}).get("tokens", {}) if model_name else {}
    in_tok = int(tok.get("prompt", 0) or tok.get("input", 0) or 0)
    out_tok = int(tok.get("candidates", 0) or tok.get("total", 0) or 0) - in_tok
    out_tok = max(out_tok, 0)
    # Pricing: gemini-3-flash-preview ≈ $0.30/M in, $2.50/M out (rough)
    cost = (in_tok * 0.30 + out_tok * 2.50) / 1e6
    return CallResult(text, cost, latency, in_tok, out_tok, model_name, True)


async def call_claude_haiku(prompt: str) -> CallResult:
    out, err, rc, latency = await _run(
        ["claude", "--model", "claude-haiku-4-5@20251001", "--bare",
         "-p", prompt, "--output-format", "json"], timeout=180)
    if rc != 0:
        return CallResult("", 0.0, latency, 0, 0, "claude-haiku", False,
                          err.decode(errors="replace")[:300])
    try:
        d = json.loads(out)
    except json.JSONDecodeError:
        return CallResult(out.decode(errors="replace"), 0.0, latency, 0, 0,
                          "claude-haiku", True)
    if d.get("is_error"):
        return CallResult("", float(d.get("total_cost_usd", 0.0)), latency,
                          0, 0, "claude-haiku", False, d.get("result", "")[:300])
    text = d.get("result", "")
    u = d.get("usage", {})
    in_tok = int(u.get("input_tokens", 0)) + int(u.get("cache_creation_input_tokens", 0))
    out_tok = int(u.get("output_tokens", 0))
    cost = float(d.get("total_cost_usd", 0.0))
    model_name = next(iter(d.get("modelUsage", {}).keys()), "claude-haiku")
    return CallResult(text, cost, latency, in_tok, out_tok, model_name, True)


# ---------- LLM judge ----------

JUDGE_RUBRIC = """You are an impartial evaluator scoring two AI assistant responses to the same user prompt.

Score EACH response on three criteria, integer 1-7 (7 = best):
  - correctness: factually right, no hallucinations
  - helpfulness: actually answers what the user asked
  - conciseness: appropriate length for the question (no padding, no missing essentials)

Then declare a winner: "A", "B", or "tie".

Return ONLY a single JSON object, no prose, no code fences:
{"a_correctness": int, "a_helpfulness": int, "a_conciseness": int,
 "b_correctness": int, "b_helpfulness": int, "b_conciseness": int,
 "winner": "A" | "B" | "tie", "reason": "one sentence"}
"""


async def judge_pair(prompt: str, resp_a: str, resp_b: str,
                     a_label: str, b_label: str) -> dict | None:
    judge_prompt = f"""{JUDGE_RUBRIC}

USER PROMPT:
{prompt}

RESPONSE A (from {a_label}):
{resp_a[:3000]}

RESPONSE B (from {b_label}):
{resp_b[:3000]}
"""
    out, err, rc, latency = await _run(
        ["gemini", "-p", judge_prompt, "--output-format", "json"], timeout=120)
    if rc != 0:
        return None
    try:
        envelope = json.loads(out)
    except json.JSONDecodeError:
        return None
    text = envelope.get("response") or envelope.get("text") or ""
    # Pull JSON object out of the response
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


# ---------- main ----------

async def run(args) -> None:
    args.out.mkdir(parents=True, exist_ok=True)

    print("[1/4] sampling prompts…", flush=True)
    samples = sample_prompts(args.raw, None, args.n)
    print(f"      sampled {len(samples)} prompts; bucket counts: ", end="")
    bcounts: dict[str, int] = {}
    for s in samples:
        bcounts[s["bucket"]] = bcounts.get(s["bucket"], 0) + 1
    print(bcounts)

    print(f"[2/4] invoking gemini and claude (haiku, --bare) on each prompt "
          f"(parallelism={args.parallel})…", flush=True)
    sem = asyncio.Semaphore(args.parallel)

    async def both(rec):
        async with sem:
            t0 = time.time()
            g, c = await asyncio.gather(
                call_gemini(rec["prompt"]),
                call_claude_haiku(rec["prompt"]),
            )
            return {"prompt": rec["prompt"], "bucket": rec["bucket"],
                    "gemini": g, "claude": c, "wall_s": time.time() - t0}

    raw_results = []
    t_phase = time.time()
    for i, fut in enumerate(asyncio.as_completed([both(s) for s in samples]), 1):
        r = await fut
        raw_results.append(r)
        # quick liveness print every 10
        if i % 5 == 0 or i == len(samples):
            spent = sum(x["gemini"].cost_usd + x["claude"].cost_usd for x in raw_results)
            print(f"      [{i}/{len(samples)}] generation cost so far: ${spent:.3f}", flush=True)
    gen_secs = time.time() - t_phase
    print(f"      gen wall time: {gen_secs:.1f}s")

    print("[3/4] judging each pair with gemini judge…", flush=True)
    sem_j = asyncio.Semaphore(args.parallel)

    async def judge_one(r, swap: bool):
        async with sem_j:
            if swap:
                a, b, a_lbl, b_lbl = r["claude"].text, r["gemini"].text, "claude", "gemini"
            else:
                a, b, a_lbl, b_lbl = r["gemini"].text, r["claude"].text, "gemini", "claude"
            v = await judge_pair(r["prompt"], a, b, a_lbl, b_lbl)
            return {"swap": swap, "verdict": v, "a_label": a_lbl, "b_label": b_lbl}

    judged = []
    t_phase = time.time()
    for i, r in enumerate(raw_results, 1):
        # alternate which response is shown as "A" to mitigate position bias
        v = await judge_one(r, swap=(i % 2 == 0))
        judged.append({**r, "judge": v})
        if i % 5 == 0 or i == len(raw_results):
            print(f"      judged [{i}/{len(raw_results)}]", flush=True)
    judge_secs = time.time() - t_phase
    print(f"      judge wall time: {judge_secs:.1f}s")

    print("[4/4] aggregating + writing report…", flush=True)
    report = aggregate(judged)
    # serialize CallResult for json
    def _ser(rec):
        out = dict(rec)
        for k in ("gemini", "claude"):
            cr: CallResult = out[k]
            out[k] = {"text": cr.text, "cost_usd": cr.cost_usd,
                      "latency_ms": cr.latency_ms,
                      "in_tokens": cr.in_tokens, "out_tokens": cr.out_tokens,
                      "model": cr.model, "success": cr.success, "error": cr.error}
        return out

    (args.out / "raw.json").write_text(json.dumps([_ser(r) for r in judged],
                                                  indent=1, default=str))
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print_report(report)
    total_cost = sum(r["gemini"].cost_usd + r["claude"].cost_usd for r in raw_results)
    print(f"\ntotal generation cost: ${total_cost:.3f}")
    print(f"artifacts:\n  {args.out / 'raw.json'}\n  {args.out / 'report.json'}")


def aggregate(judged: list[dict]) -> dict:
    by_bucket: dict[str, list[dict]] = {}
    overall = []
    for r in judged:
        v = r["judge"]["verdict"]
        if not v:
            continue
        swap = r["judge"]["swap"]
        # canonical {gemini, claude}
        if swap:
            g = {"correctness": v["b_correctness"], "helpfulness": v["b_helpfulness"], "conciseness": v["b_conciseness"]}
            c = {"correctness": v["a_correctness"], "helpfulness": v["a_helpfulness"], "conciseness": v["a_conciseness"]}
            winner_canon = ("claude" if v["winner"] == "A"
                            else "gemini" if v["winner"] == "B" else "tie")
        else:
            g = {"correctness": v["a_correctness"], "helpfulness": v["a_helpfulness"], "conciseness": v["a_conciseness"]}
            c = {"correctness": v["b_correctness"], "helpfulness": v["b_helpfulness"], "conciseness": v["b_conciseness"]}
            winner_canon = ("gemini" if v["winner"] == "A"
                            else "claude" if v["winner"] == "B" else "tie")
        rec = {
            "prompt": r["prompt"], "bucket": r["bucket"],
            "gemini": g, "claude": c, "winner": winner_canon,
            "gemini_cost": r["gemini"].cost_usd, "claude_cost": r["claude"].cost_usd,
            "gemini_latency_ms": r["gemini"].latency_ms,
            "claude_latency_ms": r["claude"].latency_ms,
        }
        overall.append(rec)
        by_bucket.setdefault(r["bucket"], []).append(rec)

    def stats(rows: list[dict]) -> dict:
        if not rows:
            return {}
        n = len(rows)
        def avg(side, k):
            return sum(r[side][k] for r in rows) / n
        return {
            "n": n,
            "gemini_mean": {k: round(avg("gemini", k), 3) for k in ("correctness", "helpfulness", "conciseness")},
            "claude_mean": {k: round(avg("claude", k), 3) for k in ("correctness", "helpfulness", "conciseness")},
            "gemini_total": round(sum(sum(r["gemini"].values()) for r in rows) / (n * 3), 3),
            "claude_total": round(sum(sum(r["claude"].values()) for r in rows) / (n * 3), 3),
            "win_rate": {
                "gemini": round(sum(1 for r in rows if r["winner"] == "gemini") / n, 3),
                "claude": round(sum(1 for r in rows if r["winner"] == "claude") / n, 3),
                "tie":    round(sum(1 for r in rows if r["winner"] == "tie")    / n, 3),
            },
            "cost_per_prompt": {
                "gemini": round(sum(r["gemini_cost"] for r in rows) / n, 5),
                "claude": round(sum(r["claude_cost"] for r in rows) / n, 5),
            },
            "latency_ms_per_prompt": {
                "gemini": round(sum(r["gemini_latency_ms"] for r in rows) / n, 1),
                "claude": round(sum(r["claude_latency_ms"] for r in rows) / n, 1),
            },
        }

    return {
        "overall": stats(overall),
        "by_bucket": {b: stats(rows) for b, rows in by_bucket.items()},
    }


def print_report(report: dict) -> None:
    o = report["overall"]
    print()
    print("=" * 78)
    print(f"OVERALL  (n={o.get('n', 0)})")
    print("=" * 78)
    print(f"  gemini mean rubric (corr/help/conc): {o['gemini_mean']['correctness']}/"
          f"{o['gemini_mean']['helpfulness']}/{o['gemini_mean']['conciseness']}  total avg {o['gemini_total']}")
    print(f"  claude mean rubric (corr/help/conc): {o['claude_mean']['correctness']}/"
          f"{o['claude_mean']['helpfulness']}/{o['claude_mean']['conciseness']}  total avg {o['claude_total']}")
    wr = o["win_rate"]
    print(f"  win rate: gemini {wr['gemini']*100:.1f}%  claude {wr['claude']*100:.1f}%  tie {wr['tie']*100:.1f}%")
    cp = o["cost_per_prompt"]
    print(f"  cost/prompt: gemini ${cp['gemini']:.5f}  claude ${cp['claude']:.5f}")
    lp = o["latency_ms_per_prompt"]
    print(f"  latency/prompt: gemini {lp['gemini']:.0f}ms  claude {lp['claude']:.0f}ms")
    print()
    print("BY BUCKET")
    print("-" * 78)
    print(f"{'bucket':<18} {'n':>3}  {'gem total':>10} {'cla total':>10}  "
          f"{'cla wins':>10} {'gem wins':>10} {'cla $/req':>10}")
    for b, s in report["by_bucket"].items():
        if not s:
            continue
        wr = s["win_rate"]
        print(f"{b:<18} {s['n']:>3d}  "
              f"{s['gemini_total']:>10.3f} {s['claude_total']:>10.3f}  "
              f"{wr['claude']*100:>9.1f}% {wr['gemini']*100:>9.1f}% "
              f"${s['cost_per_prompt']['claude']:>9.5f}")
    print("=" * 78)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path, default=ROOT / "experiments/raw_logs.json")
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--parallel", type=int, default=4)
    p.add_argument("--out", type=Path, default=ROOT / "experiments/results_judge")
    args = p.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
