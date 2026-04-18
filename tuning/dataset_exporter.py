"""Pull (prompt, teacher_response) pairs from BigQuery for a named cluster
and write SFTTrainer-format train/eval JSONL files.

Usage:
    python tuning/dataset_exporter.py --cluster cloudrun \
        --out tuning/datasets/cloudrun/v1/

    python tuning/dataset_exporter.py --cluster cloudrun \
        --out tuning/datasets/cloudrun/v1/ --synthesize 200
"""
from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


SEED = 42
TRAIN_FRAC = 0.8
DEFAULT_BQ_SOURCE = "wortz-project-352116.gemini_dreams.dream_raw_logs"


class ClusterMatch(BaseModel):
    mode: Literal["like", "regex"] = "like"
    case_sensitive: bool = False
    patterns: list[str]


class ClusterCfg(BaseModel):
    description: str = ""
    bq_source: str = DEFAULT_BQ_SOURCE
    teacher_priority: list[str] = Field(default_factory=lambda: ["claude", "gemini", "gemma4"])
    match: ClusterMatch
    synth_seed_prompts: list[str] = Field(default_factory=list)


class ClustersFile(BaseModel):
    version: int = 1
    clusters: dict[str, ClusterCfg]


def load_clusters(path: Path) -> ClustersFile:
    raw = yaml.safe_load(path.read_text())
    return ClustersFile.model_validate(raw)


# ---------- BQ extraction ----------

def _build_where(match: ClusterMatch) -> str:
    col = "log_content" if match.case_sensitive else "LOWER(log_content)"
    parts = []
    for pat in match.patterns:
        if match.mode == "like":
            p = pat if match.case_sensitive else pat.lower()
            parts.append(f"{col} LIKE @p_{len(parts)}")
        else:
            parts.append(f"REGEXP_CONTAINS({col}, @p_{len(parts)})")
    return " OR ".join(parts)


def _bq_params(match: ClusterMatch) -> list[str]:
    out = []
    for i, pat in enumerate(match.patterns):
        v = pat if match.case_sensitive else pat.lower()
        out.extend(["--parameter", f"p_{i}:STRING:{v}"])
    return out


def fetch_bq_rows(cluster: ClusterCfg) -> list[dict]:
    where = _build_where(cluster.match)
    sql = (
        "SELECT timestamp, agent_name, session_id, log_content "
        f"FROM `{cluster.bq_source}` "
        f"WHERE {where}"
    )
    argv = [
        "bq", "query", "--use_legacy_sql=false", "--format=json",
        "--max_rows=100000",
        *_bq_params(cluster.match),
        sql,
    ]
    proc = subprocess.run(argv, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"bq query failed: {proc.stderr.strip()}")
    return json.loads(proc.stdout or "[]")


# ---------- pair extraction from log_content ----------

def _pick_teacher_response(messages: list[dict], teacher_priority: list[str]) -> str | None:
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("content"):
            return str(msg["content"])
    return None


def _matches(text: str, match: ClusterMatch) -> bool:
    hay = text if match.case_sensitive else text.lower()
    for pat in match.patterns:
        needle = pat if match.case_sensitive else pat.lower()
        if match.mode == "like":
            re_pat = "^" + re.escape(needle).replace("%", ".*").replace("_", ".") + "$"
            if re.search(re_pat, hay, flags=re.DOTALL):
                return True
            if needle.strip("%") in hay:
                return True
        else:
            if re.search(needle, hay, flags=0 if match.case_sensitive else re.IGNORECASE):
                return True
    return False


def extract_pairs(rows: list[dict], cluster: ClusterCfg) -> list[dict]:
    pairs: list[dict] = []
    for row in rows:
        try:
            log = json.loads(row["log_content"])
        except (json.JSONDecodeError, TypeError):
            continue
        messages = log.get("messages") or []
        teacher = _pick_teacher_response(messages, cluster.teacher_priority)
        if not teacher:
            continue
        for i, msg in enumerate(messages):
            if msg.get("role") != "user" or not msg.get("content"):
                continue
            prompt = str(msg["content"]).strip()
            if not _matches(prompt, cluster.match):
                continue
            assistant_after = next(
                (m for m in messages[i + 1:] if m.get("role") == "assistant" and m.get("content")),
                None,
            )
            response = str(assistant_after["content"]) if assistant_after else teacher
            pairs.append({
                "prompt": prompt,
                "response": response,
                "session_id": row.get("session_id", ""),
                "agent_name": row.get("agent_name", ""),
                "source": "bq",
            })
    return pairs


# ---------- synthesis via gemini CLI ----------

_SYNTH_GEN_TEMPLATE = """You are generating training data for a small specialist language model.

CLUSTER DESCRIPTION:
{description}

SEED EXAMPLES OF IN-CLUSTER USER PROMPTS:
{seed_block}

Generate exactly {n} NEW user prompts that fit this cluster. They must be diverse
in phrasing and intent but stay strictly on-topic. Return ONLY a JSON array of
strings, no prose, no code fences.
"""

_SYNTH_ANSWER_TEMPLATE = """You are an expert SRE assistant. Answer the user's
question concisely and correctly. Prefer concrete commands and short rationale.

USER PROMPT:
{prompt}
"""


def _gemini(prompt: str, timeout: int = 180) -> str:
    proc = subprocess.run(
        ["gemini", "-p", prompt, "--output-format", "json"],
        capture_output=True, text=True, timeout=timeout, check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"gemini failed: {proc.stderr.strip()[:300]}")
    try:
        env = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return proc.stdout
    return env.get("response") or env.get("text") or ""


def _parse_json_array(text: str) -> list[str]:
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return [str(x).strip() for x in arr if str(x).strip()]


def synthesize_pairs(cluster: ClusterCfg, n: int) -> list[dict]:
    seed_block = "\n".join(f"- {p}" for p in cluster.synth_seed_prompts) or "- (none provided)"
    gen_prompt = _SYNTH_GEN_TEMPLATE.format(
        description=cluster.description or "(unspecified)",
        seed_block=seed_block,
        n=n,
    )
    raw = _gemini(gen_prompt)
    prompts = _parse_json_array(raw)[:n]
    out: list[dict] = []
    for p in prompts:
        try:
            response = _gemini(_SYNTH_ANSWER_TEMPLATE.format(prompt=p))
        except RuntimeError as e:
            print(f"  skip synth answer ({e})", file=sys.stderr)
            continue
        out.append({
            "prompt": p,
            "response": response,
            "session_id": "",
            "agent_name": "synthetic",
            "source": "synthetic",
        })
    return out


# ---------- write SFT format ----------

def _to_sft(pair: dict) -> dict:
    return {"messages": [
        {"role": "user", "content": pair["prompt"]},
        {"role": "assistant", "content": pair["response"]},
    ]}


def write_split(pairs: list[dict], out_dir: Path) -> tuple[int, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    shuffled = list(pairs)
    rng.shuffle(shuffled)
    cut = max(1, int(len(shuffled) * TRAIN_FRAC)) if len(shuffled) > 1 else len(shuffled)
    train, evalset = shuffled[:cut], shuffled[cut:]
    (out_dir / "train.jsonl").write_text(
        "\n".join(json.dumps(_to_sft(p), ensure_ascii=False) for p in train) + ("\n" if train else "")
    )
    (out_dir / "eval.jsonl").write_text(
        "\n".join(json.dumps(_to_sft(p), ensure_ascii=False) for p in evalset) + ("\n" if evalset else "")
    )
    (out_dir / "manifest.json").write_text(json.dumps({
        "seed": SEED,
        "train_frac": TRAIN_FRAC,
        "n_train": len(train),
        "n_eval": len(evalset),
        "sources": {
            s: sum(1 for p in pairs if p.get("source") == s)
            for s in {p.get("source", "unknown") for p in pairs}
        },
    }, indent=2))
    return len(train), len(evalset)


# ---------- main ----------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster", required=True)
    ap.add_argument("--clusters-file", type=Path,
                    default=Path(__file__).parent / "clusters.yaml")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--synthesize", type=int, default=0,
                    help="generate N additional synthetic pairs via `gemini -p`")
    ap.add_argument("--bq-only", action="store_true",
                    help="skip BQ fetch errors silently and continue with synth only")
    args = ap.parse_args()

    clusters = load_clusters(args.clusters_file)
    if args.cluster not in clusters.clusters:
        raise SystemExit(f"unknown cluster {args.cluster!r}; "
                         f"known: {sorted(clusters.clusters)}")
    cluster = clusters.clusters[args.cluster]

    print(f"[1/3] querying BigQuery `{cluster.bq_source}` for cluster={args.cluster}…", flush=True)
    try:
        rows = fetch_bq_rows(cluster)
    except RuntimeError as e:
        if args.bq_only:
            raise
        print(f"  bq query failed ({e}); continuing with 0 BQ rows", file=sys.stderr)
        rows = []
    pairs = extract_pairs(rows, cluster)
    print(f"      bq rows: {len(rows)}  →  in-cluster (prompt, response) pairs: {len(pairs)}")

    if args.synthesize > 0:
        print(f"[2/3] synthesizing {args.synthesize} pairs via gemini CLI…", flush=True)
        synth = synthesize_pairs(cluster, args.synthesize)
        print(f"      synthesized: {len(synth)}")
        pairs.extend(synth)
    else:
        print("[2/3] synthesize=0, skipping augmentation")

    if not pairs:
        raise SystemExit("no pairs to write; aborting")

    print(f"[3/3] writing 80/20 split (seed={SEED}) to {args.out}…", flush=True)
    n_train, n_eval = write_split(pairs, args.out)
    print(f"      wrote {n_train} train, {n_eval} eval rows")
    print(f"artifacts:\n  {args.out / 'train.jsonl'}\n  {args.out / 'eval.jsonl'}\n  {args.out / 'manifest.json'}")


if __name__ == "__main__":
    main()
