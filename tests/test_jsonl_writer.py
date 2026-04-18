import json
from pathlib import Path

from router.config_loader import LoggingCfg
from router.logging.jsonl_writer import JsonlWriter

# Field names gemini-dreams' get_recent_sessions reads off each line.
REQUIRED = {
    "timestamp", "session_id", "agent_name", "cli_type",
    "prompt", "prompt_response", "latency_ms", "skills",
}


def _writer(tmp_path: Path, redact: bool = False) -> JsonlWriter:
    return JsonlWriter(LoggingCfg(path=str(tmp_path / "log.jsonl"), redact_prompts=redact))


def _read(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text().splitlines()]


def test_jsonl_record_has_gemini_dreams_fields(tmp_path):
    w = _writer(tmp_path)
    w.write(
        session_id="01J123",
        prompt="hello",
        response_text="hi",
        latency_ms=42.0,
        router_block={"chosen_backend": "gemma4"},
        usage_block={"input_tokens": 1, "output_tokens": 1, "cost_usd": 0.0},
        backend_meta={"kind": "vllm", "model": "m", "raw_pruned": {}},
        workspace="/tmp/x",
        success=True,
    )
    recs = _read(w.path)
    assert len(recs) == 1
    r = recs[0]
    assert REQUIRED.issubset(r.keys())
    assert r["agent_name"] == "router"
    assert r["cli_type"] == "router"
    assert r["prompt"] == "hello"
    assert r["prompt_response"] == "hi"
    assert isinstance(r["skills"], dict)


def test_redaction_replaces_prompt_with_hash(tmp_path):
    w = _writer(tmp_path, redact=True)
    w.write(
        session_id="01J", prompt="secret",
        response_text="ok", latency_ms=1.0,
        router_block={}, usage_block={"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
        backend_meta={}, workspace="/tmp", success=True,
    )
    r = _read(w.path)[0]
    assert r["prompt"].startswith("sha256:")
    assert "secret" not in r["prompt"]


def test_appends_multiple_records(tmp_path):
    w = _writer(tmp_path)
    for i in range(3):
        w.write(
            session_id=f"s{i}", prompt=f"p{i}", response_text=f"r{i}",
            latency_ms=float(i),
            router_block={}, usage_block={"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
            backend_meta={}, workspace="/tmp", success=True,
        )
    assert len(_read(w.path)) == 3
