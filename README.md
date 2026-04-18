# gemini-model-router

A local-first agentic terminal that routes each prompt across **local Gemma 4
(vLLM)**, **Gemini CLI**, and **Claude Code** — picking the right one to
optimize cost, latency, and quality. Logs every decision in a format the
[gemini-dreams](https://github.com/jswortz/gemini-dreams) nightly loop can mine
to propose its own improvements.

## Install

```bash
cd gemini-model-router
pip install -e ".[dev]"
```

This wires three console scripts: `router`, `router-repl`, `router-eval`.

## Prerequisites

- Local **vLLM** serving Gemma 4 on `localhost:8000`:
  ```bash
  vllm serve google/gemma-4-it --port 8000
  ```
- `gemini` CLI on `PATH`, with `GEMINI_API_KEY` (or Google Sign-in) configured.
- `claude` CLI on `PATH`, with `ANTHROPIC_API_KEY` configured.

Override any of these in `config/router.yaml`.

## Usage

One-shot:

```bash
router "what is the capital of France"
router "refactor src/auth/login.py to use async/await" --why
router --force gemini "search the web for the latest k8s release notes"
```

Interactive REPL:

```bash
router-repl
>>> what is REST
>>> /why
>>> /route claude
>>> implement a POST /users/{id}/avatar endpoint
>>> /quit
```

## How routing works

1. **Hard rules** (`policy/rules.py`) — secrets force local, `--force` flag
   wins, candidates with insufficient context window or that are unhealthy are
   dropped.
2. **Embedding classifier** (`classifier/embed_anchors.py`) — MiniLM cosine
   against per-backend anchor centroids in `config/anchors.yaml` →
   `quality_fit[backend]`.
3. **Weighted scorer** (`policy/scorer.py`) — combines `quality_fit`,
   normalized cost, normalized latency, and per-backend capability bonuses
   (e.g., `+0.4` to `claude` when the prompt looks tool-required).
4. **Tie-break** — within `confidence_margin` (default 0.05), fall back to the
   safest cheapest backend (`gemma4`).

## Session affinity

A multi-turn conversation should not bounce between vendors — that loses prompt
caching, MCP tool registries, and (worst of all) the model's own memory of the
conversation so far. The router enforces this with a small state machine in
`src/router/session.py`:

- **Turn 1**: full classify → score → dispatch. The chosen backend is *locked*
  to that `session_id`. The vendor's own session id (e.g. Claude Code's
  `session_id` from the JSON envelope) is captured.
- **Turn 2+**: route directly to the locked backend. Pass the vendor session id
  back through the CLI's `--resume` flag (`resume_flag` in `config/router.yaml`,
  per backend) so the vendor preserves its prompt cache and tool state.
- **Shadow classification**: the embedding classifier still runs on every turn
  and its choice is logged as `router.shadow_choice` — that's training data for
  the v2 classifier without disturbing the conversation.
- **Opt out**: `/unlock` in the REPL releases the lock for the next turn;
  `--no-affinity` on the CLI bypasses the lock for one-shot probes; `--force`
  always wins.

The `Backend` Protocol takes an optional `vendor_session_id`; vLLM ignores it
(no native session), `claude_cli` and `gemini_cli` prepend it as
`[resume_flag, vendor_session_id]`. Per-vendor session ids are tracked
independently, so swapping back to a previously-used vendor mid-session resumes
*that* vendor's thread.

## Self-improvement loop

Every call appends a line to `~/.router/sessions/router_history.jsonl` in the
exact shape `gemini-dreams` reads. Wire it with one config edit:

```jsonc
// ~/.gemini-dreams/config.json
{
  "agents": {
    "router": { "logs_dir": "~/.router/sessions", "turn_threshold": 1 }
  }
}
```

Then nightly:

```bash
dream run --days 1
```

To approve an analyzer-proposed anchor edit:

```bash
router-eval list
router-eval show 42
router-eval approve 42
router-eval rebuild-anchors
```

The convention the analyzer follows in its `skill_updates` field is one
directive per line:

```
move-to gemma4: "what is the difference between let and const"
move-to claude: "trace through this stack trace and find the root cause"
```

## Sandboxing

Each backend invocation runs in a fresh `tempfile.mkdtemp("router-ws-")` with a
scrubbed environment (only `PATH`, `HOME`, `LANG`, and the per-backend auth env
vars are forwarded). This adopts the *pattern* from
[GoogleCloudPlatform/scion](https://github.com/GoogleCloudPlatform/scion) — a
multi-agent sandbox harness — without taking the dependency. The
`DockerWorkspace` class in `sandbox/workspace.py` is a Phase-2 stub for true
container isolation.

## Phase-2 migration path

The v1 invariant is: **CLI-shaped concerns never leak past the `Backend`
interface**. That keeps the upgrade mechanical:

| v1 | Phase-2 |
| --- | --- |
| Python `EmbedAnchorsClassifier` | `vllm-project/semantic-router` Envoy ExtProc as a sidecar |
| `tempfile.mkdtemp` workspace | `DockerWorkspace` → scion-style ephemeral containers |
| JSONL writer | Add a `BigQueryLogWriter` next to it |
| Single-process REPL | FastAPI `/route` behind GKE Inference Gateway, with `InferencePool`/`InferenceObjective` selecting vLLM replicas while this router still picks the *family* |

## Tests

```bash
pytest
```

Pinned to the load-bearing logic only: feature extraction, scorer math, JSONL
shape conformance.

## Why not SCION (the protocol)

`scionproto/scion` is a layer-3 inter-AS routing scheme — wrong layer for
picking which API to call. The repo the original brief linked
(`GoogleCloudPlatform/scion`) is unrelated agent-orchestration code, whose
useful idea (per-agent isolated workspaces) is what `sandbox/workspace.py`
adopts above.
