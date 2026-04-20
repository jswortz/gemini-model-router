# CLI reference

The router ships four console scripts. Most users only ever need `router`;
`router config` is the second-most-used surface, and `router-config` /
`router-eval` are specialty entry points.

## Install

**Recommended — install as a uv tool** so the scripts land on `PATH` and
work from any directory in any new shell:

```bash
cd gemini-model-router
uv tool install -e .
uv tool update-shell        # one-time PATH fix; open a new terminal after
```

`-e` keeps the install editable, so edits to `src/router/...` and changes to
`config/router.yaml` take effect immediately. To pull in the FastAPI web UI
or dev tooling (ruff, pytest), pass extras through:

```bash
uv tool install -e ".[configsite,dev]"
```

Upgrade in place after pulling new commits:

```bash
uv tool upgrade router       # picks up new entry points / deps
```

**Alternative — editable install into a project venv** (you must activate
the venv to use the scripts, or add `.venv/bin` to `PATH`):

```bash
cd gemini-model-router
uv pip install -e ".[dev]"
source .venv/bin/activate    # or: export PATH="$PWD/.venv/bin:$PATH"
```

### Environment / `.env`

On startup the CLI auto-loads a `.env` from (in order) the current
directory, the directory containing `router.yaml`, the parent of that
directory, and finally the repo root. Existing shell env vars always win,
so an explicit `export HF_TOKEN=...` overrides whatever's in the file.

Copy the template and fill in what you need — every key is optional:

```bash
cp .env.example .env
# then edit .env
```

The keys the router and its backends look for:

| Key | Purpose |
| --- | --- |
| `HF_TOKEN` | HuggingFace read token. Silences the unauthenticated-Hub warning and raises rate limits when the MiniLM anchor cache is rebuilt. |
| `GEMINI_API_KEY` | Picked up by the `gemini` CLI backend if not already authenticated. |
| `ANTHROPIC_API_KEY` | Picked up by the `claude` CLI backend if not already authenticated. |
| `TRANSFORMERS_VERBOSITY` | Set to `error` to suppress the cosmetic `embeddings.position_ids UNEXPECTED` warning from sentence-transformers. |

Either way wires four console scripts (see `pyproject.toml` `[project.scripts]`):

| Script          | Module entry              | What it does                                                                 |
| --------------- | ------------------------- | ---------------------------------------------------------------------------- |
| `router`        | `router.cli:main`         | One-shot prompt dispatch, or chat REPL when called with no args at a TTY.    |
| `router-repl`   | `router.repl:main`        | Legacy alias for `router` (no args). Always opens the chat REPL.             |
| `router-eval`   | `router.eval.review:main` | Inspect / approve analyzer-proposed anchor edits from the dreams loop.       |
| `router-config` | `router.configsite.server:main` | FastAPI web UI for editing `router.yaml` + `anchors.yaml`. Needs `[configsite]` extra. |

---

## One-shot mode

```bash
router "<prompt>"
```

If `prompt` is omitted the router reads stdin, *unless* stdin is a TTY — in
which case it drops into chat mode (see below).

| Flag            | Type / values                  | Description                                                                  |
| --------------- | ------------------------------ | ---------------------------------------------------------------------------- |
| `--config`      | path to `router.yaml`          | Override config discovery. Defaults to `find_default_config()`.              |
| `--force`       | `gemma4` \| `gemini` \| `claude` | Bypass classification and dispatch directly to one backend.                |
| `--no-stream`   | flag                           | Disable terminal streaming; print the full response after completion.        |
| `--why`         | flag                           | After answering, print backend / confidence / scores to stderr.              |
| `--session`     | session id string              | Reuse a prior `session_id` for sticky routing (vendor resume).               |
| `--no-affinity` | flag                           | Skip session affinity: re-classify even if the session is locked.            |

Examples:

```bash
# Default route + stream
router "what is the capital of France"

# Explain the choice
router "refactor src/auth/login.py to use async/await" --why

# Force a backend
router --force gemini "search the web for the latest k8s release notes"

# Disable streaming (useful when piping into another tool)
router --no-stream "summarize this README" < README.md

# Sticky multi-turn under one session id
router --session 01HX… "list the failing tests"
router --session 01HX… "now fix the first one"

# Probe what classification *would* pick, ignoring the lock
router --session 01HX… --no-affinity --why "draw an ASCII flowchart"

# Custom config
router --config ./team-router.yaml "kick off the deploy"
```

Exit codes: `0` on a successful response, `1` if the backend returned
`success=False`, `2` for an empty prompt.

---

## Chat mode

```bash
router          # no args + stdin is a TTY → chat REPL
router-repl     # always chat REPL
```

### Banner

The opening banner shows:

- `config: <absolute path to router.yaml>`
- `session: <ulid>…` (truncated to 10 chars)
- The list of backends with cost (`$in/$out per 1M`) and `ctx=<max_context>`,
  colored per backend (gemma4=cyan, gemini=magenta, claude=yellow).
- A hint to type `/help` or `?`.

### Bottom toolbar

A persistent toolbar updates after every turn:

```
 session 01HX…  ·  turn 7  ·  last: claude 1820ms $0.0143  ·  forced→gemini  ·  /help
```

Fields: current session id (truncated), turn counter, last backend +
latency (ms) + cost (USD), and the active forced route (only shown when
`/route` is set for the next turn).

### Multi-line input heuristic

`router` uses a `prompt_toolkit` `PromptSession` with a custom Enter binding:

- **Enter** submits — *unless* the buffer wants continuation, in which case
  it inserts a newline.
- The buffer "wants continuation" if either:
  - it ends with a trailing backslash (`\`), or
  - it contains an odd number of triple-backtick fences (`` ``` ``).
- **Alt-Enter** force-submits regardless.
- **Up / Down** scroll history (persisted across sessions in
  `~/.router/repl_history`).
- **Tab** completes slash commands (fuzzy).

This means you can paste a multi-line code fence and it will hold until you
close the fence, then Enter submits.

---

## Slash commands

All slash commands work inside chat mode. `?` is an alias for `/help`.

| Command              | Args                              | Behavior                                                                                          | Example                         |
| -------------------- | --------------------------------- | ------------------------------------------------------------------------------------------------- | ------------------------------- |
| `/help`              | —                                 | Print the slash-command reference and editing hints.                                              | `/help`                         |
| `/why`               | —                                 | Show the last turn's chosen backend, confidence, override, fallback, affinity, and per-backend scores. | `/why`                          |
| `/route`             | `gemma4` \| `gemini` \| `claude`  | Force a single backend for the *next* prompt only. Cleared after one turn.                        | `/route claude`                 |
| `/unroute`           | —                                 | Clear a pending `/route` force.                                                                   | `/unroute`                      |
| `/unlock`            | —                                 | Release session affinity. The next turn will re-classify even if the session was locked.          | `/unlock`                       |
| `/regen`             | —                                 | Re-dispatch the previous prompt verbatim (logged as `exact_re_ask`).                              | `/regen`                        |
| `/sessions`          | optional `N` (default 10)         | List the most recent N session ids from the JSONL log, with turn counts and prompt previews.      | `/sessions 20`                  |
| `/resume`            | `<session_id>`                    | Switch the chat to a prior session id. Subsequent turns route under that session.                 | `/resume 01HXABCDEF…`           |
| `/fork`              | —                                 | Mint a new session id (drops affinity). Useful for starting a fresh thread without exiting.       | `/fork`                         |
| `/get`               | `<dotted.path>`                   | Read a config value from `router.yaml`.                                                           | `/get policy.weights.cost`      |
| `/set`               | `<dotted.path> <yaml-value>`      | Write a config value (validated, backed up, hot-reloaded into the running orchestrator).          | `/set policy.weights.cost 0.7`  |
| `/config`            | —                                 | Pretty-print the full `router.yaml`.                                                              | `/config`                       |
| `/quit`              | —                                 | Exit the REPL (Ctrl-D / Ctrl-C also work).                                                        | `/quit`                         |

`/set` reloads the orchestrator's in-memory config after a successful write,
so subsequent turns honor the new value without a restart.

---

## `router config` subcommand reference

All edits go through `router.config_writer`: pydantic-validated, atomic
write via `os.replace`, and a timestamped backup snapshot next to the file.

| Command                                       | Behavior                                                                                                          | Example                                                              |
| --------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| `router config show`                          | Pretty-print the full `router.yaml`.                                                                              | `router config show`                                                 |
| `router config get [path]`                    | Read a value at a dotted path. Empty path returns the whole tree. Lists of dicts with `name` are name-keyed.      | `router config get policy.weights.cost`                              |
| `router config set <path> <value>`            | Set a value. `<value>` is parsed as YAML, then the whole config re-validated through `RouterConfig`.              | `router config set policy.weights.cost 0.7`                          |
| `router config edit`                          | Open `router.yaml` in `$EDITOR` (or `$VISUAL`, or `vi`). On save, validates; backs up; writes atomically.         | `EDITOR=nvim router config edit`                                     |
| `router config rebuild-anchors`               | Re-embed the anchors.yaml exemplars and refresh the centroid cache used by the embed-anchors classifier.          | `router config rebuild-anchors`                                      |
| `router config backups list`                  | List backup snapshots (`router.yaml.bak-<utc-timestamp>`), newest first.                                          | `router config backups list`                                         |
| `router config backups restore <name>`        | Validate the named backup, snapshot the current file, then copy the backup over `router.yaml`.                    | `router config backups restore router.yaml.bak-20260420T143012001Z`  |
| `router config anchor list [backend]`         | List anchors for one backend, or all backends grouped by label.                                                   | `router config anchor list claude`                                   |
| `router config anchor add <backend> <text> [--rebuild]` | Append an exemplar string to that backend's list. Pass `--rebuild` to refresh centroids in one step.    | `router config anchor add gemma4 "what does CRUD stand for"`         |
| `router config anchor remove <backend> <i> [--rebuild]` | Drop the anchor at index `i` (negative indices supported).                                              | `router config anchor remove gemini 3`                               |

Common flags:

- `--config <path>` — override config discovery (also valid on every
  `router config ...` form: `router config --config ./other.yaml show`).

Exit codes: `0` on success, `2` on any error (missing key, invalid YAML,
schema violation, missing backup, anchor index out of range).

---

## Configurable keys (cheatsheet)

The most common dotted paths, with defaults from `config/router.yaml`:

| Path                                                            | Default                              | What it does                                                                                          |
| --------------------------------------------------------------- | ------------------------------------ | ----------------------------------------------------------------------------------------------------- |
| `policy.weights.quality`                                        | `1.0`                                | Weight on the classifier's `quality_fit[backend]` term.                                               |
| `policy.weights.cost`                                           | `0.4`                                | Weight on normalized cost. Higher = stronger preference for cheaper backends.                         |
| `policy.weights.latency`                                        | `0.3`                                | Weight on normalized expected latency.                                                                |
| `policy.confidence_margin`                                      | `0.05`                               | Within this score gap, tie-break to the safest cheapest backend (`fallback_backend`).                 |
| `policy.cost_ceiling_usd_per_request`                           | `0.10`                               | Drop candidates whose estimated request cost exceeds this.                                            |
| `policy.fallback_backend`                                       | `gemma4`                             | Backend used for tie-breaks and as the safety floor.                                                  |
| `policy.sticky_bonus`                                           | `0.05`                               | Score bonus applied to the locked backend during shadow re-classification (keeps sessions sticky).   |
| `policy.capability_bonuses.local_short`                         | `0.5`                                | Bonus when backend has the `local` capability and the prompt is short / no fences / no tools.         |
| `policy.capability_bonuses.agentic_tool`                        | `0.4`                                | Bonus when backend has `agentic` and the prompt is tool-required.                                     |
| `policy.capability_bonuses.long_ctx`                            | `0.3`                                | Bonus when backend has `long_ctx` and the prompt exceeds the long-ctx token threshold.                |
| `policy.capability_bonuses.tools_url`                           | `0.1`                                | Bonus when backend has `tools` and the prompt contains a URL.                                         |
| `policy.capability_bonuses.local_short_token_threshold`         | `200`                                | Token boundary for the `local_short` bonus.                                                           |
| `policy.capability_bonuses.long_ctx_token_threshold`            | `8000`                               | Token boundary for the `long_ctx` bonus.                                                              |
| `classifier.softmax_temp`                                       | `0.07`                               | Temperature for the cosine→softmax over anchor centroids. Lower = sharper.                            |
| `classifier.model`                                              | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model used for anchors + prompts.                                                       |
| `logging.path`                                                  | `~/.router/sessions/router_history.jsonl` | Where routing telemetry is appended.                                                             |
| `logging.redact_prompts`                                        | `false`                              | If true, store hashes instead of raw prompt text.                                                     |
| `backends.<name>.cost_in_per_1m` / `cost_out_per_1m`            | per-backend                          | USD per million input / output tokens. Drives the cost term + cost ceiling.                           |
| `backends.<name>.expected_latency_ms_per_1k_out`                | per-backend                          | Expected ms per 1k output tokens; drives the latency term.                                            |
| `backends.<name>.max_context`                                   | per-backend                          | Drop backend if `prompt_tokens + expected_output > max_context`.                                      |

`router config show` (or `/config` in chat) dumps the full validated tree
when you need everything.

---

## Common workflows

### "I want Claude to handle more agentic requests"

```bash
router config get policy.capability_bonuses.agentic_tool       # was 0.4
router config set policy.capability_bonuses.agentic_tool 0.6
router "refactor src/auth/login.py to use async/await" --why   # verify chosen=claude
```

The `--why` line on stderr will show `chosen=claude` with a higher
`scores=…claude=…` than before.

### "I'm spending too much"

```bash
router config set policy.cost_ceiling_usd_per_request 0.05   # hard cap
router config set policy.weights.cost 0.7                    # also bias the soft scorer
router "explain this stack trace" --why                      # confirm gemma4/gemini, not claude
```

The cost ceiling is a hard filter; the weight is a soft preference. Use
both for sustained spend control.

### "Add a new exemplar so X always routes to gemma4"

```bash
router config anchor add gemma4 "what does CRUD stand for"
router config rebuild-anchors                # re-embed centroids
router "what does CRUD stand for" --why      # verify chosen=gemma4
```

Or do both steps at once:

```bash
router config anchor add gemma4 "what does CRUD stand for" --rebuild
```

### "Swap to a bigger (or smaller) Gemma 4 variant"

The `gemma4` backend defaults to `google/gemma-4-E4B-it` (4.5B effective
params, 128K context). The Gemma 4 lineup published on
[huggingface.co/blog/gemma4](https://huggingface.co/blog/gemma4) gives four
sizes you can drop in by editing two keys:

| Variant            | Effective / total params | Context | Best for                                        |
| ------------------ | ------------------------ | ------- | ----------------------------------------------- |
| `gemma-4-E2B-it`   | 2.3B / 5.1B              | 128K    | Edge / phones; lowest VRAM.                     |
| `gemma-4-E4B-it`   | 4.5B / 8B                | 128K    | Default. Strong quality at single-GPU cost.     |
| `gemma-4-26B-A4B-it` | 4B active / 26B (MoE)  | 128K    | Bigger quality budget, similar latency to E4B.  |
| `gemma-4-31B-it`   | 31B dense                | 256K    | Highest quality + double the context window.    |

Swap in one command, then restart vLLM with the same model id:

```bash
router config set backends.gemma4.model google/gemma-4-26B-A4B-it
router config set backends.gemma4.max_context 131072
# Then restart your vLLM server with the same model id.
vllm serve google/gemma-4-26B-A4B-it --port 8000
```

Bumping to the 31B dense model? Also raise `max_context` to `262144` to
make use of the larger window — otherwise the router caps prompts at
whatever you set here.

---

## Troubleshooting

### vLLM not running

The `gemma4` backend points at `http://localhost:8000/v1` by default. If
nothing is listening you'll get a connection-refused error on the first
prompt that routes there, and the orchestrator will fall back to
`policy.fallback_backend` (also `gemma4` by default — so the request fails).

Start vLLM with the model id from `router.yaml` (default
`google/gemma-4-E4B-it`):

```bash
vllm serve google/gemma-4-E4B-it --port 8000
```

To temporarily route around it without changing config, use
`router --force gemini "<prompt>"` or set `policy.fallback_backend` to
`gemini`.

### Locked-session confusion

If a session got locked to a backend you didn't expect:

- In chat: `/unlock`, then resend the prompt — the next turn re-classifies.
- In one-shot: pass `--no-affinity` to skip the lock for one probe, or
  `--force <backend>` to override entirely.

`/why` (chat) or `--why` (one-shot) will show `affinity=locked` vs `free`.

### Where backups go

Every validated edit (CLI, REPL `/set`, or web UI) snapshots the previous
file in the same directory:

```
config/router.yaml
config/router.yaml.bak-20260420T143012001Z
config/router.yaml.bak-20260420T140155009Z
```

List and restore:

```bash
router config backups list
router config backups restore router.yaml.bak-20260420T140155009Z
```

`backups restore` validates the backup before clobbering, and snapshots the
current file *before* overwriting (so you can always undo the undo).

### Edited config rejected

If `router config edit` fails validation on save you'll see:

```
error: edited file failed validation: <pydantic error>
your edits remain at /path/to/config/router.yaml.<random>
```

The scratch file is preserved in `cfg_path.parent` so you don't lose your
work; fix it and re-run `router config edit` (or copy it back manually).
The live `router.yaml` is untouched.

`router config set` errors with exit code `2` and leaves `router.yaml`
unchanged on validation failure.

### Where logs and history live

| File                                               | What's in it                                                                   |
| -------------------------------------------------- | ------------------------------------------------------------------------------ |
| `~/.router/sessions/router_history.jsonl`          | Routing telemetry: one JSON object per turn — prompt, scores, chosen backend, latency, cost, shadow choice, session id. Mined by `gemini-dreams`. |
| `~/.router/repl_history`                           | `prompt_toolkit` line-edit history (Up/Down recall across REPL sessions).      |
| `~/.router/cache/`                                 | Embedding-anchor centroid cache (`classifier.cache_dir`).                      |

Override `logging.path` to redirect telemetry; override `classifier.cache_dir`
to relocate the centroid cache.
