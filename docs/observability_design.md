# Cost / Latency / Quality BI + Observability — Design

> Output of the Plan-agent run, captured for future implementation work.
> Source data: every JSONL line written by `src/router/logging/jsonl_writer.py`
> to `~/.router/sessions/router_history.jsonl`.

## 1. Metrics catalog

### 1a. Per-backend operational

| Metric | Derivation | Unit | Why |
|---|---|---|---|
| `requests_total` | `count(*) WHERE success` GROUP BY `router.chosen_backend` | count | denominator |
| `requests_failed` | `count(*) WHERE NOT success` GROUP BY `router.chosen_backend` | count | failure surface per vendor |
| `latency_p50_ms` | `quantile(latency_ms, 0.5)` GROUP BY `router.chosen_backend` | ms | typical UX |
| `latency_p95_ms` | `quantile(latency_ms, 0.95)` GROUP BY `router.chosen_backend` | ms | tail; CLI hangs surface here |
| `latency_per_1k_out_ms` | `latency_ms / (usage.output_tokens / 1000.0)` | ms/1k tok | normalizes for response length |
| `cost_total_usd` | `sum(usage.cost_usd)` GROUP BY `router.chosen_backend` | USD | the headline number |
| `cost_per_request_avg` | `cost_total_usd / requests_total` | USD | per-call sticker price |
| `cost_per_1k_out_usd` | `sum(usage.cost_usd) / sum(usage.output_tokens)*1000` | USD/1k tok | vendor efficiency |
| `success_rate` | `sum(success)/count(*)` | ratio | reliability |
| `error_rate_by_class` | `count(*) WHERE error IS NOT NULL` GROUP BY `error` prefix | count | which class of failure dominates |
| `retry_rate_proxy` | sessions where consecutive rows share `session_id` and the second has `router.fallback_used = true` | ratio | true retries aren't tracked today; proxy via session traversal |

### 1b. Routing-quality

| Metric | Derivation | Unit | Why |
|---|---|---|---|
| `shadow_agreement_rate` | of rows where `router.affinity_locked = true` AND `router.shadow_choice IS NOT NULL`: `mean(router.shadow_choice == router.chosen_backend)` | ratio | <0.6 means lock is hurting more than helping |
| `shadow_disagreement_by_pair` | `count(*) WHERE shadow_choice != chosen_backend` GROUP BY (`chosen_backend`, `shadow_choice`) | count matrix | feeds gemini-dreams `move-to` directives directly |
| `fallback_used_rate` | `mean(router.fallback_used)` overall and GROUP BY backend | ratio | high = `confidence_margin` is too wide |
| `affinity_lock_share` | `mean(router.affinity_locked)` | ratio | fraction of traffic served by the cheap path |
| `hard_override_breakdown` | `count(*)` GROUP BY `router.hard_override` | count | why classifier was bypassed |
| `confidence_distribution` | histogram of `router.confidence` GROUP BY backend | histogram | low-confidence wins are the noisy ones |
| `score_margin` | `top1(router.scores) - top2(router.scores)` per row | float | how close the call was; pair with `user_followup_hint` to find regret cases |

### 1c. Cost realism (the Claude cache gap)

| Metric | Derivation | Unit | Why |
|---|---|---|---|
| `claude_cost_counted_usd` | `sum(usage.cost_usd) WHERE chosen_backend='claude'` | USD | what router billed |
| `claude_cost_raw_usd` | `sum(json_extract(backend_meta.raw_pruned, '$.total_cost_usd')) WHERE chosen_backend='claude'` | USD | what Claude actually charged (incl. cache reads/writes) |
| `claude_cache_undercount_usd` | `claude_cost_raw_usd - claude_cost_counted_usd` | USD | the known gap from `claude_cli.py` (cache tokens kept in raw, not added to billable) |
| `claude_cache_undercount_pct` | `claude_cache_undercount_usd / claude_cost_counted_usd` | ratio | sanity check on `compute_cost()` |

### 1d. Session health

| Metric | Derivation | Unit | Why |
|---|---|---|---|
| `turns_per_session` | `max(router.turn) GROUP BY session_id`, then describe | histogram | longer sessions = where affinity matters most |
| `vendor_session_reuse_rate` | per (`session_id`, `chosen_backend`): does `router.vendor_session_id` repeat across consecutive rows? | ratio | tells us prompt-cache savings are landing |
| `lock_bounce_rate` | sessions where `router.chosen_backend` changes after `affinity_locked=true` on a prior turn | per-session count | bounce = lost cache, lost MCP state |
| `first_turn_classify_count` | `count(*) WHERE router.turn = 1 AND router.affinity_locked = false` | count | cost driver — every turn-1 call pays the MiniLM round-trip |
| `session_age_at_unlock` | `router.turn` at the row immediately after a lock release | int | are users unlocking too early? |

### 1e. Followup-hint signals (regret proxy)

| Metric | Derivation | Unit | Why |
|---|---|---|---|
| `followup_hint_rate` | `mean(user_followup_hint IS NOT NULL)` GROUP BY backend | ratio | best signal we have for "user re-asked" |
| `regret_by_feature_bucket` | `followup_hint_rate` GROUP BY (backend, code_ratio_bucket) | ratio | "claude regrets cluster on code-heavy prompts" |
| `hint_taxonomy` | `count(*)` GROUP BY `user_followup_hint` | count | currently only `exact_re_ask` |

### 1f. Feature distributions

All from `router.features.*`.

| Metric | Derivation | Unit | Why |
|---|---|---|---|
| `code_ratio_hist` | histogram of `router.features.code_ratio`, GROUP BY backend | counts | high-code-ratio prompts going to gemma4 = the worry |
| `n_tokens_est_hist` | histogram of `router.features.n_tokens_est`, log buckets, GROUP BY backend | counts | long-context prompts must land on gemini/claude |
| `agent_keywords_hist` | histogram of `router.features.agent_keywords` GROUP BY backend | counts | agentic-language → expect claude |
| `tool_required_share` | `mean(router.features.tool_required) GROUP BY backend` | ratio | tool-required → gemma4 is a hard miss |
| `sensitive_share` | `mean(router.features.sensitive) GROUP BY backend` | ratio | should be ~100% gemma4 |

### 1g. Aggregate (the headline)

| Metric | Derivation | Unit | Why |
|---|---|---|---|
| `cost_savings_vs_claude_only` | counterfactual: same tokens, all routed to claude | USD | "is the router saving money?" |
| `cost_savings_vs_gemini_only` | same against gemini rates | USD | second baseline |
| `quality_floor_breach_count` | `count(*) WHERE chosen_backend='gemma4' AND tool_required=true` | count | hard miscalls |

---

## 2. Dashboards

### A. "Is the router earning its keep?"
Audience: jswortz. Tiles: total requests, total cost USD, savings-vs-claude/gemini, success rate. Charts: cost stack by backend over time (overlay raw cost line for cache gap); p50/p95 latency by backend; routing mix over time (100%-stacked area); followup-hint rate; cost-vs-latency-vs-quality scatter.

### B. "What is gemini-dreams about to flag tonight?"
Audience: jswortz pre-review. Charts: shadow disagreement matrix (heatmap); top 20 disagreement prompts (table); low-confidence wins (table); affinity-shadow regret count; feature-bucket misroutes (chosen=gemma4 with tool_required=true); pending `router-eval` items.

### C. "Is anything broken right now?"
Audience: operator. Charts: backend health pulse (sparkline, red <0.9); p95 latency 1h vs 24h baseline; error class breakdown 1h; CLI hang suspects (latency_ms > 60000); sessions on stale lock; cost anomaly (1h vs 7d median); vendor-session-id reuse rate (drop = silent cache loss).

### D. "Token economics" (monthly)
Audience: jswortz budgeting. Charts: counted vs raw cost (claude); cache undercount $ and %; tokens in/out by backend; cost_per_1k_out_usd over time; top 50 most expensive single requests.

---

## 3. Telemetry approach — DuckDB over JSONL (primary)

OpenTelemetry is the loser for v1: (1) the router process is a short-lived CLI invocation — worst possible fit for an OTLP push exporter; (2) an OTel collector is heavyweight infra the user is allergic to on a workstation; (3) the JSONL already exists and is the contract gemini-dreams reads. DuckDB reads it in place via `read_json_auto('~/.router/sessions/router_history.jsonl')`. Zero ingestion pipeline.

OTel becomes right in Phase-3 when FastAPI lands (long-lived process).

**Layer**: `src/router/observability/duck.py` — read-only sidecar that:
1. `CREATE OR REPLACE VIEW router_log AS SELECT * FROM read_json_auto('~/.router/sessions/router_history.jsonl', format='newline_delimited')`.
2. Defines named views per metric family: `v_per_backend_daily`, `v_shadow_matrix`, `v_cache_gap`, `v_feature_buckets`, `v_session_health`.
3. Optional materialization once JSONL > 200 MB.

The router never writes to DuckDB on the hot path — only the dashboard process and `router-eval` open the file, read-only.

### Future OTel span shape
- Root span `router.route` per `Orchestrator.route()` call. Attributes mirror `router_block`.
- Child `router.classify` (skipped on affinity path — its absence is itself a signal).
- Child `router.health_check`.
- Child `backend.invoke` per call.

---

## 4. Observability essentials

### 4a. Local alerts via `router doctor`
- **CLI hang**: in-flight subprocess past `hang_threshold_s` (default 90). Wrap `proc.communicate()` in `asyncio.wait_for` and emit a structured `error="timeout"` row.
- **Backend health flapping**: `success_rate < 0.8` over last 30 min.
- **Cost anomaly**: 1h cost-per-request > 3x trailing 7d median.
- **Cache undercount blowing up**: `claude_cache_undercount_pct > 50%`.
- **Lock-bounce spike**: sessions/hour with `lock_bounce_rate > 0` exceeds N.
- **Followup-hint regression**: rolling rate rises > 50% week-over-week.

Delivery: stdout for `router doctor`; opt-in `notify-send`/`osascript` desktop notifications. No SMTP, no PagerDuty.

### 4b. Health endpoints (Phase-3 FastAPI)
- `GET /healthz` → 200 if process up.
- `GET /readyz` → 200 only if all backend `health()` calls pass AND log path is writable.
- `GET /metrics` → JSON snapshot of section-1 metrics over last 1h (computed via DuckDB views).

### 4c. Logging hygiene — wire up `redact_prompts`
Currently honored only for top-level `prompt`. When the flag is on, also:
- `prompt_response` → `sha256:<16>` digest.
- `backend_meta.raw_pruned` → replace any string field >256 chars with sha256 digest; whitelist numeric fields so cost/token metrics still work.
- `error` stderr — truncate and redact.

Leave intact: `router.features.*`, `session_id`, `router.vendor_session_id`, `user_followup_hint`. Downstream consumers must be told that under redaction `prompt_response` is opaque.

---

## 5. Frontend stack

- **Streamlit** — one Python file, reads DuckDB directly, hot-reload. Default styling dated.
- **Grafana + DuckDB datasource** — proper BI feel + alerting, but Grafana service overhead and a community plugin.
- **React/Next page over FastAPI `/metrics`** — matches Phase-2 direction; highest ceiling, highest cost.

**Pick: Streamlit for v1**, plan the React/Next page for the same time the FastAPI server lands. Streamlit becomes throwaway then; that's fine — its job was to be fast in v1.

---

## 6. Phase plan

### Phase 1 — works today against existing JSONL (no router changes)
- `src/router/observability/duck.py` with the views from §3.
- `dashboards/streamlit_app.py` rendering Dashboards A and B.
- `router doctor` subcommand running §4a checks.
- Wire `router-dash` console script in `pyproject.toml`.

### Phase 2 — close data gaps surfaced in Phase 1 (small router-side changes)
- Wire `redact_prompts` gating per §4c (no schema change).
- Add CLI subprocess timeout watchdog (no schema change; emits `error="timeout"`).
- Expand `user_followup_hint` taxonomy: `near_re_ask` (cosine > 0.9), `manual_route_change`, `unlock_then_re_ask`. **Schema change: new enum values.**
- Add Dashboard C and D to Streamlit app.
- `router-eval` integration on Dashboard B.

### Phase 3 — FastAPI / OTel / real frontend
- Stand up FastAPI `/route`, `/healthz`, `/readyz`, `/metrics`.
- Add OTel exporter (process now long-lived). JSONL writer remains for gemini-dreams compatibility — both sinks coexist.
- Replace Streamlit with React/Next pages over `/metrics`, sharing chrome with the config-editing frontend.

---

## 7. Explicitly NOT recommending

1. **Datadog / New Relic** — workstation tool, single user, code in prompts the user doesn't want exfiltrated. Pricing model wrong (per-host); network shape wrong (1-second-lifetime processes).
2. **Prometheus pull-model `/metrics` in v1** — router process exits between calls; nothing for Prometheus to scrape. Push-gateway adds infra for one user.
3. **Kafka / Pub-Sub event bus** — JSONL is already the event log; only consumer is gemini-dreams which reads files.
4. **Materializing JSONL into Postgres/BigQuery in Phase-1** — the README's phase-2 table reserves BigQuery for later; DuckDB-over-JSONL gives the same SQL surface with zero infra.
5. **Sentry-style exception aggregation** — errors are categorical and small-cardinality; the §4a doctor view does the same job in 50 lines.
