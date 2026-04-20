"""Interactive chat REPL for the router.

Entry: `chat()` is called by both the `router` console script (when invoked
with no prompt at a TTY) and the legacy `router-repl` script.

Design notes — why this is not a full-screen `prompt_toolkit.Application`:
the upstream CLIs we federate (`claude`, `gemini`) are scrolling-chat apps,
not full-screen TUIs. Staying on `PromptSession` lets streaming output reuse
the existing `sys.stdout.write` path in the orchestrator, and keeps the
bottom toolbar / completion / multi-line ergonomics that define the
"feels like a real CLI" experience.
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections import OrderedDict
from pathlib import Path

import ulid
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import FuzzyCompleter, WordCompleter
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.shortcuts import print_formatted_text

try:
    from halo import Halo
except ImportError:  # halo is optional at runtime
    Halo = None  # type: ignore

from router import configcli
from router.config_loader import load_config
from router.orchestrator import Orchestrator, find_default_config

# ---------------- constants ----------------


SLASH_COMMANDS = [
    "/help",
    "/why",
    "/route",
    "/unroute",
    "/unlock",
    "/regen",
    "/sessions",
    "/resume",
    "/fork",
    "/set",
    "/get",
    "/config",
    "/quit",
]

BACKEND_COLORS = {
    "gemma4": "ansicyan",
    "gemini": "ansimagenta",
    "claude": "ansiyellow",
}

HELP_TEXT = """\
Slash commands:
  /help                          show this help
  /why                           explain the last routing decision
  /route <gemma4|gemini|claude>  force a single backend for the next prompt
  /unroute                       clear a forced route
  /unlock                        release session affinity (next turn re-classifies)
  /regen                         re-run the last prompt (logs as exact_re_ask)
  /sessions [N]                  list the last N session ids from the JSONL log
  /resume <session_id>           continue a prior session id on the next turn
  /fork                          start a new session id (drops affinity)
  /get <dotted.path>             read a config value (e.g. policy.weights.cost)
  /set <dotted.path> <yaml>      write a config value (validated, backed up)
  /config                        pretty-print the full router.yaml
  /quit                          exit

Editing:
  Enter           submit (smart: holds for unclosed code fences / trailing \\)
  Alt-Enter       force-submit
  Up / Down       history (persists across sessions in ~/.router/repl_history)
  Tab             complete slash commands and config paths
"""


# ---------------- helpers ----------------


def _expand(p: str) -> Path:
    return Path(p).expanduser()


def _print_banner(cfg_path: Path, orch: Orchestrator, sid: str) -> None:
    print_formatted_text(FormattedText([("ansibrightblack", "─" * 70)]))
    print_formatted_text(
        FormattedText(
            [
                ("bold", "  router"),
                ("", "  ·  "),
                ("ansibrightblack", f"config: {cfg_path}"),
            ]
        )
    )
    print_formatted_text(FormattedText([("ansibrightblack", f"  session: {sid[:10]}…")]))
    print_formatted_text(FormattedText([("", "  backends:")]))
    for b in orch.cfg.backends:
        color = BACKEND_COLORS.get(b.name, "ansiwhite")
        cost = (
            "$0"
            if (b.cost_in_per_1m == 0 and b.cost_out_per_1m == 0)
            else f"${b.cost_in_per_1m:g}/${b.cost_out_per_1m:g} per 1M"
        )
        print_formatted_text(
            FormattedText(
                [
                    ("", "    "),
                    (color, b.name.ljust(8)),
                    ("ansibrightblack", f"  {cost}  ·  ctx={b.max_context}"),
                ]
            )
        )
    print_formatted_text(FormattedText([("ansibrightblack", "  type /help or ? for commands")]))
    print_formatted_text(FormattedText([("ansibrightblack", "─" * 70)]))


# ---------------- multi-line submission heuristic ----------------


def _has_unclosed_fence(text: str) -> bool:
    """True if the text contains an odd number of ``` markers."""
    return text.count("```") % 2 == 1


def _wants_continuation(text: str) -> bool:
    if not text:
        return False
    stripped = text.rstrip()
    if stripped.endswith("\\"):
        return True
    return _has_unclosed_fence(text)


def _build_keybindings() -> KeyBindings:
    kb = KeyBindings()

    @kb.add(Keys.Enter)
    def _enter(event):
        buf = event.current_buffer
        if _wants_continuation(buf.text):
            buf.insert_text("\n")
        else:
            buf.validate_and_handle()

    @kb.add(Keys.Escape, Keys.Enter)
    def _alt_enter(event):
        event.current_buffer.validate_and_handle()

    return kb


# ---------------- session enumeration (for /sessions, /resume) ----------------


def _read_recent_sessions(log_path: Path, limit: int = 10) -> list[tuple[str, int, str]]:
    """Return [(session_id, turn_count, last_prompt_preview), ...] newest first."""
    if not log_path.exists():
        return []
    seen: OrderedDict[str, dict] = OrderedDict()
    try:
        for line in log_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = rec.get("session_id")
            if not sid:
                continue
            turn = rec.get("router", {}).get("turn", 0)
            prev = seen.get(sid, {"turn": 0, "preview": ""})
            if turn >= prev["turn"]:
                preview = (rec.get("prompt") or "")[:60].replace("\n", " ")
                seen[sid] = {"turn": turn, "preview": preview}
    except OSError:
        return []
    items = list(seen.items())[-limit:]
    items.reverse()
    return [(sid, v["turn"], v["preview"]) for sid, v in items]


# ---------------- chat loop ----------------


async def _ainput(session: PromptSession, prompt) -> str:
    return await asyncio.to_thread(session.prompt, prompt)


async def _amain(config_path: str | None) -> int:
    cfg_path = Path(config_path).resolve() if config_path else find_default_config().resolve()
    cfg = load_config(cfg_path)
    orch = Orchestrator(cfg)
    sid = str(ulid.new())

    history_dir = _expand("~/.router")
    history_dir.mkdir(parents=True, exist_ok=True)
    history = FileHistory(str(history_dir / "repl_history"))

    completer = FuzzyCompleter(WordCompleter(SLASH_COMMANDS, sentence=True))
    kb = _build_keybindings()

    state = {"forced": None, "last_prompt": None, "last_result": None, "turn": 0}

    def bottom_toolbar():
        last = state["last_result"]
        if last is None:
            tail = "no turns yet"
        else:
            tail = (
                f"last: {last.chosen_backend} "
                f"{last.response.latency_ms:.0f}ms "
                f"${last.response.usage.cost_usd:.4f}"
            )
        forced = f" · forced→{state['forced']}" if state["forced"] else ""
        return FormattedText(
            [
                (
                    "bg:#222 #aaa",
                    f" session {sid[:10]}… · turn {state['turn']} · {tail}{forced} · /help ",
                )
            ]
        )

    session = PromptSession(
        history=history,
        completer=completer,
        complete_while_typing=True,
        multiline=True,
        key_bindings=kb,
        bottom_toolbar=bottom_toolbar,
    )

    _print_banner(cfg_path, orch, sid)

    # Mutable reference so /resume can swap session ids in.
    current_sid = [sid]

    while True:
        try:
            line = await _ainput(session, FormattedText([("ansicyan bold", ">>> ")]))
        except (EOFError, KeyboardInterrupt):
            print_formatted_text(FormattedText([("ansibrightblack", "bye")]))
            return 0

        line = line.strip()
        if not line:
            continue

        if line.startswith("/") or line == "?":
            cmd_line = "/help" if line == "?" else line
            handled, should_quit = _handle_slash(cmd_line, state, orch, current_sid, cfg_path)
            if should_quit:
                return 0
            if state.pop("regen_pending", False) and state["last_prompt"]:
                await _dispatch_prompt(state["last_prompt"], state, orch, current_sid)
            continue

        await _dispatch_prompt(line, state, orch, current_sid)


def _handle_slash(
    line: str,
    state: dict,
    orch: Orchestrator,
    current_sid: list[str],
    cfg_path: Path,
) -> tuple[bool, bool]:
    """Returns (handled, should_quit)."""
    parts = line.split(maxsplit=2)
    cmd = parts[0]

    if cmd == "/quit":
        print_formatted_text(FormattedText([("ansibrightblack", "bye")]))
        return True, True

    if cmd == "/help":
        sys.stdout.write(HELP_TEXT)
        return True, False

    if cmd == "/why":
        last = state["last_result"]
        if not last:
            sys.stdout.write("(no previous turn)\n")
            return True, False
        affinity = "locked" if last.affinity_locked else "free"
        shadow = f" shadow={last.shadow_choice}" if last.shadow_choice else ""
        sys.stdout.write(
            f"chosen={last.chosen_backend} confidence={last.confidence:.3f} "
            f"override={last.hard_override or '-'} fallback={last.fallback_used} "
            f"affinity={affinity}{shadow}\n"
            f"scores={last.decision_scores}\n"
        )
        return True, False

    if cmd == "/route":
        if len(parts) < 2 or parts[1] not in {"gemma4", "gemini", "claude"}:
            sys.stdout.write("usage: /route <gemma4|gemini|claude>\n")
            return True, False
        state["forced"] = parts[1]
        sys.stdout.write(f"(forcing next prompt to {parts[1]})\n")
        return True, False

    if cmd == "/unroute":
        state["forced"] = None
        sys.stdout.write("(forced route cleared)\n")
        return True, False

    if cmd == "/unlock":
        if orch.unlock_session(current_sid[0]):
            sys.stdout.write("(session affinity released)\n")
        else:
            sys.stdout.write("(no active lock on this session)\n")
        return True, False

    if cmd == "/regen":
        if not state["last_prompt"]:
            sys.stdout.write("(no previous prompt)\n")
            return True, False
        # Signal the main loop to re-dispatch the last prompt.
        state["regen_pending"] = True
        return True, False

    if cmd == "/sessions":
        limit = int(parts[1]) if len(parts) >= 2 and parts[1].isdigit() else 10
        log_path = _expand(orch.cfg.logging.path)
        rows = _read_recent_sessions(log_path, limit=limit)
        if not rows:
            sys.stdout.write("(no prior sessions in log)\n")
            return True, False
        for sid_, turns, preview in rows:
            sys.stdout.write(f"  {sid_}  turns={turns}  {preview}\n")
        return True, False

    if cmd == "/resume":
        if len(parts) < 2:
            sys.stdout.write("usage: /resume <session_id>\n")
            return True, False
        current_sid[0] = parts[1]
        state["turn"] = 0  # local turn counter resets; orchestrator owns the real one
        sys.stdout.write(f"(now using session {parts[1]})\n")
        return True, False

    if cmd == "/fork":
        new_sid = str(ulid.new())
        current_sid[0] = new_sid
        state["turn"] = 0
        sys.stdout.write(f"(forked to new session {new_sid[:10]}…)\n")
        return True, False

    if cmd == "/get":
        if len(parts) < 2:
            sys.stdout.write("usage: /get <dotted.path>\n")
            return True, False
        try:
            value = configcli.repl_get(cfg_path, parts[1])
        except KeyError as e:
            sys.stdout.write(f"error: {e}\n")
            return True, False
        sys.stdout.write(f"{parts[1]} = {value!r}\n")
        return True, False

    if cmd == "/set":
        if len(parts) < 3:
            sys.stdout.write("usage: /set <dotted.path> <yaml-value>\n")
            return True, False
        dotted, raw = parts[1], parts[2]
        try:
            bak, parsed = configcli.repl_set(cfg_path, dotted, raw)
        except Exception as e:
            sys.stdout.write(f"error: {e}\n")
            return True, False
        if bak:
            sys.stdout.write(f"(backed up -> {bak.name})\n")
        sys.stdout.write(f"set {dotted} = {parsed!r}\n")
        # Reload orchestrator config so subsequent turns see the change.
        new_cfg = load_config(cfg_path)
        orch.cfg = new_cfg
        return True, False

    if cmd == "/config":
        sys.stdout.write(configcli.repl_show(cfg_path))
        return True, False

    sys.stdout.write(f"unknown command: {cmd}  (try /help)\n")
    return True, False


async def _dispatch_prompt(
    prompt: str,
    state: dict,
    orch: Orchestrator,
    current_sid: list[str],
) -> None:
    """Send a prompt to the orchestrator and render the result.

    The router picks a backend inside `route()`, so we don't know which model
    will respond until streaming has already started. To avoid the awkward
    "header appears after the answer" problem, we render a labeled footer
    after the turn completes — backend + latency + cost — which doubles as a
    visual separator between turns.
    """
    forced = state["forced"]
    spinner = None
    if Halo and forced != "gemma4":
        spinner = Halo(text="thinking…", spinner="dots")
        spinner.start()

    try:
        with patch_stdout(raw=True):
            result = await orch.route(
                prompt,
                session_id=current_sid[0],
                force=forced,
                stream=True,
            )
    finally:
        if spinner:
            spinner.stop()

    streamed = "stream" in orch.backends[result.chosen_backend].capabilities
    if not streamed:
        sys.stdout.write(result.response.text)
        if not result.response.text.endswith("\n"):
            sys.stdout.write("\n")
    else:
        sys.stdout.write("\n")

    # Footer: colored backend tag + latency + cost. Doubles as turn separator.
    color = BACKEND_COLORS.get(result.chosen_backend, "ansiwhite")
    print_formatted_text(
        FormattedText(
            [
                ("ansibrightblack", "── "),
                (color, result.chosen_backend),
                (
                    "ansibrightblack",
                    f" · {result.response.latency_ms:.0f}ms"
                    f" · ${result.response.usage.cost_usd:.4f}",
                ),
            ]
        )
    )

    state["last_prompt"] = prompt
    state["last_result"] = result
    state["turn"] += 1
    state["forced"] = None


def chat(config_path: str | None = None) -> int:
    """Public entry: drop into TUI chat. Returns process exit code."""
    return asyncio.run(_amain(config_path))


def main() -> None:
    """Console-script entry for `router-repl` (legacy alias)."""
    # Honor a future `--config` if anyone passes it, but don't get fancy.
    config = None
    if len(sys.argv) >= 3 and sys.argv[1] == "--config":
        config = sys.argv[2]
    elif "--help" in sys.argv or "-h" in sys.argv:
        sys.stdout.write("router-repl: alias for `router` (no args). See `router --help`.\n")
        sys.exit(0)
    sys.exit(chat(config_path=config))


if __name__ == "__main__":
    main()
