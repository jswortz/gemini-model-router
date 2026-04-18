from __future__ import annotations

import asyncio
import sys

import ulid
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory

try:
    from halo import Halo
except ImportError:  # halo is optional at runtime
    Halo = None  # type: ignore

from router.config_loader import load_config
from router.orchestrator import Orchestrator, find_default_config


HELP = """
Slash commands:
  /help                   show this help
  /why                    explain the last routing decision
  /route <gemma4|gemini|claude>   force a single backend for the next prompt
  /unroute                clear a forced route
  /unlock                 release session affinity so the next turn re-classifies
  /regen                  re-run the last prompt (logs as exact_re_ask)
  /quit                   exit
"""


async def _ainput(session: PromptSession, prompt: str) -> str:
    return await asyncio.to_thread(session.prompt, prompt)


async def _amain() -> int:
    cfg = load_config(find_default_config())
    orch = Orchestrator(cfg)
    sid = str(ulid.new())
    session = PromptSession(history=InMemoryHistory())
    forced: str | None = None
    last_prompt: str | None = None
    last_result = None

    sys.stdout.write(
        f"router REPL — session {sid[:10]}…  type /help for commands.\n"
    )

    while True:
        try:
            line = await _ainput(session, ">>> ")
        except (EOFError, KeyboardInterrupt):
            sys.stdout.write("\nbye\n")
            return 0

        line = line.strip()
        if not line:
            continue

        if line.startswith("/"):
            parts = line.split()
            cmd = parts[0]
            if cmd == "/quit":
                return 0
            if cmd == "/help":
                sys.stdout.write(HELP)
                continue
            if cmd == "/why":
                if not last_result:
                    sys.stdout.write("(no previous turn)\n")
                else:
                    affinity = "locked" if last_result.affinity_locked else "free"
                    shadow = (
                        f" shadow={last_result.shadow_choice}"
                        if last_result.shadow_choice
                        else ""
                    )
                    sys.stdout.write(
                        f"chosen={last_result.chosen_backend} "
                        f"confidence={last_result.confidence:.3f} "
                        f"override={last_result.hard_override or '-'} "
                        f"fallback={last_result.fallback_used} "
                        f"affinity={affinity}{shadow}\n"
                        f"scores={last_result.decision_scores}\n"
                    )
                continue
            if cmd == "/route" and len(parts) == 2 and parts[1] in {"gemma4", "gemini", "claude"}:
                forced = parts[1]
                sys.stdout.write(f"(forcing next prompt to {forced})\n")
                continue
            if cmd == "/unroute":
                forced = None
                sys.stdout.write("(forced route cleared)\n")
                continue
            if cmd == "/unlock":
                if orch.unlock_session(sid):
                    sys.stdout.write("(session affinity released)\n")
                else:
                    sys.stdout.write("(no active lock on this session)\n")
                continue
            if cmd == "/regen":
                if not last_prompt:
                    sys.stdout.write("(no previous prompt)\n")
                    continue
                line = last_prompt  # fall through to dispatch
            else:
                sys.stdout.write(f"unknown command: {cmd}\n")
                continue

        prompt = line
        backend_name = forced or "?"
        spinner = None
        if Halo and backend_name not in {"gemma4"}:
            spinner = Halo(text=f"thinking…", spinner="dots")
            spinner.start()

        try:
            result = await orch.route(prompt, session_id=sid, force=forced, stream=True)
        finally:
            if spinner:
                spinner.stop()

        if "stream" not in orch.backends[result.chosen_backend].capabilities:
            sys.stdout.write(result.response.text + "\n")
        else:
            sys.stdout.write("\n")

        last_prompt = prompt
        last_result = result
        # Forced route is one-shot.
        forced = None


def main() -> None:
    sys.exit(asyncio.run(_amain()))


if __name__ == "__main__":
    main()
