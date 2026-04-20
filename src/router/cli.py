from __future__ import annotations

import argparse
import asyncio
import sys

from router import configcli
from router.config_loader import load_config
from router.orchestrator import Orchestrator, find_default_config


def _build_oneshot_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="router",
        description=(
            "Route a single prompt across local Gemma 4, Gemini CLI, and Claude Code. "
            "Run with no prompt and a TTY to enter chat mode. "
            "Use `router config ...` to inspect/edit router.yaml."
        ),
    )
    p.add_argument("prompt", nargs="?", help="Prompt text. If omitted, read from stdin or chat.")
    p.add_argument("--config", default=None, help="Path to router.yaml.")
    p.add_argument(
        "--force",
        choices=["gemma4", "gemini", "claude"],
        default=None,
        help="Bypass the router and dispatch directly to one backend.",
    )
    p.add_argument("--no-stream", action="store_true", help="Disable terminal streaming.")
    p.add_argument(
        "--why",
        action="store_true",
        help="After answering, print why the chosen backend was selected.",
    )
    p.add_argument("--session", default=None, help="Reuse a session_id for sticky routing.")
    p.add_argument(
        "--no-affinity",
        action="store_true",
        help="Skip session affinity: re-classify even if the session is locked to a backend.",
    )
    return p




async def _run_oneshot(args: argparse.Namespace) -> int:
    cfg_path = args.config or find_default_config()
    cfg = load_config(cfg_path)
    orch = Orchestrator(cfg)

    prompt = args.prompt
    if not prompt:
        prompt = sys.stdin.read().strip()
    if not prompt:
        sys.stderr.write("error: empty prompt\n")
        return 2

    result = await orch.route(
        prompt,
        session_id=args.session,
        force=args.force,
        stream=not args.no_stream,
        no_affinity=args.no_affinity,
    )

    streamed = not args.no_stream and "stream" in orch.backends[result.chosen_backend].capabilities
    if not streamed:
        sys.stdout.write(result.response.text)
        if not result.response.text.endswith("\n"):
            sys.stdout.write("\n")
    else:
        sys.stdout.write("\n")

    if args.why:
        affinity = "locked" if result.affinity_locked else "free"
        shadow = f" shadow={result.shadow_choice}" if result.shadow_choice else ""
        sys.stderr.write(
            f"\n[router] chosen={result.chosen_backend} "
            f"confidence={result.confidence:.3f} "
            f"override={result.hard_override or '-'} "
            f"fallback={result.fallback_used} "
            f"affinity={affinity}{shadow} "
            f"scores={result.decision_scores} "
            f"latency_ms={result.response.latency_ms:.0f} "
            f"cost_usd={result.response.usage.cost_usd:.5f}\n"
        )

    return 0 if result.response.success else 1


def _should_enter_chat(args: argparse.Namespace) -> bool:
    """No prompt + no stdin pipe + isatty → drop into TUI chat."""
    if args.cmd == "config":
        return False
    if args.prompt:
        return False
    return sys.stdin.isatty()


def main() -> None:
    # `config` is dispatched off argv[0] explicitly so the one-shot `prompt`
    # positional doesn't get eaten by the subparser (e.g. `router "hello world"`
    # would otherwise be parsed as the subcommand `hello world`).
    argv = sys.argv[1:]
    if argv and argv[0] == "config":
        cfg_args = configcli.build_config_parser().parse_args(argv[1:])
        cfg_args.cmd = "config"
        sys.exit(configcli.dispatch(cfg_args))

    args = _build_oneshot_parser().parse_args(argv)
    args.cmd = None  # for _should_enter_chat()

    if _should_enter_chat(args):
        from router.repl import chat

        sys.exit(chat(config_path=args.config))

    sys.exit(asyncio.run(_run_oneshot(args)))


if __name__ == "__main__":
    main()
