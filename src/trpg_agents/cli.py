"""Test a model connection with `trpg-agents test-connection --config llm.toml`."""

import argparse
import sys
from pathlib import Path

from .config import load_model_config
from .host import ChatRequestError, complete


def main(argv: list[str] | None = None) -> int:
    """Probe the configured model and return zero only after receiving reply text."""
    parser = argparse.ArgumentParser(prog="trpg-agents")
    commands = parser.add_subparsers(dest="command", required=True)
    test = commands.add_parser("test-connection", help="Send a short test prompt to the model")
    test.add_argument(
        "--config", type=Path, default=Path("llm.toml"), help="Model config TOML file"
    )
    args = parser.parse_args(argv)

    try:
        config = load_model_config(args.config)
        complete(config, [{"role": "user", "content": "Reply with one word: OK."}])
    except (OSError, ValueError, ChatRequestError) as error:
        print(f"Connection failed: {error}", file=sys.stderr)
        return 1

    print(f"Connection OK: {config.model}")
    return 0
