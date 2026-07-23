#!/usr/bin/env python3
"""Thin wrapper: validate + summarize weak-model audit."""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from prepare_weak_model_audit import build_parser, cmd_summarize, cmd_validate  # noqa: E402


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "validate":
        sys.argv[1] = "validate"
    parser = build_parser()
    args = parser.parse_args(["summarize"] if len(sys.argv) == 1 else sys.argv[1:])
    if getattr(args, "cmd", None) == "validate":
        cmd_validate(args)
    else:
        cmd_summarize(args)


if __name__ == "__main__":
    main()
