#!/usr/bin/env python3
"""Optional held-out Stage-3 eval — requires --allow-model-inference."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_V3 = Path(__file__).resolve().parents[2]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--heldout", type=Path, required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--allow-model-inference",
        action="store_true",
        help="Required flag to run model inference (not used in offline audit phase)",
    )
    args = p.parse_args()
    if not args.allow_model_inference:
        print(
            "Refusing to run inference without --allow-model-inference. "
            "Offline audit phase only prepares heldout JSONL.",
            file=sys.stderr,
        )
        return 2
    print("Not implemented in this scaffold; wire to final_eval_v5 locally.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
