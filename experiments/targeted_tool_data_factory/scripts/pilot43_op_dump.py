"""Dump the resolved semantic signature of ops, filtered by capability prefix.

    python scripts/pilot43_op_dump.py rates arithmetic.add
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from targeted_tool_data.pilot43 import ops as O  # noqa: E402


def main() -> None:
    prefixes = sys.argv[1:] or [""]
    for pid, op in sorted(O.build_ops().items()):
        if not any(op.capability.startswith(p) or pid == p for p in prefixes):
            continue
        sig = ", ".join(f"{p.name}:{p.sem}" for p in op.params)
        print(f"{pid:32s} {op.capability:34s} ({sig}) -> {op.out_sem}")


if __name__ == "__main__":
    main()
