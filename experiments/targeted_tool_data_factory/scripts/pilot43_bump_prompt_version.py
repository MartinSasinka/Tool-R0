"""Bump the pilot43 prompt version everywhere it is asserted or configured."""
from __future__ import annotations

import pathlib
import re
import sys

FILES = (
    "src/targeted_tool_data/pilot43/orprompts.py",
    "tests/test_pilot43_openrouter.py",
    "configs/pilot4_3_openrouter.yaml",
)
NAMES = "writer|critic|critic2|rewrite"


def main(old: str, new: str) -> int:
    root = pathlib.Path(__file__).resolve().parents[1]
    pattern = re.compile(rf"(pilot43\.(?:{NAMES}))\.{re.escape(old)}\b")
    for name in FILES:
        path = root / name
        text = path.read_text(encoding="utf-8")
        bumped = pattern.sub(rf"\1.{new}", text)
        if bumped != text:
            path.write_text(bumped, encoding="utf-8")
        print(f"{name}: {len(pattern.findall(text))} occurrences")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
