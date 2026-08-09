"""Remove OpenRouter 402 (insufficient credits) stubs from llm_rendered.jsonl.

These are not real writer attempts — they must not block resume once credits
are restored, and they must not count as permanent rejects.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/pilot4_3_nestful_final")
    args = ap.parse_args()
    path = Path(args.out_dir) / "llm_rendered.jsonl"
    kept = []
    stripped = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            err = str(row.get("error") or "")
            if row.get("blocked_reason") == "writer_failed" and (
                    "402" in err or "Insufficient credits" in err):
                stripped += 1
                continue
            kept.append(row)
    bak = path.with_suffix(".jsonl.bak_pre402strip")
    if not bak.exists():
        path.replace(bak)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in kept:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    print(json.dumps({"kept": len(kept), "stripped_402": stripped,
                      "backup": str(bak)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
