"""Deduplicate llm_rendered.jsonl keeping the last record per task_id|mode."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/pilot4_3_nestful_final")
    args = ap.parse_args()
    path = Path(args.out_dir) / "llm_rendered.jsonl"
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    keys: dict[str, dict] = {}
    n_in = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            row = json.loads(line)
            keys[f"{row.get('task_id')}|{row.get('requested_mode')}"] = row
    bak = path.with_suffix(".jsonl.bak_precompact")
    if not bak.exists():
        path.replace(bak)
        src = bak
    else:
        src = path
        # rewrite in place from memory
    ordered = sorted(keys.values(), key=lambda r: str(r.get("task_id")))
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in ordered:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    print(json.dumps({"lines_in": n_in, "unique": len(ordered),
                      "backup": str(bak if bak.exists() else "")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
