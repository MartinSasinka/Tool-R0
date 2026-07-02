#!/usr/bin/env python3
"""Canonicalize gold and predicted trajectories."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from vizualisation.scripts.lib.canonicalize import canonicalize_row  # noqa: E402
from vizualisation.scripts.lib.io_utils import load_jsonl_list, log, write_jsonl  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    args = p.parse_args()
    run_dir = Path(args.run_dir)
    raw_path = run_dir / "trajectories_raw.jsonl"
    if not raw_path.is_file():
        log("canon", f"ERROR: missing {raw_path}")
        return 2

    rows = [canonicalize_row(r) for r in load_jsonl_list(raw_path)]
    out = run_dir / "trajectories_canonical.jsonl"
    write_jsonl(out, rows)
    log("canon", f"wrote {len(rows)} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
