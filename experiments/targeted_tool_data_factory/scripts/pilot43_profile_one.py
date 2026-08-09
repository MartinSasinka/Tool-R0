"""cProfile one verification, to locate the hot spot on a pathological task.

    python scripts/pilot43_profile_one.py --task-id p43_...
"""
from __future__ import annotations

import argparse
import cProfile
import pstats
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from targeted_tool_data.pilot43 import pipeline as P  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/_p43_smoke")
    ap.add_argument("--task-id", required=True)
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    rows = {r["task_id"]: r for r in P.read_jsonl(Path(args.dir) / P.SHORTLIST)}
    row = rows[args.task_id]
    prof = cProfile.Profile()
    prof.enable()
    out = P._verify_one(row, P.VerifyConfig())
    prof.disable()
    print("selectable", out["selectable"], out["reject_reason"][:120],
          "seconds", out["seconds"])
    pstats.Stats(prof).sort_stats("cumulative").print_stats(args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
