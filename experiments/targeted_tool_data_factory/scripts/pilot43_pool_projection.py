"""Project the selectable pool from the render in flight.

Answers the question the selection stage will ask later: with the LLM pass rate
observed so far, how many hard-valid tasks will each tier-relevant stratum have,
and which quota is at risk?
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
from typing import Dict


def read_jsonl(path: pathlib.Path):
    if not path.exists():
        return
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/pilot4_3_nestful_final")
    args = ap.parse_args()
    out = pathlib.Path(args.out_dir)

    verified = {r["task_id"]: r for r in read_jsonl(out / "verified_candidates.jsonl")}
    selectable = {tid for tid, r in verified.items() if r.get("selectable")}
    rendered = list(read_jsonl(out / "llm_rendered.jsonl"))
    passed = {r["task_id"] for r in rendered if not r.get("blocked")}
    attempted = {r["task_id"] for r in rendered}
    rate = len(passed) / max(1, len(attempted))

    shortlist = {r["task_id"]: r for r in read_jsonl(out / "query_render_shortlist.jsonl")}
    print(f"verified {len(verified)}, selectable {len(selectable)}, "
          f"shortlisted {len(shortlist)}")
    print(f"rendered so far {len(attempted)}, usable {len(passed)} "
          f"(pass rate {rate:.3f})")

    # strata that carry a hard quota
    buckets: Dict[str, collections.Counter] = {
        "call_bucket": collections.Counter(),
        "answer_type": collections.Counter(),
        "coding": collections.Counter(),
    }
    for tid in selectable:
        row = shortlist.get(tid)
        if row is None:
            continue
        buckets["call_bucket"][row.get("call_bucket", "?")] += 1
        buckets["answer_type"][row.get("answer_type", "?")] += 1
        buckets["coding"][bool(row.get("coding_like"))] += 1

    for name, counter in buckets.items():
        total = sum(counter.values())
        print(f"\n{name} over the selectable shortlist ({total} tasks)")
        for key, count in sorted(counter.items(), key=lambda kv: str(kv[0])):
            projected = count * rate
            print(f"  {str(key):<12} {count:6d}  projected hard-valid {projected:8.0f}")

    print("\nquota reference: PROFILE_CORE 3000 (6+ share 20-24%), "
          "LONG_HORIZON 1200 (6+ >= 65%), CAPABILITY 600 (coding 100%), "
          "CHALLENGE 200, HELDOUT ~1000, RESERVE 1000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
