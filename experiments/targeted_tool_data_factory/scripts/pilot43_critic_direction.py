"""Which critic dissents, and on what, across every archived smoke run.

Blocking is symmetric, so a single critic that rejects good queries costs as much
yield as a bad writer. This counts the direction of every disagreement.
"""
from __future__ import annotations

import argparse
import json
import pathlib
from collections import Counter

FIELDS = ("workflow_matches_query", "sink_answers_target", "all_query_facts_used",
          "all_program_nodes_required", "no_extra_conditions",
          "units_semantically_valid", "query_unambiguous", "query_natural",
          "graph_not_disclosed")


def verdict(critic: dict | None) -> str:
    if not critic:
        return "none"
    bad = [f for f in FIELDS if critic.get(f) is False]
    if bad:
        return "reject"
    return "pass" if critic.get("verdict") == "PASS" else "uncertain"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/pilot4_3_nestful_final")
    ap.add_argument("--glob", default="stage_history/llm_rendered_prompt_v5*.jsonl")
    args = ap.parse_args()

    root = pathlib.Path(args.out_dir)
    paths = sorted(root.glob(args.glob)) + [root / "llm_rendered.jsonl"]
    direction: Counter = Counter()
    fields: Counter = Counter()
    n = 0
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            first, second = verdict(row.get("critic")), verdict(
                row.get("second_critic"))
            if second == "none":
                continue
            n += 1
            direction[f"c1={first:<9} c2={second}"] += 1
            if first == "reject" and second == "pass":
                for f in FIELDS:
                    if (row.get("critic") or {}).get(f) is False:
                        fields[f"c1 alone: {f}"] += 1
            if second == "reject" and first == "pass":
                for f in FIELDS:
                    if (row.get("second_critic") or {}).get(f) is False:
                        fields[f"c2 alone: {f}"] += 1

    print(f"{n} records where both critics answered\n")
    for key, count in direction.most_common():
        print(f"{count:4d}  {key}")
    print()
    for key, count in fields.most_common(12):
        print(f"{count:4d}  {key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
