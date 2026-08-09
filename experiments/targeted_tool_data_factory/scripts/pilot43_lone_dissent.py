"""Show queries that one critic passed and the other rejected, with the reasons.

Reading these is how we decide whether a lone dissenter is catching real defects
or costing us good samples.
"""
from __future__ import annotations

import argparse
import json
import pathlib

FIELDS = ("workflow_matches_query", "sink_answers_target", "all_query_facts_used",
          "all_program_nodes_required", "no_extra_conditions",
          "units_semantically_valid", "query_unambiguous", "query_natural",
          "graph_not_disclosed")


def state(critic: dict | None) -> tuple[str, list[str]]:
    if not critic:
        return "none", []
    bad = [f for f in FIELDS if critic.get(f) is False]
    if bad:
        return "reject", bad
    return ("pass" if critic.get("verdict") == "PASS" else "uncertain"), []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/pilot4_3_nestful_final")
    ap.add_argument("--side", choices=["c1", "c2"], default="c2",
                    help="which critic is the lone dissenter")
    ap.add_argument("--limit", type=int, default=8)
    args = ap.parse_args()

    path = pathlib.Path(args.out_dir) / "llm_rendered.jsonl"
    shown = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        first, first_bad = state(row.get("critic"))
        second, second_bad = state(row.get("second_critic"))
        lone_c2 = second == "reject" and first == "pass"
        lone_c1 = first == "reject" and second == "pass"
        if not (lone_c2 if args.side == "c2" else lone_c1):
            continue
        shown += 1
        if shown > args.limit:
            break
        bad = second_bad if args.side == "c2" else first_bad
        critic = row.get("second_critic" if args.side == "c2" else "critic") or {}
        print("=" * 78)
        print(f"{row['task_id']}  {row['workflow_id']}  {row['requested_mode']}"
              f"  calls={row['call_count']}  answer={row['answer_type']}")
        print(f"QUERY: {row['query']}")
        print(f"{args.side} rejected on: {bad}")
        for node in critic.get("node_alignment") or []:
            if node.get("aligned") is False and node.get("query_evidence"):
                print(f"   {node['node_id']}: {node['query_evidence'][:150]}")
    print(f"\n{shown} lone-{args.side} rejections")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
