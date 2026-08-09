"""Dump the blocked smoke records in full, so a rejection can be read by eye."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FIELDS = ("workflow_matches_query", "sink_answers_target", "all_query_facts_used",
          "all_program_nodes_required", "no_extra_conditions",
          "units_semantically_valid", "query_unambiguous", "query_natural",
          "graph_not_disclosed")


def falses(critic: dict | None) -> list[str]:
    if not critic:
        return ["<no verdict>"]
    return [f for f in FIELDS if critic.get(f) is False]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir",
                    default=str(ROOT / "outputs" / "pilot4_3_nestful_final"))
    ap.add_argument("--reason", default="", help="only this blocked_reason prefix")
    ap.add_argument("--limit", type=int, default=8)
    args = ap.parse_args()

    path = pathlib.Path(args.out_dir) / "llm_rendered.jsonl"
    shown = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if not row.get("blocked"):
            continue
        if args.reason and not str(row.get("blocked_reason", "")).startswith(
                args.reason):
            continue
        shown += 1
        if shown > args.limit:
            break
        print("=" * 78)
        print(f"{row['task_id']}  {row['workflow_id']}  mode={row['requested_mode']}"
              f"  calls={row['call_count']}  answer={row['answer_type']}")
        print(f"blocked: {row['blocked_reason']}")
        print(f"QUERY: {row['query']}")
        val = row.get("validation") or {}
        bad = [k for k, v in (val.get("layers") or {}).items()
               if isinstance(v, dict) and v.get("passed") is False]
        if bad:
            print(f"deterministic layers failed: {bad}")
            for k in bad:
                print(f"   {k}: {val['layers'][k]}")
        print(f"critic1 false: {falses(row.get('critic'))}")
        print(f"critic2 false: {falses(row.get('second_critic'))} "
              f"(routed: {row.get('second_critic_reason')})")
        for critic, label in ((row.get("critic"), "c1"),
                              (row.get("second_critic"), "c2")):
            for node in (critic or {}).get("node_alignment") or []:
                if node.get("aligned") is False:
                    print(f"   {label} unaligned {node.get('node_id')}: "
                          f"{node.get('query_evidence')}")
        for hop in row.get("rewrite_history") or []:
            print(f"rewrite[{hop.get('reason', 'validation')}]: "
                  f"{hop.get('changes_made')}")
    print(f"\n{shown} blocked records matched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
