"""How often does a critic verdict contradict its own per-field evidence?"""
from __future__ import annotations

import argparse
import json
import pathlib
from collections import Counter

FIELDS = ("workflow_matches_query", "sink_answers_target", "all_query_facts_used",
          "all_program_nodes_required", "no_extra_conditions",
          "units_semantically_valid", "query_unambiguous", "query_natural",
          "graph_not_disclosed")


def evidence(critic: dict | None) -> list[str]:
    return [f for f in FIELDS if (critic or {}).get(f) is False]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/pilot4_3_nestful_final")
    ap.add_argument("--file", default="llm_rendered.jsonl")
    args = ap.parse_args()

    rows = [json.loads(l) for l in
            (pathlib.Path(args.out_dir) / args.file).read_text(
                encoding="utf-8").splitlines()]
    tally: Counter = Counter()
    for row in rows:
        for key, label in (("critic", "c1"), ("second_critic", "c2")):
            critic = row.get(key)
            if not critic:
                continue
            verdict = critic.get("verdict")
            bad = evidence(critic)
            tally[f"{label} {verdict} with {'evidence' if bad else 'NO evidence'}"] += 1
            unaligned = [n for n in critic.get("node_alignment") or []
                         if n.get("aligned") is False]
            if verdict != "PASS" and not bad and not unaligned:
                tally[f"{label} unsubstantiated"] += 1
            if verdict == "PASS" and bad:
                tally[f"{label} passed despite {len(bad)} false field(s)"] += 1
    for key, count in sorted(tally.items()):
        print(f"{count:4d}  {key}")
    print(f"\n{len(rows)} records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
