"""Group the failures of an OpenRouter stage by cause, with the offending text.

Aggregate pass rates say a stage failed; they never say whether the writer, the
contract or the checker is at fault. This prints each failure with the evidence
needed to tell those three apart.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/pilot4_3_nestful_final")
    ap.add_argument("--layer", default="", help="only show this failed layer")
    ap.add_argument("--critic-key", default="", help="only show this critic false")
    ap.add_argument("--show", type=int, default=8)
    args = ap.parse_args()

    path = Path(args.out_dir) / "llm_rendered.jsonl"
    records = [json.loads(line) for line in path.open(encoding="utf-8")]

    det, crit, blocked = Counter(), Counter(), Counter()
    shown_layer = shown_critic = 0
    for rec in records:
        validation = rec.get("validation") or {}
        for layer in validation.get("failed_layers", []):
            det[layer] += 1
        critic = rec.get("critic") or {}
        for key, value in critic.items():
            if value is False:
                crit[key] += 1
        if rec.get("blocked"):
            blocked[rec.get("blocked_reason", "?")] += 1

        if args.layer and args.layer in validation.get("failed_layers", []):
            if shown_layer < args.show:
                shown_layer += 1
                print(f"\n=== {rec['task_id']} {rec['requested_mode']} "
                      f"{rec['workflow_id']} ({rec['answer_type']}, "
                      f"{rec['call_count']} calls)")
                print(f"query: {rec['query']}")
                print(f"layer: {json.dumps(validation['layers'][args.layer])}")
                print(f"contract: {json.dumps(rec.get('contract', {}))[:900]}")
        if args.critic_key and critic.get(args.critic_key) is False:
            if shown_critic < args.show:
                shown_critic += 1
                print(f"\n=== {rec['task_id']} {rec['requested_mode']} "
                      f"{rec['workflow_id']}  verdict={critic.get('verdict')}")
                print(f"query: {rec['query']}")
                for node in critic.get("node_alignment") or []:
                    if not node.get("aligned"):
                        print(f"  unaligned {node['node_id']}: "
                              f"{node.get('query_evidence')}")

    n = len(records) or 1
    print(f"\nrecords: {len(records)}")
    print(f"deterministic failures: {det.most_common()}")
    print(f"critic falses: {crit.most_common()}")
    print(f"blocked reasons: {blocked.most_common()}")
    print(f"verdicts: {Counter((r.get('critic') or {}).get('verdict') for r in records).most_common()}")


if __name__ == "__main__":
    main()
