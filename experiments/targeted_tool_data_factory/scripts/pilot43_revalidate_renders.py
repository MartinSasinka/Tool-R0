"""Re-score already-rendered queries against the current deterministic validators.

The LLM text is fixed and paid for; only the checks changed. Running them offline
tells us how much a validator fix buys before another OpenRouter stage is started.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from targeted_tool_data.pilot43 import qstage
from targeted_tool_data.pilot43.qvalidate import validate_query


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/pilot4_3_nestful_final")
    ap.add_argument("--profile", default="outputs/pilot4_3_nestful_final/"
                                         "target_profile_v3.json")
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--show", type=int, default=6)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    profile = json.loads(Path(args.profile).read_text(encoding="utf-8"))

    rendered = {}
    with (out_dir / "llm_rendered.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if rec.get("query"):
                rendered[rec["task_id"]] = rec

    tasks = qstage.build_render_tasks(out_dir, profile=profile, limit=args.limit)
    before, after = Counter(), Counter()
    fixed, still, examples = 0, 0, []
    for task in tasks:
        rec = rendered.get(task["task_id"])
        if not rec:
            continue
        old = rec.get("validation") or {}
        new = validate_query(rec["query"], task["validator_contract"])
        for layer in old.get("failed_layers", []):
            before[layer] += 1
        for layer in new["failed_layers"]:
            after[layer] += 1
        if old.get("passed") and not new["passed"]:
            still += 1
            if len(examples) < args.show:
                examples.append((task["task_id"], "REGRESSED",
                                 new["failed_layers"], rec["query"]))
        elif not old.get("passed") and new["passed"]:
            fixed += 1
        elif not new["passed"] and len(examples) < args.show:
            examples.append((task["task_id"], "STILL FAILS",
                             new["failed_layers"], rec["query"]))

    n = sum(1 for t in tasks if t["task_id"] in rendered)
    print(f"rendered queries re-scored: {n}")
    print(f"deterministic pass before: "
          f"{sum(1 for t in tasks if (rendered.get(t['task_id']) or {}).get('validation', {}).get('passed')) / max(n, 1):.4f}")
    print(f"newly passing: {fixed}   newly failing: {still}")
    print(f"failed layers before: {before.most_common()}")
    print(f"failed layers after:  {after.most_common()}")
    for tid, kind, layers, query in examples:
        print(f"\n[{kind}] {tid} {layers}\n  {query}")


if __name__ == "__main__":
    main()
