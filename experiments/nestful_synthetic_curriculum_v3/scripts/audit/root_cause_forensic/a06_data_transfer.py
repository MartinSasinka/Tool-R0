"""A06 — Stage-2 / Stage-3 / NESTFUL structural distribution audit.

Compares train_subset_160 (Round-1), stage3_train_ready (pure-S3), stage2
(phase-1) against the NESTFUL diagnostic 500 + full test: call counts, tool
name overlap, argument types, variable-reference usage, answer types.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from common import MINIMAL, V3, eval_ids_500, load_jsonl, write_json

REF_RE = re.compile(r"\$var\d+")


def _gold_calls(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    for key in ("gold_calls", "output", "answers", "gold"):
        v = row.get(key)
        if isinstance(v, list) and v and isinstance(v[0], dict) and "name" in v[0]:
            return v
    return []


def _tool_names(row: Dict[str, Any]) -> List[str]:
    tools = row.get("tools") or []
    return [t.get("name") for t in tools if isinstance(t, dict) and t.get("name")]


def _arg_types(calls: List[Dict[str, Any]]) -> Counter:
    c = Counter()
    for call in calls:
        for v in (call.get("arguments") or {}).values():
            if isinstance(v, str) and REF_RE.search(v):
                c["reference"] += 1
            else:
                c[type(v).__name__] += 1
    return c


def _profile(rows: List[Dict[str, Any]], name: str) -> Dict[str, Any]:
    call_counts = Counter()
    gold_tool_names = Counter()
    offered_tools = Counter()
    arg_types = Counter()
    ref_rows = 0
    n = 0
    for r in rows:
        calls = _gold_calls(r)
        nc = r.get("num_calls") or r.get("num_gold_calls") or len(calls)
        call_counts[min(int(nc), 6) if int(nc) < 6 else 6] += 1
        for c in calls:
            gold_tool_names[c.get("name")] += 1
        for t in _tool_names(r):
            offered_tools[t] += 1
        at = _arg_types(calls)
        arg_types.update(at)
        if at.get("reference"):
            ref_rows += 1
        n += 1
    return {
        "name": name,
        "n_rows": n,
        "gold_call_count_hist": {str(k): v for k, v in sorted(call_counts.items())},
        "n_unique_gold_tools": len(gold_tool_names),
        "top_gold_tools": gold_tool_names.most_common(15),
        "n_unique_offered_tools": len(offered_tools),
        "arg_type_hist": dict(arg_types.most_common()),
        "rows_with_reference_args_frac": ref_rows / n if n else None,
        "_gold_tool_set": set(gold_tool_names),
    }


def _maybe(path: Path) -> Optional[List[Dict[str, Any]]]:
    return load_jsonl(path) if path.is_file() else None


def main() -> Dict[str, Any]:
    train160 = load_jsonl(V3 / "reports" / "reward_ablation" / "data" / "train_subset_160.jsonl")
    stage3 = _maybe(V3 / "data" / "training_ready_v5" / "filtered" / "stage3_train_ready.jsonl")
    stage2 = (_maybe(V3 / "data" / "training_ready_v5" / "filtered" / "stage2_train_ready.jsonl")
              or _maybe(V3 / "data" / "curriculum_v4_nestful_like_agentic_openrouter" / "filtered"
                        / "stage2_2call_agentic_openrouter.jsonl"))
    nestful_test = load_jsonl(MINIMAL / "data" / "splits" / "nestful_test.jsonl")
    ids = set(eval_ids_500())
    nestful_500 = [r for r in nestful_test
                   if str(r.get("sample_id") or r.get("task_id")) in ids]

    profiles = [_profile(train160, "train_subset_160 (Round-1)")]
    if stage3 is not None:
        profiles.append(_profile(stage3, "stage3_train_ready (pure-S3, 326)"))
    if stage2 is not None:
        profiles.append(_profile(stage2, "stage2 (phase-1)"))
    profiles.append(_profile(nestful_500, "nestful_diagnostic_500"))
    profiles.append(_profile(nestful_test, "nestful_test_full"))

    # overlaps vs nestful 500
    nest_tools = next(p for p in profiles if p["name"] == "nestful_diagnostic_500")["_gold_tool_set"]
    overlaps = {}
    for p in profiles:
        if p["name"].startswith("nestful"):
            continue
        inter = p["_gold_tool_set"] & nest_tools
        overlaps[p["name"]] = {
            "n_train_tools": len(p["_gold_tool_set"]),
            "n_nestful_tools": len(nest_tools),
            "n_shared": len(inter),
            "shared_sample": sorted(inter)[:20],
            "jaccard": len(inter) / len(p["_gold_tool_set"] | nest_tools)
            if (p["_gold_tool_set"] | nest_tools) else None,
        }

    for p in profiles:
        p.pop("_gold_tool_set", None)

    payload = {"profiles": profiles, "gold_tool_overlap_vs_nestful500": overlaps}
    write_json("a06_data_transfer.json", payload)
    return payload


if __name__ == "__main__":
    r = main()
    for p in r["profiles"]:
        print(p["name"], p["n_rows"], p["gold_call_count_hist"],
              "ref rows", p["rows_with_reference_args_frac"])
    for k, v in r["gold_tool_overlap_vs_nestful500"].items():
        print(k, "shared tools:", v["n_shared"], "/", v["n_train_tools"], "jaccard", v["jaccard"])
