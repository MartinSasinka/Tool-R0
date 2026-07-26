"""Dummy target adapter — proves the pipeline is not NESTFUL-bound (§20)."""
from __future__ import annotations

import random
from typing import Any, Dict, List, Set


def _mk_row(i: int) -> Dict[str, Any]:
    rng = random.Random(i)
    a, b, c = rng.randint(2, 90), rng.randint(2, 60), rng.randint(2, 40)
    return {
        "query": f"Combine {a} with {b}, then reduce by {c}.",
        "calls": [
            {"name": "dummy_combine", "arguments": {"x": a, "y": b}, "label": "$var1"},
            {"name": "dummy_reduce", "arguments": {"base": "$var1.out$", "by": c},
             "label": "$var2"},
        ],
        "tools": [
            {"name": "dummy_combine", "description": "Combines two numbers.",
             "param_types": {"x": "number", "y": "number"}, "output_fields": ["out"]},
            {"name": "dummy_reduce", "description": "Reduces a base by an amount.",
             "param_types": {"base": "number", "by": "number"}, "output_fields": ["out"]},
            {"name": "dummy_noise", "description": "Unrelated helper.",
             "param_types": {"z": "string"}, "output_fields": ["out"]},
        ],
        "gold_answer": float(a + b - c),
    }


class DummyAdapter:
    name = "dummy"
    dev_path = "dummy://dev"

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    def canonical_dev_rows(self) -> List[Dict[str, Any]]:
        return [_mk_row(i) for i in range(24)]

    def canonical_baseline_rows(self) -> List[Dict[str, Any]]:
        return [_mk_row(1000 + i) for i in range(24)]

    def blocklist(self) -> Dict[str, Any]:
        rows = self.canonical_dev_rows()
        from targeted_tool_data.util import normalize_query
        return {
            "exact": {r["query"] for r in rows},
            "normalized": {normalize_query(r["query"]) for r in rows},
            "skeletons": {tuple(c["name"] for c in r["calls"]) for r in rows},
            "queries": [r["query"] for r in rows],
        }

    def source_hashes(self) -> Dict[str, str]:
        return {"dummy": "n/a"}

    def adaptation_conventions(self) -> Dict[str, Any]:
        return {"param_styles": ["semantic"], "label_styles": ["$var{i}"]}

    def failure_profile(self) -> Dict[str, Any]:
        return {"win_rate_by_call_bucket": {"2": 0.4, "3": 0.6}}

    def target_tool_names(self) -> Set[str]:
        return {"dummy_combine", "dummy_reduce", "dummy_noise"}


def make_adapter(cfg: Dict[str, Any]) -> DummyAdapter:
    return DummyAdapter(cfg)
