"""NESTFUL target adapter.

All NESTFUL-specific code lives here + configs/targets/nestful.yaml.
- dev split (n=200): profiling source (hygiene rule D03);
- test split (n=1661): contamination blocklist ONLY;
- stage3_train_ready: baseline for profile-match comparison.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from targeted_tool_data.util import (normalize_query, read_jsonl,
                                     resolve_target_path, sha256_file)


def _canon_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for t in tools or []:
        params = t.get("parameters") or {}
        if isinstance(params, dict) and params.get("type") == "object":
            props = params.get("properties") or {}          # JSON-schema style
        else:
            props = params                                    # NESTFUL flat dict
        param_types = {}
        for pname, spec in props.items():
            param_types[pname] = (spec or {}).get("type", "unknown") if isinstance(spec, dict) else "unknown"
        out.append({
            "name": t.get("name", ""),
            "description": t.get("description", ""),
            "param_types": param_types,
            "output_fields": sorted((t.get("output_parameters") or {}).keys()),
        })
    return out


def _canon_row(row: Dict[str, Any], query_key: str, calls_key: str) -> Dict[str, Any]:
    return {
        "query": row.get(query_key, ""),
        "calls": row.get(calls_key) or [],
        "tools": _canon_tools(row.get("tools") or []),
        "gold_answer": row.get("gold_answer"),
    }


class NestfulAdapter:
    name = "nestful"

    def __init__(self, target_cfg: Dict[str, Any]):
        self.cfg = target_cfg
        self.dev_path = resolve_target_path(target_cfg["data"]["dev_path"])
        self.test_path = resolve_target_path(target_cfg["data"]["test_path"])
        self.stage3_path = resolve_target_path(target_cfg["data"]["stage3_path"])

    # profiling source — dev ONLY
    def canonical_dev_rows(self) -> List[Dict[str, Any]]:
        return [_canon_row(r, "input", "output") for r in read_jsonl(self.dev_path)]

    # baseline for comparison (old Stage-3 training data)
    def canonical_baseline_rows(self) -> List[Dict[str, Any]]:
        return [_canon_row(r, "question", "gold_calls") for r in read_jsonl(self.stage3_path)]

    # contamination blocklist: dev + test queries and gold skeletons
    def blocklist(self) -> Dict[str, Any]:
        exact: Set[str] = set()
        normalized: Set[str] = set()
        skeletons: Set[Tuple[str, ...]] = set()
        queries: List[str] = []
        for path in (self.dev_path, self.test_path):
            for r in read_jsonl(path):
                q = r.get("input", "")
                exact.add(q)
                normalized.add(normalize_query(q))
                queries.append(q)
                skeletons.add(tuple(c.get("name", "") for c in (r.get("output") or [])))
        return {"exact": exact, "normalized": normalized,
                "skeletons": skeletons, "queries": queries}

    def source_hashes(self) -> Dict[str, str]:
        return {
            "dev_sha256": sha256_file(self.dev_path),
            "test_sha256": sha256_file(self.test_path),
            "stage3_sha256": sha256_file(self.stage3_path),
        }

    # A-track surface conventions (morphology, not copies)
    def adaptation_conventions(self) -> Dict[str, Any]:
        return self.cfg.get("adaptation", {})

    def failure_profile(self) -> Dict[str, Any]:
        return self.cfg.get("failure_profile", {})

    # forbidden exact tool names for the G-track (all target tool names)
    def target_tool_names(self) -> Set[str]:
        names: Set[str] = set()
        for path in (self.dev_path, self.test_path):
            for r in read_jsonl(path):
                for t in r.get("tools") or []:
                    names.add(t.get("name", ""))
        return names


def make_adapter(target_cfg: Dict[str, Any]) -> NestfulAdapter:
    return NestfulAdapter(target_cfg)
