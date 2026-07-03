#!/usr/bin/env python3
"""Shared trajectory building and replay utilities for v3.1."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from motif_lib import (
    build_dependency_graph,
    extract_motifs,
    extract_references_from_value,
    reference_pattern_stats,
    validate_references,
)
from tool_registry_v3_1 import (
    all_tool_defs,
    execute_tool,
    infer_answer_type,
    infer_output_type,
    tool_family,
    tool_pool_for_families,
)


def ref(var_idx: int, field: str = "result") -> str:
    return f"$var_{var_idx}.{field}$"


def resolve_value(val: Any, env: Dict[int, Any]) -> Any:
    if isinstance(val, str):
        refs = extract_references_from_value(val)
        if refs:
            idx, fld = refs[0]
            out = env.get(idx)
            if fld and isinstance(out, dict):
                return out.get(fld, out)
            return out
        return val
    if isinstance(val, list):
        return [resolve_value(v, env) for v in val]
    if isinstance(val, dict):
        return {k: resolve_value(v, env) for k, v in val.items()}
    return val


def replay_calls(calls: List[Dict[str, Any]]) -> Tuple[List[Any], Any, List[str]]:
    env: Dict[int, Any] = {}
    observations: List[Any] = []
    errors: List[str] = []
    last = None
    for i, call in enumerate(calls, start=1):
        name = call.get("name", "")
        raw_args = call.get("arguments") or {}
        try:
            resolved = {k: resolve_value(v, env) for k, v in raw_args.items()}
            result = execute_tool(name, resolved)
            env[i] = result
            observations.append(result)
            last = result
        except Exception as exc:
            errors.append(f"call_{i}:{name}:{exc}")
    return observations, last, errors


def process_labels_for_calls(calls: List[Dict[str, Any]]) -> List[str]:
    labels = []
    for i, call in enumerate(calls, start=1):
        refs = []
        for v in (call.get("arguments") or {}).values():
            refs.extend(extract_references_from_value(v))
        if i == 1 and not refs:
            labels.append("atomic_step")
        elif refs:
            labels.append("reference_step")
        else:
            labels.append("independent_step")
    return labels


def pack_trajectory(
    *,
    trajectory_id: str,
    source_failure_cluster: str,
    target_full_motif: str,
    question: str,
    calls: List[Dict[str, Any]],
    seed: int,
    tool_families: Optional[List[str]] = None,
    gold_answer: Any = None,
    with_distractors: bool = False,
) -> Dict[str, Any]:
    families = tool_families or list({tool_family(c["name"]) for c in calls})
    tools = tool_pool_for_families(families)
    if with_distractors:
        tools = all_tool_defs()
    observations, last, errors = replay_calls(calls)
    if errors:
        raise ValueError(f"replay failed for {trajectory_id}: {errors}")
    ans = gold_answer if gold_answer is not None else last
    output_types = [infer_output_type(c["name"], observations[i]) for i, c in enumerate(calls)]
    row = {
        "trajectory_id": trajectory_id,
        "source_failure_cluster": source_failure_cluster,
        "target_full_motif": target_full_motif,
        "full_num_calls": len(calls),
        "question": question,
        "tools": tools,
        "gold_calls": calls,
        "observations": observations,
        "gold_answer": ans,
        "process_labels": process_labels_for_calls(calls),
        "tool_family_mix": families,
        "output_type_sequence": output_types,
        "generation_seed": seed,
        "dependency_graph": build_dependency_graph(calls),
        "reference_pattern": reference_pattern_stats(calls),
        "num_calls": len(calls),
    }
    m = extract_motifs(row)
    row["motif_type"] = target_full_motif
    row["difficulty_score"] = m["difficulty_score"]
    row["answer_type"] = infer_answer_type(ans)
    row["output_type"] = infer_answer_type(last)
    return row


def truncate_trajectory(traj: Dict[str, Any], prefix_len: int) -> Dict[str, Any]:
    calls = traj["gold_calls"][:prefix_len]
    obs, last, errors = replay_calls(calls)
    if errors:
        raise ValueError(errors)
    gold_names = {c["name"] for c in calls}
    tools = [t for t in traj["tools"] if t.get("name") in gold_names]
    used_fams = list({tool_family(c["name"]) for c in calls})
    pool = {t["name"]: t for t in tool_pool_for_families(used_fams)}
    for t in tools:
        pool[t["name"]] = t
    return {
        "gold_calls": calls,
        "observations": obs,
        "tools": list(pool.values()),
        "dependency_graph": build_dependency_graph(calls),
        "process_labels": process_labels_for_calls(calls),
        "output_type_sequence": [infer_output_type(c["name"], obs[i]) for i, c in enumerate(calls)],
        "last_observation": last,
    }


def validate_trajectory(traj: Dict[str, Any]) -> List[str]:
    errors = validate_references(traj.get("gold_calls") or [])
    _, _, replay_errs = replay_calls(traj.get("gold_calls") or [])
    errors.extend(replay_errs)
    return errors
