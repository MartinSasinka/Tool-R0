"""Shared helpers for pure Stage-3 diagnostic pack pipeline."""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from motif_lib import extract_motifs, load_jsonl, load_task_row
from rollout import Trajectory, Turn
from scripts.analysis.two_phase_root_cause_analysis import classify_failure, official_win


def load_traj_rows(eval_dir: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    with open(eval_dir / "final_eval_trajectories.jsonl", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                out[r["sample_id"]] = r
    return out


def load_tasks(path: Path) -> Dict[str, dict]:
    return {load_task_row(r)["task_id"]: load_task_row(r) for r in load_jsonl(path)}


def traj_from_dict(d: dict) -> Trajectory:
    turns: List[Turn] = []
    for t in d.get("turns") or []:
        turns.append(Turn(
            turn_idx=int(t.get("turn_idx") or 0),
            model_text=str(t.get("model_text") or ""),
            parsed_call=t.get("parsed_call"),
            observation=t.get("observation"),
            fail_reason=t.get("fail_reason"),
            is_terminal=bool(t.get("is_terminal")),
            prompt_tokens=int(t.get("prompt_tokens") or 0),
            completion_tokens=int(t.get("completion_tokens") or 0),
            clipped_completion=bool(t.get("clipped_completion")),
            teacher_forced=bool(t.get("teacher_forced")),
        ))
    return Trajectory(
        task_id=str(d.get("task_id") or ""),
        stage=int(d.get("stage") or 3),
        gold_num_turns=int(d.get("gold_num_turns") or 0),
        turns=turns,
        stop_reason=d.get("stop_reason"),
        executor_mode=str(d.get("executor_mode") or "gold_replay"),
        clipped_any=bool(d.get("clipped_any")),
        prompt_overflow=bool(d.get("prompt_overflow")),
    )


def predicted_calls(row: dict) -> List[dict]:
    turns = (row.get("_traj") or {}).get("turns") or []
    return [dict(t["parsed_call"]) for t in turns if t.get("parsed_call")]


def observations(row: dict) -> List[Any]:
    turns = (row.get("_traj") or {}).get("turns") or []
    return [
        t.get("observation")
        for t in turns
        if t.get("parsed_call") and t.get("fail_reason") is None
    ]


def compact_value(val: Any, *, max_str: int = 160) -> Any:
    if val is None or isinstance(val, (bool, int, float)):
        return val
    if isinstance(val, str):
        if len(val) <= max_str:
            return val
        return {"type": "string", "len": len(val), "preview": val[:max_str]}
    if isinstance(val, list):
        return {
            "type": "list",
            "len": len(val),
            "preview": [compact_value(x, max_str=80) for x in val[:3]],
        }
    if isinstance(val, dict):
        return {
            "type": "object",
            "keys": list(val.keys())[:12],
            "preview": {
                k: compact_value(val[k], max_str=60)
                for k in list(val.keys())[:4]
            },
        }
    return {"type": type(val).__name__}


def observation_shape(val: Any) -> str:
    if val is None:
        return "null"
    if isinstance(val, bool):
        return "boolean"
    if isinstance(val, (int, float)):
        return "scalar"
    if isinstance(val, str):
        return "string"
    if isinstance(val, list):
        return f"list[{len(val)}]"
    if isinstance(val, dict):
        return f"object[{len(val)}]"
    return type(val).__name__


def first_divergence_turn(c0_calls: List[dict], e2_calls: List[dict]) -> Optional[int]:
    for i, (a, b) in enumerate(zip(c0_calls, e2_calls), start=1):
        if (a.get("name") or "") != (b.get("name") or ""):
            return i
        pa = a.get("arguments") or {}
        pb = b.get("arguments") or {}
        if set(pa.keys()) != set(pb.keys()) or pa != pb:
            return i
    if len(c0_calls) != len(e2_calls):
        return min(len(c0_calls), len(e2_calls)) + 1
    return None


def reward_mismatch_c0_e2(r0_c0: float, r0_e2: float, w0: bool, w2: bool) -> bool:
    if w0 and not w2 and r0_e2 > r0_c0:
        return True
    if not w0 and w2 and r0_e2 < r0_c0:
        return True
    return False


def relevant_tools(task: dict, *call_lists: Iterable[List[dict]]) -> List[dict]:
    names: Set[str] = set()
    for calls in call_lists:
        for c in calls:
            n = c.get("name")
            if n:
                names.add(str(n))
    for gc in task.get("gold_calls") or []:
        if gc.get("name"):
            names.add(str(gc["name"]))
    out = []
    for tl in task.get("tools") or []:
        if tl.get("name") in names:
            slim = {
                "name": tl.get("name"),
                "description": (tl.get("description") or "")[:280],
                "parameters": tl.get("parameters"),
            }
            out.append(slim)
    return out


def arm_snapshot(
    row: dict,
    variants: Dict[str, Any],
) -> dict:
    traj = row.get("_traj") or {}
    ow = official_win(row) == 1.0
    fail = classify_failure(row)[0]
    return {
        "calls": predicted_calls(row),
        "observations": [compact_value(o) for o in observations(row)],
        "final_outcome": "win" if ow else "loss",
        "official_win": ow,
        "failure_taxonomy": fail,
        "executable": bool(traj.get("executable")),
        "num_calls": len(predicted_calls(row)),
        "reward_components": {
            k: variant_to_public(v) for k, v in variants.items()
        },
    }


def variant_to_public(v) -> dict:
    return {
        "terminal_class": v.terminal_class,
        "process_score": v.process_score,
        "total_reward": v.total_reward,
        "terminal_reward": v.terminal_reward,
        "components": v.components,
    }


def stratified_sample(
    items: List[str],
    meta: Dict[str, dict],
    n: int,
    *,
    seed: int = 20260723,
) -> List[str]:
    if len(items) <= n:
        return list(items)
    rng = random.Random(seed)
    buckets: Dict[tuple, List[str]] = defaultdict(list)
    for tid in items:
        m = meta[tid]
        key = (
            m.get("gold_call_bucket", "?"),
            m.get("motif", "?"),
            m.get("failure_type", "?")[:24],
            m.get("reward_mismatch", False),
        )
        buckets[key].append(tid)
    picked: List[str] = []
    bucket_keys = list(buckets.keys())
    rng.shuffle(bucket_keys)
    while len(picked) < n and bucket_keys:
        progressed = False
        for key in bucket_keys:
            if buckets[key]:
                picked.append(buckets[key].pop(rng.randrange(len(buckets[key]))))
                progressed = True
                if len(picked) >= n:
                    break
        if not progressed:
            break
    if len(picked) < n:
        rest = [x for x in items if x not in picked]
        rng.shuffle(rest)
        picked.extend(rest[: n - len(picked)])
    return picked[:n]


def tool_at(calls: List[dict], idx: int) -> Optional[str]:
    if idx < len(calls):
        return calls[idx].get("name")
    return None


def tool_description(task: dict, name: Optional[str]) -> str:
    if not name:
        return ""
    for tl in task.get("tools") or []:
        if tl.get("name") == name:
            return (tl.get("description") or "")[:220]
    return ""
