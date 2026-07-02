#!/usr/bin/env python3
"""Replay synthetic gold traces with local math executor (pre-training gate)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from motif_lib import (  # noqa: E402
    extract_references_from_value,
    graph_matches_refs,
    load_jsonl,
    repo_root,
    validate_references,
    write_csv,
)


from synthetic_tool_registry import execute_tool  # noqa: E402


def _resolve_value(val: Any, env: Dict[int, Any]) -> Any:
    if isinstance(val, str):
        refs = extract_references_from_value(val)
        if refs:
            idx, field = refs[0]
            out = env.get(idx)
            if field and isinstance(out, dict):
                return out.get(field, out)
            return out
        return val
    if isinstance(val, list):
        return [_resolve_value(v, env) for v in val]
    if isinstance(val, dict):
        return {k: _resolve_value(v, env) for k, v in val.items()}
    return val


def _answers_match(replay_ans: Any, gold: Any) -> bool:
    if isinstance(gold, (int, float)) and isinstance(replay_ans, (int, float)):
        return abs(float(replay_ans) - float(gold)) < 1e-6
    if isinstance(gold, list) and isinstance(replay_ans, list):
        if len(gold) != len(replay_ans):
            return False
        return all(_answers_match(a, b) for a, b in zip(gold, replay_ans))
    if isinstance(gold, dict) and isinstance(replay_ans, dict):
        return gold == replay_ans
    return replay_ans == gold


def replay_task(task: dict) -> tuple[bool, str, Any]:
    tid = task.get("task_id", "")
    try:
        json.dumps(task)
    except Exception as exc:
        return False, f"json_invalid:{exc}", None

    tools = task.get("tools") or []
    if not tools:
        return False, "missing_tools", None

    calls = task.get("gold_calls") or []
    if not calls:
        return False, "missing_gold_calls", None

    tool_by_name = {t.get("name"): t for t in tools}
    for i, c in enumerate(calls, start=1):
        if c.get("name") not in tool_by_name:
            return False, f"call_{i}:unknown_tool:{c.get('name')}", None
        params = tool_by_name[c.get("name")].get("parameters") or {}
        required = params.get("required") or []
        args = c.get("arguments") or {}
        for r in required:
            if r not in args:
                return False, f"call_{i}:missing_required:{r}", None

    ref_errs = validate_references(calls)
    if ref_errs:
        return False, f"invalid_reference:{';'.join(ref_errs)}", None

    if not graph_matches_refs(calls, task.get("dependency_graph") or {}):
        return False, "dependency_graph_mismatch", None

    env: Dict[int, Any] = {}
    for i, call in enumerate(calls, start=1):
        raw_args = call.get("arguments") or {}
        resolved = _resolve_value(raw_args, env)
        if not isinstance(resolved, dict):
            return False, f"call_{i}:bad_arguments", None
        try:
            out = execute_tool(str(call.get("name")), resolved)
        except Exception as exc:
            return False, f"call_{i}:exec_error:{exc}", None
        env[i] = {"result": out}

    replay_ans = env[len(calls)]["result"]
    gold = task.get("gold_answer")
    if not _answers_match(replay_ans, gold):
        return False, f"answer_mismatch:replay={replay_ans},gold={gold}", replay_ans
    return True, "", replay_ans


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input",
        type=Path,
        default=repo_root() / "experiments/nestful_synthetic_curriculum_v3/outputs/synthetic_motif_tasks.jsonl",
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=repo_root() / "experiments/nestful_synthetic_curriculum_v3/outputs",
    )
    args = ap.parse_args()

    if not args.input.is_file():
        cur = repo_root() / "experiments/nestful_synthetic_curriculum_v3/outputs/curriculum_v3"
        paths = sorted(cur.glob("stage*.jsonl")) if cur.is_dir() else []
        tasks = []
        for p in paths:
            tasks.extend(load_jsonl(p))
    else:
        tasks = load_jsonl(args.input)

    if not tasks:
        print("ERROR: no tasks to replay", file=sys.stderr)
        return 1

    failures = []
    invalid_refs = 0
    for t in tasks:
        ok, err, _ = replay_task(t)
        if not ok:
            failures.append({"task_id": t.get("task_id"), "error": err})
            if "invalid_reference" in err:
                invalid_refs += 1

    n = len(tasks)
    success_rate = (n - len(failures)) / max(n, 1)
    status = "PASS" if not failures else "FAIL"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "synthetic_gold_replay_failures.csv", failures, ["task_id", "error"])

    summary = {
        "status": status,
        "tasks": n,
        "successes": n - len(failures),
        "failures": len(failures),
        "gold_replay_success_rate": round(success_rate, 6),
        "invalid_reference_count": invalid_refs,
    }
    (args.out_dir / "synthetic_gold_replay_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    report = [
        "# Synthetic Gold Replay Report",
        "",
        f"Status: **{status}**",
        f"- tasks replayed: {n}",
        f"- successes: {n - len(failures)}",
        f"- failures: {len(failures)}",
        f"- gold replay success rate: {success_rate:.4f}",
        f"- invalid references: {invalid_refs}",
        "",
        "## Hard gate",
        "- gold_replay_success_rate must be 1.0",
        "- invalid_reference count must be 0",
        "- replay answer must match gold_answer",
        "",
        f"Result: **{status}**",
    ]
    (args.out_dir / "synthetic_gold_replay_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"[replay_synthetic_gold_traces] {status} rate={success_rate:.4f} fails={len(failures)}")
    if failures or invalid_refs or success_rate < 1.0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
