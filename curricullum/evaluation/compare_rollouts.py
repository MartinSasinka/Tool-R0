#!/usr/bin/env python3
"""Compare NESTFUL multiturn summaries with matched rollout counts."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

REPO = Path(__file__).resolve().parents[2]


def load_rows(path: Path, rollout_min: int = 0, rollout_max: int = 8) -> List[dict]:
    rows: List[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            ri = int(r.get("rollout_idx", 0))
            if rollout_min <= ri < rollout_max:
                rows.append(r)
    return rows


def summarize(rows: List[dict]) -> Dict:
    total = len(rows)
    passed = sum(1 for r in rows if r.get("status") == "completed")
    failed = total - passed
    by_task: Dict[str, List[bool]] = defaultdict(list)
    for r in rows:
        by_task[r["task_id"]].append(r.get("status") == "completed")
    tasks = len(by_task)
    pass_at_k = (
        sum(1 for ps in by_task.values() if any(ps)) / tasks if tasks else 0.0
    )
    no_calls = sum(1 for r in rows if int(r.get("num_tool_calls", 0)) == 0)
    return {
        "rollouts": total,
        "unique_tasks": tasks,
        "rollouts_per_task": round(total / tasks, 2) if tasks else 0,
        "accuracy_percent": round(100 * passed / total, 2) if total else 0,
        "passed": passed,
        "failed": failed,
        "pass_at_k_percent": round(100 * pass_at_k, 2),
        "no_calls_percent": round(100 * no_calls / total, 2) if total else 0,
        "avg_tool_calls": round(
            sum(int(r.get("num_tool_calls", 0)) for r in rows) / total, 2
        )
        if total
        else 0,
        "avg_steps": round(
            sum(int(r.get("num_steps", 0)) for r in rows) / total, 2
        )
        if total
        else 0,
        "top_stop": Counter(r.get("stopped") for r in rows).most_common(3),
        "verdict_reason": Counter(r.get("verdict_reason") for r in rows).most_common(5),
        "error_category_top": Counter(
            r.get("error_category") for r in rows if r.get("status") != "completed"
        ).most_common(8),
        "no_calls_failures": sum(
            1
            for r in rows
            if r.get("status") != "completed"
            and (r.get("error_category") == "no_more_calls" or int(r.get("num_tool_calls", 0)) == 0)
        ),
        "failure_buckets": _failure_buckets(rows),
    }


def _failure_buckets(rows: List[dict]) -> Dict[str, int]:
    buckets: Counter = Counter()
    for r in rows:
        if r.get("status") == "completed":
            continue
        ntc = int(r.get("num_tool_calls", 0))
        stopped = r.get("stopped", "")
        vr = r.get("verdict_reason", "")
        if ntc == 0:
            buckets["zero_tool_calls"] += 1
        elif stopped == "execution_error":
            buckets["execution_error"] += 1
        elif stopped == "step_limit":
            buckets["step_limit"] += 1
        elif stopped == "converged":
            buckets["converged_loop"] += 1
        elif vr == "executor_mismatch":
            buckets["wrong_answer_after_tools"] += 1
        elif vr == "no_final_value":
            buckets["no_final_value"] += 1
        else:
            buckets[f"other:{stopped}"] += 1
    return dict(buckets)


def task_pass_map(rows: List[dict]) -> Dict[str, bool]:
    by_task: Dict[str, List[bool]] = defaultdict(list)
    for r in rows:
        by_task[r["task_id"]].append(r.get("status") == "completed")
    return {tid: any(ps) for tid, ps in by_task.items()}


def head_to_head(
    base_rows: List[dict], other_rows: List[dict]
) -> Dict[str, int]:
    base = task_pass_map(base_rows)
    other = task_pass_map(other_rows)
    gained = lost = both = neither = 0
    for tid in base:
        bp, op = base[tid], other.get(tid, False)
        if bp and op:
            both += 1
        elif bp and not op:
            lost += 1
        elif not bp and op:
            gained += 1
        else:
            neither += 1
    return {
        "tasks_gained": gained,
        "tasks_lost": lost,
        "tasks_both_pass": both,
        "tasks_both_fail": neither,
        "net_tasks": gained - lost,
    }


def main() -> None:
    base_path = REPO / "curricullum/evaluation/results_toolr0/curriculum_baseline_multiturn_predictions.jsonl"
    s3_path = REPO / "curricullum/evaluation/results_v2_20260617/curriculum_stage_3_epoch1_multiturn_predictions.jsonl"
    s5_path = REPO / "curricullum/evaluation/results_v2_20260617/curriculum_stage_5_epoch2_multiturn_predictions.jsonl"

    configs = [
        ("baseline_4r (idx 0-3)", base_path, 0, 4),
        ("baseline_8r (all)", base_path, 0, 8),
        ("stage_3_epoch1", s3_path, 0, 4),
        ("stage_5_epoch2", s5_path, 0, 4),
    ]

    rows_by_label = {}
    out = {}
    for label, path, lo, hi in configs:
        rows = load_rows(path, lo, hi)
        rows_by_label[label] = rows
        out[label] = summarize(rows)

    base4 = rows_by_label["baseline_4r (idx 0-3)"]
    out["deltas_vs_baseline_4r"] = {
        "stage_3_epoch1": {
            "rollout_accuracy_pp": round(
                out["stage_3_epoch1"]["accuracy_percent"]
                - out["baseline_4r (idx 0-3)"]["accuracy_percent"],
                2,
            ),
            "pass_at_4_pp": round(
                out["stage_3_epoch1"]["pass_at_k_percent"]
                - out["baseline_4r (idx 0-3)"]["pass_at_k_percent"],
                2,
            ),
            "no_calls_pp": round(
                out["stage_3_epoch1"]["no_calls_percent"]
                - out["baseline_4r (idx 0-3)"]["no_calls_percent"],
                2,
            ),
            **head_to_head(base4, rows_by_label["stage_3_epoch1"]),
        },
        "stage_5_epoch2": {
            "rollout_accuracy_pp": round(
                out["stage_5_epoch2"]["accuracy_percent"]
                - out["baseline_4r (idx 0-3)"]["accuracy_percent"],
                2,
            ),
            "pass_at_4_pp": round(
                out["stage_5_epoch2"]["pass_at_k_percent"]
                - out["baseline_4r (idx 0-3)"]["pass_at_k_percent"],
                2,
            ),
            "no_calls_pp": round(
                out["stage_5_epoch2"]["no_calls_percent"]
                - out["baseline_4r (idx 0-3)"]["no_calls_percent"],
                2,
            ),
            **head_to_head(base4, rows_by_label["stage_5_epoch2"]),
        },
        "stage_5_vs_stage_3": {
            "rollout_accuracy_pp": round(
                out["stage_5_epoch2"]["accuracy_percent"]
                - out["stage_3_epoch1"]["accuracy_percent"],
                2,
            ),
            "pass_at_4_pp": round(
                out["stage_5_epoch2"]["pass_at_k_percent"]
                - out["stage_3_epoch1"]["pass_at_k_percent"],
                2,
            ),
            **head_to_head(
                rows_by_label["stage_3_epoch1"], rows_by_label["stage_5_epoch2"]
            ),
        },
    }

    out_dir = REPO / "curricullum/evaluation/results_v2_20260617"
    out_path = out_dir / "comparison_4rollouts.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    baseline4_summary = {
        "benchmark": "nestful",
        "model_profile": "curriculum_baseline_4rollouts",
        "note": "Subset of 8-rollout baseline run: rollout_idx 0-3 only",
        "num_unique_tasks": out["baseline_4r (idx 0-3)"]["unique_tasks"],
        "num_rollouts_per_task": 4,
        "total_rollouts": out["baseline_4r (idx 0-3)"]["rollouts"],
        "final_answer_accuracy": out["baseline_4r (idx 0-3)"]["accuracy_percent"] / 100,
        "final_answer_accuracy_percent": out["baseline_4r (idx 0-3)"]["accuracy_percent"],
        "passed": out["baseline_4r (idx 0-3)"]["passed"],
        "failed": out["baseline_4r (idx 0-3)"]["failed"],
        "avg_tool_calls": out["baseline_4r (idx 0-3)"]["avg_tool_calls"],
        "avg_steps": out["baseline_4r (idx 0-3)"]["avg_steps"],
    }
    (out_dir / "curriculum_baseline_4rollout_summary.json").write_text(
        json.dumps(baseline4_summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
