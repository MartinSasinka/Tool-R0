#!/usr/bin/env python3
"""Gold replay for v3.1 full trajectories and prefix samples."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from motif_lib import graph_matches_refs, load_jsonl, repo_root, validate_references, write_csv  # noqa: E402
from traj_utils_v3_1 import replay_calls  # noqa: E402
from tool_registry_v3_1 import ALL_TOOL_NAMES  # noqa: E402


def _answers_match(replay_ans: Any, gold: Any) -> bool:
    if gold is None:
        return True
    if isinstance(gold, (int, float)) and isinstance(replay_ans, (int, float)):
        return abs(float(replay_ans) - float(gold)) < 1e-6
    if isinstance(gold, list) and isinstance(replay_ans, list):
        return len(gold) == len(replay_ans) and all(_answers_match(a, b) for a, b in zip(gold, replay_ans))
    if isinstance(gold, dict) and isinstance(replay_ans, dict):
        return gold == replay_ans
    return replay_ans == gold


def replay_sample(sample: dict) -> Tuple[bool, str]:
    sid = sample.get("sample_id") or sample.get("trajectory_id") or "?"
    calls = sample.get("gold_calls") or []
    if not calls:
        return False, "missing_gold_calls"

    for i, c in enumerate(calls, start=1):
        if c.get("name") not in ALL_TOOL_NAMES:
            return False, f"call_{i}:unknown_tool"

    ref_errs = validate_references(calls)
    if ref_errs:
        return False, f"invalid_reference:{';'.join(ref_errs)}"

    if not graph_matches_refs(calls, sample.get("dependency_graph") or {}):
        return False, "dependency_graph_mismatch"

    obs, last, errs = replay_calls(calls)
    if errs:
        return False, f"exec_error:{';'.join(errs)}"

    gold = sample.get("gold_answer")
    if gold is not None and not _answers_match(last, gold):
        return False, f"answer_mismatch:replay={last},gold={gold}"

    stored_obs = sample.get("observations")
    if stored_obs is not None and len(stored_obs) == len(obs):
        for i, (a, b) in enumerate(zip(stored_obs, obs)):
            if not _answers_match(a, b):
                return False, f"observation_mismatch at {i}"

    return True, ""


def collect_tasks(in_dir: Path, use_filtered: bool) -> List[dict]:
    tasks: List[dict] = []
    full = in_dir / "full_trajectories.jsonl"
    if full.is_file():
        tasks.extend(load_jsonl(full))
    base = in_dir / ("filtered" if use_filtered else "")
    stage_dir = base if use_filtered else in_dir
    for fname in sorted(stage_dir.glob("stage*.jsonl")):
        tasks.extend(load_jsonl(fname))
    return tasks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", type=Path,
                    default=repo_root() / "experiments/nestful_synthetic_curriculum_v3/outputs/curriculum_v3_1")
    ap.add_argument("--filtered", action="store_true", default=True)
    ap.add_argument("--no-filtered", action="store_false", dest="filtered")
    args = ap.parse_args()

    tasks = collect_tasks(args.in_dir, args.filtered)
    if not tasks:
        print("ERROR: no tasks", file=sys.stderr)
        return 1

    failures = []
    invalid_refs = 0
    for t in tasks:
        ok, err = replay_sample(t)
        tid = t.get("sample_id") or t.get("trajectory_id") or t.get("task_id")
        if not ok:
            failures.append({"task_id": tid, "error": err})
            if "invalid_reference" in err:
                invalid_refs += 1

    n = len(tasks)
    rate = (n - len(failures)) / max(n, 1)
    status = "PASS" if rate >= 1.0 and invalid_refs == 0 else "FAIL"

    args.in_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.in_dir / "synthetic_gold_replay_failures.csv", failures, ["task_id", "error"])
    summary = {
        "status": status,
        "tasks": n,
        "successes": n - len(failures),
        "failures": len(failures),
        "gold_replay_success_rate": round(rate, 6),
        "invalid_reference_count": invalid_refs,
    }
    (args.in_dir / "synthetic_gold_replay_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    report = [
        "# Synthetic Gold Replay Report (v3.1)",
        "",
        f"Status: **{status}**",
        f"- tasks: {n}",
        f"- gold_replay_success_rate: {rate:.4f}",
        f"- invalid_reference_count: {invalid_refs}",
        "",
        "Gate: gold_replay_success_rate = 1.0",
    ]
    (args.in_dir / "SYNTHETIC_GOLD_REPLAY_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"[replay_v3_1] {status} rate={rate:.4f}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
