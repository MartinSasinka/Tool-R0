#!/usr/bin/env python3
"""Offline rescore: execution-aware vs partial vs strict reward vs official Win.

Reads saved ``final_eval_trajectories.jsonl`` (or rollout_eval), reconstructs
``Trajectory`` objects, scores each episode with execution / partial / strict
training rewards, and compares to per-sample ``official_win`` already stored
at eval time.

Usage (from nestful_mtgrpo_partial/):
    python scripts/rescore_execution_reward.py
    python scripts/rescore_execution_reward.py --glob "outputs/final_eval/*_react/*trajectories.jsonl"
    python scripts/rescore_execution_reward.py --out comparison/execution_reward_correlation.csv

No GPU / no re-generation. Uses ``official_win`` from the trajectory file
(head-to-head Win from the official NESTFUL scorer at eval time).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_SIBLING = os.path.join(os.path.dirname(_ROOT), "nestful_mtgrpo_minimal")
_DEFAULT_DATA = os.path.join(_SIBLING, "data", "NESTFUL-main", "data_v2", "nestful_data.jsonl")
_DEFAULT_GLOB = os.path.join(_ROOT, "outputs", "final_eval", "*_react", "final_eval_trajectories.jsonl")

for p in (_ROOT, _SIBLING, os.path.join(_SIBLING, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from data import load_tasks  # noqa: E402
from reward import compute_gold_observations, strict_gold_trace_reward  # noqa: E402
from rollout import Trajectory, Turn  # noqa: E402
import partial_reward as pr  # noqa: E402
import execution_reward as er  # noqa: E402


def _pearson(xs: List[float], ys: List[float]) -> Optional[float]:
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def _rankdata(vals: List[float]) -> List[float]:
    """Average ranks for ties (Spearman helper)."""
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _spearman(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) < 2:
        return None
    return _pearson(_rankdata(xs), _rankdata(ys))


def _traj_from_dict(d: Dict[str, Any]) -> Trajectory:
    traj = Trajectory(
        task_id=str(d.get("task_id", "")),
        stage=int(d.get("stage") or d.get("gold_num_turns") or 0),
        gold_num_turns=int(d.get("gold_num_turns") or 0),
        executor_mode=str(d.get("executor_mode", "full")),
        stop_reason=d.get("stop_reason"),
        clipped_any=bool(d.get("clipped_any", False)),
        prompt_overflow=bool(d.get("prompt_overflow", False)),
    )
    for t in d.get("turns") or []:
        traj.turns.append(Turn(
            turn_idx=int(t.get("turn_idx", 0)),
            model_text=str(t.get("model_text") or ""),
            parsed_call=t.get("parsed_call"),
            observation=t.get("observation"),
            fail_reason=t.get("fail_reason"),
            is_terminal=bool(t.get("is_terminal", False)),
            clipped_completion=bool(t.get("clipped_completion", False)),
        ))
    if d.get("pred_answer") is not None:
        traj.final_observation = d["pred_answer"]
    else:
        for t in reversed(traj.turns):
            if t.parsed_call is not None and t.fail_reason is None:
                traj.final_observation = t.observation
                break
    return traj


def _official_win(row: Dict[str, Any], traj_d: Dict[str, Any]) -> Optional[float]:
    for src in (row, traj_d):
        for key in ("official_win", "internal_win_rate"):
            if key in src and src[key] is not None:
                return float(src[key])
    return None


def score_trajectories_file(
    path: str,
    tasks_by_id: Dict[str, Dict[str, Any]],
    *,
    run_label: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows_out: List[Dict[str, Any]] = []
    skipped = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sid = str(row.get("sample_id") or row.get("task_id") or "")
            task = tasks_by_id.get(sid)
            if task is None:
                skipped += 1
                continue
            traj_d = row.get("_traj", row)
            traj = _traj_from_dict(traj_d)
            gold_obs = compute_gold_observations(task)
            rr_exec = er.execution_aware_reward(traj, task, gold_obs)
            rr_part = pr.partial_gold_trace_reward(traj, task, gold_obs)
            rr_strict = strict_gold_trace_reward(traj, task, gold_obs)
            win = _official_win(row, traj_d)
            if win is None:
                skipped += 1
                continue
            de = rr_exec.diagnostics
            rows_out.append({
                "run": run_label,
                "sample_id": sid,
                "official_win": win,
                "execution_reward": rr_exec.reward,
                "partial_reward": rr_part.reward,
                "strict_reward": rr_strict.reward,
                "final_answer_pass": float(row.get("final_answer_pass")
                                           or traj_d.get("paper", {}).get("final_answer_pass")
                                           or de.get("final_answer_pass", 0)),
                "tool_final_answer_pass": de.get("tool_final_answer_pass", 0),
                "executable_trajectory": de.get("executable_trajectory", 0),
                "tool_use_completeness": de.get("tool_use_completeness", 0),
                "valid_references": de.get("valid_references", 0),
                "small_gold_trace_progress": de.get("small_gold_trace_progress", 0),
                "execution_cap": de.get("cap_applied") or "",
                "num_successful_calls": de.get("num_successful_calls", 0),
                "gold_num_calls": task.get("num_calls", 0),
            })

    summary = _summarize(rows_out, run_label=run_label, path=path, skipped=skipped)
    return rows_out, summary


def _summarize(
    rows: List[Dict[str, Any]], *, run_label: str, path: str, skipped: int,
) -> Dict[str, Any]:
    if not rows:
        return {"run": run_label, "path": path, "n": 0, "skipped": skipped}

    win = [r["official_win"] for r in rows]
    metrics = {
        "execution_reward": [r["execution_reward"] for r in rows],
        "partial_reward": [r["partial_reward"] for r in rows],
        "strict_reward": [r["strict_reward"] for r in rows],
        "final_answer_pass": [r["final_answer_pass"] for r in rows],
        "tool_final_answer_pass": [r["tool_final_answer_pass"] for r in rows],
    }
    out: Dict[str, Any] = {
        "run": run_label,
        "path": path,
        "n": len(rows),
        "skipped": skipped,
        "mean_official_win": sum(win) / len(win),
    }
    for name, vals in metrics.items():
        out[f"mean_{name}"] = sum(vals) / len(vals)
        out[f"pearson_{name}_vs_win"] = _pearson(vals, win)
        out[f"spearman_{name}_vs_win"] = _spearman(vals, win)
    # Mean reward when win=1 vs win=0 (separation).
    for name in metrics:
        w1 = [r[name] for r in rows if r["official_win"] >= 0.5]
        w0 = [r[name] for r in rows if r["official_win"] < 0.5]
        out[f"mean_{name}_when_win"] = (sum(w1) / len(w1)) if w1 else None
        out[f"mean_{name}_when_loss"] = (sum(w0) / len(w0)) if w0 else None
    return out


def _run_label(path: str) -> str:
    parts = path.replace("\\", "/").split("/")
    for i, p in enumerate(parts):
        if p == "final_eval" and i + 1 < len(parts):
            return parts[i + 1]
        if p == "eval" and i >= 2:
            return f"{parts[i-2]}_{parts[i-1]}"
    return os.path.basename(os.path.dirname(path))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", default=_DEFAULT_GLOB,
                    help="Glob for *trajectories.jsonl files")
    ap.add_argument("--dataset", default=_DEFAULT_DATA,
                    help="Full NESTFUL dataset JSONL")
    ap.add_argument("--out", default=os.path.join(_ROOT, "..", "comparison",
                                                  "execution_reward_correlation.csv"),
                    help="Output CSV (all runs combined)")
    ap.add_argument("--summary-out",
                    default=os.path.join(_ROOT, "..", "comparison",
                                         "execution_reward_correlation_summary.csv"),
                    help="Per-run correlation summary CSV")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.glob, recursive=True))
    if not paths:
        print(f"No trajectories found for glob: {args.glob}", file=sys.stderr)
        return 1

    if not os.path.isfile(args.dataset):
        print(f"Dataset not found: {args.dataset}", file=sys.stderr)
        return 1

    print(f"[rescore] loading tasks from {args.dataset}", flush=True)
    tasks_by_id = {t["task_id"]: t for t in load_tasks(args.dataset, stage=None)}

    all_rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    for path in paths:
        label = _run_label(path)
        print(f"[rescore] {label} <- {path}", flush=True)
        rows, summary = score_trajectories_file(path, tasks_by_id, run_label=label)
        all_rows.extend(rows)
        summaries.append(summary)
        print(f"  n={summary['n']}  mean_win={summary.get('mean_official_win', 0):.3f}  "
              f"pearson(exec)={summary.get('pearson_execution_reward_vs_win')}  "
              f"pearson(partial)={summary.get('pearson_partial_reward_vs_win')}  "
              f"pearson(strict)={summary.get('pearson_strict_reward_vs_win')}", flush=True)

    # Combined summary across all runs.
    combined = _summarize(all_rows, run_label="ALL", path="combined", skipped=0)
    summaries.append(combined)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if all_rows:
        fields = list(all_rows[0].keys())
        with open(args.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(all_rows)
        print(f"[rescore] wrote {len(all_rows)} rows -> {args.out}", flush=True)

    sum_fields = sorted({k for s in summaries for k in s})
    with open(args.summary_out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=sum_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(summaries)
    print(f"[rescore] wrote summary -> {args.summary_out}", flush=True)

    print("\n=== Combined correlation with official Win (ALL runs) ===")
    for key in ("execution_reward", "partial_reward", "strict_reward", "final_answer_pass"):
        p = combined.get(f"pearson_{key}_vs_win")
        s = combined.get(f"spearman_{key}_vs_win")
        m1 = combined.get(f"mean_{key}_when_win")
        m0 = combined.get(f"mean_{key}_when_loss")
        if m1 is not None and m0 is not None:
            print(f"  {key:22s}  pearson={p!s:>8}  spearman={s!s:>8}  "
                  f"mean(win={m1:.3f} loss={m0:.3f})")
        else:
            print(f"  {key:22s}  pearson={p!s:>8}  spearman={s!s:>8}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
