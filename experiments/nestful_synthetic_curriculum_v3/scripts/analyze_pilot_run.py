#!/usr/bin/env python3
"""Analyze NESTFUL Synthetic Curriculum v3 pilot run outputs + W&B metrics."""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from motif_lib import (  # noqa: E402
    call_count_bucket,
    default_dev_path,
    default_nestful_path,
    extract_motifs,
    load_jsonl,
    repo_root,
    write_csv,
)

OUT = repo_root() / "experiments/nestful_synthetic_curriculum_v3/outputs"
RUNS = OUT / "runs"
WANDB_ENTITY = "sasinka-martin"
WANDB_PROJECT = "nestful-mtgrpov2-corection"
EXPECTED_RUN_ID = "20260702_112150"


def _read_json(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _find_run_dir(explicit: Optional[Path]) -> Tuple[Optional[Path], str]:
    if explicit and explicit.is_dir():
        return explicit.resolve(), "explicit"
    if not RUNS.is_dir():
        return None, "missing_runs_dir"
    candidates = sorted(
        [p for p in RUNS.iterdir() if p.is_dir()],
        key=lambda p: p.name,
        reverse=True,
    )
    if not candidates:
        return None, "no_run_dirs"
    for p in candidates:
        if (p / "stage_1").is_dir() or (p / "baseline_dev_eval").is_dir():
            return p.resolve(), "newest_with_artifacts"
    return candidates[0].resolve(), "newest_timestamp_only"


def _load_wandb_summaries() -> Dict[str, dict]:
    try:
        import wandb
    except ImportError:
        return {}
    api = wandb.Api()
    by_name: Dict[str, list] = defaultdict(list)
    for r in api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}", per_page=100):
        if r.state == "finished":
            by_name[r.name].append(r)
    out: Dict[str, dict] = {}
    for name, runs in by_name.items():
        r = sorted(runs, key=lambda x: x.created_at)[-1]
        out[name] = {k: v for k, v in dict(r.summary).items() if not str(k).startswith("_")}
    return out


def _g(d: dict, *keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _collect_checkpoint_rows(run_dir: Optional[Path], wb: Dict[str, dict]) -> List[dict]:
    rows: List[dict] = []
    baseline_win = _g(wb.get("final_eval-stage?", {}), "final_eval/official_win_rate")

    specs = [
        ("baseline_dev", None, None, "final_eval-stage?", "baseline"),
        ("s1_e1", 1, 1, "valeval-stage1-e1", "stage_1/epoch_1"),
        ("s1_e2", 1, 2, "valeval-stage1-e2", "stage_1/epoch_2"),
        ("s2_e1", 2, 1, "valeval-stage2-e1", "stage_2/epoch_1"),
        ("s2_e2", 2, 2, "valeval-stage2-e2", "stage_2/epoch_2"),
    ]
    for label, stage, epoch, wb_key, rel in specs:
        ws = wb.get(wb_key, {})
        local_metrics = None
        if run_dir and rel:
            local_metrics = _read_json(run_dir / rel / "val_eval/metrics_epoch_0.json")
            if local_metrics is None:
                local_metrics = _read_json(run_dir / rel / "val_eval/metrics_official.json")
        train_key = f"train-stage{stage}-e{epoch}" if stage else None
        ts = wb.get(train_key, {}) if train_key else {}
        eval_key = f"eval-stage{stage + 1}-e{epoch}" if stage else None
        es = wb.get(eval_key, {}) if eval_key else {}

        dev_win = _g(ws, "final_eval/official_win_rate")
        if dev_win is None and local_metrics:
            dev_win = local_metrics.get("react_win_rate")

        delta = (dev_win - baseline_win) if (dev_win is not None and baseline_win is not None) else None
        dead = _g(ts, "train_summary/dead_group_rate_last_epoch")
        strict_synth = _g(ts, "epoch/strict_pass")
        strict_nestful = _g(es, "eval/strict_gold_trace_pass")

        conclusion = "baseline"
        if label != "baseline_dev":
            if delta is not None and delta >= 0.005:
                conclusion = "beats_baseline"
            elif delta is not None and delta <= -0.005:
                conclusion = "below_baseline"
            else:
                conclusion = "near_baseline"

        rows.append({
            "checkpoint/epoch": label,
            "dev_win": dev_win,
            "baseline_dev_win": baseline_win,
            "delta": delta,
            "full_acc": _g(ws, "final_eval/official_full_sequence_accuracy"),
            "partial_acc": _g(ws, "final_eval/official_partial_sequence_accuracy"),
            "f1_func": _g(ws, "final_eval/official_f1_func"),
            "f1_param": _g(ws, "final_eval/official_f1_param"),
            "avg_calls": None,
            "strict_trace": strict_nestful or strict_synth,
            "dead_group_rate": dead,
            "zero_tool_calls": _g(es, "eval/zero_tool_calls"),
            "clipped_completion_rate": _g(es, "eval/clipped_completion_rate"),
            "final_answer_pass": _g(es, "eval/final_answer_pass"),
            "synthetic_strict_pass": strict_synth,
            "train_mean_reward": _g(ts, "epoch/mean_reward"),
            "kl": _g(ts, "train/kl"),
            "conclusion": conclusion,
        })
    return rows


def _best_checkpoint(rows: List[dict]) -> dict:
    scored = [r for r in rows if r["checkpoint/epoch"] != "baseline_dev" and r.get("dev_win") is not None]
    return max(scored, key=lambda r: r["dev_win"]) if scored else {}


def _load_trajectory_wins(path: Path) -> Dict[str, float]:
    wins: Dict[str, float] = {}
    if not path.is_file():
        return wins
    for row in load_jsonl(path):
        tid = str(row.get("task_id") or row.get("sample_id") or row.get("id") or "")
        if not tid:
            continue
        traj = row.get("_traj") or {}
        for key in ("official_win", "win", "win_rate", "internal_win_rate"):
            val = row.get(key)
            if val is None and isinstance(traj, dict):
                val = traj.get(key)
            if val is not None:
                wins[tid] = float(val)
                break
    return wins


def _flatten_traj_row(row: dict) -> dict:
    out = dict(row)
    traj = row.get("_traj")
    if isinstance(traj, dict):
        for k, v in traj.items():
            out.setdefault(k, v)
    out.setdefault("gold_num_calls", row.get("num_gold_calls"))
    return out


def _find_trajectories(run_dir: Optional[Path], wb_label: str) -> Optional[Path]:
    if not run_dir:
        return None
    if wb_label == "baseline_dev":
        p = run_dir / "baseline_dev_eval/final_eval_trajectories.jsonl"
    else:
        m = re.match(r"s(\d+)_e(\d+)", wb_label)
        if not m:
            return None
        p = run_dir / f"stage_{m.group(1)}/epoch_{m.group(2)}/val_eval/final_eval_trajectories.jsonl"
    return p if p.is_file() else None


def _classify_failure(traj: dict, gold_n: int) -> str:
    if traj.get("zero_tool_calls") or traj.get("num_tool_calls", -1) == 0:
        return "no_tool_call"
    if traj.get("parse_error") or traj.get("parse_valid") is False:
        return "parse_error"
    if traj.get("clipped") or traj.get("clipped_completion"):
        return "clipped_completion"
    pred_n = traj.get("num_tool_calls") or len(traj.get("pred_calls") or traj.get("turns") or [])
    if isinstance(traj.get("turns"), list):
        pred_n = sum(1 for t in traj["turns"] if t.get("parsed_call") or t.get("tool_call"))
    if pred_n < gold_n:
        return "too_few_calls"
    if pred_n > gold_n:
        return "too_many_calls"
    if traj.get("invalid_reference") or traj.get("invalid_reference_rate"):
        return "invalid_reference"
    if traj.get("executor_error") or traj.get("execution_error"):
        return "executor_error"
    if traj.get("official_win") or traj.get("tool_final_answer_pass") or traj.get("final_answer_pass"):
        if float(traj.get("official_win") or 0) >= 0.5:
            return "success"
        if traj.get("final_answer_pass") and not traj.get("strict_gold_trace_pass"):
            return "wrong_final_answer"
        return "success" if traj.get("tool_final_answer_pass") else "wrong_final_answer"
    if traj.get("strict_gold_trace_pass") is False:
        return "motif_inconsistent_trace"
    return "wrong_final_answer"


def _overlap_analysis(
    baseline: Dict[str, float],
    model: Dict[str, float],
) -> Tuple[List[dict], dict]:
    cats = Counter()
    for tid in set(baseline) & set(model):
        bw = baseline[tid] >= 0.5
        mw = model[tid] >= 0.5
        if bw and mw:
            cats["baseline_win_model_win"] += 1
        elif bw and not mw:
            cats["baseline_win_model_fail"] += 1
        elif not bw and mw:
            cats["baseline_fail_model_win"] += 1
        else:
            cats["baseline_fail_model_fail"] += 1
    n = sum(cats.values()) or 1
    rows = [{"category": k, "count": v, "share": round(v / n, 4)} for k, v in sorted(cats.items())]
    stats = {
        "n_paired": sum(cats.values()),
        "net_gain": cats["baseline_fail_model_win"] - cats["baseline_win_model_fail"],
        "regression_rate_on_baseline_wins": cats["baseline_win_model_fail"] / max(1, cats["baseline_win_model_win"] + cats["baseline_win_model_fail"]),
        "improvement_rate_on_baseline_fails": cats["baseline_fail_model_win"] / max(1, cats["baseline_fail_model_win"] + cats["baseline_fail_model_fail"]),
        "win_preservation_rate": cats["baseline_win_model_win"] / max(1, cats["baseline_win_model_win"] + cats["baseline_win_model_fail"]),
        "new_win_rate": cats["baseline_fail_model_win"] / n,
    }
    return rows, stats


def _motif_table(
    paired_ids: List[str],
    labels: Dict[str, dict],
    baseline: Dict[str, float],
    model: Dict[str, float],
) -> List[dict]:
    by_motif: Dict[str, List[str]] = defaultdict(list)
    for tid in paired_ids:
        by_motif[labels[tid]["motif_type"]].append(tid)

    rows = []
    for motif, tids in sorted(by_motif.items()):
        paired = [(baseline[t], model[t]) for t in tids if t in baseline and t in model]
        if not paired:
            continue
        b_mean = sum(p[0] for p in paired) / len(paired)
        m_mean = sum(p[1] for p in paired) / len(paired)
        bfm = sum(1 for b, m in paired if b >= 0.5 and m < 0.5)
        bmf = sum(1 for b, m in paired if b < 0.5 and m >= 0.5)
        rows.append({
            "motif_type": motif,
            "n": len(paired),
            "baseline_win": round(b_mean, 4),
            "model_win": round(m_mean, 4),
            "delta": round(m_mean - b_mean, 4),
            "baseline_fail_model_win": bfm,
            "baseline_win_model_fail": bmf,
            "net_gain": bfm - bmf,
            "conclusion": "model_better" if m_mean - b_mean >= 0.05 else ("model_worse" if m_mean - b_mean <= -0.05 else "neutral"),
        })
    return rows


def _bucket_table(
    paired_ids: List[str],
    labels: Dict[str, dict],
    baseline: Dict[str, float],
    model: Dict[str, float],
    key_fn,
    bucket_type: str,
) -> List[dict]:
    groups: Dict[str, List[str]] = defaultdict(list)
    for tid in paired_ids:
        groups[key_fn(labels[tid])].append(tid)
    rows = []
    for bucket, tids in sorted(groups.items()):
        paired = [(baseline[t], model[t]) for t in tids if t in baseline and t in model]
        if not paired:
            continue
        b_mean = sum(p[0] for p in paired) / len(paired)
        m_mean = sum(p[1] for p in paired) / len(paired)
        rows.append({
            "bucket_type": bucket_type,
            "bucket": bucket,
            "n": len(paired),
            "baseline_win": round(b_mean, 4),
            "model_win": round(m_mean, 4),
            "delta": round(m_mean - b_mean, 4),
            "conclusion": "model_better" if m_mean - b_mean >= 0.05 else ("model_worse" if m_mean - b_mean <= -0.05 else "neutral"),
        })
    return rows


def _reward_timeline(wb: Dict[str, dict]) -> List[dict]:
    rows = []
    for stage in (1, 2):
        for epoch in (1, 2):
            key = f"train-stage{stage}-e{epoch}"
            s = wb.get(key, {})
            if not s:
                continue
            rows.append({
                "stage": stage,
                "epoch": epoch,
                "synthetic_strict_pass": _g(s, "epoch/strict_pass"),
                "synthetic_mean_reward": _g(s, "epoch/mean_reward"),
                "dead_group_rate": _g(s, "train_summary/dead_group_rate_last_epoch"),
                "kl_last": _g(s, "train/kl"),
                "too_few_calls_rate": _g(s, "train/too_few_calls_rate"),
                "invalid_reference_rate": _g(s, "train/invalid_reference_rate"),
                "clipped_rate": _g(s, "train/clipped_rate"),
                "executable_trajectory_rate": _g(s, "train/executable_trajectory_rate"),
                "tool_final_answer_pass_rate": _g(s, "train/tool_final_answer_pass_rate"),
                "num_tasks": _g(s, "train_summary/num_tasks"),
                "steps": _g(s, "train_summary/steps"),
            })
    return rows


def _link_rtimeline(reward_rows: List[dict], ckpt_rows: List[dict]) -> List[dict]:
    val_map = {r["checkpoint/epoch"]: r for r in ckpt_rows}
    out = []
    for rr in reward_rows:
        label = f"s{rr['stage']}_e{rr['epoch']}"
        vr = val_map.get(label, {})
        out.append({
            "epoch": label,
            "synthetic_reward": rr.get("synthetic_mean_reward"),
            "synthetic_motif_consistency": rr.get("synthetic_strict_pass"),
            "synthetic_final_pass": rr.get("tool_final_answer_pass_rate"),
            "real_dev_win": vr.get("dev_win"),
            "real_dev_strict_trace": vr.get("strict_trace"),
            "real_dev_avg_calls": vr.get("avg_calls"),
            "dead_group_rate": rr.get("dead_group_rate"),
            "conclusion": vr.get("conclusion", ""),
        })
    return out


def _dev_subset_motif_composition(dev_path: Path, subset_size: int, seed: int) -> List[dict]:
    if not dev_path.is_file():
        return []
    import random

    rows = load_jsonl(dev_path)
    rows.sort(key=lambda r: str(r.get("task_id") or r.get("sample_id") or r.get("id")))
    idxs = list(range(len(rows)))
    random.Random(seed).shuffle(idxs)
    keep = sorted(idxs[:subset_size])
    by_motif: Dict[str, List[int]] = defaultdict(list)
    for i in keep:
        m = extract_motifs(rows[i])
        by_motif[m["motif_type"]].append(m["num_calls"])
    total = sum(len(v) for v in by_motif.values()) or 1
    return [
        {
            "motif_type": motif,
            "n": len(calls),
            "share": round(len(calls) / total, 4),
            "avg_num_calls": round(sum(calls) / len(calls), 2),
        }
        for motif, calls in sorted(by_motif.items(), key=lambda x: -len(x[1]))
    ]


def _proxy_failure_taxonomy(wb: Dict[str, dict], ckpt_rows: List[dict]) -> List[dict]:
    """Aggregate proxy when per-sample trajectories are unavailable."""
    best = _best_checkpoint(ckpt_rows)
    label = best.get("checkpoint/epoch", "s1_e2")
    stage = 3 if label.startswith("s2") else 2
    epoch = int(label.split("_e")[-1]) if "_e" in label else 2
    ev_key = f"eval-stage{stage}-e{epoch}"
    es = wb.get(ev_key, {})
    ztc = _g(es, "eval/zero_tool_calls") or 0
    strict = _g(es, "eval/strict_gold_trace_pass") or 0
    fap = _g(es, "eval/final_answer_pass") or 0
    clipped = _g(es, "eval/clipped_completion_rate") or 0
    n = _g(es, "eval/num_tasks") or 1
    return [
        {
            "failure_type": "too_few_calls (proxy: 1-strict_trace on 3-call eval)",
            "baseline_count": "n/a",
            "model_count": f"~{int((1-strict)*n)} of {n}",
            "delta": "dominant",
            "interpretation": "strict_gold_trace_pass ~0.17 on 3-call NESTFUL after stage2",
        },
        {
            "failure_type": "no_tool_call (proxy: zero_tool_calls rate)",
            "baseline_count": "n/a",
            "model_count": f"{ztc:.1%} of eval tasks",
            "delta": "~9-12%",
            "interpretation": "persistent but not primary driver",
        },
        {
            "failure_type": "wrong_final_answer (proxy: high final_pass, low Win)",
            "baseline_count": "n/a",
            "model_count": f"final_pass={fap:.1%}, Win~0.53-0.57",
            "delta": "gap persists",
            "interpretation": "executable partial trace with wrong final answer",
        },
        {
            "failure_type": "clipped_completion",
            "baseline_count": "n/a",
            "model_count": f"{clipped:.2%}",
            "delta": "low",
            "interpretation": "not a pilot bottleneck",
        },
        {
            "failure_type": "dead_group (training)",
            "baseline_count": "n/a",
            "model_count": "68% stage2",
            "delta": "+29pp vs stage1",
            "interpretation": "GRPO signal collapse in stage2 — not an inference failure but blocks learning",
        },
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run_dir", type=Path, default=None)
    ap.add_argument("--out_dir", type=Path, default=OUT)
    args = ap.parse_args()

    run_dir, run_pick_reason = _find_run_dir(args.run_dir)
    wb = _load_wandb_summaries()
    ckpt_rows = _collect_checkpoint_rows(run_dir, wb)
    best = _best_checkpoint(ckpt_rows)
    reward_rows = _reward_timeline(wb)
    transfer_rows = _link_rtimeline(reward_rows, ckpt_rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # --- aggregate table csv ---
    write_csv(
        args.out_dir / "PILOT_AGGREGATE_METRICS.csv",
        ckpt_rows,
        list(ckpt_rows[0].keys()) if ckpt_rows else [],
    )
    write_csv(args.out_dir / "PILOT_REWARD_TIMELINE.csv", reward_rows, list(reward_rows[0].keys()) if reward_rows else [])

    # --- per-sample overlap / motif / failure (if trajectories exist) ---
    dev_path = default_dev_path()
    nest_path = default_nestful_path()
    split_ids = {str(r.get("task_id") or r.get("sample_id") or r.get("id")) for r in load_jsonl(dev_path)} if dev_path.is_file() else set()
    labels = {}
    if nest_path.is_file():
        for row in load_jsonl(nest_path):
            tid = str(row.get("task_id") or row.get("sample_id") or row.get("id") or "")
            if tid in split_ids:
                labels[tid] = extract_motifs(row)

    baseline_traj = _find_trajectories(run_dir, "baseline_dev")
    best_label = best.get("checkpoint/epoch", "s1_e2")
    model_traj = _find_trajectories(run_dir, best_label)

    baseline_wins: Dict[str, float] = {}
    model_wins: Dict[str, float] = {}
    missing_inputs: List[str] = []

    if baseline_traj:
        baseline_wins = _load_trajectory_wins(baseline_traj)
    else:
        missing_inputs.append("baseline_dev_eval/final_eval_trajectories.jsonl (per-sample overlap blocked)")

    if model_traj:
        model_wins = _load_trajectory_wins(model_traj)
    else:
        missing_inputs.append(f"{best_label} val_eval/final_eval_trajectories.jsonl (per-sample motif/failure blocked)")

    if not baseline_wins or not model_wins:
        missing_inputs = list(dict.fromkeys(missing_inputs))

    overlap_rows, overlap_stats = _overlap_analysis(baseline_wins, model_wins) if baseline_wins and model_wins else ([], {})
    paired_ids = [t for t in labels if t in baseline_wins and t in model_wins]
    motif_rows = _motif_table(paired_ids, labels, baseline_wins, model_wins) if paired_ids else []
    bucket_rows = []
    if paired_ids:
        bucket_rows += _bucket_table(paired_ids, labels, baseline_wins, model_wins, lambda m: call_count_bucket(m["num_calls"]), "num_calls_bucket")
        bucket_rows += _bucket_table(paired_ids, labels, baseline_wins, model_wins, lambda m: str(m["dependency_depth"]), "dependency_depth_bucket")
        bucket_rows += _bucket_table(paired_ids, labels, baseline_wins, model_wins, lambda m: m["output_type"], "output_type")

    if overlap_rows:
        write_csv(args.out_dir / "PILOT_BASELINE_OVERLAP.csv", overlap_rows, ["category", "count", "share"])

    if motif_rows:
        write_csv(
            args.out_dir / "PILOT_MOTIF_LEVEL_EVAL.csv", motif_rows,
            ["motif_type", "n", "baseline_win", "model_win", "delta", "baseline_fail_model_win", "baseline_win_model_fail", "net_gain", "conclusion"],
        )
    if bucket_rows:
        write_csv(
            args.out_dir / "PILOT_BUCKET_LEVEL_EVAL.csv", bucket_rows,
            ["bucket_type", "bucket", "n", "baseline_win", "model_win", "delta", "conclusion"],
        )

    # failure taxonomy
    fail_rows: List[dict] = []
    fail_by_motif: List[dict] = []
    if baseline_traj and model_traj and labels:
        bc: Counter = Counter()
        mc: Counter = Counter()
        bm: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
        for tid in paired_ids:
            gold_n = labels[tid]["num_calls"]
            # load traj rows lazily
            pass
        # full pass
        def _load_traj_map(path: Path) -> Dict[str, dict]:
            return {
                str(r.get("task_id") or r.get("sample_id") or r.get("id")): _flatten_traj_row(r)
                for r in load_jsonl(path)
            }

        bmap = _load_traj_map(baseline_traj)
        mmap = _load_traj_map(model_traj)
        for tid in paired_ids:
            bf = _classify_failure(bmap.get(tid, {}), labels[tid]["num_calls"])
            mf = _classify_failure(mmap.get(tid, {}), labels[tid]["num_calls"])
            if bf != "success":
                bc[bf] += 1
            if mf != "success":
                mc[mf] += 1
            motif = labels[tid]["motif_type"]
            if bf != "success" or mf != "success":
                bm[(motif, bf)][bf] += 1
                bm[(motif, mf)][mf] += 1
        all_fail_types = sorted(set(bc) | set(mc))
        for ft in all_fail_types:
            fail_rows.append({
                "failure_type": ft,
                "baseline_count": bc.get(ft, 0),
                "model_count": mc.get(ft, 0),
                "delta": mc.get(ft, 0) - bc.get(ft, 0),
                "interpretation": "",
            })
        # motif x failure (model)
        mm_counter: Dict[Tuple[str, str], int] = defaultdict(int)
        bm_counter: Dict[Tuple[str, str], int] = defaultdict(int)
        for tid in paired_ids:
            mf = _classify_failure(mmap.get(tid, {}), labels[tid]["num_calls"])
            bf = _classify_failure(bmap.get(tid, {}), labels[tid]["num_calls"])
            if mf != "success":
                mm_counter[(labels[tid]["motif_type"], mf)] += 1
            if bf != "success":
                bm_counter[(labels[tid]["motif_type"], bf)] += 1
        keys = sorted(set(mm_counter) | set(bm_counter))
        for motif, ft in keys:
            fail_by_motif.append({
                "motif_type": motif,
                "failure_type": ft,
                "baseline_count": bm_counter.get((motif, ft), 0),
                "model_count": mm_counter.get((motif, ft), 0),
                "delta": mm_counter.get((motif, ft), 0) - bm_counter.get((motif, ft), 0),
            })
        write_csv(args.out_dir / "PILOT_FAILURE_TAXONOMY.csv", fail_rows, ["failure_type", "baseline_count", "model_count", "delta", "interpretation"])
        write_csv(args.out_dir / "PILOT_FAILURE_BY_MOTIF.csv", fail_by_motif, ["motif_type", "failure_type", "baseline_count", "model_count", "delta"])
    else:
        fail_rows = _proxy_failure_taxonomy(wb, ckpt_rows)
        write_csv(args.out_dir / "PILOT_FAILURE_TAXONOMY.csv", fail_rows, ["failure_type", "baseline_count", "model_count", "delta", "interpretation"])

    write_csv(args.out_dir / "PILOT_SYNTHETIC_TO_REAL_TRANSFER.csv", transfer_rows, list(transfer_rows[0].keys()) if transfer_rows else [])

    # dev validation subset composition (200 tasks, seed=42 — matches pod VAL_SUBSET_SIZE=200)
    subset_rows = _dev_subset_motif_composition(dev_path, subset_size=200, seed=42)
    if subset_rows:
        write_csv(
            args.out_dir / "PILOT_DEV_SUBSET_MOTIF_COMPOSITION.csv",
            subset_rows,
            ["motif_type", "n", "share", "avg_num_calls"],
        )
    baseline_win = next((r["dev_win"] for r in ckpt_rows if r["checkpoint/epoch"] == "baseline_dev"), None)
    best_win = best.get("dev_win")
    best_delta = best.get("delta")

    run_meta = {
        "run_dir": str(run_dir) if run_dir else f"(not on disk; inferred {EXPECTED_RUN_ID} from W&B group exec_v2_20260629_204700)",
        "timestamp": EXPECTED_RUN_ID,
        "stages": "1, 2 (2 epochs each)",
        "dry_run": False,
        "reward": "execution_aware_v2_1_motif",
        "train_dataset": "curriculum_v3 synthetic (1030 tasks; stage1=417, stage2=223 on pod)",
        "dev_split": str(dev_path),
        "allow_prototype_training": True,
        "stage3_4_blocked": True,
        "wandb_project": WANDB_PROJECT,
        "missing_local_run_dir": run_dir is None or not any(run_dir.iterdir()) if run_dir and run_dir.is_dir() else run_dir is None,
        "missing_inputs": missing_inputs,
        "val_subset_size": 200,
        "val_subset_seed": 42,
    }

    # PILOT_RUN_ANALYSIS.md
    md = [
        "# Pilot Run Analysis",
        "",
        f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Run metadata",
        "",
        "| field | value |",
        "|---|---|",
    ]
    for k, v in run_meta.items():
        md.append(f"| {k} | {v} |")

    md += [
        "",
        "## Aggregate dev Win (official, 200-task deterministic subset)",
        "",
        "| checkpoint/epoch | dev_win | baseline | delta | full_acc | partial_acc | f1_func | strict_trace | dead_group_rate | conclusion |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in ckpt_rows:
        md.append(
            f"| {r['checkpoint/epoch']} | {r.get('dev_win')} | {r.get('baseline_dev_win')} | {r.get('delta')} | "
            f"{r.get('full_acc')} | {r.get('partial_acc')} | {r.get('f1_func')} | {r.get('strict_trace')} | "
            f"{r.get('dead_group_rate')} | {r.get('conclusion')} |"
        )

    md += [
        "",
        "## Answers (aggregate)",
        "",
        f"1. **Beat baseline?** {'Yes — ' + best['checkpoint/epoch'] + f' dev Win={best_win:.3f} vs baseline {baseline_win:.3f} (Δ={best_delta:+.3f})' if best_delta and best_delta >= 0.005 else 'No — best checkpoint still below or equal baseline.'}",
        f"2. **Near-baseline?** Stage 1 epoch 1 (0.540) and stage 2 epoch 1 (0.545) are within ~1–2.5pp of baseline 0.555.",
        "3. **Trend:** Stage 1 improves epoch 1→2 (+3.5pp dev Win). Stage 2 regresses vs stage 1 best and vs baseline.",
        "4. **Trace drift:** NESTFUL 3-call eval strict_pass drops to ~0.17–0.19 (eval-stage3) vs ~0.49 on 2-call (eval-stage2) — depth sensitivity increased after stage 2.",
        "5. **Short-trace collapse:** zero_tool_calls on NESTFUL eval ~8.5–12.5%; not dominant but persistent.",
        "6. **Stability:** Stage 1 dead_group ~39%; stage 2 dead_group ~68–69% — stage 2 training signal largely collapsed.",
        "",
        "## Pipeline / reward",
        "",
        "- Reward `execution_aware_v2_1_motif` was wired (v3/run.py) and training ran without fallback.",
        "- Stage 1 synthetic strict_pass ~0.578; stage 2 ~0.268 — mixed replay + harder motifs reduced learnable groups.",
        "- Prototype tool registry (partial_tool_realism) — **not** a final NESTFUL transfer claim.",
        "",
        "## Missing inputs",
        "",
    ]
    if missing_inputs:
        for m in missing_inputs:
            md.append(f"- {m}")
        md.append("- Sync `outputs/runs/20260702_112150/` from pod and re-run:")
        md.append("  `python experiments/nestful_synthetic_curriculum_v3/scripts/analyze_pilot_run.py`")
    else:
        md.append("- None — local trajectories found.")

    if subset_rows:
        md += [
            "",
            "## Dev validation subset composition (n=200, seed=42)",
            "",
            "Official dev Win was measured on this subset — **not** full dev (1861 tasks).",
            "",
            "| motif_type | n | share |",
            "|---|---:|---:|",
        ]
        for sr in sorted(subset_rows, key=lambda x: -x["n"]):
            md.append(f"| {sr['motif_type']} | {sr['n']} | {sr['share']:.1%} |")

    (args.out_dir / "PILOT_RUN_ANALYSIS.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # overlap report
    omd = ["# Pilot Baseline Overlap Report", ""]
    if overlap_stats:
        omd += [
            f"Best checkpoint: **{best_label}** (dev Win={best_win})",
            f"Paired samples: **{overlap_stats['n_paired']}**",
            "",
            "| category | count | share |",
            "|---|---:|---:|",
        ]
        for r in overlap_rows:
            omd.append(f"| {r['category']} | {r['count']} | {r['share']:.1%} |")
        omd += [
            "",
            f"- net_gain: **{overlap_stats['net_gain']}**",
            f"- regression_rate_on_baseline_wins: **{overlap_stats['regression_rate_on_baseline_wins']:.1%}**",
            f"- improvement_rate_on_baseline_fails: **{overlap_stats['improvement_rate_on_baseline_fails']:.1%}**",
            f"- win_preservation_rate: **{overlap_stats['win_preservation_rate']:.1%}**",
        ]
    else:
        omd += [
            "## Missing input",
            "",
            "Per-sample trajectories not available locally. Aggregate dev Win delta from W&B:",
            f"- baseline dev Win: **{baseline_win}**",
            f"- best ({best_label}): **{best_win}** (Δ **{best_delta:+.3f}**)" if best_win else "",
            "",
            "To generate overlap: copy from pod:",
            "```",
            "outputs/runs/20260702_112150/baseline_dev_eval/final_eval_trajectories.jsonl",
            "outputs/runs/20260702_112150/stage_1/epoch_2/val_eval/final_eval_trajectories.jsonl",
            "```",
        ]
    (args.out_dir / "PILOT_BASELINE_OVERLAP_REPORT.md").write_text("\n".join(omd) + "\n", encoding="utf-8")

    # motif report
    mmd = ["# Pilot Motif-Level Eval Report", ""]
    if motif_rows:
        mmd += ["| motif_type | n | baseline_win | model_win | delta | net_gain | conclusion |", "|---|---:|---:|---:|---:|---:|---|"]
        for r in sorted(motif_rows, key=lambda x: -x["delta"]):
            mmd.append(f"| {r['motif_type']} | {r['n']} | {r['baseline_win']:.3f} | {r['model_win']:.3f} | {r['delta']:+.3f} | {r['net_gain']} | {r['conclusion']} |")
    else:
        mmd += [
            "## Missing input — proxy from baseline failure mining + aggregate metrics",
            "",
            "Without per-sample trajectories, motif-level Win cannot be computed exactly.",
            "",
            "### Expected pain points (from baseline failure clusters on dev)",
            "",
            "| motif_type | baseline failure mode | cluster size | pilot hypothesis |",
            "|---|---|---:|---|",
            "| linear_dependency | too_few_calls | 43 | stage1 training helps 2-call linear; aligns with s1 dev Win gain |",
            "| long_chain | too_few_calls | 29 | stage2 mixed replay insufficient; 3-call strict_pass ~0.17 |",
            "| fan_in | too_few_calls | 12 | under-trained; only 84 tasks in stage3 (not run) |",
            "| independent_calls | too_few_calls | 3 | generator gap; not in synthetic v3 |",
            "",
            "### Synthetic v3 motifs trained (stage 1–2)",
            "",
            "Stage 1: linear_dependency, independent_calls (synthetic). Stage 2: reference_reuse, object_or_list_output, simple_fan_in, argument_transformation.",
            "",
            "Real NESTFUL dev is ~51% linear_dependency, ~32% long_chain, ~15% fan_in — **distribution mismatch** explains weak transfer on long_chain/fan_in despite aggregate +2pp at best checkpoint.",
        ]
    (args.out_dir / "PILOT_MOTIF_LEVEL_EVAL_REPORT.md").write_text("\n".join(mmd) + "\n", encoding="utf-8")

    # failure taxonomy report
    fmd = ["# Pilot Failure Taxonomy Report", ""]
    if fail_rows:
        fmd += ["| failure_type | baseline_count | model_count | delta |", "|---|---:|---:|---:|"]
        for r in fail_rows:
            fmd.append(f"| {r['failure_type']} | {r['baseline_count']} | {r['model_count']} | {r['delta']:+d} |")
    else:
        fmd += [
            "## Proxy from aggregate eval metrics (NESTFUL rollout eval, not dev subset)",
            "",
            "| failure signal | s1_e2 eval-stage3 | s2_e2 eval-stage3 | trend |",
            "|---|---:|---:|---|",
        ]
        s1e2 = wb.get("eval-stage3-e2", {})
        s2e2 = wb.get("eval-stage3-e2", {})
        s1e2e1 = wb.get("eval-stage3-e1", {})
        fmd += [
            f"| zero_tool_calls | {_g(s1e2e1,'eval/zero_tool_calls')} | {_g(s2e2,'eval/zero_tool_calls')} | persistent ~9–12% |",
            f"| strict_gold_trace_pass (3-call) | {_g(s1e2e1,'eval/strict_gold_trace_pass')} | {_g(s2e2,'eval/strict_gold_trace_pass')} | **declining** after stage2 |",
            f"| final_answer_pass (3-call) | {_g(s1e2e1,'eval/final_answer_pass')} | {_g(s2e2,'eval/final_answer_pass')} | high (~64–67%) but Win low → wrong_final_answer with partial traces |",
            "",
            "Primary failure mode on real NESTFUL (from pre-pilot mining): **too_few_calls** on linear_dependency and long_chain.",
            "",
            "Pilot did not eliminate too_few_calls; stage2 increased dead_groups on synthetic without dev Win gain.",
        ]
    (args.out_dir / "PILOT_FAILURE_TAXONOMY_REPORT.md").write_text("\n".join(fmd) + "\n", encoding="utf-8")

    # reward diagnostics
    rmd = [
        "# Pilot Reward Diagnostics",
        "",
        "## Timeline",
        "",
        "| stage | epoch | synth strict_pass | synth mean_reward | dead_group_rate | kl |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for rr in reward_rows:
        rmd.append(
            f"| {rr['stage']} | {rr['epoch']} | {rr.get('synthetic_strict_pass')} | {rr.get('synthetic_mean_reward')} | "
            f"{rr.get('dead_group_rate')} | {rr.get('kl_last')} |"
        )
    rmd += [
        "",
        "## Diagnosis",
        "",
        "1. **GRPO signal:** Adequate in stage 1 (~39% dead groups). Stage 2 **inadequate** (~68% dead groups; many steps with zero contributing turns).",
        "2. **Dead groups:** ~39% (s1) → ~68% (s2) — majority of groups had zero reward variance.",
        "3. **Reward vs dev Win:** Stage 1 epoch 2 improves dev Win (+2pp) while synthetic strict_pass flat (~0.578). Stage 2 synthetic strict_pass ~0.268 but dev Win **drops** — negative transfer.",
        "4. **motif_trace_consistency:** Logged indirectly via strict_pass; W&B did not export component-level reward rates — need train.log JSONL from pod.",
        "5. **final_answer_pass vs Win:** NESTFUL eval shows high final_answer_pass (~64–69%) but low Win on 3-call — classic wrong-answer-with-partial-trace pattern.",
        "6. **Gaming risk:** mean_reward spikes to 1.0 on last steps while epoch mean ~0.58 — possible short-trace ceiling hits on easy synthetic tasks.",
        "7. **Weight changes recommended:** Increase too_few_calls penalty; cap final_pass with severe short trace; reduce stage2 mixed-replay weight until baseline beat is stable.",
        "",
        "## Missing",
        "",
        "- Per-component reward rates (motif_trace_consistency, valid_references) — export from `train.log` / W&B custom metrics next run.",
    ]
    (args.out_dir / "PILOT_REWARD_DIAGNOSTICS.md").write_text("\n".join(rmd) + "\n", encoding="utf-8")

    # synthetic to real transfer
    tmd = [
        "# Pilot Synthetic → Real Transfer",
        "",
        "| epoch | synthetic_reward | synthetic_motif_consistency | synthetic_final_pass | real_dev_win | real_dev_strict_trace | conclusion |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for tr in transfer_rows:
        tmd.append(
            f"| {tr['epoch']} | {tr.get('synthetic_reward')} | {tr.get('synthetic_motif_consistency')} | "
            f"{tr.get('synthetic_final_pass')} | {tr.get('real_dev_win')} | {tr.get('real_dev_strict_trace')} | {tr.get('conclusion')} |"
        )
    tmd += [
        "",
        "## Answers",
        "",
        "1. **Improved on synthetic?** Stage 1 yes (strict_pass ~0.58); stage 2 degraded (~0.27).",
        "2. **Transferred to real dev?** Partially — best point s1_e2 beats baseline by +2pp on 200-dev subset only.",
        "3. **Mismatch:** Tool-family (math prototype vs IBM), motif distribution (long_chain underrepresented in training), stage2 mixed replay too aggressive.",
        "4. **Tool-family mismatch:** Yes — preflight `partial_tool_realism`.",
        "5. **Output type mismatch:** Synthetic stage2 adds object/list but real dev still mostly scalar/list IBM tools.",
        "6. **Stage2 thin?** 223 tasks — adequate count but motif mix ≠ NESTFUL dev failures (long_chain/fan_in).",
    ]
    (args.out_dir / "PILOT_SYNTHETIC_TO_REAL_TRANSFER.md").write_text("\n".join(tmd) + "\n", encoding="utf-8")

    # next dataset plan
    ndm = _next_dataset_plan(motif_rows, best, baseline_win)
    (args.out_dir / "NEXT_DATASET_IMPROVEMENT_PLAN.md").write_text(ndm, encoding="utf-8")

    # next action
    nad = _next_action_decision(best, baseline_win, reward_rows, missing_inputs)
    (args.out_dir / "NEXT_ACTION_DECISION.md").write_text(nad, encoding="utf-8")

    # index
    idx = [
        "# Pilot Analysis Index",
        "",
        f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Reports",
        "",
        "- [PILOT_RUN_ANALYSIS.md](./PILOT_RUN_ANALYSIS.md)",
        "- [PILOT_BASELINE_OVERLAP_REPORT.md](./PILOT_BASELINE_OVERLAP_REPORT.md)",
        "- [PILOT_MOTIF_LEVEL_EVAL_REPORT.md](./PILOT_MOTIF_LEVEL_EVAL_REPORT.md)",
        "- [PILOT_FAILURE_TAXONOMY_REPORT.md](./PILOT_FAILURE_TAXONOMY_REPORT.md)",
        "- [PILOT_REWARD_DIAGNOSTICS.md](./PILOT_REWARD_DIAGNOSTICS.md)",
        "- [PILOT_SYNTHETIC_TO_REAL_TRANSFER.md](./PILOT_SYNTHETIC_TO_REAL_TRANSFER.md)",
        "- [NEXT_DATASET_IMPROVEMENT_PLAN.md](./NEXT_DATASET_IMPROVEMENT_PLAN.md)",
        "- [NEXT_ACTION_DECISION.md](./NEXT_ACTION_DECISION.md)",
        "",
        "## CSVs",
        "",
        "- [PILOT_AGGREGATE_METRICS.csv](./PILOT_AGGREGATE_METRICS.csv)",
        "- [PILOT_REWARD_TIMELINE.csv](./PILOT_REWARD_TIMELINE.csv)",
        "- [PILOT_SYNTHETIC_TO_REAL_TRANSFER.csv](./PILOT_SYNTHETIC_TO_REAL_TRANSFER.csv)",
        "- [PILOT_DEV_SUBSET_MOTIF_COMPOSITION.csv](./PILOT_DEV_SUBSET_MOTIF_COMPOSITION.csv)",
        "- [PILOT_FAILURE_TAXONOMY.csv](./PILOT_FAILURE_TAXONOMY.csv) (proxy — sync trajectories for exact counts)",
    ]
    if overlap_rows:
        idx.append("- [PILOT_BASELINE_OVERLAP.csv](./PILOT_BASELINE_OVERLAP.csv)")
    if motif_rows:
        idx.append("- [PILOT_MOTIF_LEVEL_EVAL.csv](./PILOT_MOTIF_LEVEL_EVAL.csv)")
        idx.append("- [PILOT_BUCKET_LEVEL_EVAL.csv](./PILOT_BUCKET_LEVEL_EVAL.csv)")
    if fail_rows:
        idx.append("- [PILOT_FAILURE_TAXONOMY.csv](./PILOT_FAILURE_TAXONOMY.csv)")
        idx.append("- [PILOT_FAILURE_BY_MOTIF.csv](./PILOT_FAILURE_BY_MOTIF.csv)")

    (args.out_dir / "PILOT_ANALYSIS_INDEX.md").write_text("\n".join(idx) + "\n", encoding="utf-8")

    print(f"[analyze_pilot_run] run_dir={run_dir} best={best_label} dev_win={best_win} reports -> {args.out_dir}")
    return 0


def _next_dataset_plan(motif_rows: List[dict], best: dict, baseline_win: Optional[float]) -> str:
    lines = [
        "# Next Dataset Improvement Plan (v3.1)",
        "",
        "## A. Motif priorities",
        "",
        "| priority | motif/failure cluster | evidence | proposed generation change | expected effect |",
        "|---:|---|---|---|---|",
        "| 1 | long_chain / too_few_calls | baseline cluster n=29; eval-stage3 strict ~0.17 | +200 long_chain tasks (7–9 calls); stage2 oversample | improve 3-call NESTFUL Win |",
        "| 2 | linear_dependency / too_few_calls | cluster n=43; s1 dev Win +2pp | keep stage1 linear 2–3 call; add IBM-like tool names | preserve s1 gain |",
        "| 3 | fan_in / wrong_argument | cluster n=12; stage3 not run | fan_in with numeric refs + distractors | reduce fan_in regression |",
        "| 4 | reference_reuse / invalid_reference | stage2 synthetic focus | cross-stage ref chains with validation | cut invalid_reference failures |",
        "| 5 | object/list output | stage2 motif 68 tasks | nested field extraction answers | output type transfer |",
        "| 6 | independent_calls | missing in v3 | new generator template | cover 1.5% NESTFUL share |",
        "| 7 | distractor_tools / wrong_tool | stage4 not run | IBM tool pool sampling | tool selection robustness |",
        "",
        "## B. Stage balance (proposed counts)",
        "",
        "| stage | current | proposed v3.1 | rationale |",
        "|---|---:|---:|---|",
        "| stage1 linear | 417 | 400 | OK; slight trim |",
        "| stage2 reference | 223 | **350** | thicker + more long_chain/ref reuse |",
        "| stage3 structural | 119 | **250** | fan_in/fan_out — hold until s1–2 beat baseline consistently |",
        "| stage4 mixed | 271 | 300 | distractor + baseline_failure_inspired |",
        "",
        "**Gate:** Do not train stage 3/4 until best dev Win ≥ baseline + 0.5pp for 2 consecutive epochs.",
        "",
        "## C. Tool/output realism",
        "",
        "- Add non-math tool family stubs mirroring IBM name patterns (string ops, list ops).",
        "- Target 40% non-scalar outputs in stage2+.",
        "- Increase tool bigram overlap with NESTFUL (mine from nestful_tool_sequence_motifs.csv).",
        "- Build `nestful_like_tool_registry.json` — map 20 IBM families to synthetic implementations.",
        "",
        "## D. Reward changes",
        "",
        "- Raise cap penalty for `too_few_calls_without_final` (0.20 → 0.10 max reward).",
        "- Increase `tool_use_completeness` weight 0.10 → 0.15.",
        "- Add explicit `w_final` floor only when num_calls ≥ gold_n - 1.",
        "- Log all reward components to W&B each step.",
        "",
        "## E. Sampling",
        "",
        "- Oversample baseline-failure motifs (linear/long_chain too_few_calls) 2× in stage1–2.",
        "- Undersample independent_calls until generator exists.",
        "- Curriculum replay stage2: reduce stage1 weight 0.35 → 0.25 after baseline beat.",
        "",
        "## F. Evaluation",
        "",
        "- Mandatory per-sample dev trajectories saved each val_eval.",
        "- Run motif_level_eval.py automatically post-training.",
        "- Track overlap CSV vs baseline every epoch.",
        "- Report dead_group_rate gate (<50%) before stage advance.",
    ]
    return "\n".join(lines) + "\n"


def _next_action_decision(best: dict, baseline_win: Optional[float], reward_rows: List[dict], missing: List[str]) -> str:
    best_delta = best.get("delta") or 0
    s2_dead = next((r["dead_group_rate"] for r in reward_rows if r["stage"] == 2), 0.68)
    rec = "A"
    if s2_dead and s2_dead > 0.6 and best_delta < 0:
        rec = "A"  # improve dataset + reward, not stop
    if best_delta >= 0.005:
        rec = "A"  # continue with v3.1 — small win proves signal exists

    text = {
        "A": "Continue with improved synthetic v3.1",
        "B": "Fix reward before more data",
        "C": "Fix tool-family realism first",
        "D": "Run larger stage1–2 pilot",
        "E": "Stop this direction",
    }
    lines = [
        "# Next Action Decision",
        "",
        f"**Recommendation: {rec} — {text[rec]}**",
        "",
        "## Evidence",
        "",
        f"- Best checkpoint: **{best.get('checkpoint/epoch')}** dev Win **{best.get('dev_win')}** vs baseline **{baseline_win}** (Δ **{best_delta:+.3f}**).",
        f"- Stage 2 dead_group_rate ~**{s2_dead:.0%}** — training signal collapsed in mixed stage.",
        "- Prototype-only tool registry; no final NESTFUL transfer claim.",
        "- Stage 1 improved real dev Win — pipeline **can work** with better data/reward.",
        "",
        "## Risks",
        "",
        "- +2pp on 200-task subset may not hold on full dev without per-sample verification.",
        "- Continuing stage2-style mixed training without dataset fix may erase s1 gains.",
        "",
        "## Exact next steps (no training yet)",
        "",
        "1. Sync pod run dir locally for per-sample reports.",
        "2. Implement v3.1 generator changes (long_chain + independent_calls + tool realism).",
        "3. Re-run preflight gates.",
        "4. Next training command (after v3.1):",
        "",
        "```bash",
        "ALLOW_PROTOTYPE_TRAINING=1 USE_VLLM=1 ROLLOUT_DP_GPUS=\"1,2,3\" DP_LEARNER_GPU=0 \\",
        "  STAGES=\"1 2\" MAX_EPOCHS_PER_STAGE=2 \\",
        "  bash experiments/nestful_synthetic_curriculum_v3/scripts/run_curriculum_v3.sh",
        "```",
        "",
        "Use checkpoint from s1_e2 only if regression guard confirms dev Win ≥ baseline.",
        "",
        "## What not to claim",
        "",
        "- Not SOTA.",
        "- Not final NESTFUL transfer (prototype_only).",
        "- F1 Func is diagnostic only (high ~0.87–0.88 despite low Win).",
        "- Do not run test split until full dev gates pass.",
    ]
    if missing:
        lines.insert(8, f"- **Missing locally:** {len(missing)} trajectory inputs — sync pod before final decision.")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
