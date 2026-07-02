#!/usr/bin/env python3
"""Download W&B curriculum runs and analyze GRPO learning signal."""
from __future__ import annotations

import json
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "curricullum/evaluation/wandb_analysis"

ENTITY = "sasinka-martin"
PROJECT = "nestful-curriculum-toolr0"

# Main curriculum + v2 cloud runs (skip failed/profiling/dgx unless needed)
RUN_NAME_PREFIXES = (
    "curriculum-20260612-1530-stage_",
    "qwen3-4b-curriculum-v2-cloud-stage_",
)

TAG_RE = re.compile(r"<tool_call_answer>", re.I)


@dataclass
class RunAnalysis:
    run_id: str
    run_name: str
    state: str
    stage: Optional[int]
    epoch: Optional[int]
    history_steps: int
    reward_mean: Optional[float]
    reward_final: Optional[float]
    frac_zero_std_mean: Optional[float]
    frac_zero_std_final: Optional[float]
    frac_zero_std_max: Optional[float]
    reward_std_mean: Optional[float]
    clipped_ratio_mean: Optional[float]
    completion_tables: int
    completion_rows: int
    rows_advantage_zero: int
    rows_advantage_zero_pct: Optional[float]
    rows_reward_zero: int
    rows_reward_one: int
    steps_all_adv_zero: int
    steps_with_signal: int
    steps_dead_pct: Optional[float]
    format_no_tag_pct: Optional[float]
    format_has_tag_pct: Optional[float]
    curriculum_exec_pass: Optional[float]
    curriculum_parse_fail: Optional[float]


def _parse_stage_epoch(name: str) -> Tuple[Optional[int], Optional[int]]:
    m = re.search(r"stage_(\d+)-e(\d+)", name)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def _completion_format_flags(text: str) -> Dict[str, bool]:
    t = text or ""
    low = t.lower()
    has_tag = bool(TAG_RE.search(t))
    has_think = "redacted_thinking" in low
    return {
        "no_tag": not has_tag,
        "has_tag": has_tag,
        "has_think": has_think,
    }


def _select_table_files(files: List[Any], max_files: int = 40) -> List[Any]:
    if len(files) <= max_files:
        return files
    step_nums = []
    for f in files:
        m = re.search(r"completions_(\d+)_", f.name)
        step_nums.append((int(m.group(1)) if m else 0, f))
    step_nums.sort(key=lambda x: x[0])
    if len(step_nums) <= max_files:
        return [f for _, f in step_nums]
    idxs = [round(i * (len(step_nums) - 1) / (max_files - 1)) for i in range(max_files)]
    return [step_nums[i][1] for i in idxs]


def _analyze_completion_tables(run) -> Dict[str, Any]:
    table_files = [
        f for f in run.files() if "/completions_" in f.name and f.name.endswith(".table.json")
    ]
    table_files = _select_table_files(table_files)
    rows_by_step: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    total_rows = 0
    adv_zero = 0
    reward_zero = 0
    reward_one = 0
    format_no_tag = 0
    format_has_tag = 0

    for wf in table_files:
        try:
            with wf.download(replace=True) as path:
                data = json.loads(Path(path.name).read_text(encoding="utf-8"))
        except Exception:
            continue
        cols = data.get("columns") or []
        if not cols:
            continue
        idx = {c: i for i, c in enumerate(cols)}
        for row in data.get("data") or []:
            total_rows += 1
            step = int(row[idx["step"]]) if "step" in idx else -1
            reward = float(row[idx["toolr0_reward_func"]]) if "toolr0_reward_func" in idx else 0.0
            adv = float(row[idx["advantage"]]) if "advantage" in idx else 0.0
            completion = row[idx["completion"]] if "completion" in idx else ""
            if abs(adv) < 1e-9:
                adv_zero += 1
            if reward <= 1e-9:
                reward_zero += 1
            if reward >= 0.999:
                reward_one += 1
            flags = _completion_format_flags(str(completion))
            if flags["no_tag"]:
                format_no_tag += 1
            if flags["has_tag"]:
                format_has_tag += 1
            rows_by_step[step].append({"reward": reward, "advantage": adv, **flags})

    steps_all_adv_zero = 0
    steps_with_signal = 0
    for step_rows in rows_by_step.values():
        if not step_rows:
            continue
        rewards = [r["reward"] for r in step_rows]
        advs = [r["advantage"] for r in step_rows]
        if all(abs(a) < 1e-9 for a in advs):
            steps_all_adv_zero += 1
        if len(rewards) >= 2 and statistics.pstdev(rewards) > 1e-6:
            steps_with_signal += 1

    n_steps = len(rows_by_step)
    return {
        "completion_tables": len(table_files),
        "completion_rows": total_rows,
        "rows_advantage_zero": adv_zero,
        "rows_advantage_zero_pct": (100 * adv_zero / total_rows) if total_rows else None,
        "rows_reward_zero": reward_zero,
        "rows_reward_one": reward_one,
        "steps_all_adv_zero": steps_all_adv_zero,
        "steps_with_signal": steps_with_signal,
        "steps_dead_pct": (100 * steps_all_adv_zero / n_steps) if n_steps else None,
        "format_no_tag_pct": (100 * format_no_tag / total_rows) if total_rows else None,
        "format_has_tag_pct": (100 * format_has_tag / total_rows) if total_rows else None,
    }


def _history_stats(run) -> Dict[str, Any]:
    keys = [
        "train/reward",
        "train/reward_std",
        "train/frac_reward_zero_std",
        "train/completions/clipped_ratio",
        "train/rewards/toolr0_reward_func/mean",
    ]
    rows: List[Dict[str, Any]] = []
    for row in run.scan_history(keys=keys, page_size=500):
        rows.append(row)
    if not rows:
        return {"history_steps": 0}

    df = pd.DataFrame(rows)
    out: Dict[str, Any] = {"history_steps": len(df)}

    def _series(col: str) -> Optional[pd.Series]:
        if col not in df.columns:
            return None
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        return s if len(s) else None

    for col, key_mean, key_final, key_max in [
        ("train/reward", "reward_mean", "reward_final", None),
        ("train/frac_reward_zero_std", "frac_zero_std_mean", "frac_zero_std_final", "frac_zero_std_max"),
        ("train/reward_std", "reward_std_mean", None, None),
        ("train/completions/clipped_ratio", "clipped_ratio_mean", None, None),
    ]:
        s = _series(col)
        if s is None:
            continue
        if key_mean:
            out[key_mean] = float(s.mean())
        if key_final:
            out[key_final] = float(s.iloc[-1])
        if key_max:
            out[key_max] = float(s.max())

    return out


def _local_val_metrics(stage: int, epoch: int) -> Dict[str, Optional[float]]:
    path = REPO / f"curricullum/training/results/stage_{stage}_epoch{epoch}_val.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "curriculum_exec_pass": data.get("exec_pass_rate"),
        "curriculum_parse_fail": data.get("parse_fail_rate"),
    }


def analyze_run(run) -> RunAnalysis:
    stage, epoch = _parse_stage_epoch(run.name)
    hist = _history_stats(run)
    if stage is not None and epoch is not None:
        hist.update(_local_val_metrics(stage, epoch))
    comp = _analyze_completion_tables(run)
    return RunAnalysis(
        run_id=run.id,
        run_name=run.name,
        state=run.state,
        stage=stage,
        epoch=epoch,
        history_steps=hist.get("history_steps", 0),
        reward_mean=hist.get("reward_mean"),
        reward_final=hist.get("reward_final"),
        frac_zero_std_mean=hist.get("frac_zero_std_mean"),
        frac_zero_std_final=hist.get("frac_zero_std_final"),
        frac_zero_std_max=hist.get("frac_zero_std_max"),
        reward_std_mean=hist.get("reward_std_mean"),
        clipped_ratio_mean=hist.get("clipped_ratio_mean"),
        completion_tables=comp.get("completion_tables", 0),
        completion_rows=comp.get("completion_rows", 0),
        rows_advantage_zero=comp.get("rows_advantage_zero", 0),
        rows_advantage_zero_pct=comp.get("rows_advantage_zero_pct"),
        rows_reward_zero=comp.get("rows_reward_zero", 0),
        rows_reward_one=comp.get("rows_reward_one", 0),
        steps_all_adv_zero=comp.get("steps_all_adv_zero", 0),
        steps_with_signal=comp.get("steps_with_signal", 0),
        steps_dead_pct=comp.get("steps_dead_pct"),
        format_no_tag_pct=comp.get("format_no_tag_pct"),
        format_has_tag_pct=comp.get("format_has_tag_pct"),
        curriculum_exec_pass=hist.get("curriculum_exec_pass"),
        curriculum_parse_fail=hist.get("curriculum_parse_fail"),
    )


def main() -> None:
    import wandb

    api = wandb.Api(timeout=120)
    runs = list(api.runs(f"{ENTITY}/{PROJECT}", per_page=100))
    selected = [
        r
        for r in runs
        if r.name.startswith(RUN_NAME_PREFIXES) and r.state in ("finished", "running", "crashed")
    ]
    selected.sort(key=lambda r: (r.name,))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUT_DIR / "progress.log"
    results: List[RunAnalysis] = []
    print(f"Analyzing {len(selected)} runs...", flush=True)
    with log_path.open("w", encoding="utf-8") as logf:
        logf.write(f"runs={len(selected)}\n")
        for i, run in enumerate(selected, 1):
            msg = f"[{i}/{len(selected)}] {run.name} ({run.id})"
            print(msg, flush=True)
            logf.write(msg + "\n")
            logf.flush()
            try:
                ra = analyze_run(run)
                results.append(ra)
                # incremental save
                (OUT_DIR / "wandb_grpo_signal_summary.partial.json").write_text(
                    json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8"
                )
            except Exception as exc:
                err = f"  FAILED: {exc}"
                print(err, flush=True)
                logf.write(err + "\n")
                logf.flush()

    payload = [asdict(r) for r in results]
    out_path = OUT_DIR / "wandb_grpo_signal_summary.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Stage rollup: worst epoch per stage by frac_zero_std_mean
    by_stage: Dict[int, List[RunAnalysis]] = defaultdict(list)
    for r in results:
        if r.stage is not None:
            by_stage[r.stage].append(r)

    rollup = {}
    for stage, rs in sorted(by_stage.items()):
        rs_sorted = sorted(rs, key=lambda x: x.epoch or 0)
        rollup[f"stage_{stage}"] = {
            "epochs": [asdict(r) for r in rs_sorted],
            "frac_zero_std_mean_avg": statistics.mean(
                [r.frac_zero_std_mean for r in rs_sorted if r.frac_zero_std_mean is not None]
            )
            if any(r.frac_zero_std_mean is not None for r in rs_sorted)
            else None,
            "frac_zero_std_mean_worst": max(
                (r.frac_zero_std_mean for r in rs_sorted if r.frac_zero_std_mean is not None),
                default=None,
            ),
            "steps_dead_pct_avg": statistics.mean(
                [r.steps_dead_pct for r in rs_sorted if r.steps_dead_pct is not None]
            )
            if any(r.steps_dead_pct is not None for r in rs_sorted)
            else None,
        }

    rollup_path = OUT_DIR / "wandb_grpo_signal_by_stage.json"
    rollup_path.write_text(json.dumps(rollup, indent=2), encoding="utf-8")

    print(f"\nWrote {out_path}")
    print(f"Wrote {rollup_path}")
    print(f"Log: {log_path}")
    print("\n=== Quick summary (frac_reward_zero_std mean, steps_dead %%) ===")
    for r in results:
        if r.stage is None:
            continue
        print(
            f"  S{r.stage} e{r.epoch}: zero_std={r.frac_zero_std_mean:.1%} "
            f"dead_steps={r.steps_dead_pct:.1f}% "
            f"reward={r.reward_mean:.3f} parse_fail={r.curriculum_parse_fail} "
            f"no_tag={r.format_no_tag_pct:.1f}%"
            if r.frac_zero_std_mean is not None and r.steps_dead_pct is not None
            and r.reward_mean is not None and r.format_no_tag_pct is not None
            else f"  {r.run_name}: partial data"
        )


if __name__ == "__main__":
    main()
