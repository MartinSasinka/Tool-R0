#!/usr/bin/env python3
"""Estimate per-stage max_completion_length / vllm_max_model_length from W&B + local prompts."""
from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "curricullum/evaluation/wandb_analysis"
CONFIG_PATH = REPO / "curricullum/train/configs/qwen3_4b_curriculum_v2.yaml"

ENTITY = "sasinka-martin"
PROJECT = "nestful-curriculum-toolr0"
RUN_PREFIXES = (
    "curriculum-20260612-1530-stage_",
    "qwen3-4b-curriculum-v2-cloud-stage_",
)

HISTORY_KEYS = [
    "train/completions/max_length",
    "train/completions/mean_length",
    "train/completions/max_terminated_length",
    "train/completions/mean_terminated_length",
    "train/completions/min_length",
    "train/completions/clipped_ratio",
    "train/completions/mean_terminated_length",
]


def _parse_stage_epoch(name: str) -> Tuple[Optional[int], Optional[int]]:
    import re

    m = re.search(r"stage_(\d+)-e(\d+)", name)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def _pct(vals: List[float], p: float) -> Optional[float]:
    if not vals:
        return None
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)


def _round_up(x: float, step: int = 128) -> int:
    return int(math.ceil(x / step) * step)


def _load_config_stages() -> Dict[int, Dict[str, int]]:
    import yaml

    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    out: Dict[int, Dict[str, int]] = {}
    for key, sc in (cfg.get("stages") or {}).items():
        if not key.startswith("stage_"):
            continue
        stage = int(key.split("_")[1])
        out[stage] = {
            "max_completion_length": int(sc["max_completion_length"]),
            "max_prompt_length": int(sc.get("max_prompt_length", cfg["grpo"]["max_prompt_length"])),
            "num_calls": int(sc["num_calls"]),
        }
    out["_global"] = {
        "vllm_max_model_length": int(cfg["grpo"]["vllm_max_model_length"]),
    }
    return out


def _analyze_prompt_lengths() -> Dict[str, Any]:
    import os
    import sys

    train_dir = REPO / "curricullum/train"
    data_dir = REPO / "curricullum/data"
    sys.path.insert(0, str(train_dir))
    sys.path.insert(0, str(data_dir))

    from transformers import AutoTokenizer
    from prepare_dataset_toolr0 import load_toolr0_jsonl

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507", trust_remote_code=True)
    cfg_stages = _load_config_stages()

    per_stage: Dict[int, Dict[str, Any]] = {}
    for stage in sorted(k for k in cfg_stages if isinstance(k, int)):
        sc = cfg_stages[stage]
        paths = list((REPO / "curricullum/data/filtered_toolr0_synthetic").glob(f"epoch_{stage}_*call.jsonl"))
        if not paths:
            continue
        data_path = str(paths[0])
        print(f"  prompt tokens stage {stage}...", flush=True)
        max_prompt = sc["max_prompt_length"]
        records, _ = load_toolr0_jsonl(data_path, default_num_calls=sc["num_calls"], skip_over_budget=False)
        # Stratified sample: cap per turn for speed
        by_turn_all: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for rec in records:
            by_turn_all[int(rec.get("turn_idx", 0))].append(rec)
        sampled: List[Dict[str, Any]] = []
        for turn, rs in sorted(by_turn_all.items()):
            sampled.extend(rs[:40])

        lens_all: List[int] = []
        by_turn: Dict[int, List[int]] = defaultdict(list)
        for rec in sampled:
            prompt = rec.get("prompt") or []
            text = tok.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            n = len(tok.encode(text, add_special_tokens=False))
            lens_all.append(n)
            by_turn[int(rec.get("turn_idx", 0))].append(n)

        per_stage[stage] = {
            "data_path": data_path,
            "prompt_p50": _pct(lens_all, 50),
            "prompt_p95": _pct(lens_all, 95),
            "prompt_p99": _pct(lens_all, 99),
            "prompt_max": max(lens_all) if lens_all else None,
            "prompt_over_budget": sum(1 for x in lens_all if x > max_prompt),
            "n_rows": len(lens_all),
            "n_rows_total": len(records),
            "by_turn": {
                str(t): {
                    "p95": _pct(v, 95),
                    "max": max(v),
                    "n": len(v),
                }
                for t, v in sorted(by_turn.items())
            },
        }
    return per_stage


def _fetch_wandb_completion_stats() -> List[Dict[str, Any]]:
    import wandb

    api = wandb.Api(timeout=120)
    runs = [
        r
        for r in api.runs(f"{ENTITY}/{PROJECT}", per_page=100)
        if r.name.startswith(RUN_PREFIXES) and r.state == "finished"
    ]
    rows: List[Dict[str, Any]] = []
    for run in sorted(runs, key=lambda r: r.name):
        stage, epoch = _parse_stage_epoch(run.name)
        hist_rows = list(run.scan_history(keys=HISTORY_KEYS, page_size=500))
        if not hist_rows:
            continue
        df = pd.DataFrame(hist_rows)
        entry: Dict[str, Any] = {
            "run_name": run.name,
            "stage": stage,
            "epoch": epoch,
            "history_steps": len(df),
        }
        for col, prefix in [
            ("train/completions/max_length", "comp_max"),
            ("train/completions/mean_length", "comp_mean"),
            ("train/completions/max_terminated_length", "term_max"),
            ("train/completions/mean_terminated_length", "term_mean"),
            ("train/completions/clipped_ratio", "clipped"),
        ]:
            if col not in df.columns:
                continue
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(s) == 0:
                continue
            entry[f"{prefix}_mean"] = float(s.mean())
            entry[f"{prefix}_p95"] = float(s.quantile(0.95))
            entry[f"{prefix}_max"] = float(s.max())
        rows.append(entry)
    return rows


def _recommend(
    stage: int,
    cfg: Dict[str, int],
    prompt_stats: Dict[str, Any],
    comp_runs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    stage_runs = [r for r in comp_runs if r.get("stage") == stage]
    term_max_vals = [r["term_max_max"] for r in stage_runs if r.get("term_max_max") is not None]
    comp_max_vals = [r["comp_max_max"] for r in stage_runs if r.get("comp_max_max") is not None]
    clipped_vals = [r["clipped_max"] for r in stage_runs if r.get("clipped_max") is not None]

    # Prefer terminated length (actual EOS) over raw max (may include padding to limit)
    obs_comp_p99 = _pct(term_max_vals, 99) or _pct(comp_max_vals, 99) or 0
    obs_comp_max = max(term_max_vals or comp_max_vals or [0])
    prompt_p99 = prompt_stats.get("prompt_p99") or 0
    prompt_max = prompt_stats.get("prompt_max") or 0
    worst_turn = prompt_stats.get("by_turn") or {}
    turn_max = max((v.get("max", 0) for v in worst_turn.values()), default=0)

    # Headroom: p99 + 25% margin, round to 128; floor 512 for thinking+tag
    rec_completion = max(512, _round_up(obs_comp_p99 * 1.25 if obs_comp_p99 else 768, 128))
    # Cap aggressive cuts if observed max was much higher
    rec_completion = max(rec_completion, _round_up(obs_comp_max * 1.1, 128))

    rec_prompt = max(
        int(cfg["max_prompt_length"]),
        _round_up(max(prompt_p99 * 1.05, turn_max * 1.02), 128),
    )
    # Keep prompt cap reasonable — use observed max with small buffer
    rec_prompt_tight = _round_up(prompt_max * 1.02, 128)

    rec_vllm = _round_up(rec_prompt_tight + rec_completion, 256)

    current_completion = cfg["max_completion_length"]
    current_prompt = cfg["max_prompt_length"]
    current_vllm = _load_config_stages()["_global"]["vllm_max_model_length"]

    return {
        "stage": stage,
        "num_calls": cfg["num_calls"],
        "current": {
            "max_completion_length": current_completion,
            "max_prompt_length": current_prompt,
            "vllm_max_model_length": current_vllm,
        },
        "observed_wandb": {
            "completion_term_max_p99": obs_comp_p99,
            "completion_term_max_peak": obs_comp_max,
            "clipped_ratio_peak": max(clipped_vals) if clipped_vals else 0.0,
            "n_runs": len(stage_runs),
        },
        "observed_prompt_tokens": {
            "p99": prompt_p99,
            "max": prompt_max,
            "worst_turn_max": turn_max,
            "over_current_budget": prompt_stats.get("prompt_over_budget", 0),
            "n_rows": prompt_stats.get("n_rows", 0),
        },
        "recommended": {
            "max_completion_length": min(rec_completion, current_completion),
            "max_prompt_length": min(rec_prompt_tight, rec_prompt),
            "vllm_max_model_length": rec_vllm,
        },
        "savings_tokens": {
            "completion": current_completion - min(rec_completion, current_completion),
            "vllm": current_vllm - rec_vllm,
        },
    }


def main() -> None:
    print("Fetching W&B completion length history...")
    comp_runs = _fetch_wandb_completion_stats()
    print("Analyzing local prompt token lengths...")
    prompt_by_stage = _analyze_prompt_lengths()
    cfg_stages = _load_config_stages()

    recommendations = []
    for stage in sorted(k for k in cfg_stages if isinstance(k, int)):
        rec = _recommend(stage, cfg_stages[stage], prompt_by_stage.get(stage, {}), comp_runs)
        recommendations.append(rec)

    payload = {
        "source": {
            "wandb_project": f"{ENTITY}/{PROJECT}",
            "config": str(CONFIG_PATH),
            "completion_metric": "train/completions/max_terminated_length (fallback max_length)",
        },
        "per_run_wandb": comp_runs,
        "per_stage_prompts": prompt_by_stage,
        "recommendations": recommendations,
    }

    out_path = OUT_DIR / "completion_length_recommendations.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}\n")

    print(f"{'St':>3} {'curr_comp':>10} {'rec_comp':>10} {'curr_pr':>8} {'rec_pr':>8} {'curr_vllm':>10} {'rec_vllm':>10} {'clip_pk':>8} {'comp_p99':>9}")
    for r in recommendations:
        c = r["current"]
        rc = r["recommended"]
        o = r["observed_wandb"]
        print(
            f"{r['stage']:>3} "
            f"{c['max_completion_length']:>10} "
            f"{rc['max_completion_length']:>10} "
            f"{c['max_prompt_length']:>8} "
            f"{rc['max_prompt_length']:>8} "
            f"{c['vllm_max_model_length']:>10} "
            f"{rc['vllm_max_model_length']:>10} "
            f"{o.get('clipped_ratio_peak', 0):>8.3f} "
            f"{o.get('completion_term_max_p99', 0):>9.0f}"
        )


if __name__ == "__main__":
    main()
