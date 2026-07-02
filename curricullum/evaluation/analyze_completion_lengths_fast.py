#!/usr/bin/env python3
"""Fast per-stage length recommendations: W&B history + sampled prompt tokenization."""
from __future__ import annotations

import json
import math
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "curricullum/evaluation/wandb_analysis/completion_length_recommendations.json"
CFG = REPO / "curricullum/train/configs/qwen3_4b_curriculum_v2.yaml"
DATA_DIR = REPO / "curricullum/data/filtered_toolr0_synthetic"

ENTITY, PROJECT = "sasinka-martin", "nestful-curriculum-toolr0"
PREFIXES = ("curriculum-20260612-1530-stage_", "qwen3-4b-curriculum-v2-cloud-stage_")


def _pct(vals: List[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    k = (len(s) - 1) * p / 100.0
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)


def _rup(x: float, step: int = 128) -> int:
    return int(math.ceil(max(0, x) / step) * step)


def _stage_epoch(name: str) -> Tuple[Optional[int], Optional[int]]:
    m = re.search(r"stage_(\d+)-e(\d+)", name)
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def wandb_completion_by_stage() -> Dict[int, Dict[str, float]]:
    import wandb

    keys = [
        "train/completions/max_terminated_length",
        "train/completions/max_length",
        "train/completions/mean_terminated_length",
        "train/completions/clipped_ratio",
    ]
    api = wandb.Api(timeout=120)
    by_stage: Dict[int, List[Dict[str, float]]] = defaultdict(list)
    for run in api.runs(f"{ENTITY}/{PROJECT}", per_page=100):
        if not run.name.startswith(PREFIXES) or run.state != "finished":
            continue
        stage, _ = _stage_epoch(run.name)
        if stage is None:
            continue
        rows = list(run.scan_history(keys=keys, page_size=500))
        if not rows:
            continue
        df = pd.DataFrame(rows)
        entry: Dict[str, float] = {}
        for col, key in [
            ("train/completions/max_terminated_length", "term_max"),
            ("train/completions/max_length", "raw_max"),
            ("train/completions/mean_terminated_length", "term_mean"),
            ("train/completions/clipped_ratio", "clipped"),
        ]:
            if col in df.columns:
                s = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(s):
                    entry[f"{key}_p95"] = float(s.quantile(0.95))
                    entry[f"{key}_max"] = float(s.max())
                    entry[f"{key}_mean"] = float(s.mean())
        if entry:
            by_stage[stage].append(entry)

    out: Dict[int, Dict[str, float]] = {}
    for stage, runs in by_stage.items():
        term_p95 = [r["term_max_p95"] for r in runs if "term_max_p95" in r]
        term_max = [r["term_max_max"] for r in runs if "term_max_max" in r]
        raw_max = [r["raw_max_max"] for r in runs if "raw_max_max" in r]
        clipped = [r["clipped_max"] for r in runs if "clipped_max" in r]
        out[stage] = {
            "term_max_p99": _pct(term_max, 99),
            "term_max_p95": _pct(term_p95, 95) if term_p95 else _pct(term_max, 95),
            "term_max_peak": max(term_max or raw_max or [0]),
            "clipped_peak": max(clipped or [0]),
            "n_runs": len(runs),
        }
    return out


def prompt_tokens_sampled(stage: int, num_calls: int, n_tasks: int = 60, seed: int = 42) -> Dict[str, Any]:
    import sys

    sys.path.insert(0, str(REPO / "curricullum/train"))
    sys.path.insert(0, str(REPO / "curricullum/data"))
    from prepare_dataset_toolr0 import expand_record, parse_row  # noqa
    from curricullum.data.exec_trajectory import get_ibm_registry  # noqa
    from transformers import AutoTokenizer

    paths = list(DATA_DIR.glob(f"epoch_{stage}_*call.jsonl"))
    if not paths:
        return {}
    lines = paths[0].read_text(encoding="utf-8").splitlines()
    rng = random.Random(seed + stage)
    picked = rng.sample(lines, min(n_tasks, len(lines)))

    ibm = get_ibm_registry()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507", trust_remote_code=True)
    lens: List[int] = []
    by_turn: Dict[int, List[int]] = defaultdict(list)
    for line in picked:
        row = json.loads(line)
        parsed = parse_row(row, default_num_calls=num_calls)
        if not parsed:
            continue
        for rec in expand_record(parsed, ibm):
            prompt = rec.get("prompt") or []
            text = tok.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            n = len(tok.encode(text, add_special_tokens=False))
            lens.append(n)
            by_turn[int(rec.get("turn_idx", 0))].append(n)

    return {
        "n_sampled_tasks": len(picked),
        "n_turn_rows": len(lens),
        "p95": _pct(lens, 95),
        "p99": _pct(lens, 99),
        "max": max(lens) if lens else 0,
        "by_turn": {str(t): {"max": max(v), "p95": _pct(v, 95), "n": len(v)} for t, v in sorted(by_turn.items())},
    }


def recommend(cfg: Dict[str, Any], comp: Dict[str, float], prompt: Dict[str, Any]) -> Dict[str, Any]:
    cur_comp = int(cfg["max_completion_length"])
    cur_prompt = int(cfg.get("max_prompt_length", cfg.get("_fallback_prompt", 4096)))
    cur_vllm = int(cfg.get("vllm_max_model_length", 6144))

    obs_comp = comp.get("term_max_p95") or comp.get("term_max_peak") or 0
    obs_peak = comp.get("term_max_peak") or obs_comp
    clipped = comp.get("clipped_peak") or 0

    # Completion: p95 terminated + 20% headroom; min 640 (thinking+tag); respect clips
    rec_comp = _rup(max(640, obs_comp * 1.2, obs_peak * 1.05 if clipped > 0.001 else obs_comp * 1.2), 128)
    if clipped < 0.001 and obs_peak < cur_comp * 0.4:
        rec_comp = min(rec_comp, _rup(obs_peak * 1.15, 128))

    prompt_max = prompt.get("max") or 0
    prompt_p99 = prompt.get("p99") or 0
    rec_prompt = _rup(max(prompt_p99 * 1.03, prompt_max * 1.02), 128)
    rec_prompt = min(rec_prompt, cur_prompt)  # don't expand beyond current unless needed
    if prompt_max > cur_prompt * 0.98:
        rec_prompt = _rup(prompt_max * 1.02, 128)

    rec_vllm = _rup(rec_prompt + rec_comp, 256)

    return {
        "current": {"max_completion_length": cur_comp, "max_prompt_length": cur_prompt, "vllm_max_model_length": cur_vllm},
        "observed_completion_tokens": comp,
        "observed_prompt_tokens": prompt,
        "recommended": {
            "max_completion_length": min(rec_comp, cur_comp),
            "max_prompt_length": rec_prompt,
            "vllm_max_model_length": min(rec_vllm, cur_vllm),
        },
        "delta": {
            "max_completion_length": cur_comp - min(rec_comp, cur_comp),
            "max_prompt_length": cur_prompt - rec_prompt,
            "vllm_max_model_length": cur_vllm - min(rec_vllm, cur_vllm),
        },
    }


def main() -> None:
    raw = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    global_prompt = int(raw["grpo"]["max_prompt_length"])
    global_vllm = int(raw["grpo"]["vllm_max_model_length"])

    print("W&B completion lengths...", flush=True)
    comp_by_stage = wandb_completion_by_stage()

    results = []
    for key, sc in sorted(raw["stages"].items()):
        stage = int(key.split("_")[1])
        print(f"Prompt sample stage {stage}...", flush=True)
        pr = prompt_tokens_sampled(stage, int(sc["num_calls"]))
        cfg = {
            **sc,
            "_fallback_prompt": global_prompt,
            "vllm_max_model_length": global_vllm,
        }
        rec = recommend(cfg, comp_by_stage.get(stage, {}), pr)
        rec["stage"] = stage
        results.append(rec)

    payload = {"recommendations": results, "comp_by_stage": comp_by_stage}
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}\n")
    print(f"{'St':>3} {'comp_now':>8} {'comp_rec':>8} {'pr_now':>7} {'pr_rec':>7} {'vllm_now':>8} {'vllm_rec':>8} {'clip':>6} {'comp_p95':>9}")
    for r in results:
        c, rc = r["current"], r["recommended"]
        o = r.get("observed_completion_tokens") or {}
        print(
            f"{r['stage']:>3} {c['max_completion_length']:>8} {rc['max_completion_length']:>8} "
            f"{c['max_prompt_length']:>7} {rc['max_prompt_length']:>7} "
            f"{c['vllm_max_model_length']:>8} {rc['vllm_max_model_length']:>8} "
            f"{o.get('clipped_peak', 0):>6.3f} {o.get('term_max_p95', 0):>9.0f}"
        )


if __name__ == "__main__":
    main()
