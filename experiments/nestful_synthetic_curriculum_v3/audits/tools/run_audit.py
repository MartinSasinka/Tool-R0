#!/usr/bin/env python3
"""Read-only run audit: walks outputs/runs/, extracts config + metrics per run/stage/epoch,
plus failure-mode aggregates from train_log.jsonl. Writes RUN_AUDIT.json and RUN_AUDIT.csv."""
from __future__ import annotations

import csv
import json
import math
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
V3 = os.path.join(REPO, "experiments", "nestful_synthetic_curriculum_v3")
RUNS = os.path.join(V3, "outputs", "runs")
AUDITS = os.path.join(V3, "audits")

TRAIN_RUNS = [
    "20260702_112042", "20260702_112150", "0260703_145219_v3_1",
    "20260707_103035_v3_1", "20260707_152750_v3_1", "20260707_183801_v3_1",
    "20260708_212347_v3_1",
]


def jload(p: str) -> Optional[Any]:
    try:
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def jlines(p: str) -> List[Dict[str, Any]]:
    out = []
    try:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except OSError:
        pass
    return out


MOTIF_RE = re.compile(r"traj_v3_1_([a-z_]+?)_(?:too_few_calls|wrong_tool|wrong_arguments|"
                      r"invalid_reference|wrong_condition|premature_final|parse_error|"
                      r"wrong_answer|other)_\d+")


def entropy(counter: Counter) -> float:
    tot = sum(counter.values())
    if tot == 0:
        return 0.0
    return -sum((c / tot) * math.log2(c / tot) for c in counter.values() if c)


def analyze_train_log(p: str) -> Dict[str, Any]:
    rows = [r for r in jlines(p) if "episode_rewards" in r]
    if not rows:
        return {"groups": 0}
    n = len(rows)
    dead = sum(1 for r in rows if r.get("dead_group"))
    dead_old = sum(1 for r in rows if r.get("dead_group_old_flattened"))
    mixed = sum(1 for r in rows if r.get("group_mixed"))
    all_zero = sum(1 for r in rows if r.get("group_all_zero"))
    all_one = sum(1 for r in rows if r.get("group_all_one"))
    pos_art = sum(1 for r in rows if r.get("position_artifact_detected"))
    uniq_rewards = [r.get("n_unique_episode_rewards", 0) for r in rows]
    uniq_completions = [r.get("n_unique_completion_hashes", 0) for r in rows]
    group_sizes = [len(r.get("episode_rewards", [])) for r in rows]
    mean_rewards = [r.get("mean_reward", 0.0) for r in rows]
    reward_values = Counter()
    for r in rows:
        for v in r.get("episode_rewards", []):
            reward_values[round(float(v), 4)] += 1
    ep_total = sum(group_sizes)
    too_few = sum(r.get("too_few_calls_count", 0) for r in rows)
    no_tool = sum(r.get("no_tool_call_count", 0) for r in rows)
    parse_err = sum(r.get("parse_error_count", 0) for r in rows)
    wrong_tool = sum(r.get("wrong_tool_count", 0) for r in rows)
    wrong_arg = sum(r.get("wrong_arg_count", 0) for r in rows)
    invalid_ref = sum(r.get("invalid_ref_count", 0) for r in rows)
    premature = sum(r.get("premature_final_count", 0) for r in rows)
    pred_calls = [c for r in rows for c in r.get("predicted_num_calls", [])]
    opt_steps = sum(1 for r in rows if r.get("optimizer_step_executed"))
    kl_vals = [r.get("kl", 0.0) for r in rows if isinstance(r.get("kl"), (int, float))]

    # motif-level failure aggregation (task_id encodes motif + source failure cluster)
    motif_stats: Dict[str, Dict[str, float]] = {}
    for r in rows:
        m = MOTIF_RE.search(r.get("task_id", "") or "")
        key = m.group(1) if m else "unknown"
        s = motif_stats.setdefault(key, {"groups": 0, "dead": 0, "mean_reward": 0.0})
        s["groups"] += 1
        s["dead"] += 1 if r.get("dead_group") else 0
        s["mean_reward"] += r.get("mean_reward", 0.0)
    for s in motif_stats.values():
        s["dead_rate"] = round(s["dead"] / s["groups"], 3)
        s["mean_reward"] = round(s["mean_reward"] / s["groups"], 3)

    def avg(xs):
        return round(sum(xs) / len(xs), 4) if xs else None

    return {
        "groups": n,
        "optimizer_steps": opt_steps,
        "dead_group_rate": round(dead / n, 4),
        "dead_group_rate_old_flattened": round(dead_old / n, 4),
        "mixed_group_rate": round(mixed / n, 4),
        "all_zero_group_rate": round(all_zero / n, 4),
        "all_one_group_rate": round(all_one / n, 4),
        "position_artifact_rate": round(pos_art / n, 4),
        "avg_unique_episode_rewards_per_group": avg(uniq_rewards),
        "avg_unique_completions_per_group": avg(uniq_completions),
        "avg_group_size": avg(group_sizes),
        "mean_reward_overall": avg(mean_rewards),
        "reward_value_entropy_bits": round(entropy(reward_values), 3),
        "distinct_reward_values": len(reward_values),
        "reward_value_hist_top": reward_values.most_common(8),
        "episodes_total": ep_total,
        "too_few_calls_rate": round(too_few / ep_total, 4) if ep_total else None,
        "no_tool_call_rate": round(no_tool / ep_total, 4) if ep_total else None,
        "parse_error_rate": round(parse_err / ep_total, 4) if ep_total else None,
        "wrong_tool_rate": round(wrong_tool / ep_total, 4) if ep_total else None,
        "wrong_arg_rate": round(wrong_arg / ep_total, 4) if ep_total else None,
        "invalid_ref_rate": round(invalid_ref / ep_total, 4) if ep_total else None,
        "premature_final_rate": round(premature / ep_total, 4) if ep_total else None,
        "avg_predicted_calls": avg(pred_calls),
        "mean_kl": avg(kl_vals),
        "motif_stats": dict(sorted(motif_stats.items(), key=lambda kv: -kv[1]["groups"])[:12]),
    }


def extract_epoch(run_dir: str, stage_dir: str, epoch_dir: str) -> Dict[str, Any]:
    ep_path = os.path.join(run_dir, stage_dir, epoch_dir)
    out: Dict[str, Any] = {"stage": stage_dir, "epoch": epoch_dir}
    ts = jload(os.path.join(ep_path, "train_summary.json")) or {}
    out["train_summary"] = {k: ts.get(k) for k in (
        "steps", "num_tasks", "reward_policy_resolved", "reward_fn_module", "reward_fn_name",
        "reward_fallback_used", "dead_group_rate", "position_artifact_group_rate",
        "n_unique_reward_values", "no_tool_call_rate", "too_few_calls_rate",
        "avg_predicted_calls", "contributing_turns_total", "trained", "vllm_rollout")}
    val_int = jload(os.path.join(ep_path, "val_eval", "metrics.json")) or {}
    val_off = jload(os.path.join(ep_path, "val_eval", "metrics_official.json")) or {}
    out["dev_internal_win"] = (val_int.get("internal_metrics_diagnostic") or {}).get("win_rate")
    out["dev_our_metrics"] = val_int.get("our_metrics")
    out["dev_official_win"] = val_off.get("win_rate")
    out["dev_official"] = {k: val_off.get(k) for k in (
        "f1_func", "f1_param", "partial_sequence_accuracy", "full_sequence_accuracy")} if val_off else None
    out["train_log_analysis"] = analyze_train_log(os.path.join(ep_path, "train_log.jsonl"))
    return out


def extract_run(run_id: str) -> Dict[str, Any]:
    run_dir = os.path.join(RUNS, run_id)
    out: Dict[str, Any] = {"run_id": run_id, "path": os.path.relpath(run_dir, REPO).replace("\\", "/")}

    # config from any checkpoint's config_used.json
    cfg = None
    for root, _dirs, files in os.walk(run_dir):
        if "config_used.json" in files and "checkpoints" in root:
            cfg = jload(os.path.join(root, "config_used.json"))
            if cfg:
                break
    if cfg:
        out["config"] = {
            "base_model": cfg.get("model", {}).get("base_model"),
            "seed": cfg.get("experiment", {}).get("seed"),
            "lora": {k: cfg.get("finetuning", {}).get(k) for k in
                     ("method", "lora_r", "lora_alpha", "lora_dropout", "bnb_4bit_quant_type")},
            "lr": cfg.get("training", {}).get("learning_rate"),
            "kl_beta": cfg.get("training", {}).get("kl_beta"),
            "num_generations": cfg.get("generation", {}).get("num_generations"),
            "rollout_temperature": cfg.get("generation", {}).get("temperature"),
            "rollout_top_p": cfg.get("generation", {}).get("top_p"),
            "reward_train_policy": (cfg.get("reward") or {}).get("train_policy"),
            "train_jsonl": cfg.get("paths", {}).get("train_jsonl"),
            "use_vllm": cfg.get("hardware", {}).get("use_vllm"),
            "grad_accum": cfg.get("training", {}).get("gradient_accumulation_steps"),
        }
    # baseline dev eval
    base_int = jload(os.path.join(run_dir, "baseline_dev_eval", "metrics.json")) or {}
    base_off = jload(os.path.join(run_dir, "baseline_dev_eval", "metrics_official.json")) or {}
    out["baseline_dev_internal_win"] = (base_int.get("internal_metrics_diagnostic") or {}).get("win_rate")
    out["baseline_dev_official_win"] = base_off.get("win_rate")
    out["baseline_dev_num_tasks"] = base_int.get("num_tasks")
    out["has_validation_subset"] = os.path.isfile(os.path.join(run_dir, "validation_subset.jsonl"))

    # data_base provenance
    db = os.path.join(run_dir, "data_base")
    prov = {}
    if os.path.isdir(db):
        for fn in sorted(os.listdir(db)):
            p = os.path.join(db, fn)
            try:
                with open(p, encoding="utf-8") as fh:
                    first = json.loads(fh.readline())
                sid = str(first.get("sample_id") or first.get("task_id") or "")
                if sid.startswith("prefix_v3_1"):
                    prov[fn] = "curriculum_v3_1 (dataset A)"
                elif sid.startswith("synthetic_v3"):
                    prov[fn] = "curriculum_v3 (older v3 generator)"
                elif sid.startswith("synthetic-epoch"):
                    prov[fn] = "filtered_toolr0_synthetic (dataset B)"
                else:
                    prov[fn] = f"unknown ({sid[:40]})"
            except (OSError, json.JSONDecodeError, ValueError):
                prov[fn] = "unreadable"
    out["data_base_provenance"] = prov

    # stages/epochs
    out["stages"] = []
    for stage_dir in sorted(d for d in os.listdir(run_dir) if d.startswith("stage_")):
        sp = os.path.join(run_dir, stage_dir)
        stage_info: Dict[str, Any] = {"stage": stage_dir}
        stage_info["epoch_summary"] = jlines(os.path.join(sp, "epoch_summary.jsonl"))
        gate = jload(os.path.join(sp, "stage_gate_report.json"))
        if gate:
            stage_info["stage_gate"] = {"pass": gate.get("pass"),
                                        "hard_failures": gate.get("hard_failures")}
        stage_info["epochs"] = []
        for ed in sorted(d for d in os.listdir(sp) if d.startswith("epoch_")):
            if os.path.isdir(os.path.join(sp, ed)):
                stage_info["epochs"].append(extract_epoch(run_dir, stage_dir, ed))
        out["stages"].append(stage_info)
    return out


def main() -> None:
    result: Dict[str, Any] = {"train_runs": [], "final_eval_batches": {}}
    for rid in TRAIN_RUNS:
        if os.path.isdir(os.path.join(RUNS, rid)):
            print(f"[run-audit] {rid}")
            result["train_runs"].append(extract_run(rid))

    # final eval batches
    batches = {
        "final_eval_all_runs_20260707_215620": os.path.join(RUNS, "final_eval_all_runs_20260707_215620"),
        "final_eval_all_runs_20260708_164607_temp0": os.path.join(
            RUNS, "final_eval_all_runs_20260708_164607_temp0", "final_eval_all_runs_20260708_164607_temp0"),
        "final_eval_stage3_e1e2_20260709_093453_temp0": os.path.join(
            RUNS, "final_eval_stage3_e1e2_20260709_093453_temp0", "final_eval_stage3_e1e2_20260709_093453_temp0"),
    }
    for bid, bdir in batches.items():
        cells = {}
        if os.path.isdir(bdir):
            for cell in sorted(os.listdir(bdir)):
                cd = os.path.join(bdir, cell)
                if not os.path.isdir(cd):
                    continue
                mi = jload(os.path.join(cd, "metrics.json")) or {}
                mo = jload(os.path.join(cd, "metrics_official.json"))
                cells[cell] = {
                    "num_tasks": mi.get("num_tasks"),
                    "internal_win": (mi.get("internal_metrics_diagnostic") or {}).get("win_rate"),
                    "our_metrics": mi.get("our_metrics"),
                    "official_win": mo.get("win_rate") if mo else None,
                    "official": mo,
                }
        result["final_eval_batches"][bid] = cells

    with open(os.path.join(AUDITS, "RUN_AUDIT.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)

    # CSV: one row per run/stage/epoch
    rows = []
    for run in result["train_runs"]:
        cfg = run.get("config") or {}
        for st in run["stages"]:
            for ep in st["epochs"]:
                ts = ep["train_summary"]
                tl = ep["train_log_analysis"]
                rows.append({
                    "run_id": run["run_id"],
                    "stage": st["stage"],
                    "epoch": ep["epoch"],
                    "train_jsonl": cfg.get("train_jsonl"),
                    "base_model": cfg.get("base_model"),
                    "lora": f"r{(cfg.get('lora') or {}).get('lora_r')}/a{(cfg.get('lora') or {}).get('lora_alpha')}",
                    "reward_policy": ts.get("reward_policy_resolved") or cfg.get("reward_train_policy"),
                    "lr": cfg.get("lr"),
                    "kl_beta": cfg.get("kl_beta"),
                    "num_generations": cfg.get("num_generations"),
                    "rollout_temp": cfg.get("rollout_temperature"),
                    "rollout_top_p": cfg.get("rollout_top_p"),
                    "seed": cfg.get("seed"),
                    "optimizer_steps": ts.get("steps"),
                    "dead_group_rate": ts.get("dead_group_rate") or tl.get("dead_group_rate"),
                    "mixed_group_rate": tl.get("mixed_group_rate"),
                    "n_unique_reward_values": ts.get("n_unique_reward_values"),
                    "too_few_calls_rate": ts.get("too_few_calls_rate"),
                    "avg_predicted_calls": ts.get("avg_predicted_calls"),
                    "dev_internal_win": ep.get("dev_internal_win"),
                    "dev_official_win": ep.get("dev_official_win"),
                    "baseline_dev_internal_win": run.get("baseline_dev_internal_win"),
                    "baseline_dev_official_win": run.get("baseline_dev_official_win"),
                    "baseline_same_run": bool(run.get("baseline_dev_official_win") is not None),
                })
    with open(os.path.join(AUDITS, "RUN_AUDIT.csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[run-audit] wrote RUN_AUDIT.json / RUN_AUDIT.csv ({len(rows)} epoch rows)")


if __name__ == "__main__":
    main()
