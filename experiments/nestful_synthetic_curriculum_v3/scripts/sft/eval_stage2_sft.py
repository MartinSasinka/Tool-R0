#!/usr/bin/env python3
"""Stage2 continuation SFT — unified evaluation orchestrator.

Evaluates BOTH the base model and an SFT checkpoint, on:
  (a) the synthetic Stage2 validation set (stage2_continuation/val.jsonl)
  (b) the full NESTFUL dev split (nestful_mtgrpo_minimal/data/splits/nestful_dev.jsonl)

...in TWO modes:
  1. continuation-conditioned eval (stage2 val ONLY) — via
     continuation_conditioned_eval.py. Tests whether the Stage2 continuation
     signal is learnable at all.
  2. free ReAct eval (stage2 val AND NESTFUL dev) — via the EXISTING GRPO
     eval machinery (nestful_mtgrpo_minimal/run.py --mode rollout_eval /
     val_eval), invoked as a subprocess so metric definitions
     (strict_gold_trace_pass, too_few_calls_rate, ReAct Win, ...) are
     IDENTICAL to the rest of the pipeline. Free ReAct never uses teacher
     forcing and never sees a forced prefix — the model must find call 1 on
     its own.

Temperature 0.0, free generation, no forced prefix, no teacher forcing at
eval time in EITHER mode (the continuation-conditioned mode's ONLY given
context is gold call 1 + gold observation 1, exactly mirroring the SFT
training input — see scripts/sft/sft_common.py).

Writes SFT_STAGE2_EVAL.md / .json into --out-dir, with:
  - per-mode / per-dataset / per-model metrics,
  - deltas (sft - base),
  - paired win/regression counts on NESTFUL dev (sample_id-matched).

Usage:
  python eval_stage2_sft.py \
      --checkpoint outputs/sft/stage2_continuation/run_.../adapter/epoch_1 \
      --out-dir outputs/sft/stage2_continuation/run_.../eval_20260709_130000
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from sft_common import DEFAULT_OUT_DIR, MINIMAL  # noqa: E402

RUN_PY = str(MINIMAL / "run.py")
CONFIG = str(MINIMAL / "config.yaml")
NESTFUL_DEV = str(MINIMAL / "data" / "splits" / "nestful_dev.jsonl")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="SFT LoRA adapter dir to evaluate.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--stage2-val", default=os.path.join(DEFAULT_OUT_DIR, "val.jsonl"))
    ap.add_argument("--nestful-dev", default=NESTFUL_DEV)
    ap.add_argument("--run-py", default=RUN_PY)
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--python-bin", default=sys.executable or "python")
    ap.add_argument("--eval-temperature", type=float, default=0.0)
    ap.add_argument("--use-vllm", action="store_true", default=False)
    ap.add_argument("--skip-free-react", action="store_true")
    ap.add_argument("--skip-continuation", action="store_true")
    ap.add_argument("--limit", type=int, default=None,
                    help="Debug: cap continuation-eval example count (does not affect run.py subprocess evals).")
    return ap.parse_args()


def _run(cmd: List[str], log_path: str) -> int:
    print(f"[eval_stage2_sft] $ {' '.join(cmd)}")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log_fh:
        proc = subprocess.run(cmd, stdout=log_fh, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        print(f"[eval_stage2_sft] WARNING: command exited {proc.returncode} — see {log_path}")
    return proc.returncode


def _load_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def run_free_react_stage2(args, label: str, checkpoint: Optional[str]) -> str:
    """Free ReAct eval on the synthetic Stage2 val set via run.py --mode rollout_eval.

    IMPORTANT: config.yaml's default `data.eval_stage` is 4, and
    data.load_tasks() filters strictly on `num_calls == eval_stage` (raising
    if the filter yields zero tasks) — every stage2_continuation record has
    num_calls == 2, so this MUST be overridden to 2 or rollout_eval crashes
    with "No tasks loaded ... with num_calls == 4" on this val set.
    """
    out_dir = os.path.join(args.out_dir, f"free_react_stage2_{label}")
    cmd = [
        args.python_bin, args.run_py,
        "--mode", "rollout_eval",
        "--config", args.config,
        "--override", f"experiment.output_dir={out_dir}",
        "--override", f"paths.eval_jsonl={os.path.abspath(args.stage2_val)}",
        "--override", "data.eval_stage=2",
        "--override", f"generation.temperature={args.eval_temperature}",
        "--override", f"hardware.use_vllm={'true' if args.use_vllm else 'false'}",
    ]
    if checkpoint:
        cmd += ["--checkpoint", checkpoint]
    else:
        cmd += ["--override", "model.lora_adapter=null"]
    _run(cmd, os.path.join(args.out_dir, f"free_react_stage2_{label}.log"))
    return out_dir


def run_free_react_nestful_dev(args, label: str, checkpoint: Optional[str]) -> str:
    """Free ReAct eval on the full NESTFUL dev split via run.py --mode val_eval
    (identical invocation pattern to the GRPO pilot's baseline_dev_eval)."""
    out_dir = os.path.join(args.out_dir, f"free_react_nestful_dev_{label}")
    cmd = [
        args.python_bin, args.run_py,
        "--mode", "val_eval",
        "--config", args.config,
        "--override", f"experiment.output_dir={out_dir}",
        "--override", f"paths.full_nestful_jsonl={os.path.abspath(args.nestful_dev)}",
        "--override", "validation.subset_size=0",
        "--override", "validation.require_win_rate=false",
        "--override", "validation.stage=0",
        "--override", "validation.epoch=0",
        "--override", f"generation.temperature={args.eval_temperature}",
        "--override", f"hardware.use_vllm={'true' if args.use_vllm else 'false'}",
    ]
    if checkpoint:
        cmd += ["--checkpoint", checkpoint]
    else:
        cmd += ["--override", "model.lora_adapter=null"]
    _run(cmd, os.path.join(args.out_dir, f"free_react_nestful_dev_{label}.log"))
    return out_dir


def run_continuation_eval(args, label: str, checkpoint: Optional[str]) -> str:
    out_dir = os.path.join(args.out_dir, f"continuation_eval_{label}")
    cmd = [
        args.python_bin, str(HERE / "continuation_conditioned_eval.py"),
        "--val-path", args.stage2_val,
        "--out-dir", out_dir,
        "--eval-temperature", str(args.eval_temperature),
    ]
    if checkpoint:
        cmd += ["--checkpoint", checkpoint]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    _run(cmd, os.path.join(args.out_dir, f"continuation_eval_{label}.log"))
    return out_dir


def _call_count_stats_from_final_eval(out_dir: str) -> Dict[str, Optional[float]]:
    """too_few_calls_rate / avg_predicted_calls for a `run.py --mode val_eval`
    (-> mode_final_eval) output dir.

    NOTE: these two fields are NOT part of `metrics.json`'s
    `internal_metrics_diagnostic` block for final_eval/val_eval (that block
    only carries _OFFICIAL_KEYS = f1_func/f1_param/partial_sequence_accuracy/
    full_sequence_accuracy/win_rate — see metrics.aggregate_final_eval).
    too_few_calls_rate/avg_predicted_calls are only pre-aggregated in the
    SEPARATE `mode_rollout_eval` metrics schema. For NESTFUL dev (val_eval)
    we therefore recompute them here directly from `_traj.num_tool_calls` /
    `_traj.gold_num_turns` in final_eval_trajectories.jsonl (see
    rollout.Trajectory.to_dict), matching exactly the same definition
    mode_rollout_eval uses (`num_tool_calls < gold_num_turns`).
    """
    rows = _load_jsonl(os.path.join(out_dir, "final_eval_trajectories.jsonl"))
    too_few, predicted = [], []
    for r in rows:
        traj = r.get("_traj") or {}
        n_pred, n_gold = traj.get("num_tool_calls"), traj.get("gold_num_turns")
        if n_pred is None or n_gold is None:
            continue
        predicted.append(float(n_pred))
        too_few.append(1.0 if n_pred < n_gold else 0.0)
    return {
        "too_few_calls_rate": (sum(too_few) / len(too_few)) if too_few else None,
        "avg_predicted_calls": (sum(predicted) / len(predicted)) if predicted else None,
    }


def _paired_nestful_comparison(base_dir: str, sft_dir: str) -> Dict[str, Any]:
    base_traj = {r["sample_id"]: r for r in _load_jsonl(os.path.join(base_dir, "final_eval_trajectories.jsonl"))}
    sft_traj = {r["sample_id"]: r for r in _load_jsonl(os.path.join(sft_dir, "final_eval_trajectories.jsonl"))}
    common_ids = sorted(set(base_traj) & set(sft_traj))

    def _win(row: Dict[str, Any]) -> Optional[bool]:
        traj = row.get("_traj", {})
        w = traj.get("official_win")
        if w is not None:
            return bool(w)
        # Fallback when official Win is unavailable (e.g. no IBM functions dir
        # / Windows dev machine): use final_answer_pass as a WEAKER proxy.
        # Flagged explicitly in the report — this is NOT the canonical metric.
        fap = row.get("final_answer_pass")
        return bool(fap) if fap is not None else None

    paired_wins = 0       # base lost, sft won
    paired_regressions = 0  # base won, sft lost
    both_win = 0
    both_lose = 0
    unresolved = 0
    for sid in common_ids:
        bw = _win(base_traj[sid])
        sw = _win(sft_traj[sid])
        if bw is None or sw is None:
            unresolved += 1
            continue
        if bw and sw:
            both_win += 1
        elif (not bw) and (not sw):
            both_lose += 1
        elif (not bw) and sw:
            paired_wins += 1
        elif bw and (not sw):
            paired_regressions += 1

    proxy_used = not any(
        base_traj[sid].get("_traj", {}).get("official_win") is not None for sid in common_ids
    ) if common_ids else False

    return {
        "num_common_samples": len(common_ids),
        "paired_wins": paired_wins,
        "paired_regressions": paired_regressions,
        "both_win": both_win,
        "both_lose": both_lose,
        "unresolved": unresolved,
        "used_final_answer_pass_proxy_not_official_win": proxy_used,
    }


def main() -> int:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if not os.path.isfile(args.stage2_val):
        print(f"[eval_stage2_sft] ERROR: stage2 val not found: {args.stage2_val}\n"
              f"  Build it first: python scripts/sft/build_stage2_sft_dataset.py", file=sys.stderr)
        return 1
    if not os.path.isfile(args.checkpoint.rstrip("/\\") + "/adapter_config.json"):
        print(f"[eval_stage2_sft] ERROR: --checkpoint is not a valid LoRA adapter dir "
              f"(no adapter_config.json): {args.checkpoint}", file=sys.stderr)
        return 1

    results: Dict[str, Any] = {"checkpoint": args.checkpoint, "eval_temperature": args.eval_temperature}
    dirs: Dict[str, str] = {}

    for label, ckpt in (("base", None), ("sft", args.checkpoint)):
        if not args.skip_continuation:
            dirs[f"continuation_{label}"] = run_continuation_eval(args, label, ckpt)
        if not args.skip_free_react:
            dirs[f"free_react_stage2_{label}"] = run_free_react_stage2(args, label, ckpt)
            dirs[f"free_react_nestful_dev_{label}"] = run_free_react_nestful_dev(args, label, ckpt)

    for label in ("base", "sft"):
        if f"continuation_{label}" in dirs:
            results[f"continuation_{label}"] = _load_json(
                os.path.join(dirs[f"continuation_{label}"], "continuation_eval_metrics.json"))
        if f"free_react_stage2_{label}" in dirs:
            results[f"free_react_stage2_{label}"] = _load_json(
                os.path.join(dirs[f"free_react_stage2_{label}"], "metrics.json"))
        if f"free_react_nestful_dev_{label}" in dirs:
            nd_dir = dirs[f"free_react_nestful_dev_{label}"]
            m = _load_json(os.path.join(nd_dir, "metrics.json")) or {}
            mo = _load_json(os.path.join(nd_dir, "metrics_official.json")) or {}
            ep = _load_json(os.path.join(nd_dir, "metrics_epoch_0.json")) or {}
            results[f"free_react_nestful_dev_{label}"] = {
                # f1_func/f1_param/partial_sequence_accuracy/full_sequence_accuracy/win_rate
                "internal_diagnostic": m.get("internal_metrics_diagnostic", m),
                "official": mo,
                "react_win_rate": ep.get("react_win_rate"),
                # too_few_calls_rate/avg_predicted_calls are NOT in
                # internal_metrics_diagnostic for val_eval; recomputed directly
                # from the trajectories file (see _call_count_stats_from_final_eval).
                "call_count_stats": _call_count_stats_from_final_eval(nd_dir),
            }

    if not args.skip_free_react:
        results["paired_nestful_dev"] = _paired_nestful_comparison(
            dirs.get("free_react_nestful_dev_base", ""), dirs.get("free_react_nestful_dev_sft", ""),
        )

    with open(os.path.join(args.out_dir, "SFT_STAGE2_EVAL.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False, default=str)
    _write_markdown(args.out_dir, results)

    print(f"[eval_stage2_sft] wrote {os.path.join(args.out_dir, 'SFT_STAGE2_EVAL.json')}")
    print(f"[eval_stage2_sft] wrote {os.path.join(args.out_dir, 'SFT_STAGE2_EVAL.md')}")
    return 0


def _fmt(v, digits=4):
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def _delta(a, b):
    if a is None or b is None:
        return None
    try:
        return float(a) - float(b)
    except (TypeError, ValueError):
        return None


def _write_markdown(out_dir: str, r: Dict[str, Any]) -> None:
    lines: List[str] = []
    w = lines.append
    w("# Stage2 Continuation SFT — Evaluation Summary")
    w("")
    w(f"- checkpoint (SFT): `{r.get('checkpoint')}`")
    w(f"- eval temperature: {r.get('eval_temperature')} (free generation, no forced prefix beyond "
      f"the fixed continuation-conditioned prefix; no teacher forcing in either mode)")
    w("")

    w("## 1. Continuation-conditioned eval (Stage2 val set)")
    w("")
    w("Tests whether the Stage2 continuation signal is learnable in isolation "
      "(input includes gold call 1 + gold observation 1; model must generate "
      "call 2 + the terminal finish).")
    w("")
    cb, cs = r.get("continuation_base") or {}, r.get("continuation_sft") or {}
    w("| metric | base | sft | delta |")
    w("|---|---:|---:|---:|")
    for k in ("final_answer_pass", "strict_gold_trace_pass", "too_few_calls_rate",
             "no_tool_call_rate", "parse_error_rate", "wrong_tool_rate",
             "wrong_argument_rate", "avg_predicted_continuation_calls"):
        bv, sv = cb.get(k), cs.get(k)
        w(f"| {k} | {_fmt(bv)} | {_fmt(sv)} | {_fmt(_delta(sv, bv))} |")
    w("")
    w(f"- num_scored: base={cb.get('num_scored')} sft={cs.get('num_scored')}  "
      f"executor_mode: base={cb.get('executor_mode')} sft={cs.get('executor_mode')}")
    w("")

    w("## 2. Free ReAct eval — synthetic Stage2 val set")
    w("")
    w("Model must find call 1 AND call 2 unaided (no given prefix). Uses the "
      "SAME rollout.run_episode() ReAct loop as GRPO training/eval.")
    w("")
    fb, fs = r.get("free_react_stage2_base") or {}, r.get("free_react_stage2_sft") or {}
    w("| metric | base | sft | delta |")
    w("|---|---:|---:|---:|")
    for k in ("strict_gold_trace_pass", "final_answer_pass", "too_few_calls_rate",
             "avg_predicted_calls", "zero_tool_calls", "clipped_completion_rate"):
        bv, sv = fb.get(k), fs.get(k)
        w(f"| {k} | {_fmt(bv)} | {_fmt(sv)} | {_fmt(_delta(sv, bv))} |")
    w("")

    w("## 3. Free ReAct eval — full NESTFUL dev (n=200)")
    w("")
    nb, ns = r.get("free_react_nestful_dev_base") or {}, r.get("free_react_nestful_dev_sft") or {}
    w("| metric | base | sft | delta |")
    w("|---|---:|---:|---:|")
    w(f"| ReAct Win (official) | {_fmt(nb.get('react_win_rate'))} | {_fmt(ns.get('react_win_rate'))} | "
      f"{_fmt(_delta(ns.get('react_win_rate'), nb.get('react_win_rate')))} |")
    for k in ("too_few_calls_rate", "avg_predicted_calls"):
        bv = (nb.get("call_count_stats") or {}).get(k)
        sv = (ns.get("call_count_stats") or {}).get(k)
        w(f"| {k} | {_fmt(bv)} | {_fmt(sv)} | {_fmt(_delta(sv, bv))} |")
    for k in ("win_rate", "f1_func", "f1_param", "full_sequence_accuracy", "partial_sequence_accuracy"):
        bv = (nb.get("internal_diagnostic") or {}).get(k)
        sv = (ns.get("internal_diagnostic") or {}).get(k)
        w(f"| internal_{k} | {_fmt(bv)} | {_fmt(sv)} | {_fmt(_delta(sv, bv))} |")
    w("")
    pw = r.get("paired_nestful_dev") or {}
    if pw:
        w("### Paired comparison (sample_id-matched)")
        w("")
        w(f"- common samples scored: {pw.get('num_common_samples')}")
        w(f"- paired wins (base lost -> sft won): **{pw.get('paired_wins')}**")
        w(f"- paired regressions (base won -> sft lost): **{pw.get('paired_regressions')}**")
        w(f"- both win: {pw.get('both_win')}, both lose: {pw.get('both_lose')}, "
          f"unresolved: {pw.get('unresolved')}")
        if pw.get("used_final_answer_pass_proxy_not_official_win"):
            w("- **WARNING**: official_win was unavailable for these samples "
              "(no IBM executable_functions dir / non-Linux host) — paired "
              "counts fall back to `final_answer_pass`, a WEAKER proxy. Do "
              "not report these paired counts as the canonical NESTFUL Win "
              "comparison without re-running on a host with the IBM "
              "executable-functions dir available.")
        w("")

    w("## Statistical caveats")
    w("")
    w("- NESTFUL dev is n=200 — a handful of flipped samples moves ReAct Win "
      "by ~0.5pp; treat any small delta as noise unless it is backed by a "
      "consistent shift in `too_few_calls_rate` / paired wins-regressions.")
    w("- This is a single SFT run (1 warmup epoch by default) with no seed "
      "averaging — do not generalize a single seed's result.")
    w("- Continuation-conditioned eval and free ReAct eval measure DIFFERENT "
      "things; a model can improve on one and not the other. Report both, "
      "do not average them into one number.")
    w("")

    with open(os.path.join(out_dir, "SFT_STAGE2_EVAL.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
