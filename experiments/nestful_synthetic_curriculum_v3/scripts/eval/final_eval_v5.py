#!/usr/bin/env python3
"""Final deterministic NESTFUL evaluation for curriculum v5 — ONE simple path.

Two subcommands:

  run       Evaluate ONE checkpoint (or the plain baseline) on the full
            NESTFUL test set with the official scorer. Decoding is forced to
            temperature=0.0, top_p=1.0, one rollout, ReAct, full executor —
            it can NEVER silently inherit training decoding settings, because
            every knob is passed as an explicit --override.

  compare   Read two or three finished `run` output directories (baseline,
            best, optionally final) and produce a paired comparison report:
            aggregate official metrics, metrics by call count (2 / 3 / 4+),
            gained/regressed task lists, function/parameter/executability/
            under-calling/unsupported-trace diagnostics, and a paired
            bootstrap 95% CI on the win-rate delta.

Typical use (see scripts/v5/*.sh for the shell entry points):

  python final_eval_v5.py run --label baseline --out-dir RUN/final_eval/baseline
  python final_eval_v5.py run --label best --checkpoint RUN/best_adapter \\
      --out-dir RUN/final_eval/best
  python final_eval_v5.py compare --baseline RUN/final_eval/baseline \\
      --best RUN/final_eval/best --out RUN/final_eval/compare
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from datetime import datetime, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))
_V3 = os.path.normpath(os.path.join(_HERE, "..", ".."))
_PARTIAL = os.path.normpath(os.path.join(_V3, "..", "nestful_mtgrpo_partial"))
_RUN_PY = os.path.join(_V3, "run.py")
_DEFAULT_CONFIG = os.path.join(_PARTIAL, "config.yaml")

BOOTSTRAP_ITERS = 10_000
BOOTSTRAP_SEED = 20260715


# ─────────────────────────────────────────────────────────────────── run ────

def cmd_run(args: argparse.Namespace) -> int:
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    cmd = [sys.executable, _RUN_PY, "--mode", "final_eval",
           "--config", args.config,
           "--override", f"experiment.output_dir={out_dir}",
           # Deterministic decoding — NEVER inherited from training config.
           "--override", "generation.temperature=0.0",
           "--override", "generation.top_p=1.0",
           "--override", "data.num_eval_rollouts=1",
           "--override", "data.eval_paradigm=react",
           ]
    if args.eval_set:
        cmd += ["--override", f"paths.full_nestful_jsonl={os.path.abspath(args.eval_set)}"]
    if args.max_tasks:
        cmd += ["--override", f"data.max_eval_tasks={int(args.max_tasks)}"]
    if os.environ.get("USE_VLLM", "0") == "1":
        cmd += ["--override", "hardware.use_vllm=true"]
        tp = os.environ.get("EVAL_TP", "").strip()
        if tp:
            cmd += ["--override", f"hardware.vllm_tensor_parallel_size={tp}"]
        util = os.environ.get("VLLM_GPU_UTIL", "").strip()
        if util:
            cmd += ["--override", f"hardware.vllm_gpu_memory_utilization={util}"]
    if args.checkpoint:
        ck = os.path.abspath(args.checkpoint)
        if not os.path.isfile(os.path.join(ck, "adapter_config.json")):
            print(f"[final_eval_v5] ABORT: {ck} has no adapter_config.json",
                  file=sys.stderr)
            return 2
        cmd += ["--checkpoint", ck]
    else:
        cmd += ["--override", "model.lora_adapter=null"]

    manifest = {
        "label": args.label,
        "checkpoint": os.path.abspath(args.checkpoint) if args.checkpoint else None,
        "eval_set": os.path.abspath(args.eval_set) if args.eval_set else "config default",
        "decoding": {"temperature": 0.0, "top_p": 1.0, "num_rollouts": 1,
                     "paradigm": "react"},
        "max_tasks": args.max_tasks or None,
        "command": cmd,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    print("[final_eval_v5] --- resolved configuration ---")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    with open(os.path.join(out_dir, "eval_manifest.json"), "w",
              encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    if args.dry_run:
        print("[final_eval_v5] dry run — nothing executed")
        return 0
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        print(f"[final_eval_v5] eval failed rc={rc}", file=sys.stderr)
    return rc


# ─────────────────────────────────────────────────────────────── compare ────

def _load_rows(out_dir: str) -> dict:
    """{sample_id: flattened per-sample row} from final_eval_trajectories.jsonl."""
    path = os.path.join(out_dir, "final_eval_trajectories.jsonl")
    if not os.path.isfile(path):
        raise SystemExit(f"[final_eval_v5] missing {path} — did the run finish?")
    rows = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows[r["sample_id"]] = r
    return rows


def _load_official(out_dir: str) -> dict:
    path = os.path.join(out_dir, "metrics_official.json")
    if not os.path.isfile(path):
        raise SystemExit(f"[final_eval_v5] missing {path}")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _win(row: dict):
    """Per-task win indicator: official Win if the scorer produced it,
    otherwise the internal win replica (Windows / no IBM functions dir)."""
    off = (row.get("_traj") or {}).get("official_win")
    if off is not None:
        return float(bool(off))
    v = row.get("internal_win_rate")
    return None if v is None else float(v)


def _full(row: dict):
    off = (row.get("_traj") or {}).get("official_full_match")
    if off is not None:
        return float(bool(off))
    v = row.get("internal_full_sequence_accuracy")
    return None if v is None else float(v)


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def _bucket(n_calls: int) -> str:
    if n_calls <= 2:
        return "2"
    if n_calls == 3:
        return "3"
    return "4+"


def _diagnostics(rows: dict) -> dict:
    under, unsupported = [], []
    for r in rows.values():
        traj = r.get("_traj") or {}
        npred = traj.get("num_tool_calls")
        ngold = r.get("num_gold_calls")
        if npred is not None and ngold is not None:
            under.append(float(npred < ngold))
        u = r.get("correct_answer_but_unsupported_trace")
        if u is not None:
            unsupported.append(float(bool(u)))
    return {
        "f1_func": _mean([r.get("internal_f1_func") for r in rows.values()]),
        "f1_param": _mean([r.get("internal_f1_param") for r in rows.values()]),
        "executable_rate": _mean(
            [None if (r.get("_traj") or {}).get("executable") is None
             else float(bool((r.get("_traj") or {}).get("executable")))
             for r in rows.values()]),
        "under_calling_rate": _mean(under),
        "unsupported_trace_rate": _mean(unsupported),
        "win_rate_paired_basis": _mean([_win(r) for r in rows.values()]),
        "full_sequence_accuracy_paired_basis":
            _mean([_full(r) for r in rows.values()]),
        "n_tasks": len(rows),
    }


def _by_call_count(rows: dict) -> dict:
    buckets: dict = {}
    for r in rows.values():
        b = _bucket(int(r.get("num_gold_calls") or 0))
        buckets.setdefault(b, []).append(r)
    out = {}
    for b in ("2", "3", "4+"):
        rs = buckets.get(b, [])
        out[b] = {
            "n": len(rs),
            "win_rate": _mean([_win(r) for r in rs]),
            "full_sequence_accuracy": _mean([_full(r) for r in rs]),
            "f1_func": _mean([r.get("internal_f1_func") for r in rs]),
            "f1_param": _mean([r.get("internal_f1_param") for r in rs]),
        }
    return out


def _paired(base: dict, cand: dict) -> dict:
    """Gained/regressed on the per-task win indicator + paired bootstrap CI."""
    common = sorted(set(base) & set(cand))
    pairs = []
    for tid in common:
        b, c = _win(base[tid]), _win(cand[tid])
        if b is None or c is None:
            continue
        pairs.append((tid, b, c))
    gained = [tid for tid, b, c in pairs if c > b]
    regressed = [tid for tid, b, c in pairs if c < b]
    deltas = [c - b for _, b, c in pairs]
    n = len(deltas)
    mean_delta = (sum(deltas) / n) if n else None

    ci = None
    if n:
        rng = random.Random(BOOTSTRAP_SEED)
        boots = []
        for _ in range(BOOTSTRAP_ITERS):
            s = sum(deltas[rng.randrange(n)] for _ in range(n))
            boots.append(s / n)
        boots.sort()
        ci = [boots[int(0.025 * BOOTSTRAP_ITERS)],
              boots[int(0.975 * BOOTSTRAP_ITERS) - 1]]

    return {
        "n_paired": n,
        "n_common_tasks": len(common),
        "win_delta_mean": mean_delta,
        "win_delta_bootstrap_ci95": ci,
        "bootstrap": {"iters": BOOTSTRAP_ITERS, "seed": BOOTSTRAP_SEED,
                      "statistic": "mean per-task win delta (paired resample)"},
        "n_gained": len(gained),
        "n_regressed": len(regressed),
        "gained_task_ids": gained,
        "regressed_task_ids": regressed,
    }


def cmd_compare(args: argparse.Namespace) -> int:
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    arms = {"baseline": os.path.abspath(args.baseline),
            "best": os.path.abspath(args.best)}
    if args.final:
        arms["final"] = os.path.abspath(args.final)

    rows = {label: _load_rows(d) for label, d in arms.items()}
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "arms": arms,
        "aggregate_official": {label: _load_official(d)
                               for label, d in arms.items()},
        "diagnostics": {label: _diagnostics(r) for label, r in rows.items()},
        "by_call_count": {label: _by_call_count(r) for label, r in rows.items()},
        "paired_vs_baseline": {
            label: _paired(rows["baseline"], rows[label])
            for label in arms if label != "baseline"
        },
    }
    out_path = os.path.join(out_dir, "final_compare_report.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    print(f"[final_eval_v5] report -> {out_path}")
    for label in arms:
        off = report["aggregate_official"][label]
        print(f"  {label:9s} win={off.get('win_rate')} "
              f"f1_param={off.get('f1_param')} "
              f"full={off.get('full_sequence_accuracy')}")
    for label, p in report["paired_vs_baseline"].items():
        print(f"  {label} vs baseline: d_win={p['win_delta_mean']} "
              f"CI95={p['win_delta_bootstrap_ci95']} "
              f"gained={p['n_gained']} regressed={p['n_regressed']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="deterministic final eval of one checkpoint")
    r.add_argument("--label", required=True,
                   help="arm name (baseline / best / final)")
    r.add_argument("--checkpoint", default=None,
                   help="LoRA adapter dir; omit for the plain baseline")
    r.add_argument("--out-dir", required=True)
    r.add_argument("--eval-set", default=None,
                   help="NESTFUL jsonl (default: config paths.full_nestful_jsonl)")
    r.add_argument("--max-tasks", type=int, default=0,
                   help="cap tasks (0 = full set; use only for smoke)")
    r.add_argument("--config", default=_DEFAULT_CONFIG)
    r.add_argument("--dry-run", action="store_true")
    r.set_defaults(fn=cmd_run)

    c = sub.add_parser("compare", help="paired comparison of finished runs")
    c.add_argument("--baseline", required=True)
    c.add_argument("--best", required=True)
    c.add_argument("--final", default=None)
    c.add_argument("--out", required=True)
    c.set_defaults(fn=cmd_compare)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
