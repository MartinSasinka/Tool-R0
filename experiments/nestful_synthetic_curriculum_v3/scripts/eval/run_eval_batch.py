"""Deterministic NESTFUL eval batch runner (P0 remediation).

Guarantees that every comparison is valid by construction:
- a ``baseline`` cell is MANDATORY in every batch (exit 2 without it; explicit
  ``--allow-no-baseline`` is the only escape hatch and is stamped into the report);
- temperature 0.0 by default; all decoding settings recorded in the manifest;
- the official NESTFUL scorer output (``metrics_official.json``) must exist for every
  cell or the batch fails (exit 4);
- legacy dataset B (``filtered_toolr0_synthetic``) is refused (exit 3) unless
  ``--allow-legacy-dataset``;
- one flat batch directory ``<output-root>/<batch_id>/<cell>/`` — absolute paths are
  passed to run.py so its ``_resolve_path`` re-rooting can never double-nest dirs;
- per-cell ``metrics_unified.json`` and a batch-level ``BATCH_REPORT.md`` with paired
  gained/regressed counts and binomial CIs from per-sample official wins.

Wraps ``experiments/nestful_mtgrpo_minimal/run.py --mode final_eval`` via subprocess;
no trainer/eval code is modified.

Examples (run from repo root):
  # dry run (no GPU, prints the exact commands)
  python .../run_eval_batch.py --cells "baseline,s3_e1=<adapter_dir>" --dataset nestful_test --dry-run

  # tiny smoke batch
  python .../run_eval_batch.py --cells "baseline,s3_e1=<adapter_dir>" --dataset nestful_test --max-tasks 5

  # regenerate report/unified metrics from an already-run batch directory
  python .../run_eval_batch.py --report-only <batch_dir> --baseline-cell baseline
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "lib"))

from paths import (  # noqa: E402
    EVAL_OUTPUT_ROOT, MINIMAL_ROOT, NESTFUL_DATASETS, REPO_ROOT,
    dataset_info, is_legacy_dataset_path,
)
from metrics_schema import (  # noqa: E402
    binomial_ci95, build_from_cell_dir, load_per_sample_official_wins, paired_counts,
)
from run_manifest import build_manifest, write_manifest  # noqa: E402

DEFAULT_CONFIG = os.path.join(REPO_ROOT, "experiments", "nestful_mtgrpo_partial", "config.yaml")
BASELINE_CELL = "baseline"

EXIT_NO_BASELINE = 2
EXIT_LEGACY_DATASET = 3
EXIT_MISSING_OFFICIAL = 4
EXIT_CELL_FAILED = 5


# ---------------------------------------------------------------------------
# cell spec parsing
# ---------------------------------------------------------------------------

def parse_cells(spec: str) -> List[Dict[str, Optional[str]]]:
    """'baseline,s3_e1=/path/a,s3_e2=/path/b' -> [{name, checkpoint}, ...]."""
    cells: List[Dict[str, Optional[str]]] = []
    for part in [p.strip() for p in spec.split(",") if p.strip()]:
        if "=" in part:
            name, ckpt = part.split("=", 1)
            cells.append({"name": name.strip(), "checkpoint": os.path.abspath(ckpt.strip())})
        else:
            cells.append({"name": part, "checkpoint": None})
    names = [c["name"] for c in cells]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate cell names in --cells: {names}")
    return cells


def resolve_dataset(name_or_path: str) -> str:
    if name_or_path in NESTFUL_DATASETS:
        return NESTFUL_DATASETS[name_or_path]
    return os.path.abspath(name_or_path)


# ---------------------------------------------------------------------------
# per-cell command
# ---------------------------------------------------------------------------

def build_cell_command(
    *, cell_dir: str, checkpoint: Optional[str], config: str, dataset_path: str,
    temperature: float, top_p: float, seed: int, use_vllm: bool,
    max_tasks: Optional[int],
) -> List[str]:
    cmd = [
        sys.executable, os.path.join(MINIMAL_ROOT, "run.py"),
        "--mode", "final_eval",
        "--config", os.path.abspath(config),
    ]
    if checkpoint:
        cmd += ["--checkpoint", checkpoint]
    overrides = [
        f"hardware.use_vllm={'true' if use_vllm else 'false'}",
        "data.eval_paradigm=react",
        f"generation.temperature={temperature}",
        f"generation.top_p={top_p}",
        f"experiment.seed={seed}",
        # absolute paths: run.py's _resolve_path passes absolutes through untouched,
        # which is what prevents the historical double-nested output dirs
        f"paths.full_nestful_jsonl={os.path.abspath(dataset_path)}",
        f"experiment.output_dir={os.path.abspath(cell_dir)}",
    ]
    if max_tasks:
        overrides.append(f"data.max_eval_tasks={max_tasks}")
    for ov in overrides:
        cmd += ["--override", ov]
    return cmd


def run_cell(cmd: List[str], log_path: str) -> int:
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace")
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        return proc.wait()


# ---------------------------------------------------------------------------
# diagnostics recomputed from trajectories (call-count behavior)
# ---------------------------------------------------------------------------

def call_count_stats(trajectories_path: str) -> Dict[str, Optional[float]]:
    n = 0
    too_few = 0
    no_call = 0
    parse_err = 0
    total_calls = 0
    with open(trajectories_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            traj = row.get("_traj") or {}
            gold = row.get("num_gold_calls") or traj.get("gold_num_turns")
            pred = traj.get("num_tool_calls")
            if pred is None:
                continue
            n += 1
            total_calls += pred
            if pred == 0:
                no_call += 1
            if gold is not None and pred < gold:
                too_few += 1
            if traj.get("parse_valid") is False:
                parse_err += 1
    if n == 0:
        return {}
    return {
        "too_few_calls_rate": round(too_few / n, 4),
        "avg_predicted_calls": round(total_calls / n, 4),
        "no_tool_call_rate": round(no_call / n, 4),
        "parse_error_rate": round(parse_err / n, 4),
    }


# ---------------------------------------------------------------------------
# unified metrics + report over a finished batch dir
# ---------------------------------------------------------------------------

def finalize_batch(
    batch_dir: str,
    cells: List[Dict[str, Optional[str]]],
    *,
    baseline_cell: str,
    dataset: Optional[Dict[str, Any]],
    decoding: Optional[Dict[str, Any]],
    strict_official: bool = True,
) -> int:
    """Verify official metrics, write metrics_unified.json per cell, emit BATCH_REPORT.md."""
    baseline_dir = os.path.join(batch_dir, baseline_cell)
    have_baseline = os.path.isdir(baseline_dir)

    # 1. official scorer output is mandatory per cell
    missing = [c["name"] for c in cells
               if not os.path.isfile(os.path.join(batch_dir, c["name"], "metrics_official.json"))]
    if missing:
        print(f"[eval-batch] ERROR: metrics_official.json missing for cells: {missing}. "
              "The official scorer must run for every cell (check the IBM executable-functions "
              "dir and that the cell ran on Linux — run.py skips official win on Windows).",
              file=sys.stderr)
        if strict_official:
            return EXIT_MISSING_OFFICIAL

    # 2. unified metrics per cell
    unified_by_cell: Dict[str, Dict[str, Any]] = {}
    for c in cells:
        name = c["name"]
        cell_dir = os.path.join(batch_dir, name)
        if not os.path.isfile(os.path.join(cell_dir, "metrics_official.json")):
            continue
        unified = build_from_cell_dir(
            cell_dir, cell=name, checkpoint=c.get("checkpoint"),
            dataset=dataset, decoding=decoding,
            baseline_dir=baseline_dir if (have_baseline and name != baseline_cell) else None,
        )
        # call-count behavior is not aggregated by run.py's metrics.json;
        # recompute it from the per-sample trajectories and merge into diagnostics
        traj_path = os.path.join(cell_dir, "final_eval_trajectories.jsonl")
        extra_diag = call_count_stats(traj_path) if os.path.isfile(traj_path) else {}
        for k, v in extra_diag.items():
            unified["diagnostics"][k] = v
        out = os.path.join(cell_dir, "metrics_unified.json")
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(unified, fh, indent=2, ensure_ascii=False)
        unified_by_cell[name] = unified
        print(f"[eval-batch] wrote {os.path.relpath(out, REPO_ROOT)}")

    # 3. batch report
    report_path = os.path.join(batch_dir, "BATCH_REPORT.md")
    lines = [
        f"# BATCH REPORT — {os.path.basename(os.path.normpath(batch_dir))}",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat()} by run_eval_batch.py.",
        "",
    ]
    if dataset:
        lines += [f"Dataset: `{dataset.get('path')}` (n={dataset.get('n_rows')}, "
                  f"sha256={str(dataset.get('sha256'))[:12]}…)", ""]
    if decoding:
        lines += [f"Decoding: temperature={decoding.get('temperature')}, "
                  f"top_p={decoding.get('top_p')}, seed={decoding.get('seed')}", ""]
    if not have_baseline:
        lines += ["**WARNING: no baseline cell in this batch (--allow-no-baseline was used). "
                  "Numbers below are NOT comparable to anything.**", ""]
    lines += [
        "Primary metric: `official_nestful_win_rate` (official NESTFUL scorer). "
        "`internal_final_answer_win` is a diagnostic that historically inflates the "
        "official number by ~6-7 pp — never headline it.",
        "",
        "| cell | official win | 95% CI | paired +/- vs baseline (net) | internal win (diag) | "
        "too_few_calls | avg calls | no_call | parse_err |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for c in cells:
        name = c["name"]
        u = unified_by_cell.get(name)
        if not u:
            lines.append(f"| {name} | MISSING OFFICIAL SCORE | — | — | — | — | — | — | — |")
            continue
        p = u["primary"]
        d = u["diagnostics"]
        ci = f"[{p['ci95'][0]:.3f}, {p['ci95'][1]:.3f}]" if p.get("ci95") else "—"
        pv = u.get("paired_vs_baseline")
        paired = (f"+{pv['gained']} / -{pv['regressed']} ({pv['net']:+d})"
                  if pv else ("—" if name == baseline_cell else "n/a"))
        def fmt(x):
            return f"{x:.4f}" if isinstance(x, (int, float)) else "—"
        lines.append(
            f"| {name} | {fmt(p['official_nestful_win_rate'])} | {ci} | {paired} | "
            f"{fmt(d.get('internal_final_answer_win'))} | {fmt(d.get('too_few_calls_rate'))} | "
            f"{fmt(d.get('avg_predicted_calls'))} | {fmt(d.get('no_tool_call_rate'))} | "
            f"{fmt(d.get('parse_error_rate'))} |")
    lines += [
        "",
        "Interpretation rules: an improvement claim requires the baseline row in THIS table, "
        "overlapping-CI caution, and a positive paired net with gained+regressed large enough "
        "to matter. Cross-batch comparisons are invalid.",
        "",
    ]
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"[eval-batch] wrote {os.path.relpath(report_path, REPO_ROOT)}")
    return 0 if (not missing or not strict_official) else EXIT_MISSING_OFFICIAL


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Deterministic NESTFUL eval batch runner (same-batch baseline mandatory).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--cells",
                    help="comma list: 'baseline' and/or 'name=checkpoint_dir' entries, e.g. "
                         "'baseline,s3_e1=outputs/runs/X/stage_3/checkpoints/adapter_epoch_1'")
    ap.add_argument("--dataset", default="nestful_test",
                    help="nestful_test | nestful_full | nestful_dev | path to a JSONL")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-vllm", action="store_true", help="disable vLLM backend (HF generate)")
    ap.add_argument("--max-tasks", type=int, default=None, help="smoke-test task cap")
    ap.add_argument("--batch-name", default="eval_batch")
    ap.add_argument("--output-root", default=EVAL_OUTPUT_ROOT)
    ap.add_argument("--allow-no-baseline", action="store_true",
                    help="EXPLICIT escape hatch; the report is stamped as non-comparable")
    ap.add_argument("--allow-legacy-dataset", action="store_true",
                    help="EXPLICIT escape hatch for legacy dataset B paths")
    ap.add_argument("--dry-run", action="store_true", help="print resolved commands, run nothing")
    ap.add_argument("--report-only", metavar="BATCH_DIR",
                    help="skip execution; (re)build unified metrics + BATCH_REPORT.md from an "
                         "existing batch dir (cells inferred from subdirectories)")
    ap.add_argument("--baseline-cell", default=BASELINE_CELL,
                    help="name of the baseline cell (report-only mode and paired counts)")
    args = ap.parse_args()

    # ---- report-only over an existing directory ---------------------------
    if args.report_only:
        batch_dir = os.path.abspath(args.report_only)
        cells = [{"name": d, "checkpoint": None}
                 for d in sorted(os.listdir(batch_dir))
                 if os.path.isdir(os.path.join(batch_dir, d))]
        if not cells:
            print(f"[eval-batch] no cell subdirectories in {batch_dir}", file=sys.stderr)
            return 1
        print(f"[eval-batch] report-only over {batch_dir}: cells = {[c['name'] for c in cells]}")
        if args.baseline_cell not in [c["name"] for c in cells] and not args.allow_no_baseline:
            print(f"[eval-batch] ERROR: baseline cell '{args.baseline_cell}' not found in batch. "
                  "Paired comparisons are impossible; pass --allow-no-baseline to proceed "
                  "with a non-comparable report.", file=sys.stderr)
            return EXIT_NO_BASELINE
        return finalize_batch(batch_dir, cells, baseline_cell=args.baseline_cell,
                              dataset=None, decoding=None, strict_official=False)

    # ---- normal mode -------------------------------------------------------
    if not args.cells:
        ap.error("--cells is required (or use --report-only)")
    cells = parse_cells(args.cells)

    if BASELINE_CELL not in [c["name"] for c in cells]:
        if not args.allow_no_baseline:
            print("[eval-batch] ERROR: no 'baseline' cell in --cells. Every batch must include "
                  "the base model so comparisons are same-batch. Add 'baseline' to --cells or "
                  "pass --allow-no-baseline (the report will be stamped non-comparable).",
                  file=sys.stderr)
            return EXIT_NO_BASELINE
        print("[eval-batch] WARNING: proceeding WITHOUT a baseline cell (--allow-no-baseline).")

    dataset_path = resolve_dataset(args.dataset)
    if not os.path.isfile(dataset_path):
        print(f"[eval-batch] ERROR: dataset not found: {dataset_path}", file=sys.stderr)
        return 1
    if is_legacy_dataset_path(dataset_path) and not args.allow_legacy_dataset:
        print(f"[eval-batch] ERROR: '{dataset_path}' is in the LEGACY dataset-B tree "
              "(filtered_toolr0_synthetic). It must not be used silently "
              "(audits/DATASET_AUDIT.md). Pass --allow-legacy-dataset to override.",
              file=sys.stderr)
        return EXIT_LEGACY_DATASET
    for c in cells:
        if c["checkpoint"] and not os.path.isdir(c["checkpoint"]):
            print(f"[eval-batch] ERROR: checkpoint dir for cell '{c['name']}' not found: "
                  f"{c['checkpoint']}", file=sys.stderr)
            return 1

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    temp_tag = str(args.temperature).replace(".", "p")
    batch_id = f"{args.batch_name}_{ts}_temp{temp_tag}"
    batch_dir = os.path.abspath(os.path.join(args.output_root, batch_id))

    decoding = {"temperature": args.temperature, "top_p": args.top_p,
                "seed": args.seed, "backend": "hf" if args.no_vllm else "vllm"}

    print(f"[eval-batch] repo root      : {REPO_ROOT}")
    print(f"[eval-batch] batch dir      : {batch_dir}")
    print(f"[eval-batch] dataset        : {dataset_path}")
    print(f"[eval-batch] config         : {os.path.abspath(args.config)}")
    print(f"[eval-batch] decoding       : {decoding}")
    print(f"[eval-batch] cells          : {[c['name'] for c in cells]}")
    if args.max_tasks:
        print(f"[eval-batch] max tasks      : {args.max_tasks} (SMOKE — not reportable)")

    commands = []
    for c in cells:
        cell_dir = os.path.join(batch_dir, c["name"])
        cmd = build_cell_command(
            cell_dir=cell_dir, checkpoint=c["checkpoint"], config=args.config,
            dataset_path=dataset_path, temperature=args.temperature, top_p=args.top_p,
            seed=args.seed, use_vllm=not args.no_vllm, max_tasks=args.max_tasks)
        commands.append((c, cell_dir, cmd))

    if args.dry_run:
        print("[eval-batch] DRY RUN — commands that would execute (in order):")
        for c, cell_dir, cmd in commands:
            print(f"\n# cell: {c['name']}  ->  {cell_dir}")
            print("  " + " ".join(shlex.quote(x) for x in cmd))
        print("\n[eval-batch] dry run complete; nothing executed.")
        return 0

    os.makedirs(batch_dir, exist_ok=True)
    manifest = build_manifest(
        kind="eval_batch",
        config_path=os.path.abspath(args.config),
        overrides=[f"cells={args.cells}", f"max_tasks={args.max_tasks}"],
        datasets=[dataset_path],
        seed=args.seed,
        decoding=decoding,
        extra={"batch_id": batch_id,
               "cells": [{"name": c["name"], "checkpoint": c["checkpoint"]} for c in cells],
               "smoke": bool(args.max_tasks)},
    )
    write_manifest(manifest, os.path.join(batch_dir, "manifest.json"))
    print(f"[eval-batch] manifest       : {os.path.join(batch_dir, 'manifest.json')}")

    for c, cell_dir, cmd in commands:
        os.makedirs(cell_dir, exist_ok=True)
        log_path = os.path.join(batch_dir, f"{c['name']}.log")
        print(f"\n[eval-batch] === cell {c['name']} ===")
        rc = run_cell(cmd, log_path)
        if rc != 0:
            print(f"[eval-batch] ERROR: cell '{c['name']}' exited with {rc} "
                  f"(log: {log_path})", file=sys.stderr)
            return EXIT_CELL_FAILED
        if not os.path.isfile(os.path.join(cell_dir, "metrics_official.json")):
            print(f"[eval-batch] ERROR: cell '{c['name']}' finished but produced no "
                  "metrics_official.json — official scorer did not run. Batch is invalid.",
                  file=sys.stderr)
            return EXIT_MISSING_OFFICIAL

    ds_info = dataset_info(dataset_path)
    return finalize_batch(batch_dir, cells, baseline_cell=args.baseline_cell,
                          dataset=ds_info, decoding=decoding)


if __name__ == "__main__":
    sys.exit(main())
