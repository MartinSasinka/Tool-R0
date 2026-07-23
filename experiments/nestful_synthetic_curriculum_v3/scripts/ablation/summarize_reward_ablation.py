#!/usr/bin/env python3
"""Reward ablation — per-arm eval summary + paired analysis (spec §11).

Reuses the EXISTING eval aggregation helpers from scripts/eval/final_eval_v5.py
(`_load_rows`, `_load_official`, `_diagnostics`, `_by_call_count`, `_paired`)
verbatim — this script does not reimplement scoring, it only reshapes the
already-computed `final_eval_v5.py run` outputs into the ablation's required
per-arm layout plus a cross-arm Round summary, and adds a paired McNemar test
that final_eval_v5.py's `compare` subcommand does not compute.

Per-arm outputs (spec §11), written under --out-dir (default: the eval-dir
itself, i.e. <run_dir>/eval/<arm>/<seed>/):

  task_results.jsonl     one flattened row per task (win / full-seq / f1 / taxonomy)
  metrics.json           official + diagnostics + by_call_count
  metrics.md             human-readable summary
  paired_vs_c0.json      paired win-delta vs the shared C0 baseline eval
  paired_vs_r0.json      paired win-delta vs A0_R0_CURRENT's eval (skipped for A0 itself)
  failure_taxonomy.csv   counts of parse/no-call/unsupported-tool/wrong-tool/... per arm
  bucket_metrics.csv     win/f1 by gold-call-count bucket (2 / 3 / 4+)

Cross-arm round summary (--round-summary mode): aggregates every arm's
metrics.json (+ paired_vs_c0/paired_vs_r0) into
reports/reward_ablation/round<N>/ROUND<N>_SUMMARY.{json,md}.

Usage:
  # one arm's eval-dir -> per-arm deliverables
  python summarize_reward_ablation.py arm \\
      --arm A2_R3_OUTCOME_FIRST --eval-dir <run_dir>/eval/A2_R3_OUTCOME_FIRST/20260724 \\
      --c0-dir <shared_c0_eval_dir> --r0-dir <A0_run_dir>/eval/A0_R0_CURRENT/20260724

  # cross-arm summary once every arm's per-arm deliverables exist
  python summarize_reward_ablation.py round-summary --round 1 \\
      --arm-dir A0_R0_CURRENT=<dir> --arm-dir A1_OUTCOME_ONLY=<dir> ...
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_HERE = Path(__file__).resolve().parent
_V3 = _HERE.parents[1]
if str(_V3) in sys.path:
    sys.path.remove(str(_V3))
sys.path.insert(0, str(_V3))

# final_eval_v5.py is a plain script (not a package module) — load it by path
# so we can reuse its scoring helpers without duplicating scoring logic.
_FEV5_PATH = _V3 / "scripts" / "eval" / "final_eval_v5.py"
_spec = importlib.util.spec_from_file_location("final_eval_v5", _FEV5_PATH)
fev5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fev5)  # type: ignore[union-attr]

REPORTS_DIR = _V3 / "reports" / "reward_ablation"

FAILURE_TAXONOMY_FIELDS = (
    "parse_error", "no_tool_call", "wrong_tool", "unsupported_tool",
    "executable_wrong_final", "invalid_reference", "too_few_calls",
    "too_many_calls", "official_success",
)


def _taxonomy_bucket(row: Dict[str, Any]) -> str:
    traj = row.get("_traj") or {}
    if fev5._win(row) == 1.0:  # noqa: SLF001 — deliberate reuse of final_eval_v5's win predicate
        return "official_success"
    if row.get("parse_error"):
        return "parse_error"
    if traj.get("num_tool_calls") in (0, None):
        return "no_tool_call"
    if row.get("unsupported_tool"):
        return "unsupported_tool"
    if row.get("wrong_tool"):
        return "wrong_tool"
    if row.get("invalid_reference"):
        return "invalid_reference"
    npred, ngold = traj.get("num_tool_calls"), row.get("num_gold_calls")
    if npred is not None and ngold is not None:
        if npred < ngold:
            return "too_few_calls"
        if npred > ngold:
            return "too_many_calls"
    if traj.get("executable"):
        return "executable_wrong_final"
    return "execution_failure"


def _mcnemar(base_rows: Dict[str, Any], cand_rows: Dict[str, Any]) -> Dict[str, Any]:
    """Paired McNemar test (continuity-corrected chi2, no scipy dependency)
    on the binary per-task win indicator, common tasks only."""
    common = sorted(set(base_rows) & set(cand_rows))
    b = c = 0  # b: base=1,cand=0 (regressed); c: base=0,cand=1 (gained)
    for tid in common:
        wb, wc = fev5._win(base_rows[tid]), fev5._win(cand_rows[tid])  # noqa: SLF001
        if wb is None or wc is None:
            continue
        if wb == 1.0 and wc == 0.0:
            b += 1
        elif wb == 0.0 and wc == 1.0:
            c += 1
    n_discordant = b + c
    if n_discordant == 0:
        stat, p = 0.0, 1.0
    else:
        stat = ((abs(b - c) - 1) ** 2) / n_discordant
        p = _chi2_sf_1df(stat)
    return {"n_common": len(common), "b_base_win_cand_loss": b, "c_base_loss_cand_win": c,
            "statistic": stat, "p_value": p, "note": "continuity-corrected 1df chi2, no scipy"}


def _chi2_sf_1df(x: float) -> float:
    """Survival function of chi2(df=1) = erfc(sqrt(x/2)) — closed form, no scipy."""
    import math
    if x <= 0:
        return 1.0
    return math.erfc(math.sqrt(x / 2.0))


def summarize_arm(arm: str, eval_dir: Path, out_dir: Optional[Path], c0_dir: Optional[Path],
                   r0_dir: Optional[Path]) -> Dict[str, Any]:
    out_dir = out_dir or eval_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = fev5._load_rows(str(eval_dir))  # noqa: SLF001
    official = fev5._load_official(str(eval_dir))  # noqa: SLF001
    diagnostics = fev5._diagnostics(rows)  # noqa: SLF001
    by_bucket = fev5._by_call_count(rows)  # noqa: SLF001

    with open(out_dir / "task_results.jsonl", "w", encoding="utf-8") as fh:
        for tid, r in sorted(rows.items()):
            fh.write(json.dumps({
                "task_id": tid,
                "win": fev5._win(r),  # noqa: SLF001
                "full_sequence_accuracy": fev5._full(r),  # noqa: SLF001
                "f1_func": r.get("internal_f1_func"),
                "f1_param": r.get("internal_f1_param"),
                "num_gold_calls": r.get("num_gold_calls"),
                "taxonomy": _taxonomy_bucket(r),
            }, ensure_ascii=False) + "\n")

    metrics = {"arm": arm, "official": official, "diagnostics": diagnostics, "by_call_count": by_bucket}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    tax_counts: Dict[str, int] = {k: 0 for k in FAILURE_TAXONOMY_FIELDS}
    for r in rows.values():
        tax_counts[_taxonomy_bucket(r)] = tax_counts.get(_taxonomy_bucket(r), 0) + 1
    with open(out_dir / "failure_taxonomy.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["taxonomy_class", "count", "rate"])
        n = max(len(rows), 1)
        for k, v in sorted(tax_counts.items(), key=lambda kv: -kv[1]):
            w.writerow([k, v, round(v / n, 6)])

    with open(out_dir / "bucket_metrics.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["bucket", "n", "win_rate", "full_sequence_accuracy", "f1_func", "f1_param"])
        for b, m in by_bucket.items():
            w.writerow([b, m["n"], m["win_rate"], m["full_sequence_accuracy"], m["f1_func"], m["f1_param"]])

    lines = [f"# Reward ablation eval summary — {arm}", "", f"n_tasks: {diagnostics['n_tasks']}", "",
             "## Official metrics", "", "```json", json.dumps(official, indent=2), "```", "",
             "## Diagnostics", "", "```json", json.dumps(diagnostics, indent=2), "```"]

    if c0_dir is not None:
        c0_rows = fev5._load_rows(str(c0_dir))  # noqa: SLF001
        paired_c0 = fev5._paired(c0_rows, rows)  # noqa: SLF001
        paired_c0["mcnemar"] = _mcnemar(c0_rows, rows)
        (out_dir / "paired_vs_c0.json").write_text(
            json.dumps(paired_c0, indent=2, ensure_ascii=False), encoding="utf-8")
        lines += ["", "## Paired vs C0", "", "```json", json.dumps(paired_c0, indent=2), "```"]

    if r0_dir is not None and arm != "A0_R0_CURRENT":
        r0_rows = fev5._load_rows(str(r0_dir))  # noqa: SLF001
        paired_r0 = fev5._paired(r0_rows, rows)  # noqa: SLF001
        paired_r0["mcnemar"] = _mcnemar(r0_rows, rows)
        (out_dir / "paired_vs_r0.json").write_text(
            json.dumps(paired_r0, indent=2, ensure_ascii=False), encoding="utf-8")
        lines += ["", "## Paired vs A0_R0_CURRENT", "", "```json", json.dumps(paired_r0, indent=2), "```"]

    (out_dir / "metrics.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[summarize_reward_ablation] {arm} -> {out_dir}")
    return metrics


def round_summary(round_: int, arm_dirs: Dict[str, Path]) -> Dict[str, Any]:
    out_dir = REPORTS_DIR / f"round{round_}"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_arm = {}
    for arm, d in arm_dirs.items():
        mp = d / "metrics.json"
        if not mp.is_file():
            print(f"[summarize_reward_ablation] WARNING: {mp} missing, skipping {arm}")
            continue
        entry: Dict[str, Any] = {"metrics": json.loads(mp.read_text(encoding="utf-8"))}
        for extra in ("paired_vs_c0", "paired_vs_r0"):
            ep = d / f"{extra}.json"
            if ep.is_file():
                entry[extra] = json.loads(ep.read_text(encoding="utf-8"))
        per_arm[arm] = entry
    summary = {"round": round_, "arms": per_arm}
    (out_dir / f"ROUND{round_}_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [f"# Round {round_} — cross-arm summary", "",
             "| Arm | official win_rate | full_seq_acc | f1_param | gained vs C0 | regressed vs C0 |",
             "|---|---:|---:|---:|---:|---:|"]
    for arm, entry in per_arm.items():
        off = entry["metrics"]["official"]
        pc0 = entry.get("paired_vs_c0", {})
        lines.append(f"| {arm} | {off.get('win_rate')} | {off.get('full_sequence_accuracy')} | "
                     f"{off.get('f1_param')} | {pc0.get('n_gained')} | {pc0.get('n_regressed')} |")
    (out_dir / f"ROUND{round_}_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[summarize_reward_ablation] round summary -> {out_dir}")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("arm")
    a.add_argument("--arm", required=True)
    a.add_argument("--eval-dir", type=Path, required=True)
    a.add_argument("--out-dir", type=Path, default=None)
    a.add_argument("--c0-dir", type=Path, default=None)
    a.add_argument("--r0-dir", type=Path, default=None)

    r = sub.add_parser("round-summary")
    r.add_argument("--round", type=int, required=True)
    r.add_argument("--arm-dir", action="append", required=True,
                   help="ARM_ID=path/to/per-arm/out-dir, repeatable")

    args = ap.parse_args()
    if args.cmd == "arm":
        summarize_arm(args.arm, args.eval_dir, args.out_dir, args.c0_dir, args.r0_dir)
    else:
        arm_dirs = {}
        for kv in args.arm_dir:
            k, _, v = kv.partition("=")
            arm_dirs[k] = Path(v)
        round_summary(args.round, arm_dirs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
