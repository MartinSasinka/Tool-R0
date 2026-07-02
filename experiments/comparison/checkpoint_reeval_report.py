#!/usr/bin/env python3
"""Build CHECKPOINT_REEVAL_REPORT.md from final_eval_v2 cell outputs.

Scans every cell subdir under --eval-root (each produced by
`run.py --mode final_eval`), reads:
  - metrics_official.json  (canonical NESTFUL: win_rate, full/partial acc, F1)
  - final_eval_trajectories.jsonl  (per-sample: official_win, num_tool_calls,
    parse_valid, clipped_any, fail reasons)

and emits a single markdown report with, per checkpoint:
  official Win Rate, Full Acc, Partial Acc, F1 Func / F1 Param (diagnostic),
  final_answer_pass, strict_gold_trace_pass, zero_tool_calls,
  clipped_completion_rate, parse_error_rate, invalid_reference_rate,
  avg number of calls, and a win/loss overlap vs the baseline cell.

Offline only (no model, no GPU). Safe to re-run.

Usage:
  python experiments/comparison/checkpoint_reeval_report.py \
      --eval-root experiments/nestful_mtgrpo_partial/outputs/final_eval_v2
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

# Preferred display order; any other cells are appended alphabetically.
_PREFERRED_ORDER = [
    "baseline", "best_react_win", "stage1_e3",
    "stage2_e2", "stage2_e3", "stage2_e4", "stage3_e1", "stage3_e2",
]


def _load_json(path: str) -> Optional[dict]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None


def _iter_jsonl(path: str):
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _fail_reasons(traj: dict) -> List[str]:
    out: List[str] = []
    for t in traj.get("turns", []) or []:
        fr = t.get("fail_reason")
        if fr:
            out.append(str(fr).lower())
    ee = traj.get("execution_error")
    if ee:
        out.append(str(ee).lower())
    return out


def _analyse_trajectories(path: str) -> Dict[str, Any]:
    """Per-sample aggregates from final_eval_trajectories.jsonl."""
    n = 0
    wins: Dict[str, float] = {}
    zero_calls = 0
    clipped = 0
    parse_err = 0
    invalid_ref = 0
    total_calls = 0
    strict_pass = 0
    final_pass = 0
    for row in _iter_jsonl(path):
        n += 1
        sid = str(row.get("sample_id") or "")
        traj = row.get("_traj", {}) or {}
        ow = traj.get("official_win")
        if ow is None:
            ow = row.get("internal_win_rate")
        if sid and ow is not None:
            wins[sid] = float(ow)
        ncalls = traj.get("num_tool_calls")
        if isinstance(ncalls, (int, float)):
            total_calls += ncalls
            if ncalls == 0:
                zero_calls += 1
        if traj.get("clipped_any"):
            clipped += 1
        if traj.get("parse_valid") is False:
            parse_err += 1
        reasons = _fail_reasons(traj)
        if any(("reference" in r or "$var" in r or "unresolved" in r) for r in reasons):
            invalid_ref += 1
        if row.get("strict_gold_trace_pass") is True:
            strict_pass += 1
        if row.get("final_answer_pass") is True:
            final_pass += 1
    if n == 0:
        return {"n": 0, "wins": {}}
    return {
        "n": n,
        "wins": wins,
        "zero_tool_calls": zero_calls / n,
        "clipped_completion_rate": clipped / n,
        "parse_error_rate": parse_err / n,
        "invalid_reference_rate": invalid_ref / n,
        "avg_num_calls": total_calls / n,
        "strict_gold_trace_pass": strict_pass / n,
        "final_answer_pass": final_pass / n,
    }


def _overlap(ckpt_wins: Dict[str, float], base_wins: Dict[str, float]) -> Dict[str, int]:
    """Win/loss overlap vs baseline over the shared sample ids."""
    shared = set(ckpt_wins) & set(base_wins)
    both_win = both_lose = gained = regressed = 0
    for sid in shared:
        cw = ckpt_wins[sid] >= 0.5
        bw = base_wins[sid] >= 0.5
        if cw and bw:
            both_win += 1
        elif not cw and not bw:
            both_lose += 1
        elif cw and not bw:
            gained += 1
        else:
            regressed += 1
    return {
        "shared": len(shared),
        "both_win": both_win,
        "both_lose": both_lose,
        "gained": gained,        # ckpt wins, baseline loses
        "regressed": regressed,  # ckpt loses, baseline wins
        "net": gained - regressed,
    }


def _fmt(v: Any, pct: bool = False) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        return f"{v*100:.1f}%" if pct else f"{v:.3f}"
    return str(v)


def _discover_cells(eval_root: str) -> List[str]:
    names = [
        d for d in os.listdir(eval_root)
        if os.path.isdir(os.path.join(eval_root, d))
        and os.path.isfile(os.path.join(eval_root, d, "metrics_official.json"))
    ]
    ordered = [c for c in _PREFERRED_ORDER if c in names]
    ordered += sorted(c for c in names if c not in _PREFERRED_ORDER)
    return ordered


def build_report(eval_root: str, baseline_cell: str = "baseline") -> str:
    cells = _discover_cells(eval_root)
    if not cells:
        return f"# CHECKPOINT RE-EVAL REPORT\n\nNo scored cells found under `{eval_root}`.\n"

    data: Dict[str, Dict[str, Any]] = {}
    for cell in cells:
        cdir = os.path.join(eval_root, cell)
        official = _load_json(os.path.join(cdir, "metrics_official.json")) or {}
        traj = _analyse_trajectories(os.path.join(cdir, "final_eval_trajectories.jsonl"))
        data[cell] = {"official": official, "traj": traj}

    base_wins = data.get(baseline_cell, {}).get("traj", {}).get("wins", {})
    base_win_rate = data.get(baseline_cell, {}).get("official", {}).get("win_rate")

    lines: List[str] = []
    lines.append("# CHECKPOINT RE-EVAL REPORT")
    lines.append("")
    lines.append(f"Eval root: `{eval_root}`  |  paradigm: ReAct  |  scorer: official NESTFUL")
    if base_win_rate is not None:
        lines.append(f"Baseline (`{baseline_cell}`) official Win Rate = **{base_win_rate:.3f}**.")
    lines.append("")

    # ---- Headline table (official + key diagnostics) ----
    lines.append("## Headline metrics")
    lines.append("")
    header = ("| checkpoint | Win Rate | Full Acc | Partial Acc | F1 Func* | F1 Param* "
              "| final_ans | strict_trace | zero_calls | clipped | parse_err "
              "| inv_ref | avg_calls | vs base |")
    sep = "|" + "|".join(["---"] * 13) + "|"
    lines.append(header)
    lines.append(sep)
    for cell in cells:
        off = data[cell]["official"]
        tr = data[cell]["traj"]
        wr = off.get("win_rate")
        delta = ""
        if wr is not None and base_win_rate is not None and cell != baseline_cell:
            d = wr - base_win_rate
            delta = f"{d:+.3f}"
        elif cell == baseline_cell:
            delta = "(base)"
        row = "| {name} | {wr} | {full} | {part} | {f1f} | {f1p} | {fa} | {st} | {zc} | {cl} | {pe} | {ir} | {ac} | {dl} |".format(
            name=cell,
            wr=_fmt(wr),
            full=_fmt(off.get("full_sequence_accuracy")),
            part=_fmt(off.get("partial_sequence_accuracy")),
            f1f=_fmt(off.get("f1_func")),
            f1p=_fmt(off.get("f1_param")),
            fa=_fmt(tr.get("final_answer_pass")),
            st=_fmt(tr.get("strict_gold_trace_pass")),
            zc=_fmt(tr.get("zero_tool_calls")),
            cl=_fmt(tr.get("clipped_completion_rate")),
            pe=_fmt(tr.get("parse_error_rate")),
            ir=_fmt(tr.get("invalid_reference_rate")),
            ac=_fmt(tr.get("avg_num_calls")),
            dl=delta,
        )
        lines.append(row)
    lines.append("")
    lines.append("\\* F1 Func / F1 Param are corpus-level diagnostics only (NESTFUL "
                 "headline is Win Rate + Full/Partial Acc).")
    lines.append("")

    # ---- Win/loss overlap vs baseline ----
    lines.append("## Win/loss overlap vs baseline")
    lines.append("")
    lines.append("| checkpoint | shared | both_win | both_lose | gained | regressed | net |")
    lines.append("|" + "|".join(["---"] * 7) + "|")
    for cell in cells:
        if cell == baseline_cell:
            continue
        ov = _overlap(data[cell]["traj"].get("wins", {}), base_wins)
        lines.append(
            f"| {cell} | {ov['shared']} | {ov['both_win']} | {ov['both_lose']} | "
            f"{ov['gained']} | {ov['regressed']} | {ov['net']:+d} |"
        )
    lines.append("")
    lines.append("gained = checkpoint wins where baseline loses; regressed = checkpoint "
                 "loses where baseline wins; net = gained - regressed (negative = worse "
                 "than baseline).")
    lines.append("")

    # ---- Verdict ----
    beat = [c for c in cells if c != baseline_cell
            and data[c]["official"].get("win_rate") is not None
            and base_win_rate is not None
            and data[c]["official"]["win_rate"] > base_win_rate]
    lines.append("## Verdict")
    lines.append("")
    if not base_win_rate:
        lines.append("- Baseline Win Rate unavailable — cannot rank.")
    elif beat:
        lines.append(f"- Checkpoints ABOVE baseline Win Rate: {', '.join(beat)}.")
    else:
        lines.append(f"- No checkpoint beats baseline Win Rate ({base_win_rate:.3f}). "
                     "Training degraded the model; do NOT report a trained checkpoint "
                     "as the headline result.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build CHECKPOINT_REEVAL_REPORT.md")
    ap.add_argument("--eval-root", required=True,
                    help="dir with per-cell subdirs (final_eval_v2)")
    ap.add_argument("--baseline-cell", default="baseline")
    ap.add_argument("--out", default=None,
                    help="output markdown path (default: <eval-root>/CHECKPOINT_REEVAL_REPORT.md)")
    args = ap.parse_args()

    report = build_report(args.eval_root, args.baseline_cell)
    out = args.out or os.path.join(args.eval_root, "CHECKPOINT_REEVAL_REPORT.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(report)
    print(f"[checkpoint_reeval_report] wrote {out}")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
