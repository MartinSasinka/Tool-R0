#!/usr/bin/env python3
"""Analyze curriculum multiturn prediction JSONL files.

Computes failure breakdowns, semantic rescue rates, tool-call vs depth
distributions, and curriculum-vs-baseline error-mix shifts.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "nestful_evaluation"))

from nestful_evaluation.run import (  # noqa: E402
    _NUMBER_IN_TEXT_RE,
    _extract_numeric,
    _matches_gold,
    coerce_numeric,
)
from vizualisation.scripts.lib.trajectory_metrics import compute_dependency_depth  # noqa: E402

DEFAULT_RESULTS_DIR = _REPO / "curricullum" / "evaluation" / "results"

PROFILES: Dict[str, str] = {
    "baseline": "curriculum_baseline_multiturn_predictions.jsonl",
    "stage1": "curriculum_stage1_1call_multiturn_predictions.jsonl",
    "stage2": "curriculum_stage2_2call_multiturn_predictions.jsonl",
    "stage3": "curriculum_stage3_3call_multiturn_predictions.jsonl",
}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def pct(n: int, total: int) -> float:
    return 100.0 * n / total if total else 0.0


def fmt_dist(counter: Counter, total: int, top: Optional[int] = None) -> List[Tuple[str, int, float]]:
    items = counter.most_common(top)
    return [(k, v, pct(v, total)) for k, v in items]


def join_completions(row: Dict[str, Any]) -> str:
    raw = row.get("raw_completions") or []
    if isinstance(raw, list):
        return "\n---\n".join(str(x) for x in raw)
    return str(raw)


def any_gold_in_text(text: str, gold: Any) -> bool:
    if gold is None:
        return False
    for m in _NUMBER_IN_TEXT_RE.findall(text or ""):
        if _matches_gold(coerce_numeric(m), gold):
            return True
    gold_s = str(gold).strip()
    if gold_s and gold_s in (text or ""):
        return True
    return False


def gold_dependency_depth(row: Dict[str, Any]) -> int:
    return compute_dependency_depth(row.get("gold_calls") or [])


def num_predicted_calls(row: Dict[str, Any]) -> int:
    n = row.get("num_tool_calls")
    if n is not None:
        return int(n)
    return len(row.get("predicted_calls") or [])


def num_gold_calls(row: Dict[str, Any]) -> int:
    return len(row.get("gold_calls") or [])


def is_fail(row: Dict[str, Any]) -> bool:
    return (row.get("score") or 0) < 1 or row.get("verdict") == "fail"


def print_section(title: str) -> None:
    print(f"\n{'=' * 72}")
    print(title)
    print("=" * 72)


def print_counter_table(
    label: str,
    counter: Counter,
    total: int,
    *,
    top: Optional[int] = None,
    indent: int = 2,
) -> None:
    pad = " " * indent
    print(f"{pad}{label} (n={total})")
    for key, count, share in fmt_dist(counter, total, top=top):
        print(f"{pad}  {key!s:32} {count:6d}  {share:5.1f}%")


def analyze_profile(rows: List[Dict[str, Any]], name: str) -> Dict[str, Any]:
    n = len(rows)
    fails = [r for r in rows if is_fail(r)]
    passes = [r for r in rows if not is_fail(r)]
    nf = len(fails)

    verdict_reason = Counter(r.get("verdict_reason") or "unknown" for r in fails)
    stopped = Counter(r.get("stopped") or "unknown" for r in fails)
    error_category = Counter(r.get("error_category") or "unknown" for r in fails)
    num_tool_calls_fail = Counter(num_predicted_calls(r) for r in fails)
    num_tool_calls_all = Counter(num_predicted_calls(r) for r in rows)

    # Failures: gold in raw_completions but predicted_final wrong
    gold_in_raw_wrong_pred = 0
    gold_in_raw_any_completion = 0
    pred_wrong = 0
    for r in fails:
        gold = r.get("gold_answer")
        pred = r.get("predicted_final")
        text = join_completions(r)
        pred_ok = _matches_gold(pred, gold)
        raw_ok = any_gold_in_text(text, gold)
        if not pred_ok:
            pred_wrong += 1
            if raw_ok:
                gold_in_raw_wrong_pred += 1
        if raw_ok:
            gold_in_raw_any_completion += 1

    # num_tool_calls vs gold dependency depth (all rows + fails only)
    depth_call_cross: Dict[Tuple[int, int], int] = defaultdict(int)
    depth_call_cross_fail: Dict[Tuple[int, int], int] = defaultdict(int)
    depth_dist = Counter()
    for r in rows:
        depth = gold_dependency_depth(r)
        calls = num_predicted_calls(r)
        depth_dist[depth] += 1
        depth_call_cross[(depth, calls)] += 1
        if is_fail(r):
            depth_call_cross_fail[(depth, calls)] += 1

    # wrong_call_count / empty predicted_calls
    empty_predicted = 0
    wrong_call_count = 0
    under_call = 0
    over_call = 0
    exact_call_count_pass = 0
    for r in rows:
        pred_n = num_predicted_calls(r)
        gold_n = num_gold_calls(r)
        pred_calls_empty = len(r.get("predicted_calls") or []) == 0 and pred_n == 0
        if pred_calls_empty:
            empty_predicted += 1
        if pred_n != gold_n:
            wrong_call_count += 1
            if pred_n < gold_n:
                under_call += 1
            elif pred_n > gold_n:
                over_call += 1
        elif not is_fail(r):
            exact_call_count_pass += 1

    empty_on_fail = sum(
        1
        for r in fails
        if len(r.get("predicted_calls") or []) == 0 and num_predicted_calls(r) == 0
    )

    return {
        "name": name,
        "n": n,
        "pass": len(passes),
        "fail": nf,
        "pass_rate": pct(len(passes), n),
        "verdict_reason": verdict_reason,
        "stopped": stopped,
        "error_category": error_category,
        "num_tool_calls_fail": num_tool_calls_fail,
        "num_tool_calls_all": num_tool_calls_all,
        "gold_in_raw_wrong_pred": gold_in_raw_wrong_pred,
        "gold_in_raw_any_completion": gold_in_raw_any_completion,
        "pred_wrong": pred_wrong,
        "depth_call_cross": dict(depth_call_cross),
        "depth_call_cross_fail": dict(depth_call_cross_fail),
        "depth_dist": depth_dist,
        "empty_predicted": empty_predicted,
        "wrong_call_count": wrong_call_count,
        "under_call": under_call,
        "over_call": over_call,
        "exact_call_count_pass": exact_call_count_pass,
        "empty_on_fail": empty_on_fail,
    }


def print_profile_report(stats: Dict[str, Any]) -> None:
    print_section(f"Profile: {stats['name']}")
    n, nf = stats["n"], stats["fail"]
    print(f"  Rollouts: {n}  pass={stats['pass']} ({stats['pass_rate']:.1f}%)  fail={nf}")

    print_counter_table("verdict_reason (failures)", stats["verdict_reason"], nf)
    print_counter_table("stopped (failures)", stats["stopped"], nf)
    print_counter_table("error_category (failures)", stats["error_category"], nf)
    print_counter_table("num_tool_calls (failures)", stats["num_tool_calls_fail"], nf)

    print(f"\n  Failures with wrong predicted_final: {stats['pred_wrong']} ({pct(stats['pred_wrong'], nf):.1f}% of fails)")
    print(
        f"  Gold answer in raw_completions (any fail): "
        f"{stats['gold_in_raw_any_completion']} ({pct(stats['gold_in_raw_any_completion'], nf):.1f}% of fails)"
    )
    print(
        f"  Gold in raw_completions BUT predicted_final wrong: "
        f"{stats['gold_in_raw_wrong_pred']} ({pct(stats['gold_in_raw_wrong_pred'], nf):.1f}% of fails)"
    )
    if stats["pred_wrong"]:
        print(
            f"    -> share of wrong-pred fails rescued in text: "
            f"{pct(stats['gold_in_raw_wrong_pred'], stats['pred_wrong']):.1f}%"
        )

    print("\n  num_tool_calls vs gold dependency depth (all rollouts, count):")
    cross = stats["depth_call_cross"]
    depths = sorted({k[0] for k in cross})
    max_calls = max((k[1] for k in cross), default=0)
    header = "  depth\\calls " + "".join(f"{c:>6}" for c in range(0, min(max_calls, 8) + 1))
    print(header)
    for d in depths:
        row = f"  {d:>11} "
        for c in range(0, min(max_calls, 8) + 1):
            row += f"{cross.get((d, c), 0):>6}"
        print(row)

    print("\n  Call-count patterns (all rollouts):")
    print(f"    empty predicted_calls:     {stats['empty_predicted']:6d} ({pct(stats['empty_predicted'], n):.1f}%)")
    print(f"    empty on failures only:    {stats['empty_on_fail']:6d} ({pct(stats['empty_on_fail'], nf):.1f}% of fails)")
    print(f"    wrong_call_count (!=gold): {stats['wrong_call_count']:6d} ({pct(stats['wrong_call_count'], n):.1f}%)")
    print(f"      under gold call count:   {stats['under_call']:6d}")
    print(f"      over gold call count:    {stats['over_call']:6d}")
    print(f"    exact call count + pass:   {stats['exact_call_count_pass']:6d}")


def compare_error_mix(baseline: Dict[str, Any], stage: Dict[str, Any]) -> None:
    print_section(f"Error mix shift: {stage['name']} vs baseline")
    nf_b, nf_s = baseline["fail"], stage["fail"]
    if not nf_b or not nf_s:
        print("  (insufficient failures for comparison)")
        return

    for label, key in [
        ("error_category", "error_category"),
        ("stopped", "stopped"),
        ("verdict_reason", "verdict_reason"),
    ]:
        print(f"\n  {label} (% of failures, delta pp vs baseline):")
        b_ctr: Counter = baseline[key]
        s_ctr: Counter = stage[key]
        all_keys = sorted(set(b_ctr) | set(s_ctr), key=lambda k: (-s_ctr.get(k, 0), k))
        for k in all_keys:
            b_pct = pct(b_ctr.get(k, 0), nf_b)
            s_pct = pct(s_ctr.get(k, 0), nf_s)
            delta = s_pct - b_pct
            flag = " *" if abs(delta) >= 2.0 else ""
            print(f"    {k!s:32} base {b_pct:5.1f}%  {stage['name']:7} {s_pct:5.1f}%  ({delta:+5.1f} pp){flag}")

    # Summary deltas
    b_no_tools = pct(baseline["num_tool_calls_fail"].get(0, 0), nf_b)
    s_no_tools = pct(stage["num_tool_calls_fail"].get(0, 0), nf_s)
    b_sem = pct(baseline["gold_in_raw_wrong_pred"], baseline["pred_wrong"] or 1)
    s_sem = pct(stage["gold_in_raw_wrong_pred"], stage["pred_wrong"] or 1)
    print(f"\n  Key summary:")
    print(f"    pass rate:           {baseline['pass_rate']:5.1f}% -> {stage['pass_rate']:5.1f}% ({stage['pass_rate']-baseline['pass_rate']:+.1f} pp)")
    print(f"    no_tool fails:       {b_no_tools:5.1f}% -> {s_no_tools:5.1f}% ({s_no_tools-b_no_tools:+.1f} pp)")
    print(f"    semantic rescue*:    {b_sem:5.1f}% -> {s_sem:5.1f}% ({s_sem-b_sem:+.1f} pp)")
    print("    * gold in raw_completions among wrong-predicted failures")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing *_multiturn_predictions.jsonl files",
    )
    parser.add_argument(
        "--profiles",
        nargs="*",
        default=list(PROFILES.keys()),
        help="Profiles to analyze (default: all)",
    )
    parser.add_argument("--baseline-only", action="store_true", help="Only analyze baseline")
    args = parser.parse_args()

    profiles = ["baseline"] if args.baseline_only else args.profiles
    all_stats: Dict[str, Dict[str, Any]] = {}

    for name in profiles:
        if name not in PROFILES:
            print(f"Unknown profile: {name}", file=sys.stderr)
            return 1
        path = args.results_dir / PROFILES[name]
        if not path.exists():
            print(f"Missing file: {path}", file=sys.stderr)
            return 1
        rows = load_jsonl(path)
        all_stats[name] = analyze_profile(rows, name)

    # Baseline detailed report
    if "baseline" in all_stats:
        print_profile_report(all_stats["baseline"])

    # Stage comparison (if not baseline-only)
    if not args.baseline_only and "baseline" in all_stats:
        for stage in ["stage1", "stage2", "stage3"]:
            if stage in all_stats:
                compare_error_mix(all_stats["baseline"], all_stats[stage])

    # Compact cross-profile summary table
    if len(all_stats) > 1:
        print_section("Cross-profile summary")
        print(f"  {'profile':10} {'pass%':>7} {'fail':>7} {'no_tool%':>9} {'empty_pred%':>12} {'wrong_cnt%':>11} {'sem_rescue%':>12}")
        for name, s in all_stats.items():
            nf = s["fail"] or 1
            no_tool = pct(s["num_tool_calls_fail"].get(0, 0), nf)
            sem = pct(s["gold_in_raw_wrong_pred"], s["pred_wrong"] or 1)
            print(
                f"  {name:10} {s['pass_rate']:6.1f}% {s['fail']:7d} "
                f"{no_tool:8.1f}% {pct(s['empty_predicted'], s['n']):11.1f}% "
                f"{pct(s['wrong_call_count'], s['n']):10.1f}% {sem:11.1f}%"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
