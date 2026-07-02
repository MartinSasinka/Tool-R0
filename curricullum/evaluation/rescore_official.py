"""Re-score OLD curriculum eval predictions with TODAY's metrics — zero GPU.

Motivation
----------
The old curriculum eval (`*_multiturn_predictions.jsonl`) only ever reported
`final_answer_accuracy` (executor pass-rate). It never computed the NESTFUL paper
metrics (F1 Func, F1 Param, Partial/Full Sequence Accuracy, Win Rate). Because the
prediction files already store everything we need per task:
    gold_calls (with labels), tools, gold_answer, predicted_calls, raw_completions
we can recompute the paper-comparable metrics OFFLINE, using the exact same
official NESTFUL scorer the new MT-GRPO run uses. This makes the old curriculum
stages an apples-to-apples ablation against the new run — without re-running any
inference.

Two ablation dimensions
-----------------------
  metric : old custom metric  ->  official NESTFUL scorer
  parser : old (stored predicted_calls)  ->  new lenient parser (re-parse
           raw_completions per turn)

By default we score the stored `predicted_calls` (isolates the metric change).
Pass --reparse to ALSO re-extract calls from `raw_completions` with the new
lenient parser and score that (shows how much the new parser recovers).

Multiple rollouts
-----------------
The official scorer expects ONE prediction per task (its F1 is corpus-level). The
old files contain several rollouts per task (`rollout_idx`). We deduplicate to one
row per task_id: rollout 0 by default, or the best final-answer-pass rollout with
--best.

Usage
-----
    # Windows (no Win Rate — official scorer's executor needs Unix SIGALRM):
    python curricullum/evaluation/rescore_official.py --no-win-rate

    # RunPod / Linux (with Win Rate):
    python curricullum/evaluation/rescore_official.py

    # Also show new-parser recovery, pick best rollout:
    python curricullum/evaluation/rescore_official.py --reparse --best --no-win-rate
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
# The official-scorer adapter + new parser live in the experiments package.
_EXP = os.path.join(_REPO_ROOT, "experiments", "nestful_mtgrpo_minimal")
if _EXP not in sys.path:
    sys.path.insert(0, _EXP)

import nestful_official_score as nos  # noqa: E402
from parser import parse_tool_call  # noqa: E402

# Default glob set: every curriculum predictions file we know about.
_DEFAULT_GLOBS = [
    os.path.join(_HERE, "results1", "*_multiturn_predictions.jsonl"),
    os.path.join(_HERE, "results_v2_20260617", "*_multiturn_predictions.jsonl"),
    os.path.join(_HERE, "results_toolr0", "*_multiturn_predictions.jsonl"),
]


def _profile_name(path: str) -> str:
    base = os.path.basename(path).replace("_multiturn_predictions.jsonl", "")
    parent = os.path.basename(os.path.dirname(path))
    return f"{parent}/{base}"


def _load_dedup(path: str, pick_best: bool) -> List[Dict[str, Any]]:
    """Load a predictions file and keep ONE row per task_id.

    pick_best=False -> first occurrence of rollout_idx 0 (or first seen).
    pick_best=True  -> the rollout with the highest score / a passing verdict.
    """
    by_task: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            tid = r.get("task_id")
            if tid is None:
                continue
            cur = by_task.get(tid)
            if cur is None:
                by_task[tid] = r
                continue
            if pick_best:
                if _row_score(r) > _row_score(cur):
                    by_task[tid] = r
            else:
                # prefer rollout_idx == 0 when available
                if r.get("rollout_idx") == 0 and cur.get("rollout_idx") != 0:
                    by_task[tid] = r
    return list(by_task.values())


def _row_score(r: Dict[str, Any]) -> float:
    if r.get("verdict") == "pass":
        return 2.0
    try:
        return float(r.get("score", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _json_safe(obj: Any) -> Any:
    """Make a value JSON-serializable. The lenient parser uses ast.literal_eval,
    which can turn a model's "{1, 2, 3}" into a Python set; sets (and tuples) are
    not JSON types, so coerce them to lists recursively."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (set, frozenset)):
        try:
            return sorted(_json_safe(v) for v in obj)
        except TypeError:
            return [_json_safe(v) for v in obj]
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _reparse_calls(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Rebuild the predicted call sequence by re-parsing every stored per-turn
    completion with the NEW lenient parser.

    The old files store ONE completion per turn taken (len(raw_completions) ==
    num_steps), so we have the full trajectory text. Unlike a live rollout we must
    NOT stop at the first failed/terminal turn: we already know the model kept
    going, so we collect every recoverable call across all turns. This measures
    exactly how many calls the new lenient parser extracts from the same raw text
    the old pipeline saw."""
    completions = row.get("raw_completions") or []
    if isinstance(completions, str):
        completions = [completions]
    calls: List[Dict[str, Any]] = []
    for text in completions:
        pr = parse_tool_call(text or "", lenient=True)
        if pr.ok and pr.call is not None and not pr.is_terminal:
            calls.append(pr.call)
    return calls


_DEFAULT_DATASET = os.path.join(
    _EXP, "data", "NESTFUL-main", "data_v2", "nestful_data.jsonl"
)
_DATASET_CACHE: Optional[Dict[str, Dict[str, Any]]] = None


def _load_nestful_dataset() -> Dict[str, Dict[str, Any]]:
    """task_id/sample_id -> raw row (tools with output_parameters, output, gold_answer)."""
    global _DATASET_CACHE
    if _DATASET_CACHE is not None:
        return _DATASET_CACHE
    path = _DEFAULT_DATASET
    if not os.path.isfile(path):
        _DATASET_CACHE = {}
        return _DATASET_CACHE
    _DATASET_CACHE = nos.load_raw_dataset(path)
    return _DATASET_CACHE


def _gold_row(row: Dict[str, Any], dataset: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Adapt an old prediction row into the raw-dataset shape build_item expects.

    Predictions store ``tools`` without ``output_parameters``; Win Rate needs the
    full IBM spec from the official NESTFUL JSONL (matched by task_id).
    """
    tid = str(row.get("task_id") or row.get("sample_id") or "")
    ds = dataset.get(tid) or {}
    tools = ds.get("tools", row.get("tools", []))
    gold_answer = ds.get("gold_answer", row.get("gold_answer", ""))
    output = ds.get("output")
    if output is None:
        output = json.dumps(row.get("gold_calls", []), ensure_ascii=False)
    return {
        "output": output,
        "tools": tools,
        "gold_answer": gold_answer,
    }


def _score(rows: List[Dict[str, Any]], use_reparse: bool,
           func_dir: Optional[str], win_rate: bool,
           dataset: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    items = []
    n_calls_pred = 0
    for r in rows:
        pred = _reparse_calls(r) if use_reparse else (r.get("predicted_calls") or [])
        pred = _json_safe(pred)
        n_calls_pred += len(pred)
        items.append(nos.build_item(pred, _gold_row(r, dataset)))
    metrics = nos.score_items(items, executable_func_dir=func_dir, win_rate=win_rate)
    metrics["avg_pred_calls"] = round(n_calls_pred / max(len(rows), 1), 3)
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", action="append", default=None,
                    help="prediction file glob(s); repeatable. Default: all known result dirs.")
    ap.add_argument("--reparse", action="store_true",
                    help="also re-extract calls from raw_completions with the new lenient parser")
    ap.add_argument("--best", action="store_true",
                    help="dedup rollouts by best score instead of rollout_idx 0")
    ap.add_argument("--no-win-rate", action="store_true",
                    help="skip Win Rate (official IBM re-execution of predicted calls)")
    ap.add_argument("--func-dir", default=nos._DEFAULT_FUNC_DIR,
                    help="executable_functions dir for Win Rate")
    ap.add_argument("--out", default=os.path.join(_HERE, "rescored_official.json"))
    args = ap.parse_args()

    globs = args.glob or _DEFAULT_GLOBS
    paths: List[str] = []
    for g in globs:
        paths.extend(sorted(glob.glob(g)))
    if not paths:
        print("No prediction files matched.")
        return 1

    win_rate = not args.no_win_rate
    func_dir = args.func_dir if (win_rate and os.path.isdir(args.func_dir or "")) else None
    if win_rate and not func_dir:
        print("[note] executable_functions dir missing -> Win Rate skipped.\n")
        win_rate = False

    dataset = _load_nestful_dataset()
    if win_rate and not dataset:
        print(f"[note] NESTFUL dataset missing at {_DEFAULT_DATASET} -> Win Rate skipped.\n")
        win_rate = False

    results: Dict[str, Any] = {}
    # (metric_key, short_header) — short headers keep the table readable.
    col_defs = [
        ("f1_func", "F1_func"),
        ("f1_param", "F1_param"),
        ("partial_sequence_accuracy", "PartAcc"),
        ("full_sequence_accuracy", "FullAcc"),
    ]
    if win_rate:
        col_defs.append(("win_rate", "WinRate"))
    col_defs.append(("avg_pred_calls", "avgCalls"))
    cols = [k for k, _ in col_defs]

    print(f"Scoring {len(paths)} file(s)  | dedup={'best' if args.best else 'rollout0'}"
          f"  | win_rate={'on' if win_rate else 'off'}\n")

    hdr = f"{'profile':<42} {'parser':<8}" + "".join(f"{h:>10}" for _, h in col_defs) + f"{'n':>7}"
    print(hdr)
    print("-" * len(hdr))

    for path in paths:
        name = _profile_name(path)
        rows = _load_dedup(path, pick_best=args.best)
        entry: Dict[str, Any] = {"num_tasks": len(rows)}

        modes = [("stored", False)]
        if args.reparse:
            modes.append(("reparse", True))

        for label, use_reparse in modes:
            try:
                m = _score(rows, use_reparse, func_dir, win_rate, dataset)
            except Exception as exc:  # noqa: BLE001
                print(f"{name:<44}{label:<10}  ERROR: {exc!r}")
                continue
            entry[label] = m
            line = f"{name:<42} {label:<8}"
            for c in cols:
                v = m.get(c)
                line += f"{v:>10.4f}" if isinstance(v, (int, float)) else f"{'-':>10}"
            line += f"{len(rows):>7}"
            print(line)
        results[name] = entry

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
