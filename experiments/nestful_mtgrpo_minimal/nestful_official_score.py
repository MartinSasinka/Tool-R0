"""Score predictions with the REAL NESTFUL scorer (data/NESTFUL-main/src).

Why this exists
---------------
`metrics.py` computes *definition-inspired* metrics that are NOT identical to the
official NESTFUL evaluator:
  - F1 Func/Param: ours = mean per-sample multiset F1; official = corpus-level
    set-based macro-F1 via sklearn MultiLabelBinarizer.
  - Part./Full Acc: official applies variable GROUNDING (a `$var_2.result$` ref is
    rewritten to `$<producing_fn>.result$`) and length alignment before matching.
For paper numbers comparable to NESTFUL Table 1/2 we must run the official code.

This module adapts our predicted call lists into the exact item format the
official `calculate_scores` expects (the LLaMa-3.1 parser path, which applies
grounding when a `label` field is present) and returns the five paper metrics.

Win Rate executes the IBM functions and uses `signal.SIGALRM`. On Unix this is
native; on Windows we install a small threading-based ``signal.alarm`` shim so
Win Rate can be computed offline without WSL/Linux.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_NESTFUL_SRC = os.path.join(_HERE, "data", "NESTFUL-main", "src")
_DEFAULT_FUNC_DIR = os.path.join(_HERE, "data", "NESTFUL-main", "data_v2", "executable_functions")

# Model name whose official parser reads `generated_text` as a JSON list of calls
# and applies variable grounding when a `label` field is present.
_SCORER_MODEL = "Llama-3.1-8B-Instruct"
_SIGNAL_PATCHED = False


def _patch_signal_alarm_for_win_rate() -> None:
    """Install a threading ``signal.alarm`` shim on Windows (no SIGALRM).

    The official IBM ``calculate_ans`` uses ``signal.alarm(10)`` for timeouts.
    Must run before ``from scorer import ...`` on platforms without SIGALRM.
    """
    global _SIGNAL_PATCHED
    if _SIGNAL_PATCHED:
        return
    import signal
    import threading

    if hasattr(signal, "SIGALRM"):
        _SIGNAL_PATCHED = True
        return

    signal.SIGALRM = 14  # type: ignore[attr-defined]
    _state: Dict[str, Any] = {"timer": None, "handler": None}
    _orig_signal = signal.signal

    def _patched_signal(signum, handler):
        if signum == signal.SIGALRM:
            _state["handler"] = handler
            return _state["handler"]
        return _orig_signal(signum, handler)

    def _patched_alarm(seconds):
        t = _state["timer"]
        if t is not None:
            t.cancel()
            _state["timer"] = None
        if seconds <= 0:
            return 0

        def _fire():
            h = _state["handler"]
            if h is not None:
                h(signal.SIGALRM, None)

        nt = threading.Timer(float(seconds), _fire)
        nt.daemon = True
        _state["timer"] = nt
        nt.start()
        return 0

    signal.signal = _patched_signal  # type: ignore[assignment]
    signal.alarm = _patched_alarm  # type: ignore[attr-defined]
    _SIGNAL_PATCHED = True


def _ensure_scorer_on_path() -> None:
    if _NESTFUL_SRC not in sys.path:
        sys.path.insert(0, _NESTFUL_SRC)
    _patch_signal_alarm_for_win_rate()


def _json_safe(value: Any) -> Any:
    """Recursively coerce predicted-call values to JSON-serializable Python types.

    ``parse_tool_calls_all`` may use ``ast.literal_eval``, which can produce sets
    and tuples inside ``arguments`` — plain ``json.dumps`` then raises.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, bytearray):
        return bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, (set, frozenset)):
        return sorted([_json_safe(x) for x in value], key=lambda x: (type(x).__name__, str(x)))
    if isinstance(value, tuple):
        return [_json_safe(x) for x in value]
    if isinstance(value, list):
        return [_json_safe(x) for x in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    try:
        import numpy as np
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, np.ndarray):
            return _json_safe(value.tolist())
    except ImportError:
        pass
    return str(value)


def add_labels(calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Attach sequential labels ($var1, $var2, ...) so the official grounding can
    map variable references to the producing function. Multi-turn predictions
    have one call per turn and no label, so we assign by position."""
    out = []
    for i, c in enumerate(calls):
        lbl = c.get("label") or f"$var{i + 1}"
        out.append({
            "name": c.get("name", ""),
            "arguments": _json_safe(c.get("arguments", {}) or {}),
            "label": lbl,
        })
    return out


def _json_field_str(value: Any) -> str:
    """Serialize a dataset field for the official scorer's ``json.loads()`` calls.

    The NESTFUL scorer always does ``json.loads(item['gold_answer'])`` (and the
    same for ``tools`` / ``output`` in Win Rate).  Bare strings from the JSONL
    (datetimes like ``022-01-01T00:00:00``, paths, …) are NOT valid JSON text on
    their own — ``json.loads('022-…')`` parses ``0`` and raises Extra data.
    Structured values (list/dict/number) are ``json.dumps``'d once; strings that
    are already complete JSON documents are passed through unchanged.
    """
    if isinstance(value, str):
        stripped = value.lstrip()
        try:
            _obj, end = json.JSONDecoder().raw_decode(stripped)
            if end == len(stripped):
                return value
        except json.JSONDecodeError:
            pass
        return json.dumps(value, ensure_ascii=False)
    return json.dumps(value, ensure_ascii=False)


def build_item(pred_calls: List[Dict[str, Any]], gold_row: Dict[str, Any]) -> Dict[str, str]:
    """Build one official-scorer item from predicted calls + a raw dataset row.

    `gold_row` is a raw line from nestful_data.jsonl with keys output/tools/gold_answer.
    """
    return {
        "generated_text": json.dumps(_json_safe(add_labels(pred_calls)), ensure_ascii=False),
        "output": _json_field_str(gold_row["output"]),
        "tools": _json_field_str(gold_row["tools"]),
        "gold_answer": _json_field_str(gold_row["gold_answer"]),
    }


def score_items(
    items: List[Dict[str, str]],
    executable_func_dir: str = _DEFAULT_FUNC_DIR,
    win_rate: bool = True,
    per_sample: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, float]:
    """Run the official NESTFUL scorer on prepared items. Returns paper metrics.

    Robust Win Rate
    ---------------
    The official ``calculate_scores`` computes Win Rate inline via
    ``calculate_win_score`` -> ``if pred_ans == gold_ans`` (scorer.py:127). When an
    executed answer is a numpy array / list-of-arrays, that comparison returns an
    array and Python raises ``ValueError: truth value of an array ... is ambiguous``,
    which aborts scoring for the ENTIRE batch (one bad sample nulls the whole
    epoch's ``react_win_rate`` and silently breaks checkpoint selection).

    To make Win Rate crash-proof we compute the corpus F1/Acc metrics WITHOUT the
    fragile inline win pass (``win_rate_flag=False``) and derive Win Rate from the
    per-sample path (:func:`score_items_per_sample`), which isolates each sample in
    its own try/except. The number is identical to the official aggregate when
    nothing crashes (same ``calculate_win_score``); a sample whose answer cannot be
    compared counts as a loss (0.0) instead of aborting the epoch.

    Pass ``per_sample`` to reuse an already-computed per-sample result and avoid
    re-executing the IBM functions twice.
    """
    _ensure_scorer_on_path()
    if not items:
        out: Dict[str, float] = {
            "f1_func": 0.0,
            "f1_param": 0.0,
            "partial_sequence_accuracy": 0.0,
            "full_sequence_accuracy": 0.0,
            "num_examples": 0,
            "num_pred_parsing_errors": 0,
        }
        if win_rate:
            out["win_rate"] = 0.0
        return out
    import warnings

    from scorer import calculate_scores  # official code

    with warnings.catch_warnings():
        # sklearn MultiLabelBinarizer is noisy about unseen label classes; that
        # is expected for corpus-level macro-F1 and not an error.
        warnings.simplefilter("ignore")
        res = calculate_scores(
            items,
            _SCORER_MODEL,
            executable_func_dir,
            intents_only=False,
            win_rate_flag=False,   # Win computed robustly below (per-sample)
            alt_traj_flag=False,
        )
    out = {
        "f1_func": float(res["f1_intent"]),
        "f1_param": float(res["f1_slot"]) if res["f1_slot"] else 0.0,
        "partial_sequence_accuracy": float(res["accuracy_combined"]),
        "full_sequence_accuracy": float(res["percentage_times_full_score"]),
        "num_examples": int(res["num_examples"]),
        "num_pred_parsing_errors": int(res["num_pred_examples_w_parsing_errors"]),
    }
    if win_rate:
        if per_sample is None:
            per_sample = score_items_per_sample(
                items, executable_func_dir=executable_func_dir, win_rate=True
            )
        wins = [r["official_win"] for r in per_sample
                if r.get("official_win") is not None]
        # Never emit a null Win: an empty/failed win list scores 0.0 so downstream
        # checkpoint selection has a concrete (conservative) number.
        out["win_rate"] = (sum(wins) / len(wins)) if wins else 0.0
    return out


def _canonical_api_strings(func_calls: List[str]) -> List[str]:
    """Replicate the official canonical step string `name(sorted("k = v"))`
    (scorer.py 259-282) from a list of grounded call JSON strings."""
    out: List[str] = []
    for f in func_calls:
        if not f:
            continue
        try:
            fd = json.loads(f.replace("<|endoftext|>", "").strip())
            f_name = str(fd["name"])
            arg_list = []
            for key, val in (fd.get("arguments") or {}).items():
                if isinstance(val, str) and val.startswith("$") and not val.endswith("$"):
                    val = val + "$"
                arg_list.append(f"{key} = {val}")
            args = ", ".join(sorted(arg_list))
            out.append(f"{f_name}({args})")
        except Exception:  # noqa: BLE001 - mirror official's per-call skip
            continue
    return out


def score_items_per_sample(
    items: List[Dict[str, str]],
    executable_func_dir: str = _DEFAULT_FUNC_DIR,
    win_rate: bool = True,
) -> List[Dict[str, Any]]:
    """Per-sample OFFICIAL diagnostics, computed with the official helpers.

    Returns one dict per item, aligned with `items`, containing:
      parse_valid, n_pred_calls, official_partial_match, official_full_match,
      official_win, pred_answer, executable, execution_error.

    NOTE: F1 Func / F1 Param are *corpus-level macro* metrics and cannot be
    expressed per-sample; they are only available as aggregates from
    `score_items`. Win/pred_answer require executing the IBM functions, which
    uses `signal.SIGALRM` (with a Windows shim when needed); when `win_rate=False`
    those fields are None. `execution_error` is coarse ("execution_failed") because the
    official `calculate_ans` swallows the traceback and returns False on failure.
    """
    _ensure_scorer_on_path()
    import warnings

    from output_parsers import parse_llama_3_output, ground_seq_nested_repsonse  # noqa: F401
    from scorer import calculate_win_score, calculate_ans
    from utils import post_process_api_with_args

    results: List[Dict[str, Any]] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for item in items:
            (pred_func_calls, gold_func_calls, pred_dict_list, gold_dict_list,
             _n_err, pred_has_parsing_errors) = parse_llama_3_output(item, 0)

            api_gold = _canonical_api_strings(gold_func_calls)
            api_pred = _canonical_api_strings(pred_func_calls)
            g2, p2 = post_process_api_with_args(list(api_gold), list(api_pred))
            n = len(g2)
            if n == 0:
                partial = 0.0
            else:
                partial = sum(1 for a, b in zip(g2, p2) if a == b) / n
            full = 1.0 if (n > 0 and partial == 1.0) else 0.0

            rec: Dict[str, Any] = {
                "parse_valid": (not pred_has_parsing_errors),
                "n_pred_calls": len(pred_dict_list) if isinstance(pred_dict_list, list) else 0,
                "official_partial_match": float(partial),
                "official_full_match": float(full),
                "official_win": None,
                "pred_answer": None,
                "executable": None,
                "execution_error": None,
            }

            if win_rate:
                try:
                    win = calculate_win_score(
                        pred_dict_list, item["gold_answer"], item["tools"], executable_func_dir
                    )
                    rec["official_win"] = 1.0 if win else 0.0
                    pred_ans = calculate_ans(
                        pred_dict_list, json.loads(item["tools"]), executable_func_dir
                    )
                    executable = pred_ans is not False
                    rec["executable"] = bool(executable)
                    rec["pred_answer"] = pred_ans if executable else None
                    rec["execution_error"] = None if executable else "execution_failed"
                except Exception as exc:  # noqa: BLE001 - never let one sample abort scoring
                    rec["official_win"] = 0.0
                    rec["executable"] = False
                    rec["execution_error"] = f"{type(exc).__name__}: {exc}"

            results.append(rec)
    return results


def load_raw_dataset(path: str) -> Dict[str, Dict[str, Any]]:
    """Map sample_id -> raw dataset row (output/tools/gold_answer kept as-is)."""
    rows: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            sid = r.get("sample_id") or r.get("task_id") or r.get("id")
            if sid is not None:
                rows[str(sid)] = r
    return rows


# ---------------------------------------------------------------------------
#  Standalone re-scoring of an existing trajectories.jsonl
# ---------------------------------------------------------------------------

def _predicted_calls_from_traj(traj: Dict[str, Any], lenient: bool = True) -> List[Dict[str, Any]]:
    """Re-extract the predicted call sequence from a stored trajectory's turns."""
    from parser import parse_tool_call

    calls: List[Dict[str, Any]] = []
    for t in traj.get("turns", []):
        txt = t.get("model_text", "")
        pr = parse_tool_call(txt, lenient=lenient)
        if pr.is_terminal:
            break
        if pr.ok and pr.call is not None:
            calls.append(pr.call)
        else:
            break
    return calls


def rescore_trajectories(
    trajectories_path: str,
    dataset_path: str,
    executable_func_dir: str = _DEFAULT_FUNC_DIR,
    win_rate: bool = True,
    lenient: bool = True,
) -> Dict[str, float]:
    raw = load_raw_dataset(dataset_path)
    items: List[Dict[str, str]] = []
    missing = 0
    with open(trajectories_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sid = str(row.get("sample_id") or row.get("task_id") or "")
            gold_row = raw.get(sid)
            if gold_row is None:
                missing += 1
                continue
            traj = row.get("_traj", row)
            pred = _predicted_calls_from_traj(traj, lenient=lenient)
            items.append(build_item(pred, gold_row))
    metrics = score_items(items, executable_func_dir, win_rate=win_rate)
    metrics["num_missing_in_dataset"] = missing
    return metrics


def rescore_direct_predictions(
    predictions_path: str,
    dataset_path: str,
    executable_func_dir: str = _DEFAULT_FUNC_DIR,
    win_rate: bool = True,
) -> Dict[str, float]:
    """Re-score a ``direct_predictions.jsonl`` without re-running generation."""
    raw = load_raw_dataset(dataset_path)
    items: List[Dict[str, str]] = []
    missing = 0
    with open(predictions_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sid = str(row.get("sample_id") or row.get("task_id") or "")
            gold_row = raw.get(sid)
            if gold_row is None:
                missing += 1
                continue
            pred = row.get("predicted_calls") or []
            items.append(build_item(pred, gold_row))
    metrics = score_items(items, executable_func_dir, win_rate=win_rate)
    metrics["paradigm"] = "direct"
    metrics["num_missing_in_dataset"] = missing
    return metrics


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Official NESTFUL scoring of saved eval outputs.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--trajectories", help="ReAct *_trajectories.jsonl from an eval run")
    src.add_argument("--direct-predictions",
                     help="direct_predictions.jsonl (no re-generation needed)")
    ap.add_argument("--dataset", default=os.path.join(_HERE, "data", "NESTFUL-main", "data_v2", "nestful_data.jsonl"))
    ap.add_argument("--executable_func_dir", default=_DEFAULT_FUNC_DIR)
    ap.add_argument("--no-win-rate", action="store_true",
                    help="skip Win Rate (required on Windows; SIGALRM is Unix-only)")
    ap.add_argument("--strict", action="store_true", help="use strict parser instead of lenient")
    ap.add_argument("--out", default=None, help="optional path to write metrics JSON")
    args = ap.parse_args()

    if args.direct_predictions:
        metrics = rescore_direct_predictions(
            args.direct_predictions, args.dataset, args.executable_func_dir,
            win_rate=not args.no_win_rate,
        )
    else:
        metrics = rescore_trajectories(
            args.trajectories, args.dataset, args.executable_func_dir,
            win_rate=not args.no_win_rate, lenient=not args.strict,
        )
    print(json.dumps(metrics, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
