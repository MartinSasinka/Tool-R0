"""Final-evaluation metrics (EVAL ONLY — never used as training reward).

Two families:

  A. NESTFUL official-compatible metrics:
        f1_func, f1_param, partial_sequence_accuracy,
        full_sequence_accuracy, win_rate
     Implemented to be DEFINITION-compatible with NESTFUL, but not guaranteed
     byte-identical to the official evaluator.

  B. Our paper-specific alternative-path metrics:
        strict_gold_trace_pass, solution_equivalent_pass,
        alternative_valid_solution_pass, strict_fail_but_solution_equivalent_pass,
        correct_answer_but_unsupported_trace, final_answer_pass

This file is a minimal standalone reimplementation; it imports nothing from
curricullum/ or nestful_evaluation/. There is NO LLM judge anywhere.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from executor import matches_gold, coerce_numeric


# =====================================================================
#  Official-compatibility helpers (internal diagnostic replica)
#
#  These replicate the official NESTFUL scorer semantics so metrics.py can act
#  as a cross-check. They are NOT the source of truth — paper tables use the
#  official adapter (nestful_official_score.py). Where the official helper is
#  importable we prefer it (see _align_canonical).
# =====================================================================

def decimal_aware_equal(pred: Any, gold: Any) -> bool:
    """Replicate the official float comparison (scorer.py 124-127): if gold is a
    float, round pred to gold's decimal-place count then compare exactly. Falls
    back to the tolerant matches_gold for non-floats / odd types."""
    try:
        if (
            isinstance(pred, (int, float)) and isinstance(gold, (int, float))
            and not isinstance(pred, bool) and not isinstance(gold, bool)
        ):
            if isinstance(gold, float):
                s = repr(gold)
                dec = len(s.split(".")[1]) if "." in s else 0
                return round(float(pred), dec) == gold
            return pred == gold
    except (ValueError, TypeError, OverflowError):
        pass
    return matches_gold(pred, gold)


def _check_label_in_slot(label: str, slot_v: Any) -> bool:
    """Mirror output_parsers.ground_seq_nested_repsonse.check_label_in_slot."""
    if isinstance(slot_v, str) and slot_v.startswith("$var") and "." in slot_v:
        return slot_v.split(".", 1)[0].replace("$", "") == label
    return False


def ground_calls(calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Replicate official variable grounding: rewrite a `$var_2.result$` slot to
    `$<producing_fn>.result$` so comparison is label-name independent. Calls
    without an explicit label get positional `$varN` labels (matching the
    official adapter's add_labels)."""
    label_to_name: Dict[str, str] = {}
    for i, api in enumerate(calls):
        name = api.get("name", "") or ""
        if name in ("varResult", "var_result"):
            continue
        lbl = (api.get("label") or f"$var{i + 1}").replace("$", "")
        if lbl:
            label_to_name[lbl] = name

    grounded: List[Dict[str, Any]] = []
    for api in calls:
        if api.get("name") == "var_result":
            continue
        args = api.get("arguments") or {}
        new_args: Dict[str, Any] = {}
        for s_n, s_v in args.items():
            v = s_v
            for lbl, nm in label_to_name.items():
                if isinstance(v, str) and _check_label_in_slot(lbl, v):
                    v = v.replace(lbl, nm)
                elif isinstance(v, list):
                    nv = []
                    for it in v:
                        if isinstance(it, str) and _check_label_in_slot(lbl, it):
                            it = it.replace(lbl, nm)
                        nv.append(it)
                    v = nv
            new_args[s_n] = v
        grounded.append({"name": api.get("name", "") or "", "arguments": new_args})
    return grounded


def _canonical_step(call: Dict[str, Any]) -> str:
    """Official canonical step string: name(sorted("k = v")) (scorer.py 259-282)."""
    name = str(call.get("name", ""))
    parts = []
    for k, v in (call.get("arguments") or {}).items():
        if isinstance(v, str) and v.startswith("$") and not v.endswith("$"):
            v = v + "$"
        parts.append(f"{k} = {v}")
    return f"{name}(" + ", ".join(sorted(parts)) + ")"


def _align_canonical(gold: List[str], pred: List[str]) -> Tuple[List[str], List[str]]:
    """Length-align two canonical-string lists. Prefer the official
    post_process_api_with_args (utils.py); fall back to right-padding."""
    if len(gold) == len(pred):
        return gold, pred
    try:
        import os
        import sys
        _src = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data", "NESTFUL-main", "src",
        )
        if _src not in sys.path:
            sys.path.insert(0, _src)
        from utils import post_process_api_with_args  # official aligner
        return post_process_api_with_args(list(gold), list(pred))
    except Exception:  # noqa: BLE001 - fallback when NESTFUL-main is absent
        n = max(len(gold), len(pred))
        return gold + [""] * (n - len(gold)), pred + [""] * (n - len(pred))


def _grounded_func_names(calls: List[Dict[str, Any]]) -> List[str]:
    return [c.get("name", "") for c in ground_calls(calls)]


def _grounded_param_slots(calls: List[Dict[str, Any]]) -> List[str]:
    """Value-aware (name, "k = v") slot tokens, grounded — used for F1 Param."""
    out: List[str] = []
    for c in ground_calls(calls):
        name = c.get("name", "")
        for k, v in (c.get("arguments") or {}).items():
            if isinstance(v, str) and v.startswith("$") and not v.endswith("$"):
                v = v + "$"
            out.append(f"{name}|{k} = {v}")
    return out


def internal_corpus_macro_f1(
    gold_lists: List[List[str]], pred_lists: List[List[str]]
) -> Optional[float]:
    """Corpus-level set-based macro-F1 (sklearn MultiLabelBinarizer), replicating
    the official compute_score_sklearn. Returns None if sklearn is unavailable.
    DIAGNOSTIC ONLY — the canonical F1 comes from the official adapter."""
    try:
        import warnings

        from sklearn.metrics import f1_score
        from sklearn.preprocessing import MultiLabelBinarizer

        mlb = MultiLabelBinarizer()
        mlb.fit(gold_lists)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(
                f1_score(
                    mlb.transform(gold_lists),
                    mlb.transform(pred_lists),
                    average="macro",
                    zero_division=0,
                )
            )
    except Exception:  # noqa: BLE001
        return None


# =====================================================================
#  Evidence check
# =====================================================================

def _collect_scalars(obj: Any, out: List[Any]) -> None:
    if isinstance(obj, bool):
        out.append(obj)
    elif isinstance(obj, (int, float)):
        out.append(obj)
    elif isinstance(obj, str):
        s = obj.strip()
        if s:
            out.append(coerce_numeric(s))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _collect_scalars(k, out)
            _collect_scalars(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect_scalars(v, out)


def answer_supported_by_observations(
    final_answer: Any, observations: List[Any], gold_answer: Any
) -> bool:
    """Deterministic evidence check (no LLM).

    Returns True only if the final answer matches the gold answer AND every
    scalar value in the final answer also appears among the model's own
    observation values. If support cannot be established, returns False.
    """
    if not matches_gold(final_answer, gold_answer):
        return False

    obs_scalars: List[Any] = []
    for obs in observations:
        _collect_scalars(obs, obs_scalars)

    target_scalars: List[Any] = []
    _collect_scalars(final_answer, target_scalars)
    if not target_scalars:
        # Cannot verify support -> conservative False.
        return False

    for tv in target_scalars:
        if not any(matches_gold(tv, ov) for ov in obs_scalars):
            return False
    return True


# =====================================================================
#  Trajectory health
# =====================================================================

def is_noop_or_bruteforce(trajectory, task: Dict[str, Any]) -> bool:
    if trajectory.zero_tool_calls:
        return True
    calls = trajectory.predicted_calls
    # Repeated identical call (noop loop).
    sigs = [
        (c.get("name"), tuple(sorted((c.get("arguments") or {}).keys())))
        for c in calls
    ]
    counts = Counter(sigs)
    if any(v >= 3 for v in counts.values()):
        return True
    # Brute-force: called (almost) every available tool while exceeding gold len.
    gold_n = len(task.get("gold_calls", []))
    n_tools = len(task.get("tools", []))
    unique_tools = len({c.get("name") for c in calls})
    if n_tools and unique_tools >= n_tools and trajectory.num_tool_calls > gold_n:
        return True
    return False


# =====================================================================
#  NESTFUL official-compatible metrics
# =====================================================================

def _multiset_f1(pred: List[Any], gold: List[Any]) -> Dict[str, float]:
    pc, gc = Counter(pred), Counter(gold)
    matched = sum((pc & gc).values())
    p = matched / len(pred) if pred else 0.0
    r = matched / len(gold) if gold else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def _args_equal(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    if set(a.keys()) != set(b.keys()):
        return False
    for k in a:
        if a[k] == b[k]:
            continue
        if matches_gold(a[k], b[k]):
            continue
        return False
    return True


def compute_nestful_official_metrics(
    predicted_calls: List[Dict[str, Any]],
    gold_calls: List[Dict[str, Any]],
    trajectory=None,
    task: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """INTERNAL diagnostic replica of the official NESTFUL metrics.

    Replicates official semantics as closely as practical: variable grounding,
    value-aware F1 Param slots, canonical step strings with length alignment for
    Partial/Full, and a decimal-aware Win check. These outputs are stored as
    `internal_*` and are NOT the reported paper numbers — the official adapter
    (nestful_official_score.py) is canonical. On any malformed input the function
    degrades to the failure mode (zeros) and never raises.

    Per-sample `f1_func` / `f1_param` here are count-aware multiset F1 and are
    DIAGNOSTIC ONLY; the official F1 is corpus-level macro (see
    internal_corpus_macro_f1, computed at aggregation).
    """
    predicted_calls = predicted_calls or []
    gold_calls = gold_calls or []

    try:
        # Grounded function-name and value-aware slot views (official-style).
        gold_names = _grounded_func_names(gold_calls)
        pred_names = _grounded_func_names(predicted_calls)
        f1_func = _multiset_f1(pred_names, gold_names)["f1"]

        gold_slots = _grounded_param_slots(gold_calls)
        pred_slots = _grounded_param_slots(predicted_calls)
        f1_param = _multiset_f1(pred_slots, gold_slots)["f1"]

        # Partial/Full via grounded canonical strings + official length alignment.
        gold_canon = [_canonical_step(c) for c in ground_calls(gold_calls)]
        pred_canon = [_canonical_step(c) for c in ground_calls(predicted_calls)]
        g2, p2 = _align_canonical(gold_canon, pred_canon)
        n = len(g2)
        partial_seq = (sum(1 for a, b in zip(g2, p2) if a == b) / n) if n else 0.0
        full_seq = 1.0 if (n > 0 and partial_seq == 1.0) else 0.0
    except Exception:  # noqa: BLE001 - failure mode: never throw, score as miss
        f1_func = f1_param = partial_seq = full_seq = 0.0
        gold_names = pred_names = gold_slots = pred_slots = []

    # win_rate (internal): executable success reaching gold answer, decimal-aware.
    # NOTE: the reported Win Rate comes from the official scorer's re-execution.
    win = 0.0
    if trajectory is not None and task is not None:
        executable = (
            not trajectory.executor_error
            and not trajectory.zero_tool_calls
            and trajectory.stop_reason not in ("parse_fail",)
        )
        final_ok = decimal_aware_equal(
            trajectory.final_observation, task.get("gold_answer")
        )
        if executable and final_ok:
            win = 1.0

    return {
        "f1_func": f1_func,
        "f1_param": f1_param,
        "partial_sequence_accuracy": partial_seq,
        "full_sequence_accuracy": full_seq,
        "win_rate": win,
        # Grounded lists for corpus-level macro-F1 aggregation (diagnostic).
        "_gold_func_names": gold_names,
        "_pred_func_names": pred_names,
        "_gold_param_slots": gold_slots,
        "_pred_param_slots": pred_slots,
    }


# =====================================================================
#  Solution-equivalent + paper metrics
# =====================================================================

@dataclass
class MetricResult:
    passed: bool
    limited: bool
    reason: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)


def solution_equivalent_score(trajectory, task: Dict[str, Any]) -> MetricResult:
    """EVAL-ONLY. 1 iff trajectory is executable, final answer == gold, and the
    answer is supported by the model's own observations.

    `limited=True` when the executor ran in gold_replay mode (observations are
    synthetic), in which case this metric must NOT be reported as a real result.
    """
    limited = getattr(trajectory, "executor_mode", "gold_replay") != "full"

    if trajectory.zero_tool_calls:
        return MetricResult(False, limited, "zero_tool_calls")
    if trajectory.executor_error:
        return MetricResult(False, limited, "executor_error")
    if trajectory.stop_reason == "parse_fail":
        return MetricResult(False, limited, "parse_fail")
    if is_noop_or_bruteforce(trajectory, task):
        return MetricResult(False, limited, "noop_or_bruteforce")

    final_ok = matches_gold(trajectory.final_observation, task.get("gold_answer"))
    if not final_ok:
        return MetricResult(False, limited, "final_answer_mismatch")

    supported = answer_supported_by_observations(
        trajectory.final_observation, trajectory.observations, task.get("gold_answer")
    )
    if not supported:
        return MetricResult(False, limited, "answer_not_supported_by_own_trace")

    return MetricResult(True, limited, None)


def compute_paper_metrics(
    trajectory,
    task: Dict[str, Any],
    strict_reward,
    official: Dict[str, float],
) -> Dict[str, Any]:
    """Combine strict reward, solution-equivalent, and NESTFUL official metrics
    into the per-sample paper record.
    """
    strict_pass = bool(strict_reward.reward >= 1.0)
    final_pass = bool(strict_reward.diagnostics.get("final_answer_pass"))
    full_seq = bool(official.get("full_sequence_accuracy", 0.0) >= 1.0)

    se = solution_equivalent_score(trajectory, task)
    solution_equivalent_pass = bool(se.passed)

    strict_fail_but_equiv = bool((not full_seq) and solution_equivalent_pass)
    # Alternative valid solution = solved equivalently but NOT via the gold path.
    alternative_valid = strict_fail_but_equiv
    correct_answer_unsupported = bool(final_pass and not solution_equivalent_pass)

    return {
        "strict_gold_trace_pass": strict_pass,
        "solution_equivalent_pass": solution_equivalent_pass,
        "alternative_valid_solution_pass": alternative_valid,
        "strict_fail_but_solution_equivalent_pass": strict_fail_but_equiv,
        "correct_answer_but_unsupported_trace": correct_answer_unsupported,
        "final_answer_pass": final_pass,
        "solution_equivalent_limited": se.limited,
        "solution_equivalent_reason": se.reason,
    }


# =====================================================================
#  Aggregation
# =====================================================================

_OFFICIAL_KEYS = [
    "f1_func", "f1_param", "partial_sequence_accuracy",
    "full_sequence_accuracy", "win_rate",
]
_PAPER_KEYS = [
    "strict_gold_trace_pass", "solution_equivalent_pass",
    "alternative_valid_solution_pass", "strict_fail_but_solution_equivalent_pass",
    "correct_answer_but_unsupported_trace", "final_answer_pass",
]


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate_final_eval(
    per_sample: List[Dict[str, Any]], executor_mode: str
) -> Dict[str, Any]:
    def _agg(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        internal = {k: _mean([float(r["internal"][k]) for r in rows]) for k in _OFFICIAL_KEYS}
        # Corpus-level macro-F1 (diagnostic replica of official compute_score_sklearn).
        gl_f = [r["_internal_lists"]["gold_func"] for r in rows if r.get("_internal_lists")]
        pl_f = [r["_internal_lists"]["pred_func"] for r in rows if r.get("_internal_lists")]
        gl_s = [r["_internal_lists"]["gold_slots"] for r in rows if r.get("_internal_lists")]
        pl_s = [r["_internal_lists"]["pred_slots"] for r in rows if r.get("_internal_lists")]
        if gl_f:
            internal["f1_func_corpus_macro"] = internal_corpus_macro_f1(gl_f, pl_f)
            internal["f1_param_corpus_macro"] = internal_corpus_macro_f1(gl_s, pl_s)
        ours = {k: _mean([1.0 if r["paper"][k] else 0.0 for r in rows]) for k in _PAPER_KEYS}
        # Label makes the source-of-truth policy explicit: this is the INTERNAL
        # diagnostic replica; the canonical paper metrics are in metrics_official.json.
        return {"internal_metrics_diagnostic": internal, "our_metrics": ours}

    overall = _agg(per_sample) if per_sample else {
        "internal_metrics_diagnostic": {k: 0.0 for k in _OFFICIAL_KEYS},
        "our_metrics": {k: 0.0 for k in _PAPER_KEYS},
    }

    by_calls: Dict[str, Any] = {}
    buckets: Dict[int, List[Dict[str, Any]]] = {}
    for r in per_sample:
        buckets.setdefault(int(r["num_gold_calls"]), []).append(r)
    for n in sorted(buckets):
        by_calls[str(n)] = _agg(buckets[n])

    reportable = executor_mode == "full"
    report = {
        "num_tasks": len(per_sample),
        "executor_mode": executor_mode,
        "solution_equivalent_reportable": reportable,
        "win_rate_reportable": reportable,
        "win_rate_and_solution_equivalent_limited": not reportable,
        **overall,
        "by_num_calls": by_calls,
    }
    if not reportable:
        report["warning"] = (
            "Alternative-path metrics are limited because non-gold calls cannot "
            "be genuinely executed."
        )
    else:
        report["note"] = (
            "Executor ran in full mode: win_rate and solution_equivalent_pass "
            "are real (definition-compatible with NESTFUL, not byte-identical)."
        )
    return report
