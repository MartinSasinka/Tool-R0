"""Deterministic failure taxonomy over stored trajectories."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .graph_features import is_reference, parse_reference
from .io import as_bool
from .trajectory_features import _parsed_calls


# Priority order for primary category (first match wins among applicable).
PRIMARY_PRIORITY = [
    "SUCCESS_STRICT_GOLD",
    "SUCCESS_ALTERNATIVE_VALID",
    "SUCCESS_OTHER_OFFICIAL",
    "FAIL_PARSE_INVALID",
    "FAIL_NO_TOOL_CALL",
    "FAIL_CLIPPED_OUTPUT",
    "FAIL_PROMPT_OVERFLOW",
    "FAIL_REFERENCE_SYNTAX",
    "FAIL_REFERENCE_TO_UNKNOWN_LABEL",
    "FAIL_REFERENCE_WRONG_SOURCE",
    "FAIL_REFERENCE_WRONG_OUTPUT_KEY",
    "FAIL_EXECUTOR_ERROR",
    "FAIL_WRONG_FIRST_TOOL",
    "FAIL_WRONG_LATER_TOOL",
    "FAIL_WRONG_TOOL_SEQUENCE",
    "FAIL_MISSING_REQUIRED_ARGUMENT",
    "FAIL_EXTRA_ARGUMENT",
    "FAIL_WRONG_ARGUMENT_KEY",
    "FAIL_WRONG_ARGUMENT_VALUE",
    "FAIL_TOO_FEW_CALLS",
    "FAIL_TOO_MANY_CALLS",
    "FAIL_PREMATURE_TERMINATION",
    "FAIL_NON_TERMINATION",
    "FAIL_CORRECT_ANSWER_UNSUPPORTED_TRACE",
    "FAIL_EXECUTABLE_WRONG_ANSWER",
    "FAIL_OTHER",
]


def _fail_blob(traj: Dict[str, Any]) -> str:
    parts = [str(traj.get("execution_error") or ""), str(traj.get("mismatch_reason") or "")]
    for t in traj.get("turns") or []:
        if t.get("fail_reason"):
            parts.append(str(t["fail_reason"]))
    return " | ".join(parts).lower()


def classify_trajectory(
    row: Dict[str, Any],
    *,
    gold_calls: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    traj = row.get("_traj") or {}
    gold_n = int(row.get("num_gold_calls") or traj.get("gold_num_turns") or (len(gold_calls) if gold_calls else 0))
    pred = _parsed_calls(traj)
    n_pred = len(pred)
    blob = _fail_blob(traj)
    flags: List[str] = []

    official = bool(as_bool(traj.get("official_win")))
    strict = bool(as_bool(row.get("strict_gold_trace_pass")))
    alt = bool(as_bool(row.get("alternative_valid_solution_pass")))
    sol_eq = bool(as_bool(row.get("solution_equivalent_pass")))
    ans = bool(as_bool(row.get("final_answer_pass")))
    unsupported = bool(as_bool(row.get("correct_answer_but_unsupported_trace")))

    candidates: List[str] = []

    if official and strict:
        candidates.append("SUCCESS_STRICT_GOLD")
    elif official and (alt or sol_eq):
        candidates.append("SUCCESS_ALTERNATIVE_VALID")
    elif official:
        candidates.append("SUCCESS_OTHER_OFFICIAL")

    if traj.get("parse_valid") is False or traj.get("stop_reason") == "parse_fail":
        candidates.append("FAIL_PARSE_INVALID")
        flags.append("parse_invalid")
    if n_pred == 0 and not official:
        candidates.append("FAIL_NO_TOOL_CALL")
        flags.append("no_tool_call")
    if as_bool(traj.get("clipped_any")):
        candidates.append("FAIL_CLIPPED_OUTPUT")
        flags.append("clipped")
    if as_bool(traj.get("prompt_overflow")):
        candidates.append("FAIL_PROMPT_OVERFLOW")
        flags.append("prompt_overflow")

    # reference heuristics from blob + parsed refs
    if "unresolved" in blob or "bad_ref" in blob or "missing_ref" in blob:
        if "unknown" in blob and "label" in blob:
            candidates.append("FAIL_REFERENCE_TO_UNKNOWN_LABEL")
        elif "output" in blob or "key" in blob:
            candidates.append("FAIL_REFERENCE_WRONG_OUTPUT_KEY")
        else:
            candidates.append("FAIL_REFERENCE_SYNTAX")
        flags.append("reference_issue")
    for c in pred:
        for v in (c.get("arguments") or {}).values() if isinstance(c.get("arguments"), dict) else []:
            if isinstance(v, str) and "$" in v and not is_reference(v):
                candidates.append("FAIL_REFERENCE_SYNTAX")
                flags.append("malformed_reference_string")

    if traj.get("executable") is False and "reference" not in " ".join(flags):
        if "unknown_tool" not in blob:
            candidates.append("FAIL_EXECUTOR_ERROR")
            flags.append("executor_error")

    gold_names = [str(c.get("name") or "") for c in (gold_calls or [])]
    pred_names = [str(c.get("name") or "") for c in pred]
    if gold_names and pred_names and not official:
        if pred_names[0] != gold_names[0]:
            candidates.append("FAIL_WRONG_FIRST_TOOL")
            flags.append("wrong_first_tool")
        elif pred_names != gold_names:
            if len(pred_names) == len(gold_names):
                # same length, different later
                if any(a != b for a, b in zip(pred_names, gold_names)):
                    if pred_names[0] == gold_names[0]:
                        candidates.append("FAIL_WRONG_LATER_TOOL")
                    candidates.append("FAIL_WRONG_TOOL_SEQUENCE")
            else:
                candidates.append("FAIL_WRONG_TOOL_SEQUENCE")
            flags.append("wrong_tool_sequence")

    if not official:
        if n_pred < gold_n:
            candidates.append("FAIL_TOO_FEW_CALLS")
            flags.append("too_few_calls")
        if n_pred > gold_n:
            candidates.append("FAIL_TOO_MANY_CALLS")
            flags.append("too_many_calls")
        stop = str(traj.get("stop_reason") or "")
        if stop in ("final_answer", "stop", "answer") and n_pred < gold_n:
            candidates.append("FAIL_PREMATURE_TERMINATION")
            flags.append("premature_termination")
        if stop in ("max_turns", "max_tokens", "length"):
            candidates.append("FAIL_NON_TERMINATION")
            flags.append("non_termination")

    if "missing" in blob and "arg" in blob:
        candidates.append("FAIL_MISSING_REQUIRED_ARGUMENT")
        flags.append("missing_arg")
    if "extra" in blob and "arg" in blob:
        candidates.append("FAIL_EXTRA_ARGUMENT")
        flags.append("extra_arg")
    if ("arg" in blob or "type" in blob or "schema" in blob) and traj.get("executable") is False:
        candidates.append("FAIL_WRONG_ARGUMENT_VALUE")
        flags.append("wrong_arg")

    if unsupported and not official:
        candidates.append("FAIL_CORRECT_ANSWER_UNSUPPORTED_TRACE")
        flags.append("unsupported_trace")
    if as_bool(traj.get("executable")) and not ans and not official:
        candidates.append("FAIL_EXECUTABLE_WRONG_ANSWER")
        flags.append("wrong_answer")

    if not candidates:
        candidates.append("FAIL_OTHER" if not official else "SUCCESS_OTHER_OFFICIAL")

    # choose primary by priority list
    primary = "FAIL_OTHER"
    for cat in PRIMARY_PRIORITY:
        if cat in candidates:
            primary = cat
            break

    secondary = sorted({c for c in candidates if c != primary})
    return {
        "primary_failure": primary,
        "secondary_flags": secondary,
        "all_flags": sorted(set(flags)),
        "official_win": official,
        "n_pred_calls": n_pred,
        "n_gold_calls": gold_n,
    }


def build_failure_tables(
    ids: Sequence[str],
    c0_rows: Dict[str, Dict[str, Any]],
    d1_rows: Dict[str, Dict[str, Any]],
    diag_gold: Dict[str, List[Dict[str, Any]]],
    outcomes: Dict[str, str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any], str]:
    per_task = []
    transitions = []
    matrix_counter: Counter = Counter()
    for sid in ids:
        gold = diag_gold.get(sid) or []
        c0c = classify_trajectory(c0_rows[sid], gold_calls=gold)
        d1c = classify_trajectory(d1_rows[sid], gold_calls=gold)
        out = outcomes[sid]
        per_task.append({
            "sample_id": sid,
            "outcome": out,
            "c0_primary": c0c["primary_failure"],
            "d1_primary": d1c["primary_failure"],
            "c0_secondary": "|".join(c0c["secondary_flags"]),
            "d1_secondary": "|".join(d1c["secondary_flags"]),
            "c0_flags": "|".join(c0c["all_flags"]),
            "d1_flags": "|".join(d1c["all_flags"]),
            "c0_win": int(c0c["official_win"]),
            "d1_win": int(d1c["official_win"]),
        })
        transitions.append({
            "sample_id": sid,
            "outcome": out,
            "c0_primary": c0c["primary_failure"],
            "d1_primary": d1c["primary_failure"],
            "transition": f"{c0c['primary_failure']} -> {d1c['primary_failure']}",
        })
        matrix_counter[(c0c["primary_failure"], d1c["primary_failure"])] += 1

    matrix_rows = [
        {"c0_primary": a, "d1_primary": b, "count": cnt}
        for (a, b), cnt in sorted(matrix_counter.items(), key=lambda x: -x[1])
    ]

    # summaries with absolute counts
    c0_dist = Counter(r["c0_primary"] for r in per_task)
    d1_dist = Counter(r["d1_primary"] for r in per_task)
    gained = [r for r in per_task if r["outcome"] == "loss_to_win"]
    lost = [r for r in per_task if r["outcome"] == "win_to_loss"]
    unchanged_fail = [r for r in per_task if r["outcome"] == "loss_to_loss"]

    removed = Counter()
    introduced = Counter()
    same_fail = Counter()
    for r in per_task:
        if r["c0_primary"].startswith("FAIL") and r["d1_primary"].startswith("SUCCESS"):
            removed[r["c0_primary"]] += 1
        if r["c0_primary"].startswith("SUCCESS") and r["d1_primary"].startswith("FAIL"):
            introduced[r["d1_primary"]] += 1
        if r["c0_primary"] == r["d1_primary"] and r["c0_primary"].startswith("FAIL"):
            same_fail[r["c0_primary"]] += 1

    summary = {
        "c0_primary_counts": dict(c0_dist),
        "d1_primary_counts": dict(d1_dist),
        "failures_removed_by_d1": dict(removed),
        "failures_introduced_by_d1": dict(introduced),
        "failures_unchanged_same_primary": dict(same_fail),
        "n_gained": len(gained),
        "n_lost": len(lost),
        "n_unchanged_fail": len(unchanged_fail),
        "top_transitions": [
            {"transition": f"{a} -> {b}", "count": cnt}
            for (a, b), cnt in matrix_counter.most_common(30)
        ],
    }

    md_lines = [
        "# FAILURE_ANALYSIS",
        "",
        "## C0 primary failure counts (absolute)",
        "",
    ]
    for k, v in c0_dist.most_common():
        md_lines.append(f"- `{k}`: {v}")
    md_lines += ["", "## D1 primary failure counts (absolute)", ""]
    for k, v in d1_dist.most_common():
        md_lines.append(f"- `{k}`: {v}")
    md_lines += ["", "## Failures removed by D1 (FAIL→SUCCESS)", ""]
    for k, v in removed.most_common():
        md_lines.append(f"- `{k}`: {v}")
    md_lines += ["", "## Failures introduced by D1 (SUCCESS→FAIL)", ""]
    for k, v in introduced.most_common():
        md_lines.append(f"- `{k}`: {v}")
    md_lines += ["", "## Top transitions", ""]
    for item in summary["top_transitions"][:20]:
        md_lines.append(f"- {item['transition']}: {item['count']}")
    md_lines += [
        "",
        "## Priority rule",
        "",
        "Primary category is the first matching label in PRIMARY_PRIORITY.",
        "",
    ]
    return per_task, transitions, matrix_rows, summary, "\n".join(md_lines) + "\n"
