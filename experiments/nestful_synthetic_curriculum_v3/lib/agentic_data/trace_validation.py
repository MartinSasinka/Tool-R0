"""Hard structural validation of gold-call traces.

Independent of execution — runs BEFORE the (more expensive) deterministic
executor so malformed traces never reach it, and again as defense-in-depth
over every accepted row before the final dataset is written.

Enforces (per the 2026-07-11 hardening audit):
  * unique call labels — a label may never be reused/overwritten;
  * sequential labels — call i must be labeled exactly "$var{i+1}$";
  * references may only point to a PREVIOUS call's label (never itself or a
    future call);
  * the referenced output field must be the field the producing tool
    actually emits (not just any string that looks like a reference);
  * the gold call count must match both the stage's expected range and the
    challenger's own declared `tool_names` length.

Two real (accepted) pilot rows motivated this module:
  agentic_v4_stage2_000007: gold_calls labeled ["$var1", "$var1"] (duplicate,
    non-sequential — the second call should have been "$var2").
  agentic_v4_stage2_000008: same duplicate-label defect
    (["$var1", "$var1"]), also referencing "$var1.result$" from within the
    call that OWNS that same label.
Both executed successfully (the executor tolerates label reuse because the
scope dict is only overwritten AFTER the referencing happens), which is
exactly why a structural check is needed on top of execution success.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

_REF_RE = re.compile(r"^\$([A-Za-z_]\w*)\.(\w+)\$$")


def expected_label(index: int) -> str:
    """0-based call index -> the ONLY label it may legally use."""
    return f"$var{index + 1}"


def _norm_label(raw: Any) -> str:
    return "$" + str(raw if raw is not None else "").lstrip("$")


def label_errors(gold_calls: List[Dict[str, Any]]) -> List[str]:
    """Unique + sequential label checks ($var1, $var2, ... in call order)."""
    errs: List[str] = []
    seen: set = set()
    for i, call in enumerate(gold_calls):
        if not isinstance(call, dict):
            errs.append(f"call {i + 1}: not an object")
            continue
        label = _norm_label(call.get("label"))
        want = expected_label(i)
        if label != want:
            errs.append(f"call {i + 1}: label {label!r} must be sequential "
                        f"{want!r}")
        if label in seen:
            errs.append(f"call {i + 1}: label {label!r} reused — a label "
                        "may never be overwritten by a later call")
        seen.add(label)
    return errs


def _valid_ref_fields(tool_spec: Dict[str, Any]) -> Optional[set]:
    """Legal `.field$` names for a producer tool, or ``None`` if the tool is
    unknown (skip the field check — caller has no schema to validate against).

    Object-typed outputs (``out_fields`` set) may ONLY be referenced by one
    of their nested field names (e.g. ``$var1.area$``, never ``$var1.result$``
    even though ``result`` is the tool's own ``out_key``). Scalar outputs may
    only be referenced by their single ``out_key``."""
    if not tool_spec:
        return None
    out_fields = tool_spec.get("out_fields")
    if out_fields:
        return set(out_fields.keys())
    out_key = tool_spec.get("out_key")
    return {out_key} if out_key is not None else None


def reference_errors(gold_calls: List[Dict[str, Any]],
                     tools: Dict[str, Dict[str, Any]]) -> List[str]:
    """Every $varN.key$ reference must point to a PRIOR call's label, using
    a field the call's tool actually exposes (its sole ``out_key`` for a
    scalar output, or one of its ``out_fields`` for an object output)."""
    errs: List[str] = []
    label_to_tool: Dict[str, str] = {}
    for i, call in enumerate(gold_calls):
        if not isinstance(call, dict):
            continue
        name = call.get("name")
        args = call.get("arguments")
        own_label = _norm_label(call.get("label"))
        if isinstance(args, dict):
            for k, v in args.items():
                if not isinstance(v, str):
                    continue
                m = _REF_RE.match(v.strip())
                if not m:
                    continue
                ref_label, ref_key = "$" + m.group(1), m.group(2)
                if ref_label == own_label:
                    errs.append(f"call {i + 1} arg {k!r}: reference {v!r} "
                                "points to its OWN label (self-reference)")
                    continue
                if ref_label not in label_to_tool:
                    errs.append(
                        f"call {i + 1} arg {k!r}: reference {v!r} does not "
                        "point to any PRIOR call's label (forward/unknown "
                        "reference)")
                    continue
                producer = label_to_tool[ref_label]
                valid_fields = _valid_ref_fields(tools.get(producer))
                if valid_fields is not None and ref_key not in valid_fields:
                    errs.append(
                        f"call {i + 1} arg {k!r}: reference {v!r} uses field "
                        f"'.{ref_key}$' but {ref_label} ({producer}) only "
                        f"outputs {sorted(valid_fields)}")
        if name:
            label_to_tool[own_label] = name
    return errs


def call_count_errors(gold_calls: List[Dict[str, Any]], cand: Dict[str, Any],
                      expected_range: Tuple[int, int]) -> List[str]:
    errs: List[str] = []
    lo, hi = expected_range
    n = len(gold_calls)
    if not (lo <= n <= hi):
        errs.append(f"call count {n} outside expected stage range [{lo},{hi}]")
    tool_names = cand.get("tool_names")
    if isinstance(tool_names, list) and len(tool_names) != n:
        errs.append(f"declared tool_names length {len(tool_names)} != "
                    f"gold_calls length {n}")
    return errs


def hard_trace_errors(cand: Dict[str, Any], tools: Dict[str, Dict[str, Any]],
                      expected_range: Tuple[int, int]) -> List[str]:
    """All hard structural checks combined; used both as a generation-time
    gate (cheap, pre-execution) and as final defense-in-depth validation over
    every accepted row. ANY violation must reject/flag the row — there is no
    partial credit for trace structure."""
    calls = cand.get("gold_calls")
    if not isinstance(calls, list) or not calls:
        return ["gold_calls missing/empty/not a list"]
    errs = call_count_errors(calls, cand, expected_range)
    errs += label_errors(calls)
    errs += reference_errors(calls, tools)
    return errs
