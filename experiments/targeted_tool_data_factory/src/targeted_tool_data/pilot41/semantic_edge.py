"""Semantic edge validation: runtime ∧ semantic ∧ unit ∧ workflow ∧ role."""
from __future__ import annotations

from typing import Any, Dict, Optional

from .. import registry as reg
from .primitive_semantics import semantics_for
from .semantic_types import (SemanticType, TypedValue, semantic_compatible,
                             unit_compatible)


def validate_semantic_edge(
        source_node: Dict[str, Any],
        target_node: Dict[str, Any],
        target_arg: str,
        workflow_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Validate one DAG edge from source output into a target parameter."""
    ctx = workflow_context or {}
    src_sid = source_node.get("primitive_id") or source_node.get("semantic_id")
    tgt_sid = target_node.get("primitive_id") or target_node.get("semantic_id")
    src_sem = semantics_for(str(src_sid))
    tgt_sem = semantics_for(str(tgt_sid))

    # locate target arg index
    tgt_prim = reg.get(str(tgt_sid))
    arg_idx = next((i for i, (n, _t, _s) in enumerate(tgt_prim.params)
                    if n == target_arg), 0)
    if arg_idx >= len(tgt_sem.semantic_input_types):
        dst_type = SemanticType.parse("GenericScalar")
        dst_runtime = "number"
        dst_unit = ""
    else:
        dst_type = tgt_sem.semantic_input_types[arg_idx]
        dst_runtime = tgt_sem.runtime_input_types[arg_idx]
        dst_unit = (tgt_prim.param_units[arg_idx]
                    if arg_idx < len(tgt_prim.param_units) else "")

    src_type = src_sem.semantic_output_type
    # workflow may already have typed the source variable
    typed = (ctx.get("typed_outputs") or {}).get(
        source_node.get("node_id") or source_node.get("label"))
    if typed:
        if isinstance(typed, TypedValue):
            src_type = typed.semantic_type
            src_unit = typed.unit
            src_runtime = typed.runtime_type
        elif isinstance(typed, dict):
            src_type = SemanticType.parse(str(typed.get("semantic_type")))
            src_unit = str(typed.get("unit") or "")
            src_runtime = str(typed.get("runtime_type") or src_sem.runtime_output_type)
        else:
            src_unit = ""
            src_runtime = src_sem.runtime_output_type
    else:
        src_unit = ""
        src_runtime = src_sem.runtime_output_type

    # runtime
    runtime_ok = _runtime_ok(src_runtime, dst_runtime)
    # semantic — GenericScalar→Duration* forbidden unless conversion or typed
    allow_generic = bool(ctx.get("allow_generic_numeric", False))
    if dst_type.name.startswith("Duration") and src_type.name == "GenericScalar":
        if not ctx.get("source_is_duration"):
            sem_ok, sem_reason = False, "forbidden_generic_to_Duration"
        else:
            sem_ok, sem_reason = True, "workflow_typed_duration"
    else:
        sem_ok, sem_reason = semantic_compatible(
            src_type, dst_type, allow_generic=allow_generic)

    unit_ok, unit_reason = unit_compatible(
        src_unit, dst_unit, unit_behavior=tgt_sem.unit_behavior)

    # workflow family gate
    wf_family = str(ctx.get("workflow_family") or ctx.get("domain") or "*")
    allowed = tgt_sem.allowed_workflow_families
    wf_ok = ("*" in allowed) or (wf_family in allowed) or not allowed
    src_cap = src_sem.capability_family
    if src_cap in tgt_sem.forbidden_successor_families:
        wf_ok = False
        wf_reason = f"forbidden_successor_{src_cap}"
    else:
        wf_reason = "workflow_ok" if wf_ok else f"workflow_{wf_family}_not_in_{allowed}"

    # role
    role_ok = True
    role_reason = "role_unconstrained"
    need_roles = tgt_sem.allowed_input_roles
    src_role = ""
    if typed and isinstance(typed, TypedValue):
        src_role = typed.role
    elif isinstance(typed, dict):
        src_role = str(typed.get("role") or "")
    if need_roles and src_role and src_role not in need_roles:
        role_ok = False
        role_reason = f"role_{src_role}_not_in_{need_roles}"

    accepted = all((runtime_ok, sem_ok, unit_ok, wf_ok, role_ok))
    return {
        "runtime_compatible": runtime_ok,
        "semantic_compatible": sem_ok,
        "unit_compatible": unit_ok,
        "workflow_compatible": wf_ok,
        "role_compatible": role_ok,
        "accepted": accepted,
        "reason": (sem_reason if not sem_ok else
                   unit_reason if not unit_ok else
                   wf_reason if not wf_ok else
                   role_reason if not role_ok else
                   "accepted"),
        "source_semantic": str(src_type),
        "target_semantic": str(dst_type),
        "target_arg": target_arg,
    }


def _runtime_ok(src: str, dst: str) -> bool:
    if src == dst:
        return True
    if dst == "number" and src in ("number", "integer"):
        return True
    if dst.startswith("enum:"):
        return False
    if dst == "array" and src == "array":
        return True
    return False


def validate_program_edges(nodes: list, edges: list,
                           workflow_context: Optional[Dict[str, Any]] = None
                           ) -> Dict[str, Any]:
    """Validate every edge; return aggregate + per-edge results."""
    by_id = {n.get("node_id"): n for n in nodes}
    results = []
    for e in edges:
        src = by_id.get(e.get("from"))
        tgt = by_id.get(e.get("to"))
        if not src or not tgt:
            results.append({"accepted": False, "reason": "missing_node",
                            "edge": e})
            continue
        results.append(validate_semantic_edge(
            src, tgt, str(e.get("param") or "arg_0"), workflow_context))
    n = len(results) or 1
    return {
        "n_edges": len(results),
        "n_accepted": sum(1 for r in results if r.get("accepted")),
        "rejection_rate": round(
            sum(1 for r in results if not r.get("accepted")) / n, 4),
        "edges": results,
        "all_accepted": all(r.get("accepted") for r in results) if results else True,
    }
