"""Hard workflow/program semantic validators."""
from __future__ import annotations

from typing import Any, Dict, List

from ..executor import execute, replay_consistent
from ..schemas import GraphNode, SemanticProgram
from .primitives_v2 import bind_capability
from .semantic_types import semantic_compatible, unit_compatible
from .workflows_v2 import workflows_by_id


def _program(record: Dict[str, Any]) -> SemanticProgram:
    sp = record["semantic_program"]
    return SemanticProgram(
        nodes=[GraphNode(node_id=n["node_id"], semantic_id=n["primitive_id"],
                         inputs=n["inputs"], output_type=n["output_type"])
               for n in sp["nodes"]],
        sink=sp["sink"], motif=record["pattern_family"],
        depth=max(0, len(sp["nodes"]) - 1))


def workflow_program_query_alignment(record: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    blueprint = workflows_by_id().get(record.get("workflow_id"))
    if blueprint is None:
        return ["unknown workflow_id"]
    if not record.get("was_generated_from_workflow"):
        errors.append("workflow-first provenance flag absent")
    if record.get("workflow_instance", {}).get("workflow_id") != blueprint.workflow_id:
        errors.append("workflow instance mismatch")
    if record.get("pattern_family") not in blueprint.allowed_structural_patterns:
        errors.append("structural pattern outside workflow")
    n = len(record.get("semantic_program", {}).get("nodes", []))
    lo, hi = blueprint.allowed_call_count_range
    if not lo <= n <= hi:
        errors.append("call count outside workflow")
    nodes = record.get("semantic_program", {}).get("nodes", [])
    if len(nodes) != len(blueprint.plan_template):
        errors.append("semantic plan length mismatch")
    for node, plan in zip(nodes, blueprint.plan_template):
        if node.get("capability") != plan.capability:
            errors.append(f"{node.get('node_id')}: capability differs from plan")
        allowed = bind_capability(plan.capability).primitive_id
        if node.get("primitive_id") != allowed:
            errors.append(f"{node.get('node_id')}: primitive not bound to capability")
        if tuple(node.get("input_roles") or []) != plan.input_roles:
            errors.append(f"{node.get('node_id')}: input roles differ from plan")
        if node.get("output_role") != plan.output_role:
            errors.append(f"{node.get('node_id')}: output role differs from plan")
    if not nodes or nodes[-1].get("output_role") != blueprint.target_role:
        errors.append("sink does not produce workflow target")
    contract = record.get("query_contract") or {}
    if contract.get("workflow_id") != blueprint.workflow_id:
        errors.append("query contract workflow mismatch")
    if contract.get("target_role") != blueprint.target_role:
        errors.append("query target mismatch")
    return errors


def semantic_edges(record: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    nodes = record.get("semantic_program", {}).get("nodes", [])
    outputs = {n["output_role"]: n for n in nodes}
    facts = (record.get("workflow_instance") or {}).get("facts") or {}
    for node in nodes:
        for role, expected in zip(node.get("input_roles", []),
                                  node.get("input_semantic_types", [])):
            if role in facts:
                actual, unit = facts[role]["semantic_type"], facts[role].get("unit", "")
            elif role in outputs and nodes.index(outputs[role]) < nodes.index(node):
                actual, unit = outputs[role]["output_semantic_type"], ""
            else:
                errors.append(f"{node['node_id']}: role {role} has no prior producer")
                continue
            ok, reason = semantic_compatible(actual, expected)
            if not ok:
                errors.append(f"{node['node_id']}:{role}:{reason}")
            expected_unit = facts.get(role, {}).get("unit", "")
            ok, reason = unit_compatible(unit, expected_unit)
            if not ok:
                errors.append(f"{node['node_id']}:{role}:{reason}")
    return errors


def executor_replay(record: Dict[str, Any]) -> List[str]:
    try:
        program = _program(record)
        observations, answer = execute(program)
        if observations != record.get("oracle_observations"):
            return ["oracle observations differ on replay"]
        if answer != record.get("gold_answer"):
            return ["gold answer differs on replay"]
        if not replay_consistent(program, 2):
            return ["executor replay is nondeterministic"]
        return []
    except Exception as exc:  # noqa: BLE001
        return [f"executor failure: {exc}"]


def node_necessity(record: Dict[str, Any]) -> List[str]:
    program = _program(record)
    target = record.get("gold_answer")
    errors: List[str] = []
    for removed in program.nodes:
        kept = [n for n in program.nodes if n.node_id != removed.node_id]
        if not kept or removed.node_id == program.sink:
            continue
        try:
            _, answer = execute(SemanticProgram(
                nodes=kept, sink=program.sink, motif=program.motif, depth=program.depth))
            if answer == target:
                errors.append(f"{removed.node_id}: deletion leaves answer unchanged")
        except Exception:
            pass
    return errors


def validate_record(record: Dict[str, Any]) -> Dict[str, Any]:
    checks = {
        "V_WORKFLOW_PROGRAM_QUERY_ALIGNMENT": workflow_program_query_alignment(record),
        "V_SEMANTIC_EDGE_UNIT_ROLE": semantic_edges(record),
        "V_EXECUTOR_REPLAY_X2": executor_replay(record),
        "V_NODE_NECESSITY": node_necessity(record),
    }
    layers = {name: {"passed": not errors, "reasons": errors}
              for name, errors in checks.items()}
    return {"passed": all(v["passed"] for v in layers.values()), "layers": layers}
