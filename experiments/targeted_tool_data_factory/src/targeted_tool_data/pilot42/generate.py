"""WorkflowBlueprint -> instance -> semantic plan -> typed executable DAG."""
from __future__ import annotations

import random
from typing import Any, Dict, Iterable, List

from .. import registry as reg
from ..executor import execute
from ..graph import REF
from ..schemas import GraphNode, SemanticProgram
from ..util import short_hash
from . import QUERY_MODES
from .primitives_v2 import bind_capability
from .query_contract import build_query_contract
from .query_render import query_template_fingerprint, render_query
from .semantic_types import SemanticType, TypedValue, semantic_compatible
from .workflows_v2 import WorkflowBlueprint, get_workflows, workflows_by_id


def _make_value(role: str, semantic_type: str, rng: random.Random) -> TypedValue:
    st = SemanticType.parse(semantic_type)
    if st.name == "Percentage":
        value, unit = rng.randint(5, 35), "percent"
    elif st.name == "Money":
        value, unit = rng.randint(80, 1800), "USD"
    elif st.name == "Boolean":
        value, unit = bool(rng.randrange(2)), ""
    elif st.name == "Count":
        value, unit = rng.randint(3, 90), "items"
    else:
        value, unit = rng.randint(10, 900), ""
    return TypedValue(value=value, semantic_type=st, role=role, unit=unit)


def instantiate_workflow(blueprint: WorkflowBlueprint, seed: int) -> Dict[str, Any]:
    rng = random.Random(f"pilot42-instance:{blueprint.workflow_id}:{seed}")
    entity = rng.choice(blueprint.entity_pools)
    facts = {}
    for role in blueprint.input_roles:
        tv = _make_value(role, blueprint.role_semantic_types[role], rng)
        facts[role] = tv.as_dict()
    return {"workflow_id": blueprint.workflow_id, "entity": entity, "facts": facts,
            "instance_id": "wi42_" + short_hash([blueprint.workflow_id, seed, facts])}


def _tool(primitive_id: str) -> Dict[str, Any]:
    p = reg.get(primitive_id)
    surface = p.surfaces_g[0] if p.surfaces_g else p.surfaces_a[0]
    properties = {}
    for (canonical, typ, semantic), shown in zip(p.params, surface.param_names):
        properties[shown] = {"type": typ.split(":")[0], "description": semantic}
    return {"name": surface.name, "description": surface.description,
            "parameters": {"type": "object", "properties": properties,
                           "required": list(properties)},
            "output_field": surface.output_field, "output_type": p.out_type,
            "semantic_id": primitive_id, "is_distractor": False}


def generate_program_from_workflow(
        blueprint: WorkflowBlueprint, instance: Dict[str, Any],
        structural_skill: str, difficulty: str, seed: int) -> Dict[str, Any]:
    if instance.get("workflow_id") != blueprint.workflow_id:
        raise ValueError("workflow instance/blueprint mismatch")
    if structural_skill not in blueprint.allowed_structural_patterns:
        raise ValueError(f"pattern {structural_skill} not allowed by {blueprint.workflow_id}")
    roles: Dict[str, Dict[str, Any]] = dict(instance["facts"])
    nodes: List[GraphNode] = []
    typed_nodes: List[Dict[str, Any]] = []
    gold_calls: List[Dict[str, Any]] = []
    tools: Dict[str, Dict[str, Any]] = {}
    producer: Dict[str, str] = {}  # role -> node_id
    producer_label: Dict[str, str] = {}  # role -> $var_k
    producer_out_field: Dict[str, str] = {}
    for index, plan in enumerate(blueprint.plan_template):
        binding = bind_capability(plan.capability, seed=seed + index)
        primitive = reg.get(binding.primitive_id)
        if len(primitive.params) != len(plan.input_roles):
            raise ValueError(f"{binding.primitive_id}: arity does not match semantic plan")
        inputs: Dict[str, Any] = {}
        call_args: Dict[str, Any] = {}
        surface = primitive.surfaces_g[0] if primitive.surfaces_g else primitive.surfaces_a[0]
        for pos, (role, expected) in enumerate(zip(
                plan.input_roles, plan.input_semantic_types)):
            if role not in roles:
                raise ValueError(f"plan role {role!r} has no fact or prior output")
            actual = roles[role]["semantic_type"]
            ok, reason = semantic_compatible(actual, expected)
            if not ok:
                raise ValueError(f"semantic input mismatch {role}: {reason}")
            pname = primitive.params[pos][0]
            shown = surface.param_names[pos]
            if role in producer:
                inputs[pname] = {REF: producer[role]}
                call_args[shown] = (
                    f"{producer_label[role]}.{producer_out_field[role]}$")
            else:
                inputs[pname] = roles[role]["value"]
                call_args[shown] = roles[role]["value"]
        node_id = f"n{index}"
        label = f"$var_{index + 1}"
        node = GraphNode(node_id=node_id, semantic_id=binding.primitive_id,
                         inputs=inputs, output_type=primitive.out_type)
        nodes.append(node)
        producer[plan.output_role] = node_id
        producer_label[plan.output_role] = label
        producer_out_field[plan.output_role] = surface.output_field
        roles[plan.output_role] = {
            "semantic_type": plan.output_semantic_type, "role": plan.output_role,
            "unit": next((roles[r].get("unit", "") for r in plan.input_roles
                          if roles[r].get("unit") not in ("percent", "")), ""),
            "value": None,
        }
        typed_nodes.append({
            "node_id": node_id, "primitive_id": binding.primitive_id,
            "capability": plan.capability, "capability_family": binding.capability_family,
            "inputs": inputs, "input_roles": list(plan.input_roles),
            "input_semantic_types": list(plan.input_semantic_types),
            "output_role": plan.output_role,
            "output_semantic_type": plan.output_semantic_type,
            "output_type": primitive.out_type,
        })
        tool = _tool(binding.primitive_id)
        tools[tool["name"]] = tool
        gold_calls.append({"name": tool["name"], "arguments": call_args,
                           "label": label})
    if not nodes or producer.get(blueprint.target_role) != nodes[-1].node_id:
        raise ValueError("semantic plan sink does not produce workflow target")
    program = SemanticProgram(nodes=nodes, sink=nodes[-1].node_id,
                              motif=structural_skill, depth=len(nodes) - 1)
    observations, answer = execute(program)
    mode = QUERY_MODES[seed % len(QUERY_MODES)]
    contract = build_query_contract(instance, blueprint, mode)
    question = render_query(contract, mode)
    semantic_program_id = "sp42_" + short_hash({
        "workflow_id": blueprint.workflow_id,
        "nodes": [n.model_dump() for n in nodes], "sink": program.sink})
    return {
        "schema_version": "ttdf.pilot42.record.v1",
        "task_id": "p42_" + short_hash([semantic_program_id, mode]),
        "workflow_id": blueprint.workflow_id, "workflow_instance_id": instance["instance_id"],
        "was_generated_from_workflow": True, "semantic_plan_id": "plan42_" + short_hash(
            [blueprint.workflow_id, [n.capability for n in blueprint.plan_template]]),
        "primitive_binding_ids": [n["primitive_id"] for n in typed_nodes],
        "semantic_program_id": semantic_program_id,
        "program_family_id": "pf42_" + short_hash(
            [blueprint.workflow_id, structural_skill,
             [n["primitive_id"] for n in typed_nodes]]),
        "pattern_family": structural_skill, "structural_skill": structural_skill,
        "difficulty": difficulty, "call_count": len(nodes),
        "semantic_program": {"nodes": typed_nodes, "sink": program.sink},
        "workflow_instance": instance, "query_contract": contract,
        "question": question, "query": question, "requested_query_mode": mode,
        "query_template_fingerprint": query_template_fingerprint(question),
        "gold_calls": gold_calls, "tools": list(tools.values()),
        "oracle_observations": observations, "gold_answer": answer,
        "verifier_spec": {"kind": "exact_or_tolerance", "oracle": answer,
                          "tolerance": 1e-6 if isinstance(answer, float) else 0},
        "provenance": {"workflow_id": blueprint.workflow_id,
                       "instance_id": instance["instance_id"],
                       "semantic_plan_source": "workflow_blueprint",
                       "generator_seed": seed},
    }


def _blueprint_for_cell(cell: Any, index: int) -> WorkflowBlueprint:
    if isinstance(cell, WorkflowBlueprint):
        return cell
    wid = cell.get("workflow_id") if isinstance(cell, dict) else getattr(cell, "workflow_id", None)
    if wid:
        return workflows_by_id()[wid]
    return get_workflows()[index % len(get_workflows())]


def generate_semantic_pool(cells: Iterable[Any], candidate_target: int = 20_000,
                           seed: int = 20260731,
                           max_attempts_factor: int = 8) -> List[Dict[str, Any]]:
    cells = list(cells) or get_workflows()
    rows: List[Dict[str, Any]] = []
    seen = set()
    attempts = 0
    while len(rows) < candidate_target and attempts < candidate_target * max_attempts_factor:
        cell = cells[attempts % len(cells)]
        blueprint = _blueprint_for_cell(cell, attempts)
        if hasattr(cell, "structural_skill") and cell.structural_skill in (
                blueprint.allowed_structural_patterns):
            pattern = cell.structural_skill
        else:
            pattern = blueprint.allowed_structural_patterns[
                attempts % len(blueprint.allowed_structural_patterns)]
        mode = getattr(cell, "query_mode", None) or QUERY_MODES[attempts % len(QUERY_MODES)]
        try:
            instance = instantiate_workflow(blueprint, seed + attempts)
            row = generate_program_from_workflow(
                blueprint, instance, pattern,
                blueprint.difficulty_variants[attempts % len(blueprint.difficulty_variants)],
                seed + attempts)
            # re-render with cell query mode for coverage
            from .query_contract import build_query_contract
            contract = build_query_contract(instance, blueprint, mode)
            question = render_query(contract, mode)
            row["query_contract"] = contract
            row["question"] = question
            row["query"] = question
            row["requested_query_mode"] = mode
            row["query_template_fingerprint"] = query_template_fingerprint(question)
            row["generation_cell"] = getattr(cell, "cell_id", f"cell_{attempts}")
            row["cell_tier"] = getattr(cell, "tier", "CORE_PROFILE")
            row["call_bucket"] = str(row["call_count"] if row["call_count"] < 6 else "6+")
            row["surface_track"] = getattr(cell, "track", "A_NATIVE")
            row["capability_families"] = sorted({
                n["capability_family"] for n in row["semantic_program"]["nodes"]})
            dedupe = (row["semantic_program_id"], mode, row["generation_cell"])
            if dedupe not in seen:
                seen.add(dedupe)
                row["task_id"] = "p42_" + short_hash(list(dedupe))
                rows.append(row)
        except (ValueError, ArithmeticError, KeyError, ZeroDivisionError):
            pass
        attempts += 1
    return rows
