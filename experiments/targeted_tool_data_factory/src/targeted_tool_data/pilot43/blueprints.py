"""Workflow blueprint DSL: the plan owns the edges, so the workflow *is* the program.

A blueprint holds one or more :class:`Plan` variants. A plan is an explicit
capability DAG: every step names the capability it needs and, per parameter,
where the value comes from -- either a named input role (a fact the user states)
or ``@node`` (the output of an earlier step). There is no post-hoc pattern
label and no post-hoc capability label: call count, structural pattern, answer
type and capability coverage are all derived by building and executing the plan.

Blueprint modules live in :mod:`targeted_tool_data.pilot43.wf` and are collected
by :func:`all_blueprints`.
"""
from __future__ import annotations

import importlib
import os
import pkgutil
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from ..repro import sha256_obj
from . import SURFACE_TRACKS
from . import semtypes as st
from .ops import build_ops, ops_by_capability
from .values import CALIBRATED_HINTS, sem_of_hint


class BlueprintError(ValueError):
    """A blueprint that cannot be trusted to generate what it declares."""


@dataclass(frozen=True)
class Role:
    """A fact the user supplies in the query."""
    name: str
    hint: str
    description: str = ""

    @property
    def sem(self) -> str:
        return sem_of_hint(self.hint)

    @property
    def calibrated(self) -> bool:
        return self.hint in CALIBRATED_HINTS


@dataclass(frozen=True)
class Step:
    node_id: str
    capability: str
    args: Tuple[str, ...]
    purpose: str = ""

    def refs(self) -> Tuple[str, ...]:
        return tuple(a[1:] for a in self.args if a.startswith("@"))

    def roles(self) -> Tuple[str, ...]:
        return tuple(a for a in self.args if not a.startswith("@"))


@dataclass(frozen=True)
class Plan:
    plan_id: str
    roles: Tuple[Role, ...]
    steps: Tuple[Step, ...]
    sink: str
    intent: str = ""

    @property
    def call_count(self) -> int:
        return len(self.steps)

    def role(self, name: str) -> Role:
        for r in self.roles:
            if r.name == name:
                return r
        raise BlueprintError(f"unknown role {name}")

    def step(self, node_id: str) -> Step:
        for s in self.steps:
            if s.node_id == node_id:
                return s
        raise BlueprintError(f"unknown step {node_id}")

    def capability_plan(self) -> Tuple[str, ...]:
        return tuple(s.capability for s in self.steps)

    def normalized_capability_plan(self) -> Tuple[str, ...]:
        """Capability families in call order; used by the program-plan holdout."""
        return tuple(s.capability.split(".")[0] for s in self.steps)


#: Blueprint modules were written independently and drifted into two names for the
#: same domain ("paths" and "path_processing"). Reporting both would inflate the
#: domain count, so every domain label is folded onto the taxonomy the spec names.
DOMAIN_ALIASES = {
    "arithmetic_core": "arithmetic",
    "dates": "date_time",
    "files": "file_processing",
    "paths": "path_processing",
    "urls": "url_processing",
    "text": "text_processing",
    "rates_ratios": "rates_and_ratios",
    "statistics_summary": "statistics",
    "resources": "resource_allocation",
    "validation_rules": "validation",
}


def canonical_domain(name: str) -> str:
    return DOMAIN_ALIASES.get(name, name)


@dataclass(frozen=True)
class Blueprint:
    workflow_id: str
    domain: str
    natural_user_goal: str
    target_description: str
    plans: Tuple[Plan, ...]
    value_generator_id: str
    query_asset_family: str
    hard_distractor_families: Tuple[str, ...] = ()
    surface_compatibility: Tuple[str, ...] = SURFACE_TRACKS
    boolean_balancing_strategy: str | None = None
    entity_family: str = "operations"

    @property
    def family(self) -> str:
        return self.workflow_id.split(".")[0]

    def allowed_call_counts(self) -> Tuple[int, ...]:
        return tuple(sorted({p.call_count for p in self.plans}))

    def capability_families(self) -> Tuple[str, ...]:
        out = set()
        for p in self.plans:
            out.update(p.normalized_capability_plan())
        return tuple(sorted(out))

    def coding_like(self) -> bool:
        from .ops import CODING_FAMILIES
        return any(f in CODING_FAMILIES for f in self.capability_families())


def validate_blueprint(bp: Blueprint) -> List[str]:
    """Fail-closed checks. A blueprint that cannot bind is never used."""
    caps = ops_by_capability()
    ops = build_ops()
    errs: List[str] = []
    if not bp.plans:
        errs.append(f"{bp.workflow_id}: no plans")
    for plan in bp.plans:
        tag = f"{bp.workflow_id}/{plan.plan_id}"
        seen: Dict[str, int] = {}
        role_names = {r.name for r in plan.roles}
        if len(role_names) != len(plan.roles):
            errs.append(f"{tag}: duplicate role names")
        used_roles: set[str] = set()
        for i, step in enumerate(plan.steps):
            if step.node_id in seen:
                errs.append(f"{tag}: duplicate node {step.node_id}")
            seen[step.node_id] = i
            candidates = [pid for pid in caps.get(step.capability, [])
                          if ops[pid].arity == len(step.args)]
            if not candidates:
                errs.append(f"{tag}/{step.node_id}: no op for "
                            f"{step.capability} with arity {len(step.args)}")
            for arg in step.args:
                if arg.startswith("@"):
                    ref = arg[1:]
                    if ref not in seen or seen[ref] >= i:
                        errs.append(f"{tag}/{step.node_id}: bad ref {arg}")
                elif arg not in role_names:
                    errs.append(f"{tag}/{step.node_id}: unknown role {arg}")
                else:
                    used_roles.add(arg)
        if plan.sink not in seen:
            errs.append(f"{tag}: sink {plan.sink} is not a step")
        idle = role_names - used_roles
        if idle:
            errs.append(f"{tag}: roles never used: {sorted(idle)}")
        # every step must reach the sink
        reach = {plan.sink}
        for step in reversed(plan.steps):
            if step.node_id in reach:
                reach.update(step.refs())
        dead = [s.node_id for s in plan.steps if s.node_id not in reach]
        if dead:
            errs.append(f"{tag}: steps not feeding the sink: {dead}")
    return errs


# ── registry ─────────────────────────────────────────────────────────────
_CACHE: Tuple[Blueprint, ...] | None = None


def loaded_modules() -> Tuple[str, ...]:
    """Blueprint modules that will be imported, in load order.

    ``P43_WF_MODULES`` restricts the set to a comma-separated allowlist. It exists
    so a single family can be developed while others are mid-edit; the pipeline
    refuses to run with it set (see :func:`assert_full_registry`) because a
    silently smaller registry would invalidate every coverage number.
    """
    from . import wf

    names = sorted(m.name for m in pkgutil.iter_modules(wf.__path__))
    allow = os.environ.get("P43_WF_MODULES", "").strip()
    if allow:
        wanted = {n.strip() for n in allow.split(",") if n.strip()}
        names = [n for n in names if n in wanted]
    return tuple(names)


def assert_full_registry() -> None:
    if os.environ.get("P43_WF_MODULES", "").strip():
        raise BlueprintError(
            "P43_WF_MODULES is set: the registry would be a development subset. "
            "Unset it before generating, selecting or auditing.")


def all_blueprints(*, reload: bool = False) -> Tuple[Blueprint, ...]:
    global _CACHE
    if _CACHE is not None and not reload:
        return _CACHE
    from . import wf

    found: List[Blueprint] = []
    for mod in loaded_modules():
        module = importlib.import_module(f"{wf.__name__}.{mod}")
        if reload:
            module = importlib.reload(module)
        blueprints = getattr(module, "blueprints", None)
        if blueprints is None:
            continue
        found.extend(blueprints())
    found = [bp if bp.domain == canonical_domain(bp.domain)
             else replace(bp, domain=canonical_domain(bp.domain))
             for bp in found]
    ids = [bp.workflow_id for bp in found]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise BlueprintError(f"duplicate workflow ids: {dupes}")
    errs: List[str] = []
    for bp in found:
        errs.extend(validate_blueprint(bp))
    if errs:
        raise BlueprintError("invalid blueprints: " + "; ".join(errs[:20]))
    _CACHE = tuple(sorted(found, key=lambda b: b.workflow_id))
    return _CACHE


def by_id(workflow_id: str) -> Blueprint:
    for bp in all_blueprints():
        if bp.workflow_id == workflow_id:
            return bp
    raise BlueprintError(f"unknown workflow {workflow_id}")


def registry_hash() -> str:
    payload = []
    for bp in all_blueprints():
        payload.append({
            "workflow_id": bp.workflow_id,
            "domain": bp.domain,
            "goal": bp.natural_user_goal,
            "generator": bp.value_generator_id,
            "plans": [{
                "plan_id": p.plan_id,
                "roles": [(r.name, r.hint) for r in p.roles],
                "steps": [(s.node_id, s.capability, list(s.args)) for s in p.steps],
                "sink": p.sink,
            } for p in bp.plans],
        })
    return sha256_obj(payload)


def export_registry(samples: Dict[str, Dict[str, Any]] | None = None
                    ) -> Dict[str, Any]:
    """Export the registry with *derived* (never declared) plan properties."""
    rows = []
    for bp in all_blueprints():
        derived = (samples or {}).get(bp.workflow_id, {})
        rows.append({
            "workflow_id": bp.workflow_id,
            "domain": bp.domain,
            "natural_user_goal": bp.natural_user_goal,
            "target_description": bp.target_description,
            "value_generator_id": bp.value_generator_id,
            "query_asset_family": bp.query_asset_family,
            "boolean_balancing_strategy": bp.boolean_balancing_strategy,
            "hard_distractor_families": list(bp.hard_distractor_families),
            "surface_compatibility": list(bp.surface_compatibility),
            "entity_family": bp.entity_family,
            "coding_like": bp.coding_like(),
            "capability_families": list(bp.capability_families()),
            "allowed_call_counts": list(bp.allowed_call_counts()),
            "plans": [{
                "plan_id": p.plan_id,
                "intent": p.intent,
                "call_count": p.call_count,
                "input_roles": [{"name": r.name, "hint": r.hint,
                                 "semantic_type": r.sem,
                                 "calibrated": r.calibrated,
                                 "description": r.description} for r in p.roles],
                "capability_plan": list(p.capability_plan()),
                "normalized_capability_plan": list(p.normalized_capability_plan()),
                "steps": [{"node_id": s.node_id, "capability": s.capability,
                           "args": list(s.args), "purpose": s.purpose}
                          for s in p.steps],
                "sink": p.sink,
                "derived": derived.get(p.plan_id, {}),
            } for p in bp.plans],
        })
    return {
        "schema_version": "ttdf.pilot43.workflow_registry.v3",
        "n_workflows": len(rows),
        "n_plans": sum(len(bp.plans) for bp in all_blueprints()),
        "domains": sorted({bp.domain for bp in all_blueprints()}),
        "registry_hash": registry_hash(),
        "workflows": rows,
    }
