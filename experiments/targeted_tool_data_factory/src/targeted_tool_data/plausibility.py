"""Semantic plausibility: unit propagation over a task graph and
classification into natural / abstract_coherent / artificial_composition.

Motivation (pilot2): pilot1 produced chains such as
``square(44) -> increase 1931 by that percent -> km_to_meters -> seconds_to_
minutes``, which are executable and internally valid but semantically absurd.
Units make the mismatch machine-detectable, so such compositions can be
avoided during generation and capped during selection.

Classes:
  natural               at least one unit-typed operation, every typed
                        consumer received a matching unit, no mismatch;
  abstract_coherent     dimensionless arithmetic only (NESTFUL's math core
                        looks exactly like this) — perfectly coherent;
  artificial_composition  at least one unit mismatch (e.g. a temperature fed
                        into a duration converter, or a physical measurement
                        used as a percentage rate).
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from . import registry as reg
from .graph import REF, _refs_in
from .schemas import SemanticProgram

NATURAL = "natural"
ABSTRACT = "abstract_coherent"
ARTIFICIAL = "artificial_composition"


def transition_class(expect: str, incoming_unit: str, semantic: str) -> str:
    """Classify one argument binding."""
    if expect == reg.E_ANY:
        # a physical measurement used as a rate/exponent/precision is absurd
        if semantic in ("percentage", "precision", "count") and \
                incoming_unit in reg.PHYSICAL_UNITS:
            return ARTIFICIAL
        return NATURAL if incoming_unit in reg.PHYSICAL_UNITS else ABSTRACT
    if expect == reg.E_NEUTRAL:
        if incoming_unit in reg.PHYSICAL_UNITS:
            return ARTIFICIAL
        return ABSTRACT
    # expect is a specific unit
    if incoming_unit == expect:
        return NATURAL
    if incoming_unit in (reg.U_ABSTRACT, reg.U_COUNT):
        # an unlabeled number can be read as the required unit only when it
        # comes straight from the question (handled by caller for constants)
        return ARTIFICIAL
    return ARTIFICIAL


def unit_of_constant(expect: str) -> str:
    """A literal in the question carries whatever unit the tool expects."""
    return expect if expect not in (reg.E_ANY, reg.E_NEUTRAL) else reg.U_ABSTRACT


def analyze(prog: SemanticProgram) -> Dict[str, Any]:
    """Propagate units through the DAG and classify the whole task."""
    node_unit: Dict[str, str] = {}
    transitions: List[Dict[str, str]] = []
    has_typed = False
    worst = ABSTRACT

    for nd in prog.nodes:
        prim = reg.get(nd.semantic_id)
        in_units: List[str] = []
        for (pname, ptype, semantic), expect in zip(prim.params, prim.param_units):
            if expect not in (reg.E_ANY, reg.E_NEUTRAL):
                has_typed = True
            value = nd.inputs[pname]
            refs = _refs_in(value)
            if not refs:
                u = unit_of_constant(expect)
                in_units.append(u)
                continue
            if ptype == reg.ARR and len(refs) > 1:
                # aggregation over independent branches: comparable only if
                # every branch carries the same unit (summing meters with a
                # ratio is the classic artificial composition)
                incoming = [node_unit.get(r, reg.U_ABSTRACT) for r in refs]
                distinct = set(incoming)
                if distinct <= reg.NEUTRAL_UNITS:
                    cls = ABSTRACT
                elif len(distinct) == 1:
                    cls = NATURAL
                    has_typed = True
                else:
                    cls = ARTIFICIAL
                for r, u in zip(refs, incoming):
                    transitions.append({"from": r, "to": nd.node_id,
                                        "param": pname, "incoming": u,
                                        "expect": "aggregate", "class": cls})
                if cls == ARTIFICIAL:
                    worst = ARTIFICIAL
                elif cls == NATURAL and worst != ARTIFICIAL:
                    worst = NATURAL
                in_units.extend(incoming)
                continue
            for ref in refs:
                incoming = node_unit.get(ref, reg.U_ABSTRACT)
                cls = transition_class(expect, incoming, semantic)
                transitions.append({"from": ref, "to": nd.node_id,
                                    "param": pname, "incoming": incoming,
                                    "expect": expect, "class": cls})
                if cls == ARTIFICIAL:
                    worst = ARTIFICIAL
                elif cls == NATURAL and worst != ARTIFICIAL:
                    worst = NATURAL
                in_units.append(incoming)
        if prim.unit_out not in (reg.U_ABSTRACT, reg.PRESERVE):
            has_typed = True
        node_unit[nd.node_id] = prim.unit_of_output(in_units)

    if worst == ARTIFICIAL:
        cls = ARTIFICIAL
    elif has_typed and worst == NATURAL:
        cls = NATURAL
    else:
        cls = ABSTRACT
    return {
        "plausibility_class": cls,
        "unit_trace": {k: v for k, v in node_unit.items()},
        "artificial_transitions": [t for t in transitions if t["class"] == ARTIFICIAL],
        "sink_unit": node_unit.get(prog.sink, reg.U_ABSTRACT),
    }


def compatible_consumers(incoming_unit: str, candidate_sids: List[str],
                         *, allow_artificial: bool = False
                         ) -> List[Tuple[str, int]]:
    """(sid, param_index) pairs that can consume `incoming_unit` without
    creating an artificial transition."""
    out: List[Tuple[str, int]] = []
    for sid in candidate_sids:
        p = reg.get(sid)
        for i, ((_pn, ptype, semantic), expect) in enumerate(
                zip(p.params, p.param_units)):
            if ptype not in (reg.NUM, reg.INT):
                continue
            cls = transition_class(expect, incoming_unit, semantic)
            if cls != ARTIFICIAL or allow_artificial:
                out.append((sid, i))
    return out
