"""The SemanticProgram layer (Phase F).

A ``ProgramSpec`` is everything about a task that is true *before* anyone
decides how to phrase the question or what to call the tools: the typed DAG,
its primitives, its constants, its dependency edges, the oracle trace and the
structural feature vector. No natural language, no surface names.

The same ProgramSpec can therefore be rendered as several different tasks
(A_NATIVE + goal-based, G_GENERAL + semi-implicit, ...) that share an oracle
and must never be split across train/heldout.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .. import registry as reg
from ..capability import family_of
from ..graph import REF, _refs_in
from ..schemas import SemanticProgram
from ..util import short_hash
from .patterns import ProgramResult, Shape, shape_signature


@dataclass
class ProgramSpec:
    program: SemanticProgram
    shape: Shape
    pattern_family: str
    transformations: List[str]
    observations: List[Any]
    answer: Any
    capability_families: List[str]
    constants: List[Any] = field(default_factory=list)
    edges: List[Dict[str, Any]] = field(default_factory=list)
    features: Dict[str, Any] = field(default_factory=dict)
    semantic_program_id: str = ""
    program_family_id: str = ""
    graph_template_id: str = ""

    @property
    def call_count(self) -> int:
        return len(self.program.nodes)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "semantic_program_id": self.semantic_program_id,
            "program_family_id": self.program_family_id,
            "graph_template_id": self.graph_template_id,
            "pattern_family": self.pattern_family,
            "transformations": list(self.transformations),
            "shape": [list(p) for p in self.shape],
            "nodes": [
                {"node_id": nd.node_id, "primitive_id": nd.semantic_id,
                 "capability_family": family_of(nd.semantic_id),
                 "output_type": nd.output_type,
                 "inputs": nd.inputs}
                for nd in self.program.nodes
            ],
            "edges": list(self.edges),
            "sink": self.program.sink,
            "constants": list(self.constants),
            "oracle_observations": list(self.observations),
            "oracle_answer": self.answer,
            "capability_families": list(self.capability_families),
            "structural_features": dict(self.features),
        }


def _edges_of(prog: SemanticProgram) -> List[Dict[str, Any]]:
    pos = {nd.node_id: i for i, nd in enumerate(prog.nodes)}
    edges = []
    for nd in prog.nodes:
        for pname, value in nd.inputs.items():
            for ref in _refs_in(value):
                edges.append({"from": ref, "to": nd.node_id, "param": pname,
                              "distance": pos[nd.node_id] - pos[ref]})
    return edges


def _constants_of(prog: SemanticProgram) -> List[Any]:
    out: List[Any] = []
    for nd in prog.nodes:
        for value in nd.inputs.values():
            if not _refs_in(value):
                out.append(value)
    return out


def structural_features(prog: SemanticProgram, shape: Shape) -> Dict[str, Any]:
    from collections import Counter

    n = len(shape)
    indeg = [len(p) for p in shape]
    outdeg = Counter(p for parents in shape for p in parents)
    distances = [i - p for i, parents in enumerate(shape) for p in parents]
    types = [nd.output_type for nd in prog.nodes]
    transitions = sum(1 for i, parents in enumerate(shape)
                      for p in parents if types[p] != types[i])
    longest = [0] * n
    for i, parents in enumerate(shape):
        if parents:
            longest[i] = 1 + max(longest[p] for p in parents)
    depth = max(longest) if longest else 0
    prim_counts = Counter(nd.semantic_id for nd in prog.nodes)
    fam_counts = Counter(family_of(nd.semantic_id) for nd in prog.nodes)
    return {
        "n_nodes": n,
        "n_edges": sum(indeg),
        "depth": depth,
        "critical_path": depth + 1 if n else 0,
        "n_roots": sum(1 for d in indeg if d == 0),
        "n_leaves": sum(1 for i in range(n) if outdeg.get(i, 0) == 0),
        "n_joins": sum(1 for d in indeg if d >= 2),
        "max_indegree": max(indeg) if indeg else 0,
        "n_fan_out_nodes": sum(1 for c in outdeg.values() if c >= 2),
        "max_outdegree": max(outdeg.values()) if outdeg else 0,
        "n_reused_outputs": sum(1 for c in outdeg.values() if c >= 2),
        "n_late_references": sum(1 for d in distances if d >= 2),
        "mean_reference_distance": round(sum(distances) / len(distances), 4)
                                   if distances else 0.0,
        "max_reference_distance": max(distances) if distances else 0,
        "n_parallel_branches": max(sum(1 for d in indeg if d == 0), 1),
        "n_type_transitions": transitions,
        "n_distinct_primitives": len(prim_counts),
        "max_primitive_repeat": max(prim_counts.values()) if prim_counts else 0,
        "n_distinct_capability_families": len(fam_counts),
        "capability_family_counts": dict(sorted(fam_counts.items())),
    }


def canonical_topology_hash(shape: Shape) -> str:
    return "topo_" + short_hash(shape_signature(shape))


def make_spec(result: ProgramResult) -> ProgramSpec:
    prog = result.program
    edges = _edges_of(prog)
    feats = structural_features(prog, result.shape)
    # family = topology + primitive multiset (surface- and value-independent)
    fam_payload = {
        "topology": shape_signature(result.shape),
        "primitives": sorted(nd.semantic_id for nd in prog.nodes),
    }
    spec = ProgramSpec(
        program=prog, shape=result.shape, pattern_family=result.pattern_family,
        transformations=list(result.transformations),
        observations=list(result.observations), answer=result.answer,
        capability_families=list(result.capability_families),
        constants=_constants_of(prog), edges=edges, features=feats,
        program_family_id="pf4_" + short_hash(fam_payload),
        graph_template_id=canonical_topology_hash(result.shape),
    )
    spec.semantic_program_id = "sp4_" + short_hash({
        **fam_payload,
        "inputs": [nd.inputs for nd in prog.nodes],
        "sink": prog.sink,
    })
    return spec


def sink_primitive(spec: ProgramSpec) -> reg.Primitive:
    by_id = {nd.node_id: nd for nd in spec.program.nodes}
    return reg.get(by_id[spec.program.sink].semantic_id)
