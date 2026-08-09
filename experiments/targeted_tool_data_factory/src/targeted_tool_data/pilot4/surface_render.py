"""Tool surface renderers (Phase H): A_NATIVE and G_GENERAL.

A_NATIVE keeps the abstract conventions of the target family (short functional
names, ``arg_i`` style parameters, flat JSON schemas, ``$var_n.<field>$``
references). G_GENERAL keeps the identical semantic program but changes the
whole surface: different names, different parameter morphology, different
valid output keys.

Output keys are a deterministic function of ``surface_id``, so a tool NAME
still maps to exactly one signature and one output key globally — the trainer's
executor resolves calls by name against a single registry.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .. import registry as reg
from ..graph import REF, _refs_in
from ..schemas import Call, ToolParam, ToolSpec
from ..util import short_hash
from .program import ProgramSpec

SURFACE_TRACKS = ["A_NATIVE", "G_GENERAL"]

TRACK_CODE = {"A_NATIVE": "A", "G_GENERAL": "G"}

# Controlled output-key vocabulary for the generalisation track. A_NATIVE keeps
# the target-like ``output_0``.
_G_OUTPUT_KEYS = ["result", "value", "output", "computed", "outcome",
                  "returned_value", "result_value"]

REFERENCE_PROFILES = {
    "A_NATIVE": "$var{i}.{field}$",
    "G_GENERAL": "$var_{i}.{field}$",
}


def output_field_for(track: str, surface_id_global: str, default: str) -> str:
    """Deterministic per-surface output key; identical for every task."""
    if track == "A_NATIVE":
        return default
    idx = int(short_hash(surface_id_global)[:8], 16) % len(_G_OUTPUT_KEYS)
    return _G_OUTPUT_KEYS[idx]


def _global_surface_id(sid: str, surf: reg.Surface) -> str:
    return f"{sid}::{surf.surface_id}::{surf.name}"


def output_key_map(track: str = "G_GENERAL") -> Dict[str, str]:
    """Tool name -> output key, exported alongside the dataset."""
    out: Dict[str, str] = {}
    for sid, tcode, surf in reg.all_surfaces():
        if tcode != TRACK_CODE.get(track, "G"):
            continue
        out[surf.name] = output_field_for(track, _global_surface_id(sid, surf),
                                          surf.output_field)
    return out


def render_tool(sid: str, track: str, surf: reg.Surface) -> ToolSpec:
    prim = reg.get(sid)
    params: List[ToolParam] = []
    for (canon, ptype, sem), pname in zip(prim.params, surf.param_names):
        if ptype.startswith("enum:"):
            params.append(ToolParam(name=pname, type="string", semantic=sem,
                                    description=f"One of: {ptype[5:]}.",
                                    enum=ptype[5:].split(",")))
        elif ptype == reg.ARR:
            params.append(ToolParam(name=pname, type="array", semantic=sem,
                                    description="The list of numeric values.",
                                    items_type="number"))
        else:
            params.append(ToolParam(name=pname, type=ptype, semantic=sem,
                                    description=f"{pname.replace('_', ' ')}."))
    return ToolSpec(
        name=surf.name, description=surf.description, params=params,
        output_field=output_field_for(track, _global_surface_id(sid, surf),
                                      surf.output_field),
        output_type=prim.out_type, output_description=surf.description,
        semantic_id=sid, surface_id=surf.surface_id)


def pick_surfaces(spec: ProgramSpec, track: str, rng: random.Random
                  ) -> Dict[str, ToolSpec]:
    """One consistent surface per primitive per task."""
    code = TRACK_CODE.get(track)
    if code is None:
        raise ValueError(f"unknown surface track {track!r}")
    out: Dict[str, ToolSpec] = {}
    for nd in spec.program.nodes:
        if nd.semantic_id in out:
            continue
        options = reg.get(nd.semantic_id).surfaces(code)
        if not options:
            options = reg.get(nd.semantic_id).surfaces("A" if code == "G" else "G")
        out[nd.semantic_id] = render_tool(nd.semantic_id, track, rng.choice(options))
    return out


def render_calls(spec: ProgramSpec, tools: Dict[str, ToolSpec], track: str
                 ) -> List[Call]:
    fmt = REFERENCE_PROFILES.get(track, "$var{i}.{field}$")
    labels = {nd.node_id: fmt.split(".")[0].format(i=i + 1)
              for i, nd in enumerate(spec.program.nodes)}
    fields = {nd.node_id: tools[nd.semantic_id].output_field
              for nd in spec.program.nodes}

    def _render(v: Any) -> Any:
        if isinstance(v, dict) and REF in v:
            nid = v[REF]
            return f"{labels[nid]}.{fields[nid]}$"
        if isinstance(v, list):
            return [_render(x) for x in v]
        return v

    calls: List[Call] = []
    for nd in spec.program.nodes:
        prim = reg.get(nd.semantic_id)
        tool = tools[nd.semantic_id]
        args: Dict[str, Any] = {}
        for (canon, _t, _s), tp in zip(prim.params, tool.params):
            args[tp.name] = _render(nd.inputs[canon])
        calls.append(Call(name=tool.name, arguments=args,
                          label=labels[nd.node_id]))
    return calls


def paired_variants(query_modes: Sequence[str] = ("PROCEDURAL_EXPLICIT",
                                                  "SEMI_IMPLICIT",
                                                  "GOAL_BASED_IMPLICIT"),
                    tracks: Sequence[str] = SURFACE_TRACKS
                    ) -> List[Tuple[str, str]]:
    """The (track, query_mode) combinations a single program may be rendered as."""
    return [(t, m) for t in tracks for m in query_modes]


def surface_signature(tools: Dict[str, ToolSpec]) -> str:
    payload = sorted((t.name, tuple(p.name for p in t.params), t.output_field)
                     for t in tools.values())
    return "sig_" + short_hash(payload)
