"""Surface rendering: tool schemas, gold calls, deterministic query templates.

Separates semantic primitives from surface representation (DESIGN.md §10).
The query is rendered LAST, from the already-executed program (capability 6).
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from .. import registry as reg
from ..graph import REF, _refs_in
from ..schemas import Call, GraphNode, SemanticProgram, ToolParam, ToolSpec


# ── tool schema rendering ─────────────────────────────────────────────────
def render_tool(sid: str, track: str, rng: random.Random,
                surface: Optional[reg.Surface] = None,
                param_style: str = "semantic") -> ToolSpec:
    """param_style="as_surface" (engine v2) keeps the surface's own parameter
    names, so a tool NAME maps to exactly one signature globally — required by
    the trainer's synthetic executor, which resolves calls by name against one
    global registry."""
    p = reg.get(sid)
    surf = surface or rng.choice(p.surfaces(track))
    pnames = (
        [f"arg_{i}" for i in range(len(p.params))]
        if param_style == "generic" else list(surf.param_names))
    params = []
    for (canon, ptype, sem), pname in zip(p.params, pnames):
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
        output_field=surf.output_field, output_type=p.out_type,
        output_description=surf.description, semantic_id=sid,
        surface_id=surf.surface_id)


def pick_surfaces(prog: SemanticProgram, track: str, rng: random.Random,
                  param_style: str) -> Dict[str, ToolSpec]:
    """One consistent surface per semantic id per task."""
    out: Dict[str, ToolSpec] = {}
    for nd in prog.nodes:
        if nd.semantic_id not in out:
            out[nd.semantic_id] = render_tool(nd.semantic_id, track, rng,
                                              param_style=param_style)
    return out


# ── gold call rendering ───────────────────────────────────────────────────
def render_calls(prog: SemanticProgram, tools: Dict[str, ToolSpec],
                 label_style: str = "$var{i}") -> List[Call]:
    labels = {nd.node_id: label_style.format(i=i + 1)
              for i, nd in enumerate(prog.nodes)}
    fields = {nd.node_id: tools[nd.semantic_id].output_field for nd in prog.nodes}

    def _render_val(v: Any) -> Any:
        if isinstance(v, dict) and REF in v:
            nid = v[REF]
            return f"{labels[nid]}.{fields[nid]}$"
        if isinstance(v, list):
            return [_render_val(item) for item in v]
        return v

    calls = []
    for nd in prog.nodes:
        prim = reg.get(nd.semantic_id)
        spec = tools[nd.semantic_id]
        args = {}
        for (canon, _t, _s), tp in zip(prim.params, spec.params):
            args[tp.name] = _render_val(nd.inputs[canon])
        calls.append(Call(name=spec.name, arguments=args, label=labels[nd.node_id]))
    return calls


# ── query realization (deterministic templates) ───────────────────────────
_WRAPPERS_ENTITIES = [
    ("A warehouse audit produced these numbers.", "warehouse"),
    ("A lab notebook lists these measurements.", "lab"),
    ("A shop reviews its figures.", "shop"),
    ("A hiking club recorded these values.", "club"),
    ("An engineer checks a report.", "report"),
    ("A delivery service reviews yesterday's log.", "delivery"),
    ("A school tracks these figures for its yearbook.", "school"),
]
_IRRELEVANT = [
    "(Unrelated: the office moved in 2019.)",
    "(Note: the team has 7 members, which is not needed here.)",
    "(Background: the device serial is AX-40, irrelevant to the math.)",
    "(For context only: the meeting lasted 90 minutes.)",
]
# mostly empty: NESTFUL questions rarely carry an output-format directive
_FINAL_FMT = ["", "", "", "", " What is the final result?",
              " Report only the final value."]

# per-family connector variants -> distinct surface template ids.
# Leads must be grammatical in front of a bare imperative verb phrase
# ("Start by compute ..." was a pilot1 artifact and is gone).
_IMP_LEADS = [
    ("First, ", "Then ", "Finally, "),
    ("To begin, ", "Next, ", "Lastly, "),
    ("Step one: ", "After that, ", "To finish, "),
    ("Please ", "then ", "and finally "),
    ("Initially, ", "Subsequently, ", "In the end, "),
]
_SEQ_CONN = [
    (", then ", ", and finally "),
    (", after that ", ", and at the end "),
    (", followed by ", ", and lastly "),
    ("; next ", "; finally "),
    (", and then ", ", concluding by "),
]
_IND_FRAMES = [
    "I need the final value after the following steps: {body}.",
    "Work out the end result of these operations: {body}.",
    "What do I get if I {body}?",
    "Determine the outcome of this procedure: {body}.",
    "Give me the result of this calculation: {body}.",
    "Tell me what comes out when I {body}.",
]
_GOAL_FRAMES = [
    "The end goal is to {last}. To get there, {body} first.",
    "Ultimately I want to {last}. Before that, {body}.",
    "My target is to {last}, and the way there is: {body}.",
    "What I finally need is to {last}; the preparation is to {body}.",
]


def _fmt_val(v: Any, numeric_string: bool = False) -> str:
    if isinstance(v, str):
        return f"'{v}'"
    if isinstance(v, list):
        return "[" + ", ".join(_fmt_val(x) for x in v) + "]"
    if isinstance(v, float) and v == int(v):
        return str(int(v))
    return str(v)


def _node_phrase(nd: GraphNode, idx: int, prog: SemanticProgram,
                 prev_text: str = "that result") -> str:
    prim = reg.get(nd.semantic_id)
    order = {n.node_id: i for i, n in enumerate(prog.nodes)}
    fills: Dict[str, str] = {}
    agg_over_all = False
    any_ref = False
    for i, (canon, _t, _s) in enumerate(prim.params):
        v = nd.inputs[canon]
        refs = _refs_in(v)
        if refs:
            any_ref = True
            if isinstance(v, list) and len(refs) > 1:
                agg_over_all = True
                fills[canon] = "all of the previous results"
            elif order[refs[0]] == idx - 1:
                fills[canon] = prev_text
            else:
                fills[canon] = f"the result of step {order[refs[0]] + 1}"
        else:
            fills[canon] = _fmt_val(v)
    # unit conversions read wrong with a reference in the value slot
    # ("convert that result kilometers to meters"); phrase_ref fixes the
    # preposition ("convert that result from kilometers into meters").
    template = prim.phrase_ref if (any_ref and prim.phrase_ref) else prim.phrase
    phrase = template.format(**{k.strip("{}"): v for k, v in fills.items()})
    if agg_over_all:
        action = {"sum_values": "sum", "mean_values": "average",
                  "max_values": "maximum", "min_values": "minimum",
                  "range_spread": "spread (max minus min)"}.get(nd.semantic_id, "aggregate")
        phrase = f"compute the {action} of all the previous results"
    return phrase


_FAMILY_WEIGHTS = [
    ("imperative", len(_IMP_LEADS)),
    ("sequence", len(_SEQ_CONN)),
    ("word_problem", len(_WRAPPERS_ENTITIES)),
    ("indirect", len(_IND_FRAMES)),
    ("goal_first", len(_GOAL_FRAMES)),
]
TEMPLATE_COUNT = sum(w for _f, w in _FAMILY_WEIGHTS)


def _pick_family(rng: random.Random) -> str:
    """Weighted by variant count, so every individual template is equally
    likely and no template can dominate the pool (>5 % gate)."""
    total = TEMPLATE_COUNT
    x = rng.randrange(total)
    acc = 0
    for fam, w in _FAMILY_WEIGHTS:
        acc += w
        if x < acc:
            return fam
    return _FAMILY_WEIGHTS[-1][0]


def render_query(prog: SemanticProgram, rng: random.Random,
                 with_irrelevant: bool = False) -> Tuple[str, str, str]:
    """Returns (query, template_id, paraphrase_family)."""
    phrases = [_node_phrase(nd, i, prog) for i, nd in enumerate(prog.nodes)]
    family = _pick_family(rng)
    fmt = rng.choice(_FINAL_FMT)
    irr = (" " + rng.choice(_IRRELEVANT)) if with_irrelevant else ""

    if family == "imperative":
        k = rng.randrange(len(_IMP_LEADS))
        first, mid, last = _IMP_LEADS[k]
        parts = []
        for i, ph in enumerate(phrases):
            lead = first if i == 0 else (mid if i < len(phrases) - 1 else last)
            parts.append(lead + ph + ".")
        q = " ".join(parts)
        tid = f"imperative_v{k + 1}"
    elif family == "sequence":
        k = rng.randrange(len(_SEQ_CONN))
        mid, last = _SEQ_CONN[k]
        body = mid.join(phrases[:-1])
        q = f"{body[0].upper()}{body[1:]}{last}{phrases[-1]}."
        tid = f"sequence_v{k + 1}"
    elif family == "word_problem":
        wrap, ent = rng.choice(_WRAPPERS_ENTITIES)
        steps = " ".join(f"Step {i + 1}: {ph}." for i, ph in enumerate(phrases))
        q = f"{wrap} {steps}"
        tid = f"word_problem_{ent}"
    elif family == "indirect":
        k = rng.randrange(len(_IND_FRAMES))
        body = "; ".join(phrases)
        q = _IND_FRAMES[k].format(body=body)
        tid = f"indirect_v{k + 1}"
    else:  # goal_first
        k = rng.randrange(len(_GOAL_FRAMES))
        body = ", then ".join(phrases[:-1])
        # the goal is stated BEFORE its input exists, so "that result" would
        # be dangling; name it explicitly instead (pilot1 artifact).
        last = _node_phrase(prog.nodes[-1], len(prog.nodes) - 1, prog,
                            prev_text="the value those steps produce")
        q = _GOAL_FRAMES[k].format(last=last, body=body[0].lower() + body[1:])
        tid = f"goal_first_v{k + 1}"

    q = (q + irr + fmt).strip()
    return q, tid, family


# ── export shapes ─────────────────────────────────────────────────────────
def tool_to_nestful(spec: ToolSpec) -> Dict[str, Any]:
    """NESTFUL flat-dict parameters format."""
    params = {}
    for p in spec.params:
        entry: Dict[str, Any] = {"description": p.description, "type": p.type}
        if p.type == "array":
            entry["items"] = {"type": p.items_type or "number"}
        if p.enum:
            entry["enum"] = p.enum
        params[p.name] = entry
    return {
        "name": spec.name,
        "description": spec.description,
        "parameters": params,
        "output_parameters": {
            spec.output_field: {"description": spec.output_description or spec.description,
                                "type": spec.output_type}},
    }


def tool_to_jsonschema(spec: ToolSpec) -> Dict[str, Any]:
    """GRPO train-ready JSON-Schema style (stage3_train_ready format)."""
    props = {}
    required = []
    for p in spec.params:
        entry: Dict[str, Any] = {"type": p.type, "description": p.description}
        if p.type == "array":
            entry["items"] = {"type": p.items_type or "number"}
        if p.enum:
            entry["enum"] = p.enum
        props[p.name] = entry
        if p.required:
            required.append(p.name)
    return {
        "name": spec.name,
        "description": spec.description,
        "parameters": {"type": "object", "properties": props, "required": required},
        "output_parameters": {
            spec.output_field: {"type": spec.output_type,
                                "description": spec.output_description or spec.description}},
    }
