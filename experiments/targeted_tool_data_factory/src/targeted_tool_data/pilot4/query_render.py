"""Query renderers (Phase G): procedural, semi-implicit and goal-based.

Pilot3 rendered every question as an ordered recipe, which made the question a
transcript of the gold program. The renderers here take the same
``ProgramSpec`` and expose three different amounts of the plan:

    PROCEDURAL_EXPLICIT   names every operation, in order        (easy bucket)
    SEMI_IMPLICIT         names some relations, never the recipe (medium)
    GOAL_BASED_IMPLICIT   states inputs and the goal only        (hard)

None of them may change the program, its constants, its oracle or its answer.
"""
from __future__ import annotations

import random
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .. import registry as reg
from ..graph import _refs_in
from ..query_realism import lexicon_for
from ..schemas import GraphNode
from .program import ProgramSpec

QUERY_RENDERERS = ["PROCEDURAL_EXPLICIT", "SEMI_IMPLICIT", "GOAL_BASED_IMPLICIT"]

# Domain frames are controlled and reusable; they carry no operation words.
_DOMAINS = [
    ("a warehouse audit", "the audit"),
    ("a laboratory log", "the log"),
    ("a delivery roster", "the roster"),
    ("a field survey", "the survey"),
    ("a maintenance sheet", "the sheet"),
    ("a store inventory", "the inventory"),
    ("a training record", "the record"),
    ("a sensor readout", "the readout"),
]

_PROC_LEADS = [
    ("First, ", "Then ", "Finally, "),
    ("To begin, ", "Next, ", "Lastly, "),
    ("Step one: ", "After that, ", "To finish, "),
    ("Please ", "then ", "and finally "),
]

_SEMI_OPENERS = [
    "Working from {domain}, {clause}.",
    "{domain_cap} gives {values}. {clause}.",
    "From {domain}: {clause}.",
    "Given {domain}, {clause}.",
]

# Goal phrases are either noun phrases ("the overall total") or embedded
# clauses ("how much remains"), which do not fit the same carrier sentences.
_GOAL_FRAMES_NOUN = [
    "{domain_cap} lists {values}. Work out {goal}.",
    "These figures come from {domain}: {values}. Determine {goal}.",
    "{domain_cap} recorded {values}. What is {goal}?",
    "Using {domain} with {values}, report {goal}.",
    "{domain_cap} holds {values}. Establish {goal}.",
]
_GOAL_FRAMES_CLAUSE = [
    "{domain_cap} lists {values}. Work out {goal}.",
    "These figures come from {domain}: {values}. Determine {goal}.",
    "{domain_cap} recorded {values}. Establish {goal}.",
    "Using {domain} with {values}, work out {goal}.",
    "{domain_cap} holds {values}. Determine {goal}.",
]
_CLAUSE_STARTS = ("how ", "what ", "which ", "whether ", "is it ", "does ", "do ")

_GENERIC_GOALS = {
    "number": "the resulting figure",
    "integer": "the resulting whole number",
    "boolean": "whether the condition holds",
    "string": "the resulting label",
    "array": "the resulting series",
}


def _fmt(v: Any) -> str:
    if isinstance(v, str):
        return f"'{v}'"
    if isinstance(v, list):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    if isinstance(v, float) and v == int(v):
        return str(int(v))
    return str(v)


def _constant_phrases(spec: ProgramSpec) -> List[str]:
    """Every direct constant, in node order, so the task stays solvable."""
    out: List[str] = []
    for nd in spec.program.nodes:
        prim = reg.get(nd.semantic_id)
        for pname, _t, _s in prim.params:
            v = nd.inputs[pname]
            if not _refs_in(v):
                out.append(_fmt(v))
    return out


def _node_index(spec: ProgramSpec) -> Dict[str, int]:
    return {nd.node_id: i for i, nd in enumerate(spec.program.nodes)}


def _exact_clause(spec: ProgramSpec, nd: GraphNode, idx: int,
                  order: Dict[str, int], prev_text: str = "that result") -> str:
    prim = reg.get(nd.semantic_id)
    fills: Dict[str, str] = {}
    any_ref = False
    for pname, _t, _s in prim.params:
        v = nd.inputs[pname]
        refs = _refs_in(v)
        if refs:
            any_ref = True
            src = order[refs[0]]
            fills[pname] = (prev_text if src == idx - 1
                            else f"the result of step {src + 1}")
        else:
            fills[pname] = _fmt(v)
    template = prim.phrase_ref if (any_ref and prim.phrase_ref) else prim.phrase
    try:
        return template.format(**fills)
    except (KeyError, IndexError):
        return prim.phrase.format(**fills)


_LEADING_VERB_RE = re.compile(r"^(compute|find|take|convert|round|check|count|"
                              r"sum|add|subtract|multiply|divide|average|join|"
                              r"parse|format|label|build|keep|scale|sort|append|"
                              r"invert|negate|square|clamp|map|express|shift|"
                              r"concatenate|uppercase|look up|apply)\b", re.I)


def _semantic_clause(spec: ProgramSpec, nd: GraphNode, idx: int,
                     order: Dict[str, int]) -> str:
    """Operation-free description: which quantities feed this intermediate."""
    prim = reg.get(nd.semantic_id)
    lex = lexicon_for(nd.semantic_id)
    parts: List[str] = []
    for pname, _t, _s in prim.params:
        v = nd.inputs[pname]
        refs = _refs_in(v)
        if refs:
            parts.append(f"the figure from stage {order[refs[0]] + 1}")
        else:
            parts.append(_fmt(v))
    inputs = " and ".join(parts) if len(parts) <= 2 else (
        ", ".join(parts[:-1]) + " and " + parts[-1])
    goal = lex["semantic"][0] if lex["semantic"] else None
    if goal:
        return f"stage {idx + 1} derives {goal} from {inputs}"
    return f"stage {idx + 1} follows from {inputs}"


def _goal_phrase(spec: ProgramSpec) -> str:
    from .program import sink_primitive

    prim = sink_primitive(spec)
    lex = lexicon_for(prim.sid)
    if lex["semantic"]:
        return lex["semantic"][0]
    return _GENERIC_GOALS.get(prim.out_type, "the final value")


def goal_is_underspecified(spec: ProgramSpec) -> bool:
    """True when the sink has no semantic cue, so the goal is purely generic."""
    from .program import sink_primitive

    return not lexicon_for(sink_primitive(spec).sid)["semantic"]


def _answer_str(ans: Any) -> str:
    if isinstance(ans, float) and ans == int(ans):
        return str(int(ans))
    return str(ans)


# ── renderers ─────────────────────────────────────────────────────────────
def render_procedural_explicit(spec: ProgramSpec, rng: random.Random
                               ) -> Tuple[str, str]:
    order = _node_index(spec)
    clauses = [_exact_clause(spec, nd, i, order)
               for i, nd in enumerate(spec.program.nodes)]
    k = rng.randrange(len(_PROC_LEADS))
    first, mid, last = _PROC_LEADS[k]
    parts = []
    for i, c in enumerate(clauses):
        lead = first if i == 0 else (mid if i < len(clauses) - 1 else last)
        parts.append(lead + c + ".")
    return " ".join(parts), f"p4_procedural_v{k + 1}"


def render_semi_implicit(spec: ProgramSpec, rng: random.Random) -> Tuple[str, str]:
    """Name a minority of relations; never one text step per gold call."""
    order = _node_index(spec)
    nodes = list(spec.program.nodes)
    n = len(nodes)
    domain, domain_short = rng.choice(_DOMAINS)
    # at most ~40 % of the calls may be described operationally, and never the
    # sink (naming the last operation is what makes the plan readable)
    budget = max(1, int(n * 0.4)) if n > 2 else 1
    candidates = list(range(n - 1)) or [0]
    rng.shuffle(candidates)
    explicit = sorted(candidates[:budget])

    described: List[str] = []
    for i in explicit:
        described.append(_exact_clause(spec, nodes[i], i, order,
                                       prev_text="the earlier figure"))
    implicit_idx = [i for i in range(n) if i not in explicit]
    if implicit_idx:
        described.append(
            f"the remaining {len(implicit_idx)} intermediate figures follow from "
            f"{_goal_phrase(spec)}"
            if len(implicit_idx) > 1 else
            f"one further intermediate figure leads to {_goal_phrase(spec)}")
    clause = "; ".join(described)
    values = ", ".join(_constant_phrases(spec)) or "no direct figures"
    k = rng.randrange(len(_SEMI_OPENERS))
    text = _SEMI_OPENERS[k].format(
        domain=domain, domain_cap=domain[0].upper() + domain[1:],
        values=values, clause=clause)
    if "{values}" not in _SEMI_OPENERS[k] and values not in text:
        text += f" The available figures are {values}."
    text += f" Report {_goal_phrase(spec)}."
    return text, f"p4_semi_v{k + 1}"


def render_goal_based_implicit(spec: ProgramSpec, rng: random.Random
                               ) -> Tuple[str, str]:
    """State the inputs, the dependency skeleton and the goal — no operations."""
    order = _node_index(spec)
    domain, _short = rng.choice(_DOMAINS)
    values = ", ".join(_constant_phrases(spec))
    goal = _goal_phrase(spec)
    frames = (_GOAL_FRAMES_CLAUSE if goal.lower().startswith(_CLAUSE_STARTS)
              else _GOAL_FRAMES_NOUN)
    k = rng.randrange(len(frames))
    text = frames[k].format(
        domain=domain, domain_cap=domain[0].upper() + domain[1:],
        values=values, goal=goal)
    # Dependency hints keep the task well posed: they say which quantities feed
    # which intermediate figure, without naming a single operation. Without
    # them a multi-stage goal question is not answerable at all.
    hints = [_semantic_clause(spec, nd, i, order)
             for i, nd in enumerate(spec.program.nodes)]
    if hints:
        text += " The stages are related as follows: " + "; ".join(hints) + "."
    return text, f"p4_goal_v{k + 1}"


_RENDERERS = {
    "PROCEDURAL_EXPLICIT": render_procedural_explicit,
    "SEMI_IMPLICIT": render_semi_implicit,
    "GOAL_BASED_IMPLICIT": render_goal_based_implicit,
}


def render_query(spec: ProgramSpec, mode: str, rng: random.Random
                 ) -> Dict[str, Any]:
    fn = _RENDERERS.get(mode)
    if fn is None:
        raise ValueError(f"unknown query renderer {mode!r}")
    text, template_id = fn(spec, rng)
    text = re.sub(r"\s+", " ", text).strip()
    return {
        "query": text,
        "template_id": template_id,
        "requested_query_mode": mode,
        "query_skeleton": _skeleton(text),
    }


def _skeleton(text: str) -> str:
    """Template fingerprint with all literals removed (concentration control)."""
    from ..util import short_hash

    masked = re.sub(r"-?\d+(?:\.\d+)?", "#", text)
    masked = re.sub(r"'[^']*'", "'S'", masked)
    masked = re.sub(r"\[[^\]]*\]", "[L]", masked)
    return "qs_" + short_hash(masked)


def answer_leaks_into_query(query: str, answer: Any) -> bool:
    return _answer_str(answer) in query
