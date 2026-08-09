"""Pilot4.1 query renderers — no stage/DAG edge dumps in implicit modes."""
from __future__ import annotations

import random
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .. import registry as reg
from ..graph import _refs_in
from . import QUERY_MODES
from .workflows import WorkflowFamily

# Phrases forbidden in implicit modes (checked again by V11).
FORBIDDEN_IMPLICIT = [
    r"\bstage\b", r"\bstep\s*\d", r"\bfirst calculate\b", r"\bthen calculate\b",
    r"\bnext compute\b", r"\bthe previous result\b", r"\bthe figure from\b",
    r"\bthe remaining \d+ intermediate\b", r"\bthe stages are related\b",
    r"\buse the result from\b",
]
FORBIDDEN_IMPLICIT_RE = re.compile("|".join(FORBIDDEN_IMPLICIT), re.I)

_ENTITY_FALLBACK = [
    "the order", "the shipment", "the account", "the sample", "the booking",
    "the batch", "the listing", "the report", "the ticket", "the project",
]


def _fmt(v: Any) -> str:
    if isinstance(v, str):
        return f"'{v}'" if " " in v or not v.isdigit() else v
    if isinstance(v, list):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    if isinstance(v, float) and v == int(v):
        return str(int(v))
    return str(v)


def _constants(nodes: Sequence[Dict[str, Any]]) -> List[Any]:
    out: List[Any] = []
    for nd in nodes:
        for v in (nd.get("inputs") or {}).values():
            if not _refs_in(v):
                out.append(v)
    return out


def _facts_from_contract(contract: Dict[str, Any], rng: random.Random) -> List[str]:
    facts = list(contract.get("facts") or [])
    if facts:
        return [str(f) for f in facts]
    # fallback: enumerate constants neutrally without stage labels
    vals = contract.get("entities") or []
    consts = contract.get("constants") or []
    phrases = []
    for i, c in enumerate(consts):
        label = vals[i % len(vals)] if vals else "value"
        phrases.append(f"{label} is {_fmt(c)}")
    rng.shuffle(phrases)
    return phrases


def _target_phrase(contract: Dict[str, Any]) -> str:
    tv = contract.get("target_variable") or {}
    if isinstance(tv, dict) and tv.get("role"):
        return str(tv["role"]).replace("_", " ")
    return str(contract.get("user_goal") or "the result")


def render_graph_explicit(contract: Dict[str, Any], rng: random.Random
                          ) -> Tuple[str, str]:
    """Curriculum bucket: may mention ordered operations lightly."""
    facts = _facts_from_contract(contract, rng)
    ops = [str(s.get("primitive_id") or s.get("capability") or "operation")
           for s in (contract.get("semantic_program_summary") or [])]
    body = "Known figures: " + "; ".join(facts) + ". "
    if ops:
        body += "Carry out these operations in order: " + ", ".join(ops[:6]) + ". "
    body += f"Return {_target_phrase(contract)}."
    return body.strip(), "p41_graph_explicit_v1"


def render_operation_explicit_graph_implicit(contract: Dict[str, Any],
                                             rng: random.Random
                                             ) -> Tuple[str, str]:
    facts = _facts_from_contract(contract, rng)
    caps = []
    for s in (contract.get("semantic_program_summary") or []):
        c = s.get("capability_family") or s.get("primitive_id")
        if c and c not in caps:
            caps.append(str(c).split(".")[-1].replace("_", " "))
    opener = rng.choice([
        "Given that {facts}, {goal}.",
        "With {facts}, please {goal}.",
        "{facts}. I need {goal}.",
    ])
    goal = _target_phrase(contract)
    hint = ""
    if caps and rng.random() < 0.7:
        hint = f" This involves {caps[0]}" + (f" and {caps[1]}" if len(caps) > 1 else "") + "."
    text = opener.format(facts="; ".join(facts), goal=f"find {goal}") + hint
    return text.strip(), "p41_op_explicit_graph_implicit_v1"


def render_semi_implicit(contract: Dict[str, Any], rng: random.Random
                         ) -> Tuple[str, str]:
    facts = _facts_from_contract(contract, rng)
    domain = contract.get("domain") or "the case"
    goal = _target_phrase(contract)
    text = rng.choice([
        f"In {domain}, {'; '.join(facts)}. What is {goal}?",
        f"Working from {domain}: {'; '.join(facts)}. Determine {goal}.",
        f"{'; '.join(facts)}. From these, obtain {goal}.",
    ])
    return text.strip(), "p41_semi_implicit_v1"


def render_goal_based_implicit(contract: Dict[str, Any], rng: random.Random
                               ) -> Tuple[str, str]:
    """Facts + goal only. No stages, no dependency edges, no step labels."""
    facts = _facts_from_contract(contract, rng)
    goal = contract.get("user_goal") or f"find {_target_phrase(contract)}"
    text = rng.choice([
        f"{'; '.join(facts)}. {goal[0].upper() + goal[1:]}.",
        f"I have the following: {'; '.join(facts)}. Please {goal}.",
        f"{'; '.join(facts)}. Can you {goal}?",
    ])
    assert not FORBIDDEN_IMPLICIT_RE.search(text), "goal renderer leaked graph cues"
    return text.strip(), "p41_goal_implicit_v1"


def render_domain_grounded_implicit(contract: Dict[str, Any], rng: random.Random
                                    ) -> Tuple[str, str]:
    domain = contract.get("domain") or "operations"
    entities = list(contract.get("entities") or _ENTITY_FALLBACK)
    ent = rng.choice(entities) if entities else "the case"
    facts = _facts_from_contract(contract, rng)
    goal = contract.get("user_goal") or f"determine {_target_phrase(contract)}"
    text = rng.choice([
        f"For {ent} in {domain}, {'; '.join(facts)}. {goal[0].upper() + goal[1:]}.",
        f"{ent[0].upper() + ent[1:]} records that {'; '.join(facts)}. "
        f"We need to {goal}.",
        f"Regarding {ent}: {'; '.join(facts)}. Kindly {goal}.",
    ])
    assert not FORBIDDEN_IMPLICIT_RE.search(text), "domain renderer leaked graph cues"
    return text.strip(), "p41_domain_grounded_v1"


_RENDERERS = {
    "GRAPH_EXPLICIT": render_graph_explicit,
    "OPERATION_EXPLICIT_GRAPH_IMPLICIT": render_operation_explicit_graph_implicit,
    "SEMI_IMPLICIT": render_semi_implicit,
    "GOAL_BASED_IMPLICIT": render_goal_based_implicit,
    "DOMAIN_GROUNDED_IMPLICIT": render_domain_grounded_implicit,
}


def render_query(contract: Dict[str, Any], query_mode: str,
                 rng: random.Random) -> Tuple[str, str]:
    fn = _RENDERERS.get(query_mode)
    if fn is None:
        raise ValueError(f"unknown query mode {query_mode!r}")
    return fn(contract, rng)


def build_semantic_contract(task_id: str, workflow: WorkflowFamily,
                            nodes: Sequence[Dict[str, Any]],
                            answer: Any, *,
                            query_mode: str,
                            rng: random.Random,
                            style_seed: str = "") -> Dict[str, Any]:
    consts = _constants(nodes)
    entities = list(workflow.entity_pool) or list(_ENTITY_FALLBACK)
    rng.shuffle(entities)
    facts = []
    for i, tmpl in enumerate(workflow.fact_templates):
        if i >= len(consts):
            break
        try:
            facts.append(tmpl.format(
                **{workflow.required_roles[j]: _fmt(consts[j])
                   for j in range(min(len(workflow.required_roles), len(consts)))},
                **{f"v{j}": _fmt(consts[j]) for j in range(len(consts))}))
        except (KeyError, IndexError):
            facts.append(f"{workflow.required_roles[i] if i < len(workflow.required_roles) else 'value'} is {_fmt(consts[i])}")
    if not facts:
        for i, c in enumerate(consts):
            role = workflow.required_roles[i] if i < len(workflow.required_roles) else f"value_{i}"
            facts.append(f"{role.replace('_', ' ')} is {_fmt(c)}")
    summary = [{"primitive_id": n.get("primitive_id"),
                "capability_family": n.get("capability_family"),
                "role": n.get("output_role") or ""} for n in nodes]
    return {
        "task_id": task_id,
        "language": "en",
        "domain": workflow.domain,
        "user_goal": workflow.user_goal_template,
        "workflow_id": workflow.workflow_id,
        "entities": entities[:4],
        "facts": facts,
        "constants": consts,
        "units": list(workflow.allowed_units),
        "target_variable": {"role": workflow.target_role, "value_type": type(answer).__name__},
        "semantic_program_summary": summary,
        "required_relations": list(workflow.semantic_constraints),
        "query_mode": query_mode,
        "forbidden_terms": ["stage", "step 1", "the stages are related",
                            "the figure from", "use the result from"],
        "style_seed": style_seed or short_style(rng),
        "max_words": 100,
    }


def short_style(rng: random.Random) -> str:
    return rng.choice(["concise", "polite", "direct", "business", "neutral"])


def query_template_fingerprint(text: str) -> str:
    """Normalize away numbers, quoted strings, currency codes."""
    t = text.lower()
    t = re.sub(r"'[^']*'", "<STR>", t)
    t = re.sub(r"\b\d+(?:\.\d+)?\b", "<NUM>", t)
    t = re.sub(r"\b(?:eur|usd|czk|gbp|km|m|cm|kg|hours?|minutes?|days?)\b",
               "<UNIT>", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t
