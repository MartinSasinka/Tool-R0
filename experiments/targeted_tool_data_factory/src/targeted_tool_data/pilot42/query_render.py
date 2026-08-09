"""Deterministic, non-graph-leaking renderers for Pilot4.2 query modes."""
from __future__ import annotations

import re
from typing import Any, Dict

from ..util import short_hash

FORBIDDEN_GRAPH_PHRASES = (
    "stages are related", "stage 1", "stage 2", "first call", "second call",
    "output of", "feed the result", "$var", "dependency graph", "tool named",
    "first calculate", "then calculate", "next compute", "use the previous result",
    "call the function", "use tool",
)
FORBIDDEN_IMPLICIT = re.compile(
    r"\b(stage|step\s*\d|first calculate|then calculate|next compute|"
    r"use the previous result|the stages are related|call the function|"
    r"use tool|fan-?in|fan-?out|\$var_)\b",
    re.I,
)


def _value(fact: Dict[str, Any]) -> str:
    value = fact["value"]
    text = (str(int(value)) if isinstance(value, float) and value.is_integer()
            else str(value))
    return f"{text} {fact.get('unit', '')}".strip()


def render_query(contract: Dict[str, Any], mode: str | None = None) -> str:
    mode = mode or contract["query_mode"]
    entity = contract["entity"]
    facts = contract["facts"]
    statements = [f"{f['role'].replace('_', ' ')} is {_value(f)}" for f in facts]
    target = contract["natural_language_assets"].get(
        "target_phrase", contract["target_role"].replace("_", " "))
    joined = "; ".join(statements)
    goal = contract["user_goal"]
    domain = contract["domain"].replace("_", " ")
    if mode == "GRAPH_EXPLICIT":
        text = (f"For the {entity}, given {joined}, perform the workflow to "
                f"{goal}. Report {target}.")
    elif mode == "OPERATION_EXPLICIT_GRAPH_IMPLICIT":
        text = (f"Using {joined} for the {entity}, {goal}. "
                f"Do not invent extra values. Report {target}.")
    elif mode == "SEMI_IMPLICIT":
        text = (f"The {entity} has {joined}. {goal.capitalize()}. "
                f"What is {target}?")
    elif mode == "GOAL_BASED_IMPLICIT":
        text = (f"{joined}. {goal.capitalize()} for the {entity}. "
                f"Report {target}.")
    else:  # DOMAIN_GROUNDED_IMPLICIT
        text = (f"In this {domain} setting, the {entity} has {joined}. "
                f"Determine {target}.")
    if mode != "GRAPH_EXPLICIT" and FORBIDDEN_IMPLICIT.search(text):
        raise ValueError("renderer produced graph-leaking language")
    return text


def query_template_fingerprint(text: str) -> str:
    normalized = re.sub(r"[-+]?\d+(?:\.\d+)?", "<NUM>", text.lower())
    normalized = re.sub(
        r"\b(?:usd|eur|czk|percent|items?|minutes?|hours?|days?)\b",
        "<UNIT>", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return "qtf42_" + short_hash(normalized)
