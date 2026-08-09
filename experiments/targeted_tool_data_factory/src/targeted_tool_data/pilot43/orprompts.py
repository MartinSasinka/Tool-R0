"""Prompt builders for the Pilot4.3 writer, critics and rewriter.

The failure mode these prompts prevent is a writer that quietly *changes the
problem*: it rounds a number, swaps a unit, adds "but no more than 500", or
answers the question inside the query. Every such edit makes the executed
program a wrong oracle for the text, and no downstream metric can see it,
because the program still runs. The constraints are therefore stated as
prohibitions the model must self-report on, and the same list is what
``qvalidate`` checks deterministically afterwards.

The second prohibition group is disclosure: naming a tool, a variable, a node
id, the number of calls or the dependency order turns an implicit query into an
explicit one and destroys the mode distribution the dataset is built around.

Prompt text is versioned and hashed (:data:`PROMPT_HASHES`) so the freeze
manifest records exactly which wording produced the corpus.
"""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Dict, Mapping, NamedTuple, Sequence

from . import PROMPT_VERSION_PREFIX

WRITER_VERSION = "pilot43.writer.v6"
CRITIC_VERSION = "pilot43.critic.v6"
CRITIC2_VERSION = "pilot43.critic2.v6"
REWRITE_VERSION = "pilot43.rewrite.v6"


class Prompt(NamedTuple):
    """``(system, user, json_schema, prompt_version)`` -- a tuple on purpose."""

    system: str
    user: str
    schema: Dict[str, Any]
    prompt_version: str


# ── shared hard constraints ──────────────────────────────────────────────
WRITER_RULES = """\
HARD CONSTRAINTS (a single violation makes the sample unusable):
1. Pairing: each entry of `stated_facts` is one fact. Write its `value` next to
   wording taken from its own `means`, so a reader can tell which quantity is
   which. Never attach a value to a different fact's meaning, and never merge
   two facts into one number or list.
2. Numbers: reproduce every value exactly as given. Do not add a number, drop a
   number, round, re-scale, average, reorder a list or reformat one.
3. Units: keep the unit written next to each value. Never convert, translate or
   drop a unit, and never attach a unit to a value that had none.
4. Target: ask for exactly the stated `target`. Do not narrow it, widen it,
   split it into parts or replace it with a related quantity.
5. Conditions: add no constraint, threshold, limit, deadline, tolerance,
   preference or assumption that the contract does not state.
6. Names: never write a tool name, function name, capability name, parameter
   name, variable label, node id, or any identifier containing an underscore.
7. Procedure: never disclose how many operations are needed, never number or
   order steps, and never refer to "the result of the previous step" or any
   other intermediate value.
8. Answer: never state, approximate or hint at the answer.
9. Coverage: every fact must appear, and the request must end with one
   unambiguous question for the target.
10. Rules: when `rules_to_state` is present, state every rule it lists, in your
   own words, as the user's own convention ("we treat anything above ... as
   ..."). These are not extra conditions: without them the request cannot be
   answered. Keep the quantities they mention; invent no new ones.
11. Register: ordinary working English, 2-5 sentences, no bullet lists, no
   placeholders, no markdown, no restating the same fact twice. Somebody with
   the facts in front of them should be able to ask this out loud.
12. Prose only: never copy the layout below into the request. No field-and-value
   pairs, no field names lifted from the contract. Every value belongs inside a
   sentence that says what it is, the way a colleague would say it.
"""

MODE_INSTRUCTIONS: Dict[str, str] = {
    "DOMAIN_GROUNDED_IMPLICIT": """\
MODE DOMAIN_GROUNDED_IMPLICIT: write the request as it would arrive inside the
scenario. At least one of the named people, sites, organisations or periods must
appear, so the facts are embedded in a real situation rather than listed. Name no
operation at all: the reader learns what has to be done only from the situation
and the target.
""",
    "GOAL_BASED_IMPLICIT": """\
MODE GOAL_BASED_IMPLICIT: state the goal and the facts plainly, with little or
no scenario colour. Names are optional here and usually better left out. Name no
operation and give no hint of a sequence; the request is "here is the situation,
I need this outcome".
""",
    "SEMI_IMPLICIT": """\
MODE SEMI_IMPLICIT: you may mention at most one broad kind of work involved
(for example that there is a currency side to it, or that a share is involved),
phrased the way a colleague would say it. You may not name the individual
operations, their order, or how many there are.
""",
}

WRITER_SYSTEM = """\
You turn a query contract into one natural user request in English.
The contract lists the facts the user already knows -- each with the meaning it
has in the situation -- the people, places and periods involved, and the single
target the user wants. A hidden program computes that target from those facts;
you never see it and must never imply its shape.
Write the way a colleague writes when they have the figures in front of them and
need the answer: the situation first, the figures where they belong, the question
last.
Return only JSON matching the provided schema.
"""

REWRITE_SYSTEM = """\
You repair a user request that failed validation. You keep everything that was
correct and change only what the findings identify. The same hard constraints
apply to the repaired text as to the original; a repair that fixes one finding
by breaking another rule is worse than the text you were given.
Return only JSON matching the provided schema.
"""

CRITIC_SYSTEM = """\
You audit one synthetic tool-use sample. You see the workflow goal, the exact
program that will be executed as ground truth, its dependency edges, the
purpose of each node, the input facts with units, the observed value of every
node, the final answer, the query written for that program, and the findings of
a deterministic validator.
Your job is to decide whether the query and the program are the same problem.
Judge the sample, not the writing style, and do not repair anything.

How to read the query: it is meant to be *implicit*. A correct query states the
situation, the facts and the target, and says nothing about the operations, their
order or their number. A node is "required by the query" when reaching the
target from the stated facts needs that node's contribution -- not when the query
mentions it. Intermediate quantities are supposed to be unnamed; that is the
design, not a defect. An arithmetic step that must happen on the way from the
facts to the target is required, and the evidence for it is the span of the query
that makes it unavoidable (the facts it consumes, or the target it feeds).

`workflow_goal` is background: it describes the family of work the sample comes
from, and several different questions are asked within it. Judge
`workflow_matches_query` and `sink_answers_target` against `target`, not against
the goal line. A query whose closing question matches the target is correct even
when the background sentence would suggest a different question.

Some samples carry `rules_the_query_must_state`: conventions the user sets, such
as which line counts as "over" or how a reference is composed. The query is
required to state those, and the nodes that compute them are required *because*
the query states them. Treat such a rule as part of the problem, never as an
extra condition the writer invented.

PASS: the program answers exactly what the query asks, every fact in the query is
used by the program, every node contributes on the way to the target, nothing was
invented, and neither the graph nor the answer is disclosed.
REWRITE: the pairing is sound but the text is ambiguous, unnatural, pairs a value
with the wrong meaning, or leaks a hint of the procedure.
REJECT: the query and the program are different problems, a fact was changed or
invented, a node cannot contribute to the target at all, or the answer is given
away.
Return only JSON matching the provided schema.
"""

CRITIC2_SYSTEM = CRITIC_SYSTEM + """\
You are the second, independent critic. You were not shown the first critic's
verdict; form your own. Disagreement is expected and useful, so do not soften a
finding to look reasonable.
"""


# ── schemas ──────────────────────────────────────────────────────────────
_SELF_CHECK_KEYS = (
    "no_new_or_changed_numbers", "no_unit_changes", "target_unchanged",
    "no_added_conditions", "no_tool_or_variable_names",
    "call_count_not_disclosed", "graph_not_disclosed", "answer_not_stated",
)

WRITER_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["query", "facts_stated", "units_stated", "target_sentence",
                 "self_check", "notes"],
    "properties": {
        "query": {"type": "string"},
        "facts_stated": {"type": "array", "items": {"type": "string"}},
        "units_stated": {"type": "array", "items": {"type": "string"}},
        "target_sentence": {"type": "string"},
        "self_check": {
            "type": "object",
            "additionalProperties": False,
            "required": list(_SELF_CHECK_KEYS),
            "properties": {k: {"type": "boolean"} for k in _SELF_CHECK_KEYS},
        },
        "notes": {"type": "array", "items": {"type": "string"}},
    },
}

REWRITE_SCHEMA: Dict[str, Any] = copy.deepcopy(WRITER_SCHEMA)
REWRITE_SCHEMA["properties"]["changes_made"] = {"type": "array",
                                                "items": {"type": "string"}}
REWRITE_SCHEMA["required"] = list(REWRITE_SCHEMA["required"]) + ["changes_made"]

CRITIC_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["workflow_matches_query", "sink_answers_target",
                 "all_query_facts_used", "all_program_nodes_required",
                 "no_extra_conditions", "units_semantically_valid",
                 "query_unambiguous", "query_natural", "graph_not_disclosed",
                 "node_alignment", "verdict"],
    "properties": {
        "workflow_matches_query": {"type": "boolean"},
        "sink_answers_target": {"type": "boolean"},
        "all_query_facts_used": {"type": "boolean"},
        "all_program_nodes_required": {"type": "boolean"},
        "no_extra_conditions": {"type": "boolean"},
        "units_semantically_valid": {"type": "boolean"},
        "query_unambiguous": {"type": "boolean"},
        "query_natural": {"type": "boolean"},
        "graph_not_disclosed": {"type": "boolean"},
        "node_alignment": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["node_id", "required_by_query", "query_evidence",
                             "semantic_role_matches", "aligned"],
                "properties": {
                    "node_id": {"type": "string"},
                    "required_by_query": {"type": "boolean"},
                    "query_evidence": {"type": "string"},
                    "semantic_role_matches": {"type": "boolean"},
                    "aligned": {"type": "boolean"},
                },
            },
        },
        "verdict": {"type": "string", "enum": ["PASS", "REWRITE", "REJECT"]},
    },
}


# ── builders ─────────────────────────────────────────────────────────────
def _json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str,
                      sort_keys=True)


def _fact_block(facts: Sequence[Mapping[str, Any]]) -> str:
    """One line per fact, meaning and value side by side.

    A JSON dump of parallel lists is what let a writer pair the wrong value with
    the wrong meaning, so the pairing is made syntactically unavoidable here.

    A dotted separator was tried instead of the equals sign, to stop writers
    copying "meaning = value" into the query. On the same 50 smoke tasks it cost
    six points of deterministic pass rate and six of critic pass rate: the equals
    sign is what makes the pairing unmistakable, and a handful of spreadsheet-ish
    queries is the cheaper problem.
    """
    lines = []
    for fact in facts:
        unit = str(fact.get("unit") or "")
        suffix = f"   [unit: {unit}]" if unit else ""
        lines.append(f'- {fact.get("means") or fact.get("name")}'
                     f' = {fact.get("value")}{suffix}')
    return "\n".join(lines) or "- (no facts)"


def _rule_block(rules: Sequence[str]) -> str:
    """The conventions the reader could not otherwise know.

    These read like extra conditions and are the opposite: the task is
    unanswerable without them, because the rule lives in the hidden program.
    """
    if not rules:
        return ""
    lines = "\n".join(f"- {rule}" for rule in rules)
    return ("\nRULES THE USER'S OWN PRACTICE SETS -- state every one of them in "
            f"your own words:\n{lines}\n")


def _contract_block(view: Mapping[str, Any]) -> str:
    entities = view.get("scenario_entities") or {}
    names = (sorted(entities.values()) if isinstance(entities, dict)
             else sorted(str(e) for e in entities))
    return "\n".join([
        # background, not the question: a workflow goal covers a family of plans,
        # so a writer that turns it into the question asks for the wrong thing
        f"BACKGROUND, why the user is looking at this (never the question): "
        f"{view.get('user_goal', '')}",
        f"AREA OF WORK: {view.get('domain', '')}",
        "PEOPLE, PLACES AND PERIODS AVAILABLE (use the ones that fit, at least "
        "one): " + ", ".join(names),
        "",
        "FACTS THE USER ALREADY HAS -- each value belongs with its own meaning:",
        _fact_block(view.get("stated_facts") or []),
        _rule_block(view.get("rules_to_state") or []),
        f"THE ONE QUESTION TO ASK -- the target: {view.get('target', '')}",
        f"SHAPE OF THE ANSWER (never state it): {view.get('answer_type', '')}",
        "",
        "WORDS YOU MAY NEVER WRITE:",
        _json(view.get("must_not_disclose") or []),
    ])


def writer_prompt(contract_payload: Mapping[str, Any], mode: str) -> Prompt:
    """Turn a :class:`~.queries.QueryContract` payload into an implicit query."""
    if mode not in MODE_INSTRUCTIONS:
        raise ValueError(f"no writer instructions for mode {mode!r}")
    user = "\n".join([
        WRITER_RULES,
        MODE_INSTRUCTIONS[mode],
        _contract_block(contract_payload),
        "",
        "Write the request now. Fill `facts_stated` with the fact meanings you "
        "used, `units_stated` with the units you kept, `target_sentence` with "
        "the sentence that asks for the target, and set every `self_check` flag "
        "truthfully; a false flag is a rejected sample.",
    ])
    return Prompt(WRITER_SYSTEM, user, WRITER_SCHEMA, WRITER_VERSION)


def critic_prompt(context: Mapping[str, Any], query: str,
                  validator_findings: Mapping[str, Any], *,
                  second_opinion: bool = False) -> Prompt:
    """Audit one (program, query) pair against the executed oracle."""
    payload = {
        "workflow_goal": context.get("workflow_goal", ""),
        "target": context.get("target", ""),
        "answer_type": context.get("answer_type", ""),
        "program": list(context.get("program") or []),
        "reconstructed_edges": [list(e) for e in (context.get("edges") or [])],
        "node_purposes": dict(context.get("node_purposes") or {}),
        "input_facts": list(context.get("input_facts") or []),
        "oracle_observations": dict(context.get("observations") or {}),
        "answer": context.get("answer"),
        "query": query,
        # the conventions the query was required to state; a node that exists only
        # to apply one of these is required *because* the query states the rule
        "rules_the_query_must_state": list(context.get("specification") or []),
        "deterministic_validator": _slim_findings(validator_findings),
    }
    user = "\n".join([
        "SAMPLE UNDER AUDIT:",
        _json(payload),
        "",
        "Return one `node_alignment` entry per program node, in program order. "
        "`query_evidence` is the shortest span of the query that makes the "
        "node's contribution unavoidable: the facts it consumes, or the part of "
        "the target it produces. Mark `required_by_query` false only when the "
        "node's output could be dropped and the query still answered.",
    ])
    system = CRITIC2_SYSTEM if second_opinion else CRITIC_SYSTEM
    version = CRITIC2_VERSION if second_opinion else CRITIC_VERSION
    return Prompt(system, user, CRITIC_SCHEMA, version)


def rewrite_prompt(contract_payload: Mapping[str, Any], mode: str, query: str,
                   critic_findings: Mapping[str, Any],
                   validator_findings: Mapping[str, Any]) -> Prompt:
    """Repair a query under exactly the writer's constraints."""
    if mode not in MODE_INSTRUCTIONS:
        raise ValueError(f"no writer instructions for mode {mode!r}")
    user = "\n".join([
        WRITER_RULES,
        MODE_INSTRUCTIONS[mode],
        _contract_block(contract_payload),
        "",
        "REJECTED QUERY:",
        query,
        "",
        "CRITIC FINDINGS:",
        _json(_slim_critic(critic_findings)),
        "",
        "DETERMINISTIC VALIDATOR FINDINGS:",
        _json(_slim_findings(validator_findings)),
        "",
        "Rewrite the request so that every finding is resolved and no other "
        "constraint is broken. List what you changed in `changes_made`.",
    ])
    return Prompt(REWRITE_SYSTEM, user, REWRITE_SCHEMA, REWRITE_VERSION)


def _slim_findings(findings: Mapping[str, Any]) -> Dict[str, Any]:
    """Only the failures. A full dump lets the critic pattern-match the
    validator instead of reading the sample."""
    layers = dict(findings.get("layers") or {})
    failed = {name: layer for name, layer in layers.items()
              if not (layer or {}).get("passed", True)}
    return {
        "passed": bool(findings.get("passed")),
        "failed_layers": sorted(failed),
        "details": failed,
        "classified_mode": (findings.get("classification") or {}).get(
            "actual_query_mode", ""),
    }


def _slim_critic(critic: Mapping[str, Any]) -> Dict[str, Any]:
    misaligned = [n for n in (critic.get("node_alignment") or [])
                  if not n.get("aligned", True)]
    return {
        "verdict": critic.get("verdict", ""),
        "failed_checks": sorted(k for k, v in critic.items()
                                if isinstance(v, bool) and not v),
        "misaligned_nodes": misaligned,
    }


# ── freeze support ───────────────────────────────────────────────────────
def _hash_prompt(*parts: Any) -> str:
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


PROMPT_HASHES: Dict[str, str] = {
    WRITER_VERSION: _hash_prompt(WRITER_SYSTEM, WRITER_RULES, MODE_INSTRUCTIONS,
                                 WRITER_SCHEMA),
    CRITIC_VERSION: _hash_prompt(CRITIC_SYSTEM, CRITIC_SCHEMA),
    CRITIC2_VERSION: _hash_prompt(CRITIC2_SYSTEM, CRITIC_SCHEMA),
    REWRITE_VERSION: _hash_prompt(REWRITE_SYSTEM, WRITER_RULES,
                                  MODE_INSTRUCTIONS, REWRITE_SCHEMA),
}

PROMPT_VERSIONS: Sequence[str] = tuple(PROMPT_HASHES)

# a version that does not carry the run prefix would pass the client's isolation
# check by accident, so it is caught at import time rather than at request time
for _version in PROMPT_VERSIONS:
    if not _version.startswith(PROMPT_VERSION_PREFIX + "."):
        raise RuntimeError(f"prompt version {_version!r} is not a Pilot4.3 version")
