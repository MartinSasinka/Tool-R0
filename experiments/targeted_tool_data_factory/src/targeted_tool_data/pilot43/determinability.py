"""Can a query determine this program, and if not, what rule has to be stated?

Two plan shapes are not answerable from a goal and a bag of facts, however well
the query is written:

* a **computed criterion** -- a threshold, minimum or pad width produced by
  another step. "Which parts came out oversized?" cannot be answered unless the
  reader is told that oversized means *above the average plus the spread*: the
  rule exists only in the graph.
* an **opaque composite sink** -- a formatting or concatenation step that welds
  unrelated quantities together. "Give me the audit reference" cannot be answered
  unless the reader is told that the reference is *the scheme and host, a dash,
  then the path depth plus the length of the last segment plus the port*.

Both were rejected by the first critic in the smoke stage, correctly: the query
did not entail the program. The answer is not to weaken the critic but to *state
the rule*, and to route such tasks to the modes that are allowed to state it. A
task whose rule is stated is a task whose nodes are all required.

The rule text is derived from the program, never authored per plan, so it cannot
drift away from what the program does. It is phrased in domain words ("the spread
of the measured lengths") rather than capability names, and it becomes part of the
contract's own vocabulary -- the leakage checks treat it exactly like a fact
description, because that is what it is.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from .blueprints import Plan
from .ops import Op, build_ops
from .program import Program, Ref

#: A task whose rule is inherent in the goal; every mode may ask for it.
SELF_EVIDENT = "SELF_EVIDENT"
#: A task that is answerable only when the query states the rule below.
NEEDS_RULE = "NEEDS_RULE"
#: A task whose rule cannot be stated as a sentence a person would say -- a
#: composite of a composite of a composite. Such a task is dropped rather than
#: rendered: a query nobody could read is not a hard sample, it is a broken one.
NOT_STATABLE = "NOT_STATABLE"
#: Beyond this the rule reads as a program listing rather than a working
#: convention. Measured on the smoke stage: the readable rules sat under 180
#: characters, the unreadable ones ran past 300.
MAX_RULE_CHARS = 240

#: parameter names whose value *is* a rule rather than a subject of the request
CRITERION_PARAMS = frozenset({"threshold", "minimum", "cutoff", "limit",
                              "tolerance", "band"})
#: ``format.pad.width`` is a layout rule; ``geometry.rectangle_area.width`` is a
#: measurement the user states, so the parameter name alone cannot separate them
CRITERION_WIDTH_CAPABILITIES = frozenset({"format.pad"})
#: capabilities that only *present* a value: welding two computed values together
#: with one of these yields an answer shape nobody would ask for by name
COMPOSING_CAPABILITIES = frozenset({
    "format.tag", "format.join_fields", "string.concat", "string.join",
    "format.pad", "list.combine_append", "list.combine_concat",
})


@dataclass(frozen=True)
class Determinability:
    level: str
    reasons: Tuple[str, ...]
    specification: Tuple[str, ...]

    @property
    def needs_rule(self) -> bool:
        return self.level == NEEDS_RULE

    def as_row(self) -> Dict[str, Any]:
        return {"determinability": self.level, "reasons": list(self.reasons),
                "specification": list(self.specification)}


# ── verbalising a value ──────────────────────────────────────────────────
#: How the *output* of a capability reads as a noun phrase. Keys are matched
#: against the capability tail, longest-specific first, the same way
#: :func:`.queries.operation_phrase` works. These are domain words on purpose: the
#: reader has to understand the rule, not identify the tool.
VALUE_PHRASE: Tuple[Tuple[str, str], ...] = (
    ("aggregate_count", "how many of {0} qualify"),
    ("aggregate_sum", "the total of {0}"),
    ("aggregate_mean", "the average of {0}"),
    ("aggregate_max", "the largest of {0}"),
    ("aggregate_min", "the smallest of {0}"),
    ("reduce_count_above", "how many of {0} sit above {1}"),
    ("reduce_count", "how many entries {0} has"),
    ("reduce_sum", "the total of {0}"),
    ("reduce_mean", "the average of {0}"),
    ("reduce_max", "the largest of {0}"),
    ("reduce_min", "the smallest of {0}"),
    ("count_length", "the length of {0}"),
    ("count_occurrences", "how often {1} occurs in {0}"),
    ("stdev", "the spread of {0}"),
    ("variance", "the spread of {0}"),
    ("median", "the middle value of {0}"),
    ("mean", "the average of {0}"),
    ("sum_three", "{0} plus {1} plus {2}"),
    ("sum", "the total of {0}"),
    ("basename", "the last segment of {0}"),
    ("depth", "how many levels deep {0} runs"),
    ("extension", "the extension of {0}"),
    ("parent", "the folder holding {0}"),
    ("scheme", "the scheme of {0}"),
    ("domain", "the host of {0}"),
    ("parse_port", "the port of {0}"),
    ("path", "the path of {0}"),
    ("extract_digits", "the digits in {0}"),
    ("parse_number", "{0} read as a number"),
    ("apply_tax", "{0} with tax added"),
    ("compound_growth", "{0} carried forward the stated number of periods"),
    ("share_percent", "{0} as a share of {1}"),
    ("percent_of", "{1} percent of {0}"),
    ("increase_by_percent", "{0} raised by {1} percent"),
    ("decrease_by_percent", "{0} cut by {1} percent"),
    ("abs_difference", "the gap between {0} and {1}"),
    ("difference", "the difference between {0} and {1}"),
    ("subtract", "{0} less {1}"),
    ("multiply", "{0} times {1}"),
    ("divide", "{0} over {1}"),
    ("average_two", "the average of {0} and {1}"),
    ("add", "{0} plus {1}"),
    ("fixed", "{0} written to the stated number of decimals"),
    ("currency", "{0} written as an amount"),
    ("round", "{0} rounded"),
    ("values", "the figures in {0}"),
    ("keys", "the names in {0}"),
    ("lookup", "what {0} holds for {1}"),
    ("concat", "{0} followed by {1}"),
    ("tag", "{0} followed by {1}"),
    ("pad", "{0} padded out to {1}"),
)

#: How a criterion consumer reads as a rule. Keyed by capability.
CRITERION_RULE: Dict[str, str] = {
    "list.filter": "only the values above {limit} count",
    "list.reduce_count_above": "a value counts as over the line "
                              "when it is above {limit}",
    "dictionary.aggregate_filter": "a category counts as over the line "
                                   "when it is above {limit}",
    "record.aggregate_count": "a record counts when its figure "
                              "reaches at least {limit}",
    "comparison.at_least": "the figure has to reach {limit}",
    "format.pad": "pad it out to {limit}",
}
#: Fallback when a criterion consumer has no rule template yet.
GENERIC_RULE = "the line to compare against is {limit}"

MAX_DEPTH = 4
#: Below this depth the phrase would read as a program listing rather than a rule.
DEEP_FALLBACK = "the figure that follows from the numbers above"


def _phrase_for(capability: str) -> str:
    tail = capability.split(".", 1)[-1]
    for key, template in VALUE_PHRASE:
        if key in tail:
            return template
    return ""


class _Verbaliser:
    """Turns a node's output into a noun phrase, recursing through its inputs."""

    def __init__(self, prog: Program, ops: Dict[str, Op], plan: Plan,
                 role_values: Dict[str, Any], render: Any) -> None:
        self.prog = prog
        self.ops = ops
        self.render = render
        #: an intermediate the recursion cannot afford to unfold is still
        #: nameable: the plan author gave the step a purpose in domain words
        self.purposes = {s.node_id: (s.purpose or "").strip()
                         for s in plan.steps}
        self.by_value: List[Tuple[Any, str]] = []
        for role in plan.roles:
            if role.name in role_values:
                self.by_value.append((role_values[role.name],
                                      role.description or
                                      role.name.replace("_", " ")))

    def leaf(self, value: Any) -> str:
        for known, description in self.by_value:
            if type(known) is type(value) and known == value:
                return f"the {description}" if not description.startswith(
                    ("the ", "a ", "an ", "how ", "which ")) else description
        return self.render(value)

    def named(self, node_id: str) -> str:
        """The step's own purpose as a noun phrase, or "" when it has none."""
        purpose = self.purposes.get(node_id, "")
        if len(purpose) < 4:
            return ""
        if purpose.startswith(("the ", "a ", "an ", "how ", "which ", "what ")):
            return purpose
        return f"the {purpose}"

    def node(self, node_id: str, depth: int = 0) -> str:
        if depth >= MAX_DEPTH:
            return self.named(node_id) or DEEP_FALLBACK
        nd = self.prog.node(node_id)
        op = self.ops[nd.op]
        template = _phrase_for(op.capability)
        parts: List[str] = []
        for param in op.params:
            value = nd.args.get(param.name)
            if isinstance(value, Ref):
                parts.append(self.node(value.node_id, depth + 1))
            else:
                parts.append(self.leaf(value))
        if not template:
            return parts[0] if parts else (self.named(node_id) or DEEP_FALLBACK)
        while len(parts) < 3:
            parts.append("")
        return template.format(*parts).strip()


# ── classification ──────────────────────────────────────────────────────
def _criterion_hits(prog: Program, ops: Dict[str, Op]
                    ) -> List[Tuple[str, str, str]]:
    """``(node_id, capability, producing_node)`` per computed criterion."""
    out: List[Tuple[str, str, str]] = []
    for nd in prog.nodes:
        op = ops[nd.op]
        for param in op.params:
            value = nd.args.get(param.name)
            if not isinstance(value, Ref):
                continue
            is_criterion = (param.name in CRITERION_PARAMS
                            or (param.name == "width" and op.capability
                                in CRITERION_WIDTH_CAPABILITIES))
            if is_criterion:
                out.append((nd.node_id, op.capability, value.node_id))
    return out


def _composite_sink(prog: Program, ops: Dict[str, Op]) -> str:
    sink = prog.node(prog.sink)
    capability = ops[sink.op].capability
    if capability not in COMPOSING_CAPABILITIES:
        return ""
    refs = [v for v in sink.args.values() if isinstance(v, Ref)]
    return capability if len(refs) >= 2 else ""


def _tautological(rule: str) -> bool:
    """True when a rule explains a thing as itself.

    The composite-sink verbaliser can bottom out in the same role it started
    from, yielding "the average appended as a reference row is the quantity on
    each line" -- a sentence that tells the reader nothing, and which a writer
    then reproduces verbatim. Such a rule is not statable.
    """
    subject, _, predicate = rule.partition(" is ")
    if not predicate:
        return False
    subject = subject.strip().lower().removeprefix("the ")
    predicate = predicate.strip().lower()
    return bool(subject) and (subject == predicate.removeprefix("the ")
                              or subject in predicate.split(" followed by ")[0])


def classify(prog: Program, plan: Plan, role_values: Dict[str, Any], *,
             target_phrase: str = "", render: Any = None,
             ops: Dict[str, Op] | None = None) -> Determinability:
    """Level, reasons and the rule sentences a query must state."""
    ops = ops or build_ops()
    render = render or (lambda v: str(v))
    verb = _Verbaliser(prog, ops, plan, role_values, render)

    reasons: List[str] = []
    spec: List[str] = []
    for node_id, capability, producer in _criterion_hits(prog, ops):
        reasons.append(f"computed_criterion:{capability}@{node_id}")
        rule = CRITERION_RULE.get(capability, GENERIC_RULE)
        spec.append(rule.format(limit=verb.node(producer, 1)))

    composite = _composite_sink(prog, ops)
    if composite:
        reasons.append(f"composite_sink:{composite}")
        subject = target_phrase or "the answer"
        spec.append(f"{subject} is {verb.node(prog.sink, 1)}")

    spec = list(dict.fromkeys(spec))
    unstatable = [rule for rule in spec
                  if len(rule) > MAX_RULE_CHARS or DEEP_FALLBACK in rule
                  or _tautological(rule)]
    if unstatable:
        reasons.append(f"rule_not_statable:{len(unstatable)}")
        level = NOT_STATABLE
    elif reasons:
        level = NEEDS_RULE
    else:
        level = SELF_EVIDENT
    return Determinability(level=level, reasons=tuple(reasons),
                           specification=tuple(spec))


def plan_needs_rule(plan: Plan, ops: Dict[str, Op] | None = None) -> bool:
    """Plan-level screen, usable before any instance exists.

    Mode assignment runs over shortlist rows rather than built instances, so the
    routing decision has to be answerable from the plan alone. It is the same
    test: both shapes are properties of the wiring, not of the values.
    """
    ops = ops or build_ops()
    by_capability = {op.capability: op for op in ops.values()}
    for step in plan.steps:
        op = by_capability.get(step.capability)
        if op is None:
            continue
        for param, arg in zip(op.params, step.args):
            if not arg.startswith("@"):
                continue
            if (param.name in CRITERION_PARAMS
                    or (param.name == "width"
                        and step.capability in CRITERION_WIDTH_CAPABILITIES)):
                return True
    sink = plan.step(plan.sink)
    if sink.capability in COMPOSING_CAPABILITIES:
        if sum(1 for a in sink.args if a.startswith("@")) >= 2:
            return True
    return False


def needs_rule_plan_ids(ops: Dict[str, Op] | None = None
                        ) -> Dict[Tuple[str, str], bool]:
    """``(workflow_id, plan_id) -> needs_rule`` for the whole registry."""
    from .blueprints import all_blueprints

    ops = ops or build_ops()
    out: Dict[Tuple[str, str], bool] = {}
    for bp in all_blueprints():
        for plan in bp.plans:
            out[(bp.workflow_id, plan.plan_id)] = plan_needs_rule(plan, ops)
    return out
