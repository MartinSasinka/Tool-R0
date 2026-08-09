"""Query contracts and deterministic renderers.

A :class:`QueryContract` is the only thing a renderer -- deterministic or LLM --
is allowed to see: the facts the user states, their units, the entities of the
scenario, and the target. What it must *not* disclose travels with it explicitly
(tool names, capability names, node ids, call count, edges), so both the writer
prompt and the deterministic validator work from the same list.

Pilot4.2 rendered every task from one template family, which produced the
"``a`` is 12, ``b`` is 7, scale, adjust, then compare" register that the audit
flagged. Here the deterministic renderer is a *composition*: an opening frame, a
fact-presentation pattern, an ordering and a closing question form, all drawn
per task. That yields ~1000 distinct syntactic skeletons, which is what the
1 %-per-skeleton diversity gate needs, and none of the banned phrasings appear
anywhere in the module.

Deterministic rendering is intended for the explicit modes; the implicit modes
are written by the OpenRouter writer and only fall back to a deterministic
scenario when the writer is unavailable (which is recorded, never hidden).
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

from ..repro import sha256_obj
from . import determinability, semtypes as st
from .blueprints import Blueprint, Plan
from .build import Instance
from .ops import build_ops

# ── scenario assets ──────────────────────────────────────────────────────
ORGS = ("Neomark", "Vantera", "Brightloom", "Kestrel Foods", "Orlin Systems",
        "Aurea Labs", "Northwind Rail", "Calder & Sons", "Petrichor Media",
        "Solvay Print", "Mistral Freight", "Halden Care", "Torvik Energy",
        "Blue Harbour", "Ardent Tools", "Verdana Hotels", "Kirin Optics",
        "Sable Textiles", "Onyx Analytics", "Fenwick Motors")
SITES = ("the Brno depot", "the Leipzig warehouse", "the harbour branch",
         "the north workshop", "the Tallinn office", "the airport kiosk",
         "the Riverside plant", "the central lab", "the Utrecht hub",
         "the seasonal pop-up", "the service yard", "the training centre")
PEOPLE = ("Marta", "Dan", "Priya", "Tomas", "Ines", "Karel", "Yara", "Nils",
          "Ada", "Bruno", "Lena", "Osman")
ROLES_PEOPLE = ("the shift lead", "our accountant", "the site manager",
                "the duty planner", "a client", "the auditor", "the intern",
                "the regional buyer")
PROJECTS = ("the spring rollout", "the Q3 refit", "the ISO review",
            "the pilot batch", "the winter campaign", "the migration",
            "the maintenance window", "the tender response")
PERIODS = ("last week", "this month", "yesterday", "the last quarter",
           "the current sprint", "the previous shift", "this morning")

#: entity_family -> flavour words. Unknown families fall back to the pools above,
#: which keeps a new workflow module readable without touching this file.
FAMILY_FLAVOUR: Dict[str, Tuple[str, ...]] = {
    "procurement": ("the purchasing team", "our supplier", "the framework contract"),
    "retail": ("the shop floor", "the loyalty programme", "the weekend promotion"),
    "finance": ("the ledger", "the month-end close", "the reimbursement batch"),
    "warehouse": ("the picking line", "the overflow rack", "the inbound dock"),
    "operations": ("the duty roster", "the incident log", "the handover notes"),
    "logistics": ("the delivery round", "the return leg", "the courier slot"),
    "engineering": ("the calibration rig", "the test bench", "the spare kit"),
    "data": ("the export", "the nightly job", "the reporting sheet"),
    "web": ("the campaign link", "the tracking parameters", "the landing page"),
    "files": ("the archive folder", "the upload batch", "the backup set"),
    "lab": ("the sample tray", "the reference standard", "the QC sheet"),
    "hr": ("the timesheet", "the onboarding pack", "the shift swap"),
}

UNIT_WORD: Dict[str, str] = {
    st.PERCENTAGE: "%", st.DUR_S: "seconds", st.DUR_MIN: "minutes",
    st.DUR_H: "hours", st.DUR_D: "days", st.LEN_M: "m", st.LEN_KM: "km",
    st.MASS_KG: "kg", st.MASS_G: "g", st.VOL_L: "l", st.VOL_ML: "ml",
    st.TEMP_C: "°C", st.TEMP_F: "°F", st.BYTES: "bytes", st.SCORE: "points",
    st.AREA: "m²",
}

BANNED_PHRASES = (
    "a is ", "b is ", "v1 is ", "v2 is ", "do not invent",
    "report the result", "scale, adjust", "then compare the result",
)


def format_value(value: Any, sem: str = st.GENERIC) -> str:
    """Canonical rendering of a fact. Kept plain so validation can match it."""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        text = f"{value:.4f}".rstrip("0").rstrip(".")
        return text if text not in ("", "-") else "0"
    if isinstance(value, list):
        return ", ".join(format_value(v) for v in value)
    if isinstance(value, dict):
        return "; ".join(f"{k}: {format_value(v)}" for k, v in value.items())
    return str(value)


def _nested_numbers(value: Any) -> List[str]:
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [format_value(value)]
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            out.extend(_nested_numbers(item))
        return out
    if isinstance(value, dict):
        out = []
        for key, item in value.items():
            out.extend(_nested_numbers(key))
            out.extend(_nested_numbers(item))
        return out
    if isinstance(value, str):
        # dates and urls carry digits that the query must reproduce verbatim
        return [m for m in re.findall(r"-?\d+(?:\.\d+)?", value)]
    return []


def unit_of(sem: str, currency: str) -> str:
    if sem == st.MONEY:
        return currency
    return UNIT_WORD.get(sem, "")


def with_unit(value: Any, sem: str, currency: str) -> str:
    text = format_value(value, sem)
    unit = unit_of(sem, currency)
    if not unit:
        return text
    if unit == "%":
        return f"{text} %"
    if sem == st.MONEY:
        return f"{text} {unit}"
    return f"{text} {unit}"


@dataclass
class Fact:
    role: str
    value: Any
    sem: str
    description: str
    rendered: str
    unit: str


@dataclass
class QueryContract:
    task_id: str
    workflow_id: str
    plan_id: str
    domain: str
    natural_user_goal: str
    target_phrase: str
    answer_type: str
    requested_mode: str
    currency: str
    entities: Dict[str, str]
    facts: List[Fact]
    hidden_plan: List[Dict[str, str]]
    forbidden_terms: List[str]
    call_count: int
    seed: int
    #: rule sentences the query must state for the program to be determined; empty
    #: for a self-evident task. See :mod:`.determinability`.
    specification: Tuple[str, ...] = ()
    determinability: str = determinability.SELF_EVIDENT

    def fact_lines(self) -> List[str]:
        return [f"{f.description}: {f.rendered}" for f in self.facts]

    def numbers(self) -> List[str]:
        """Every number the query is allowed to contain, however deeply nested.

        Record lists (``[{"label": ..., "amount": 12.5}, ...]``) are the reason this
        walks recursively: a shallow version missed their numbers, and the fact check
        then read the query's own table as invented values.
        """
        out: List[str] = []
        for f in self.facts:
            if isinstance(f.value, (int, float)) and not isinstance(f.value, bool):
                out.append(format_value(f.value, f.sem))
            else:
                out.extend(_nested_numbers(f.value))
        return out

    def units(self) -> List[str]:
        return sorted({f.unit for f in self.facts if f.unit})

    def as_payload(self) -> Dict[str, Any]:
        """What the LLM writer and critic receive (never the tool surface names)."""
        return {
            "task_id": self.task_id,
            "domain": self.domain,
            "user_goal": self.natural_user_goal,
            "scenario_entities": self.entities,
            "stated_facts": [{"name": f.role, "means": f.description,
                              "value": f.rendered, "unit": f.unit}
                             for f in self.facts],
            "target": self.target_phrase,
            "answer_type": self.answer_type,
            "requested_style": self.requested_mode,
            "must_not_disclose": self.forbidden_terms,
            # the rule the reader could not otherwise know; stating it is what
            # makes the request answerable, so it is required, not optional
            "rules_to_state": list(self.specification),
        }


def _humanise(text: str) -> str:
    return re.sub(r"[_.]+", " ", text).strip()


VERDICT_NOUNS = ("verdict", "decision", "check", "outcome", "call", "answer")


#: Blueprint authors sometimes phrase a sink's purpose the way the tool is named
#: ("both conditions hold" next to a ``both_conditions_hold`` surface), which the
#: leakage validator correctly reads as naming the tool. The wording is rephrased
#: rather than the check relaxed.
PARAPHRASE: Tuple[Tuple[str, str], ...] = (
    ("both conditions hold", "both requirements are met"),
    ("all conditions hold", "every requirement is met"),
    ("either condition holds", "at least one requirement is met"),
    ("any condition holds", "at least one requirement is met"),
    ("conditions hold", "requirements are met"),
    ("condition holds", "requirement is met"),
    ("within tolerance", "close enough to the reference"),
    ("in range", "inside the accepted span"),
    ("is at least", "reaches"),
    ("is at most", "stays under"),
)


def _spaced_forbidden(forbidden: Sequence[str]) -> List[str]:
    out = {re.sub(r"[_.]+", " ", t).strip().lower() for t in forbidden}
    return [t for t in out if len(t) >= 5 and " " in t]


def _avoid_forbidden(phrase: str, forbidden: Sequence[str],
                     answer_type: str) -> str:
    spaced = _spaced_forbidden(forbidden)

    def hits(text: str) -> List[str]:
        low = text.lower()
        return [t for t in spaced if re.search(rf"\b{re.escape(t)}\b", low)]

    if not hits(phrase):
        return phrase
    for src, dst in PARAPHRASE:
        phrase = re.sub(re.escape(src), dst, phrase, flags=re.IGNORECASE)
    remaining = hits(phrase)
    if not remaining:
        return phrase
    generic = ("whether the stated requirement is met" if answer_type == "boolean"
               else "the requested figure")
    for term in remaining:
        phrase = re.sub(rf"\b{re.escape(term)}\b", generic, phrase,
                        flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", phrase).strip()


#: A target that says nothing ("the positions in the requested figure") is what a
#: leakage substitution leaves behind when the sink's purpose was itself named
#: after its tool. No wording can rescue it: the query would have no question in
#: it, and both critics rightly reject that. Such a task is dropped instead.
VACUOUS_TARGETS: Tuple[str, ...] = (
    "the figure", "the value", "the result", "the output", "the number",
    "the answer", "the amount", "the outcome",
)
#: only ever produced by the substitution above, so their presence anywhere in
#: the phrase means the meaning has already been lost
PLACEHOLDER_MARKERS: Tuple[str, ...] = (
    "requested figure", "same figure", "stated requirement is met",
)


def target_is_vacuous(phrase: str) -> bool:
    """True when the target names no quantity a user could ask for."""
    text = re.sub(r"\s+", " ", phrase.strip().lower().rstrip("?.")).strip()
    if len(text) < 6 or text in VACUOUS_TARGETS:
        return True
    return any(marker in text for marker in PLACEHOLDER_MARKERS)


def target_phrase(inst: Instance, bp: Blueprint, plan: Plan) -> str:
    """What the user asks for, taken from the sink's own purpose where given.

    Boolean sinks need care: blueprint authors write purposes both as noun phrases
    ("budget verdict") and as clauses ("both conditions must hold"), and gluing
    "whether" onto the second form produced "does both conditions must hold hold
    here?" in the smoke run. Clauses are normalised, noun phrases are kept.
    """
    sink = plan.step(inst.program.sink)
    purpose = (sink.purpose or "").strip().rstrip("?").strip().rstrip(".")
    if not purpose or len(purpose) < 6:
        purpose = _humanise(plan.intent) or bp.target_description
    if inst.answer_type == "boolean":
        return _boolean_target(purpose)
    if purpose.startswith("the "):
        return purpose
    return f"the {purpose}"


def _boolean_target(purpose: str) -> str:
    text = purpose.strip()
    if text.lower().startswith("whether "):
        return text
    if any(noun in text.lower() for noun in VERDICT_NOUNS):
        return text if text.startswith("the ") else f"the {text}"
    clause = re.sub(r"\b(must|should|has to|have to)\s+", "", text)
    clause = re.sub(r"\bhold(s)?\b\s*$", "hold", clause).strip()
    if clause.startswith(("both ", "either ", "all ", "every ", "the ", "a ",
                          "an ", "it ", "they ")):
        return f"whether {clause}"
    return f"whether {clause} is the case"


def _currency(inst: Instance, rng: random.Random) -> str:
    for role, hint in inst.role_hints.items():
        if hint == "currency_code":
            return str(inst.role_values[role])
    return rng.choice(("EUR", "GBP", "USD", "CZK", "PLN"))


def _entities(bp: Blueprint, rng: random.Random) -> Dict[str, str]:
    flavour = FAMILY_FLAVOUR.get(bp.entity_family, ())
    ents = {
        "org": rng.choice(ORGS),
        "site": rng.choice(SITES),
        "person": rng.choice(PEOPLE),
        "role": rng.choice(ROLES_PEOPLE),
        "project": rng.choice(PROJECTS),
        "period": rng.choice(PERIODS),
    }
    if flavour:
        ents["context"] = rng.choice(flavour)
    return ents


def forbidden_terms(inst: Instance) -> List[str]:
    ops = build_ops()
    out: List[str] = []
    for nd in inst.program.nodes:
        op = ops[nd.op]
        out.append(nd.node_id)
        out.append(op.capability)
        for surf in op.surfaces:
            out.append(surf.name)
    return sorted(set(out))


def build_contract(inst: Instance, bp: Blueprint, plan: Plan, *, mode: str,
                   task_id: str, seed: int) -> QueryContract:
    rng = random.Random(f"contract:{seed}:{task_id}")
    currency = _currency(inst, rng)
    facts: List[Fact] = []
    for role in plan.roles:
        value = inst.role_values[role.name]
        sem = role.sem
        facts.append(Fact(
            role=role.name, value=value, sem=sem,
            description=role.description or _humanise(role.name),
            rendered=with_unit(value, sem, currency),
            unit=unit_of(sem, currency)))
    hidden = [{"node_id": s.node_id, "capability": s.capability,
               "purpose": s.purpose or ""} for s in plan.steps]
    forbidden = forbidden_terms(inst)
    target = _avoid_forbidden(target_phrase(inst, bp, plan), forbidden,
                              inst.answer_type)
    if target_is_vacuous(target):
        # the sink's own purpose was named after its tool; the blueprint's
        # target description is the same quantity said in domain words
        fallback = (bp.target_description or "").strip().rstrip("?.")
        if fallback and not fallback.startswith("the "):
            fallback = f"the {fallback}"
        candidate = _avoid_forbidden(fallback, forbidden, inst.answer_type)
        if not target_is_vacuous(candidate):
            target = candidate
    rule = determinability.classify(
        inst.program, plan, inst.role_values, target_phrase=target,
        render=lambda v: with_unit(v, st.GENERIC, currency))
    level = (determinability.NOT_STATABLE if target_is_vacuous(target)
             else rule.level)
    return QueryContract(
        task_id=task_id, workflow_id=bp.workflow_id, plan_id=plan.plan_id,
        domain=bp.domain, natural_user_goal=bp.natural_user_goal,
        target_phrase=target,
        answer_type=inst.answer_type, requested_mode=mode, currency=currency,
        entities=_entities(bp, rng), facts=facts, hidden_plan=hidden,
        forbidden_terms=forbidden, call_count=inst.call_count,
        seed=seed, specification=rule.specification,
        determinability=level)


# ── deterministic renderers ──────────────────────────────────────────────
#: opening frames. Each is a distinct syntactic skeleton once numbers, entities
#: and domain nouns are normalised away, which is what the diversity gate counts.
FRAMES: Tuple[Tuple[str, str], ...] = (
    ("handover", "{person} left me the numbers for {project} at {site}."),
    ("callback", "{role} rang about {context_or_project}."),
    ("ledger", "Going through {org}'s figures for {period}."),
    ("ticket", "New ticket from {site}, opened {period}."),
    ("standup", "We picked this up at the {period} stand-up at {site}."),
    ("mail", "{person} from {org} wrote in about {project}."),
    ("audit", "{role} is reviewing {context_or_project} before sign-off."),
    ("handoff", "Taking over {project} from {person} today."),
    ("floor", "Quick one from {site} while {project} is running."),
    ("quote", "{org} sent through their side of {project}."),
    ("review", "Sitting down with {role} over {context_or_project}."),
    ("log", "The {period} log at {site} has these entries."),
    ("prep", "Before I answer {person}, I want this straight."),
    ("meeting", "{org} wants an answer in the {period} review."),
    ("phone", "Still on the phone with {role} about {project}."),
    ("note", "A note came down from {site} regarding {context_or_project}."),
    ("dept", "Our side of {project} is being questioned by {role}."),
    ("shift", "{person} handed over {context_or_project} at the end of the shift."),
    ("client", "A client of {org} queried {project} {period}."),
    ("desk", "This landed on my desk at {site} {period}."),
)

#: how the facts are laid out. ``{lines}`` is a joined fact rendering.
FACT_PATTERNS: Tuple[Tuple[str, str], ...] = (
    ("inline_and", "{joined_and}."),
    ("inline_semi", "{joined_semi}."),
    ("recorded", "What is recorded: {joined_semi}."),
    ("we_have", "We have {joined_and}."),
    ("stands_at", "{first_clause}, while {rest_and}."),
    ("known", "Known so far: {joined_semi}."),
    ("per_line", "{bullets}"),
    ("came_with", "It came with {joined_and}."),
    ("figures", "The figures are {joined_and}."),
    ("says", "It says {joined_semi}."),
    ("noted", "{first_clause}. Also noted: {rest_and}."),
    ("attached", "Attached to it: {joined_semi}."),
)

QUESTION_FORMS: Tuple[Tuple[str, str], ...] = (
    ("what_is", "What is {target}?"),
    ("could_you", "Could you work out {target}?"),
    ("need", "I need {target} before I reply."),
    ("tell_me", "Tell me {target}, please."),
    ("asking", "{role_cap} is asking for {target}."),
    ("give", "Give me {target} so I can close this off."),
    ("figure", "Can you figure out {target}?"),
    ("wondering", "I am wondering about {target}."),
    ("chase", "They will chase me for {target} today."),
    ("confirm", "Please confirm {target}."),
    ("how_much", "How much is {target}?"),
    ("end_up", "What does {target_bare} come to?"),
    ("work_back", "Work {target} out for me, if you can."),
    ("after_all", "So what is {target} after all this?"),
    ("send", "I have to send {target} over shortly."),
    ("stuck", "I am stuck on {target}."),
    ("double", "Double-check {target} for me."),
    ("landing", "Where does {target_bare} land?"),
    ("worth", "It would help to know {target}."),
    ("report", "{role_cap} expects {target} in writing."),
    ("quick", "Quickest question: {target}?"),
    ("settle", "Let us settle {target} first."),
    ("what_exactly", "What exactly is {target}?"),
    ("sum_up", "Sum up {target} for me."),
    ("before", "I cannot reply before I have {target}."),
    ("please_give", "Please give me {target}."),
    ("owed", "They are owed {target} in the reply."),
    ("run", "Run {target} for me."),
)

BOOL_FORMS: Tuple[Tuple[str, str], ...] = (
    ("yesno", "I need a yes or no on {target}."),
    ("does_hold", "Does {target_bare} hold here?"),
    ("check", "Can you check {target}?"),
    ("confirm_b", "Please confirm {target}."),
    ("decide", "Decide {target} for me."),
    ("tell_b", "Tell me {target}."),
    ("is_that_so", "Is {target_bare} the case here?"),
    ("green", "Can I give this a green light, that is, {target}?"),
    ("worried", "I am worried about {target}."),
    ("settle_b", "Let us settle {target}."),
    ("stand", "Where do we stand on {target}?"),
    ("hold_up", "Does {target_bare} still hold?"),
    ("sign", "Can I sign this off, given {target}?"),
    ("ask_b", "They are asking {target}."),
    ("verify", "Verify {target} for me."),
    ("true_b", "Is it true that {target_bare}?"),
    ("either", "Yes or no: {target_bare}?"),
    ("need_b", "I need to know {target}."),
)


def _join_and(parts: Sequence[str]) -> str:
    parts = [p for p in parts if p]
    if len(parts) <= 1:
        return parts[0] if parts else ""
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def _fact_clause(fact: Fact, style: int) -> str:
    desc = fact.description
    if style == 0:
        return f"{desc} is {fact.rendered}"
    if style == 1:
        return f"{fact.rendered} for {desc}"
    if style == 2:
        return f"{desc} comes to {fact.rendered}"
    return f"{desc} at {fact.rendered}"


def _fact_block(contract: QueryContract, pattern: str, rng: random.Random) -> str:
    facts = list(contract.facts)
    if rng.random() < 0.5:
        rng.shuffle(facts)
    styles = [rng.randrange(4) for _ in facts]
    clauses = [_fact_clause(f, s) for f, s in zip(facts, styles)]
    joined_and = _join_and(clauses)
    joined_semi = "; ".join(clauses)
    bullets = " ".join(f"{f.description.capitalize()} {f.rendered}." for f in facts)
    first = clauses[0] if clauses else ""
    rest = _join_and(clauses[1:]) if len(clauses) > 1 else ""
    text = pattern.format(joined_and=joined_and, joined_semi=joined_semi,
                          bullets=bullets, first_clause=first,
                          rest_and=rest or joined_and)
    return text.replace("  ", " ").strip()


def _frame(contract: QueryContract, template: str) -> str:
    ents = contract.entities
    context_or_project = ents.get("context") or ents["project"]
    return template.format(context_or_project=context_or_project, **ents)


def _question(contract: QueryContract, rng: random.Random) -> Tuple[str, str]:
    target = contract.target_phrase
    bare = re.sub(r"^(the|whether)\s+", "", target)
    # "does <x> hold hold here?" -- the form supplies the verb, so drop a trailing one
    bare = re.sub(r"\s+(hold|holds|is the case)$", "", bare)
    forms = BOOL_FORMS if contract.answer_type == "boolean" else QUESTION_FORMS
    key, template = forms[rng.randrange(len(forms))]
    text = template.format(target=target, target_bare=bare,
                           role_cap=contract.entities["role"].capitalize())
    return key, text[0].upper() + text[1:]


#: capability keyword -> how a person would name that operation.
#:
#: Two constraints pull against each other here. The phrase must not be the tool
#: identifier, and it must still *name the operation* -- otherwise a query rendered
#: for ``OPERATION_EXPLICIT_GRAPH_IMPLICIT`` reads as fully implicit and the
#: independent classifier rightly says so, leaving that mode unreachable. Each
#: phrase therefore shares a word stem with its capability ("a multiplication" for
#: ``multiply``) without ever reproducing the tool name.
OPERATION_PHRASE: Tuple[Tuple[str, str], ...] = (
    ("apply_tax", "a taxable uplift"),
    ("decrease_by_percent", "a percentage decrease"),
    ("increase_by_percent", "a percentage increase"),
    ("share_percent", "a percentage share"),
    ("percent_of", "a percentage of the amount"),
    ("ratio_of", "a ratio between two amounts"),
    ("reduce_sum", "a summation over the entries"),
    ("reduce_max", "a maximum over the entries"),
    ("reduce_min", "a minimum over the entries"),
    ("reduce_mean", "a mean over the entries"),
    ("map_percent", "a percentage mapping over every entry"),
    ("abs_difference", "an absolute difference"),
    ("multiply", "a multiplication"),
    ("divide", "a division"),
    ("subtract", "a subtraction"),
    ("add", "an addition"),
    ("three_bands", "a banded classification"),
    ("ratio_band", "a banded ratio classification"),
    ("at_least", "a comparison against a limit"),
    ("less_than", "a comparison against a limit"),
    ("greater", "a comparison against a limit"),
    ("equal", "an equality comparison"),
    ("concat", "a concatenation"),
    ("normalize", "a normalisation of the text"),
    ("currency", "a currency formatting step"),
    ("round", "a rounding step"),
    ("join", "a joining step"),
    ("split", "a splitting step"),
    ("extract", "an extraction"),
    ("lookup", "a lookup"),
    ("count", "a count"),
    ("convert", "a conversion"),
    ("difference", "a difference"),
    ("and", "a conjunction of conditions"),
    ("or", "a disjunction of conditions"),
    ("not", "a negation"),
)

FAMILY_PHRASE: Dict[str, str] = {
    "arithmetic": "an arithmetic step", "rates": "a rate calculation",
    "comparison": "a comparison", "statistics": "a statistical step",
    "list": "a list operation", "dictionary": "a dictionary lookup",
    "record": "a record read", "string": "a string operation",
    "path": "a path operation", "url": "a url breakdown",
    "date": "a date calculation", "duration": "a duration calculation",
    "boolean": "a boolean combination", "decision": "a decision rule",
    "classification": "a classification step", "format": "a formatting step",
    "validation": "a validation check", "unit": "a unit conversion",
    "geometry": "a geometric calculation", "lookup": "a lookup",
}


def operation_phrase(capability: str) -> str:
    tail = capability.split(".", 1)[-1]
    for key, phrase in OPERATION_PHRASE:
        if key in tail:
            return phrase
    return FAMILY_PHRASE.get(capability.split(".")[0], "a calculation")


def _operation_hint(contract: QueryContract, rng: random.Random,
                    ordered: bool) -> str:
    """Names the *kinds* of operation without echoing a tool identifier."""
    seen: List[str] = []
    for step in contract.hidden_plan:
        phrase = operation_phrase(step["capability"])
        if phrase not in seen:
            seen.append(phrase)
    if not ordered:
        rng.shuffle(seen)
        openers = ("It involves ", "There is ", "Expect ")
        return openers[rng.randrange(len(openers))] + _join_and(seen) + "."
    return "Work through " + _join_and(seen) + "."


def _stage_hint(contract: QueryContract, rng: random.Random) -> str:
    families: List[str] = []
    for step in contract.hidden_plan:
        fam = step["capability"].split(".")[0]
        if fam not in families:
            families.append(fam)
    phrase = _join_and([_humanise(f) for f in families[:3]])
    options = (f"Take the {phrase} side of it into account.",
               f"Mind the {phrase} part.",
               f"There is a {phrase} element to this.")
    return options[rng.randrange(len(options))]


def tidy(text: str) -> str:
    """Collapse the seams left by composing a frame with an entity phrase.

    Period and context entities read as noun phrases ("the last quarter"), and a
    frame that supplies its own article produced "at the the last quarter". Fixing
    it here rather than in every frame keeps the frame list free to be written
    naturally.
    """
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\b(the|a|an)\s+(the|a|an)\b", r"\2", text, flags=re.IGNORECASE)
    text = re.sub(r"\b([A-Za-z]{2,})\s+\1\b", r"\1", text)
    text = re.sub(r"\s+([,.;:?])", r"\1", text)
    # only a capital letter may follow punctuation without a space: "files.example"
    # inside a URL and "co:8080" must survive untouched
    return re.sub(r"([,.;:?])([A-Z])", r"\1 \2", text).strip()


def render_deterministic(contract: QueryContract, mode: str, *,
                         seed: int = 0) -> Dict[str, Any]:
    """Compose a query for ``mode``; returns the text plus its template identity."""
    rng = random.Random(f"render:{seed}:{contract.task_id}:{mode}")
    frame_key, frame_tpl = FRAMES[rng.randrange(len(FRAMES))]
    fact_key, fact_tpl = FACT_PATTERNS[rng.randrange(len(FACT_PATTERNS))]
    q_key, question = _question(contract, rng)
    parts = [_frame(contract, frame_tpl), _fact_block(contract, fact_tpl, rng)]
    extras: List[str] = [_rule_sentence(contract, rng)]
    if mode == "GRAPH_EXPLICIT":
        extras.append(_graph_disclosure(contract))
    elif mode == "OPERATION_EXPLICIT_GRAPH_IMPLICIT":
        extras.append(_operation_hint(contract, rng, ordered=False))
    elif mode == "SEMI_IMPLICIT":
        extras.append(_stage_hint(contract, rng))
    parts.extend(extras)
    parts.append(question)
    text = tidy(" ".join(p for p in parts if p))
    return {
        "query": text,
        "renderer": "deterministic",
        "renderer_family": f"{frame_key}|{fact_key}|{q_key}",
        "template_id": f"det.{mode.lower()}.{frame_key}.{fact_key}.{q_key}",
        "requested_mode": mode,
    }


#: how a user introduces their own convention; drawn per task so the rule sentence
#: does not become a template of its own
RULE_OPENERS = ("Round here, {rule}.", "The rule we work to is that {rule}.",
                "For what it is worth, {rule}.", "We take it that {rule}.",
                "House rule: {rule}.", "As we do it, {rule}.")


def _rule_sentence(contract: QueryContract, rng: random.Random) -> str:
    """State the conventions the reader could not otherwise know (spec 17).

    Empty for a self-evident task, which is most of them.
    """
    if not contract.specification:
        return ""
    joined = _join_and([r.rstrip(".") for r in contract.specification])
    opener = RULE_OPENERS[rng.randrange(len(RULE_OPENERS))]
    return opener.format(rule=joined)


def _graph_disclosure(contract: QueryContract) -> str:
    """Only for GRAPH_EXPLICIT, which is capped at 3 % of the dataset."""
    steps = []
    for i, step in enumerate(contract.hidden_plan, 1):
        steps.append(f"step {i} {operation_phrase(step['capability'])}")
    return "In order: " + ", ".join(steps) + "."


def banned_phrase_hits(text: str) -> List[str]:
    low = text.lower()
    return [p.strip() for p in BANNED_PHRASES if p in low]


def contract_hash(contract: QueryContract) -> str:
    return sha256_obj({
        "workflow": contract.workflow_id, "plan": contract.plan_id,
        "facts": [(f.role, f.rendered) for f in contract.facts],
        "target": contract.target_phrase, "mode": contract.requested_mode,
    })
