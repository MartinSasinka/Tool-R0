"""Hard deterministic query validation, independent mode classification, fingerprints.

Every check here is a *gate*, not a score: a query that fails one is rewritten or
dropped. The three groups matter for different Pilot4.2 defects.

* **Preservation** (facts, numbers, entities, units, target, no new conditions,
  no answer leak) catches a writer that quietly changed the problem.
* **Leakage** (tool names, variable labels, operations, graph edges, call count,
  reference sources) is measured *and then used to classify the mode from the
  text itself*. The requested mode is never trusted; selection quotas run on the
  classified mode, which is what Pilot4.2 got wrong.
* **Diversity** fingerprints are three-layered on purpose: exact text, a lexical
  skeleton with all content words removed, and an intent family. Exact
  uniqueness alone let one template dominate a whole pilot.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

NUM_RE = re.compile(r"-?\d+(?:[.,]\d+)?")
VAR_LEAK_RE = re.compile(r"\$[A-Za-z_][A-Za-z0-9_]*\$|\bvar\d+\b|\boutput_\d+\b"
                         r"|\bn[1-9]\d?\b", re.IGNORECASE)
STEP_RE = re.compile(r"\bstep\s*\d+\b|\bstage\s*\d+\b", re.IGNORECASE)
ORDER_CUES = ("then", "after that", "afterwards", "next", "finally", "first",
              "second", "third", "lastly", "subsequently", "once you have",
              "in order:")
REFERENCE_CUES = ("the result of", "that result", "the previous result",
                  "the value you get", "the output of", "the number you get",
                  "use the result", "take the result", "resulting value")
COUNT_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
               "seven": 7, "eight": 8, "nine": 9, "ten": 10}
CALL_COUNT_CUES = ("steps", "step", "calls", "call", "operations", "stages",
                   "tool calls")
#: phrases that add a *constraint* the contract never stated. Bare prepositions
#: ("over", "under") were tried and removed: they occur in ordinary framing
#: ("sitting down with the auditor over the ledger") and produced only noise.
CONDITION_CUES = ("at least", "no more than", "not exceed", "must not", "only if",
                  "unless", "provided that", "as long as", "if and only if",
                  "should not", "must stay", "cap of", "capped at", "limit of",
                  "threshold", "minimum of", "maximum of", "no lower than",
                  "no higher than", "must remain")

FUNCTION_WORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "that", "this",
    "these", "those", "of", "for", "to", "in", "on", "at", "by", "with", "from",
    "as", "is", "are", "was", "were", "be", "been", "being", "do", "does", "did",
    "can", "could", "should", "would", "will", "shall", "may", "might", "must",
    "have", "has", "had", "i", "we", "you", "they", "he", "she", "it", "me",
    "my", "our", "your", "their", "his", "her", "its", "what", "which", "who",
    "whom", "whose", "when", "where", "why", "how", "whether", "not", "no",
    "yes", "there", "here", "so", "up", "out", "about", "into", "over", "after",
    "before", "again", "just", "still", "only", "also", "each", "every", "any",
    "some", "all", "both", "how much", "how many", "please", "need", "needs",
    "want", "wants", "give", "tell", "confirm", "check", "work", "figure",
    "come", "comes", "left", "sent", "rang", "wrote", "asking", "chase",
    "wondering", "decide", "let", "know", "get", "got", "make", "made", "take",
    "takes", "put", "keep", "stay", "stays", "hold", "holds", "run", "running",
    "going", "sitting", "taking", "quick", "new", "last", "next", "per",
    "between", "against", "without", "within", "across", "off", "down",
}

UNIT_TOKENS = ("%", "eur", "gbp", "usd", "czk", "pln", "hours", "hour", "minutes",
               "minute", "seconds", "second", "days", "day", "km", "m", "kg", "g",
               "l", "ml", "°c", "°f", "bytes", "points", "m²")


# ── helpers ──────────────────────────────────────────────────────────────
def numbers_in(text: str) -> List[str]:
    return [m.group(0).replace(",", ".") for m in NUM_RE.finditer(text)]


def _norm_num(text: str) -> str:
    try:
        value = float(str(text).replace(",", "."))
    except ValueError:
        return str(text).strip()
    if value == int(value):
        return str(int(value))
    return f"{value:.4f}".rstrip("0").rstrip(".")


def tokens(text: str) -> List[str]:
    return re.findall(r"[a-z°²]+", text.lower())


def content_words(text: str) -> List[str]:
    return [t for t in tokens(text) if t not in FUNCTION_WORDS and len(t) > 2]


# ── preservation checks ──────────────────────────────────────────────────
def _strip_entities(query: str, entities: Sequence[str]) -> str:
    """Remove entity mentions before number checks.

    Scenario names legitimately contain digits ("the Q3 refit"), and counting the
    3 as an invented fact would reject a perfectly good query.
    """
    text = query
    for ent in sorted((e for e in entities if e), key=len, reverse=True):
        text = re.sub(re.escape(str(ent)), " ", text, flags=re.IGNORECASE)
    return text


#: A small whole number is often written out, and a positional fact is normally
#: spoken as an ordinal: "the second team" states the value 2 exactly as faithfully
#: as the digit does. Rejecting those queries cost real samples in the smoke run
#: and taught the writer to write worse English.
CARDINAL_WORDS: Dict[str, str] = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
    "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten",
    "11": "eleven", "12": "twelve",
}
ORDINAL_WORDS: Dict[str, str] = {
    "1": "first", "2": "second", "3": "third", "4": "fourth", "5": "fifth",
    "6": "sixth", "7": "seventh", "8": "eighth", "9": "ninth", "10": "tenth",
    "11": "eleventh", "12": "twelfth",
}


def _spelled_out(number: str, query_words: Set[str]) -> bool:
    return (CARDINAL_WORDS.get(number) in query_words
            or ORDINAL_WORDS.get(number) in query_words)


def check_facts(query: str, contract: Dict[str, Any]) -> Dict[str, Any]:
    """Every stated fact must survive verbatim; nothing numeric may be invented."""
    q_nums = {_norm_num(n) for n in
              numbers_in(_strip_entities(query, contract["entities"]))}
    expected = {_norm_num(n) for n in contract["expected_numbers"]}
    words = set(tokens(query))
    missing = sorted(n for n in expected - q_nums if not _spelled_out(n, words))
    extra = sorted(q_nums - expected)
    allowed_extra: Set[str] = set()
    if contract.get("mode") == "GRAPH_EXPLICIT":
        allowed_extra = {str(i) for i in range(1, contract["call_count"] + 1)}
    extra = [x for x in extra if x not in allowed_extra]
    # a string fact may carry deliberate padding ("  office chair  ") because the
    # task is about trimming it; the query states it verbatim but any renderer or
    # writer collapses runs of spaces, so the comparison is whitespace-insensitive
    low = re.sub(r"\s+", " ", query.lower())
    missing_strings = [s for s in contract["expected_strings"]
                       if s and re.sub(r"\s+", " ", str(s).lower().strip())
                       not in low]
    return {
        "passed": not missing and not extra and not missing_strings,
        "missing_numbers": missing, "extra_numbers": extra,
        "missing_strings": missing_strings,
    }


def check_units(query: str, contract: Dict[str, Any]) -> Dict[str, Any]:
    """Required units must be present; no *other* unit may be attached to a number.

    A unit only counts as used when it follows a number: "the second supplier" is
    not a duration, and treating the word alone as a unit token rejected honest
    queries in the smoke run.

    A unit named by a fact's own description is permitted even when the value
    carries no unit itself: a list role described as "kilometres each leg runs" has
    no scalar unit to attach, and a query that writes "2.49 km" is preserving that
    description rather than inventing a dimension.
    """
    low = query.lower()
    required = [u.lower() for u in contract["expected_units"]]
    described = _units_in_vocabulary(contract)
    allowed = set(required) | described
    missing = [u for u in required if u not in low]
    foreign = [u for u in UNIT_TOKENS
               if u not in allowed and
               re.search(rf"\d\s*{re.escape(u)}\b", low)]
    return {"passed": not missing and not foreign,
            "missing_units": missing, "foreign_units": foreign,
            "allowed_by_description": sorted(described - set(required))}


#: unit words as they appear in prose, mapped to the unit token they license
UNIT_SYNONYMS: Dict[str, Tuple[str, ...]] = {
    "km": ("kilometre", "kilometres", "kilometer", "kilometers", "km"),
    "m": ("metre", "metres", "meter", "meters"),
    "kg": ("kilogram", "kilograms", "kilo", "kilos", "kg"),
    "g": ("gram", "grams"),
    "l": ("litre", "litres", "liter", "liters"),
    "ml": ("millilitre", "millilitres", "milliliter", "milliliters", "ml"),
    "hours": ("hour", "hours"),
    "minutes": ("minute", "minutes"),
    "seconds": ("second", "seconds"),
    "days": ("day", "days"),
    "%": ("percent", "percentage", "per cent"),
    "bytes": ("byte", "bytes"),
    "points": ("point", "points"),
}


def _units_in_vocabulary(contract: Dict[str, Any]) -> Set[str]:
    text = " ".join(str(v) for v in contract.get("domain_vocabulary") or ()).lower()
    out: Set[str] = set()
    for unit, words in UNIT_SYNONYMS.items():
        if any(re.search(rf"\b{re.escape(w)}\b", text) for w in words):
            out.add(unit)
    return out


def check_target(query: str, contract: Dict[str, Any]) -> Dict[str, Any]:
    target_words = set(content_words(contract["target_phrase"]))
    q_words = set(content_words(query))
    if not target_words:
        return {"passed": False, "overlap": 0.0, "reason": "empty target"}
    overlap = len(target_words & q_words) / len(target_words)
    ok = overlap >= 0.5 or len(target_words & q_words) >= 2
    return {"passed": bool(ok), "overlap": round(overlap, 3)}


#: Only the scenario-grounded mode is *supposed* to name people and places. The
#: other modes are defined by the absence of scenario colour, so requiring an
#: entity there contradicted the mode instruction the writer was given, and
#: rejected queries that were doing exactly what the mode asked for.
ENTITY_REQUIRED_MODES = ("DOMAIN_GROUNDED_IMPLICIT",)


def check_entities(query: str, contract: Dict[str, Any]) -> Dict[str, Any]:
    low = query.lower()
    present = [e for e in contract["entities"] if e and e.lower() in low]
    required = contract.get("mode") in ENTITY_REQUIRED_MODES
    return {"passed": bool(present) or not required,
            "required": required, "entities_present": present}


def check_answer_leak(query: str, contract: Dict[str, Any]) -> Dict[str, Any]:
    answer = contract.get("answer_rendered") or ""
    if not answer:
        return {"passed": True, "leaked": False}
    # A lookup or selection task answers *with* one of the stated facts, so the
    # value appearing in the query is not a leak -- it is the question.
    stated = {_norm_num(n) for n in contract["expected_numbers"]}
    stated_text = {re.sub(r"\s+", " ", str(s).strip().lower())
                   for s in contract.get("expected_strings", ())}
    if _norm_num(answer) in stated or \
            re.sub(r"\s+", " ", answer.strip().lower()) in stated_text:
        return {"passed": True, "leaked": False, "note": "answer equals a stated fact"}
    if _is_number(answer):
        leaked = _norm_num(answer) in {_norm_num(n) for n in numbers_in(query)}
    else:
        leaked = len(answer) > 3 and answer.lower() in query.lower()
    return {"passed": not leaked, "leaked": bool(leaked)}


def _is_number(text: str) -> bool:
    try:
        float(str(text).replace(",", "."))
        return True
    except ValueError:
        return False


def strip_vocabulary(query: str, vocabulary: Sequence[str]) -> str:
    """Remove the task's own domain wording before measuring leakage.

    Every leakage signal below is about wording the query did not need. A workflow
    whose fact is called "unit price of the first supplier" forces the words
    "first" and "limit" into the text, and counting those as an ordered step list
    or as an invented condition -- which is what the first smoke run did -- makes
    the metric meaningless. Stripping the vocabulary first is the single rule that
    fixes operation coverage, order cues and condition cues at once, and the
    vocabulary is exported with the task so the audit strips exactly the same words.
    """
    text = query
    for phrase in sorted((str(p) for p in vocabulary if p), key=len, reverse=True):
        if len(phrase) < 4:
            continue
        text = re.sub(re.escape(phrase), " ", text, flags=re.IGNORECASE)
    return text


def check_new_conditions(query: str, contract: Dict[str, Any]) -> Dict[str, Any]:
    residual = strip_vocabulary(query, contract.get("domain_vocabulary", ())).lower()
    hits = [c for c in CONDITION_CUES if c in residual]
    budget = int(contract.get("predicate_steps", 0))
    ok = len(hits) <= max(budget, 0) + (1 if budget else 0)
    return {"passed": bool(ok), "condition_cues": hits,
            "predicate_budget": budget}


def check_tool_leak(query: str, contract: Dict[str, Any]) -> Dict[str, Any]:
    """Tool *identifiers* are always forbidden; naming an operation is mode-dependent.

    ``apply_tax`` or ``list_reduce_sum`` in the text means the writer saw the tool
    surface, so it is rejected everywhere. The de-underscored phrase ("apply tax")
    is only rejected for the modes that are supposed to hide the operations --
    otherwise ``OPERATION_EXPLICIT_GRAPH_IMPLICIT``, whose whole point is to name
    what has to be done, could never pass.
    """
    low = query.lower()
    explicit_ok = contract.get("mode") in ("GRAPH_EXPLICIT",
                                           "OPERATION_EXPLICIT_GRAPH_IMPLICIT")
    identifiers: List[str] = []
    phrases: List[str] = []
    for term in contract["forbidden_terms"]:
        t = term.lower()
        if len(t) < 5:
            continue
        if "_" in t and t in low:
            identifiers.append(term)
            continue
        spaced = t.replace("_", " ")
        if spaced != t and spaced in low:
            phrases.append(term)
        elif spaced == t and re.search(rf"\b{re.escape(t)}\b", low):
            phrases.append(term)
    hits = identifiers + ([] if explicit_ok else phrases)
    return {"passed": not hits,
            "leaked_identifiers": sorted(set(identifiers))[:8],
            "leaked_phrases": sorted(set(phrases))[:8],
            "phrases_allowed_in_mode": explicit_ok}


def check_var_leak(query: str) -> Dict[str, Any]:
    hits = [m.group(0) for m in VAR_LEAK_RE.finditer(query)]
    return {"passed": not hits, "hits": hits[:8]}


# ── leakage measurement ──────────────────────────────────────────────────
def _stems(words: Iterable[str]) -> Set[str]:
    """Crude 5-character stems so "a multiplication" matches ``arithmetic.multiply``."""
    return {w[:5] for w in words if len(w) > 3}


def operation_coverage(query: str, capabilities: Sequence[str],
                       domain_vocabulary: Sequence[str] = ()) -> float:
    """Share of gold operations whose own words appear in the query.

    Words the query *has* to contain are excluded: a tax workflow must say "tax
    rate" to state its facts, so sharing that word with ``rates.apply_tax`` is not
    disclosure. Matching is by stem, because a query naming an operation says
    "a multiplication", not "multiply", and the metric has to see that.
    """
    if not capabilities:
        return 0.0
    residual = strip_vocabulary(query, domain_vocabulary)
    q = _stems(tokens(residual))
    hit = 0
    for cap in capabilities:
        words = {w for w in re.split(r"[._]", cap.lower()) if len(w) > 3}
        if _stems(words) & q:
            hit += 1
    return round(hit / len(capabilities), 4)


def graph_edge_coverage(query: str, call_count: int,
                        domain_vocabulary: Sequence[str] = ()) -> float:
    """How much of the dependency order the text hands over."""
    residual = strip_vocabulary(query, domain_vocabulary).lower()
    markers = sum(residual.count(c) for c in ORDER_CUES)
    markers += len(STEP_RE.findall(residual))
    markers += sum(residual.count(c) for c in REFERENCE_CUES)
    denom = max(call_count - 1, 1)
    return round(min(markers / denom, 1.0), 4)


def stage_label_count(query: str) -> int:
    return len(STEP_RE.findall(query))


def call_count_disclosed(query: str, call_count: int) -> bool:
    low = query.lower()
    for m in re.finditer(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
                         r"(steps?|calls?|operations?|stages?|tool calls?)\b", low):
        word = m.group(1)
        value = COUNT_WORDS.get(word, None)
        if value is None:
            try:
                value = int(word)
            except ValueError:
                continue
        if value == call_count:
            return True
    return bool(STEP_RE.findall(query)) and len(STEP_RE.findall(query)) >= call_count


def reference_source_leak(query: str) -> List[str]:
    low = query.lower()
    return [c for c in REFERENCE_CUES if c in low]


def classify_mode(query: str, capabilities: Sequence[str], call_count: int,
                  entities: Sequence[str],
                  domain_vocabulary: Sequence[str] = ()) -> Dict[str, Any]:
    """Classify from the text alone. The requested label is never an input."""
    op_cov = operation_coverage(query, capabilities, domain_vocabulary)
    graph_cov = graph_edge_coverage(query, call_count, domain_vocabulary)
    stages = stage_label_count(query)
    refs = reference_source_leak(strip_vocabulary(query, domain_vocabulary))
    grounded = any(e and e.lower() in query.lower() for e in entities)
    if graph_cov >= 0.5 or stages >= max(call_count - 1, 2):
        mode = "GRAPH_EXPLICIT"
    elif op_cov >= 0.5:
        mode = "OPERATION_EXPLICIT_GRAPH_IMPLICIT"
    elif op_cov >= 0.2 or graph_cov >= 0.25 or refs:
        mode = "SEMI_IMPLICIT"
    elif grounded:
        mode = "DOMAIN_GROUNDED_IMPLICIT"
    else:
        mode = "GOAL_BASED_IMPLICIT"
    return {
        "actual_query_mode": mode,
        "operation_leakage": op_cov,
        "graph_edge_coverage": graph_cov,
        "sequence_leakage": round(min(graph_cov + 0.25 * len(refs), 1.0), 4),
        "call_count_leakage": call_count_disclosed(query, call_count),
        "stage_label_count": stages,
        "reference_source_leakage": refs[:4],
        "scenario_grounded": bool(grounded),
    }


MODE_LIMITS = {
    "DOMAIN_GROUNDED_IMPLICIT": {"operation_leakage": 0.20, "graph_edge_coverage": 0.20},
    "GOAL_BASED_IMPLICIT": {"operation_leakage": 0.25, "graph_edge_coverage": 0.20},
    "SEMI_IMPLICIT": {"operation_leakage": 0.55, "graph_edge_coverage": 0.45},
    "OPERATION_EXPLICIT_GRAPH_IMPLICIT": {"operation_leakage": 1.0,
                                          "graph_edge_coverage": 0.50},
    "GRAPH_EXPLICIT": {"operation_leakage": 1.0, "graph_edge_coverage": 1.0},
}


def check_mode_limits(cls: Dict[str, Any], mode: str) -> Dict[str, Any]:
    limits = MODE_LIMITS[mode]
    fails = []
    for key, cap in limits.items():
        if cls[key] > cap:
            fails.append(f"{key}={cls[key]} > {cap}")
    if mode in ("DOMAIN_GROUNDED_IMPLICIT", "GOAL_BASED_IMPLICIT"):
        if cls["stage_label_count"] != 0:
            fails.append("stage labels present in an implicit mode")
        if cls["call_count_leakage"]:
            fails.append("call count disclosed in an implicit mode")
    return {"passed": not fails, "violations": fails}


#: share of a rule's *distinctive* words that must survive into the query. The
#: writer states the rule in its own words, so an exact-substring check would
#: reject good paraphrases; below this the rule is simply not there.
RULE_COVERAGE = 0.6


def check_specification(query: str, contract: Dict[str, Any]) -> Dict[str, Any]:
    """Every rule the query has to state must actually be stated.

    A task with a computed criterion or a composite answer shape is unanswerable
    without its rule, and a query that omits it is not a hard question but a
    broken one -- which is exactly what the first critic kept rejecting.

    Coverage counts only the words the rule *adds*. A rule about the host, port
    and path of a link repeats the words of the facts it is built from, so
    counting those let a query that never mentioned the rule pass on the strength
    of restating its own inputs.
    """
    rules = contract.get("specification") or ()
    words = set(tokens(query))
    given: Set[str] = set()
    for text in (contract.get("target_phrase", ""),
                 *(contract.get("domain_vocabulary") or ())):
        if text not in rules:
            given.update(content_words(str(text)))
    missing: List[Dict[str, Any]] = []
    for rule in rules:
        needed = [w for w in content_words(rule) if w not in given]
        if not needed:
            # the rule adds no word of its own: nothing distinctive to look for
            needed = content_words(rule)
        if not needed:
            continue
        covered = sum(1 for w in needed if w in words) / len(needed)
        if covered < RULE_COVERAGE:
            missing.append({"rule": rule, "coverage": round(covered, 3),
                            "distinctive_words": needed})
    return {"passed": not missing, "rules": len(rules), "missing_rules": missing}


# ── language sanity ──────────────────────────────────────────────────────
def check_language(query: str) -> Dict[str, Any]:
    issues: List[str] = []
    text = query.strip()
    if len(text) < 40:
        issues.append("too short")
    if len(text) > 1200:
        issues.append("too long")
    if "{" in text or "}" in text:
        issues.append("unfilled placeholder")
    if "  " in text:
        issues.append("double space")
    if not re.search(r"[.?!]$", text):
        issues.append("no final punctuation")
    if re.search(r"\b([a-z]{2,})\s+\1\b", text.lower()):
        issues.append("doubled word")
    letters = sum(ch.isalpha() for ch in text)
    # a query that states a list of readings is legitimately digit-heavy
    if letters < 0.30 * len(text):
        issues.append("low letter ratio")
    if re.search(r"[\u0400-\u04FF\u4e00-\u9fff]", text):
        issues.append("non-latin script")
    if not re.search(r"[?.]", text):
        issues.append("no sentence end")
    if text.count("?") > 2:
        issues.append("too many questions")
    return {"passed": not issues, "issues": issues}


# ── fingerprints ─────────────────────────────────────────────────────────
def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def exact_fingerprint(query: str) -> str:
    return _sha(re.sub(r"\s+", " ", query.strip().lower()))


def lexical_skeleton(query: str) -> str:
    """Content words, numbers, units and entities removed; syntax preserved."""
    text = query.lower()
    text = NUM_RE.sub(" # ", text)
    out: List[str] = []
    for token in re.findall(r"[a-z°²#]+|[.,;:?!]", text):
        if token in ".,;:?!":
            out.append(token)
        elif token == "#":
            out.append("#")
        elif token in FUNCTION_WORDS:
            out.append(token)
        else:
            out.append("X")
    collapsed: List[str] = []
    for token in out:
        if token == "X" and collapsed and collapsed[-1] == "X":
            continue
        collapsed.append(token)
    return " ".join(collapsed)


def skeleton_fingerprint(query: str) -> str:
    return _sha(lexical_skeleton(query))


def question_form(query: str) -> str:
    last = re.split(r"(?<=[.?!])\s+", query.strip())[-1].lower()
    if last.startswith(("what", "which", "how", "who", "when", "where", "why")):
        return "wh_question"
    if last.endswith("?") and last.split()[0] in (
            "is", "are", "does", "do", "did", "can", "could", "should", "would",
            "will", "has", "have"):
        return "yes_no_question"
    if last.endswith("?"):
        return "other_question"
    if last.split() and last.split()[0] in ("give", "tell", "confirm", "check",
                                            "decide", "please", "work", "let"):
        return "imperative"
    return "statement"


def intent_fingerprint(query: str) -> str:
    """Coarse family: question form + the function-word spine of the last sentence."""
    sentences = [s for s in re.split(r"(?<=[.?!])\s+", query.strip()) if s]
    last = sentences[-1] if sentences else query
    spine = [t for t in tokens(last) if t in FUNCTION_WORDS][:8]
    bucket = "short" if len(sentences) <= 2 else ("medium" if len(sentences) <= 4
                                                  else "long")
    return _sha(f"{question_form(query)}|{bucket}|{' '.join(spine)}")


def fingerprints(query: str) -> Dict[str, str]:
    return {
        "exact_fingerprint": exact_fingerprint(query),
        "skeleton_fingerprint": skeleton_fingerprint(query),
        "intent_fingerprint": intent_fingerprint(query),
        "question_form": question_form(query),
        "lexical_skeleton": lexical_skeleton(query)[:240],
    }


# ── the whole gate ───────────────────────────────────────────────────────
LAYERS = ("facts", "units", "target", "entities", "answer_leak", "new_conditions",
          "tool_leak", "var_leak", "mode_limits", "specification", "language")


def validate_query(query: str, contract: Dict[str, Any]) -> Dict[str, Any]:
    """Run every deterministic check. ``contract`` is the flat payload dict."""
    cls = classify_mode(query, contract["gold_capabilities"],
                        contract["call_count"], contract["entities"],
                        contract.get("domain_vocabulary", ()))
    layers = {
        "facts": check_facts(query, contract),
        "units": check_units(query, contract),
        "target": check_target(query, contract),
        "entities": check_entities(query, contract),
        "answer_leak": check_answer_leak(query, contract),
        "new_conditions": check_new_conditions(query, contract),
        "tool_leak": check_tool_leak(query, contract),
        "var_leak": check_var_leak(query),
        "mode_limits": check_mode_limits(cls, cls["actual_query_mode"]),
        "specification": check_specification(query, contract),
        "language": check_language(query),
    }
    failed = [k for k, v in layers.items() if not v["passed"]]
    return {
        "passed": not failed,
        "failed_layers": failed,
        "layers": layers,
        "classification": cls,
        "fingerprints": fingerprints(query),
    }


def _mapping_strings(value: Dict[Any, Any]) -> List[str]:
    """The side of a mapping that carries data the user would say out loud.

    A rate card (``{"dispatch": 7.9}``) is keyed by data, so its keys have to
    survive into the query. A record (``{"label": "desk lamp", "site": "west"}``)
    is keyed by its schema: demanding the word "label" asked the writer to recite
    a column name, which is exactly the disclosure the other checks forbid.
    """
    if value and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                     for v in value.values()):
        return [str(k) for k in value]
    return [v for v in value.values() if isinstance(v, str)]


def contract_payload(contract, *, answer: Any, gold_capabilities: Sequence[str],
                     predicate_steps: int) -> Dict[str, Any]:
    """Flatten a :class:`~.queries.QueryContract` for the validators above."""
    from .queries import format_value

    expected_strings: List[str] = []
    for f in contract.facts:
        if isinstance(f.value, str):
            expected_strings.append(f.value)
        elif isinstance(f.value, list):
            for item in f.value:
                if isinstance(item, str):
                    expected_strings.append(item)
                elif isinstance(item, dict):
                    expected_strings.extend(_mapping_strings(item))
        elif isinstance(f.value, dict):
            expected_strings.extend(_mapping_strings(f.value))
    return {
        "mode": contract.requested_mode,
        "call_count": contract.call_count,
        "target_phrase": contract.target_phrase,
        "expected_numbers": contract.numbers(),
        "expected_strings": expected_strings,
        "expected_units": contract.units(),
        "entities": list(contract.entities.values()),
        "forbidden_terms": contract.forbidden_terms,
        "gold_capabilities": list(gold_capabilities),
        "predicate_steps": predicate_steps,
        "answer_rendered": format_value(answer),
        "specification": list(getattr(contract, "specification", ())),
        # the words the query must be allowed to use: they are the task's own
        # domain vocabulary, not disclosure of the program. The rule sentences
        # belong here for the same reason a fact description does -- without them
        # the request cannot be answered, so naming them is not leakage.
        "domain_vocabulary": ([f.description for f in contract.facts]
                              + list(getattr(contract, "specification", ()))
                              + [contract.target_phrase, contract.domain,
                                 contract.natural_user_goal]),
    }


# ── pool-level diversity ─────────────────────────────────────────────────
def diversity_report(queries: Sequence[str]) -> Dict[str, Any]:
    from collections import Counter

    n = len(queries) or 1
    exact = Counter(exact_fingerprint(q) for q in queries)
    skel = Counter(skeleton_fingerprint(q) for q in queries)
    intent = Counter(intent_fingerprint(q) for q in queries)
    top10 = sum(c for _k, c in intent.most_common(10))
    return {
        "n": len(queries),
        "exact_duplicate_rate": round(sum(c - 1 for c in exact.values()) / n, 5),
        "distinct_exact": len(exact),
        "distinct_skeletons": len(skel),
        "distinct_intent_templates": len(intent),
        "max_skeleton_share": round(max(skel.values()) / n, 5) if skel else 0.0,
        "max_intent_share": round(max(intent.values()) / n, 5) if intent else 0.0,
        "top10_intent_share": round(top10 / n, 5),
        "question_form_distribution": dict(Counter(question_form(q)
                                                   for q in queries)),
    }


DIVERSITY_GATES = {
    "exact_duplicate_rate": ("<=", 0.0),
    "max_skeleton_share": ("<=", 0.01),
    "max_intent_share": ("<=", 0.02),
    "top10_intent_share": ("<=", 0.15),
}


def check_diversity(report: Dict[str, Any]) -> Dict[str, Any]:
    fails = []
    for key, (_op, cap) in DIVERSITY_GATES.items():
        if float(report.get(key, 1.0)) > cap:
            fails.append(f"{key}={report.get(key)} > {cap}")
    return {"passed": not fails, "violations": fails}
