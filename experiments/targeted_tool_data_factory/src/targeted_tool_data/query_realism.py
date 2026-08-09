"""Query realism and plan-leak auditing (Phase B).

A synthetic question that spells out the gold program turns tool-use into
transcription. This module measures how much of the plan a question gives
away, using a versioned, hand-auditable operation lexicon — never an LLM and
never a lexicon scraped from the evaluation benchmark.

Three cue strengths per operation:

    exact     the tool/primitive operation word itself ("subtract")
    lexical   an unambiguous paraphrase of the operation ("minus")
    semantic  a goal-level hint that implies it ("how much remains")

Only ``exact`` and ``lexical`` count as leakage; ``semantic`` is what a
realistic user question is expected to contain.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .capability import family_of

SCHEMA_VERSION = "ttdf.query_realism.v1"
LEXICON_VERSION = "ttdf.operation_lexicon.v1"

QUERY_MODES = ["PROCEDURAL_EXPLICIT", "PROCEDURAL_PARTIAL", "SEMI_IMPLICIT",
               "GOAL_BASED_IMPLICIT", "UNCLASSIFIED"]

# ── operation lexicon (hand maintained, versioned) ────────────────────────
# {primitive_id: (exact terms, lexical cues, semantic cues)}
OPERATION_LEXICON: Dict[str, Dict[str, List[str]]] = {}


def _lex(sid: str, exact: Sequence[str], lexical: Sequence[str],
         semantic: Sequence[str] = ()) -> None:
    OPERATION_LEXICON[sid] = {
        "exact": sorted({sid.replace("_", " "), *exact}),
        "lexical": sorted(set(lexical)),
        "semantic": sorted(set(semantic)),
    }


_lex("add", ["add", "sum"], ["plus", "added to", "combine", "total of", "altogether"],
     ["how many in total", "overall amount"])
_lex("subtract", ["subtract"], ["minus", "difference", "decrease by", "take away",
                                "reduce by", "less than"],
     ["how much remains", "how much more", "what is left"])
_lex("multiply", ["multiply"], ["times", "product", "scaled by", "multiplied by"],
     ["how much in all", "per unit total"])
_lex("divide", ["divide"], ["divided by", "quotient", "split into", "per"],
     ["how much each", "share equally"])
_lex("power", ["power", "exponent"], ["raised to", "to the power"], [])
_lex("modulo", ["modulo", "remainder"], ["mod", "left over after dividing"], ["what is left over"])
_lex("floor_divide", ["floor divide"], ["whole times", "how many whole", "fits into"],
     ["how many complete"])
_lex("percent_of", ["percent of"], ["percentage of", "% of"], ["what share of"])
_lex("ratio_of", ["ratio"], ["ratio of", "relative to"], ["how does it compare"])
_lex("increase_by_percent", ["increase by percent"], ["increase by", "raise by",
                                                      "grow by", "markup"],
     ["after the rise", "new higher value"])
_lex("decrease_by_percent", ["decrease by percent"], ["decrease by", "discount",
                                                      "reduce by", "cut by"],
     ["after the discount", "new lower value"])
_lex("abs_difference", ["absolute difference"], ["how far apart", "gap between",
                                                 "distance between"],
     ["how much they differ"])
_lex("average_two", ["average"], ["mean of", "average of"], ["typical value"])
_lex("mean_three", ["average", "mean"], ["mean of", "average of"], ["typical value"])
_lex("mean_values", ["average", "mean"], ["mean of", "average of"], ["typical value"])
_lex("median_values", ["median"], ["middle value"], ["typical value"])
_lex("sum_three", ["add up", "sum"], ["total of", "plus", "altogether"], ["overall total"])
_lex("product_three", ["multiply"], ["product of", "times"], [])
_lex("sum_values", ["sum"], ["total of", "add up", "altogether"], ["overall total"])
_lex("count_values", ["count"], ["how many items", "number of entries"], ["how many"])
_lex("max_values", ["maximum"], ["largest", "highest", "biggest", "peak"], ["best result"])
_lex("min_values", ["minimum"], ["smallest", "lowest", "least"], ["worst result"])
_lex("max_two", ["maximum", "larger"], ["larger of", "greater of", "higher of"], ["the better one"])
_lex("min_two", ["minimum", "smaller"], ["smaller of", "lesser of", "lower of"], ["the worse one"])
_lex("range_spread", ["range", "spread"], ["max minus min", "spread of", "range of"],
     ["how wide the values are"])
_lex("range_three", ["range", "spread"], ["spread between", "largest and smallest"],
     ["how wide the values are"])
_lex("negate", ["negate"], ["opposite sign", "negative of"], [])
_lex("inverse", ["reciprocal", "inverse"], ["one over"], [])
_lex("square", ["square"], ["squared", "to the second power"], [])
_lex("sqrt", ["square root"], ["root of"], ["side length"])
_lex("digit_sum", ["digit sum"], ["sum of the digits"], [])
_lex("ratio_to_percent", ["percentage"], ["as a percentage", "express as percent"], [])
_lex("ceil_value", ["round up"], ["ceiling", "rounded up"], ["next whole"])
_lex("floor_value", ["round down"], ["floor", "rounded down"], ["previous whole"])
_lex("round_to_int", ["round"], ["nearest whole", "rounded to integer"], [])
_lex("round_places", ["round"], ["decimal places", "rounded to"], [])
_lex("round_direction", ["round"], ["rounding mode", "round using"], [])
_lex("clamp", ["clamp"], ["limit between", "cap at", "bound between"], ["keep within limits"])
_lex("is_within_range", ["within range"], ["lies between", "is between", "inside the range"],
     ["is it acceptable"])
_lex("is_greater", ["is greater"], ["greater than", "larger than", "exceeds"],
     ["does it beat"])
_lex("is_divisible_by", ["divisible"], ["divides evenly", "is a multiple of"], [])
_lex("is_non_negative", ["non negative"], ["not below zero", "zero or positive"], ["is it valid"])
_lex("logical_and", ["logical and"], ["both", "and also", "at the same time"],
     ["do both hold"])
_lex("logical_or", ["logical or"], ["either", "at least one"], ["does any hold"])
_lex("logical_not", ["logical not"], ["invert", "negation of", "does not hold"], [])
_lex("scale_list", ["scale"], ["scale every", "multiply each"], [])
_lex("offset_list", ["offset"], ["add to every", "shift every"], [])
_lex("cumulative_sums", ["running totals", "cumulative"], ["running sum"], [])
_lex("sort_values_desc", ["sort"], ["from largest to smallest", "descending order"],
     ["ranked list"])
_lex("filter_above", ["filter"], ["keep the items above", "only those above"],
     ["which ones qualify"])
_lex("top_k_values", ["top"], ["largest items", "top k"], ["the best few"])
_lex("index_of_max", ["index of max", "position"], ["position of the largest"],
     ["which one is best"])
_lex("value_at_position", ["value at position"], ["item at position", "entry at rank"], [])
_lex("append_value", ["append"], ["add to the list", "attach to the list"], [])
_lex("concat_lists", ["concatenate"], ["merge the lists", "join the series"], [])
_lex("join_values", ["join"], ["joined into one string", "separated by"], [])
_lex("concat_texts", ["concatenate"], ["join the texts", "stick together"], [])
_lex("tag_value", ["tag"], ["build an identifier", "prefix with"], ["label it"])
_lex("format_with_unit", ["format with unit"], ["label with the unit", "add the unit"], [])
_lex("format_fixed", ["format"], ["decimal places as text", "formatted to"], [])
_lex("number_to_string", ["convert to text"], ["as text", "stringify"], [])
_lex("parse_number", ["parse"], ["read the number from", "numeric text"], [])
_lex("text_length", ["count the characters"], ["character count", "length of the text"], [])
_lex("text_upper", ["uppercase"], ["upper case", "capitalise"], [])
_lex("join_path_segments", ["join path"], ["path segments", "folder path"], [])
_lex("file_extension", ["extension"], ["file suffix", "after the last dot"], [])
_lex("domain_of_url", ["domain", "host"], ["host of", "domain of"], [])
_lex("celsius_to_fahrenheit", ["convert celsius to fahrenheit"], ["to fahrenheit"], [])
_lex("fahrenheit_to_celsius", ["convert fahrenheit to celsius"], ["to celsius"], [])
_lex("km_to_meters", ["convert kilometers to meters"], ["to meters", "in metres"], [])
_lex("meters_to_km", ["convert meters to kilometers"], ["to kilometers"], [])
_lex("hours_to_minutes", ["convert hours to minutes"], ["in minutes"], [])
_lex("minutes_to_seconds", ["convert minutes to seconds"], ["in seconds"], [])
_lex("seconds_to_minutes", ["convert seconds to minutes"], ["whole minutes"], [])
_lex("days_to_hours", ["convert days to hours"], ["in hours"], [])
_lex("weeks_to_days", ["convert weeks to days"], ["in days"], [])
_lex("minutes_since_midnight", ["minutes since midnight"], ["minute offset"], [])
_lex("rectangle_area", ["area"], ["area of a rectangle", "surface of"], ["how much space"])
_lex("rectangle_perimeter", ["perimeter"], ["outline length", "around the rectangle"], [])
_lex("circle_area", ["area"], ["area of a circle", "disc surface"], ["how much space"])
_lex("hypotenuse", ["hypotenuse"], ["diagonal", "right triangle"], ["straight-line distance"])
_lex("bitwise_and", ["bitwise and"], ["mask and", "bits set in both"], [])
_lex("bitwise_or", ["bitwise or"], ["mask or", "bits set in either"], [])
_lex("bitwise_xor", ["bitwise xor"], ["exclusive or"], [])
_lex("left_shift", ["shift left"], ["shift the bits", "bits up"], [])
_lex("lookup_unit_factor", ["look up"], ["metre factor", "unit factor"], [])
_lex("apply_rate_override", ["override"], ["apply the policy", "adjusted by"], [])
_lex("classify_threshold", ["classify"], ["above or", "band label", "against the threshold"],
     ["which category"])
_lex("grade_band", ["grade band"], ["performance tier", "low medium or high"],
     ["which category"])


def lexicon_for(sid: str) -> Dict[str, List[str]]:
    entry = OPERATION_LEXICON.get(sid)
    if entry:
        return entry
    return {"exact": [sid.replace("_", " ")], "lexical": [], "semantic": []}


# ── procedural cue detection ──────────────────────────────────────────────
_STEP_NUMBER_RE = re.compile(
    r"\b(?:step\s*(?:one|two|three|four|five|six|seven|eight|\d+)|"
    r"\(\s*\d\s*\)|\d\s*[\).]\s+)", re.I)
_ORDINAL_RE = re.compile(
    r"\b(?:first|firstly|second|secondly|third|thirdly|fourth|fifth|"
    r"initially|to begin|lastly|finally|in the end|to finish)\b", re.I)
_CONNECTIVE_RE = re.compile(
    r"\b(?:then|next|after that|afterwards|subsequently|once completed|"
    r"once done|followed by|and finally|concluding by|at the end)\b", re.I)
_INTERMEDIATE_REF_RE = re.compile(
    r"\b(?:that result|the result of (?:the )?(?:previous |prior |step ?\d+)?"
    r"(?:step)?|use the result|the previous result|the value those steps produce|"
    r"all of the previous results|the outcome of the previous)\b", re.I)


def procedural_cues(question: str) -> Dict[str, int]:
    q = question or ""
    return {
        "step_number_count": len(_STEP_NUMBER_RE.findall(q)),
        "ordinal_cue_count": len(_ORDINAL_RE.findall(q)),
        "connective_cue_count": len(_CONNECTIVE_RE.findall(q)),
        "explicit_intermediate_reference_count": len(_INTERMEDIATE_REF_RE.findall(q)),
    }


def procedural_cue_count(question: str) -> int:
    c = procedural_cues(question)
    return (c["step_number_count"] + c["ordinal_cue_count"]
            + c["connective_cue_count"] + c["explicit_intermediate_reference_count"])


# ── operation explicitness ────────────────────────────────────────────────
def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9$%. ]+", " ", (text or "").lower())


def _find_first(hay: str, needles: Iterable[str]) -> int:
    best = -1
    for n in needles:
        if not n:
            continue
        pos = hay.find(n.lower())
        if pos >= 0 and (best < 0 or pos < best):
            best = pos
    return best


def operation_explicitness(question: str, gold_sids: Sequence[str]) -> Dict[str, Any]:
    """Per-operation cue strength, plus the cue positions used for ordering."""
    hay = " " + re.sub(r"\s+", " ", _norm(question)) + " "
    per_op: List[Dict[str, Any]] = []
    for i, sid in enumerate(gold_sids):
        lex = lexicon_for(sid)
        p_exact = _find_first(hay, lex["exact"])
        p_lex = _find_first(hay, lex["lexical"])
        p_sem = _find_first(hay, lex["semantic"])
        if p_exact >= 0:
            level, pos = "exact", p_exact
        elif p_lex >= 0:
            level, pos = "lexical", p_lex
        elif p_sem >= 0:
            level, pos = "semantic", p_sem
        else:
            level, pos = "implicit", -1
        per_op.append({"step": i, "primitive_id": sid,
                       "capability_family": family_of(sid),
                       "cue_level": level, "cue_position": pos})
    n = len(gold_sids) or 1
    n_exact = sum(1 for o in per_op if o["cue_level"] == "exact")
    n_lexical = sum(1 for o in per_op if o["cue_level"] == "lexical")
    n_semantic = sum(1 for o in per_op if o["cue_level"] == "semantic")
    n_implicit = sum(1 for o in per_op if o["cue_level"] == "implicit")
    return {
        "n_gold_operations": len(gold_sids),
        "n_exactly_named_operations": n_exact,
        "n_lexically_cued_operations": n_lexical,
        "n_semantically_cued_operations": n_semantic,
        "n_implicit_operations": n_implicit,
        "exact_operation_coverage": round(n_exact / n, 4),
        "lexical_operation_coverage": round((n_exact + n_lexical) / n, 4),
        "implicit_operation_rate": round((n_semantic + n_implicit) / n, 4),
        "per_operation": per_op,
    }


# ── sequence leakage ──────────────────────────────────────────────────────
def _lcs_len(a: Sequence[int], b: Sequence[int]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def sequence_leakage(explicitness: Dict[str, Any]) -> Dict[str, Any]:
    """How closely the order of textual cues tracks the gold call order."""
    cued = [o for o in explicitness["per_operation"]
            if o["cue_level"] in ("exact", "lexical") and o["cue_position"] >= 0]
    n_gold = explicitness["n_gold_operations"] or 1
    if len(cued) < 2:
        return {
            "n_ordered_cues": len(cued),
            "lcs_ratio": 0.0,
            "kendall_agreement": 0.0,
            "exact_ordered_operation_coverage": round(len(cued) / n_gold, 4)
                                                if len(cued) == 1 else 0.0,
            "sequence_leakage": round(len(cued) / n_gold * 0.25, 4),
        }
    by_text = sorted(cued, key=lambda o: o["cue_position"])
    gold_order = [o["step"] for o in cued]
    text_order = [o["step"] for o in by_text]
    lcs = _lcs_len(gold_order, text_order)
    lcs_ratio = lcs / len(gold_order)
    conc = disc = 0
    for i in range(len(text_order)):
        for j in range(i + 1, len(text_order)):
            if text_order[i] < text_order[j]:
                conc += 1
            else:
                disc += 1
    kendall = (conc - disc) / max(conc + disc, 1)
    ordered_cov = (lcs / n_gold)
    leak = 0.5 * ordered_cov + 0.3 * lcs_ratio + 0.2 * max(kendall, 0.0)
    return {
        "n_ordered_cues": len(cued),
        "lcs_ratio": round(lcs_ratio, 4),
        "kendall_agreement": round(kendall, 4),
        "exact_ordered_operation_coverage": round(ordered_cov, 4),
        "sequence_leakage": round(min(leak, 1.0), 4),
    }


# ── query-mode classification ─────────────────────────────────────────────
def classify_query_mode(question: str, gold_sids: Sequence[str],
                        explicitness: Optional[Dict[str, Any]] = None,
                        leak: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Deterministic rule-based classifier with evidence flags and confidence."""
    ex = explicitness or operation_explicitness(question, gold_sids)
    lk = leak or sequence_leakage(ex)
    cues = procedural_cues(question)
    n_gold = ex["n_gold_operations"] or 1
    n_cue = cues["step_number_count"] + cues["ordinal_cue_count"] + cues["connective_cue_count"]
    lex_cov = ex["lexical_operation_coverage"]
    seq = lk["sequence_leakage"]

    flags = {
        "has_step_numbers": cues["step_number_count"] >= 2,
        "one_text_step_per_gold_call": n_cue >= max(n_gold - 1, 1) and lex_cov >= 0.8,
        "names_every_operation": lex_cov >= 0.95,
        "names_most_operations": lex_cov >= 0.6,
        "explicit_intermediate_reference":
            cues["explicit_intermediate_reference_count"] > 0,
        "ordered_cues": seq >= 0.6,
        "no_operation_named": lex_cov <= 0.2,
    }

    if flags["names_every_operation"] and (flags["has_step_numbers"] or flags["ordered_cues"]):
        mode, conf = "PROCEDURAL_EXPLICIT", 0.95
    elif lex_cov >= 0.8 and n_cue >= 1:
        mode, conf = "PROCEDURAL_EXPLICIT", 0.8
    elif lex_cov >= 0.5 and (n_cue >= 1 or seq >= 0.4):
        mode, conf = "PROCEDURAL_PARTIAL", 0.75
    elif 0.2 < lex_cov < 0.5:
        mode, conf = "SEMI_IMPLICIT", 0.7
    elif flags["no_operation_named"]:
        mode, conf = "GOAL_BASED_IMPLICIT", 0.8 if n_cue == 0 else 0.6
    else:
        mode, conf = "UNCLASSIFIED", 0.3

    return {
        "query_mode": mode,
        "confidence": conf,
        "evidence_flags": flags,
        "procedural_cue_count": n_cue + cues["explicit_intermediate_reference_count"],
        **cues,
    }


# ── per-task audit ────────────────────────────────────────────────────────
def audit_task(question: str, gold_sids: Sequence[str]) -> Dict[str, Any]:
    ex = operation_explicitness(question, gold_sids)
    lk = sequence_leakage(ex)
    cls = classify_query_mode(question, gold_sids, ex, lk)
    return {
        "schema_version": SCHEMA_VERSION,
        "lexicon_version": LEXICON_VERSION,
        **{k: v for k, v in ex.items() if k != "per_operation"},
        **lk,
        **cls,
        "per_operation": ex["per_operation"],
    }


def gold_sids_from_row(row: Dict[str, Any],
                       name_to_sid: Optional[Dict[str, str]] = None) -> List[str]:
    """Map a row's gold calls back onto registry primitive ids."""
    if name_to_sid is None:
        name_to_sid = surface_name_index()
    calls = row.get("gold_calls") or row.get("output") or row.get("canonical_calls") or []
    out = []
    for c in calls:
        name = str((c or {}).get("name") or "")
        sid = name_to_sid.get(name)
        if sid:
            out.append(sid)
    return out


_NAME_INDEX: Optional[Dict[str, str]] = None


def surface_name_index() -> Dict[str, str]:
    global _NAME_INDEX
    if _NAME_INDEX is None:
        from . import registry as reg

        _NAME_INDEX = {surf.name: sid for sid, _track, surf in reg.all_surfaces()}
    return dict(_NAME_INDEX)


# ── dataset-level aggregation ─────────────────────────────────────────────
def _bucket(x: float, edges: Sequence[float], labels: Sequence[str]) -> str:
    for e, lab in zip(edges, labels):
        if x <= e:
            return lab
    return labels[-1]


EXPLICITNESS_BUCKETS = ([0.0, 0.25, 0.5, 0.75, 1.01],
                        ["none", "low", "medium", "high", "full"])
LEAKAGE_BUCKETS = ([0.0, 0.2, 0.4, 0.7, 1.01],
                   ["none", "low", "medium", "high", "full"])
CUE_BUCKETS = ([0, 1, 3, 6, 10 ** 6], ["0", "1", "2-3", "4-6", "7+"])


def aggregate(audits: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    from collections import Counter

    n = len(audits) or 1
    modes = Counter(a["query_mode"] for a in audits)
    expl = Counter(_bucket(a["lexical_operation_coverage"], *EXPLICITNESS_BUCKETS)
                   for a in audits)
    leak = Counter(_bucket(a["sequence_leakage"], *LEAKAGE_BUCKETS) for a in audits)
    cues = Counter(_bucket(a["procedural_cue_count"], *CUE_BUCKETS) for a in audits)
    interm = Counter("present" if a["explicit_intermediate_reference_count"] > 0
                     else "absent" for a in audits)
    return {
        "n_tasks": len(audits),
        "query_mode_distribution": {k: round(v / n, 4) for k, v in sorted(modes.items())},
        "operation_explicitness_distribution": {k: round(v / n, 4)
                                                for k, v in sorted(expl.items())},
        "sequence_leakage_distribution": {k: round(v / n, 4) for k, v in sorted(leak.items())},
        "procedural_cue_distribution": {k: round(v / n, 4) for k, v in sorted(cues.items())},
        "intermediate_reference_explicitness": {k: round(v / n, 4)
                                                for k, v in sorted(interm.items())},
        "mean_exact_operation_coverage": round(
            sum(a["exact_operation_coverage"] for a in audits) / n, 4),
        "mean_lexical_operation_coverage": round(
            sum(a["lexical_operation_coverage"] for a in audits) / n, 4),
        "mean_implicit_operation_rate": round(
            sum(a["implicit_operation_rate"] for a in audits) / n, 4),
        "mean_sequence_leakage": round(sum(a["sequence_leakage"] for a in audits) / n, 4),
        "mean_procedural_cue_count": round(
            sum(a["procedural_cue_count"] for a in audits) / n, 4),
        "plan_leak_rate": round(
            sum(1 for a in audits
                if a["lexical_operation_coverage"] >= 0.8 and a["sequence_leakage"] >= 0.5)
            / n, 4),
    }


def audit_dataset(rows: Sequence[Dict[str, Any]], label: str,
                  *, keep_text: bool = True) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Returns (per-task rows, aggregate profile).

    ``keep_text=False`` is used for benchmark-derived corpora: only aggregates
    and hashed task ids leave the audit.
    """
    import hashlib

    idx = surface_name_index()
    per_task: List[Dict[str, Any]] = []
    audits: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        q = str(row.get("question") or row.get("input") or "")
        sids = gold_sids_from_row(row, idx)
        if not sids:
            sids = _fallback_sids(row)
        a = audit_task(q, sids)
        audits.append(a)
        rid = str(row.get("sample_id") or row.get("task_id") or i)
        per_task.append({
            "dataset": label,
            "task_ref": rid if keep_text else hashlib.sha256(rid.encode()).hexdigest()[:16],
            "call_count": len(sids),
            "call_bucket": _call_bucket(len(sids)),
            "query_mode": a["query_mode"],
            "query_mode_confidence": a["confidence"],
            "exact_operation_coverage": a["exact_operation_coverage"],
            "lexical_operation_coverage": a["lexical_operation_coverage"],
            "implicit_operation_rate": a["implicit_operation_rate"],
            "sequence_leakage": a["sequence_leakage"],
            "lcs_ratio": a["lcs_ratio"],
            "kendall_agreement": a["kendall_agreement"],
            "procedural_cue_count": a["procedural_cue_count"],
            "step_number_count": a["step_number_count"],
            "explicit_intermediate_reference_count":
                a["explicit_intermediate_reference_count"],
            "question_chars": len(q),
            "question": q if keep_text else "",
        })
    return per_task, {"dataset": label, **aggregate(audits)}


def _fallback_sids(row: Dict[str, Any]) -> List[str]:
    """Benchmark rows use foreign tool names; fall back to the raw name."""
    calls = row.get("gold_calls") or row.get("output") or []
    return [str((c or {}).get("name") or "") for c in calls if isinstance(c, dict)]


def _call_bucket(n: int) -> str:
    if n <= 2:
        return "2"
    if n >= 6:
        return "6+"
    return str(n)


def select_examples(per_task: Sequence[Dict[str, Any]], per_mode: int = 2
                    ) -> List[Dict[str, Any]]:
    """Deterministic, category-balanced examples — not just the worst offenders."""
    out: List[Dict[str, Any]] = []
    for mode in QUERY_MODES:
        pool = [r for r in per_task if r["query_mode"] == mode and r.get("question")]
        if not pool:
            continue
        pool = sorted(pool, key=lambda r: str(r["task_ref"]))
        step = max(1, len(pool) // max(per_mode, 1))
        out.extend(pool[i * step] for i in range(min(per_mode, len(pool))))
    return out


def markdown_report(title: str, aggregates: Sequence[Dict[str, Any]],
                    notes: Sequence[str] = ()) -> str:
    lines = [f"# {title}", "",
             f"Lexicon version: `{LEXICON_VERSION}` — schema `{SCHEMA_VERSION}`", ""]
    if notes:
        lines += list(notes) + [""]
    lines += ["| dataset | n | mean lexical cov | mean seq leak | plan-leak rate | "
              "procedural cues |", "|---|---:|---:|---:|---:|---:|"]
    for a in aggregates:
        lines.append(
            f"| {a['dataset']} | {a['n_tasks']} | {a['mean_lexical_operation_coverage']:.3f} | "
            f"{a['mean_sequence_leakage']:.3f} | {a['plan_leak_rate']:.3f} | "
            f"{a['mean_procedural_cue_count']:.2f} |")
    lines += ["", "## Query-mode distribution", "",
              "| dataset | " + " | ".join(QUERY_MODES) + " |",
              "|---|" + "---:|" * len(QUERY_MODES)]
    for a in aggregates:
        d = a["query_mode_distribution"]
        lines.append(f"| {a['dataset']} | "
                     + " | ".join(f"{d.get(m, 0.0):.3f}" for m in QUERY_MODES) + " |")
    lines += ["", "## Reading", "",
              "- `lexical_operation_coverage` = share of gold operations the question",
              "  names exactly or by an unambiguous paraphrase.",
              "- `sequence_leakage` combines ordered coverage, LCS ratio and Kendall",
              "  agreement between cue order and gold call order.",
              "- `plan_leak_rate` = share of tasks with coverage >= 0.8 AND leakage >= 0.5.",
              ""]
    return "\n".join(lines) + "\n"


def examples_markdown(examples: Sequence[Dict[str, Any]]) -> str:
    lines = ["# PLAN_LEAK_EXAMPLES", "",
             "Deterministically sampled, balanced across query modes.", ""]
    for ex in examples:
        lines += [
            f"## `{ex['query_mode']}` — {ex['dataset']} / {ex['task_ref']}", "",
            f"- lexical coverage: {ex['lexical_operation_coverage']:.2f}",
            f"- sequence leakage: {ex['sequence_leakage']:.2f}",
            f"- procedural cues: {ex['procedural_cue_count']}",
            "", "```text", ex["question"][:900], "```", "",
        ]
    return "\n".join(lines) + "\n"
