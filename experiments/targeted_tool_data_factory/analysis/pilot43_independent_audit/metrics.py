"""Pure recomputation of dataset metrics from exported records.

Every function here takes exported record dicts (or plain values) and recomputes
a metric from the record CONTENT. Declared metadata is never used as a source of
truth; where a declared value is needed it is fetched explicitly through
:func:`get_path` so the auditor can compare it against the recomputed value.

Only the standard library is used.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

from .graph_recon import literal_arguments, parse_ref
from .pattern_rules import VALUE_KIND

MISSING = object()

# Declared answer-type spellings that mean the same thing as a recomputed kind.
_ANSWER_TYPE_ALIASES: Dict[str, str] = {
    "category": "string",
    "categorical": "string",
    "text": "string",
    "str": "string",
    "bool": "boolean",
    "int": "integer",
    "number": "float",
    "numeric": "float",
    "array": "list",
    "dict": "object",
}


def sha256_hex(text: str) -> str:
    """Stable hex digest of a text value."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_path(rec: Any, path: str) -> Any:
    """Fetch a dotted path from nested dicts, returning ``MISSING`` if absent."""
    cur = rec
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return MISSING
    return cur


def recompute_call_count(rec: Dict[str, Any]) -> int:
    """Number of gold calls actually present in the record."""
    calls = rec.get("gold_calls")
    return len(calls) if isinstance(calls, list) else 0


def answer_type_of(rec: Dict[str, Any]) -> str:
    """Recompute the answer type from the exported ``gold_answer`` value."""
    return VALUE_KIND(rec.get("gold_answer"))


def normalize_answer_type(declared: Any) -> str:
    """Normalise a declared answer type for comparison.

    A declared ``"category"`` is treated as ``"string"``; the raw declared value
    is kept by the caller so that the report can show both.
    """
    if not isinstance(declared, str):
        return ""
    key = declared.strip().lower()
    return _ANSWER_TYPE_ALIASES.get(key, key)


class PrimitiveUsage(NamedTuple):
    """Result of :func:`primitive_usage`."""

    counts: Counter
    disagreements: int
    unmapped: Counter
    source: str


def _call_primitive(
    call: Dict[str, Any],
    surface_to_primitive: Optional[Dict[str, str]],
) -> Tuple[Optional[str], Optional[str]]:
    """Return ``(mapped_primitive, declared_primitive)`` for one gold call."""
    declared = call.get("primitive_id")
    declared = declared if isinstance(declared, str) and declared else None
    mapped = None
    if surface_to_primitive is not None:
        mapped = surface_to_primitive.get(str(call.get("name", "")))
    return mapped, declared


def primitive_usage(
    records: Sequence[Dict[str, Any]],
    surface_to_primitive: Optional[Dict[str, str]] = None,
) -> PrimitiveUsage:
    """Count primitives actually used by gold calls.

    When a surface-name -> primitive map is supplied it is the primary source;
    the per-call ``primitive_id`` field (when present) is used as a cross-check
    and every mismatch increments ``disagreements``. Without a map the declared
    ``primitive_id`` is used, and surface names that cannot be resolved at all
    are collected in ``unmapped``.
    """
    counts: Counter = Counter()
    unmapped: Counter = Counter()
    disagreements = 0
    for rec in records:
        for call in rec.get("gold_calls") or []:
            if not isinstance(call, dict):
                continue
            mapped, declared = _call_primitive(call, surface_to_primitive)
            if mapped is not None and declared is not None and mapped != declared:
                disagreements += 1
            chosen = mapped if mapped is not None else declared
            if chosen is None:
                unmapped[str(call.get("name", ""))] += 1
            else:
                counts[chosen] += 1
    source = "surface_to_primitive" if surface_to_primitive else "declared_primitive_id"
    return PrimitiveUsage(counts=counts, disagreements=disagreements, unmapped=unmapped, source=source)


def capability_usage(
    records: Sequence[Dict[str, Any]],
    primitive_to_capability: Dict[str, str],
    surface_to_primitive: Optional[Dict[str, str]] = None,
) -> Counter:
    """Count capability families actually exercised by gold calls."""
    counts: Counter = Counter()
    for rec in records:
        for call in rec.get("gold_calls") or []:
            if not isinstance(call, dict):
                continue
            mapped, declared = _call_primitive(call, surface_to_primitive)
            prim = mapped if mapped is not None else declared
            if prim is None:
                counts["<unmapped>"] += 1
                continue
            counts[primitive_to_capability.get(prim, "<unknown_capability>")] += 1
    return counts


def concentration(counter: Counter) -> Dict[str, float]:
    """Top-1 and top-10 mass share of a count distribution."""
    total = sum(counter.values())
    if total <= 0:
        return {"total": 0, "distinct": 0, "top1_share": 0.0, "top10_share": 0.0}
    ordered = [c for _, c in counter.most_common()]
    return {
        "total": total,
        "distinct": len(counter),
        "top1_share": ordered[0] / total,
        "top10_share": sum(ordered[:10]) / total,
    }


def primitive_sequences(
    records: Sequence[Dict[str, Any]],
    surface_to_primitive: Optional[Dict[str, str]] = None,
    primitive_to_capability: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Count exact primitive sequences and normalised capability sequences."""
    prim_seqs: Counter = Counter()
    cap_seqs: Counter = Counter()
    for rec in records:
        prims: List[str] = []
        caps: List[str] = []
        for call in rec.get("gold_calls") or []:
            if not isinstance(call, dict):
                continue
            mapped, declared = _call_primitive(call, surface_to_primitive)
            prim = mapped if mapped is not None else declared
            if prim is None:
                prim = f"surface:{call.get('name', '')}"
            prims.append(prim)
            if primitive_to_capability is not None:
                caps.append(primitive_to_capability.get(prim, "<unknown_capability>"))
        prim_seqs[tuple(prims)] += 1
        if primitive_to_capability is not None:
            cap_seqs[tuple(caps)] += 1
    return {
        "primitive_sequences": prim_seqs,
        "capability_sequences": cap_seqs,
        "primitive_sequence_concentration": concentration(prim_seqs),
        "capability_sequence_concentration": concentration(cap_seqs),
    }


# ---------------------------------------------------------------------------
# Query fingerprints
# ---------------------------------------------------------------------------

_URL_RE = re.compile(r"(?:[a-z][a-z0-9+.\-]*://\S+|www\.[^\s]+)")
_WINPATH_RE = re.compile(r"[a-z]:\\[^\s]+")
_POSIXPATH_RE = re.compile(r"(?<![\w])/[\w.\-]+(?:/[\w.\-]+)+")
_QUOTED_RE = re.compile(r"\"[^\"]*\"|'[^']*'")
_CURRENCY_RE = re.compile(r"[$\u20ac\u00a3\u00a5\u20b9]|\b(?:usd|eur|gbp|jpy|inr)\b")
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
_WS_RE = re.compile(r"\s+")
_SENTENCE_END_RE = re.compile(r"[.?!]+")

_PLACEHOLDER_SAFE = set("<>_")

#: Closed list of function words used for the coarse intent fingerprint.
FUNCTION_WORDS: Tuple[str, ...] = (
    "a", "about", "after", "all", "am", "an", "and", "any", "are", "as", "at", "be",
    "been", "before", "below", "but", "by", "can", "compute", "could", "did", "do",
    "does", "each", "either", "else", "every", "for", "from", "given", "had", "has",
    "have", "how", "if", "in", "into", "is", "it", "its", "many", "much", "must",
    "of", "on", "once", "only", "or", "over", "per", "report", "should", "since",
    "so", "than", "that", "the", "their", "then", "there", "these", "this", "those",
    "to", "under", "until", "up", "using", "was", "we", "were", "what", "when",
    "where", "which", "while", "who", "why", "will", "with", "within", "would",
    "you", "your",
)
_FUNCTION_WORD_SET = frozenset(FUNCTION_WORDS)


def _strip_punctuation(text: str) -> str:
    """Drop punctuation, keeping sentence-final ``?``/``.`` and placeholders."""
    out: List[str] = []
    for i, ch in enumerate(text):
        if ch.isalnum() or ch.isspace() or ch in _PLACEHOLDER_SAFE:
            out.append(ch)
            continue
        if ch in ".?":
            nxt = text[i + 1 : i + 2]
            if nxt == "" or nxt.isspace():
                out.append(ch)
            continue
    return "".join(out)


def lexical_skeleton(text: str) -> str:
    """Normalised lexical skeleton of a query.

    Lowercases, replaces URLs / filesystem paths / quoted spans with ``<str>``,
    currency symbols and ISO codes with ``<cur>``, all numbers with ``<n>``,
    strips punctuation except sentence-final ``?``/``.``, and collapses
    whitespace. Two queries with the same skeleton differ only in surface
    values, which makes skeleton concentration a direct measure of templating.
    """
    s = str(text or "").lower()
    s = _URL_RE.sub(" <str> ", s)
    s = _WINPATH_RE.sub(" <str> ", s)
    s = _POSIXPATH_RE.sub(" <str> ", s)
    s = _QUOTED_RE.sub(" <str> ", s)
    s = _CURRENCY_RE.sub(" <cur> ", s)
    s = _NUMBER_RE.sub(" <n> ", s)
    s = _strip_punctuation(s)
    return _WS_RE.sub(" ", s).strip()


def intent_signature(text: str) -> str:
    """Coarse intent fingerprint of a query.

    Built from the first six function words of the query, a question-form
    marker (``Q`` when the text contains ``?``, else ``S``) and the sentence
    count. This deliberately ignores content words so that queries built from
    the same phrasing template collide even when their domain wording differs.
    """
    s = str(text or "").lower()
    words = re.findall(r"[a-z']+", s)
    picked = [w for w in words if w in _FUNCTION_WORD_SET][:6]
    marker = "Q" if "?" in str(text or "") else "S"
    sentences = len([p for p in _SENTENCE_END_RE.split(str(text or "")) if p.strip()])
    return "|".join(picked) + f"#{marker}#{max(sentences, 1)}"


def query_fingerprints(text: str) -> Dict[str, str]:
    """Exact / skeleton / intent fingerprints of one query string."""
    raw = str(text or "")
    skeleton = lexical_skeleton(raw)
    intent = intent_signature(raw)
    return {
        "exact": sha256_hex(raw),
        "skeleton": skeleton,
        "skeleton_hash": sha256_hex(skeleton),
        "intent": intent,
        "intent_hash": sha256_hex(intent),
    }


def duplicate_rates(
    records: Sequence[Dict[str, Any]],
    text_key: str = "question",
) -> Dict[str, Any]:
    """Query repetition measures recomputed from the exported text.

    ``exact_duplicate_rate`` is the share of records whose exact query text is
    not unique in the collection.
    """
    exact: Counter = Counter()
    skeleton: Counter = Counter()
    intent: Counter = Counter()
    for rec in records:
        fp = query_fingerprints(rec.get(text_key, ""))
        exact[fp["exact"]] += 1
        skeleton[fp["skeleton_hash"]] += 1
        intent[fp["intent_hash"]] += 1
    n = max(len(records), 1)
    n_dup_records = sum(c for c in exact.values() if c >= 2)
    return {
        "n": len(records),
        "n_distinct_exact": len(exact),
        "exact_duplicate_rate": n_dup_records / n,
        "n_distinct_skeleton": len(skeleton),
        "top1_skeleton_share": concentration(skeleton)["top1_share"],
        "top10_skeleton_share": concentration(skeleton)["top10_share"],
        "n_distinct_intent": len(intent),
        "top1_intent_share": concentration(intent)["top1_share"],
        "top10_intent_share": concentration(intent)["top10_share"],
    }


def boolean_balance(
    records: Sequence[Dict[str, Any]],
    workflow_path: str = "workflow_id",
    cell_path: str = "cell_tier",
    text_key: str = "question",
) -> Dict[str, Any]:
    """True-share tables for boolean answers, overall and per grouping.

    Groupings: workflow id, coarse intent template (from the query text) and
    generation cell / tier. A balanced boolean dataset sits near 0.5 in every
    non-trivial group; a strong skew means the answer is guessable.
    """
    overall = [0, 0]
    by_workflow: Dict[str, List[int]] = {}
    by_intent: Dict[str, List[int]] = {}
    by_cell: Dict[str, List[int]] = {}

    def bump(table: Dict[str, List[int]], key: str, is_true: bool) -> None:
        slot = table.setdefault(key, [0, 0])
        slot[0] += 1
        if is_true:
            slot[1] += 1

    for rec in records:
        answer = rec.get("gold_answer")
        if not isinstance(answer, bool):
            continue
        overall[0] += 1
        if answer:
            overall[1] += 1
        wf = get_path(rec, workflow_path)
        cell = get_path(rec, cell_path)
        bump(by_workflow, str(wf) if wf is not MISSING else "<missing>", answer)
        bump(by_cell, str(cell) if cell is not MISSING else "<missing>", answer)
        bump(by_intent, intent_signature(rec.get(text_key, "")), answer)

    def render(table: Dict[str, List[int]]) -> Dict[str, Dict[str, float]]:
        return {
            key: {"n": slot[0], "n_true": slot[1], "true_share": slot[1] / slot[0]}
            for key, slot in sorted(table.items())
        }

    return {
        "n_boolean": overall[0],
        "n_true": overall[1],
        "overall_true_share": (overall[1] / overall[0]) if overall[0] else 0.0,
        "by_workflow": render(by_workflow),
        "by_intent_template": render(by_intent),
        "by_cell": render(by_cell),
    }


def split_overlap(
    splits: Dict[str, List[Dict[str, Any]]],
    keys: Sequence[str],
    train_key: str = "train",
) -> Dict[str, Dict[str, int]]:
    """Count key values shared between the train split and every other split.

    ``keys`` are dotted record paths. A non-zero count for a key that is meant
    to be disjoint across splits is leakage.
    """
    result: Dict[str, Dict[str, int]] = {}
    train_records = splits.get(train_key, [])
    for key in keys:
        train_values = {
            str(get_path(rec, key))
            for rec in train_records
            if get_path(rec, key) is not MISSING
        }
        per_split: Dict[str, int] = {}
        for name, records in sorted(splits.items()):
            if name == train_key:
                continue
            other = {
                str(get_path(rec, key))
                for rec in records
                if get_path(rec, key) is not MISSING
            }
            per_split[name] = len(train_values & other)
        result[key] = per_split
    return result


def tv_distance(p: Dict[str, float], q: Dict[str, float]) -> float:
    """Total variation distance between two distributions over a shared key set.

    Both inputs are normalised to sum to 1 over the union of their keys; an
    all-zero input is treated as the uniform-zero distribution, which makes the
    distance 1 against any non-empty distribution.
    """
    keys = set(p) | set(q)
    tp = float(sum(p.values()))
    tq = float(sum(q.values()))
    if tp <= 0 and tq <= 0:
        return 0.0
    if tp <= 0 or tq <= 0:
        return 1.0
    return 0.5 * sum(abs(p.get(k, 0.0) / tp - q.get(k, 0.0) / tq) for k in keys)


def numeric_literal_stats(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Distribution of literal numeric arguments across all gold calls.

    Used as a value-realism probe: real-world quantities are rarely uniform
    integers in a narrow band, and a high share of "generic" integers in
    1..2000 indicates values drawn from a single random range.
    """
    values: List[float] = []
    for rec in records:
        for call in rec.get("gold_calls") or []:
            if not isinstance(call, dict):
                continue
            for lit in literal_arguments(call.get("arguments") or {}):
                if isinstance(lit, bool):
                    continue
                if isinstance(lit, (int, float)):
                    values.append(float(lit))
    n = len(values)
    if n == 0:
        return {"n": 0}
    ints = [v for v in values if float(v).is_integer()]
    round_vals = [v for v in ints if int(v) % 10 == 0]
    generic = [v for v in ints if 1 <= v <= 2000]
    ordered = sorted(values)
    return {
        "n": n,
        "min": ordered[0],
        "max": ordered[-1],
        "mean": sum(values) / n,
        "median": ordered[n // 2],
        "share_integer": len(ints) / n,
        "share_round_multiple_of_10": len(round_vals) / n,
        "share_generic_int_1_2000": len(generic) / n,
        "n_distinct": len(set(values)),
    }


def surface_names(records: Sequence[Dict[str, Any]]) -> Counter:
    """Count distinct gold tool surface names actually called."""
    counts: Counter = Counter()
    for rec in records:
        for call in rec.get("gold_calls") or []:
            if isinstance(call, dict):
                counts[str(call.get("name", ""))] += 1
    return counts


def surface_to_primitive_from_tools(
    records: Sequence[Dict[str, Any]],
    semantic_id_key: str = "semantic_id",
) -> Dict[str, Any]:
    """Derive a surface-name -> primitive map from the exported tool schemas.

    Exported records carry their own tool list, and each tool may declare the
    primitive it realises. This is exported DATA, not producer code, so using it
    keeps the audit independent. Surface names that map to more than one
    primitive are reported as collisions and resolved by majority count.
    """
    observed: Dict[str, Counter] = {}
    for rec in records:
        for tool in rec.get("tools") or []:
            if not isinstance(tool, dict):
                continue
            name = str(tool.get("name", ""))
            prim = tool.get(semantic_id_key)
            if not name or not isinstance(prim, str) or not prim:
                continue
            observed.setdefault(name, Counter())[prim] += 1
    mapping = {name: counter.most_common(1)[0][0] for name, counter in observed.items()}
    collisions = {
        name: dict(counter) for name, counter in observed.items() if len(counter) > 1
    }
    return {"mapping": mapping, "collisions": collisions, "n_surfaces": len(mapping)}


def json_safe(value: Any) -> Any:
    """Convert counters and tuple keys into JSON-serialisable structures."""
    if isinstance(value, Counter):
        return {json.dumps(k) if isinstance(k, tuple) else str(k): v for k, v in value.items()}
    if isinstance(value, dict):
        return {
            (json.dumps(k) if isinstance(k, tuple) else str(k)): json_safe(v)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def is_reference_value(value: Any) -> bool:
    """True when a value is a tool-call reference string."""
    return parse_ref(value) is not None
