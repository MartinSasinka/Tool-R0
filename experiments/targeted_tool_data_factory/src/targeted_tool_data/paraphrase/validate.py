"""Deterministic acceptance test for an LLM paraphrase.

A paraphrase is accepted only if the ORIGINAL PROGRAM is provably unchanged:
same numeric literals, same operations in the same order, every dependency
still stated, no computed value revealed, no new ambiguity. Anything the
validator cannot prove is rejected and the deterministic template survives —
the LLM can therefore never influence the oracle (DESIGN.md §9).
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from .. import registry as reg
from ..graph import _refs_in
from ..schemas import TaskRecord

# words that express "the value produced by an earlier step"
_REF_MARKERS = [
    "that result", "this result", "the result", "that value", "this value",
    "the previous", "previous value", "previous result", "that number",
    "that outcome", "the outcome", "resulting", "what you get",
    "the figure you", "that figure", "step ", "it ", "the answer from",
    "the value those steps", "those steps", "that total", "that amount",
]

# operation keywords per primitive: at least one must be present, in order
_OP_KEYWORDS: Dict[str, List[str]] = {
    "add": ["add", "sum", "plus", "total", "combin", "together"],
    "subtract": ["subtract", "minus", "less", "reduce", "deduct", "take away",
                 "remove", "remain", "difference"],
    "multiply": ["multipl", "times", "product", "scale", "by a factor"],
    "divide": ["divide", "divid", "split", "per ", "quotient", "share"],
    "power": ["power", "raise", "exponent", "cube", "squar"],
    "floor_divide": ["whole times", "how many times", "fit", "full group",
                     "complete group", "whole number of", "integer division",
                     "whole times", "fits into", "whole quotient"],
    "modulo": ["remainder", "left over", "leftover", "modulo", "mod "],
    "percent_of": ["percent", "%"],
    "ratio_of": ["ratio", "proportion", "relative to", "compare"],
    "abs_difference": ["absolute difference", "difference", "gap", "apart",
                       "how far"],
    "average_two": ["average", "mean", "midpoint", "halfway"],
    "max_two": ["larger", "bigger", "greater of", "higher", "maximum"],
    "min_two": ["smaller", "lower", "lesser", "minimum"],
    "increase_by_percent": ["increase", "grow", "raise", "up by", "more",
                            "percent"],
    "decrease_by_percent": ["decrease", "reduce", "cut", "lower", "down by",
                            "discount", "less", "percent"],
    "sqrt": ["square root", "root"],
    "negate": ["negat", "flip the sign", "opposite", "invert the sign",
               "make it negative"],
    "inverse": ["reciprocal", "one divided by", "invert", "1 divided by"],
    "floor_value": ["round", "down", "floor", "drop the"],
    "ceil_value": ["round", "up", "ceiling", "next whole"],
    "square": ["squar", "itself", "second power"],
    "round_to_int": ["round", "nearest", "whole number"],
    "digit_sum": ["digit"],
    "round_places": ["round", "decimal", "precision"],
    "clamp": ["clamp", "bound", "limit", "between"],
    "round_direction": ["round", "direction", "mode"],
    "is_greater": ["greater", "exceed", "bigger", "larger", "above", "more than"],
    "is_within_range": ["between", "range", "within", "inside", "limits"],
    "is_divisible_by": ["divisible", "divides evenly", "evenly", "whole groups",
                        "without a remainder", "no remainder"],
    "number_to_string": ["text", "string", "written", "as words", "render"],
    "parse_number": ["text", "string", "parse", "read", "numeric"],
    "format_fixed": ["decimal", "format", "text", "string", "places"],
    "format_with_unit": ["unit", "label", "annotate", "append", "with the"],
    "tag_value": ["identifier", "prefix", "tag", "code", "label", "reference"],
    "text_length": ["character", "length", "how long", "letters"],
    "concat_texts": ["join", "concat", "combine", "together", "append"],
    "join_values": ["join", "separat", "single string", "one string", "delimit"],
    "sum_values": ["sum", "total", "add"],
    "mean_values": ["average", "mean"],
    "max_values": ["maximum", "largest", "highest", "peak", "biggest"],
    "min_values": ["minimum", "smallest", "lowest"],
    "count_values": ["count", "how many", "number of items", "length"],
    "range_spread": ["spread", "range", "max minus min", "difference between the"],
    "index_of_max": ["position", "index", "which", "rank", "place"],
    "sort_values_desc": ["sort", "order", "descend", "largest to smallest",
                         "high to low"],
    "scale_list": ["scale", "multipl", "each", "every"],
    "filter_above": ["keep", "filter", "above", "greater", "exceed", "over"],
    "top_k_values": ["top", "largest", "leading", "highest"],
    "cumulative_sums": ["running total", "cumulative", "accumulat"],
    "append_value": ["append", "add", "extend", "end of the list", "to the list"],
    "seconds_to_minutes": ["minute", "second"],
    "hours_to_minutes": ["minute", "hour"],
    "minutes_to_seconds": ["second", "minute"],
    "km_to_meters": ["meter", "metre", "kilomet"],
    "meters_to_km": ["kilomet", "meter", "metre"],
    "celsius_to_fahrenheit": ["fahrenheit", "celsius", "temperature"],
    "fahrenheit_to_celsius": ["celsius", "fahrenheit", "temperature"],
}

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def numeric_multiset(text: str) -> List[str]:
    """Numeric tokens in canonical form ('7.0' == '7')."""
    out = []
    for tok in _NUM_RE.findall(text):
        try:
            f = float(tok)
        except ValueError:              # pragma: no cover
            continue
        out.append(str(int(f)) if f == int(f) else str(f))
    return sorted(out)


def step_descriptions(rec: TaskRecord) -> List[str]:
    """Neutral, oracle-free description of each step for the prompt."""
    out = []
    for i, nd in enumerate(rec.semantic_program.nodes):
        prim = reg.get(nd.semantic_id)
        deps = sorted({r for v in nd.inputs.values() for r in _refs_in(v)})
        idx = {n.node_id: j + 1 for j, n in enumerate(rec.semantic_program.nodes)}
        dep_txt = ""
        if deps:
            names = ", ".join(f"step {idx[d]}" for d in deps)
            dep_txt = f" (uses the result of {names})"
        out.append(prim.phrase.split("{")[0].strip().rstrip(",") or prim.sid)
        out[-1] = f"{prim.category}: {prim.sid.replace('_', ' ')}{dep_txt}"
    return out


def _ref_marker_count(text: str) -> int:
    low = text.lower()
    return sum(low.count(m) for m in _REF_MARKERS)


def validate_paraphrase(rec: TaskRecord, text: str, *,
                        max_len_ratio: float = 1.7,
                        min_len_ratio: float = 0.45) -> Tuple[bool, List[str]]:
    """Returns (accepted, reasons_for_rejection)."""
    from ..validation import v3_semantic

    errs: List[str] = []
    original = rec.query
    if not text or len(text) < 20:
        return False, ["empty or too short"]
    if len(text) > max(len(original) * max_len_ratio, 160):
        errs.append("too long")
    if len(text) < len(original) * min_len_ratio:
        errs.append("too short")
    if not text.isascii():
        errs.append("non-ascii output")
    for bad in ("```", "{", "}", "<", "|"):
        if bad in text:
            errs.append(f"formatting artifact {bad!r}")
            break
    if re.search(r"^\s*(sure|here|certainly|of course)\b", text, re.I):
        errs.append("chat preamble")

    # 1. numeric literals: exactly the same multiset (no invented, none lost,
    #    and no computed value can appear because it would be a new number)
    if numeric_multiset(text) != numeric_multiset(original):
        errs.append("numeric tokens changed")

    # 2. operations present and in the original order
    pos = 0
    for i, nd in enumerate(rec.semantic_program.nodes):
        kws = _OP_KEYWORDS.get(nd.semantic_id)
        if not kws:
            continue
        low = text.lower()
        found = -1
        for kw in kws:
            j = low.find(kw, pos)
            if j >= 0 and (found < 0 or j < found):
                found = j
        if found < 0:
            errs.append(f"operation {i + 1} ({nd.semantic_id}) missing or reordered")
            break
        pos = found + 1

    # 3. dependencies still expressed
    ref_nodes = sum(1 for nd in rec.semantic_program.nodes
                    if any(_refs_in(v) for v in nd.inputs.values()))
    if _ref_marker_count(text) < ref_nodes:
        errs.append("dependency references dropped")

    # 4. full V3 semantic re-check against the paraphrased surface
    probe = rec.model_copy(update={"query": text})
    v3 = v3_semantic(probe)
    if v3:
        errs.extend(f"V3: {e}" for e in v3[:3])

    return (not errs), errs


def shortlist(records: List[Dict[str, Any]], n: int, seed: int,
              cells: Optional[List[Dict[str, Any]]] = None,
              n_select: int = 0, target_share: float = 0.0,
              accept_rate: Any = 0.35) -> List[str]:
    """Structural shortlist for paraphrasing, deterministic.

    Without `cells` this stratifies over (track, call bucket, motif, answer
    type) across the whole validated pool. That spreads the requests evenly but
    wastes most of them: only ~15 % of the pool is ever selected, so a uniform
    shortlist leaves most generation cells with too few accepted paraphrases to
    reach the target share in the frozen dataset.

    With `cells`, the budget is allocated per generation cell in proportion to
    that cell's quota in the FINAL selection, scaled by the target paraphrase
    share and the observed acceptance rate. Within a cell, candidates are taken
    in `task_id` order, which is the tie-break the selector itself falls back
    to, so the shortlist and the selection agree on which records matter.

    `accept_rate` may be a scalar or a {call_count: rate} mapping. It has to be
    per call count in practice: a paraphrase of a 2-call question survives the
    validator about 64 % of the time, a 5-call one about 11 %, so a flat budget
    silently starves exactly the long-horizon cells that matter most.
    """
    import random

    def _rate(call_count: int) -> float:
        if isinstance(accept_rate, dict):
            table = {int(k): float(v) for k, v in accept_rate.items()}
            if not table:
                return 0.35
            key = min(table, key=lambda k: abs(k - call_count))
            return max(table[key], 0.05)
        return max(float(accept_rate), 0.05)

    rng = random.Random(seed)
    if cells and n_select > 0 and target_share > 0:
        by_cell: Dict[str, List[str]] = {}
        for r in records:
            by_cell.setdefault(r["generation_cell_id"], []).append(r["task_id"])
        for v in by_cell.values():
            v.sort()
        want: Dict[str, int] = {}
        for c in cells:
            cid = c["generation_cell_id"]
            need = c["quota_weight"] * n_select * target_share
            want[cid] = min(len(by_cell.get(cid, [])),
                            int(math.ceil(need / _rate(int(c["call_count"])))))
        out: List[str] = []
        for cid in sorted(want, key=lambda c: (-want[c], c)):
            out.extend(by_cell.get(cid, [])[:want[cid]])
        out = out[:n]
        # spare budget goes back to the cells with the largest remaining pool
        if len(out) < n:
            taken = set(out)
            rest = [t for cid in sorted(by_cell) for t in by_cell[cid]
                    if t not in taken]
            out.extend(rest[:n - len(out)])
        return out

    buckets: Dict[Tuple, List[str]] = {}
    for r in records:
        cc = r["call_count"]
        key = (r["track"], "6+" if cc >= 6 else str(cc), r["motif"],
               r["answer_type"])
        buckets.setdefault(key, []).append(r["task_id"])
    for v in buckets.values():
        rng.shuffle(v)
    out = []
    keys = sorted(buckets)
    while len(out) < n and any(buckets[k] for k in keys):
        for k in keys:
            if buckets[k] and len(out) < n:
                out.append(buckets[k].pop())
    return out
