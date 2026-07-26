"""Hard-distractor synthesis and per-task offered-tool set control.

Hard distractor = tool that is semantically/schematically close to a gold
tool but wrong for the step (capabilities 1, 8). Random unrelated tools are
easy distractors, counted separately (DESIGN.md §11).
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from .. import registry as reg
from ..render import render_tool
from ..schemas import ToolSpec
from ..util import char_ngrams, jaccard

# near-semantics confusion pairs (same domain, different aggregation/rounding)
CONFUSION_PAIRS: Dict[str, List[str]] = {
    "floor_value": ["ceil_value", "round_places"],
    "ceil_value": ["floor_value", "round_places"],
    "round_places": ["floor_value", "ceil_value"],
    "sum_values": ["mean_values", "max_values"],
    "mean_values": ["sum_values", "range_spread"],
    "max_values": ["min_values", "range_spread"],
    "min_values": ["max_values", "count_values"],
    "max_two": ["min_two", "average_two"],
    "min_two": ["max_two", "average_two"],
    "increase_by_percent": ["decrease_by_percent", "percent_of"],
    "decrease_by_percent": ["increase_by_percent", "percent_of"],
    "percent_of": ["ratio_of", "increase_by_percent"],
    "ratio_of": ["divide", "percent_of"],
    "divide": ["floor_divide", "modulo", "ratio_of"],
    "floor_divide": ["divide", "modulo"],
    "modulo": ["floor_divide", "divide"],
    "add": ["subtract", "multiply", "average_two"],
    "subtract": ["add", "abs_difference"],
    "abs_difference": ["subtract", "average_two"],
    "multiply": ["add", "power", "square"],
    "power": ["multiply", "square"],
    "square": ["power", "sqrt"],
    "sqrt": ["square", "inverse"],
    "inverse": ["negate", "sqrt"],
    "negate": ["inverse", "abs_difference"],
    "seconds_to_minutes": ["hours_to_minutes", "floor_divide"],
    "hours_to_minutes": ["seconds_to_minutes", "multiply"],
    "number_to_string": ["format_fixed", "parse_number"],
    "format_fixed": ["number_to_string", "round_places"],
    "parse_number": ["number_to_string", "text_length"],
    "concat_texts": ["text_length", "number_to_string"],
    "text_length": ["count_values", "parse_number"],
    "average_two": ["add", "mean_values"],
    "km_to_meters": ["hours_to_minutes", "multiply"],
    "celsius_to_fahrenheit": ["km_to_meters", "multiply"],
    "count_values": ["sum_values", "text_length"],
    "range_spread": ["abs_difference", "max_values"],
    "clamp": ["max_two", "min_two"],
    "round_direction": ["round_places", "floor_value"],
    # pilot2 typed primitives
    "is_greater": ["is_within_range", "is_divisible_by", "max_two"],
    "is_within_range": ["is_greater", "clamp"],
    "is_divisible_by": ["modulo", "is_greater"],
    "round_to_int": ["floor_value", "ceil_value", "round_places"],
    "digit_sum": ["sum_values", "text_length"],
    "index_of_max": ["max_values", "count_values"],
    "sort_values_desc": ["top_k_values", "cumulative_sums"],
    "top_k_values": ["sort_values_desc", "filter_above"],
    "filter_above": ["top_k_values", "sort_values_desc"],
    "scale_list": ["cumulative_sums", "sort_values_desc"],
    "cumulative_sums": ["scale_list", "sum_values"],
    "join_values": ["concat_texts", "sort_values_desc"],
    "format_with_unit": ["number_to_string", "format_fixed"],
    "meters_to_km": ["km_to_meters", "divide"],
    "minutes_to_seconds": ["hours_to_minutes", "seconds_to_minutes"],
    "fahrenheit_to_celsius": ["celsius_to_fahrenheit", "subtract"],
    "sum_three": ["mean_three", "range_three"],
    "mean_three": ["sum_three", "range_three"],
    "range_three": ["sum_three", "mean_three"],
}


def _sig(p: reg.Primitive) -> Tuple:
    return (tuple(sorted(t for _n, t, _s in p.params)), p.out_type)


def name_similarity(a: str, b: str) -> float:
    return jaccard(char_ngrams(a, 3), char_ngrams(b, 3))


def desc_similarity(a: str, b: str) -> float:
    return jaccard(set(a.lower().split()), set(b.lower().split()))


def sig_similarity(a: ToolSpec, b: ToolSpec) -> float:
    score = 0.0
    if len(a.params) == len(b.params):
        score += 0.4
    if sorted(p.type for p in a.params) == sorted(p.type for p in b.params):
        score += 0.4
    if a.output_type == b.output_type:
        score += 0.2
    return score


def _candidates_same_signature(gold_sids: List[str]) -> List[str]:
    prims = reg.all_primitives()
    out = []
    gold_sigs = {_sig(prims[s]) for s in gold_sids if s in prims}
    for sid, p in prims.items():
        if sid not in gold_sids and _sig(p) in gold_sigs:
            out.append(sid)
    return sorted(out)


def build_offered_set(
    gold_tools: List[ToolSpec], track: str, rng: random.Random,
    offered_count: int, hard_distractor_type: Optional[str],
    param_style: str = "semantic",
) -> Tuple[List[ToolSpec], List[int], Dict[str, float]]:
    """Returns (offered tools shuffled, gold positions, hard-distractor similarity)."""
    gold_sids = [t.semantic_id for t in gold_tools]
    gold_names = [t.name for t in gold_tools]
    prims = reg.all_primitives()
    used_names = set(gold_names)
    hard: List[ToolSpec] = []
    n_hard_wanted = max(2, (offered_count - len(gold_tools)) // 3) if hard_distractor_type else 0

    def _add_hard(sid: str, dtype: str) -> bool:
        if len(hard) >= n_hard_wanted:
            return False
        spec = render_tool(sid, track, rng, param_style=param_style)
        if spec.name in used_names:
            for surf in prims[sid].surfaces(track):
                if surf.name not in used_names:
                    spec = render_tool(sid, track, rng, surface=surf, param_style=param_style)
                    break
            else:
                return False
        spec.is_distractor = True
        spec.distractor_type = dtype
        spec.similarity_to_gold = {
            "name": max(name_similarity(spec.name, g) for g in gold_names),
            "description": max(desc_similarity(spec.description, t.description) for t in gold_tools),
            "signature": max(sig_similarity(spec, t) for t in gold_tools),
        }
        used_names.add(spec.name)
        hard.append(spec)
        return True

    # 1) near-semantics confusion partners of gold ops (strongest confusers)
    partners = []
    for sid in gold_sids:
        partners.extend(s for s in CONFUSION_PAIRS.get(sid, []) if s not in gold_sids)
    rng.shuffle(partners)
    for sid in partners:
        _add_hard(sid, hard_distractor_type or "near_semantics")

    # 2) same signature different semantics
    if hard_distractor_type in (None, "same_signature_different_semantics") or len(hard) < n_hard_wanted:
        for sid in _candidates_same_signature(gold_sids):
            if sid not in {h.semantic_id for h in hard}:
                _add_hard(sid, "same_signature_different_semantics")

    # 3) similar-name pick: remaining prims ranked by lexical closeness to gold names
    if len(hard) < n_hard_wanted:
        scored = []
        for sid, p in prims.items():
            if sid in gold_sids or sid in {h.semantic_id for h in hard}:
                continue
            for surf in p.surfaces(track):
                s = max(name_similarity(surf.name, g) for g in gold_names)
                scored.append((s, sid))
        scored.sort(reverse=True)
        for _s, sid in scored:
            if not _add_hard(sid, "similar_name"):
                break

    # easy distractors fill the rest
    easy: List[ToolSpec] = []
    pool = sorted(set(prims) - set(gold_sids) - {h.semantic_id for h in hard})
    rng.shuffle(pool)
    while len(gold_tools) + len(hard) + len(easy) < offered_count and pool:
        sid = pool.pop()
        spec = render_tool(sid, track, rng, param_style=param_style)
        if spec.name in used_names:
            continue
        spec.is_distractor = True
        spec.distractor_type = "easy"
        used_names.add(spec.name)
        easy.append(spec)

    offered = list(gold_tools) + hard + easy
    rng.shuffle(offered)
    gold_positions = [i for i, t in enumerate(offered) if not t.is_distractor]
    sims = {"name": 0.0, "description": 0.0, "signature": 0.0}
    if hard:
        for k in sims:
            sims[k] = round(sum(h.similarity_to_gold[k] for h in hard) / len(hard), 4)
    return offered, gold_positions, sims
