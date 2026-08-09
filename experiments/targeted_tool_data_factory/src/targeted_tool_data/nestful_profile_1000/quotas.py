"""Target quotas from nestful_dev_200 TargetProfile (Hamilton / largest-remainder)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from . import CALL_HARD, N_TRAIN

# Explicit audit trail: nestful_dev classifier → Pilot4.3 modes.
# GOAL_BASED_IMPLICIT in NESTFUL mixes scenario-grounded and pure-goal phrasing;
# split 70/30 into DOMAIN_GROUNDED / GOAL_BASED to match Pilot4.3 taxonomy without
# dumping the entire 70% into one label.
DEV_MODE_MAP: Dict[str, Tuple[Tuple[str, float], ...]] = {
    "GOAL_BASED_IMPLICIT": (
        ("DOMAIN_GROUNDED_IMPLICIT", 0.70),
        ("GOAL_BASED_IMPLICIT", 0.30),
    ),
    "SEMI_IMPLICIT": (("SEMI_IMPLICIT", 1.0),),
    "PROCEDURAL_PARTIAL": (
        ("OPERATION_EXPLICIT_GRAPH_IMPLICIT", 0.65),
        ("SEMI_IMPLICIT", 0.35),
    ),
    "PROCEDURAL_EXPLICIT": (("GRAPH_EXPLICIT", 1.0),),
    "UNCLASSIFIED": (("DOMAIN_GROUNDED_IMPLICIT", 1.0),),
}

DEV_ANSWER_MAP = {
    "bool": "boolean", "float": "float", "int": "integer",
    "list": "list", "string": "string", "numeric_string": "string",
    "object": "object", "other": "string",
}

TOOL_BANDS = [("<=9", 0, 9), ("10-12", 10, 12), ("13-18", 13, 18), ("19+", 19, 10**9)]
REF_BANDS = [("0-0.25", 0.0, 0.25), ("0.25-0.5", 0.25, 0.5),
             ("0.5-0.75", 0.5, 0.75), ("0.75-1", 0.75, 1.01)]
DEPTH_KEYS = ("1", "2", "3", "4+")
JOIN_KEYS = ("0", "1", "2", "3", "4+")
MOTIF_KEYS = ("linear", "fan_in", "multi_join", "mixed")


def hamilton(shares: Mapping[str, float], n: int) -> Dict[str, int]:
    """Largest-remainder integer allocation; keys with 0 share stay 0."""
    items = [(k, max(0.0, float(v))) for k, v in shares.items()]
    total = sum(v for _, v in items)
    if n <= 0:
        return {k: 0 for k, _ in items}
    if total <= 0:
        # dump into first key deterministically
        out = {k: 0 for k, _ in items}
        if items:
            out[items[0][0]] = n
        return out
    norm = [(k, v / total) for k, v in items]
    raw = [(k, s * n) for k, s in norm]
    floors = {k: int(x) for k, x in raw}
    rem = n - sum(floors.values())
    order = sorted(raw, key=lambda kv: (-(kv[1] - int(kv[1])), kv[0]))
    for i in range(rem):
        floors[order[i % len(order)][0]] += 1
    return floors


def tool_band(n: int) -> str:
    for name, lo, hi in TOOL_BANDS:
        if lo <= n <= hi:
            return name
    return "19+"


def ref_band(x: float) -> str:
    for name, lo, hi in REF_BANDS:
        if lo <= x < hi:
            return name
    return REF_BANDS[-1][0]


def depth_bucket(d: int) -> str:
    if d <= 1:
        return "1"
    if d == 2:
        return "2"
    if d == 3:
        return "3"
    return "4+"


def join_bucket(j: int) -> str:
    if j >= 4:
        return "4+"
    return str(max(0, j))


def map_motif(primary: str, join_n: int, multi_join: bool) -> str:
    p = (primary or "").upper()
    if p in {"LINEAR_CHAIN"} and join_n == 0:
        return "linear"
    if p in {"MULTI_JOIN", "NESTED_AGGREGATION", "TWO_STAGE_AGGREGATION"} or multi_join:
        return "multi_join"
    if p in {"MIXED_INDEPENDENT_DEPENDENT"}:
        return "mixed"
    if join_n >= 1 or p in {
        "FAN_IN_SINGLE", "FAN_IN_MULTIPLE", "LATE_REFERENCE", "DIAMOND",
        "PARALLEL_THEN_MERGE", "REUSE_EARLY_OUTPUT",
    }:
        return "fan_in"
    return "linear" if join_n == 0 else "fan_in"


def _map_mode_dist(dev: Mapping[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for label, share in dev.items():
        for dest, w in DEV_MODE_MAP.get(label, (("DOMAIN_GROUNDED_IMPLICIT", 1.0),)):
            out[dest] = out.get(dest, 0.0) + float(share) * w
    s = sum(out.values()) or 1.0
    return {k: v / s for k, v in out.items()}


def _map_answer_dist(dev: Mapping[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for label, share in dev.items():
        key = DEV_ANSWER_MAP.get(label, "string")
        out[key] = out.get(key, 0.0) + float(share)
    s = sum(out.values()) or 1.0
    return {k: v / s for k, v in out.items()}


def load_profiles(pilot_out: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    v3 = json.loads((pilot_out / "target_profile_v3.json").read_text(encoding="utf-8"))
    # v2 path is recorded in v3 sources
    v2_path = Path("outputs/profiles/target_profile_v2.json")
    for src in v3.get("sources") or []:
        if src.get("role") == "dev_200_aggregate_profile":
            cand = Path(src["path"])
            if not cand.is_file():
                cand = Path("experiments/targeted_tool_data_factory") / src["path"]
            if cand.is_file():
                v2_path = cand
                break
    if not v2_path.is_file():
        # relative to factory root
        alt = Path("outputs/profiles/target_profile_v2.json")
        if alt.is_file():
            v2_path = alt
    v2 = json.loads(v2_path.read_text(encoding="utf-8"))
    if v2.get("mode") != "PROFILE_SAFE":
        raise ValueError(f"refusing non PROFILE_SAFE profile: {v2.get('mode')}")
    if v2.get("source") not in (None, "nestful_dev_200") and "nestful_dev" not in str(
            v2.get("source", "")):
        # still accept if n_rows=200 and PROFILE_SAFE
        if int(v2.get("n_rows") or 0) != 200:
            raise ValueError(f"unexpected profile source {v2.get('source')}")
    return v3, v2


def build_target_quotas(v2: Mapping[str, Any], v3: Mapping[str, Any],
                        n: int = N_TRAIN) -> Dict[str, Any]:
    """Integer quotas for N. Call counts are hard-fixed; conditionals via Hamilton."""
    cond = v2["conditional"]
    call = dict(CALL_HARD)
    assert sum(call.values()) == n

    answer: Dict[str, Dict[str, int]] = {}
    mode: Dict[str, Dict[str, int]] = {}
    tools: Dict[str, Dict[str, int]] = {}
    depth: Dict[str, Dict[str, int]] = {}
    joins: Dict[str, Dict[str, int]] = {}
    dens: Dict[str, Dict[str, int]] = {}
    motif: Dict[str, Dict[str, int]] = {}
    schema: Dict[str, Dict[str, int]] = {}

    for bucket, bn in call.items():
        answer[bucket] = hamilton(
            _map_answer_dist(cond["P(answer_type|call_count)"].get(bucket) or {"float": 1.0}),
            bn)
        mode[bucket] = hamilton(
            _map_mode_dist(cond["P(query_mode|call_count)"].get(bucket) or {
                "GOAL_BASED_IMPLICIT": 1.0}),
            bn)
        tools[bucket] = hamilton(
            cond["P(offered_tool_count|call_count)"].get(bucket) or {"<=9": 1.0}, bn)
        depth[bucket] = hamilton(
            cond["P(depth|call_count)"].get(bucket) or {"1": 1.0}, bn)
        joins[bucket] = hamilton(
            cond["P(join_count|call_count)"].get(bucket) or {"0": 1.0}, bn)
        dens[bucket] = hamilton(
            cond["P(reference_density|call_count)"].get(bucket) or {"0.25-0.5": 1.0}, bn)
        motif[bucket] = hamilton(
            cond["P(motif|call_count)"].get(bucket) or {"linear": 1.0}, bn)
        schema[bucket] = hamilton(
            cond["P(schema_complexity|call_count)"].get(bucket) or {"high": 1.0}, bn)

    # 6+ internal length shares from pool will be filled later; placeholder from
    # uniform-over-available is computed at selection time.
    return {
        "n": n,
        "dataset_source": "nestful_dev_200",
        "call_count_hard": call,
        "answer_type_source": "P(answer_type|call_count) mapped from nestful_dev_200 (dev_raw, no Pilot4.3 floor)",
        "P(answer_type|call_count)": answer,
        "P(query_mode|call_count)": mode,
        "P(offered_tool_count|call_count)": tools,
        "P(depth|call_count)": depth,
        "P(join_count|call_count)": joins,
        "P(reference_density|call_count)": dens,
        "P(motif|call_count)": motif,
        "P(schema_complexity|call_count)": schema,
        "query_mode_mapping": {
            src: [{"pilot43_mode": d, "weight": w} for d, w in dests]
            for src, dests in DEV_MODE_MAP.items()
        },
        "marginal_query_mode_dev": dict(v2.get("marginal", {}).get("query_mode") or {}),
        "v3_mapped_overall_modes": dict((v3.get("query_mode") or {}).get("overall") or {}),
        "note": (
            "Call-count integers are the hard NESTFUL PROFILE_CORE shares for N=1000. "
            "Conditionals use largest-remainder on nestful_dev_200 PROFILE_SAFE "
            "aggregates. Answer-type uses raw dev (float-only on 4/5/6+), not the "
            "Pilot4.3 non-float floor."
        ),
    }


def write_mode_mapping(out_dir: Path) -> Path:
    payload = {
        "schema": "ttdf.nestful_profile_query_mode_mapping.v1",
        "source_taxonomy": sorted(DEV_MODE_MAP),
        "dest_taxonomy": sorted({
            d for dests in DEV_MODE_MAP.values() for d, _ in dests
        }),
        "mapping": {
            src: [{"pilot43_mode": d, "weight": w} for d, w in dests]
            for src, dests in DEV_MODE_MAP.items()
        },
        "rationale": {
            "GOAL_BASED_IMPLICIT": (
                "NESTFUL's goal-based bucket mixes domain-grounded scenarios and "
                "pure goals; split 70/30 into DOMAIN_GROUNDED_IMPLICIT and "
                "GOAL_BASED_IMPLICIT."
            ),
            "PROCEDURAL_PARTIAL": (
                "Partial procedures name operations without full graphs → mostly "
                "OPERATION_EXPLICIT_GRAPH_IMPLICIT, remainder SEMI_IMPLICIT."
            ),
            "PROCEDURAL_EXPLICIT": "Full procedural graph disclosure → GRAPH_EXPLICIT.",
            "SEMI_IMPLICIT": "Direct 1:1.",
            "UNCLASSIFIED": "Conservative assign to DOMAIN_GROUNDED_IMPLICIT.",
        },
    }
    path = out_dir / "NESTFUL_PROFILE_QUERY_MODE_MAPPING.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
