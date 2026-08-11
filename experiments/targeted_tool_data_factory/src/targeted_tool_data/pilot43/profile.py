"""TargetProfile v3: the dev-200 conditionals Pilot4.3 must reproduce.

The profile is *read*, never invented. Its source is the frozen PROFILE_SAFE
profile built from the NESTFUL dev split (aggregates only, no questions and no
gold programs are copied), and every number this module exposes is either taken
from that file or derived from it by a rule that is written down here.

Three things are added on top of the raw dev conditionals.

* **Query-mode mapping.** The dev classifier uses its own label set
  (``GOAL_BASED_IMPLICIT`` / ``SEMI_IMPLICIT`` / ``PROCEDURAL_*``); Pilot4.3 needs
  the five-mode taxonomy of the spec. The mapping is explicit and the resulting
  target shares are clipped into the spec's bands.
* **Answer-type floor.** Dev has literally 100 % float answers for 4, 5 and 6+
  calls. Reproducing that exactly would make three quarters of PROFILE_CORE a
  single answer type, so a small floor for non-float answers is applied. The
  floor is a *declared deviation*: :func:`answer_type_targets` reports both the
  raw dev distribution and the floored target, and the deviation is printed in
  the quality report rather than hidden.
* **Structural minima for long tasks.** Dev's own 6+ join / multi-join /
  late-reference rates are the floor; the spec's control minima are applied on
  top with :func:`long_horizon_minima`.

The fourth thing v3 adds is a *classifier-matched* structural target. The
pattern distribution inside ``target_profile_v2.json`` came from the older
``profile_v2`` motif classifier, so comparing a Pilot4.3 pattern histogram
against it would compare two different vocabularies. :func:`build_profile_v3`
therefore re-reads the raw NESTFUL dev-200 programs, rebuilds each dependency
DAG from its own argument references, and re-classifies it with the Pilot4.3
invariants via :mod:`.dev_patterns`. When the raw dev split cannot be found the
v2 numbers are kept and flagged with ``classifier_mismatch: true`` rather than
being silently reused as if they were comparable.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from ..repro import sha256_file, sha256_obj
from . import (ANSWER_TYPES, CALL_BUCKETS, PROFILE_CALL_TARGETS, QUERY_MODES,
               QUERY_MODE_TARGETS)
from . import dev_patterns as dp

SCHEMA_VERSION = "ttdf.pilot43.target_profile.v3"

#: dev classifier label -> Pilot4.3 query mode. Dev's "goal based implicit" bucket
#: mixes scenario-grounded and pure-goal phrasing, so it is split by the spec's
#: relative weights instead of being dumped into one mode.
DEV_MODE_MAP: Dict[str, Tuple[Tuple[str, float], ...]] = {
    "GOAL_BASED_IMPLICIT": (("DOMAIN_GROUNDED_IMPLICIT", 0.70),
                            ("GOAL_BASED_IMPLICIT", 0.30)),
    "SEMI_IMPLICIT": (("SEMI_IMPLICIT", 1.0),),
    "PROCEDURAL_PARTIAL": (("OPERATION_EXPLICIT_GRAPH_IMPLICIT", 0.65),
                           ("SEMI_IMPLICIT", 0.35)),
    "PROCEDURAL_EXPLICIT": (("GRAPH_EXPLICIT", 1.0),),
    "UNCLASSIFIED": (("DOMAIN_GROUNDED_IMPLICIT", 1.0),),
}

#: minimum share of non-float answers per call bucket for PROFILE_CORE. Dev is
#: float-only above three calls; a pool with one answer type cannot exercise the
#: boolean/string/list verifiers at all, so this floor is applied and declared.
ANSWER_TYPE_FLOOR: Dict[str, float] = {"2": 0.0, "3": 0.10, "4": 0.14,
                                       "5": 0.14, "6+": 0.16}

#: dev answer-type labels -> Pilot4.3 answer types
DEV_ANSWER_MAP = {"bool": "boolean", "float": "float", "int": "integer",
                  "list": "list", "string": "string", "numeric_string": "string",
                  "object": "object", "other": "string"}

_FACTORY_ROOT = Path(__file__).resolve().parents[3]
_REPO_ROOT = _FACTORY_ROOT.parent.parent
DEFAULT_SOURCE = _FACTORY_ROOT / "data" / "target_profile_v2.json"

# Frozen source rows used to derive the aggregate profile. Keeping both inputs
# in the factory makes Pilot 4.3 generation independent of deleted experiments.
DEFAULT_DEV_ROWS = _FACTORY_ROOT / "data" / "nestful_dev.jsonl"
DEFAULT_OUTPUT = Path("outputs/pilot4_3_nestful_final/target_profile_v3.json")

#: reported vocabulary of the recomputed P(answer_type|call_count)
_ANSWER_KIND_ORDER = ("boolean", "integer", "float", "string", "list", "object")

#: identical to profile_v2's numeric-string test, so a numeric-looking string is
#: bucketed the same way on both sides of the comparison
_NUMERIC_STRING_RE = re.compile(r"-?\d+(\.\d+)?")


class ProfileError(ValueError):
    """The profile source is missing or does not carry the required conditionals."""


def load_source(path: Path | None = None) -> Dict[str, Any]:
    p = Path(path or DEFAULT_SOURCE)
    if not p.is_file():
        raise ProfileError(f"dev profile not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if data.get("mode") != "PROFILE_SAFE":
        raise ProfileError(f"refusing non PROFILE_SAFE profile: {data.get('mode')}")
    for key in ("P(answer_type|call_count)", "P(offered_tool_count|call_count)",
                "P(query_mode|call_count)", "P(motif|call_count)"):
        if key not in (data.get("conditional") or {}):
            raise ProfileError(f"profile source lacks {key}")
    return data


def call_count_targets(source: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Dev call-count shares with the spec's tolerance attached."""
    dist = source["call_count_dist"]
    out: Dict[str, Dict[str, float]] = {}
    for bucket in CALL_BUCKETS:
        share, tol = PROFILE_CALL_TARGETS[bucket]
        observed = float(dist.get(bucket, 0.0))
        out[bucket] = {"dev_share": observed, "target_share": observed,
                       "tolerance_pp": tol, "spec_share": share}
    return out


def answer_type_targets(source: Dict[str, Any]) -> Dict[str, Any]:
    """P(answer_type | call_count) from dev, with the declared non-float floor."""
    cond = source["conditional"]["P(answer_type|call_count)"]
    raw: Dict[str, Dict[str, float]] = {}
    floored: Dict[str, Dict[str, float]] = {}
    for bucket in CALL_BUCKETS:
        dev = cond.get(bucket) or {}
        mapped: Dict[str, float] = {}
        for label, share in dev.items():
            key = DEV_ANSWER_MAP.get(label, "string")
            mapped[key] = round(mapped.get(key, 0.0) + float(share), 5)
        raw[bucket] = mapped
        floored[bucket] = _apply_floor(mapped, ANSWER_TYPE_FLOOR.get(bucket, 0.0))
    return {"dev_raw": raw, "target": floored,
            "floor_applied": ANSWER_TYPE_FLOOR,
            "deviation_note": (
                "dev-200 is float-only for 4, 5 and 6+ calls; a non-float floor is "
                "applied so boolean/string/list/object verifiers are exercised in "
                "PROFILE_CORE. Both distributions are reported.")}


def _apply_floor(dist: Dict[str, float], floor: float) -> Dict[str, float]:
    if floor <= 0:
        return dict(dist)
    non_float = sum(v for k, v in dist.items() if k != "float")
    if non_float >= floor:
        return dict(dist)
    deficit = floor - non_float
    out = {k: v for k, v in dist.items()}
    out["float"] = max(round(out.get("float", 0.0) - deficit, 5), 0.0)
    # spread the deficit over the answer types dev does show at short call counts
    spread = ["boolean", "integer", "string", "list", "object", "category"]
    per = round(deficit / len(spread), 5)
    for key in spread:
        out[key] = round(out.get(key, 0.0) + per, 5)
    total = sum(out.values()) or 1.0
    return {k: round(v / total, 5) for k, v in out.items() if v > 0}


def query_mode_targets(source: Dict[str, Any]) -> Dict[str, Any]:
    """Dev query modes mapped onto the Pilot4.3 taxonomy and clipped to the bands."""
    cond = source["conditional"]["P(query_mode|call_count)"]
    per_bucket: Dict[str, Dict[str, float]] = {}
    for bucket in CALL_BUCKETS:
        dev = cond.get(bucket) or {}
        mapped: Dict[str, float] = {m: 0.0 for m in QUERY_MODES}
        for label, share in dev.items():
            for mode, weight in DEV_MODE_MAP.get(label, DEV_MODE_MAP["UNCLASSIFIED"]):
                mapped[mode] = round(mapped[mode] + float(share) * weight, 5)
        per_bucket[bucket] = _clip_to_bands(mapped)
    overall: Dict[str, float] = {m: 0.0 for m in QUERY_MODES}
    dist = source["call_count_dist"]
    for bucket, shares in per_bucket.items():
        w = float(dist.get(bucket, 0.0))
        for mode, share in shares.items():
            overall[mode] = round(overall[mode] + w * share, 5)
    return {"per_call_bucket": per_bucket, "overall": _clip_to_bands(overall),
            "spec_bands": {m: list(QUERY_MODE_TARGETS[m]) for m in QUERY_MODES}}


def _clip_to_bands(dist: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for mode, share in dist.items():
        lo, hi = QUERY_MODE_TARGETS[mode]
        out[mode] = min(max(share, lo), hi)
    total = sum(out.values()) or 1.0
    scaled = {k: v / total for k, v in out.items()}
    # rescaling can push a mode back out of its band; one corrective pass is enough
    for mode in scaled:
        lo, hi = QUERY_MODE_TARGETS[mode]
        scaled[mode] = min(max(scaled[mode], lo), hi)
    total = sum(scaled.values()) or 1.0
    return {k: round(v / total, 5) for k, v in scaled.items()}


def offered_tool_targets(source: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """P(offered_tool_count | call_count); bands are dev's own bucket labels."""
    cond = source["conditional"]["P(offered_tool_count|call_count)"]
    return {b: {k: float(v) for k, v in (cond.get(b) or {}).items()}
            for b in CALL_BUCKETS if cond.get(b)}


#: dev's own band edges, kept identical so producer and audit bucket the same way
TOOL_BANDS: Tuple[Tuple[str, int, int], ...] = (
    ("<=9", 0, 9), ("10-12", 10, 12), ("13-18", 13, 18), ("19+", 19, 10 ** 6))


def tool_band(count: int) -> str:
    for label, lo, hi in TOOL_BANDS:
        if lo <= count <= hi:
            return label
    return "19+"


def sample_tool_count(band: str, rng) -> int:
    for label, lo, hi in TOOL_BANDS:
        if label == band:
            return rng.randint(max(lo, 6), min(hi, 22))
    return 10


def long_horizon_minima(source: Dict[str, Any]) -> Dict[str, float]:
    """Structural floors for 6+ PROFILE_CORE.

    The gate is the spec's control minimum, not dev's own rate: dev-200 happens to
    have a join and a late reference in *every* 6+ task, and turning an n=22
    observation into a 100 % requirement would force a single topology family --
    the very failure mode Pilot4.2 had. Dev's rates are carried alongside as the
    aspiration and are reported next to the achieved values.
    """
    dev = (source.get("topology_diversity_by_bucket") or {}).get("6+") or {}
    out: Dict[str, Any] = {"join_rate": 0.55, "multi_join_rate": 0.25,
                           "late_reference_rate": 0.50, "fan_out_rate": 0.15,
                           "reuse_rate": 0.12, "minimum_pattern_families": 10}
    out["dev_observed"] = {k: dev.get(k) for k in
                           ("join_rate", "multi_join_rate", "late_reference_rate",
                            "fan_out_rate", "reuse_rate", "n_distinct_topologies")}
    return out


# ── dev-200 recomputed with the Pilot4.3 classifier ──────────────────────
def call_bucket(n: int) -> str:
    """Same bucketing the dev profile used, so buckets line up exactly."""
    if n <= 2:
        return "2"
    if n >= 6:
        return "6+"
    return str(n)


def _count_bucket(value: int) -> str:
    """profile_v2's bucketing for join / fan-out / reuse counts."""
    return str(value) if value <= 3 else "4+"


def answer_kind(value: Any) -> str:
    """Pilot4.3 answer type of a raw dev gold answer.

    Mirrors ``profile_v2._answer_type`` and then :data:`DEV_ANSWER_MAP`, so the
    recomputed conditional is directly comparable with the frozen v2 profile.
    In particular a numeric-looking string stays a string instead of being
    promoted to a number, which is what the v2 mapping does.
    """
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return DEV_ANSWER_MAP["int"]
    if isinstance(value, float):
        return DEV_ANSWER_MAP["float"]
    if isinstance(value, list):
        return DEV_ANSWER_MAP["list"]
    if isinstance(value, dict):
        return DEV_ANSWER_MAP["object"]
    if isinstance(value, str):
        text = value.strip()
        numeric = bool(_NUMERIC_STRING_RE.fullmatch(text))
        return DEV_ANSWER_MAP["numeric_string" if numeric else "string"]
    return DEV_ANSWER_MAP["other"]


def _share(counter: Counter) -> Dict[str, float]:
    total = sum(counter.values()) or 1
    return {k: round(v / total, 5) for k, v in sorted(counter.items())}


def repo_relative(path: Path) -> str:
    """Path as written into the profile.

    Rendered relative to the repo root whenever possible: ``profile_hash``
    covers the source paths, so an absolute path would make the hash depend on
    the checkout location and on how the caller happened to spell the argument.
    """
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(_REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(resolved).replace("\\", "/")


def read_dev_rows(path: Path) -> List[Dict[str, Any]]:
    """Read the raw NESTFUL dev split; no question or program is ever copied."""
    rows: List[Dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def dev_conditionals(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """P(structure | call_count) over dev-200, classified by Pilot4.3 rules.

    Only aggregate shares leave this function; the dev questions and gold
    programs are read, counted and discarded.
    """
    per_bucket: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    n_unresolved_tools = 0
    for row in rows:
        item = dp.classify_row(row)
        item["answer_type"] = answer_kind(row.get("gold_answer"))
        n_unresolved_tools += item["features"]["n_unresolved_tools"]
        per_bucket[call_bucket(item["call_count"])].append(item)

    primary: Dict[str, Dict[str, float]] = {}
    satisfied: Dict[str, Dict[str, float]] = {}
    joins: Dict[str, Dict[str, float]] = {}
    reuse: Dict[str, Dict[str, float]] = {}
    fan_out: Dict[str, Dict[str, float]] = {}
    late: Dict[str, Dict[str, float]] = {}
    tools: Dict[str, Dict[str, float]] = {}
    answers: Dict[str, Dict[str, float]] = {}
    counts: Dict[str, int] = {}
    for bucket in CALL_BUCKETS:
        items = per_bucket.get(bucket, [])
        if not items:
            continue
        n = len(items)
        counts[bucket] = n
        primary[bucket] = _share(Counter(i["primary_pattern"] for i in items))
        # a graph satisfies several invariants at once, so this is a per-bucket
        # marginal rate per pattern and does NOT sum to 1
        satisfied[bucket] = {
            name: round(count / n, 5)
            for name, count in sorted(dp.pattern_counts(items).items())
        }
        joins[bucket] = _share(Counter(_count_bucket(i["join_count"])
                                       for i in items))
        reuse[bucket] = _share(Counter(_count_bucket(i["reuse_count"])
                                       for i in items))
        fan_out[bucket] = _share(Counter(_count_bucket(i["fan_out_count"])
                                         for i in items))
        late[bucket] = _share(Counter("true" if i["late_reference"] else "false"
                                      for i in items))
        tools[bucket] = _share(Counter(tool_band(i["offered_tool_count"])
                                       for i in items))
        answers[bucket] = _share(Counter(i["answer_type"] for i in items))
    return {
        "classifier": dp.CLASSIFIER_ID,
        "n_rows": len(rows),
        "n_by_call_bucket": counts,
        "call_count_dist": _share(Counter(
            call_bucket(len(row.get("output") or [])) for row in rows)),
        "n_calls_with_unresolved_tool_schema": n_unresolved_tools,
        "value_kind_source": (
            "declared output_parameters type of the offered dev tool; dev "
            "programs are not executed, so nodes whose type cannot be resolved "
            "stay 'unknown' and are skipped by the type-transition rule"),
        "P(primary_pattern|call_count)": primary,
        "P(satisfied_pattern|call_count)": satisfied,
        "P(join_count|call_count)": joins,
        "P(reuse_count|call_count)": reuse,
        "P(fan_out_count|call_count)": fan_out,
        "P(late_reference|call_count)": late,
        "P(offered_tool_count|call_count)": tools,
        "P(answer_type|call_count)": answers,
    }


def consistency_with_v2(structural: Dict[str, Any],
                        source: Dict[str, Any]) -> Dict[str, Any]:
    """Self-audit: the classifier-independent conditionals must match v2 exactly.

    The dev edge reconstruction here is the same rule ``profile_v2.build_dag``
    used, so join / fan-out / reuse / offered-tool conditionals recomputed from
    the raw split must equal the frozen ones. Only the *pattern* vocabulary is
    allowed to differ. A False here means the raw split is not the file
    ``target_profile_v2.json`` was built from.
    """
    cond = source.get("conditional") or {}
    checks: Dict[str, Any] = {
        "call_count_dist_matches_v2":
            structural["call_count_dist"] == source.get("call_count_dist"),
    }
    for key in ("P(join_count|call_count)", "P(fan_out_count|call_count)",
                "P(reuse_count|call_count)", "P(offered_tool_count|call_count)"):
        checks[f"{key}_matches_v2"] = structural[key] == cond.get(key)
    mapped_v2 = {
        bucket: _share(Counter({DEV_ANSWER_MAP.get(label, "string"): 0
                                for label in dist}))
        for bucket, dist in (cond.get("P(answer_type|call_count)") or {}).items()
    }
    checks["answer_type_labels_match_v2_after_mapping"] = all(
        set(structural["P(answer_type|call_count)"].get(bucket, {})) == set(labels)
        for bucket, labels in mapped_v2.items())
    checks["note"] = (
        "classifier-independent conditionals are recomputed from the raw split "
        "and compared against the frozen v2 profile; only the pattern "
        "vocabulary is expected to differ")
    return checks


def _v2_pattern_fallback(source: Dict[str, Any], reason: str) -> Dict[str, Any]:
    """Keep the v2 motif distribution, marked as produced by another classifier."""
    return {
        "classifier": "profile_v2.motif_of (NOT the Pilot4.3 classifier)",
        "classifier_mismatch": True,
        "explanation": (
            f"{reason} The structural distribution below is the profile_v2 motif "
            "distribution (linear / fan_in / multi_join / ...), which shares no "
            "vocabulary with the Pilot4.3 15-invariant classifier. A Pilot4.3 "
            "pattern histogram must NOT be compared against it."),
        "P(motif|call_count)": source["conditional"]["P(motif|call_count)"],
        "P(offered_tool_count|call_count)":
            source["conditional"]["P(offered_tool_count|call_count)"],
        "P(answer_type|call_count)":
            source["conditional"]["P(answer_type|call_count)"],
    }


def build_profile_v3(source_path: Path | None = None,
                     write_to: Path | None = None,
                     dev_rows_path: Path | None = None) -> Dict[str, Any]:
    """The Pilot4.3 target profile, optionally written to ``write_to``.

    ``source_path`` stays the first positional argument so existing callers that
    pass a v2 profile path (or nothing) keep working.
    """
    src_path = Path(source_path or DEFAULT_SOURCE)
    source = load_source(src_path)
    dev_path = Path(dev_rows_path or DEFAULT_DEV_ROWS)

    sources: List[Dict[str, Any]] = [{
        "role": "dev_200_aggregate_profile",
        "path": repo_relative(src_path),
        "sha256": sha256_file(src_path),
    }]
    if dev_path.is_file():
        dev_rows = read_dev_rows(dev_path)
        structural = dev_conditionals(dev_rows)
        structural["classifier_mismatch"] = False
        structural["raw_dev_programs_found"] = True
        structural["consistency_with_v2"] = consistency_with_v2(structural, source)
        sources.append({
            "role": "dev_200_raw_programs",
            "path": repo_relative(dev_path),
            "sha256": sha256_file(dev_path),
            "n_rows": len(dev_rows),
        })
        n_rows = len(dev_rows)
    else:
        structural = _v2_pattern_fallback(
            source, f"the raw dev-200 split was not found at {dev_path}.")
        structural["raw_dev_programs_found"] = False
        n_rows = int(source.get("n_rows") or 0)

    profile = {
        "schema_version": SCHEMA_VERSION,
        "sources": sources,
        # kept for backward compatibility with readers of the v2-only profile
        "source_file": repo_relative(src_path),
        "source_sha256": sha256_file(src_path),
        "source_profile_schema": source.get("schema_version"),
        "source_mode": source.get("mode"),
        "source_n_rows": source.get("n_rows"),
        "n_rows": n_rows,
        "dataset_source": source.get("source"),
        "call_count": call_count_targets(source),
        "answer_type": answer_type_targets(source),
        "query_mode": query_mode_targets(source),
        "offered_tool_count": offered_tool_targets(source),
        "tool_bands": [list(b) for b in TOOL_BANDS],
        "long_horizon_minima": long_horizon_minima(source),
        "structural_patterns": structural,
        "motif_conditional": source["conditional"]["P(motif|call_count)"],
        "graph_features": source.get("graph_features"),
        "topology_diversity_by_bucket": source.get("topology_diversity_by_bucket"),
        "answer_types_supported": list(ANSWER_TYPES),
        "answer_kind_order": list(_ANSWER_KIND_ORDER),
        "notes": [
            "diagnostic-500 is never a quota source; only the dev split profile is",
            "no dev question or gold program is copied into Pilot4.3",
            "structural_patterns is the only classifier-matched pattern target; "
            "motif_conditional is retained for continuity with Pilot4.2 reports",
        ],
    }
    profile["profile_hash"] = sha256_obj(
        {k: v for k, v in profile.items() if k != "profile_hash"})
    if write_to is not None:
        path = Path(write_to)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(profile, indent=2, ensure_ascii=False,
                                   default=str), encoding="utf-8")
    return profile


def tv_distance(observed: Dict[str, float], target: Dict[str, float]) -> float:
    keys = set(observed) | set(target)
    return round(0.5 * sum(abs(float(observed.get(k, 0.0)) - float(target.get(k, 0.0)))
                           for k in keys), 5)


def bucket_counts(total: int, shares: Dict[str, float],
                  order: Sequence[str]) -> Dict[str, int]:
    """Largest-remainder allocation so the counts sum to ``total`` exactly."""
    raw = {k: total * float(shares.get(k, 0.0)) for k in order}
    base = {k: int(v) for k, v in raw.items()}
    rest = total - sum(base.values())
    for k in sorted(order, key=lambda k: (-(raw[k] - base[k]), k))[:max(rest, 0)]:
        base[k] += 1
    return base
