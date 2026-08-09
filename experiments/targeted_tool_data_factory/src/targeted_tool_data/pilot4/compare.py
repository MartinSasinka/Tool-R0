"""Phase N: offline pilot3 vs pilot4 comparison.

Structural, linguistic and schema statistics only. No model is run, so nothing
here says anything about NESTFUL accuracy; every row carries the caveat that
makes its interpretation explicit.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from functools import lru_cache
from itertools import combinations, product
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .. import query_realism as qr
from ..profile_v2 import (CALL_BUCKETS, build_profile_v2, featurize,
                          topology_signature)
from ..repro import stamp, write_csv, write_json, write_text
from . import SCHEMA_VERSION
from .cells import BUCKET_CALLS
from .patterns import PatternError, validate_shape

# higher / lower / match: what "better" means for each metric.
# COVERAGE is for structures the factory is asked to cover even though the
# dev-200 profile shows none of them (fan-out and output reuse): matching the
# profile would mean generating zero, so those rows are judged as coverage
# added rather than as a distance to a target of 0.
HIGHER, LOWER, MATCH, NEUTRAL = "higher_is_better", "lower_is_better", \
    "closer_to_target_is_better", "descriptive_only"
COVERAGE = "coverage_required_above_profile"


def _load(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _norm_entropy(counts: Counter) -> float:
    total = sum(counts.values()) or 1
    if len(counts) <= 1:
        return 0.0
    h = -sum((v / total) * math.log(v / total) for v in counts.values())
    return h / math.log(len(counts))


def _tv_distance(a: Dict[str, float], b: Dict[str, float]) -> float:
    keys = set(a) | set(b)
    return round(0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys), 4)


_ENUMERATION_LIMIT = 6


def _count_shapes(n: int) -> int:
    """Every dead-call-free DAG on n calls in topological order, counted exactly."""
    total = 0
    parent_sets = [[tuple(c) for r in range(i + 1)
                    for c in combinations(range(i), r)] for i in range(n)]
    for combo in product(*parent_sets):
        try:
            validate_shape([list(p) for p in combo])
        except PatternError:
            continue
        total += 1
    return total


@lru_cache(maxsize=None)
def admissible_topologies(bucket: str) -> Optional[int]:
    """How many shapes a dead-call-free program can take in this bucket.

    At two calls the only shape is a chain; at three calls only three shapes
    survive, because any other 3-node DAG leaves a call unreachable from the
    sink. Diversity rows for such buckets are capped by arithmetic rather than by
    the generator, and once every shape is covered the top-1 share is pinned by
    the join rate the profile asks for. Returns None for the long buckets, where
    the space is far too large to enumerate and no cap can be claimed.
    """
    calls = BUCKET_CALLS[bucket]
    if max(calls) > _ENUMERATION_LIMIT:
        return None
    return sum(_count_shapes(n) for n in calls)


_CAPPED_KEYS = ("n_distinct_topologies", "top1_topology_share",
                "normalized_entropy")


def dataset_stats(rows: Sequence[Dict[str, Any]], label: str) -> Dict[str, Any]:
    feats = [featurize(r) for r in rows]
    n = len(feats) or 1
    by_bucket: Dict[str, List[Dict[str, Any]]] = {b: [] for b in CALL_BUCKETS}
    for f in feats:
        by_bucket.setdefault(f["call_bucket"], []).append(f)

    def share(pred: Callable[[Dict[str, Any]], bool],
              subset: Optional[Sequence[Dict[str, Any]]] = None) -> float:
        pool = feats if subset is None else subset
        return round(sum(1 for f in pool if pred(f)) / max(len(pool), 1), 4)

    def mean(key: str, subset: Optional[Sequence[Dict[str, Any]]] = None) -> float:
        pool = feats if subset is None else subset
        return round(sum(float(f.get(key, 0)) for f in pool) / max(len(pool), 1), 4)

    tool_names: Counter = Counter()
    output_keys: Counter = Counter()
    tool_combos: Counter = Counter()
    skeletons: Counter = Counter()
    families: Counter = Counter()
    primitives: Counter = Counter()
    capabilities: Counter = Counter()
    schema_compatible = 0
    hard_distractors = 0
    for row in rows:
        tools = row.get("tools") or row.get("offered_tools") or []
        names = sorted(str(t.get("name")) for t in tools if isinstance(t, dict))
        tool_names.update(names)
        for t in tools:
            for key in (t.get("output_parameters") or {}):
                output_keys[str(key)] += 1
        calls = row.get("gold_calls") or row.get("output") or []
        tool_combos["|".join(sorted({str(c.get("name")) for c in calls}))] += 1
        skeletons[row.get("query_skeleton") or _skeleton_fallback(row)] += 1
        families[str(row.get("program_family_id")
                     or (row.get("provenance") or {}).get("semantic_program_family")
                     or row.get("task_id"))] += 1
        for node in (row.get("semantic_program") or {}).get("nodes", []):
            primitives[str(node.get("primitive_id"))] += 1
        capabilities.update(row.get("capability_families") or [])
        if row.get("schema_compatible_distractor_count", 0) > 0:
            schema_compatible += 1
        hard_distractors += int(row.get("hard_distractor_count") or 0)

    topo_by_bucket = {}
    for bucket, rows_b in by_bucket.items():
        if not rows_b:
            continue
        sigs = Counter(topology_signature(f) for f in rows_b)
        topo_by_bucket[bucket] = {
            "n": len(rows_b),
            "n_distinct_topologies": len(sigs),
            "top1_topology_share": round(sigs.most_common(1)[0][1] / len(rows_b), 4),
            "normalized_entropy": round(_norm_entropy(sigs), 4),
            "join_rate": share(lambda f: f["n_joins"] > 0, rows_b),
            "multi_join_rate": share(lambda f: f["n_joins"] > 1, rows_b),
            "fan_out_rate": share(lambda f: f["n_fan_out_nodes"] > 0, rows_b),
            "reuse_rate": share(lambda f: f["n_reused_outputs"] > 0, rows_b),
            "late_reference_rate": share(lambda f: f["n_late_references"] > 0, rows_b),
            "mean_reference_distance": mean("mean_reference_distance", rows_b),
        }

    return {
        "label": label,
        "n": len(rows),
        "call_count_dist": {b: round(len(by_bucket.get(b, [])) / n, 4)
                            for b in CALL_BUCKETS},
        "mean_call_count": mean("call_count"),
        "topology_by_bucket": topo_by_bucket,
        "join_rate": share(lambda f: f["n_joins"] > 0),
        "multi_join_rate": share(lambda f: f["n_joins"] > 1),
        "fan_out_rate": share(lambda f: f["n_fan_out_nodes"] > 0),
        "reuse_rate": share(lambda f: f["n_reused_outputs"] > 0),
        "late_reference_rate": share(lambda f: f["n_late_references"] > 0),
        "mean_reference_distance": mean("mean_reference_distance"),
        "max_reference_distance": mean("max_reference_distance"),
        "mean_depth": mean("depth"),
        "mean_type_transitions": mean("n_type_transitions"),
        "query_mode_dist": {k: round(v / n, 4) for k, v in
                            Counter(f.get("query_mode") for f in feats).items()},
        "mean_operation_explicitness": mean("operation_explicitness"),
        "mean_sequence_leakage": mean("sequence_leakage"),
        "mean_procedural_cue_count": mean("procedural_cue_count"),
        "plan_leak_rate": share(lambda f: f.get("query_mode") == "PROCEDURAL_EXPLICIT"),
        "goal_based_share": share(
            lambda f: f.get("query_mode") == "GOAL_BASED_IMPLICIT"),
        "n_distinct_tool_names": len(tool_names),
        "n_distinct_output_keys": len(output_keys),
        "output_key_entropy": round(_norm_entropy(output_keys), 4),
        "n_distinct_tool_combinations": len(tool_combos),
        "tool_combination_entropy": round(_norm_entropy(tool_combos), 4),
        "n_distinct_primitives": len(primitives),
        "primitive_entropy": round(_norm_entropy(primitives), 4),
        "n_capability_families": len(capabilities),
        "top1_query_skeleton_share": round(
            (skeletons.most_common(1)[0][1] / n) if skeletons else 0.0, 4),
        "top1_program_family_share": round(
            (families.most_common(1)[0][1] / n) if families else 0.0, 4),
        "mean_offered_tool_count": mean("offered_tool_count"),
        "mean_schema_complexity": mean("schema_complexity"),
        "mean_parameter_count": mean("parameter_count"),
        "mean_nested_schema_depth": mean("nested_schema_depth"),
        "schema_compatible_distractor_share": round(schema_compatible / n, 4),
        "mean_hard_distractor_count": round(hard_distractors / n, 4),
    }


def _skeleton_fallback(row: Dict[str, Any]) -> str:
    import re

    q = str(row.get("question") or row.get("input") or "")
    return re.sub(r"\d+(\.\d+)?", "#", q.lower())[:120]


_METRICS: List[Tuple[str, str, str, str]] = [
    # (key, direction, target key in profile or "", caveat)
    ("mean_call_count", MATCH, "mean_call_count",
     "structural only; matching the profile is the goal, not maximisation"),
    ("join_rate", MATCH, "join_rate", "measured over the whole set, not per bucket"),
    ("multi_join_rate", HIGHER, "multi_join_rate",
     "pilot3 weakness; higher is only better up to the profile"),
    ("fan_out_rate", COVERAGE, "fan_out_rate",
     "dev-200 contains no fan-out, so this is deliberate coverage beyond the "
     "measured profile and a known distribution-mismatch risk"),
    ("reuse_rate", COVERAGE, "reuse_rate",
     "dev-200 contains no output reuse, so this is deliberate coverage beyond "
     "the measured profile and a known distribution-mismatch risk"),
    ("late_reference_rate", MATCH, "late_reference_rate", "profile-relative"),
    ("mean_reference_distance", MATCH, "mean_reference_distance",
     "long references are harder to track but also rarer in the benchmark"),
    ("mean_depth", MATCH, "", "descriptive"),
    ("mean_type_transitions", HIGHER, "mean_type_transitions",
     "type transitions exercise conversion capabilities"),
    ("plan_leak_rate", LOWER, "plan_leak_rate",
     "rule-based classifier, not a human judgement of realism"),
    ("goal_based_share", MATCH, "goal_based_share",
     "target taken from the dev-200 profile"),
    ("mean_operation_explicitness", LOWER, "mean_operation_explicitness",
     "lexicon-based; a low value is not proof the task is implicit"),
    ("mean_sequence_leakage", LOWER, "mean_sequence_leakage",
     "only defined when at least two operations are cued"),
    ("mean_procedural_cue_count", LOWER, "mean_procedural_cue_count",
     "counts surface cues, not reasoning difficulty"),
    ("n_distinct_tool_names", HIGHER, "", "raw surface vocabulary size"),
    ("n_distinct_output_keys", HIGHER, "", "affects reference-format diversity"),
    ("output_key_entropy", HIGHER, "", "normalised entropy, comparable across sizes"),
    ("n_distinct_tool_combinations", HIGHER, "", "absolute count scales with n"),
    ("tool_combination_entropy", HIGHER, "", "normalised, size-robust"),
    ("n_distinct_primitives", HIGHER, "", "capability breadth proxy"),
    ("primitive_entropy", HIGHER, "", "normalised, size-robust"),
    ("n_capability_families", HIGHER, "",
     "pilot3 rows predate the taxonomy, so this is 0 for pilot3 by construction"),
    ("top1_query_skeleton_share", LOWER, "", "template concentration"),
    ("top1_program_family_share", LOWER, "", "program-family concentration"),
    ("mean_offered_tool_count", MATCH, "mean_offered_tool_count",
     "environment size, profile-relative"),
    ("mean_schema_complexity", MATCH, "mean_schema_complexity", "profile-relative"),
    ("mean_parameter_count", MATCH, "mean_parameter_count", "profile-relative"),
    ("mean_nested_schema_depth", MATCH, "mean_nested_schema_depth",
     "profile-relative"),
    ("schema_compatible_distractor_share", HIGHER, "",
     "pilot3 has no distractor metadata, so this is 0 for pilot3 by construction"),
    ("mean_hard_distractor_count", HIGHER, "",
     "hardness labels differ between pilots; not directly comparable"),
]


def _direction_verdict(direction: str, p3: Any, p4: Any,
                       target: Any) -> str:
    try:
        a, b = float(p3), float(p4)
    except (TypeError, ValueError):
        return "n/a"
    if direction == HIGHER:
        return "improved" if b > a else ("unchanged" if b == a else "regressed")
    if direction == LOWER:
        return "improved" if b < a else ("unchanged" if b == a else "regressed")
    if direction == COVERAGE:
        return "coverage_added" if b > a else ("unchanged" if b == a
                                               else "coverage_lost")
    if direction == MATCH and target not in (None, ""):
        try:
            t = float(target)
        except (TypeError, ValueError):
            return "n/a"
        return ("closer_to_target" if abs(b - t) < abs(a - t)
                else "unchanged" if abs(b - t) == abs(a - t) else "further_from_target")
    return "descriptive"


DEFAULT_SOURCES = {
    "pilot3": [
        "reports/pilot3_provenance/_git_revisions/train_grpo_pilot3@e83f57de.jsonl",
        "outputs/selected/export_pilot3/train_grpo_pilot3.jsonl",
    ],
    "pilot4_profile_safe": ["outputs/pilot4_profile_safe/canonical.jsonl"],
}


def _resolve(module_root: Path, name: str, explicit: Optional[str]) -> Optional[Path]:
    if explicit and Path(explicit).exists():
        return Path(explicit)
    for rel in DEFAULT_SOURCES.get(name, []):
        p = module_root / rel
        if p.exists():
            return p
    return None


def run_comparison(repo_root: Path, out_dir: Path, *, baseline: str,
                   candidate: str, nestful_dev: Optional[Path] = None,
                   baseline_path: Optional[str] = None,
                   candidate_path: Optional[str] = None,
                   cli_args: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    module_root = repo_root / "experiments" / "targeted_tool_data_factory"
    p3_path = _resolve(module_root, baseline, baseline_path)
    p4_path = _resolve(module_root, candidate, candidate_path)
    if p3_path is None or p4_path is None:
        raise FileNotFoundError(
            f"could not resolve baseline={baseline} ({p3_path}) or "
            f"candidate={candidate} ({p4_path})")

    p3_rows, p4_rows = _load(p3_path), _load(p4_path)
    # compare like with like: pilot3 train-600 against the pilot4 train split
    p4_train = [r for r in p4_rows if r.get("split") == "train"] or p4_rows

    p3 = dataset_stats(p3_rows, f"{baseline}:{p3_path.name}")
    p4 = dataset_stats(p4_train, f"{candidate}:train")
    p4_all = dataset_stats(p4_rows, f"{candidate}:selected_{len(p4_rows)}")

    dev_stats: Dict[str, Any] = {}
    profile: Dict[str, Any] = {}
    if nestful_dev and Path(nestful_dev).exists():
        dev_rows = _load(Path(nestful_dev))
        dev_stats = dataset_stats(dev_rows, "nestful_dev_200")
        profile = build_profile_v2(dev_rows, source="nestful_dev_200",
                                   mode="PROFILE_SAFE")

    rows: List[Dict[str, Any]] = []
    for key, direction, target_key, caveat in _METRICS:
        target = dev_stats.get(target_key) if target_key else ""
        rows.append({
            "metric": key,
            "pilot3": p3.get(key),
            "pilot4_train600": p4.get(key),
            "pilot4_selected": p4_all.get(key),
            "target_dev200": target,
            "direction": direction,
            "verdict": _direction_verdict(direction, p3.get(key), p4.get(key), target),
            "caveat": caveat,
        })

    # per-bucket conditional topology rows
    for bucket in CALL_BUCKETS:
        b3 = p3["topology_by_bucket"].get(bucket, {})
        b4 = p4["topology_by_bucket"].get(bucket, {})
        bd = (dev_stats.get("topology_by_bucket") or {}).get(bucket, {})
        n_admissible = admissible_topologies(bucket)
        capped = (n_admissible is not None
                  and int(b4.get("n_distinct_topologies") or 0) >= n_admissible)
        for key, direction in (("n_distinct_topologies", HIGHER),
                               ("top1_topology_share", LOWER),
                               ("normalized_entropy", HIGHER),
                               ("join_rate", MATCH), ("multi_join_rate", HIGHER),
                               ("fan_out_rate", COVERAGE), ("reuse_rate", COVERAGE),
                               ("late_reference_rate", MATCH),
                               ("mean_reference_distance", MATCH)):
            verdict = _direction_verdict(direction, b3.get(key), b4.get(key),
                                         bd.get(key, ""))
            caveat = ("distinct-topology counts scale with the number of "
                      "rows in the bucket; compare the entropy row too"
                      if key == "n_distinct_topologies"
                      else "conditional on the call bucket")
            if capped and key in _CAPPED_KEYS:
                verdict = "structurally_capped"
                caveat = (f"all {n_admissible} dead-call-free topologies that exist "
                          f"at {bucket} calls are covered, so the remaining spread "
                          "is fixed by the join rate the profile asks for; pilot3 "
                          "scores lower on the share only because its join rate "
                          "overshot that target")
            rows.append({
                "metric": f"bucket[{bucket}].{key}",
                "pilot3": b3.get(key), "pilot4_train600": b4.get(key),
                "pilot4_selected": (p4_all["topology_by_bucket"]
                                    .get(bucket, {}).get(key)),
                "target_dev200": bd.get(key, ""),
                "direction": direction,
                "verdict": verdict,
                "caveat": caveat,
                "n_admissible_topologies": ("" if n_admissible is None
                                            else n_admissible),
            })

    distances = {}
    if dev_stats:
        distances = {
            "call_count_tv_pilot3": _tv_distance(p3["call_count_dist"],
                                                 dev_stats["call_count_dist"]),
            "call_count_tv_pilot4": _tv_distance(p4["call_count_dist"],
                                                 dev_stats["call_count_dist"]),
            "query_mode_tv_pilot3": _tv_distance(p3["query_mode_dist"],
                                                 dev_stats["query_mode_dist"]),
            "query_mode_tv_pilot4": _tv_distance(p4["query_mode_dist"],
                                                 dev_stats["query_mode_dist"]),
        }

    payload = {
        "schema_version": SCHEMA_VERSION,
        "baseline": {"name": baseline, "path": str(p3_path), "n": len(p3_rows)},
        "candidate": {"name": candidate, "path": str(p4_path), "n": len(p4_rows)},
        "profile_reference": {"path": str(nestful_dev) if nestful_dev else None,
                              "n": dev_stats.get("n", 0)},
        "stats": {"pilot3": p3, "pilot4_train600": p4,
                  "pilot4_selected": p4_all, "nestful_dev_200": dev_stats},
        "distribution_distances": distances,
        "metrics": rows,
        "profile_query_realism": (profile.get("query_realism") or {}),
        "provenance": stamp(repo_root, schema_version=SCHEMA_VERSION,
                            cli_args=cli_args, input_paths=[p3_path, p4_path]),
    }
    write_json(out_dir / "PILOT3_VS_PILOT4_DATA_AUDIT.json", payload)
    write_csv(out_dir / "PILOT3_VS_PILOT4_METRICS.csv", rows,
              ["metric", "pilot3", "pilot4_train600", "pilot4_selected",
               "target_dev200", "direction", "verdict",
               "n_admissible_topologies", "caveat"])
    write_text(out_dir / "PILOT3_VS_PILOT4_DATA_AUDIT.md",
               _markdown(payload))
    return {"n_metrics": len(rows), "out_dir": str(out_dir), "payload": payload}


def _markdown(payload: Dict[str, Any]) -> str:
    p3 = payload["stats"]["pilot3"]
    p4 = payload["stats"]["pilot4_train600"]
    dev = payload["stats"]["nestful_dev_200"]
    lines = [
        "# PILOT3_VS_PILOT4_DATA_AUDIT", "",
        "Offline dataset statistics only. No model was run, so nothing in this "
        "report predicts NESTFUL accuracy.", "",
        f"- baseline: `{payload['baseline']['path']}` "
        f"(n={payload['baseline']['n']})",
        f"- candidate: `{payload['candidate']['path']}` "
        f"(n={payload['candidate']['n']}, train split n={p4['n']})",
        f"- profile reference: dev-200 (n={dev.get('n', 0)}), aggregates only",
        "",
        "## Distribution distance to the profile", "",
        "| distribution | pilot3 TV | pilot4 TV |", "|---|---|---|",
    ]
    d = payload["distribution_distances"]
    if d:
        lines += [f"| call count | {d['call_count_tv_pilot3']} | "
                  f"{d['call_count_tv_pilot4']} |",
                  f"| query mode | {d['query_mode_tv_pilot3']} | "
                  f"{d['query_mode_tv_pilot4']} |"]
    lines += ["", "## Metrics", "",
              "| metric | pilot3 | pilot4 train-600 | dev-200 target | direction | "
              "verdict |", "|---|---|---|---|---|---|"]
    for row in payload["metrics"]:
        lines.append(
            f"| `{row['metric']}` | {row['pilot3']} | {row['pilot4_train600']} | "
            f"{row['target_dev200']} | {row['direction']} | {row['verdict']} |")
    lines += ["", "## Caveats", ""]
    seen = set()
    for row in payload["metrics"]:
        c = row["caveat"]
        if c and c not in seen:
            seen.add(c)
            lines.append(f"- {c}")
    lines += ["",
              "- diagnostic-500 was not used anywhere in this comparison.",
              "- pilot3 rows predate several pilot4 fields; those metrics are "
              "marked as zero by construction rather than as a regression.", ""]
    return "\n".join(lines)
