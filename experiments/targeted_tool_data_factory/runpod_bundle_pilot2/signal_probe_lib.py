#!/usr/bin/env python3
"""Pure analysis core of the Pilot2 inference-only signal probe.

Everything in this module is deterministic, dependency-free (stdlib only) and
free of torch/vLLM imports, so the whole decision layer — group metrics, P3
selection, Phase-1 selection, reward-ordering validation, verdict — is unit
testable on a laptop while the GPU workers only produce raw rollout records.

Split of responsibility:
    signal_probe_worker.py   GPU: rollouts -> rollout records (this module's
                             ``derive_rollout_metrics`` shapes each record)
    signal_probe_analyze.py  CPU: rollout records -> groups, report, subsets
    signal_probe_lib.py      the logic both of them share (here)
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SCHEMA_VERSION = 1

# A group whose reward spread is below this has no GRPO gradient at all:
# advantages are computed from between-completion contrast, so an identical
# reward for every completion means the group contributes nothing.
DEAD_EPS = 1e-6
# Numeric tolerance when comparing a predicted argument/observation to gold.
VALUE_TOL = 1e-3

# Verdict thresholds (spec §8).
DEAD_PASS_MAX = 0.50
DEAD_CONDITIONAL_MAX = 0.70
# Reward ordering is judged on Pareto-comparable rollout pairs only, so a
# non-zero rate is a real contradiction, not a modelling artefact. A small
# tolerance absorbs float noise in reward reconstruction.
ORDERING_INVERSION_TOL = 0.02
ORDERING_MIN_PAIRS = 20
# CONDITIONAL requires that the process component really does separate
# trajectories even where the terminal outcome does not.
PROCESS_VARIANCE_MIN_RATE = 0.10

FAILURE_CLASSES: Tuple[str, ...] = (
    "success",
    "prompt_overflow",
    "clipped_completion",
    "parse_error",
    "no_tool_call",
    "unknown_tool",
    "invalid_reference",
    "arg_key_error",
    "arg_type_error",
    "arg_range_error",
    "exec_division_by_zero",
    "exec_error",
    "wrong_first_tool",
    "too_few_calls",
    "too_many_calls",
    "wrong_args",
    "wrong_final_answer",
)


# ─────────────────────────────────────────────────────────── hashing / cache ──

def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str)


def sha256_str(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def content_hash(obj: Any) -> str:
    return sha256_str(canonical_json(obj))


def rollout_cache_key(*, row_hash: str, task_id: str, rollout_idx: int,
                      phase: str, probe_signature: Dict[str, Any]) -> str:
    """Content-addressed identity of one rollout.

    Keyed by the task CONTENT (not just its id) plus every knob that can change
    what the model would produce, so ``--resume`` reuses a record only when it
    is genuinely the same experiment. Changing the model, decoding, reward arm
    or registry invalidates the cache automatically.
    """
    return content_hash({
        "schema": SCHEMA_VERSION,
        "row_hash": row_hash,
        "task_id": task_id,
        "rollout_idx": int(rollout_idx),
        "phase": phase,
        "probe": probe_signature,
    })


# ───────────────────────────────────────────────────────────────── metadata ──

def extract_task_meta(row: Dict[str, Any]) -> Dict[str, Any]:
    """Structural metadata of one pilot2 task, tolerant of top-level vs
    ``provenance`` placement."""
    prov = row.get("provenance") or {}
    gold_calls = row.get("gold_calls") or row.get("output") or []
    call_count = row.get("num_calls")
    if not isinstance(call_count, int):
        call_count = len(gold_calls)
    return {
        "task_id": str(row.get("sample_id") or row.get("task_id") or ""),
        "track": row.get("track") or prov.get("track") or "?",
        "call_count": int(call_count),
        "motif": row.get("motif_type") or prov.get("motif_type") or "?",
        "answer_type": row.get("answer_type") or prov.get("answer_type") or "?",
        "generation_cell": prov.get("generation_cell_id") or "?",
        "target_skill": prov.get("target_skill") or "?",
        "target_failure_mode": prov.get("target_failure_mode") or "?",
    }


def structural_key(meta: Dict[str, Any]) -> str:
    """Primary axis whose distribution the Phase-1 subset must preserve."""
    return f"{meta.get('track', '?')}|{meta.get('call_count', 0)}call"


# ────────────────────────────────────────────────────── argument comparison ──

def values_equal(a: Any, b: Any, *, tol: float = VALUE_TOL) -> bool:
    """Structural equality with a numeric tolerance (bools stay strict)."""
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b or a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= tol
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(values_equal(x, y, tol=tol)
                                       for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(values_equal(a[k], b[k], tol=tol)
                                        for k in a)
    return a == b


def _json_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (list, tuple)):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "null"


def arg_error_counts(pred_args: Optional[Dict[str, Any]],
                     gold_args: Optional[Dict[str, Any]]) -> Dict[str, int]:
    """Key / type / value error counts of one call against its gold call.

    A key error is a missing or extra argument name; a type error is the right
    name with the wrong JSON type; a value error is the right name and type
    with the wrong value. The three are mutually exclusive per argument.
    """
    pred_args = pred_args or {}
    gold_args = gold_args or {}
    missing = set(gold_args) - set(pred_args)
    extra = set(pred_args) - set(gold_args)
    type_errors = 0
    value_errors = 0
    for key in set(pred_args) & set(gold_args):
        if _json_type(pred_args[key]) != _json_type(gold_args[key]):
            type_errors += 1
        elif not values_equal(pred_args[key], gold_args[key]):
            value_errors += 1
    return {
        "key_errors": len(missing) + len(extra),
        "missing_keys": sorted(missing),
        "extra_keys": sorted(extra),
        "type_errors": type_errors,
        "value_errors": value_errors,
    }


def correct_prefix_len(pred_calls: Sequence[Dict[str, Any]],
                       gold_calls: Sequence[Dict[str, Any]]) -> int:
    """Length of the leading run of calls that match gold exactly.

    Both sides carry RESOLVED arguments (references already substituted by the
    executor), so a correct call counts as correct regardless of which ``$varN``
    label style the model chose.
    """
    n = 0
    for pred, gold in zip(pred_calls, gold_calls):
        if (pred.get("name") or "") != (gold.get("name") or ""):
            break
        errs = arg_error_counts(pred.get("arguments"), gold.get("arguments"))
        if errs["key_errors"] or errs["type_errors"] or errs["value_errors"]:
            break
        n += 1
    return n


# ─────────────────────────────────────────────────── failure classification ──

def classify_failure(rec: Dict[str, Any]) -> str:
    """Single mutually-exclusive failure label for one rollout.

    Ordered most-proximate-cause first: a trajectory that never produced a
    parsable call is a parse failure, not a "wrong args" failure, even though
    its arguments are trivially wrong too.
    """
    if rec.get("success"):
        return "success"
    if rec.get("prompt_overflow"):
        return "prompt_overflow"
    if rec.get("clipped"):
        return "clipped_completion"
    if rec.get("parse_error") or rec.get("stop_reason") == "parse_fail":
        return "parse_error"
    n_pred = int(rec.get("n_pred_calls") or 0)
    if n_pred == 0:
        return "no_tool_call"

    errors = [str(c.get("error")) for c in (rec.get("resolved_calls") or [])
              if c.get("error")]
    if errors:
        err = errors[0]
        if "unknown_tool" in err or "unregistered_tool" in err:
            return "unknown_tool"
        if "unresolved_variable" in err or "unresolved_field" in err:
            return "invalid_reference"
        if "missing_required_argument" in err or "unknown_argument" in err:
            return "arg_key_error"
        if "argument_type_mismatch" in err or "invalid_arguments_type" in err:
            return "arg_type_error"
        if ("below_min" in err or "above_max" in err
                or "array_too_short" in err or "array_element_type" in err):
            return "arg_range_error"
        if "division_by_zero" in err:
            return "exec_division_by_zero"
        return "exec_error"

    if not rec.get("first_tool_correct"):
        return "wrong_first_tool"
    gold_n = int(rec.get("call_count") or 0)
    if gold_n and n_pred < gold_n:
        return "too_few_calls"
    if gold_n and n_pred > gold_n:
        return "too_many_calls"
    if (int(rec.get("arg_key_errors") or 0) or int(rec.get("arg_type_errors") or 0)
            or int(rec.get("arg_value_errors") or 0)):
        return "wrong_args"
    return "wrong_final_answer"


def derive_rollout_metrics(rec: Dict[str, Any],
                           gold_calls: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Fill the derived, gold-comparing fields of a rollout record in place.

    ``rec`` must already carry what only the GPU worker can know (raw
    completion, parsed/resolved calls, observations, reward components).
    """
    resolved = rec.get("resolved_calls") or []
    pred_resolved = [{"name": c.get("name"), "arguments": c.get("arguments_resolved") or {}}
                     for c in resolved]
    gold_resolved = [{"name": c.get("name"), "arguments": c.get("arguments") or {}}
                     for c in gold_calls]

    first_correct = bool(pred_resolved) and bool(gold_resolved) and (
        (pred_resolved[0].get("name") or "") == (gold_resolved[0].get("name") or ""))

    key_e = type_e = value_e = 0
    per_call: List[Dict[str, Any]] = []
    for i, pred in enumerate(pred_resolved):
        gold = gold_resolved[i] if i < len(gold_resolved) else None
        if gold is None:
            per_call.append({"position": i, "beyond_gold": True})
            continue
        errs = arg_error_counts(pred.get("arguments"), gold.get("arguments"))
        key_e += errs["key_errors"]
        type_e += errs["type_errors"]
        value_e += errs["value_errors"]
        per_call.append({
            "position": i,
            "name_ok": (pred.get("name") or "") == (gold.get("name") or ""),
            **errs,
        })

    prefix = correct_prefix_len(pred_resolved, gold_resolved)
    gold_n = len(gold_resolved) or 1

    rec["first_tool_correct"] = first_correct
    rec["correct_prefix_len"] = prefix
    rec["correct_prefix_frac"] = round(prefix / gold_n, 6)
    rec["arg_key_errors"] = key_e
    rec["arg_type_errors"] = type_e
    rec["arg_value_errors"] = value_e
    rec["per_call_errors"] = per_call
    rec["failure_class"] = classify_failure(rec)
    return rec


# ─────────────────────────────────────────────────────── objective quality ──

def objective_quality(rec: Dict[str, Any]) -> Tuple[float, ...]:
    """Reward-independent quality vector, used ONLY to audit reward ordering.

    Every component is derived from the executor and the gold trace, never from
    the reward function, so agreement between this vector and the reward is
    real evidence and not a tautology. Higher is better on every axis.
    """
    return (
        1.0 if rec.get("success") else 0.0,
        float(rec.get("correct_prefix_len") or 0),
        float(rec.get("n_successful_calls") or 0),
        -float((rec.get("arg_key_errors") or 0) + (rec.get("arg_type_errors") or 0)
               + (rec.get("arg_value_errors") or 0)),
        1.0 if not rec.get("parse_error") and not rec.get("clipped") else 0.0,
    )


def pareto_compare(a: Tuple[float, ...], b: Tuple[float, ...]) -> int:
    """1 if ``a`` dominates ``b``, -1 if dominated, 0 if incomparable/equal."""
    ge = all(x >= y for x, y in zip(a, b))
    le = all(x <= y for x, y in zip(a, b))
    if ge and not le:
        return 1
    if le and not ge:
        return -1
    return 0


def reward_ordering_audit(records_by_task: Dict[str, List[Dict[str, Any]]], *,
                          tol: float = ORDERING_INVERSION_TOL,
                          min_pairs: int = ORDERING_MIN_PAIRS) -> Dict[str, Any]:
    """Does a higher reward really mean an objectively better trajectory?

    Only Pareto-comparable pairs inside the same group are judged: if rollout A
    is at least as good as B on every objective axis and strictly better on one,
    then A's reward must not be lower than B's. Anything else is an inversion.
    """
    comparable = 0
    inversions = 0
    concordant = 0
    ties = 0
    examples: List[Dict[str, Any]] = []
    for task_id, recs in sorted(records_by_task.items()):
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                a, b = recs[i], recs[j]
                dom = pareto_compare(objective_quality(a), objective_quality(b))
                if dom == 0:
                    continue
                better, worse = (a, b) if dom == 1 else (b, a)
                comparable += 1
                delta = float(better.get("episode_reward") or 0.0) - \
                    float(worse.get("episode_reward") or 0.0)
                if delta < -DEAD_EPS:
                    inversions += 1
                    if len(examples) < 10:
                        examples.append({
                            "task_id": task_id,
                            "better_rollout": better.get("rollout_idx"),
                            "worse_rollout": worse.get("rollout_idx"),
                            "better_reward": better.get("episode_reward"),
                            "worse_reward": worse.get("episode_reward"),
                            "better_quality": list(objective_quality(better)),
                            "worse_quality": list(objective_quality(worse)),
                        })
                elif abs(delta) <= DEAD_EPS:
                    ties += 1
                else:
                    concordant += 1

    rate = round(inversions / comparable, 6) if comparable else None
    if comparable < min_pairs:
        valid: Optional[bool] = None
    else:
        valid = bool(rate is not None and rate <= tol)
    return {
        "comparable_pairs": comparable,
        "concordant_pairs": concordant,
        "tied_pairs": ties,
        "inversions": inversions,
        "inversion_rate": rate,
        "inversion_tolerance": tol,
        "min_pairs_required": min_pairs,
        "ordering_valid": valid,
        "inversion_examples": examples,
        "note": ("Only Pareto-comparable within-group pairs are judged: an "
                 "inversion means a strictly better trajectory received a "
                 "strictly lower reward."),
    }


# ────────────────────────────────────────────────────────────── statistics ──

def shannon_entropy(labels: Iterable[Any]) -> float:
    counts = Counter(labels)
    n = sum(counts.values())
    if n <= 0:
        return 0.0
    return round(-sum((c / n) * math.log2(c / n) for c in counts.values()), 6)


def _mean(values: Sequence[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    return round(sum(vals) / len(vals), 6) if vals else None


def build_group(meta: Dict[str, Any], records: List[Dict[str, Any]], *,
                phase: str, eps: float = DEAD_EPS) -> Dict[str, Any]:
    """Per-group signal metrics (spec §3)."""
    recs = sorted(records, key=lambda r: int(r.get("rollout_idx") or 0))
    rewards = [float(r.get("episode_reward") or 0.0) for r in recs]
    successes = [bool(r.get("success")) for r in recs]
    failure_classes = [str(r.get("failure_class") or "?") for r in recs]
    n = len(recs)
    success_count = sum(1 for s in successes if s)
    r_min = min(rewards) if rewards else 0.0
    r_max = max(rewards) if rewards else 0.0
    r_range = round(r_max - r_min, 9)
    dead = r_range <= eps
    process_scores = [float(r.get("process_reward") or 0.0) for r in recs]
    process_range = round((max(process_scores) - min(process_scores)) if process_scores else 0.0, 9)

    return {
        "task_id": meta["task_id"],
        "phase": phase,
        "track": meta.get("track"),
        "call_count": meta.get("call_count"),
        "motif": meta.get("motif"),
        "answer_type": meta.get("answer_type"),
        "generation_cell": meta.get("generation_cell"),
        "structural_key": structural_key(meta),
        "n_rollouts": n,
        "success_count": success_count,
        "success_rate": round(success_count / n, 6) if n else None,
        "reward_min": round(r_min, 6),
        "reward_max": round(r_max, 6),
        "reward_range": r_range,
        "reward_mean": _mean(rewards),
        "reward_values": [round(r, 6) for r in rewards],
        "unique_rewards": len({round(r, 6) for r in rewards}),
        "process_reward_range": process_range,
        "dead_group": dead,
        "terminal_mixed": bool(0 < success_count < n),
        "process_only_mixed": bool(success_count == 0 and r_range > eps),
        "all_failure_dead": bool(success_count == 0 and dead),
        "all_success_dead": bool(success_count == n and n > 0 and dead),
        "failure_entropy_bits": shannon_entropy(failure_classes),
        "failure_classes": dict(sorted(Counter(failure_classes).items())),
        "mean_correct_prefix_frac": _mean([r.get("correct_prefix_frac") for r in recs]),
        "first_tool_correct_rate": _mean([1.0 if r.get("first_tool_correct") else 0.0
                                          for r in recs]),
        "mean_arg_errors": _mean([(r.get("arg_key_errors") or 0)
                                  + (r.get("arg_type_errors") or 0)
                                  + (r.get("arg_value_errors") or 0) for r in recs]),
    }


def success_bucket(group: Dict[str, Any]) -> str:
    n = int(group.get("n_rollouts") or 0)
    s = int(group.get("success_count") or 0)
    if s == 0:
        return "none"
    if n and s >= n:
        return "all"
    return "partial"


def distribution_by(groups: Sequence[Dict[str, Any]], field: str) -> Dict[str, Any]:
    """Group counts and signal rates bucketed by one metadata field."""
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for g in groups:
        buckets[str(g.get(field))].append(g)
    out: Dict[str, Any] = {}
    for key, gs in sorted(buckets.items()):
        out[key] = {
            "n_groups": len(gs),
            "dead_rate": _mean([1.0 if g["dead_group"] else 0.0 for g in gs]),
            "terminal_mixed_rate": _mean([1.0 if g["terminal_mixed"] else 0.0 for g in gs]),
            "process_only_mixed_rate": _mean([1.0 if g["process_only_mixed"] else 0.0
                                              for g in gs]),
            "mean_success_rate": _mean([g.get("success_rate") for g in gs]),
            "mean_reward_range": _mean([g.get("reward_range") for g in gs]),
        }
    return out


def reward_range_histogram(groups: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    """Reward-range distribution — how much contrast each group actually has."""
    edges = [(0.0, "0 (dead)"), (1e-6, "(0, 0.01]"), (0.01, "(0.01, 0.05]"),
             (0.05, "(0.05, 0.15]"), (0.15, "(0.15, 0.40]"), (0.40, "> 0.40")]
    hist: Dict[str, int] = {label: 0 for _, label in edges}
    for g in groups:
        r = float(g.get("reward_range") or 0.0)
        if r <= DEAD_EPS:
            hist["0 (dead)"] += 1
        elif r <= 0.01:
            hist["(0, 0.01]"] += 1
        elif r <= 0.05:
            hist["(0.01, 0.05]"] += 1
        elif r <= 0.15:
            hist["(0.05, 0.15]"] += 1
        elif r <= 0.40:
            hist["(0.15, 0.40]"] += 1
        else:
            hist["> 0.40"] += 1
    return hist


def phase_summary(groups: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(groups)
    if not n:
        return {"n_groups": 0}
    buckets = Counter(success_bucket(g) for g in groups)
    return {
        "n_groups": n,
        "n_rollouts": sum(int(g.get("n_rollouts") or 0) for g in groups),
        "dead_group_rate": round(sum(1 for g in groups if g["dead_group"]) / n, 6),
        "terminal_mixed_rate": round(sum(1 for g in groups if g["terminal_mixed"]) / n, 6),
        "process_only_mixed_rate": round(
            sum(1 for g in groups if g["process_only_mixed"]) / n, 6),
        "all_failure_dead_rate": round(
            sum(1 for g in groups if g["all_failure_dead"]) / n, 6),
        "all_success_dead_rate": round(
            sum(1 for g in groups if g["all_success_dead"]) / n, 6),
        "success_bucket_counts": {
            "zero_success": buckets.get("none", 0),
            "partial_success": buckets.get("partial", 0),
            "all_success": buckets.get("all", 0),
        },
        "success_bucket_rates": {
            "zero_success": round(buckets.get("none", 0) / n, 6),
            "partial_success": round(buckets.get("partial", 0) / n, 6),
            "all_success": round(buckets.get("all", 0) / n, 6),
        },
        "mean_success_rate": _mean([g.get("success_rate") for g in groups]),
        "mean_reward_range": _mean([g.get("reward_range") for g in groups]),
        "mean_failure_entropy_bits": _mean([g.get("failure_entropy_bits") for g in groups]),
        "reward_range_histogram": reward_range_histogram(groups),
    }


# ────────────────────────────────────────────────────────────── selection ──

def _round_robin_by_stratum(groups: Sequence[Dict[str, Any]],
                            limit: int) -> List[Dict[str, Any]]:
    """Take up to ``limit`` groups, cycling strata so no single cell dominates.

    Diversity is enforced across all four structural axes at once by keying on
    (track, call_count, motif, answer_type).
    """
    strata: Dict[Tuple, List[Dict[str, Any]]] = defaultdict(list)
    for g in sorted(groups, key=lambda x: str(x.get("task_id"))):
        key = (g.get("track"), g.get("call_count"), g.get("motif"), g.get("answer_type"))
        strata[key].append(g)
    order = sorted(strata)
    picked: List[Dict[str, Any]] = []
    idx = 0
    while len(picked) < limit and any(strata[k] for k in order):
        key = order[idx % len(order)]
        if strata[key]:
            picked.append(strata[key].pop(0))
        idx += 1
    return picked[:limit]


def select_p3_tasks(groups: Sequence[Dict[str, Any]], *, limit: int = 64
                    ) -> Dict[str, Any]:
    """Pick the boundary tasks worth re-probing at 8 rollouts (spec §4).

    Priority: groups the model already sometimes solves (1/4-3/4), then
    all-failure groups that at least separate on process reward, then a
    stratified sample of dead groups as a control. Diversity across track /
    call count / motif / answer type is preserved inside every bucket.
    """
    boundary, all_fail_spread, dead = [], [], []
    for g in groups:
        n = int(g.get("n_rollouts") or 0)
        s = int(g.get("success_count") or 0)
        if n and 0 < s < n:
            boundary.append(g)
        elif s == 0 and float(g.get("reward_range") or 0.0) > DEAD_EPS:
            all_fail_spread.append(g)
        elif g.get("dead_group"):
            dead.append(g)

    selected: List[Dict[str, Any]] = []
    reasons: Dict[str, str] = {}

    # A stratified dead sample is a REQUIRED part of P3 (spec §4): it is the
    # control that shows whether a dead group stays dead at 8 rollouts, and it
    # is the audit material for the report. Reserve its slots up front,
    # otherwise a large boundary bucket fills the whole budget and the probe
    # can no longer distinguish "genuinely dead" from "under-sampled at 4".
    dead_reserve = min(len(dead), max(1, limit // 8)) if dead else 0

    for name, bucket, room in (
        ("p2_boundary_success", boundary, limit - dead_reserve),
        ("all_failure_reward_spread", all_fail_spread, limit - dead_reserve),
        ("dead_group_stratified_control", dead, limit),
    ):
        room = min(max(0, room - len(selected)), limit - len(selected))
        for g in _round_robin_by_stratum(bucket, room):
            selected.append(g)
            reasons[str(g["task_id"])] = name

    return {
        "task_ids": [str(g["task_id"]) for g in selected],
        "reasons": reasons,
        "bucket_sizes": {
            "p2_boundary_success": len(boundary),
            "all_failure_reward_spread": len(all_fail_spread),
            "dead_group_stratified_control": len(dead),
        },
        "selected_bucket_counts": dict(sorted(Counter(reasons.values()).items())),
        "dead_control_slots_reserved": dead_reserve,
        "limit": limit,
    }


def allocate_proportional(counts: Dict[str, int], total: int) -> Dict[str, int]:
    """Largest-remainder allocation of ``total`` across ``counts`` shares."""
    grand = sum(counts.values())
    if grand <= 0 or total <= 0:
        return {k: 0 for k in counts}
    exact = {k: (v / grand) * total for k, v in counts.items()}
    alloc = {k: int(math.floor(v)) for k, v in exact.items()}
    left = total - sum(alloc.values())
    order = sorted(counts, key=lambda k: (-(exact[k] - alloc[k]), k))
    for k in order[:max(0, left)]:
        alloc[k] += 1
    return alloc


def select_phase1(groups: Sequence[Dict[str, Any]], *, target: int = 100,
                  min_size: int = 80, max_size: int = 120,
                  anchor_frac: float = 0.10, control_frac: float = 0.10
                  ) -> Dict[str, Any]:
    """Recommend the Phase-1 GRPO train subset (spec §7).

    Selection is by SIGNAL CATEGORY and structural stratum, never by ranking a
    single reward score: the groups that can actually move a GRPO update are
    the mixed ones, and a subset chosen by "highest mean reward" would just be
    the easy tail. Easy anchors and hard all-failure controls are added in
    small, capped quantities so the subset stays diagnostic.
    """
    target = max(min_size, min(max_size, target))
    by_id = {str(g["task_id"]): g for g in groups}
    if len(by_id) <= min_size:
        # Nothing to choose from — take everything and say so.
        counts = Counter(str(g.get("structural_key")) for g in groups)
        n_all = len(groups) or 1
        return {
            "task_ids": sorted(by_id),
            "reasons": {t: "all_available_below_min_size" for t in by_id},
            "reason_counts": {"all_available_below_min_size": len(by_id)},
            "target": target,
            "n_selected": len(by_id),
            "quotas": {},
            "note": "fewer groups probed than the minimum subset size",
            "structural_distribution": {
                k: {"original_share": round(v / n_all, 4),
                    "selected_share": round(v / n_all, 4), "selected_n": v}
                for k, v in sorted(counts.items())},
            "max_structural_share_delta": 0.0,
            "selection_basis": ("every probed group (pool smaller than the "
                                "minimum subset size)"),
        }
    target = min(target, len(by_id))

    tier1 = [g for g in groups if g.get("terminal_mixed")]
    tier2 = [g for g in groups if g.get("process_only_mixed") and not g.get("terminal_mixed")]
    anchors = [g for g in groups if g.get("all_success_dead")]
    controls = [g for g in groups if g.get("all_failure_dead")]

    anchor_quota = min(len(anchors), int(round(target * anchor_frac)))
    control_quota = min(len(controls), int(round(target * control_frac)))
    signal_quota = max(0, target - anchor_quota - control_quota)

    # Preserve the ORIGINAL structural distribution of the probed pool.
    original_counts = Counter(str(g.get("structural_key")) for g in groups)
    signal_pool = tier1 + tier2
    stratum_targets = allocate_proportional(dict(original_counts), signal_quota)

    reasons: Dict[str, str] = {}
    selected: List[str] = []

    # Per-stratum budget is shared by both signal tiers, so tier2 fills the
    # strata tier1 could not cover instead of restarting the allocation.
    stratum_remaining = dict(stratum_targets)

    def _take(pool: Sequence[Dict[str, Any]], reason: str, quota: int,
              respect_strata: bool) -> int:
        """Select up to ``quota`` unclaimed groups; returns how many were taken."""
        if quota <= 0:
            return 0
        taken = 0
        ordered = sorted((g for g in pool if str(g["task_id"]) not in reasons),
                         key=lambda g: str(g.get("task_id")))
        if respect_strata:
            # Preferred pass: honour the per-stratum budget.
            for g in ordered:
                if taken >= quota:
                    break
                key = str(g.get("structural_key"))
                if stratum_remaining.get(key, 0) <= 0:
                    continue
                tid = str(g["task_id"])
                reasons[tid] = reason
                selected.append(tid)
                stratum_remaining[key] -= 1
                taken += 1
        # Fallback pass: a stratum may be under-populated in this tier, so top
        # up round-robin instead of leaving the quota unfilled.
        rest = [g for g in ordered if str(g["task_id"]) not in reasons]
        for g in _round_robin_by_stratum(rest, quota - taken):
            tid = str(g["task_id"])
            reasons[tid] = reason
            selected.append(tid)
            taken += 1
        return taken

    n_signal = _take(tier1, "terminal_mixed", signal_quota, True)
    n_signal += _take(tier2, "process_only_mixed", signal_quota - n_signal, True)
    _take(anchors, "easy_anchor", anchor_quota, False)
    _take(controls, "hard_all_failure_control", control_quota, False)

    # Top up to the minimum with the least-bad remainder if the signal buckets
    # were too small, keeping structural diversity.
    if len(selected) < min_size:
        rest = [g for g in groups if str(g["task_id"]) not in reasons]
        for g in _round_robin_by_stratum(rest, min_size - len(selected)):
            reasons[str(g["task_id"])] = "structural_topup"
            selected.append(str(g["task_id"]))

    sel_counts = Counter(str(by_id[t].get("structural_key")) for t in selected)
    n_sel = len(selected) or 1
    n_all = len(groups) or 1
    structural = {}
    for key in sorted(set(original_counts) | set(sel_counts)):
        structural[key] = {
            "original_share": round(original_counts.get(key, 0) / n_all, 4),
            "selected_share": round(sel_counts.get(key, 0) / n_sel, 4),
            "selected_n": sel_counts.get(key, 0),
        }

    return {
        "task_ids": sorted(selected),
        "reasons": reasons,
        "target": target,
        "n_selected": len(selected),
        "quotas": {"signal": signal_quota, "easy_anchor": anchor_quota,
                   "hard_all_failure_control": control_quota},
        "reason_counts": dict(sorted(Counter(reasons.values()).items())),
        "structural_distribution": structural,
        "max_structural_share_delta": max(
            (abs(v["original_share"] - v["selected_share"]) for v in structural.values()),
            default=0.0),
        "selection_basis": ("group signal category + structural stratum "
                            "(NOT a ranking of any single reward score)"),
    }


# ──────────────────────────────────────────────────────────────── verdict ──

def compute_verdict(summary: Dict[str, Any], ordering: Dict[str, Any]) -> Dict[str, Any]:
    """PASS / CONDITIONAL / STOP per spec §8."""
    dead = summary.get("dead_group_rate")
    ordering_valid = ordering.get("ordering_valid")
    process_rate = summary.get("process_only_mixed_rate") or 0.0
    reasons: List[str] = []

    if dead is None:
        return {"verdict": "STOP", "reasons": ["no groups probed"],
                "dead_group_rate": None, "ordering_valid": ordering_valid}

    if ordering_valid is None:
        reasons.append(
            f"reward ordering unproven: only {ordering.get('comparable_pairs')} "
            f"Pareto-comparable pairs (< {ordering.get('min_pairs_required')})")
        verdict = "STOP"
    elif not ordering_valid:
        reasons.append(
            f"reward ordering INVALID: {ordering.get('inversions')} inversions "
            f"({ordering.get('inversion_rate')}) exceed tolerance "
            f"{ordering.get('inversion_tolerance')}")
        verdict = "STOP"
    elif dead > DEAD_CONDITIONAL_MAX:
        reasons.append(f"dead-group rate {dead:.3f} > {DEAD_CONDITIONAL_MAX}")
        verdict = "STOP"
    elif dead > DEAD_PASS_MAX:
        if process_rate >= PROCESS_VARIANCE_MIN_RATE:
            reasons.append(
                f"dead-group rate {dead:.3f} in ({DEAD_PASS_MAX}, "
                f"{DEAD_CONDITIONAL_MAX}] with process-only-mixed rate "
                f"{process_rate:.3f} >= {PROCESS_VARIANCE_MIN_RATE}")
            verdict = "CONDITIONAL"
        else:
            reasons.append(
                f"dead-group rate {dead:.3f} above {DEAD_PASS_MAX} and "
                f"process-only-mixed rate {process_rate:.3f} < "
                f"{PROCESS_VARIANCE_MIN_RATE} — no usable process variance")
            verdict = "STOP"
    else:
        reasons.append(f"dead-group rate {dead:.3f} <= {DEAD_PASS_MAX} "
                       "and reward ordering valid")
        verdict = "PASS"

    return {
        "verdict": verdict,
        "reasons": reasons,
        "dead_group_rate": dead,
        "ordering_valid": ordering_valid,
        "process_only_mixed_rate": process_rate,
        "gates": {
            "PASS": f"dead <= {DEAD_PASS_MAX} AND reward ordering valid",
            "CONDITIONAL": (f"dead in ({DEAD_PASS_MAX}, {DEAD_CONDITIONAL_MAX}] AND "
                            f"process_only_mixed_rate >= {PROCESS_VARIANCE_MIN_RATE}"),
            "STOP": (f"dead > {DEAD_CONDITIONAL_MAX} OR reward ordering "
                     "invalid/unproven"),
        },
    }


# ───────────────────────────────────────────────────────────── group audit ──

def audit_groups(groups: Sequence[Dict[str, Any]],
                 records_by_task: Dict[str, List[Dict[str, Any]]], *,
                 n_dead: int = 3, n_alive: int = 3) -> Dict[str, Any]:
    """Concrete per-rollout evidence for a few dead and non-dead groups.

    The aggregate rates are only trustworthy if a human can open a handful of
    groups and see that "dead" really means "identical reward for identical
    quality" and not a bug in the reward plumbing.
    """
    def _detail(group: Dict[str, Any]) -> Dict[str, Any]:
        recs = sorted(records_by_task.get(str(group["task_id"]), []),
                      key=lambda r: int(r.get("rollout_idx") or 0))
        return {
            "task_id": group["task_id"],
            "track": group.get("track"),
            "call_count": group.get("call_count"),
            "motif": group.get("motif"),
            "answer_type": group.get("answer_type"),
            "generation_cell": group.get("generation_cell"),
            "dead_group": group.get("dead_group"),
            "success_count": group.get("success_count"),
            "reward_values": group.get("reward_values"),
            "reward_range": group.get("reward_range"),
            "failure_classes": group.get("failure_classes"),
            "rollouts": [{
                "rollout_idx": r.get("rollout_idx"),
                "failure_class": r.get("failure_class"),
                "terminal_outcome": r.get("terminal_outcome"),
                "episode_reward": r.get("episode_reward"),
                "process_reward": r.get("process_reward"),
                "return_t0": r.get("return_t0"),
                "first_tool_correct": r.get("first_tool_correct"),
                "correct_prefix_len": r.get("correct_prefix_len"),
                "n_pred_calls": r.get("n_pred_calls"),
                "n_successful_calls": r.get("n_successful_calls"),
                "arg_errors": {"key": r.get("arg_key_errors"),
                               "type": r.get("arg_type_errors"),
                               "value": r.get("arg_value_errors")},
                "completion_hash": (r.get("completion_hash") or "")[:16],
                "completion_excerpt": (r.get("raw_completion") or "")[:400],
            } for r in recs],
        }

    dead = [g for g in groups if g.get("dead_group")]
    alive = [g for g in groups if not g.get("dead_group")]
    # Prefer the most informative examples: dead groups with the most distinct
    # failure classes, live groups with the widest reward range.
    dead.sort(key=lambda g: (-len(g.get("failure_classes") or {}), str(g["task_id"])))
    alive.sort(key=lambda g: (-float(g.get("reward_range") or 0.0), str(g["task_id"])))
    return {
        "dead_groups": [_detail(g) for g in dead[:n_dead]],
        "non_dead_groups": [_detail(g) for g in alive[:n_alive]],
        "n_dead_available": len(dead),
        "n_non_dead_available": len(alive),
    }


# ────────────────────────────────────────────────────────── markdown report ──

def _md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> List[str]:
    out = ["| " + " | ".join(str(h) for h in headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        out.append("| " + " | ".join("—" if c is None else str(c) for c in row) + " |")
    return out


def _pct(value: Optional[float]) -> str:
    return "—" if value is None else f"{100.0 * float(value):.1f} %"


def render_report_md(report: Dict[str, Any]) -> str:
    v = report["verdict"]
    md: List[str] = [
        "# Pilot2 signal probe — inference only",
        "",
        f"**VERDICT: {v['verdict']}**",
        "",
    ]
    for reason in v["reasons"]:
        md.append(f"- {reason}")
    md += [
        "",
        "No optimizer step, no gradient and no LoRA update was performed: this "
        "run only asks whether the frozen Pilot2 train set produces a usable "
        "GRPO rollout/reward signal for the base checkpoint.",
        "",
        "## Setup",
        "",
    ]
    prov = report.get("provenance", {})
    md += _md_table(["property", "value"], [
        ["model", prov.get("model")],
        ["dtype", prov.get("dtype")],
        ["LoRA adapter", prov.get("lora_adapter") or "none (C0 base)"],
        ["dataset", prov.get("dataset")],
        ["dataset sha256", (prov.get("dataset_sha256") or "")[:16] + "…"],
        ["reward arm", prov.get("reward_arm")],
        ["resolved reward policy", prov.get("resolved_reward_policy")],
        ["executor mode", prov.get("executor_mode")],
        ["tool registry", prov.get("synthetic_tools_dir")],
        ["registry hash", (prov.get("registry_hash") or "")[:16] + "…"],
        ["decoding", f"T={prov.get('temperature')} top_p={prov.get('top_p')}"],
        ["seed", prov.get("seed")],
        ["P2", f"{report['phases'].get('P2', {}).get('n_groups', 0)} tasks × "
               f"{prov.get('p2_rollouts')} rollouts"],
        ["P3", f"{report['phases'].get('P3', {}).get('n_groups', 0)} tasks × "
               f"{prov.get('p3_rollouts')} rollouts"],
    ])

    for phase in ("P2", "P3"):
        summ = report["phases"].get(phase) or {}
        if not summ.get("n_groups"):
            continue
        md += ["", f"## {phase} — group signal "
                   f"({summ['n_groups']} groups, {summ['n_rollouts']} rollouts)", ""]
        md += _md_table(["metric", "value"], [
            ["dead-group rate", _pct(summ.get("dead_group_rate"))],
            ["terminal-mixed rate", _pct(summ.get("terminal_mixed_rate"))],
            ["process-only-mixed rate", _pct(summ.get("process_only_mixed_rate"))],
            ["all-failure dead rate", _pct(summ.get("all_failure_dead_rate"))],
            ["all-success dead rate", _pct(summ.get("all_success_dead_rate"))],
            ["mean success rate", _pct(summ.get("mean_success_rate"))],
            ["mean reward range", summ.get("mean_reward_range")],
            ["mean failure entropy (bits)", summ.get("mean_failure_entropy_bits")],
        ])
        n_roll = (report.get("provenance", {}).get("p3_rollouts")
                  if phase == "P3" else report.get("provenance", {}).get("p2_rollouts"))
        counts = summ.get("success_bucket_counts", {})
        rates = summ.get("success_bucket_rates", {})
        md += ["", f"### {phase} success distribution (out of {n_roll} rollouts)", ""]
        md += _md_table(["bucket", "groups", "share"], [
            [f"0/{n_roll}", counts.get("zero_success"), _pct(rates.get("zero_success"))],
            [f"1–{int(n_roll) - 1}/{n_roll}" if n_roll else "partial",
             counts.get("partial_success"), _pct(rates.get("partial_success"))],
            [f"{n_roll}/{n_roll}", counts.get("all_success"), _pct(rates.get("all_success"))],
        ])
        md += ["", f"### {phase} reward-range distribution", ""]
        md += _md_table(["reward range", "groups"],
                        [[k, val] for k, val in
                         (summ.get("reward_range_histogram") or {}).items()])

    md += ["", "## Breakdown (all probed groups)", ""]
    for field, title in (("track", "Track (A = in-distribution, G = generalization)"),
                         ("call_count", "Call count"),
                         ("motif", "Motif"),
                         ("answer_type", "Answer type")):
        dist = report["breakdowns"].get(field) or {}
        if not dist:
            continue
        md += [f"### {title}", ""]
        md += _md_table(["value", "groups", "dead", "terminal-mixed",
                         "process-only-mixed", "mean success"],
                        [[k, s["n_groups"], _pct(s["dead_rate"]),
                          _pct(s["terminal_mixed_rate"]),
                          _pct(s["process_only_mixed_rate"]),
                          _pct(s["mean_success_rate"])]
                         for k, s in dist.items()])
        md += [""]

    cells = report["breakdowns"].get("generation_cell") or {}
    if cells:
        worst = sorted(cells.items(),
                       key=lambda kv: (-(kv[1]["dead_rate"] or 0.0), kv[0]))[:15]
        md += ["### Generation cell (15 worst by dead rate)", ""]
        md += _md_table(["cell", "groups", "dead", "mean success"],
                        [[k, s["n_groups"], _pct(s["dead_rate"]),
                          _pct(s["mean_success_rate"])] for k, s in worst])
        md += [""]

    md += ["## Failure classes (all rollouts)", ""]
    md += _md_table(["failure class", "rollouts", "share"],
                    [[k, c["count"], _pct(c["share"])]
                     for k, c in (report.get("failure_classes") or {}).items()])

    ordering = report["reward_ordering"]
    md += [
        "",
        "## Reward ordering — does a higher reward mean a better trajectory?",
        "",
        "Judged only on Pareto-comparable rollout pairs inside the same group: "
        "if one rollout is at least as good as another on success, correct-prefix "
        "length, successful calls, argument errors and format validity — and "
        "strictly better on one — then its reward must not be lower.",
        "",
    ]
    md += _md_table(["metric", "value"], [
        ["comparable pairs", ordering.get("comparable_pairs")],
        ["concordant", ordering.get("concordant_pairs")],
        ["tied", ordering.get("tied_pairs")],
        ["inversions", ordering.get("inversions")],
        ["inversion rate", ordering.get("inversion_rate")],
        ["tolerance", ordering.get("inversion_tolerance")],
        ["ordering valid", ordering.get("ordering_valid")],
    ])
    if ordering.get("inversion_examples"):
        md += ["", "Inversion examples (better trajectory, lower reward):", ""]
        md += _md_table(["task", "better rollout", "reward", "worse rollout", "reward"],
                        [[e["task_id"], e["better_rollout"], e["better_reward"],
                          e["worse_rollout"], e["worse_reward"]]
                         for e in ordering["inversion_examples"][:5]])

    audit = report.get("group_audit") or {}
    md += ["", "## Group audit", ""]
    for label, key in (("Dead groups", "dead_groups"), ("Non-dead groups", "non_dead_groups")):
        entries = audit.get(key) or []
        md += [f"### {label} ({len(entries)} shown of {audit.get('n_dead_available') if key == 'dead_groups' else audit.get('n_non_dead_available')})", ""]
        for g in entries:
            md += [
                f"**`{g['task_id']}`** — track {g['track']}, {g['call_count']} calls, "
                f"motif `{g['motif']}`, answer `{g['answer_type']}`, cell `{g['generation_cell']}`",
                "",
                f"rewards {g['reward_values']} (range {g['reward_range']}), "
                f"successes {g['success_count']}, failures {g['failure_classes']}",
                "",
            ]
            md += _md_table(["rollout", "failure class", "terminal", "reward",
                             "process", "prefix", "calls", "ok calls", "arg errors"],
                            [[r["rollout_idx"], r["failure_class"], r["terminal_outcome"],
                              r["episode_reward"], r["process_reward"],
                              r["correct_prefix_len"], r["n_pred_calls"],
                              r["n_successful_calls"],
                              sum(v or 0 for v in r["arg_errors"].values())]
                             for r in g["rollouts"]])
            md += [""]

    p1 = report["phase1_selection"]
    md += [
        "## Recommended Phase-1 subset",
        "",
        f"{p1.get('n_selected')} tasks -> `recommended_phase1_train.jsonl` "
        f"(deferred: {report['deferred_count']} -> `deferred_phase2_tasks.jsonl`).",
        "",
        f"Basis: {p1.get('selection_basis', 'n/a')}.",
        "",
    ]
    md += _md_table(["reason", "tasks"],
                    [[k, val] for k, val in (p1.get("reason_counts") or {}).items()])
    md += ["", "### Structural distribution preserved", "",
           f"largest share delta vs the probed pool: "
           f"{p1.get('max_structural_share_delta', 0.0)}", ""]
    md += _md_table(["stratum", "original share", "selected share", "selected n"],
                    [[k, s["original_share"], s["selected_share"], s["selected_n"]]
                     for k, s in (p1.get("structural_distribution") or {}).items()])

    md += [
        "",
        "## What to do next",
        "",
        "- **PASS** — start the GRPO run on `recommended_phase1_train.jsonl`.",
        "- **CONDITIONAL** — the signal lives in the process component only; "
        "start only with A4 (gated verifiable) and expect slow movement.",
        "- **STOP** — do not spend GPU hours on GRPO: fix the reward or the "
        "task mix first.",
        "",
    ]
    return "\n".join(md) + "\n"


def build_report(*, groups: Sequence[Dict[str, Any]],
                 records: Sequence[Dict[str, Any]],
                 provenance: Dict[str, Any],
                 phase1: Dict[str, Any],
                 p3_selection: Optional[Dict[str, Any]] = None,
                 deferred_count: int = 0) -> Dict[str, Any]:
    """Assemble the machine-readable report (spec §6)."""
    records_by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        records_by_task[str(r.get("task_id"))].append(r)

    by_phase: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for g in groups:
        by_phase[str(g.get("phase"))].append(g)

    # Verdict is judged on the deepest evidence available: P3 when it ran
    # (8 rollouts resolve borderline groups), else P2.
    decision_phase = "P3" if by_phase.get("P3") else "P2"
    decision_groups = by_phase.get(decision_phase) or []

    fc = Counter(str(r.get("failure_class")) for r in records)
    n_rec = sum(fc.values()) or 1
    failure_classes = {
        k: {"count": c, "share": round(c / n_rec, 6)}
        for k, c in sorted(fc.items(), key=lambda kv: (-kv[1], kv[0]))
    }

    ordering = reward_ordering_audit(
        {t: rs for t, rs in records_by_task.items() if len(rs) > 1})
    summary = phase_summary(decision_groups)
    verdict = compute_verdict(summary, ordering)

    return {
        "schema_version": SCHEMA_VERSION,
        "provenance": provenance,
        "decision_phase": decision_phase,
        "summary": summary,
        "phases": {p: phase_summary(gs) for p, gs in sorted(by_phase.items())},
        "breakdowns": {
            field: distribution_by(decision_groups or list(groups), field)
            for field in ("track", "call_count", "motif", "answer_type",
                          "generation_cell")
        },
        "failure_classes": failure_classes,
        "reward_ordering": ordering,
        "group_audit": audit_groups(decision_groups or list(groups), records_by_task),
        "p3_selection": p3_selection or {},
        "phase1_selection": phase1,
        "deferred_count": deferred_count,
        "verdict": verdict,
    }
