#!/usr/bin/env python3
"""Tests for the Pilot2 signal probe — no GPU, no model, no network.

Covers the whole decision layer (the part that can silently produce a wrong
GO/NO-GO answer): argument-error classification, correct-prefix length, failure
classes, group metrics, reward-ordering audit, P3 selection, Phase-1 selection,
verdict thresholds, the content-hash cache and turn-return parity with the real
trainer. Ends with an end-to-end analyze run over synthetic shards and a worker
dry run.

Usage:
    python experiments/targeted_tool_data_factory/runpod_bundle_pilot2/test_signal_probe.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

BUNDLE = Path(__file__).resolve().parent
FACTORY = BUNDLE.parent
EXPERIMENTS = FACTORY.parent
sys.path.insert(0, str(BUNDLE))

from signal_probe_lib import (  # noqa: E402
    FAILURE_CLASSES, arg_error_counts, allocate_proportional, build_group, build_report,
    classify_failure, compute_verdict, correct_prefix_len, derive_rollout_metrics,
    extract_task_meta, objective_quality, pareto_compare, phase_summary,
    render_report_md, reward_ordering_audit, rollout_cache_key, select_p3_tasks,
    select_phase1, shannon_entropy, structural_key,
)
from signal_probe_worker import turn_returns  # noqa: E402

FAILED: List[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAILED.append(msg)
        print(f"  FAIL: {msg}")


# ─────────────────────────────────────────────────── argument comparison ──

def test_arg_error_counts() -> None:
    gold = {"a": 1, "b": "x", "c": [1, 2]}
    same = arg_error_counts(dict(gold), gold)
    check(same["key_errors"] == 0 and same["type_errors"] == 0
          and same["value_errors"] == 0, "identical args must be error free")

    missing = arg_error_counts({"a": 1, "b": "x"}, gold)
    check(missing["key_errors"] == 1 and missing["missing_keys"] == ["c"],
          "a missing argument is one key error")

    extra = arg_error_counts({**gold, "d": 9}, gold)
    check(extra["key_errors"] == 1 and extra["extra_keys"] == ["d"],
          "an extra argument is one key error")

    wrong_type = arg_error_counts({"a": "1", "b": "x", "c": [1, 2]}, gold)
    check(wrong_type["type_errors"] == 1 and wrong_type["value_errors"] == 0,
          "number vs string is a type error, not a value error")

    wrong_value = arg_error_counts({"a": 2, "b": "x", "c": [1, 2]}, gold)
    check(wrong_value["value_errors"] == 1 and wrong_value["type_errors"] == 0,
          "right type wrong number is a value error")

    # A float within tolerance is the same value: the executor rounds.
    tol = arg_error_counts({"a": 1.0002, "b": "x", "c": [1, 2]}, gold)
    check(tol["value_errors"] == 0, "numeric tolerance must absorb rounding")

    # bool must never be compared as the number 1
    strict_bool = arg_error_counts({"flag": True}, {"flag": 1})
    check(strict_bool["type_errors"] == 1, "True must not equal 1")


def test_correct_prefix_len() -> None:
    gold = [{"name": "add", "arguments": {"a": 1}},
            {"name": "mul", "arguments": {"b": 2}},
            {"name": "sub", "arguments": {"c": 3}}]
    check(correct_prefix_len(gold, gold) == 3, "identical trace = full prefix")
    check(correct_prefix_len(gold[:2], gold) == 2, "truncated trace = its length")
    check(correct_prefix_len([], gold) == 0, "no calls = zero prefix")

    wrong_second_name = [gold[0], {"name": "div", "arguments": {"b": 2}}, gold[2]]
    check(correct_prefix_len(wrong_second_name, gold) == 1,
          "prefix stops at the first wrong name and does not resume")

    wrong_first_arg = [{"name": "add", "arguments": {"a": 99}}] + gold[1:]
    check(correct_prefix_len(wrong_first_arg, gold) == 0,
          "prefix stops at the first wrong argument")


# ──────────────────────────────────────────────── failure classification ──

def _rec(**kw: Any) -> Dict[str, Any]:
    base = {"success": False, "prompt_overflow": False, "clipped": False,
            "parse_error": False, "stop_reason": "terminal", "n_pred_calls": 2,
            "resolved_calls": [], "first_tool_correct": True, "call_count": 2,
            "arg_key_errors": 0, "arg_type_errors": 0, "arg_value_errors": 0}
    base.update(kw)
    return base


def test_classify_failure() -> None:
    check(classify_failure(_rec(success=True)) == "success", "success wins")
    check(classify_failure(_rec(prompt_overflow=True)) == "prompt_overflow",
          "prompt overflow is its own class")
    check(classify_failure(_rec(clipped=True)) == "clipped_completion",
          "clipped completion is its own class")
    check(classify_failure(_rec(stop_reason="parse_fail")) == "parse_error",
          "parse failure detected from stop_reason")
    check(classify_failure(_rec(n_pred_calls=0, stop_reason="terminal"))
          == "no_tool_call", "zero calls is no_tool_call")

    errs = {
        "unknown_tool:nope": "unknown_tool",
        "unresolved_variable:var9": "invalid_reference",
        "synthetic:missing_required_argument:x": "arg_key_error",
        "synthetic:argument_type_mismatch:x": "arg_type_error",
        "synthetic:argument_below_min:x": "arg_range_error",
        "synthetic:division_by_zero:x": "exec_division_by_zero",
        "synthetic:runtime_error:x": "exec_error",
    }
    for err, want in errs.items():
        got = classify_failure(_rec(resolved_calls=[{"error": err}]))
        check(got == want, f"executor error {err!r} -> {want}, got {got}")

    check(classify_failure(_rec(first_tool_correct=False)) == "wrong_first_tool",
          "wrong first tool is reported before arg errors")
    check(classify_failure(_rec(n_pred_calls=1, call_count=3)) == "too_few_calls",
          "fewer calls than gold is too_few_calls")
    check(classify_failure(_rec(n_pred_calls=5, call_count=3)) == "too_many_calls",
          "more calls than gold is too_many_calls")
    check(classify_failure(_rec(arg_value_errors=2)) == "wrong_args",
          "wrong argument values classify as wrong_args")
    check(classify_failure(_rec()) == "wrong_final_answer",
          "fully executable but unsuccessful is wrong_final_answer")

    # Every label the classifier can emit must be in the declared taxonomy, or
    # the report's failure-class table silently grows unknown buckets.
    emitted = {classify_failure(_rec(success=True)),
               classify_failure(_rec(prompt_overflow=True)),
               classify_failure(_rec(clipped=True)),
               classify_failure(_rec(stop_reason="parse_fail")),
               classify_failure(_rec(n_pred_calls=0)),
               classify_failure(_rec(first_tool_correct=False)),
               classify_failure(_rec(n_pred_calls=1, call_count=3)),
               classify_failure(_rec(n_pred_calls=5, call_count=3)),
               classify_failure(_rec(arg_value_errors=2)),
               classify_failure(_rec())} | set(errs.values())
    unknown = emitted - set(FAILURE_CLASSES)
    check(not unknown, f"undeclared failure classes: {sorted(unknown)}")


def test_derive_rollout_metrics() -> None:
    gold = [{"name": "add", "arguments": {"a": 1, "b": 2}},
            {"name": "mul", "arguments": {"x": 3.0}}]
    rec = {
        "success": False, "n_pred_calls": 2, "n_successful_calls": 2,
        "call_count": 2, "stop_reason": "terminal",
        "resolved_calls": [
            {"name": "add", "arguments_resolved": {"a": 1, "b": 2}, "error": None},
            {"name": "mul", "arguments_resolved": {"x": 4.0}, "error": None},
        ],
    }
    out = derive_rollout_metrics(dict(rec), gold)
    check(out["first_tool_correct"] is True, "first tool matches gold")
    check(out["correct_prefix_len"] == 1, "prefix is 1 (second call has wrong value)")
    check(out["correct_prefix_frac"] == 0.5, "prefix fraction is 1/2")
    check(out["arg_value_errors"] == 1, "one wrong argument value counted")
    check(out["failure_class"] == "wrong_args", "classified as wrong_args")


# ─────────────────────────────────────────────────────────── group metrics ──

def _meta(task_id: str, *, track: str = "A", call_count: int = 3,
          motif: str = "linear", answer_type: str = "float",
          cell: str = "A_3call_linear_00") -> Dict[str, Any]:
    return {"task_id": task_id, "track": track, "call_count": call_count,
            "motif": motif, "answer_type": answer_type, "generation_cell": cell}


def _rollout(task_id: str, idx: int, *, reward: float, success: bool,
             failure: str = "wrong_final_answer", process: float = 0.5,
             prefix: int = 1, ok_calls: int = 1, phase: str = "P2",
             **extra: Any) -> Dict[str, Any]:
    rec = {
        "phase": phase, "task_id": task_id, "rollout_idx": idx,
        "cache_key": f"{phase}|{task_id}|{idx}",
        "episode_reward": reward, "process_reward": process,
        "return_t0": reward, "success": success,
        "failure_class": "success" if success else failure,
        "terminal_outcome": "official_success" if success else "executable_wrong_result",
        "correct_prefix_len": prefix, "correct_prefix_frac": prefix / 3.0,
        "first_tool_correct": prefix > 0, "n_pred_calls": 3,
        "n_successful_calls": ok_calls, "arg_key_errors": 0,
        "arg_type_errors": 0, "arg_value_errors": 0,
        "parse_error": False, "clipped": False, "prompt_overflow": False,
        "raw_completion": f"completion {task_id} #{idx}",
        "completion_hash": f"hash{task_id}{idx}",
    }
    rec.update(extra)
    return rec


def test_group_metrics() -> None:
    # terminal mixed: 2 success, 2 failure
    recs = [_rollout("t1", 0, reward=0.95, success=True),
            _rollout("t1", 1, reward=0.95, success=True),
            _rollout("t1", 2, reward=0.30, success=False),
            _rollout("t1", 3, reward=0.10, success=False, failure="parse_error")]
    g = build_group(_meta("t1"), recs, phase="P2")
    check(g["success_count"] == 2, "success count")
    check(g["reward_min"] == 0.10 and g["reward_max"] == 0.95, "reward min/max")
    check(abs(g["reward_range"] - 0.85) < 1e-9, "reward range")
    check(g["dead_group"] is False, "group with spread is not dead")
    check(g["terminal_mixed"] is True, "some success and some failure = terminal mixed")
    check(g["process_only_mixed"] is False, "terminal mixed is not process-only mixed")
    check(g["all_failure_dead"] is False and g["all_success_dead"] is False,
          "mixed group is neither all-failure nor all-success dead")
    check(g["failure_entropy_bits"] > 0, "three distinct classes give entropy > 0")
    check(g["structural_key"] == structural_key(_meta("t1")), "structural key recorded")

    # process-only mixed: no success, but reward still separates
    recs = [_rollout("t2", i, reward=0.30 + 0.01 * i, success=False) for i in range(4)]
    g = build_group(_meta("t2"), recs, phase="P2")
    check(g["success_count"] == 0, "no successes")
    check(g["process_only_mixed"] is True, "all failures with spread = process-only mixed")
    check(g["dead_group"] is False, "spread means not dead")
    check(g["all_failure_dead"] is False, "spread means not all-failure dead")

    # all-failure dead
    recs = [_rollout("t3", i, reward=0.05, success=False, failure="no_tool_call")
            for i in range(4)]
    g = build_group(_meta("t3"), recs, phase="P2")
    check(g["dead_group"] is True and g["all_failure_dead"] is True,
          "identical failing rewards = all-failure dead")
    check(g["failure_entropy_bits"] == 0.0, "single failure class = zero entropy")

    # all-success dead (saturated)
    recs = [_rollout("t4", i, reward=0.95, success=True) for i in range(4)]
    g = build_group(_meta("t4"), recs, phase="P2")
    check(g["dead_group"] is True and g["all_success_dead"] is True,
          "identical winning rewards = all-success dead")
    check(g["terminal_mixed"] is False, "all success is not terminal mixed")

    check(abs(shannon_entropy(["a", "a", "b", "b"]) - 1.0) < 1e-9,
          "two equal classes = 1 bit")


# ───────────────────────────────────────────────────── reward ordering ──

def test_reward_ordering_audit() -> None:
    check(pareto_compare((1, 2, 3), (1, 1, 3)) == 1, "dominating vector detected")
    check(pareto_compare((1, 1, 3), (1, 2, 3)) == -1, "dominated vector detected")
    check(pareto_compare((1, 2), (2, 1)) == 0, "incomparable vectors")

    # Monotone: better trajectory always has the higher reward.
    good = {f"t{t}": [
        _rollout(f"t{t}", 0, reward=0.95, success=True, prefix=3, ok_calls=3),
        _rollout(f"t{t}", 1, reward=0.30, success=False, prefix=1, ok_calls=1),
        _rollout(f"t{t}", 2, reward=0.10, success=False, prefix=0, ok_calls=0),
    ] for t in range(10)}
    ok = reward_ordering_audit(good)
    check(ok["inversions"] == 0, "monotone rewards give zero inversions")
    check(ok["ordering_valid"] is True, "monotone rewards are valid ordering")
    check(ok["comparable_pairs"] >= 20, "fixture yields enough comparable pairs")

    # Inverted: the objectively best trajectory gets the lowest reward.
    bad = {f"t{t}": [
        _rollout(f"t{t}", 0, reward=0.10, success=True, prefix=3, ok_calls=3),
        _rollout(f"t{t}", 1, reward=0.95, success=False, prefix=0, ok_calls=0),
    ] for t in range(30)}
    inv = reward_ordering_audit(bad)
    check(inv["inversions"] == 30, "every inverted pair is counted")
    check(inv["ordering_valid"] is False, "inverted rewards are invalid ordering")
    check(len(inv["inversion_examples"]) > 0, "inversion examples are reported")

    # Too little evidence must not be reported as valid.
    thin = reward_ordering_audit({"t0": [
        _rollout("t0", 0, reward=0.9, success=True, prefix=3, ok_calls=3),
        _rollout("t0", 1, reward=0.1, success=False, prefix=0, ok_calls=0)]})
    check(thin["ordering_valid"] is None,
          "fewer than the minimum comparable pairs = unproven, not valid")

    q = objective_quality(_rollout("t", 0, reward=0.9, success=True, prefix=3,
                                   ok_calls=3))
    check(q[0] == 1.0 and q[1] == 3.0, "objective quality reads success and prefix")


# ────────────────────────────────────────────────────────── P3 selection ──

def _fake_records(meta: Dict[str, Any], kind: int, n_rollouts: int,
                  phase: str = "P2") -> List[Dict[str, Any]]:
    """Rollout records for one synthetic group.

    Reward is always a monotone function of the objective quality, so the
    fixture itself never contains a reward inversion — an inversion reported by
    a test therefore means the audit found a real one.
    """
    recs: List[Dict[str, Any]] = []
    for j in range(n_rollouts):
        failure = "wrong_final_answer"
        if kind == 0:        # terminal mixed
            if j == 0:
                reward, success, prefix = 0.95, True, 3
            elif j % 2 == 1:
                reward, success, prefix = 0.30, False, 1
            else:
                reward, success, prefix = 0.10, False, 0
        elif kind == 1:      # process-only mixed: reward tracks the prefix
            prefix, success = j % 3, False
            reward = 0.30 + 0.005 * prefix
        elif kind == 2:      # all-failure dead
            reward, success, prefix = 0.05, False, 0
            failure = "no_tool_call"
        elif kind == 3:      # all-success dead (easy anchor)
            reward, success, prefix = 0.95, True, 3
        else:                # dead in the middle
            reward, success, prefix = 0.30, False, 1
        rec = _rollout(meta["task_id"], j, reward=round(reward, 6), success=success,
                       failure=failure, prefix=prefix, ok_calls=prefix, phase=phase)
        rec.update({"track": meta["track"], "call_count": meta["call_count"],
                    "motif": meta["motif"], "answer_type": meta["answer_type"],
                    "generation_cell": meta["generation_cell"]})
        recs.append(rec)
    return recs


def _pool_specs(n: int = 160) -> List[tuple]:
    """(meta, kind) for a realistic pool.

    The structural axes (track / call count / motif / answer type) are
    deliberately DECORRELATED from the group kind, so every stratum contains a
    mix of mixed / dead / saturated groups exactly as the real data does.
    """
    tracks, calls = ["A", "G"], [2, 3, 4, 5, 6]
    motifs = ["linear", "fan_in", "branch_aggregate"]
    answers = ["float", "int", "bool", "string"]
    specs = []
    for i in range(n):
        track, nc, motif = tracks[i % 2], calls[i % 5], motifs[i % 3]
        meta = _meta(f"ttdf_{i:04d}", track=track, call_count=nc, motif=motif,
                     answer_type=answers[i % 4],
                     cell=f"{track}_{nc}call_{motif}_0{i % 3}")
        specs.append((meta, (i // 5) % 5))
    return specs


def _pool(n: int = 160, n_rollouts: int = 4) -> List[Dict[str, Any]]:
    """A realistic P2 group pool: mixed / process-only / dead / saturated."""
    return [build_group(meta, _fake_records(meta, kind, n_rollouts), phase="P2")
            for meta, kind in _pool_specs(n)]


def test_select_p3() -> None:
    groups = _pool()
    sel = select_p3_tasks(groups, limit=64)
    ids = sel["task_ids"]
    check(len(ids) <= 64, f"P3 respects the limit, got {len(ids)}")
    check(len(ids) == len(set(ids)), "no duplicate P3 tasks")

    by_id = {str(g["task_id"]): g for g in groups}
    boundary = [t for t in ids if by_id[t]["terminal_mixed"]]
    check(len(boundary) == len([g for g in groups if g["terminal_mixed"]]),
          "every boundary (1/4-3/4) group is selected first")

    counts = sel["selected_bucket_counts"]
    check(counts.get("p2_boundary_success", 0) > 0, "boundary bucket used")
    check(counts.get("all_failure_reward_spread", 0) > 0, "spread bucket used")
    dead_picked = counts.get("dead_group_stratified_control", 0)
    # Dead controls must always be present, even when the informative buckets
    # could fill the entire budget on their own.
    check(dead_picked >= sel["dead_control_slots_reserved"] > 0,
          f"dead controls are reserved and selected, got {dead_picked}")
    check(dead_picked < len(ids) // 2, "dead controls are a minority of P3")

    # Diversity: the selection must not collapse onto one stratum.
    tracks = {by_id[t]["track"] for t in ids}
    motifs = {by_id[t]["motif"] for t in ids}
    answers = {by_id[t]["answer_type"] for t in ids}
    counts_calls = {by_id[t]["call_count"] for t in ids}
    check(tracks == {"A", "G"}, "both tracks represented in P3")
    check(len(motifs) == 3, "all motifs represented in P3")
    check(len(answers) == 4, "all answer types represented in P3")
    check(len(counts_calls) == 5, "all call counts represented in P3")

    # Determinism.
    check(select_p3_tasks(groups, limit=64)["task_ids"] == ids,
          "P3 selection is deterministic")


# ──────────────────────────────────────────────────── Phase-1 selection ──

def test_select_phase1() -> None:
    groups = _pool()
    sel = select_phase1(groups, target=100, min_size=80, max_size=120)
    ids = sel["task_ids"]
    check(80 <= len(ids) <= 120, f"Phase-1 size in [80,120], got {len(ids)}")
    check(len(ids) == len(set(ids)), "no duplicate Phase-1 tasks")

    reasons = sel["reason_counts"]
    check(reasons.get("terminal_mixed", 0) > 0, "terminal-mixed groups preferred")
    check(reasons.get("process_only_mixed", 0) > 0, "process-only-mixed included")
    check(0 < reasons.get("easy_anchor", 0) <= 15, "a few easy anchors added")
    check(0 < reasons.get("hard_all_failure_control", 0) <= 15,
          "a few hard all-failure controls added")

    by_id = {str(g["task_id"]): g for g in groups}
    n_signal = sum(1 for t in ids if by_id[t]["terminal_mixed"]
                   or by_id[t]["process_only_mixed"])
    check(n_signal > len(ids) // 2, "most of the subset carries usable signal")

    # Not chosen by a single reward score: the subset must NOT be the top-reward
    # tail, so it has to include low-mean-reward groups too.
    means = sorted(by_id[t]["reward_mean"] for t in ids)
    check(means[0] < 0.2, "subset includes hard low-reward groups")
    check(means[-1] > 0.5, "subset includes easy high-reward groups")

    check(sel["max_structural_share_delta"] <= 0.10,
          f"structural distribution preserved (delta "
          f"{sel['max_structural_share_delta']})")
    for key, stats in sel["structural_distribution"].items():
        check(stats["selected_n"] > 0 or stats["original_share"] == 0,
              f"stratum {key} is not dropped entirely")

    check(select_phase1(groups, target=100)["task_ids"] == ids,
          "Phase-1 selection is deterministic")

    small = select_phase1(groups[:20], target=100, min_size=80)
    check(len(small["task_ids"]) == 20, "tiny pool returns everything, not a crash")

    alloc = allocate_proportional({"a": 5, "b": 5}, 7)
    check(sum(alloc.values()) == 7, "largest-remainder allocation totals exactly")


# ─────────────────────────────────────────────────────────────── verdict ──

def test_verdict() -> None:
    valid = {"ordering_valid": True, "comparable_pairs": 500, "inversions": 0,
             "inversion_rate": 0.0, "inversion_tolerance": 0.02,
             "min_pairs_required": 20}
    invalid = {**valid, "ordering_valid": False, "inversions": 50,
               "inversion_rate": 0.10}
    unproven = {**valid, "ordering_valid": None, "comparable_pairs": 3}

    v = compute_verdict({"dead_group_rate": 0.40, "process_only_mixed_rate": 0.3}, valid)
    check(v["verdict"] == "PASS", f"dead 0.40 + valid ordering = PASS, got {v}")

    v = compute_verdict({"dead_group_rate": 0.50, "process_only_mixed_rate": 0.3}, valid)
    check(v["verdict"] == "PASS", "dead exactly at 0.50 still passes")

    v = compute_verdict({"dead_group_rate": 0.60, "process_only_mixed_rate": 0.25}, valid)
    check(v["verdict"] == "CONDITIONAL",
          f"dead 0.60 with process variance = CONDITIONAL, got {v}")

    v = compute_verdict({"dead_group_rate": 0.60, "process_only_mixed_rate": 0.01}, valid)
    check(v["verdict"] == "STOP",
          "dead 0.60 without usable process variance = STOP")

    v = compute_verdict({"dead_group_rate": 0.80, "process_only_mixed_rate": 0.5}, valid)
    check(v["verdict"] == "STOP", "dead 0.80 = STOP regardless of process variance")

    v = compute_verdict({"dead_group_rate": 0.10, "process_only_mixed_rate": 0.5}, invalid)
    check(v["verdict"] == "STOP",
          "broken reward ordering = STOP even with a low dead rate")

    v = compute_verdict({"dead_group_rate": 0.10, "process_only_mixed_rate": 0.5}, unproven)
    check(v["verdict"] == "STOP", "unproven ordering must not be reported as PASS")


# ───────────────────────────────────────────── cache key / return parity ──

def test_cache_key() -> None:
    sig = {"model": "Qwen/Qwen3-4B-Instruct-2507", "temperature": 0.7}
    base = rollout_cache_key(row_hash="abc", task_id="t1", rollout_idx=0,
                             phase="P2", probe_signature=sig)
    check(base == rollout_cache_key(row_hash="abc", task_id="t1", rollout_idx=0,
                                    phase="P2", probe_signature=dict(sig)),
          "cache key is stable for identical inputs")
    variants = {
        "row content": rollout_cache_key(row_hash="def", task_id="t1", rollout_idx=0,
                                         phase="P2", probe_signature=sig),
        "rollout index": rollout_cache_key(row_hash="abc", task_id="t1", rollout_idx=1,
                                           phase="P2", probe_signature=sig),
        "phase": rollout_cache_key(row_hash="abc", task_id="t1", rollout_idx=0,
                                   phase="P3", probe_signature=sig),
        "decoding": rollout_cache_key(row_hash="abc", task_id="t1", rollout_idx=0,
                                      phase="P2",
                                      probe_signature={**sig, "temperature": 1.0}),
        "model": rollout_cache_key(row_hash="abc", task_id="t1", rollout_idx=0,
                                   phase="P2",
                                   probe_signature={**sig, "model": "other"}),
    }
    for label, key in variants.items():
        check(key != base, f"cache key must change when {label} changes")


def test_turn_returns_matches_trainer() -> None:
    r_seq = [0.0, 0.0, 0.0]
    got = turn_returns(r_seq, 0.83, 1.0, 1.0)
    check(all(abs(v - 0.83) < 1e-9 for v in got),
          "sparse r_seq with gamma=lambda=1 gives the episode reward at every turn")

    dense = turn_returns([0.1, 0.2], 0.5, 0.9, 1.0)
    check(len(dense) == 2, "one return per turn")

    minimal = EXPERIMENTS / "nestful_mtgrpo_minimal"
    if str(minimal) not in sys.path:
        sys.path.insert(0, str(minimal))
    try:
        from grpo_train import _turn_returns as trainer_fn  # type: ignore
    except Exception as exc:  # noqa: BLE001 - torch/trl not installed locally
        print(f"  (skipped trainer parity: {type(exc).__name__})")
        return
    for r_seq, ep, gamma, lam in (([0.0, 0.0, 0.0], 0.83, 1.0, 1.0),
                                  ([0.1, 0.2], 0.5, 0.9, 1.0),
                                  ([0.3], 0.7, 0.95, 0.5)):
        mine = turn_returns(r_seq, ep, gamma, lam)
        theirs = trainer_fn(list(r_seq), ep, gamma, lam)
        check(all(abs(a - b) < 1e-12 for a, b in zip(mine, theirs))
              and len(mine) == len(theirs),
              f"turn_returns must match the trainer for r_seq={r_seq}")


def test_extract_task_meta() -> None:
    row = {"sample_id": "ttdf_x", "answer_type": "float", "motif_type": "fan_in",
           "num_calls": 4, "gold_calls": [{}] * 4,
           "provenance": {"track": "G", "generation_cell_id": "G_4call_fan_in_00",
                          "target_skill": "variable_planning",
                          "target_failure_mode": "too_few_calls"}}
    meta = extract_task_meta(row)
    check(meta["task_id"] == "ttdf_x", "task id read")
    check(meta["track"] == "G", "track read from provenance")
    check(meta["call_count"] == 4 and meta["motif"] == "fan_in", "structure read")
    check(meta["answer_type"] == "float", "answer type read")
    check(meta["generation_cell"] == "G_4call_fan_in_00", "generation cell read")
    check(extract_task_meta({"sample_id": "y", "gold_calls": [{}, {}]})["call_count"] == 2,
          "call count falls back to len(gold_calls)")


# ────────────────────────────────────────────────────────── end to end ──

def _fake_dataset(path: Path, n: int = 160) -> List[str]:
    tracks, calls = ["A", "G"], [2, 3, 4, 5, 6]
    motifs = ["linear", "fan_in", "branch_aggregate"]
    answers = ["float", "int", "bool", "string"]
    ids = []
    with path.open("w", encoding="utf-8") as fh:
        for i in range(n):
            tid = f"ttdf_{i:04d}"
            ids.append(tid)
            nc = calls[i % 5]
            fh.write(json.dumps({
                "sample_id": tid,
                "question": f"question {i}",
                "answer_type": answers[i % 4],
                "motif_type": motifs[i % 3],
                "num_calls": nc,
                "gold_answer": 1.0,
                "gold_calls": [{"name": "add", "arguments": {"a": 1},
                                "label": f"$var{j + 1}"} for j in range(nc)],
                "observations": [1.0] * nc,
                "tools": [{"name": "add", "parameters": {}, "output_parameters": {}}],
                "provenance": {
                    "track": tracks[i % 2],
                    "generation_cell_id": f"{tracks[i % 2]}_{nc}call_{motifs[i % 3]}_0{i % 3}",
                    "target_skill": "tool_catalog",
                    "target_failure_mode": "wrong_args",
                },
            }, ensure_ascii=False) + "\n")
    return ids


def _write_fake_shards(probe_dir: Path, specs: List[tuple], phase: str,
                       n_rollouts: int, n_shards: int = 4) -> None:
    """Write synthetic rollout records the way the GPU workers would."""
    sinks = [(probe_dir / f"shard_{phase.lower()}_{i}.jsonl").open("w", encoding="utf-8")
             for i in range(n_shards)]
    try:
        for gi, (meta, kind) in enumerate(specs):
            sink = sinks[gi % n_shards]
            for rec in _fake_records(meta, kind, n_rollouts, phase=phase):
                sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
    finally:
        for s in sinks:
            s.close()


def _manifest(phase: str, data: Path, rollouts: int) -> str:
    return json.dumps({
        "worker_version": "test", "phase": phase, "dataset": str(data),
        "rollouts_per_task": rollouts, "backend": "vllm",
        "executor_mode": "synthetic", "overrides": ["executor.mode=synthetic"],
        "mt_grpo": {"gamma": 1.0, "lambda_episode": 1.0},
        "reward": {"resolved_policy": "reward_ablation_A4_GATED_VERIFIABLE"},
        "probe_signature": {"model": "Qwen/Qwen3-4B-Instruct-2507",
                            "dtype": "bfloat16", "reward_arm": "A4_GATED_VERIFIABLE",
                            "temperature": 0.7, "top_p": 0.95, "seed": 20260726,
                            "registry_hash": "deadbeef" * 8,
                            "dataset_sha256": "cafe" * 16},
    })


def test_analyze_end_to_end() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        data = root / "train_grpo_pilot2.jsonl"
        _fake_dataset(data)
        probe_dir = root / "signal_probe"
        probe_dir.mkdir()

        specs = _pool_specs()
        _write_fake_shards(probe_dir, specs, "P2", 4)
        (probe_dir / "manifest_p2_0.json").write_text(
            _manifest("P2", data, 4), encoding="utf-8")

        # ── select-p3 ──
        rc = subprocess.run([sys.executable, str(BUNDLE / "signal_probe_analyze.py"),
                             "--mode", "select-p3", "--probe-dir", str(probe_dir),
                             "--data", str(data), "--p3-limit", "64"],
                            capture_output=True, text=True)
        check(rc.returncode == 0, f"select-p3 exits 0 ({rc.stderr[-500:]})")
        p3_ids = (probe_dir / "p3_task_ids.txt").read_text(encoding="utf-8").split()
        check(0 < len(p3_ids) <= 64, f"p3_task_ids.txt has 1..64 ids, got {len(p3_ids)}")

        # ── P3 shards for the selected tasks (8 rollouts), then the report ──
        keep = set(p3_ids)
        p3_specs = [(meta, kind) for meta, kind in specs if meta["task_id"] in keep]
        _write_fake_shards(probe_dir, p3_specs, "P3", 8)
        (probe_dir / "manifest_p3_0.json").write_text(
            _manifest("P3", data, 8), encoding="utf-8")

        rc = subprocess.run([sys.executable, str(BUNDLE / "signal_probe_analyze.py"),
                             "--mode", "report", "--probe-dir", str(probe_dir),
                             "--data", str(data)],
                            capture_output=True, text=True)
        check(rc.returncode == 0, f"report exits 0 ({rc.stderr[-800:]})")

        for name in ("rollouts.jsonl", "groups.jsonl", "SIGNAL_PROBE_REPORT.md",
                     "SIGNAL_PROBE_REPORT.json", "recommended_phase1_train.jsonl",
                     "deferred_phase2_tasks.jsonl"):
            check((probe_dir / name).is_file(), f"{name} written")

        report = json.loads((probe_dir / "SIGNAL_PROBE_REPORT.json").read_text(
            encoding="utf-8"))
        summ = report["summary"]
        for key in ("dead_group_rate", "terminal_mixed_rate",
                    "process_only_mixed_rate", "reward_range_histogram",
                    "success_bucket_counts"):
            check(key in summ, f"report summary contains {key}")
        for field in ("track", "call_count", "motif", "answer_type",
                      "generation_cell"):
            check(report["breakdowns"].get(field), f"breakdown by {field} present")
        check(len(report["group_audit"]["dead_groups"]) >= 3, "at least 3 dead audits")
        check(len(report["group_audit"]["non_dead_groups"]) >= 3,
              "at least 3 non-dead audits")
        check(report["failure_classes"], "failure classes reported")
        check(report["verdict"]["verdict"] in ("PASS", "CONDITIONAL", "STOP"),
              "verdict is one of the three")
        check(report["reward_ordering"]["ordering_valid"] is True,
              "consistent fixture rewards are judged a valid ordering")
        check(report["decision_phase"] == "P3", "P3 supersedes P2 for the decision")

        phase1 = [json.loads(l) for l in
                  (probe_dir / "recommended_phase1_train.jsonl").read_text(
                      encoding="utf-8").splitlines() if l.strip()]
        deferred = [json.loads(l) for l in
                    (probe_dir / "deferred_phase2_tasks.jsonl").read_text(
                        encoding="utf-8").splitlines() if l.strip()]
        check(80 <= len(phase1) <= 120, f"phase1 file has 80..120 rows, got {len(phase1)}")
        check(len(phase1) + len(deferred) == 160,
              "phase1 + deferred covers every task exactly once")
        src_first = json.loads(data.read_text(encoding="utf-8").splitlines()[0])
        by_id = {r["sample_id"]: r for r in phase1 + deferred}
        check(by_id[src_first["sample_id"]] == src_first,
              "subset rows are verbatim copies of the frozen source rows")

        md = (probe_dir / "SIGNAL_PROBE_REPORT.md").read_text(encoding="utf-8")
        for needle in ("VERDICT", "dead-group rate", "terminal-mixed rate",
                       "process-only-mixed rate", "reward-range distribution",
                       "Reward ordering", "Group audit",
                       "Recommended Phase-1 subset", "0/8", "8/8"):
            check(needle in md, f"report markdown mentions {needle!r}")

        # Re-running must be idempotent (deterministic outputs).
        before = (probe_dir / "SIGNAL_PROBE_REPORT.json").read_text(encoding="utf-8")
        subprocess.run([sys.executable, str(BUNDLE / "signal_probe_analyze.py"),
                        "--mode", "report", "--probe-dir", str(probe_dir),
                        "--data", str(data)], capture_output=True, text=True)
        check((probe_dir / "SIGNAL_PROBE_REPORT.json").read_text(encoding="utf-8")
              == before, "analyze is idempotent")


def test_worker_dry_run() -> None:
    """The worker must resolve its whole plan without torch, vLLM or a GPU."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        data = root / "train_grpo_pilot2.jsonl"
        _fake_dataset(data, n=8)
        out = root / "probe" / "shard_p2_0.jsonl"
        rc = subprocess.run([sys.executable, str(BUNDLE / "signal_probe_worker.py"),
                             "--data", str(data), "--out", str(out),
                             "--phase", "P2", "--rollouts", "4",
                             "--shard-index", "0", "--shard-count", "4",
                             "--dry-run"], capture_output=True, text=True)
        check(rc.returncode == 0, f"worker dry run exits 0 ({rc.stderr[-800:]})")
        check("DRY RUN" in rc.stdout, "worker dry run says so")
        check("bfloat16" in rc.stdout, "worker reports the BF16 dtype")
        check("targeted_tool_data_factory" in rc.stdout
              or "trainer_adapter" in rc.stdout,
              "worker reports the factory registry")
        check(not out.exists(), "worker dry run writes no rollouts")


def test_worker_resume_cache() -> None:
    """Resume must read the existing shard and survive a truncated last line."""
    from signal_probe_worker import read_cached_rollouts

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        shard = root / "shard_p2_0.jsonl"
        with shard.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps({"cache_key": "k1", "task_id": "a"}) + "\n")
            fh.write(json.dumps({"cache_key": "k2", "task_id": "b"}) + "\n")
            fh.write('{"cache_key": "k3", "task_i')   # killed mid-write
        recs = read_cached_rollouts(shard)
        check(len(recs) == 2, f"truncated final record dropped, got {len(recs)}")
        check([r["cache_key"] for r in recs] == ["k1", "k2"],
              "intact records are preserved in order")

        data = root / "train.jsonl"
        _fake_dataset(data, n=8)
        rc = subprocess.run([sys.executable, str(BUNDLE / "signal_probe_worker.py"),
                             "--data", str(data), "--out", str(shard),
                             "--phase", "P2", "--rollouts", "4",
                             "--shard-index", "0", "--shard-count", "4",
                             "--resume", "--dry-run"],
                            capture_output=True, text=True)
        check(rc.returncode == 0, f"resume dry run exits 0 ({rc.stderr[-500:]})")
        check("resume: 2 cached rollouts" in rc.stdout,
              f"resume reports the cache size, got: {rc.stdout[-300:]}")
        check("would generate 6 of 8 rollouts" in rc.stdout,
              f"resume subtracts cached work from the plan, got: {rc.stdout[-300:]}")


def test_analyze_dry_run() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        data = root / "d.jsonl"
        _fake_dataset(data, n=4)
        probe_dir = root / "probe"
        rc = subprocess.run([sys.executable, str(BUNDLE / "signal_probe_analyze.py"),
                             "--mode", "report", "--probe-dir", str(probe_dir),
                             "--data", str(data), "--dry-run"],
                            capture_output=True, text=True)
        check(rc.returncode == 0, f"analyze dry run exits 0 ({rc.stderr[-500:]})")
        check("DRY RUN" in rc.stdout, "analyze dry run says so")
        check(not (probe_dir / "SIGNAL_PROBE_REPORT.md").exists(),
              "analyze dry run writes nothing")


def test_report_renders_without_p3() -> None:
    """P2-only evidence must still produce a complete report."""
    specs = _pool_specs(40)
    groups = [build_group(meta, _fake_records(meta, kind, 4), phase="P2")
              for meta, kind in specs]
    records: List[Dict[str, Any]] = []
    for meta, kind in specs:
        records.extend(_fake_records(meta, kind, 4))
    report = build_report(groups=groups, records=records,
                          provenance={"model": "m", "p2_rollouts": 4},
                          phase1=select_phase1(groups, target=100),
                          deferred_count=0)
    check(report["decision_phase"] == "P2", "P2 is the decision phase without P3")
    md = render_report_md(report)
    check("VERDICT" in md and "0/4" in md,
          "P2-only report renders its own rollout count in the buckets")


def main() -> int:
    tests = [
        test_arg_error_counts,
        test_correct_prefix_len,
        test_classify_failure,
        test_derive_rollout_metrics,
        test_group_metrics,
        test_reward_ordering_audit,
        test_select_p3,
        test_select_phase1,
        test_verdict,
        test_cache_key,
        test_turn_returns_matches_trainer,
        test_extract_task_meta,
        test_analyze_end_to_end,
        test_worker_dry_run,
        test_worker_resume_cache,
        test_analyze_dry_run,
        test_report_renders_without_p3,
    ]
    for fn in tests:
        print(f"[test] {fn.__name__}")
        fn()
    if FAILED:
        print(f"\n{len(FAILED)} CHECK(S) FAILED:")
        for msg in FAILED:
            print(f"  - {msg}")
        return 1
    print(f"\nok: all {len(tests)} signal-probe tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
