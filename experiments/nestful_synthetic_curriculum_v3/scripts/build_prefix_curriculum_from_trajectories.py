#!/usr/bin/env python3
"""Decompose full trajectories into stage-wise prefix samples (v3.1)."""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
except ImportError:
    yaml = None

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from motif_lib import load_jsonl, repo_root  # noqa: E402
from question_templates_v3_1 import (  # noqa: E402
    is_non_scalar_answer,
    question_for_prefix,
    question_signature,
    stage2_task_category,
)
from generate_full_motif_trajectories_v3_1 import FAILURE_FAMILIES, generate_one  # noqa: E402
from tool_registry_v3_1 import default_lookup_table, infer_answer_type, tool_family  # noqa: E402
from traj_utils_v3_1 import pack_trajectory, ref, truncate_trajectory  # noqa: E402
from uniqueness_utils_v3_1 import StageDedupRegistry, analyze_all_stages, compute_signatures  # noqa: E402

MAX_DEDUP_RETRIES = 50
STAGE_PREFIX_LEN = {
    "stage1_1call_atomic": 1,
    "stage2_2call_dependency": 2,
    "stage3_3call_composition": 3,
    "stage4_4to6call_persistence": 4,
}
STAGE2_CLUSTER_PREF = {
    "reference_dependency": (
        "reference_reuse__invalid_reference",
        "fan_in__wrong_argument",
        "linear_dependency__too_few_calls",
        "long_chain__too_few_calls",
    ),
    "transform": (
        "string_output__wrong_answer",
        "list_output__wrong_field",
        "object_list__wrong_extraction",
        "boolean_output__wrong_condition",
    ),
    "independent": (
        "independent_calls__premature_final",
        "linear_dependency__too_few_calls",
    ),
}
NON_SCALAR_CLUSTERS = (
    "object_list__wrong_extraction",
    "string_output__wrong_answer",
    "list_output__wrong_field",
    "boolean_output__wrong_condition",
    "lookup_query__wrong_field",
)

STAGE_FILES = {
    "stage1_1call_atomic": "stage1_1call_atomic.jsonl",
    "stage2_2call_dependency": "stage2_2call_dependency.jsonl",
    "stage3_3call_composition": "stage3_3call_composition.jsonl",
    "stage4_4to6call_persistence": "stage4_4to6call_persistence.jsonl",
}

MOTIF_TYPE_MAP = {
    (1, "long_chain"): "atomic_from_long_chain",
    (1, "linear_dependency"): "atomic_from_linear",
    (1, "fan_in"): "atomic_from_fan_in",
    (1, "object_or_list_output"): "atomic_from_object_list",
    (1, "distractor_tools"): "atomic_tool_selection_with_distractors",
    (2, "long_chain"): "two_call_long_chain_prefix",
    (2, "linear_dependency"): "two_call_linear_prefix",
    (2, "reference_reuse"): "two_call_reference_passing",
    (2, "object_or_list_output"): "two_call_object_field_prefix",
    (3, "long_chain"): "three_call_long_chain_prefix",
    (3, "linear_dependency"): "three_call_linear_chain",
    (3, "fan_in"): "three_call_fan_in",
    (3, "reference_reuse"): "three_call_reference_reuse",
    (3, "object_or_list_output"): "three_call_object_list",
    (4, "long_chain"): "four_to_six_call_long_chain",
    (5, "long_chain"): "four_to_six_call_long_chain",
    (6, "long_chain"): "four_to_six_call_long_chain",
}

STAGE2_TARGETS = {
    "reference_dependency": (0.70, 0.80),
    "independent": (0.10, 0.15),
    "transform": (0.10, 0.15),
}


def _motif_type(prefix_len: int, target_motif: str, stage: str) -> str:
    key = (prefix_len, target_motif)
    if key in MOTIF_TYPE_MAP:
        return MOTIF_TYPE_MAP[key]
    defaults = {
        "stage1_1call_atomic": "atomic_from_baseline_failure",
        "stage2_2call_dependency": "two_call_baseline_failure_prefix",
        "stage3_3call_composition": "three_call_argument_transformation",
        "stage4_4to6call_persistence": "baseline_failure_inspired",
    }
    return defaults.get(stage, f"prefix_{prefix_len}")


def _make_prefix_sample(
    traj: Dict[str, Any],
    prefix_len: int,
    stage: str,
    sample_idx: int,
    rng: random.Random,
    *,
    terminal: bool,
) -> Dict[str, Any]:
    truncated = truncate_trajectory(traj, prefix_len)
    motif = traj.get("target_full_motif", "long_chain")
    mt = _motif_type(prefix_len, motif, stage)
    seed = int(traj.get("generation_seed", 0)) + sample_idx * 17 + prefix_len
    gold_answer = truncated["last_observation"]
    question = question_for_prefix(
        rng,
        truncated["gold_calls"],
        prefix_len,
        stage,
        seed=seed,
        motif_type=mt,
        observations=truncated.get("observations"),
    )
    return {
        "sample_id": f"prefix_v3_1_{traj['trajectory_id']}_{stage}_p{prefix_len}_{sample_idx:05d}",
        "trajectory_id": traj["trajectory_id"],
        "stage": stage,
        "num_calls": prefix_len,
        "prefix_length": prefix_len,
        "source_prefix_length": prefix_len,
        "target_full_motif": motif,
        "source_failure_cluster": traj.get("source_failure_cluster"),
        "prefix_of_motif": True,
        "question": question,
        "tools": truncated["tools"],
        "gold_calls": truncated["gold_calls"],
        "observations": truncated["observations"],
        "gold_answer": gold_answer,
        "answer_type": infer_answer_type(gold_answer),
        "dependency_graph": truncated["dependency_graph"],
        "motif_type": mt,
        "process_labels": truncated["process_labels"],
        "terminal_stage": terminal,
        "generation_seed": seed,
        "output_type_sequence": truncated["output_type_sequence"],
    }


def decompose_trajectory(
    traj: Dict[str, Any], sample_counter: List[int], rng: random.Random
) -> Dict[str, List[dict]]:
    L = traj["full_num_calls"]
    out: Dict[str, List[dict]] = defaultdict(list)
    if L >= 1:
        sample_counter[0] += 1
        out["stage1_1call_atomic"].append(
            _make_prefix_sample(
                traj, 1, "stage1_1call_atomic", sample_counter[0], rng, terminal=True
            )
        )
    if L >= 2:
        sample_counter[0] += 1
        out["stage2_2call_dependency"].append(
            _make_prefix_sample(
                traj, 2, "stage2_2call_dependency", sample_counter[0], rng, terminal=True
            )
        )
    if L >= 3:
        sample_counter[0] += 1
        out["stage3_3call_composition"].append(
            _make_prefix_sample(
                traj, 3, "stage3_3call_composition", sample_counter[0], rng, terminal=True
            )
        )
    for pl in range(4, min(L, 6) + 1):
        sample_counter[0] += 1
        terminal = pl == L or pl == 6
        out["stage4_4to6call_persistence"].append(
            _make_prefix_sample(
                traj, pl, "stage4_4to6call_persistence", sample_counter[0], rng, terminal=terminal
            )
        )
    return out


def _stage_prefix_lengths(stage: str, rng: random.Random) -> List[int]:
    if stage == "stage4_4to6call_persistence":
        return [rng.randint(4, 6)]
    return [STAGE_PREFIX_LEN[stage]]


def _rerender_question(sample: dict, rng: random.Random, seed_offset: int = 0) -> dict:
    s = dict(sample)
    seed = int(s.get("generation_seed", 0)) + seed_offset
    s["generation_seed"] = seed
    s["question"] = question_for_prefix(
        rng,
        s["gold_calls"],
        int(s["num_calls"]),
        str(s["stage"]),
        seed=seed,
        motif_type=str(s.get("motif_type", "")),
        observations=s.get("observations"),
    )
    return s


def _try_register(registry: StageDedupRegistry, sample: dict, rng: random.Random) -> Optional[dict]:
    for attempt in range(MAX_DEDUP_RETRIES):
        candidate = _rerender_question(sample, rng, seed_offset=attempt * 997) if attempt else dict(sample)
        sigs = compute_signatures(candidate)
        if registry.can_add(sigs, candidate):
            registry.register(candidate, sigs)
            return candidate
    registry.note_warning(f"could not register unique sample for {sample.get('stage')} id={sample.get('sample_id')}")
    return None


def _dedupe_stage(samples: List[dict], stage: str, target: int, rng: random.Random) -> Tuple[List[dict], StageDedupRegistry]:
    registry = StageDedupRegistry(
        stage, stage_target=target, max_trace_count=1, max_template_count=16, max_tool_seq_ratio=0.16,
    )
    out: List[dict] = []
    for s in samples:
        kept = _try_register(registry, s, rng)
        if kept:
            out.append(kept)
    return out, registry


def _synthesize_stage1(rng: random.Random, idx: int) -> dict:
    """Diverse single-call atomic sample."""
    choice = rng.randint(0, 15)
    if choice == 0:
        calls = [{"name": "add", "arguments": {"arg_0": rng.randint(-15, 30), "arg_1": rng.randint(-15, 30)}, "label": "$var_1"}]
        fam, cluster, motif = ["math"], "linear_dependency__too_few_calls", "linear_dependency"
    elif choice == 1:
        calls = [{"name": "multiply", "arguments": {"arg_0": rng.randint(2, 12), "arg_1": rng.randint(2, 9)}, "label": "$var_1"}]
        fam, cluster, motif = ["math"], "linear_dependency__too_few_calls", "linear_dependency"
    elif choice == 2:
        calls = [{"name": "subtract", "arguments": {"arg_0": rng.randint(0, 25), "arg_1": rng.randint(1, 10)}, "label": "$var_1"}]
        fam, cluster, motif = ["math"], "linear_dependency__too_few_calls", "linear_dependency"
    elif choice == 3:
        calls = [{"name": "divide_safe", "arguments": {"arg_0": rng.randint(4, 40), "arg_1": rng.choice([2, 3, 4, 5])}, "label": "$var_1"}]
        fam, cluster, motif = ["math"], "long_chain__too_few_calls", "long_chain"
    elif choice == 4:
        vals = [rng.randint(-5, 20) for _ in range(rng.randint(3, 6))]
        calls = [{"name": "sum_list", "arguments": {"values": vals}, "label": "$var_1"}]
        fam, cluster, motif = ["math"], "long_chain__too_few_calls", "long_chain"
    elif choice == 5:
        w1, w2 = rng.choice(["tool", "nest", "call", "data"]), rng.choice(["use", "ful", "ing", "sync"])
        calls = [{"name": "concat", "arguments": {"a": w1, "b": w2}, "label": "$var_1"}]
        fam, cluster, motif = ["string"], "string_output__wrong_answer", "reference_reuse"
    elif choice == 6:
        text = rng.choice(["Hello", "World", "Nest", "Tool"])
        calls = [{"name": "uppercase", "arguments": {"text": text}, "label": "$var_1"}]
        fam, cluster, motif = ["string"], "string_output__wrong_answer", "reference_reuse"
    elif choice == 7:
        vals = [rng.randint(-5, 20) for _ in range(rng.randint(4, 7))]
        calls = [{"name": "filter_greater_than", "arguments": {"values": vals, "threshold": rng.randint(-3, 10)}, "label": "$var_1"}]
        fam, cluster, motif = ["list"], "list_output__wrong_field", "object_or_list_output"
    elif choice == 8:
        vals = [rng.randint(1, 9) for _ in range(rng.randint(3, 6))]
        calls = [{"name": "sort_list", "arguments": {"values": vals}, "label": "$var_1"}]
        fam, cluster, motif = ["list"], "list_output__wrong_field", "object_or_list_output"
    elif choice == 9:
        key = rng.choice(["name", "score", "label", "status"])
        val = rng.choice(["alpha", "beta", "gamma", "delta"])
        calls = [{"name": "make_object", "arguments": {"key": key, "value": val}, "label": "$var_1"}]
        fam, cluster, motif = ["object"], "object_list__wrong_extraction", "object_or_list_output"
    elif choice == 10:
        a, b = rng.randint(-5, 20), rng.randint(-5, 20)
        calls = [{"name": "greater_than", "arguments": {"a": a, "b": b}, "label": "$var_1"}]
        fam, cluster, motif = ["boolean"], "boolean_output__wrong_condition", "independent_calls"
    elif choice == 11:
        a, b = rng.randint(-5, 20), rng.randint(-5, 20)
        calls = [{"name": "less_than", "arguments": {"a": a, "b": b}, "label": "$var_1"}]
        fam, cluster, motif = ["boolean"], "boolean_output__wrong_condition", "independent_calls"
    elif choice == 12:
        table = default_lookup_table()
        key = rng.choice(list(table.keys()))
        calls = [{"name": "lookup_by_key", "arguments": {"table": table, "key": key}, "label": "$var_1"}]
        fam, cluster, motif = ["lookup"], "lookup_query__wrong_field", "reference_reuse"
    elif choice == 13:
        vals = [rng.randint(1, 9) for _ in range(rng.randint(3, 6))]
        calls = [{"name": "max_list", "arguments": {"values": vals}, "label": "$var_1"}]
        fam, cluster, motif = ["math"], "long_chain__too_few_calls", "long_chain"
    else:
        text = rng.choice(["alpha", "beta", "gamma", "tooluse"])
        calls = [{"name": "string_length", "arguments": {"text": text}, "label": "$var_1"}]
        fam, cluster, motif = ["string"], "string_output__wrong_answer", "reference_reuse"
    traj = pack_trajectory(
        trajectory_id=f"traj_v3_1_synth_s1_{idx:06d}",
        source_failure_cluster=f"synthetic_gap_{cluster}",
        target_full_motif=motif,
        question="placeholder",
        calls=calls,
        seed=120000 + idx,
        tool_families=fam,
    )
    return _make_prefix_sample(traj, 1, "stage1_1call_atomic", idx, rng, terminal=True)


def _synthesize_stage2_short(rng: random.Random, idx: int, *, ref_step: bool = True) -> dict:
    a, b = rng.randint(-12, 25), rng.randint(-12, 25)
    if ref_step:
        calls = [
            {"name": "add", "arguments": {"arg_0": a, "arg_1": b}, "label": "$var_1"},
            {"name": "multiply", "arguments": {"arg_0": ref(1), "arg_1": rng.choice([2, 3, 4, 5, -2])}, "label": "$var_2"},
        ]
        cluster = "reference_reuse__invalid_reference"
        motif = "reference_reuse"
    else:
        c, d = rng.randint(-12, 25), rng.randint(-12, 25)
        calls = [
            {"name": "add", "arguments": {"arg_0": a, "arg_1": b}, "label": "$var_1"},
            {"name": "add", "arguments": {"arg_0": c, "arg_1": d}, "label": "$var_2"},
        ]
        cluster = "independent_calls__premature_final"
        motif = "independent_calls"
    traj = pack_trajectory(
        trajectory_id=f"traj_v3_1_synth_s2_{idx:06d}",
        source_failure_cluster=f"synthetic_gap_{cluster}",
        target_full_motif=motif,
        question="placeholder",
        calls=calls,
        seed=130000 + idx,
        tool_families=["math"],
    )
    return _make_prefix_sample(traj, 2, "stage2_2call_dependency", idx, rng, terminal=True)


def _generate_unique_sample(
    stage: str,
    registry: StageDedupRegistry,
    rng: random.Random,
    counter: List[int],
    *,
    cluster: Optional[str] = None,
    prefix_len: Optional[int] = None,
    require_non_scalar: bool = False,
) -> Optional[dict]:
    clusters = [cluster] if cluster else list(FAILURE_CLUSTERS := FAILURE_FAMILIES)
    if require_non_scalar:
        clusters = [c for c in clusters if c in NON_SCALAR_CLUSTERS] or list(NON_SCALAR_CLUSTERS)
    for attempt in range(MAX_DEDUP_RETRIES):
        counter[0] += 1
        sid = counter[0]
        if stage == "stage1_1call_atomic" and attempt < 30:
            sample = _synthesize_stage1(rng, sid)
            kept = _try_register(registry, sample, rng)
            if kept:
                return kept
        if stage == "stage2_2call_dependency" and attempt < 20:
            sample = _synthesize_stage2_short(rng, sid, ref_step=(attempt % 2 == 0))
            kept = _try_register(registry, sample, rng)
            if kept:
                return kept
        cid = rng.choice(clusters)
        pl = prefix_len
        if pl is None:
            pl = _stage_prefix_lengths(stage, rng)[0]
        min_calls = pl if stage != "stage4_4to6call_persistence" else pl
        max_calls = pl if stage != "stage4_4to6call_persistence" else max(pl, 6)
        num_calls = rng.randint(min_calls, max(min_calls + 2, max_calls))
        try:
            traj = generate_one(rng, 880000 + counter[0], cid, num_calls=num_calls)
        except Exception:
            continue
        if traj.get("full_num_calls", 0) < pl:
            continue
        sample = _make_prefix_sample(
            traj,
            pl,
            stage,
            counter[0],
            rng,
            terminal=(pl >= traj.get("full_num_calls", pl)),
        )
        kept = _try_register(registry, sample, rng)
        if kept:
            return kept
    registry.note_warning(f"generate_unique_sample exhausted retries for {stage}")
    return None


def _fill_stage1_only_synth(
    registry: StageDedupRegistry,
    target: int,
    rng: random.Random,
    counter: List[int],
    existing: List[dict],
) -> List[dict]:
    out = list(existing)
    sid = counter[0]
    fails = 0
    while len(out) < target and fails < 8000:
        sid += 1
        sample = _synthesize_stage1(rng, sid)
        kept = _try_register(registry, sample, rng)
        if kept:
            out.append(kept)
            fails = 0
        else:
            fails += 1
            if fails % 50 == 0:
                registry.max_tool_seq_ratio = min(0.25, registry.max_tool_seq_ratio + 0.01)
                registry.max_template_count += 1
    counter[0] = sid
    return out[:target]


def _fill_stage2_synth(
    registry: StageDedupRegistry,
    target: int,
    rng: random.Random,
    counter: List[int],
    existing: List[dict],
) -> List[dict]:
    out = list(existing)
    sid = counter[0]
    fails = 0
    while len(out) < target and fails < 2500:
        sid += 1
        sample = _synthesize_stage2_short(rng, sid, ref_step=(sid % 3 != 0))
        kept = _try_register(registry, sample, rng)
        if kept:
            out.append(kept)
            fails = 0
        else:
            fails += 1
            if fails % 100 == 0:
                registry.max_tool_seq_ratio = min(0.22, registry.max_tool_seq_ratio + 0.02)
                registry.max_trace_count = 2
    counter[0] = sid
    return out[:target]


def _fill_stage_to_target(
    samples: List[dict],
    stage: str,
    target: int,
    rng: random.Random,
    counter: List[int],
    *,
    cluster: Optional[str] = None,
    require_non_scalar: bool = False,
) -> Tuple[List[dict], StageDedupRegistry]:
    deduped, registry = _dedupe_stage(samples, stage, target, rng)
    if stage == "stage1_1call_atomic":
        registry.max_tool_seq_ratio = 0.20
        if len(deduped) < target:
            deduped = _fill_stage1_only_synth(registry, target, rng, counter, deduped)
        return deduped[:target], registry
    prefix_pool = list(range(4, 7)) if stage == "stage4_4to6call_persistence" else [STAGE_PREFIX_LEN[stage]]
    idx = 0
    fails = 0
    while len(deduped) < target:
        pl = prefix_pool[idx % len(prefix_pool)]
        extra = _generate_unique_sample(
            stage,
            registry,
            rng,
            counter,
            cluster=cluster,
            prefix_len=pl,
            require_non_scalar=require_non_scalar and len(deduped) < int(target * 0.35),
        )
        if extra:
            deduped.append(extra)
            fails = 0
        else:
            fails += 1
            if stage == "stage1_1call_atomic":
                synth = _synthesize_stage1(rng, counter[0])
                counter[0] += 1
                kept = _try_register(registry, synth, rng)
                if kept:
                    deduped.append(kept)
                    fails = 0
            elif stage == "stage2_2call_dependency" and fails % 3 == 0:
                synth = _synthesize_stage2_short(rng, counter[0], ref_step=(fails % 2 == 0))
                counter[0] += 1
                kept = _try_register(registry, synth, rng)
                if kept:
                    deduped.append(kept)
                    fails = 0
            if fails >= 300:
                registry.max_trace_count = 2
                registry.max_tool_seq_ratio = 0.20
            if fails >= 800:
                break
        idx += 1
        if idx > target * MAX_DEDUP_RETRIES * 2:
            break
    if len(deduped) < target:
        registry.note_warning(f"{stage} filled {len(deduped)}/{target} unique samples")
    return deduped[:target], registry


def _synthesize_stage2_independent(rng: random.Random, idx: int) -> dict:
    """Pure 2-call independent task for stage2 balance."""
    a, b = rng.randint(-8, 20), rng.randint(-8, 20)
    c, d = rng.randint(-8, 20), rng.randint(-8, 20)
    while a == c and b == d:
        c, d = rng.randint(-8, 20), rng.randint(-8, 20)
    calls = [
        {"name": "add", "arguments": {"arg_0": a, "arg_1": b}, "label": "$var_1"},
        {"name": "add", "arguments": {"arg_0": c, "arg_1": d}, "label": "$var_2"},
    ]
    traj = pack_trajectory(
        trajectory_id=f"traj_v3_1_synth_indep_{idx:05d}",
        source_failure_cluster="synthetic_gap_stage2_independent",
        target_full_motif="independent_calls",
        question="placeholder",
        calls=calls,
        seed=90000 + idx,
        tool_families=["math"],
    )
    sample = _make_prefix_sample(
        traj, 2, "stage2_2call_dependency", idx, rng, terminal=True
    )
    sample["motif_type"] = "two_call_baseline_failure_prefix"
    return sample


def _rebalance_stage2(samples: List[dict], target: int, rng: random.Random, counter: List[int]) -> Tuple[List[dict], StageDedupRegistry]:
    buckets: Dict[str, List[dict]] = defaultdict(list)
    for s in samples:
        buckets[stage2_task_category(s)].append(s)

    n_indep = int(round(target * 0.125))
    n_transform = int(round(target * 0.125))
    n_ref = target - n_indep - n_transform

    registry = StageDedupRegistry("stage2_2call_dependency", stage_target=target, max_trace_count=1, max_template_count=12)
    out: List[dict] = []

    def _take(pool: List[dict], n: int) -> None:
        added = 0
        for s in pool:
            if added >= n or len(out) >= target:
                return
            kept = _try_register(registry, s, rng)
            if kept:
                out.append(kept)
                added += 1

    _take(buckets.get("reference_dependency", []), n_ref)
    _take(buckets.get("transform", []), n_transform)
    _take(buckets.get("independent", []), n_indep)

    quotas = (
        ("reference_dependency", n_ref, STAGE2_CLUSTER_PREF["reference_dependency"]),
        ("transform", n_transform, STAGE2_CLUSTER_PREF["transform"]),
        ("independent", n_indep, STAGE2_CLUSTER_PREF["independent"]),
    )
    for cat, need, clusters in quotas:
        have = sum(1 for s in out if stage2_task_category(s) == cat)
        while have < need and len(out) < target:
            if cat == "independent":
                synth = _synthesize_stage2_independent(rng, counter[0])
                counter[0] += 1
                kept = _try_register(registry, synth, rng)
                if kept:
                    out.append(kept)
                    have += 1
                    continue
            kept = _generate_unique_sample(
                "stage2_2call_dependency",
                registry,
                rng,
                counter,
                cluster=rng.choice(clusters),
            )
            if kept:
                out.append(kept)
                have += 1
            else:
                break

    while len(out) < target:
        kept = _generate_unique_sample("stage2_2call_dependency", registry, rng, counter)
        if kept:
            out.append(kept)
        else:
            break

    if len(out) < target:
        out = _fill_stage2_synth(registry, target, rng, counter, out)

    rng.shuffle(out)
    return out[:target], registry


def _compute_diversity_stats(
    all_samples: List[dict], trajectories: List[dict]
) -> dict:
    questions = [s.get("question", "") for s in all_samples]
    sigs = [question_signature(s) for s in all_samples]
    traj_ids = {s.get("trajectory_id") for s in all_samples}
    prefix_per_traj = Counter(s.get("trajectory_id") for s in all_samples)
    sig_counts = Counter(sigs)
    dup_rate = sum(c - 1 for c in sig_counts.values() if c > 1) / max(len(all_samples), 1)

    stage2 = [s for s in all_samples if s.get("stage") == "stage2_2call_dependency"]
    s2_cats = Counter(stage2_task_category(s) for s in stage2)
    s2_total = max(len(stage2), 1)

    s2plus = [s for s in all_samples if int(s.get("num_calls", 0)) >= 2]
    ns_share = sum(1 for s in s2plus if is_non_scalar_answer(s.get("gold_answer"))) / max(len(s2plus), 1)

    null_gold = sum(1 for s in all_samples if s.get("gold_answer") is None)

    return {
        "unique_questions": len(set(questions)),
        "total_prefix_samples": len(all_samples),
        "duplicate_signature_rate": round(dup_rate, 4),
        "unique_full_trajectories": len(trajectories),
        "unique_trajectories_in_prefixes": len(traj_ids),
        "mean_prefix_samples_per_trajectory": round(
            sum(prefix_per_traj.values()) / max(len(prefix_per_traj), 1), 2
        ),
        "prefix_samples_per_trajectory_histogram": dict(prefix_per_traj),
        "stage2_category_share": {
            k: round(v / s2_total, 4) for k, v in s2_cats.items()
        },
        "non_scalar_gold_answer_share_stage2_plus": round(ns_share, 4),
        "null_gold_answer_count": null_gold,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path,
                    default=repo_root() / "experiments/nestful_synthetic_curriculum_v3/outputs/curriculum_v3_1/full_trajectories.jsonl")
    ap.add_argument("--config", type=Path,
                    default=repo_root() / "experiments/nestful_synthetic_curriculum_v3/configs/curriculum_v3_1.yaml")
    ap.add_argument("--out-dir", type=Path,
                    default=repo_root() / "experiments/nestful_synthetic_curriculum_v3/outputs/curriculum_v3_1")
    args = ap.parse_args()

    before_path = args.out_dir / "dataset_uniqueness_summary.json"
    before_snapshot = None
    if before_path.is_file():
        before_snapshot = json.loads(before_path.read_text(encoding="utf-8"))

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) if yaml and args.config.is_file() else {}
    stages_cfg = cfg.get("stages", {})
    min_tasks = {k: int(v.get("min_tasks", 800)) for k, v in stages_cfg.items() if k.startswith("stage")}

    trajectories = load_jsonl(args.input)
    seed = int(cfg.get("generation", {}).get("random_seed", 31042))
    rng = random.Random(seed + 999)

    stage_samples: Dict[str, List[dict]] = {k: [] for k in STAGE_FILES}
    counter = [0]
    dedup_warnings: List[str] = []
    for traj in trajectories:
        for stage, samples in decompose_trajectory(traj, counter, rng).items():
            stage_samples[stage].extend(samples)

    stage_registries: Dict[str, StageDedupRegistry] = {}

    if "stage2_2call_dependency" in min_tasks:
        stage_samples["stage2_2call_dependency"], reg = _rebalance_stage2(
            stage_samples.get("stage2_2call_dependency", []),
            min_tasks["stage2_2call_dependency"],
            rng,
            counter,
        )
        stage_registries["stage2_2call_dependency"] = reg
        dedup_warnings.extend(reg.warnings)

    for stage in ("stage3_3call_composition", "stage4_4to6call_persistence"):
        if stage in min_tasks:
            filled, reg = _fill_stage_to_target(
                stage_samples.get(stage, []),
                stage,
                min_tasks[stage],
                rng,
                counter,
                require_non_scalar=True,
            )
            stage_samples[stage] = filled
            stage_registries[stage] = reg
            dedup_warnings.extend(reg.warnings)

    for stage, minimum in min_tasks.items():
        if stage in ("stage2_2call_dependency", "stage3_3call_composition", "stage4_4to6call_persistence"):
            continue
        filled, reg = _fill_stage_to_target(
            stage_samples.get(stage, []),
            stage,
            minimum,
            rng,
            counter,
        )
        stage_samples[stage] = filled
        stage_registries[stage] = reg
        dedup_warnings.extend(reg.warnings)

    all_samples = [s for samples in stage_samples.values() for s in samples]
    diversity = _compute_diversity_stats(all_samples, trajectories)
    uniqueness = analyze_all_stages(stage_samples)
    diversity["uniqueness"] = uniqueness["overall"]
    diversity["uniqueness_per_stage"] = {
        k: {
            "unique_question_ratio": v["unique_question_ratio"],
            "trace_duplicate_ratio": v["trace_duplicate_ratio"],
            "exact_duplicate_count": v["exact_duplicate_count"],
        }
        for k, v in uniqueness["per_stage"].items()
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for stage, fname in STAGE_FILES.items():
        path = args.out_dir / fname
        samples = stage_samples.get(stage, [])
        with open(path, "w", encoding="utf-8") as fh:
            for s in samples:
                fh.write(json.dumps(s, ensure_ascii=False) + "\n")
        counts[stage] = len(samples)

    manifest = {
        "version": "v3_1",
        "source_trajectories": len(trajectories),
        "stages": counts,
        "total_prefix_samples": sum(counts.values()),
        "diversity": diversity,
    }
    (args.out_dir / "curriculum_v3_1_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    report_lines = [
        "# Question Diversity Report (v3.1)",
        "",
        f"- unique_questions: {diversity['unique_questions']}",
        f"- duplicate_signature_rate: {diversity['duplicate_signature_rate']}",
        f"- unique_full_trajectories: {diversity['unique_full_trajectories']}",
        f"- mean_prefix_samples_per_trajectory: {diversity['mean_prefix_samples_per_trajectory']}",
        f"- stage2_category_share: {diversity['stage2_category_share']}",
        f"- non_scalar_gold_answer_share_stage2_plus: {diversity['non_scalar_gold_answer_share_stage2_plus']}",
        f"- null_gold_answer_count: {diversity['null_gold_answer_count']}",
    ]
    (args.out_dir / "QUESTION_DIVERSITY_REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    if dedup_warnings:
        (args.out_dir / "dedup_generation_warnings.txt").write_text("\n".join(dedup_warnings[:200]) + "\n", encoding="utf-8")

    imp_lines = [
        "# Uniqueness Improvement Report (v3.1)",
        "",
        "| Stage | before_uq_ratio | after_uq_ratio | before_trace_dup | after_trace_dup | exact_dup_after | status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for stage, after in uniqueness["per_stage"].items():
        before_uq = before_snapshot.get("per_stage", {}).get(stage, {}).get("unique_question_ratio", "n/a") if before_snapshot else "n/a"
        before_trace = before_snapshot.get("per_stage", {}).get(stage, {}).get("trace_duplicate_ratio", "n/a") if before_snapshot else "n/a"
        imp_lines.append(
            f"| {stage} | {before_uq} | {after['unique_question_ratio']} | {before_trace} | "
            f"{after['trace_duplicate_ratio']} | {after['exact_duplicate_count']} | {after['status']} |"
        )
    imp_lines += [
        "",
        f"- curriculum integrity preserved: stage counts={counts}",
        f"- dedup warnings: {len(dedup_warnings)}",
        f"- overall uniqueness status: {uniqueness['status']}",
    ]
    (args.out_dir / "UNIQUENESS_IMPROVEMENT_REPORT.md").write_text("\n".join(imp_lines) + "\n", encoding="utf-8")

    print(f"[build_prefix_curriculum] stages={counts}")
    print(f"[build_prefix_curriculum] unique_questions={diversity['unique_questions']} "
          f"exact_dup={uniqueness['overall']['exact_duplicate_count']} "
          f"mean_uq_ratio={uniqueness['overall']['mean_unique_question_ratio']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
