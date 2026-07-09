#!/usr/bin/env python3
"""Generate motif-aligned synthetic tasks (mixed prototype tool registry)."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:
    import yaml
except ImportError:
    yaml = None

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from motif_lib import build_dependency_graph, extract_motifs, reference_pattern_stats, repo_root  # noqa: E402
from synthetic_tool_registry import (  # noqa: E402
    all_tool_defs,
    infer_answer_type,
    tool_pool_for_families,
)

NESTFUL_MOTIF_TYPES = (
    "linear_dependency", "long_chain", "fan_in", "fan_out", "independent_calls",
)


def _ref(var_idx: int, field: str = "result") -> str:
    return f"$var_{var_idx}.{field}$"


def _pack(
    motif_type: str,
    calls: list,
    ans: Any,
    seed: int,
    question: str,
    *,
    tool_families: Optional[List[str]] = None,
    output_type: Optional[str] = None,
    target_stage: Optional[str] = None,
) -> dict:
    families = tool_families or ["math"]
    tools = tool_pool_for_families(families)
    ref = reference_pattern_stats(calls)
    ans_type = infer_answer_type(ans)
    out_type = output_type or ans_type
    row = {
        "task_id": f"synthetic_v3_{motif_type}_{seed:06d}",
        "question": question,
        "tools": tools,
        "gold_calls": calls,
        "gold_answer": ans,
        "num_calls": len(calls),
        "motif_type": motif_type,
        "dependency_graph": build_dependency_graph(calls),
        "reference_pattern": ref,
        "output_type": out_type,
        "answer_type": ans_type,
        "generation_seed": seed,
        "source_motif_cluster": None,
        "tool_families_used": families,
        "target_stage": target_stage,
    }
    m = extract_motifs(row)
    row["difficulty_score"] = m["difficulty_score"]
    return row


def _gen_linear(rng: random.Random, seed: int, target_stage: str = "stage1_linear_simple") -> dict:
    a, b = rng.randint(1, 20), rng.randint(1, 20)
    calls = [
        {"name": "add", "arguments": {"arg_0": a, "arg_1": b}, "label": "$var_1"},
        {"name": "multiply", "arguments": {"arg_0": _ref(1), "arg_1": 3}, "label": "$var_2"},
    ]
    return _pack("linear_dependency", calls, (a + b) * 3, seed, f"Compute ({a}+{b})*3", target_stage=target_stage)


def _gen_independent_calls(rng: random.Random, seed: int, n_calls: int = 2, target_stage: str = "stage1_linear_simple") -> dict:
    calls, partials = [], []
    for i in range(n_calls):
        a, b = rng.randint(1, 12), rng.randint(1, 12)
        calls.append({"name": "add", "arguments": {"arg_0": a, "arg_1": b}, "label": f"$var_{i + 1}"})
        partials.append(a + b)
    return _pack(
        "independent_calls", calls, partials[-1], seed,
        f"Compute {n_calls} independent sums; return the last",
        target_stage=target_stage,
    )


def _gen_boolean_compare(rng: random.Random, seed: int) -> dict:
    a, b = rng.randint(1, 15), rng.randint(1, 15)
    result = a > b
    calls = [
        {"name": "greater_than", "arguments": {"a": a, "b": b}, "label": "$var_1"},
        {"name": "equals", "arguments": {"a": str(result).lower(), "b": str(result).lower()}, "label": "$var_2"},
    ]
    return _pack(
        "independent_calls", calls, True, seed, f"Verify comparison {a} > {b}",
        tool_families=["boolean", "string"], output_type="boolean", target_stage="stage1_linear_simple",
    )


def _gen_reference_reuse(rng: random.Random, seed: int) -> dict:
    a, b = rng.randint(2, 15), rng.randint(2, 15)
    calls = [
        {"name": "add", "arguments": {"arg_0": a, "arg_1": b}, "label": "$var_1"},
        {"name": "multiply", "arguments": {"arg_0": _ref(1), "arg_1": _ref(1)}, "label": "$var_2"},
        {"name": "add", "arguments": {"arg_0": _ref(1), "arg_1": _ref(2)}, "label": "$var_3"},
    ]
    s = a + b
    return _pack("reference_reuse", calls, s + s * s, seed, f"Given x={a}+{b}, compute x + x^2", target_stage="stage2_reference_reuse")


def _gen_fan_in(rng: random.Random, seed: int, target_stage: str = "stage2_reference_reuse") -> dict:
    a, b, c = rng.randint(1, 10), rng.randint(1, 10), rng.randint(1, 10)
    calls = [
        {"name": "add", "arguments": {"arg_0": a, "arg_1": b}, "label": "$var_1"},
        {"name": "multiply", "arguments": {"arg_0": a, "arg_1": c}, "label": "$var_2"},
        {"name": "add", "arguments": {"arg_0": _ref(1), "arg_1": _ref(2)}, "label": "$var_3"},
    ]
    motif = "simple_fan_in" if target_stage == "stage2_reference_reuse" else "fan_in"
    return _pack(motif, calls, (a + b) + (a * c), seed, "Combine two parallel computations", target_stage=target_stage)


def _gen_short_linear_ref(rng: random.Random, seed: int) -> dict:
    x = rng.randint(2, 9)
    calls = [
        {"name": "add", "arguments": {"arg_0": x, "arg_1": 1}, "label": "$var_1"},
        {"name": "multiply", "arguments": {"arg_0": _ref(1), "arg_1": 2}, "label": "$var_2"},
        {"name": "add", "arguments": {"arg_0": _ref(2), "arg_1": _ref(1)}, "label": "$var_3"},
    ]
    v1, v2 = x + 1, (x + 1) * 2
    return _pack("linear_dependency", calls, v2 + v1, seed, "Short linear chain with reuse", target_stage="stage2_reference_reuse")


def _gen_object_output(rng: random.Random, seed: int, target_stage: str = "stage2_reference_reuse") -> dict:
    vals = [rng.randint(1, 9) for _ in range(4)]
    calls = [
        {"name": "sum_list", "arguments": {"values": vals}, "label": "$var_1"},
        {"name": "scale", "arguments": {"arg_0": _ref(1), "factor": 2}, "label": "$var_2"},
    ]
    return _pack("object_or_list_output", calls, sum(vals) * 2, seed, f"Sum list {vals} and double", target_stage=target_stage)


def _gen_list_output(rng: random.Random, seed: int) -> dict:
    vals = [rng.randint(1, 8) for _ in range(5)]
    thr = rng.randint(2, 5)
    calls = [
        {"name": "filter_greater_than", "arguments": {"values": vals, "threshold": thr}, "label": "$var_1"},
        {"name": "sort_list", "arguments": {"values": _ref(1)}, "label": "$var_2"},
    ]
    ans = sorted([v for v in vals if v > thr])
    return _pack(
        "object_or_list_output", calls, ans, seed, f"Filter {vals} > {thr} and sort",
        tool_families=["list", "math"], output_type="list", target_stage="stage2_reference_reuse",
    )


def _gen_arg_transform(rng: random.Random, seed: int, target_stage: str = "stage2_reference_reuse") -> dict:
    a, b = rng.randint(2, 8), rng.randint(2, 8)
    calls = [
        {"name": "add", "arguments": {"arg_0": a, "arg_1": b}, "label": "$var_1"},
        {"name": "subtract", "arguments": {"arg_0": _ref(1), "arg_1": 1}, "label": "$var_2"},
        {"name": "divide", "arguments": {"arg_0": _ref(2), "arg_1": 2}, "label": "$var_3"},
    ]
    return _pack("argument_transformation", calls, ((a + b) - 1) / 2, seed, "Transform intermediate result", target_stage=target_stage)


def _gen_independent_aggregate(rng: random.Random, seed: int) -> dict:
    a, b, c, d = rng.randint(1, 9), rng.randint(1, 9), rng.randint(1, 9), rng.randint(1, 9)
    calls = [
        {"name": "add", "arguments": {"arg_0": a, "arg_1": b}, "label": "$var_1"},
        {"name": "add", "arguments": {"arg_0": c, "arg_1": d}, "label": "$var_2"},
        {"name": "add", "arguments": {"arg_0": _ref(1), "arg_1": _ref(2)}, "label": "$var_3"},
    ]
    return _pack("simple_fan_in", calls, (a + b) + (c + d), seed, "Aggregate two independent sums", target_stage="stage2_reference_reuse")


def _gen_string_chain(rng: random.Random, seed: int) -> dict:
    w1, w2 = rng.choice(["tool", "nest", "call"]), rng.choice(["use", "ful", "ing"])
    calls = [
        {"name": "concat", "arguments": {"a": w1, "b": w2}, "label": "$var_1"},
        {"name": "uppercase", "arguments": {"text": _ref(1)}, "label": "$var_2"},
    ]
    ans = (w1 + w2).upper()
    return _pack(
        "reference_reuse", calls, ans, seed, f"Concat '{w1}' and '{w2}' then uppercase",
        tool_families=["string"], output_type="string", target_stage="stage2_reference_reuse",
    )


def _gen_object_field(rng: random.Random, seed: int) -> dict:
    key, val = "name", rng.choice(["alpha", "beta", "gamma"])
    calls = [
        {"name": "make_object", "arguments": {"key": key, "value": val}, "label": "$var_1"},
        {"name": "get_field", "arguments": {"obj": _ref(1), "key": key}, "label": "$var_2"},
    ]
    return _pack(
        "object_or_list_output", calls, val, seed, f"Build object with {key}={val} and read field",
        tool_families=["object", "string"], output_type="string", target_stage="stage2_reference_reuse",
    )


def _gen_fan_out(rng: random.Random, seed: int) -> dict:
    x = rng.randint(3, 12)
    calls = [
        {"name": "scale", "arguments": {"arg_0": x, "factor": 2}, "label": "$var_1"},
        {"name": "add", "arguments": {"arg_0": _ref(1), "arg_1": 1}, "label": "$var_2"},
        {"name": "multiply", "arguments": {"arg_0": _ref(1), "arg_1": 3}, "label": "$var_3"},
    ]
    return _pack("fan_out", calls, x * 2 * 3, seed, f"From {x}, scale by 2 then triple scaled value", target_stage="stage3_structural_motifs")


def _gen_fan_in_deep(rng: random.Random, seed: int) -> dict:
    vals = [rng.randint(1, 6) for _ in range(4)]
    calls = [
        {"name": "add", "arguments": {"arg_0": vals[0], "arg_1": vals[1]}, "label": "$var_1"},
        {"name": "add", "arguments": {"arg_0": vals[2], "arg_1": vals[3]}, "label": "$var_2"},
        {"name": "multiply", "arguments": {"arg_0": _ref(1), "arg_1": 2}, "label": "$var_3"},
        {"name": "add", "arguments": {"arg_0": _ref(3), "arg_1": _ref(2)}, "label": "$var_4"},
    ]
    left = vals[0] + vals[1]
    ans = left * 2 + (vals[2] + vals[3])
    return _pack("fan_in", calls, ans, seed, "Four-way fan-in aggregation", target_stage="stage3_structural_motifs")


def _gen_merge_objects(rng: random.Random, seed: int) -> dict:
    calls = [
        {"name": "make_object", "arguments": {"key": "a", "value": "1"}, "label": "$var_1"},
        {"name": "make_object", "arguments": {"key": "b", "value": "2"}, "label": "$var_2"},
        {"name": "merge_objects", "arguments": {"a": _ref(1), "b": _ref(2)}, "label": "$var_3"},
        {"name": "get_field", "arguments": {"obj": _ref(3), "key": "b"}, "label": "$var_4"},
    ]
    return _pack(
        "argument_transformation", calls, "2", seed, "Merge two objects and read field b",
        tool_families=["object", "string"], output_type="string", target_stage="stage3_structural_motifs",
    )


def _gen_long_chain(rng: random.Random, seed: int, n_steps: int = 5) -> dict:
    x = rng.randint(2, 5)
    calls = []
    for i in range(1, n_steps + 1):
        if i == 1:
            calls.append({"name": "add", "arguments": {"arg_0": x, "arg_1": 1}, "label": f"$var_{i}"})
        else:
            calls.append({"name": "add", "arguments": {"arg_0": _ref(i - 1), "arg_1": 1}, "label": f"$var_{i}"})
    return _pack("long_chain", calls, x + n_steps, seed, f"Increment {x} {n_steps} times", target_stage="stage4_nestful_like_mixed")


def _gen_distractor(rng: random.Random, seed: int) -> dict:
    task = _gen_linear(rng, seed, target_stage="stage4_nestful_like_mixed")
    task["motif_type"] = "distractor_tools"
    task["task_id"] = f"synthetic_v3_distractor_tools_{seed:06d}"
    task["tools"] = all_tool_defs()
    return task


def _gen_alt_trace(rng: random.Random, seed: int) -> dict:
    a, b = rng.randint(3, 12), rng.randint(3, 12)
    calls = [
        {"name": "multiply", "arguments": {"arg_0": a, "arg_1": b}, "label": "$var_1"},
        {"name": "add", "arguments": {"arg_0": _ref(1), "arg_1": 0}, "label": "$var_2"},
    ]
    return _pack("alternative_valid_traces", calls, a * b, seed, f"Compute {a}*{b}", target_stage="stage4_nestful_like_mixed")


def _gen_baseline_inspired(rng: random.Random, seed: int, cluster: dict | None) -> dict:
    motif = (cluster or {}).get("motif_type", "long_chain")
    typical = int(round(float((cluster or {}).get("typical_num_calls", 5))))
    if motif == "long_chain":
        task = _gen_long_chain(rng, seed, n_steps=max(5, min(typical, 7)))
    elif motif == "independent_calls":
        task = _gen_independent_calls(rng, seed, n_calls=max(2, typical), target_stage="stage4_nestful_like_mixed")
    elif motif == "fan_in":
        task = _gen_fan_in_deep(rng, seed)
    elif motif == "linear_dependency":
        task = _gen_long_chain(rng, seed, n_steps=4)
    else:
        task = _gen_long_chain(rng, seed, n_steps=5)
    task["motif_type"] = "baseline_failure_inspired"
    task["source_motif_cluster"] = (cluster or {}).get("cluster_id")
    task["source_motif_type"] = motif
    task["task_id"] = f"synthetic_v3_baseline_bf_{seed:06d}"
    task["target_stage"] = "stage4_nestful_like_mixed"
    task["generation_family"] = "baseline_failure_inspired"
    m = extract_motifs(task)
    task["difficulty_score"] = m["difficulty_score"]
    return task


def _weighted_quotas(total: int, counts: Dict[str, int], min_per: int = 1) -> Dict[str, int]:
    keys = list(NESTFUL_MOTIF_TYPES)
    denom = sum(max(counts.get(k, 0), 1) for k in keys) or 1
    raw = {
        k: max(min_per if counts.get(k, 0) > 0 else 0, int(round(total * counts.get(k, 0) / denom)))
        for k in keys
    }
    while sum(raw.values()) < total:
        raw[max(keys, key=lambda k: counts.get(k, 0))] += 1
    while sum(raw.values()) > total:
        k = max((x for x in keys if raw[x] > min_per), key=lambda x: raw[x], default=keys[0])
        raw[k] -= 1
    return raw


NESTFUL_GENERATORS = {
    "linear_dependency": lambda r, s: _gen_linear(r, s, "stage1_linear_simple"),
    "long_chain": lambda r, s: _gen_long_chain(r, s, n_steps=r.randint(5, 7)),
    "fan_in": lambda r, s: _gen_fan_in(r, s, "stage3_structural_motifs"),
    "fan_out": _gen_fan_out,
    "independent_calls": lambda r, s: _gen_independent_calls(r, s, 2, "stage1_linear_simple"),
}


def _gen_nestful_topup(rng: random.Random, seed: int, motif: str) -> dict:
    task = NESTFUL_GENERATORS.get(motif, NESTFUL_GENERATORS["linear_dependency"])(rng, seed)
    task["motif_type"] = motif
    task["generation_family"] = "nestful_topup"
    return task


STAGE_GENERATORS: Dict[str, List[Callable[[random.Random, int], dict]]] = {
    "stage1_linear_simple": [
        lambda r, s: _gen_linear(r, s, "stage1_linear_simple"),
        lambda r, s: _gen_independent_calls(r, s, 2, "stage1_linear_simple"),
        _gen_boolean_compare,
    ],
    "stage2_reference_reuse": [
        _gen_reference_reuse,
        _gen_fan_in,
        _gen_short_linear_ref,
        _gen_object_output,
        _gen_list_output,
        _gen_arg_transform,
        _gen_independent_aggregate,
        _gen_string_chain,
        _gen_object_field,
    ],
    "stage3_structural_motifs": [
        _gen_fan_out,
        _gen_fan_in_deep,
        _gen_merge_objects,
        lambda r, s: _gen_arg_transform(r, s, "stage3_structural_motifs"),
        lambda r, s: _gen_fan_in(r, s, "stage3_structural_motifs"),
    ],
    "stage4_nestful_like_mixed": [
        lambda r, s: _gen_long_chain(r, s, n_steps=5),
        lambda r, s: _gen_long_chain(r, s, n_steps=7),
        _gen_distractor,
        _gen_alt_trace,
        lambda r, s: _gen_independent_calls(r, s, 3, "stage4_nestful_like_mixed"),
    ],
}


def _weighted_pick(rng: random.Random, weights: Dict[str, float], generators: Dict[str, Callable]) -> Callable:
    keys = list(weights.keys())
    ws = [weights[k] for k in keys]
    choice = rng.choices(keys, weights=ws, k=1)[0]
    return generators[choice]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path,
                    default=repo_root() / "experiments/nestful_synthetic_curriculum_v3/configs/motif_generation.yaml")
    ap.add_argument("--out", type=Path,
                    default=repo_root() / "experiments/nestful_synthetic_curriculum_v3/outputs/synthetic_motif_tasks.jsonl")
    ap.add_argument("--curriculum-config", type=Path,
                    default=repo_root() / "experiments/nestful_synthetic_curriculum_v3/configs/curriculum_v3.yaml")
    ap.add_argument("--specs", type=Path, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) if yaml and args.config.is_file() else {}
    curr_cfg = yaml.safe_load(args.curriculum_config.read_text(encoding="utf-8")) if yaml else {}
    stage_mins = cfg.get("stage_minimums") or curr_cfg.get("stage_minimums") or {}
    total = int(cfg.get("generation", {}).get("total_tasks", sum(stage_mins.values()) or 700))
    seed_base = int(cfg.get("generation", {}).get("random_seed", 42))

    dist_path = repo_root() / (cfg.get("target_distribution", {}).get(
        "source", "experiments/nestful_synthetic_curriculum_v3/outputs/nestful_motif_distribution.json"
    ))
    nest_counts = {}
    if dist_path.is_file():
        nest_counts = json.loads(dist_path.read_text(encoding="utf-8")).get("motif_type", {})

    clusters = []
    specs_path = args.specs or (args.out.parent / "baseline_failure_motif_specs.json")
    if specs_path.is_file():
        clusters = json.loads(specs_path.read_text(encoding="utf-8"))

    rng = random.Random(seed_base)
    tasks: List[dict] = []
    sid = 0
    stage_counts: Dict[str, int] = {}

    for stage, minimum in stage_mins.items():
        gens = STAGE_GENERATORS.get(stage, STAGE_GENERATORS["stage4_nestful_like_mixed"])
        for i in range(int(minimum)):
            sid += 1
            if stage == "stage4_nestful_like_mixed" and i % 5 == 0 and clusters:
                tasks.append(_gen_baseline_inspired(rng, sid, clusters[i % len(clusters)]))
            else:
                tasks.append(gens[i % len(gens)](rng, sid))
        stage_counts[stage] = int(minimum)

    topup_n = int(cfg.get("nestful_topup_tasks", 0))
    if topup_n and nest_counts:
        quotas = _weighted_quotas(topup_n, nest_counts, min_per=1)
        for motif, count in quotas.items():
            for _ in range(count):
                sid += 1
                tasks.append(_gen_nestful_topup(rng, sid, motif))

    boost = cfg.get("nestful_motif_boost") or {}
    for motif, count in boost.items():
        for _ in range(int(count)):
            sid += 1
            tasks.append(_gen_nestful_topup(rng, sid, str(motif)))

    while len(tasks) < total:
        sid += 1
        stage = rng.choice(list(stage_mins.keys()))
        gens = STAGE_GENERATORS[stage]
        tasks.append(gens[sid % len(gens)](rng, sid))

    rng.shuffle(tasks)
    for i, t in enumerate(tasks, start=1):
        t["generation_seed"] = seed_base + i

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for t in tasks:
            fh.write(json.dumps(t, ensure_ascii=False) + "\n")

    summary = {"total": len(tasks), "stage_minimums": stage_counts, "stage_mins_config": stage_mins}
    (args.out.parent / "generation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[generate_motif_synthetic_tasks] wrote {len(tasks)} tasks -> {args.out}")
    print(f"  stage targets: {stage_counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
