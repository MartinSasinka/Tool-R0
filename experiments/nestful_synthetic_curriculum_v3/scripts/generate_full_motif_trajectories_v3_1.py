#!/usr/bin/env python3
"""Generate full synthetic trajectories from NESTFUL failure motifs (v3.1)."""
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

from motif_lib import repo_root  # noqa: E402
from question_templates_v3_1 import question_for_trajectory  # noqa: E402
from traj_utils_v3_1 import pack_trajectory, ref, validate_trajectory  # noqa: E402
from tool_registry_v3_1 import default_lookup_table  # noqa: E402

FAILURE_FAMILIES = [
    "linear_dependency__too_few_calls",
    "long_chain__too_few_calls",
    "fan_in__wrong_argument",
    "reference_reuse__invalid_reference",
    "object_list__wrong_extraction",
    "distractor_tools__wrong_tool",
    "independent_calls__premature_final",
    "string_output__wrong_answer",
    "list_output__wrong_field",
    "boolean_output__wrong_condition",
    "lookup_query__wrong_field",
]

SYNTHETIC_GAP_CLUSTERS = {
    "fan_in__wrong_argument",
    "reference_reuse__invalid_reference",
    "object_list__wrong_extraction",
    "distractor_tools__wrong_tool",
    "string_output__wrong_answer",
    "list_output__wrong_field",
    "boolean_output__wrong_condition",
}


def _label(i: int) -> str:
    return f"$var_{i}"


def _ri(rng: random.Random, lo: int = -12, hi: int = 25) -> int:
    return rng.randint(lo, hi)


def _pos(rng: random.Random, lo: int = 1, hi: int = 25) -> int:
    return rng.randint(lo, hi)


def _divisor(rng: random.Random) -> int:
    return rng.choice([2, 3, 4, 5, 10])


def _list_vals(rng: random.Random, n: Optional[int] = None) -> List[int]:
    n = n or rng.randint(3, 8)
    return [rng.randint(-5, 20) for _ in range(n)]


OBJECT_FIELDS = ["name", "score", "label", "status", "category", "value", "id"]
STRING_WORDS = ["tool", "nest", "call", "use", "ful", "ing", "data", "sync", "alpha", "beta"]


def _gen_linear(rng: random.Random, n: int = 3) -> List[dict]:
    a, b = _ri(rng), _ri(rng)
    calls = [{"name": "add", "arguments": {"arg_0": a, "arg_1": b}, "label": _label(1)}]
    for i in range(2, n + 1):
        calls.append({
            "name": "multiply",
            "arguments": {"arg_0": ref(i - 1), "arg_1": rng.choice([2, 3, 4, 5, -2])},
            "label": _label(i),
        })
    return calls


def _gen_long_chain(rng: random.Random, n: int) -> List[dict]:
    x = _ri(rng, 2, 20)
    calls = [{"name": "add", "arguments": {"arg_0": x, "arg_1": _pos(rng, 1, 5)}, "label": _label(1)}]
    for i in range(2, n + 1):
        op = rng.choice(["add", "multiply", "subtract", "divide_safe"])
        if op == "add":
            calls.append({"name": "add", "arguments": {"arg_0": ref(i - 1), "arg_1": 1}, "label": _label(i)})
        elif op == "subtract":
            calls.append({"name": "subtract", "arguments": {"arg_0": ref(i - 1), "arg_1": 1}, "label": _label(i)})
        elif op == "divide_safe":
            calls.append({"name": "divide_safe", "arguments": {"arg_0": ref(i - 1), "arg_1": _divisor(rng)}, "label": _label(i)})
        else:
            calls.append({
                "name": "multiply",
                "arguments": {"arg_0": ref(i - 1), "arg_1": 2},
                "label": _label(i),
            })
    return calls


def _gen_fan_in(rng: random.Random, n: int = 4) -> List[dict]:
    a, b, c = rng.randint(1, 10), rng.randint(1, 10), rng.randint(1, 10)
    calls = [
        {"name": "add", "arguments": {"arg_0": a, "arg_1": b}, "label": _label(1)},
        {"name": "add", "arguments": {"arg_0": ref(1), "arg_1": c}, "label": _label(2)},
        {"name": "multiply", "arguments": {"arg_0": ref(1), "arg_1": ref(2)}, "label": _label(3)},
    ]
    if n >= 4:
        calls.append({"name": "add", "arguments": {"arg_0": ref(3), "arg_1": ref(2)}, "label": _label(4)})
    if n >= 5:
        calls.append({"name": "multiply", "arguments": {"arg_0": ref(4), "arg_1": 2}, "label": _label(5)})
    return calls


def _gen_reference_reuse(rng: random.Random, n: int = 4) -> List[dict]:
    a, b = rng.randint(2, 12), rng.randint(2, 12)
    calls = [
        {"name": "add", "arguments": {"arg_0": a, "arg_1": b}, "label": _label(1)},
        {"name": "multiply", "arguments": {"arg_0": ref(1), "arg_1": ref(1)}, "label": _label(2)},
        {"name": "add", "arguments": {"arg_0": ref(1), "arg_1": ref(2)}, "label": _label(3)},
    ]
    if n >= 4:
        calls.append({"name": "multiply", "arguments": {"arg_0": ref(3), "arg_1": ref(1)}, "label": _label(4)})
    return calls


def _gen_object_list(rng: random.Random, n: int = 4) -> List[dict]:
    key = rng.choice(OBJECT_FIELDS)
    val = rng.choice(STRING_WORDS)
    vals = _list_vals(rng)
    calls = [
        {"name": "make_object", "arguments": {"key": key, "value": val}, "label": _label(1)},
        {"name": "get_field", "arguments": {"obj": ref(1), "key": key}, "label": _label(2)},
        {"name": "filter_greater_than", "arguments": {"values": vals, "threshold": rng.randint(2, 5)}, "label": _label(3)},
        {"name": "sort_list", "arguments": {"values": ref(3)}, "label": _label(4)},
    ]
    if n >= 5:
        calls.append({"name": "update_field", "arguments": {"obj": ref(1), "key": "score", "value": 99}, "label": _label(5)})
    return calls[:n]


def _gen_distractor(rng: random.Random, n: int = 4) -> List[dict]:
    return _gen_linear(rng, n)


def _gen_independent(rng: random.Random, n: int = 3) -> List[dict]:
    """Mostly dependent chain; last step may be independent only when n>=3."""
    a, b = rng.randint(1, 10), rng.randint(1, 10)
    calls = [{"name": "add", "arguments": {"arg_0": a, "arg_1": b}, "label": _label(1)}]
    for i in range(2, n + 1):
        if i == 2:
            calls.append({
                "name": "multiply",
                "arguments": {"arg_0": ref(1), "arg_1": rng.randint(2, 4)},
                "label": _label(i),
            })
        else:
            calls.append({
                "name": "add",
                "arguments": {"arg_0": ref(i - 1), "arg_1": rng.randint(1, 3)},
                "label": _label(i),
            })
    return calls


def _gen_string_chain(rng: random.Random, n: int = 3) -> List[dict]:
    w1, w2 = rng.choice(STRING_WORDS), rng.choice(STRING_WORDS)
    calls = [
        {"name": "concat", "arguments": {"a": w1, "b": w2}, "label": _label(1)},
    ]
    transform = rng.choice(["uppercase", "lowercase"])
    calls.append({"name": transform, "arguments": {"text": ref(1)}, "label": _label(2)})
    if n >= 3:
        calls.append({"name": "string_length", "arguments": {"text": ref(2)}, "label": _label(3)})
    if n >= 4:
        calls.append({"name": "add", "arguments": {"arg_0": ref(3), "arg_1": 1}, "label": _label(4)})
    if n >= 5:
        calls.append({"name": "extract_prefix", "arguments": {"text": ref(1), "n": 2}, "label": _label(5)})
    return calls[:n]


def _gen_list_chain(rng: random.Random, n: int = 4) -> List[dict]:
    vals = _list_vals(rng, rng.randint(4, 9))
    thr = rng.randint(-2, 8)
    calls = [
        {"name": "filter_greater_than", "arguments": {"values": vals, "threshold": thr}, "label": _label(1)},
        {"name": "sort_list", "arguments": {"values": ref(1)}, "label": _label(2)},
        {"name": "get_item", "arguments": {"values": ref(2), "index": rng.randint(0, min(2, len(vals) - 1))}, "label": _label(3)},
    ]
    if n >= 4:
        tail = rng.choice(["multiply", "join_list"])
        if tail == "join_list":
            calls.append({"name": "join_list", "arguments": {"values": ref(2), "sep": "-"}, "label": _label(4)})
        else:
            calls.append({"name": "multiply", "arguments": {"arg_0": ref(3), "arg_1": 2}, "label": _label(4)})
    if n >= 5:
        calls.append({"name": "list_length", "arguments": {"values": ref(1)}, "label": _label(5)})
    return calls[:n]


def _gen_boolean_chain(rng: random.Random, n: int = 3) -> List[dict]:
    a, b = rng.randint(1, 15), rng.randint(1, 15)
    calls = [
        {"name": "greater_than", "arguments": {"a": a, "b": b}, "label": _label(1)},
        {"name": "less_than", "arguments": {"a": b, "b": a}, "label": _label(2)},
    ]
    if n >= 3:
        calls.append({"name": "and_bool", "arguments": {"a": ref(1), "b": ref(2)}, "label": _label(3)})
    if n >= 4:
        calls.append({"name": "or_bool", "arguments": {"a": ref(3), "b": ref(1)}, "label": _label(4)})
    return calls[:n]


def _gen_lookup_chain(rng: random.Random, n: int = 4) -> List[dict]:
    table = default_lookup_table()
    key = rng.choice(list(table.keys()))
    recs = list(table.values())
    calls = [
        {"name": "lookup_by_key", "arguments": {"table": table, "key": key}, "label": _label(1)},
        {"name": "get_field", "arguments": {"obj": ref(1), "key": "score"}, "label": _label(2)},
        {"name": "count_records", "arguments": {"records": recs}, "label": _label(3)},
    ]
    if n >= 4:
        calls.append({"name": "aggregate_field", "arguments": {"records": recs, "field": "score"}, "label": _label(4)})
    if n >= 5:
        calls.append({"name": "add", "arguments": {"arg_0": ref(2), "arg_1": ref(3)}, "label": _label(5)})
    return calls[:n]


GENERATORS: Dict[str, Callable[[random.Random, int], List[dict]]] = {
    "linear_dependency__too_few_calls": lambda r, n: _gen_linear(r, max(2, min(n, 4))),
    "long_chain__too_few_calls": lambda r, n: _gen_long_chain(r, max(6, min(n, 9))),
    "fan_in__wrong_argument": lambda r, n: _gen_fan_in(r, max(3, min(n, 5))),
    "reference_reuse__invalid_reference": lambda r, n: _gen_reference_reuse(r, max(3, min(n, 5))),
    "object_list__wrong_extraction": lambda r, n: _gen_object_list(r, max(3, min(n, 5))),
    "distractor_tools__wrong_tool": lambda r, n: _gen_distractor(r, max(3, min(n, 5))),
    "independent_calls__premature_final": lambda r, n: _gen_independent(r, max(2, min(n, 4))),
    "string_output__wrong_answer": lambda r, n: _gen_string_chain(r, max(2, min(n, 4))),
    "list_output__wrong_field": lambda r, n: _gen_list_chain(r, max(3, min(n, 5))),
    "boolean_output__wrong_condition": lambda r, n: _gen_boolean_chain(r, max(2, min(n, 4))),
    "lookup_query__wrong_field": lambda r, n: _gen_lookup_chain(r, max(3, min(n, 5))),
}

CLUSTER_WEIGHTS: Dict[str, float] = {
    "linear_dependency__too_few_calls": 0.9,
    "long_chain__too_few_calls": 1.0,
    "fan_in__wrong_argument": 1.2,
    "reference_reuse__invalid_reference": 1.4,
    "object_list__wrong_extraction": 2.2,
    "distractor_tools__wrong_tool": 0.7,
    "independent_calls__premature_final": 0.2,
    "string_output__wrong_answer": 2.0,
    "list_output__wrong_field": 2.0,
    "boolean_output__wrong_condition": 1.8,
    "lookup_query__wrong_field": 1.6,
}


def _weighted_cluster_counts(clusters: List[str], total: int) -> Dict[str, int]:
    weights = {c: CLUSTER_WEIGHTS.get(c, 1.0) for c in clusters}
    denom = sum(weights.values()) or 1.0
    raw = {c: max(1, int(round(total * weights[c] / denom))) for c in clusters}
    while sum(raw.values()) < total:
        c = max(clusters, key=lambda x: weights[x])
        raw[c] += 1
    while sum(raw.values()) > total:
        c = max(clusters, key=lambda x: raw[x])
        if raw[c] > 1:
            raw[c] -= 1
        else:
            break
    return raw


MOTIF_MAP = {
    "linear_dependency__too_few_calls": "linear_dependency",
    "long_chain__too_few_calls": "long_chain",
    "fan_in__wrong_argument": "fan_in",
    "reference_reuse__invalid_reference": "reference_reuse",
    "object_list__wrong_extraction": "object_or_list_output",
    "distractor_tools__wrong_tool": "distractor_tools",
    "independent_calls__premature_final": "independent_calls",
    "string_output__wrong_answer": "reference_reuse",
    "list_output__wrong_field": "object_or_list_output",
    "boolean_output__wrong_condition": "independent_calls",
    "lookup_query__wrong_field": "reference_reuse",
}

FAMILY_TOOLS = {
    "linear_dependency__too_few_calls": ["math"],
    "long_chain__too_few_calls": ["math"],
    "fan_in__wrong_argument": ["math"],
    "reference_reuse__invalid_reference": ["math"],
    "object_list__wrong_extraction": ["object", "list", "math"],
    "distractor_tools__wrong_tool": ["math", "string", "list"],
    "independent_calls__premature_final": ["math"],
    "string_output__wrong_answer": ["string", "math"],
    "list_output__wrong_field": ["list", "math"],
    "boolean_output__wrong_condition": ["boolean", "math"],
    "lookup_query__wrong_field": ["lookup", "object", "math"],
}


def _resolve_cluster_id(cluster: str, known: set) -> str:
    if cluster in known:
        return cluster
    base = cluster.split("__")[0] if "__" in cluster else cluster
    for k in FAILURE_FAMILIES:
        if k.startswith(base):
            return k
    return f"synthetic_gap_{cluster}"


def generate_one(
    rng: random.Random,
    seed: int,
    cluster: str,
    *,
    num_calls: Optional[int] = None,
    known_clusters: Optional[set] = None,
) -> dict:
    known = known_clusters or set()
    src = cluster if cluster in GENERATORS else _resolve_cluster_id(cluster, known)
    if src not in GENERATORS:
        src = rng.choice(FAILURE_FAMILIES)
    gen = GENERATORS[src]
    if num_calls is None:
        if "long_chain" in src:
            num_calls = rng.randint(6, 9)
        elif "linear" in src or "independent" in src:
            num_calls = rng.randint(3, 5)
        else:
            num_calls = rng.randint(4, 6)
    calls = gen(rng, num_calls)
    motif = MOTIF_MAP.get(src, "long_chain")
    families = FAMILY_TOOLS.get(src, ["math"])
    with_distr = "distractor" in src
    traj_id = f"traj_v3_1_{src.replace('__', '_')}_{seed:06d}"
    display_cluster = src if src in known or src in FAILURE_FAMILIES else f"synthetic_gap_{src}"
    if src in SYNTHETIC_GAP_CLUSTERS and src not in known:
        display_cluster = f"synthetic_gap_{src}"
    return pack_trajectory(
        trajectory_id=traj_id,
        source_failure_cluster=display_cluster,
        target_full_motif=motif,
        question=question_for_trajectory(rng, calls, seed=seed),
        calls=calls,
        seed=seed,
        tool_families=families,
        with_distractors=with_distr,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path,
                    default=repo_root() / "experiments/nestful_synthetic_curriculum_v3/configs/curriculum_v3_1.yaml")
    ap.add_argument("--motif-config", type=Path,
                    default=repo_root() / "experiments/nestful_synthetic_curriculum_v3/configs/motif_generation.yaml")
    ap.add_argument("--specs", type=Path,
                    default=repo_root() / "experiments/nestful_synthetic_curriculum_v3/outputs/baseline_failure_motif_specs.json")
    ap.add_argument("--out-dir", type=Path,
                    default=repo_root() / "experiments/nestful_synthetic_curriculum_v3/outputs/curriculum_v3_1")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) if yaml and args.config.is_file() else {}
    gen_cfg = cfg.get("generation", {})
    seed_base = int(gen_cfg.get("random_seed", 31042))
    min_traj = int(gen_cfg.get("min_full_trajectories", 500))
    target_traj = int(gen_cfg.get("target_full_trajectories", 700))

    known_clusters = set()
    clusters = []
    if args.specs.is_file():
        for row in json.loads(args.specs.read_text(encoding="utf-8")):
            cid = row.get("cluster_id", "")
            if cid:
                known_clusters.add(cid)
                clusters.append(cid)
    for fam in FAILURE_FAMILIES:
        if fam not in clusters:
            clusters.append(fam)

    rng = random.Random(seed_base)
    trajectories: List[dict] = []
    sid = 0
    quota = _weighted_cluster_counts(clusters, target_traj)

    for cluster, count in quota.items():
        for _ in range(count):
            sid += 1
            try:
                t = generate_one(rng, seed_base + sid, cluster, known_clusters=known_clusters)
                errs = validate_trajectory(t)
                if not errs:
                    trajectories.append(t)
            except Exception:
                continue

    while len(trajectories) < min_traj:
        sid += 1
        cluster = rng.choice(clusters)
        try:
            t = generate_one(rng, seed_base + sid, cluster, known_clusters=known_clusters)
            if not validate_trajectory(t):
                trajectories.append(t)
        except Exception:
            continue
        if sid > min_traj * 3:
            break

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "full_trajectories.jsonl"
    with open(out_path, "w", encoding="utf-8") as fh:
        for t in trajectories:
            fh.write(json.dumps(t, ensure_ascii=False) + "\n")

    by_cluster: Dict[str, int] = {}
    by_motif: Dict[str, int] = {}
    for t in trajectories:
        by_cluster[t["source_failure_cluster"]] = by_cluster.get(t["source_failure_cluster"], 0) + 1
        by_motif[t["target_full_motif"]] = by_motif.get(t["target_full_motif"], 0) + 1

    manifest = {
        "total_trajectories": len(trajectories),
        "by_source_failure_cluster": by_cluster,
        "by_target_full_motif": by_motif,
        "call_count_histogram": {},
        "validation_failures": 0,
    }
    for t in trajectories:
        n = t["full_num_calls"]
        manifest["call_count_histogram"][str(n)] = manifest["call_count_histogram"].get(str(n), 0) + 1

    (args.out_dir / "full_trajectory_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"[generate_full_motif_trajectories_v3_1] wrote {len(trajectories)} -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
