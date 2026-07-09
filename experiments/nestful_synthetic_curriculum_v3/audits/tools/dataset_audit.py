#!/usr/bin/env python3
"""Read-only dataset audit for the NESTFUL synthetic curriculum experiments.

Compares:
  A) experiments/nestful_synthetic_curriculum_v3/outputs/curriculum_v3_1/filtered/stage*.jsonl
  B) experiments/nestful_mtgrpo_minimal/data/filtered_toolr0_synthetic/*.jsonl
  plus NESTFUL splits (dev/test/full) for overlap checks.

Writes DATASET_AUDIT.json next to the audits folder. Markdown is rendered
separately from the JSON. This script only READS the datasets.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from collections import Counter
from typing import Any, Dict, List, Optional

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
V3 = os.path.join(REPO, "experiments", "nestful_synthetic_curriculum_v3")
MINIMAL = os.path.join(REPO, "experiments", "nestful_mtgrpo_minimal")

A_DIR = os.path.join(V3, "outputs", "curriculum_v3_1", "filtered")
B_DIR = os.path.join(MINIMAL, "data", "filtered_toolr0_synthetic")
SPLITS = os.path.join(MINIMAL, "data", "splits")
NESTFUL_FULL = os.path.join(MINIMAL, "data", "NESTFUL-main", "data_v2", "nestful_data.jsonl")

A_FILES = [
    "stage1_1call_atomic.jsonl",
    "stage2_2call_dependency.jsonl",
    "stage3_3call_composition.jsonl",
    "stage4_4to6call_persistence.jsonl",
]
B_FILES = [
    "curriculum_toolr0_all.jsonl",
    "epoch_1_1call.jsonl",
    "epoch_2_2call.jsonl",
    "epoch_3_3call.jsonl",
    "epoch_4_4call.jsonl",
    "epoch_5_5call.jsonl",
    "epoch_6_6call.jsonl",
]

VAR_REF_RE = re.compile(r"\$var_?\d+(\.[A-Za-z0-9_]+)?\$")
LEAK_PATTERNS = ("motif", "cluster", "trajectory_id", "sample_id", "stage1", "stage2",
                 "stage3", "stage4", "prefix_of_motif", "failure_cluster")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def qhash(s: str) -> str:
    return hashlib.sha1(norm_text(s).encode("utf-8")).hexdigest()[:16]


def trace_hash(calls: Any) -> str:
    """Canonical hash of a gold call sequence: name + sorted arg keys + arg values."""
    if isinstance(calls, str):
        try:
            calls = json.loads(calls)
        except (json.JSONDecodeError, TypeError):
            return "unparseable"
    if not isinstance(calls, list):
        return "not-a-list"
    canon = []
    for c in calls:
        if not isinstance(c, dict):
            canon.append(str(c))
            continue
        args = c.get("arguments") or {}
        canon.append({
            "name": c.get("name"),
            "args": {k: args[k] for k in sorted(args)} if isinstance(args, dict) else args,
        })
    return hashlib.sha1(json.dumps(canon, sort_keys=True, ensure_ascii=False,
                                   default=str).encode("utf-8")).hexdigest()[:16]


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_question(row: Dict[str, Any]) -> str:
    return row.get("question") or row.get("input") or ""


def get_calls(row: Dict[str, Any]) -> Any:
    return row.get("gold_calls") if "gold_calls" in row else row.get("output")


def parse_tools(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    t = row.get("tools")
    if isinstance(t, str):
        try:
            t = json.loads(t)
        except (json.JSONDecodeError, TypeError):
            return []
    return t if isinstance(t, list) else []


def audit_file(path: str) -> Dict[str, Any]:
    rows = read_jsonl(path)
    n = len(rows)
    schema = Counter()
    call_counts = Counter()
    qhashes = []
    thashes = []
    sample_ids = []
    null_answers = 0
    unresolved_var_in_answer = 0
    tools_used = Counter()
    tools_offered = Counter()
    n_tools_offered = []
    motifs = Counter()
    clusters = Counter()
    answer_types = Counter()
    stages = Counter()
    leaks = Counter()
    tools_as_string = 0

    for row in rows:
        for k in row:
            schema[k] += 1
        calls = get_calls(row)
        if isinstance(calls, str):
            try:
                calls = json.loads(calls)
            except (json.JSONDecodeError, TypeError):
                calls = None
        ncalls = len(calls) if isinstance(calls, list) else -1
        call_counts[ncalls] += 1
        q = get_question(row)
        qhashes.append(qhash(q))
        thashes.append(trace_hash(get_calls(row)))
        sample_ids.append(str(row.get("sample_id") or row.get("id") or ""))

        ga = row.get("gold_answer")
        if ga is None:
            null_answers += 1
        elif VAR_REF_RE.search(json.dumps(ga, ensure_ascii=False, default=str)):
            unresolved_var_in_answer += 1

        if isinstance(calls, list):
            for c in calls:
                if isinstance(c, dict):
                    tools_used[c.get("name", "?")] += 1
        if isinstance(row.get("tools"), str):
            tools_as_string += 1
        toolz = parse_tools(row)
        n_tools_offered.append(len(toolz))
        for t in toolz:
            tools_offered[t.get("name", "?")] += 1

        if row.get("target_full_motif"):
            motifs[row["target_full_motif"]] += 1
        elif row.get("motif_type"):
            motifs[row["motif_type"]] += 1
        if row.get("source_failure_cluster"):
            clusters[row["source_failure_cluster"]] += 1
        if row.get("answer_type"):
            answer_types[row["answer_type"]] += 1
        if row.get("stage"):
            stages[row["stage"]] += 1

        ql = norm_text(q)
        for pat in LEAK_PATTERNS:
            if pat in ql:
                leaks[pat] += 1

    qc = Counter(qhashes)
    tc = Counter(thashes)
    ic = Counter(sample_ids)
    return {
        "path": os.path.relpath(path, REPO).replace("\\", "/"),
        "sha256": sha256_file(path),
        "rows": n,
        "schema_fields": {k: v for k, v in sorted(schema.items())},
        "call_count_distribution": {str(k): v for k, v in sorted(call_counts.items())},
        "unique_questions": len(qc),
        "duplicate_question_rows": n - len(qc),
        "top_dup_questions": qc.most_common(3) if n - len(qc) else [],
        "unique_gold_traces": len(tc),
        "duplicate_gold_trace_rows": n - len(tc),
        "unique_sample_ids": len(ic),
        "duplicate_sample_ids": n - len(ic),
        "null_gold_answers": null_answers,
        "gold_answer_contains_unresolved_var_ref": unresolved_var_in_answer,
        "tools_field_is_string_not_list": tools_as_string,
        "tools_used_distinct": len(tools_used),
        "tools_used_top": tools_used.most_common(15),
        "tools_offered_distinct": len(tools_offered),
        "avg_tools_offered_per_task": round(sum(n_tools_offered) / n, 2) if n else 0,
        "motif_distribution": dict(motifs.most_common()),
        "failure_cluster_distribution": dict(clusters.most_common()),
        "answer_type_distribution": dict(answer_types.most_common()),
        "stage_field_distribution": dict(stages.most_common()),
        "prompt_metadata_leaks": dict(leaks),
        "_qhashes": qhashes,
        "_thashes": thashes,
        "_sample_ids": sample_ids,
    }


def overlap(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    qa, qb = set(a["_qhashes"]), set(b["_qhashes"])
    ta, tb = set(a["_thashes"]), set(b["_thashes"])
    ia, ib = set(a["_sample_ids"]), set(b["_sample_ids"])
    return {
        "question_hash_overlap": len(qa & qb),
        "gold_trace_hash_overlap": len(ta & tb),
        "sample_id_overlap": len(ia & ib),
        "a_unique_questions": len(qa),
        "b_unique_questions": len(qb),
    }


def main() -> int:
    out: Dict[str, Any] = {"datasets": {}, "overlaps": {}, "notes": []}

    audited: Dict[str, Dict[str, Any]] = {}
    for name in A_FILES:
        p = os.path.join(A_DIR, name)
        if os.path.isfile(p):
            print(f"[audit] A: {name}")
            audited[f"A/{name}"] = audit_file(p)
    for name in B_FILES:
        p = os.path.join(B_DIR, name)
        if os.path.isfile(p):
            print(f"[audit] B: {name}")
            audited[f"B/{name}"] = audit_file(p)
    for name in ("nestful_dev.jsonl", "nestful_test.jsonl"):
        p = os.path.join(SPLITS, name)
        if os.path.isfile(p):
            print(f"[audit] NESTFUL split: {name}")
            audited[f"NESTFUL/{name}"] = audit_file(p)
    if os.path.isfile(NESTFUL_FULL):
        print("[audit] NESTFUL full benchmark")
        audited["NESTFUL/nestful_data.jsonl"] = audit_file(NESTFUL_FULL)

    # Pairwise overlaps of interest
    pairs = []
    a_keys = [k for k in audited if k.startswith("A/")]
    b_keys = [k for k in audited if k.startswith("B/")]
    n_keys = [k for k in audited if k.startswith("NESTFUL/")]
    for ak in a_keys:
        for bk in b_keys:
            pairs.append((ak, bk))
    for k in a_keys + b_keys:
        for nk in n_keys:
            pairs.append((k, nk))
    # A stage-level self-consistency: stage2 vs stage3 etc. (cross-stage dup check)
    for i in range(len(a_keys)):
        for j in range(i + 1, len(a_keys)):
            pairs.append((a_keys[i], a_keys[j]))
    for i in range(len(b_keys)):
        for j in range(i + 1, len(b_keys)):
            if "curriculum_toolr0_all" in b_keys[i] or "curriculum_toolr0_all" in b_keys[j]:
                continue
            pairs.append((b_keys[i], b_keys[j]))
    # all-file vs its epoch shards
    if "B/curriculum_toolr0_all.jsonl" in audited:
        for bk in b_keys:
            if bk != "B/curriculum_toolr0_all.jsonl":
                pairs.append(("B/curriculum_toolr0_all.jsonl", bk))

    for ak, bk in pairs:
        key = f"{ak} <-> {bk}"
        out["overlaps"][key] = overlap(audited[ak], audited[bk])

    for k, v in audited.items():
        v.pop("_qhashes"); v.pop("_thashes"); v.pop("_sample_ids")
        out["datasets"][k] = v

    dst = os.path.join(V3, "audits", "DATASET_AUDIT.json")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(f"[audit] wrote {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
