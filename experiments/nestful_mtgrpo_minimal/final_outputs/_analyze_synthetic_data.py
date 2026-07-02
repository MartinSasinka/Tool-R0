#!/usr/bin/env python3
"""One-off analysis of filtered_toolr0_synthetic data suitability."""
from __future__ import annotations

import collections
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "filtered_toolr0_synthetic"
NESTFUL = ROOT / "data" / "NESTFUL-main" / "data_v2" / "nestful_data.jsonl"
sys.path.insert(0, str(ROOT))

from data import normalize_task  # noqa: E402
from executor import (  # noqa: E402
    IBMFunctionRegistry,
    ToolExecutor,
    detect_ibm_functions_dir,
    matches_gold,
)

VAR_REF = re.compile(r"\$var_(\d+)")


def load_tasks(path: Path, limit: int | None = None) -> list:
    out = []
    with open(path, encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            if limit is not None and idx >= limit:
                break
            line = line.strip()
            if line:
                out.append(normalize_task(json.loads(line), idx))
    return out


def gold_tool_in_prompt(task: dict) -> bool:
    names = {t["name"] for t in task["tools"]}
    return all(c["name"] in names for c in task["gold_calls"])


def chain_deps_ok(task: dict) -> bool:
    for i, call in enumerate(task["gold_calls"]):
        for val in (call.get("arguments") or {}).values():
            if isinstance(val, str):
                for vid in VAR_REF.findall(val):
                    if int(vid) > i + 1:
                        return False
    return True


def replay_gold(task: dict, reg: IBMFunctionRegistry | None) -> str:
    ex = ToolExecutor(task, registry=reg, mode="full")
    obs = []
    for call in task["gold_calls"]:
        r = ex.execute(call)
        if r.error:
            return "exec_fail"
        obs.append(r.observation)
    final = obs[-1] if obs else None
    return "ok" if matches_gold(final, task["gold_answer"]) else "mismatch"


def main() -> None:
    funcs_dir = detect_ibm_functions_dir(
        explicit=str(ROOT / "data" / "NESTFUL-main" / "data_v2" / "executable_functions"),
        repo_root=str(ROOT),
    )
    reg = IBMFunctionRegistry(funcs_dir) if funcs_dir else None
    nestful_tasks = load_tasks(NESTFUL) if NESTFUL.is_file() else []

    print("=== DATASET INVENTORY ===")
    total = 0
    for stage in range(1, 7):
        p = DATA / f"epoch_{stage}_{stage}call.jsonl"
        if p.is_file():
            n = len(load_tasks(p))
            total += n
            print(f"  epoch_{stage}_{stage}call.jsonl: {n} tasks")
    all_p = DATA / "curriculum_toolr0_all.jsonl"
    if all_p.is_file():
        rows = load_tasks(all_p)
        stages = collections.Counter(t["num_calls"] for t in rows)
        print(f"  curriculum_toolr0_all.jsonl: {len(rows)} tasks stages={dict(stages)}")
    print(f"  TOTAL synthetic tasks (epoch files): {total}")

    print("\n=== STAGE QUALITY (IBM full replay) ===")
    for stage in range(1, 7):
        p = DATA / f"epoch_{stage}_{stage}call.jsonl"
        if not p.is_file():
            continue
        tasks = load_tasks(p)
        n = len(tasks)
        in_prompt = sum(gold_tool_in_prompt(t) for t in tasks)
        chain_ok = sum(chain_deps_ok(t) for t in tasks)
        tool_hits = tool_total = 0
        for t in tasks:
            for c in t["gold_calls"]:
                tool_total += 1
                if reg and reg.get(c["name"]):
                    tool_hits += 1
        rng = random.Random(42)
        sample = tasks if n <= 120 else rng.sample(tasks, 120)
        replay = collections.Counter(replay_gold(t, reg) for t in sample)
        print(
            f"  stage {stage}: n={n} | gold_tools_in_prompt={100*in_prompt/n:.1f}% "
            f"| chain_ok={100*chain_ok/n:.1f}% | ibm_registry={100*tool_hits/tool_total:.1f}% "
            f"| replay(sample={len(sample)})={dict(replay)}"
        )

    if nestful_tasks:
        print("\n=== NESTFUL REPLAY BASELINE (same metric) ===")
        by_stage: dict[int, list] = collections.defaultdict(list)
        for t in nestful_tasks:
            by_stage[t["num_calls"]].append(t)
        rng = random.Random(42)
        for stage in [2, 3, 4, 5, 6]:
            pool = by_stage.get(stage, [])
            if not pool:
                continue
            sample = rng.sample(pool, min(120, len(pool)))
            replay = collections.Counter(replay_gold(t, reg) for t in sample)
            print(f"  NESTFUL {stage}-call pool={len(pool)} replay(sample={len(sample)})={dict(replay)}")

        print("\n=== TRAIN vs IN-CURRICULUM EVAL SIZE ===")
        for stage in [1, 2, 3, 4, 5]:
            syn_n = len(load_tasks(DATA / f"epoch_{stage}_{stage}call.jsonl"))
            nest_n = len([t for t in nestful_tasks if t["num_calls"] == stage + 1])
            print(f"  train stage {stage}: {syn_n} synthetic | eval subset {stage+1}-call: {nest_n} NESTFUL")

        syn_tools = set()
        for stage in range(1, 7):
            for t in load_tasks(DATA / f"epoch_{stage}_{stage}call.jsonl"):
                for c in t["gold_calls"]:
                    syn_tools.add(c["name"])
        nest_tools = {c["name"] for t in nestful_tasks for c in t["gold_calls"]}
        syn_ids = set()
        for stage in range(1, 7):
            for t in load_tasks(DATA / f"epoch_{stage}_{stage}call.jsonl"):
                syn_ids.add(t["task_id"])
        nest_ids = {t["task_id"] for t in nestful_tasks}
        print("\n=== OVERLAP WITH NESTFUL BENCHMARK ===")
        print(f"  task_id overlap: {len(syn_ids & nest_ids)}")
        syn_q = {t["question"][:80].lower() for t in load_tasks(DATA / "epoch_1_1call.jsonl")}
        for stage in range(2, 7):
            for t in load_tasks(DATA / f"epoch_{stage}_{stage}call.jsonl"):
                syn_q.add(t["question"][:80].lower())
        nest_q = {t["question"][:80].lower() for t in nestful_tasks}
        print(f"  question-prefix overlap (80 chars): {len(syn_q & nest_q)} / {len(syn_q)}")
        print(f"  gold tool names: syn={len(syn_tools)} nest={len(nest_tools)} overlap={len(syn_tools & nest_tools)}")

    print("\n=== GOLD ANSWER / FORMAT ISSUES ===")
    missing = unresolved = dup_q = 0
    all_q: list[str] = []
    for stage in range(1, 7):
        for t in load_tasks(DATA / f"epoch_{stage}_{stage}call.jsonl"):
            all_q.append(t["question"])
            if t["gold_answer"] is None:
                missing += 1
            elif isinstance(t["gold_answer"], str) and "$var_" in t["gold_answer"]:
                unresolved += 1
    dup_q = sum(1 for c in collections.Counter(all_q).values() if c > 1)
    print(f"  missing gold_answer: {missing}")
    print(f"  unresolved $var_ in gold_answer string: {unresolved}")
    print(f"  duplicate questions: {dup_q}")


if __name__ == "__main__":
    main()
