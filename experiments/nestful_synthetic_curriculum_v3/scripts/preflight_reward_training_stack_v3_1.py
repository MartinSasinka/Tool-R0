#!/usr/bin/env python3
"""Preflight: dataset + reward + training-stack verification (post-audit).

Runs BEFORE any training and hard-fails when the stack cannot honour the
configured reward or the dataset gates fail. Checks:

  1. Dataset HARD gates (direct, on the stage JSONLs):
     counts >= 800 per stage, exact call counts (1/2/3, 4-6 for stage4),
     gold_answer nulls == 0, duplicate sample ids == 0, exact duplicates == 0.
     Plus deep-audit summary (final_dataset_audit_summary.json) when present:
     invalid refs / alignment / gold replay / leakage / placeholders.
  2. Reward dispatch smoke test (runs smoke_test_reward_dispatch_v3_1.py in a
     subprocess): resolved == configured, fractional rewards present.
  3. Dead-group proxy on 50 stage1 tasks: fraction of simulated 8-rollout
     groups with zero reward variance must not be catastrophic (>0.90).
  4. Reward component logging: required diagnostic keys present.
  5. Replay-ratio semantics: 0.20 on [stage1, stage2] must give a 20/80 mix.
  6. Metadata visibility: normalize_task preserves stage/motif metadata and
     the reward can infer stage from metadata or TRAIN_STAGE.
  7. Guard env: REGRESSION_GUARD=1, STAGE_GATES=1, ALLOW_STRICT_REWARD_FALLBACK=0.

Outputs:
  outputs/curriculum_v3_1/PREFLIGHT_REWARD_TRAINING_STACK.md
  outputs/curriculum_v3_1/preflight_reward_training_stack_summary.json
Exit 0 = pass, 1 = hard fail.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
V3 = HERE.parent
EXPERIMENTS = V3.parent
MINIMAL = EXPERIMENTS / "nestful_mtgrpo_minimal"
PARTIAL = EXPERIMENTS / "nestful_mtgrpo_partial"
for p in (str(MINIMAL), str(PARTIAL), str(V3), str(EXPERIMENTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

STAGE_FILES = {
    1: ("stage1_1call_atomic.jsonl", (1, 1)),
    2: ("stage2_2call_dependency.jsonl", (2, 2)),
    3: ("stage3_3call_composition.jsonl", (3, 3)),
    4: ("stage4_4to6call_persistence.jsonl", (4, 6)),
}

REQUIRED_DIAG_KEYS = (
    "reward_total", "reward_format", "reward_tool_match", "reward_arg_match",
    "reward_executable", "reward_final_answer", "reward_valid_refs",
    "reward_num_calls", "reward_premature_final", "reward_cap_reason",
    "reward_floor_reason", "reward_seen_stage", "reward_seen_num_calls",
    "reward_seen_motif_type", "reward_seen_terminal_stage", "turn_scores",
)


def _data_dir() -> Path:
    d = V3 / "outputs/curriculum_v3_1/filtered"
    if d.is_dir() and list(d.glob("stage*.jsonl")):
        return d
    return V3 / "outputs/curriculum_v3_1"


def _load_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def check_dataset_gates(results: dict) -> bool:
    from data import normalize_task
    data_dir = _data_dir()
    gates = {}
    ok = True
    seen_ids = set()
    dup_ids = 0
    exact_dups = 0
    seen_rows = set()
    null_answers = 0
    for stage, (fname, (lo, hi)) in STAGE_FILES.items():
        path = data_dir / fname
        if not path.is_file():
            gates[f"stage{stage}_file"] = {"pass": False, "detail": f"missing {path}"}
            ok = False
            continue
        rows = _load_jsonl(path)
        gates[f"stage{stage}_count_ge_800"] = {
            "pass": len(rows) >= 800, "detail": f"{len(rows)} samples"}
        ok &= len(rows) >= 800
        bad_calls = 0
        for i, row in enumerate(rows):
            task = normalize_task(row, i)
            n = task["num_calls"]
            if not (lo <= n <= hi):
                bad_calls += 1
            if task.get("gold_answer") is None:
                null_answers += 1
            sid = task["task_id"]
            if sid in seen_ids:
                dup_ids += 1
            seen_ids.add(sid)
            key = (task["question"], json.dumps(task["gold_calls"], sort_keys=True))
            if key in seen_rows:
                exact_dups += 1
            seen_rows.add(key)
        gates[f"stage{stage}_call_counts_{lo}_{hi}"] = {
            "pass": bad_calls == 0, "detail": f"{bad_calls} out-of-range rows"}
        ok &= bad_calls == 0
    gates["gold_answer_null_eq_0"] = {"pass": null_answers == 0,
                                      "detail": f"{null_answers} nulls"}
    gates["duplicate_sample_ids_eq_0"] = {"pass": dup_ids == 0,
                                          "detail": f"{dup_ids} duplicates"}
    gates["exact_duplicates_eq_0"] = {"pass": exact_dups == 0,
                                      "detail": f"{exact_dups} duplicates"}
    ok &= null_answers == 0 and dup_ids == 0 and exact_dups == 0

    # Deep-audit summary (produced by final_dataset_audit_v3_1.py).
    audit = V3 / "outputs/curriculum_v3_1/final_dataset_audit_summary.json"
    if audit.is_file():
        a = json.loads(audit.read_text(encoding="utf-8"))
        deep = {
            "invalid_reference_count": a.get("invalid_reference_count") == 0,
            "question_trace_alignment_failures": a.get("question_trace_alignment_failures") == 0,
            "gold_replay_success_rate": a.get("gold_replay_success_rate") == 1.0,
            "metadata_leakage_count": a.get("metadata_leakage_count") == 0,
            "unresolved_placeholder_count": a.get("unresolved_placeholder_count") == 0,
        }
        for k, v in deep.items():
            gates[f"audit_{k}"] = {"pass": bool(v), "detail": f"{k}={a.get(k)}"}
            ok &= bool(v)
    else:
        gates["audit_summary"] = {"pass": None,
                                  "detail": "final_dataset_audit_summary.json missing "
                                            "(run final_dataset_audit_v3_1.py)"}

    results["dataset_gates"] = gates
    results["dataset_hard_gates_pass"] = ok
    return ok


def check_reward_dispatch(results: dict, reward_policy: str) -> bool:
    proc = subprocess.run(
        [sys.executable, str(HERE / "smoke_test_reward_dispatch_v3_1.py"),
         "--reward-policy", reward_policy, "--n-tasks", "6"],
        capture_output=True, text=True)
    smoke_json = V3 / "outputs/curriculum_v3_1/smoke_test_reward_dispatch_summary.json"
    smoke = {}
    if smoke_json.is_file():
        smoke = json.loads(smoke_json.read_text(encoding="utf-8"))
    ok = proc.returncode == 0
    results["reward_dispatch_ok"] = ok
    results["reward_policy_configured"] = reward_policy
    results["reward_policy_resolved"] = smoke.get("reward_policy_resolved")
    results["resolved_reward_fn"] = smoke.get("resolved_reward_fn")
    results["fractional_rewards_present"] = bool(smoke.get("fractional_rewards_present"))
    results["smoke_test"] = {
        "returncode": proc.returncode,
        "n_unique_reward_values": smoke.get("n_unique_reward_values"),
        "failures": smoke.get("failures"),
    }
    if not ok:
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:])
    return ok and results["fractional_rewards_present"]


def check_dead_group_proxy(results: dict, reward_policy: str, n_tasks: int = 50) -> bool:
    """Model-free proxy: for each stage1 task simulate an 8-rollout group with
    a plausible mixture of failure modes; a group is 'dead' when all 8 rewards
    are identical. With the fixed graded reward this should be far below 0.90
    (the old strict-fallback setup was 1.00)."""
    sys.path.insert(0, str(HERE))
    from smoke_test_reward_dispatch_v3_1 import build_variants, load_stage_tasks
    import vllm_dp_pool

    fn, _ = vllm_dp_pool.resolve_reward_info(
        {"reward": {"train_policy": reward_policy}})
    data_dir = _data_dir()
    tasks = load_stage_tasks(data_dir / STAGE_FILES[1][0], n_tasks)
    os.environ["TRAIN_STAGE"] = "1"
    dead = 0
    group_plan = ["correct", "wrong_tool", "wrong_args", "wrong_final",
                  "no_tool_call", "parse_error", "wrong_args", "wrong_final"]
    for task in tasks:
        variants = build_variants(task)
        rewards = []
        for name in group_plan:
            traj = variants.get(name) or variants["correct"]
            rewards.append(round(float(fn(traj, task, None)["episode_reward"]), 6))
        if len(set(rewards)) == 1:
            dead += 1
    os.environ.pop("TRAIN_STAGE", None)
    rate = dead / max(1, len(tasks))
    results["dead_group_proxy_rate_stage1"] = rate
    results["dead_group_proxy_n_tasks"] = len(tasks)
    ok = rate <= 0.90
    print(f"[preflight] dead-group proxy (stage1, {len(tasks)} tasks): "
          f"{rate:.3f} {'OK' if ok else 'CATASTROPHIC'}")
    return ok


def check_component_logging(results: dict) -> bool:
    from smoke_test_reward_dispatch_v3_1 import build_variants, load_stage_tasks
    from lib.reward_v3_1 import execution_aware_v3_1_stepwise
    data_dir = _data_dir()
    tasks = load_stage_tasks(data_dir / STAGE_FILES[1][0], 1)
    traj = build_variants(tasks[0])["correct"]
    diag = execution_aware_v3_1_stepwise(traj, tasks[0], train_stage=1).diagnostics
    missing = [k for k in REQUIRED_DIAG_KEYS if k not in diag]
    results["reward_component_logging_ok"] = not missing
    results["reward_component_missing_keys"] = missing
    return not missing


def check_replay_ratio(results: dict) -> bool:
    from data import load_tasks_mixed
    data_dir = _data_dir()
    files = [str(data_dir / STAGE_FILES[1][0]), str(data_dir / STAGE_FILES[2][0])]
    try:
        mix = load_tasks_mixed(files, replay_ratio=0.20, seed=7)
    except Exception as exc:  # noqa: BLE001
        results["replay_ratio_ok"] = False
        results["replay_ratio_error"] = str(exc)
        return False
    eff = mix["effective_mix"]
    ok = abs(eff[0] - 0.20) <= 0.01 and abs(eff[1] - 0.80) <= 0.01
    # The old scalar bug must also stay fixed: a single weight must raise.
    try:
        load_tasks_mixed(files, weights=[0.20], seed=7)
        scalar_rejected = False
    except ValueError:
        scalar_rejected = True
    results["replay_ratio_ok"] = ok and scalar_rejected
    results["replay_effective_mix"] = eff
    results["replay_scalar_weight_rejected"] = scalar_rejected
    return ok and scalar_rejected


def check_metadata_visibility(results: dict) -> bool:
    from data import normalize_task
    from lib.reward_v3_1 import detect_stage
    data_dir = _data_dir()
    rows = _load_jsonl(data_dir / STAGE_FILES[2][0])[:5]
    ok = True
    for i, row in enumerate(rows):
        task = normalize_task(row, i)
        for key in ("stage", "terminal_stage", "motif_type"):
            if key in row and key not in task:
                ok = False
        if detect_stage(task) != "stage2":
            ok = False
    # TRAIN_STAGE env fallback path.
    os.environ["TRAIN_STAGE"] = "3"
    bare = {"gold_calls": [], "num_calls": 0}
    ok &= detect_stage(bare) == "stage3"
    os.environ.pop("TRAIN_STAGE", None)
    results["train_stage_metadata_visible"] = ok
    return ok


def check_guard_env(results: dict) -> bool:
    guard = os.environ.get("REGRESSION_GUARD", "1") == "1"
    gates = os.environ.get("STAGE_GATES", "1") == "1"
    no_fb = os.environ.get("ALLOW_STRICT_REWARD_FALLBACK", "0") != "1"
    results["regression_guard_enabled"] = guard
    results["stage_advancement_gates_enabled"] = gates
    results["strict_fallback_disallowed"] = no_fb
    results["checkpoint_guard_enabled"] = True  # enforced in code (checkpoint_eligibility.py)
    return guard and gates and no_fb


def write_report(results: dict, out_dir: Path) -> None:
    md = out_dir / "PREFLIGHT_REWARD_TRAINING_STACK.md"
    lines = [
        "# Preflight: reward + training stack (v3.1, post-audit)",
        "",
        f"- generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- status: **{results['status']}**",
        f"- reward policy: `{results.get('reward_policy_configured')}` -> "
        f"`{results.get('resolved_reward_fn')}`",
        "",
        "| check | result |",
        "|---|---|",
    ]
    for key in ("dataset_hard_gates_pass", "reward_dispatch_ok",
                "fractional_rewards_present", "dead_group_proxy_rate_stage1",
                "reward_component_logging_ok", "replay_ratio_ok",
                "train_stage_metadata_visible", "regression_guard_enabled",
                "stage_advancement_gates_enabled", "strict_fallback_disallowed",
                "checkpoint_guard_enabled"):
        lines.append(f"| {key} | {results.get(key)} |")
    lines += ["", "## Dataset gates", "", "| gate | pass | detail |", "|---|---|---|"]
    for name, g in (results.get("dataset_gates") or {}).items():
        lines.append(f"| {name} | {g['pass']} | {g['detail']} |")
    if results.get("hard_failures"):
        lines += ["", "## HARD FAILURES", ""]
        lines += [f"- {f}" for f in results["hard_failures"]]
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[preflight] report -> {md}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward-policy", default="execution_aware_v3_1_stepwise")
    ap.add_argument("--curriculum-version", default="v3_1")
    args = ap.parse_args()

    out_dir = V3 / "outputs/curriculum_v3_1"
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {"curriculum_version": args.curriculum_version}
    hard_failures = []

    print("[preflight] 1/7 dataset hard gates ...")
    if not check_dataset_gates(results):
        hard_failures.append("dataset_hard_gates_pass=false")
    print("[preflight] 2/7 reward dispatch smoke test ...")
    if not check_reward_dispatch(results, args.reward_policy):
        hard_failures.append("reward_dispatch_ok=false or fractional_rewards_present=false")
    print("[preflight] 3/7 dead-group proxy ...")
    if not check_dead_group_proxy(results, args.reward_policy):
        hard_failures.append("dead_group_proxy catastrophic (>0.90)")
    print("[preflight] 4/7 reward component logging ...")
    if not check_component_logging(results):
        hard_failures.append(f"reward diagnostics missing keys: "
                             f"{results.get('reward_component_missing_keys')}")
    print("[preflight] 5/7 replay-ratio semantics ...")
    if not check_replay_ratio(results):
        hard_failures.append("replay_ratio semantics broken")
    print("[preflight] 6/7 stage metadata visibility ...")
    if not check_metadata_visibility(results):
        hard_failures.append("train stage metadata not visible to reward")
    print("[preflight] 7/7 guard environment ...")
    if not check_guard_env(results):
        hard_failures.append("regression guard / stage gates / fallback env misconfigured")

    results["hard_failures"] = hard_failures
    results["status"] = "PASS" if not hard_failures else "FAIL"
    summary = out_dir / "preflight_reward_training_stack_summary.json"
    summary.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                       encoding="utf-8")
    print(f"[preflight] summary -> {summary}")
    write_report(results, out_dir)
    if hard_failures:
        print("[preflight] HARD FAIL:")
        for f in hard_failures:
            print(f"[preflight]   - {f}")
        return 1
    print("[preflight] PASS — stack verified, safe to start the smoke pilot")
    return 0


if __name__ == "__main__":
    sys.exit(main())
