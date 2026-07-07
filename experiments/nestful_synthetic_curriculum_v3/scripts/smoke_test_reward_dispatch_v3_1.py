#!/usr/bin/env python3
"""Reward-dispatch smoke test (audit Bug 2).

Verifies — BEFORE any GPU training — that:

  1. The configured reward policy resolves through the EXACT same resolver the
     DP rollout workers use (vllm_dp_pool.resolve_reward_info), NOT to the
     strict binary reward, with fallback_used=false.
  2. An unknown policy HARD-FAILS (no silent strict fallback).
  3. Scoring realistic trajectory variants on real stage1/stage2 tasks yields
     FRACTIONAL rewards (values other than 0/1) with several unique values.

Exit code 0 = pass; 1 = hard fail. Writes a JSON summary next to the dataset.

Usage:
    python smoke_test_reward_dispatch_v3_1.py \
        --reward-policy execution_aware_v3_1_stepwise [--n-tasks 8] [--out ...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V3 = HERE.parent
EXPERIMENTS = V3.parent
MINIMAL = EXPERIMENTS / "nestful_mtgrpo_minimal"
PARTIAL = EXPERIMENTS / "nestful_mtgrpo_partial"
for p in (str(MINIMAL), str(PARTIAL), str(V3), str(EXPERIMENTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from data import normalize_task  # noqa: E402  (minimal experiment loader)
from rollout import Trajectory, Turn  # noqa: E402
import vllm_dp_pool  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
#  Synthetic trajectory variants (deterministic, executor-free)
# ─────────────────────────────────────────────────────────────────────────────

def _turn_with_call(idx, call, fail_reason=None, observation=None):
    t = Turn(idx, model_text=json.dumps(call))
    t.parsed_call = dict(call)
    t.fail_reason = fail_reason
    t.observation = observation
    return t


def _terminal_turn(idx):
    t = Turn(idx, model_text="[]")
    t.is_terminal = True
    return t


def _mk_traj(task, turns, final_observation=None, stop_reason="terminal"):
    traj = Trajectory(task["task_id"], task["num_calls"], task["num_calls"])
    traj.turns = turns
    traj.final_observation = final_observation
    traj.stop_reason = stop_reason
    return traj


def build_variants(task):
    """Deterministic trajectory variants spanning the intended reward bands.

    Respects ``terminal_stage``: non-terminal (prefix) tasks must NOT end with
    a terminal/final turn — doing so is the premature_final failure mode (and
    is exercised separately below).
    """
    gold = task["gold_calls"]
    n = len(gold)
    terminal = bool(task.get("terminal_stage", True))
    variants = {}

    def _tail(idx):
        return [_terminal_turn(idx)] if terminal else []

    def _stop():
        return "terminal" if terminal else "max_turns"

    # 1. fully correct: gold calls, clean execution, gold final answer.
    turns = [_turn_with_call(i, c) for i, c in enumerate(gold)] + _tail(n)
    variants["correct"] = _mk_traj(task, turns, final_observation=task.get("gold_answer"),
                                   stop_reason=_stop())

    # 2. wrong tool on the first call.
    wrong = dict(gold[0]); wrong["name"] = "definitely_not_a_gold_tool"
    turns = [_turn_with_call(0, wrong)]
    turns += [_turn_with_call(i, c) for i, c in enumerate(gold[1:], start=1)]
    turns += _tail(n)
    variants["wrong_tool"] = _mk_traj(task, turns, final_observation=task.get("gold_answer"),
                                      stop_reason=_stop())

    # 3. correct tools, wrong argument values.
    turns = []
    for i, c in enumerate(gold):
        bad = dict(c)
        bad["arguments"] = {k: "WRONG_VALUE_XYZ" for k in (c.get("arguments") or {})}
        turns.append(_turn_with_call(i, bad))
    turns += _tail(n)
    variants["wrong_args"] = _mk_traj(task, turns, final_observation=None,
                                      stop_reason=_stop())

    # 4. executable gold trace but WRONG final answer.
    turns = [_turn_with_call(i, c) for i, c in enumerate(gold)] + _tail(n)
    variants["wrong_final"] = _mk_traj(task, turns, final_observation="WRONG_FINAL_ANSWER",
                                       stop_reason=_stop())

    # 5. too few calls (only for multi-call tasks).
    if n >= 2:
        turns = [_turn_with_call(0, gold[0])] + _tail(1)
        variants["too_few"] = _mk_traj(task, turns, final_observation=None,
                                       stop_reason=_stop())

    # 6. no tool call at all (immediate terminal).
    variants["no_tool_call"] = _mk_traj(task, [_terminal_turn(0)])

    # 7. parse error on the first turn.
    t = Turn(0, model_text="{ this is not json")
    t.fail_reason = "parse:invalid_json"
    variants["parse_error"] = _mk_traj(task, [t], stop_reason="parse_fail")

    # 8. premature final on a PREFIX (non-terminal) task — must be 0.0.
    if not terminal:
        turns = [_turn_with_call(i, c) for i, c in enumerate(gold)]
        turns.append(_terminal_turn(n))
        variants["premature_final"] = _mk_traj(
            task, turns, final_observation=task.get("gold_answer"))

    return variants


def load_stage_tasks(stage_file: Path, n: int):
    tasks = []
    with open(stage_file, "r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            tasks.append(normalize_task(json.loads(line), idx))
            if len(tasks) >= n:
                break
    return tasks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward-policy", default="execution_aware_v3_1_stepwise")
    ap.add_argument("--curriculum-version", default="v3_1")
    ap.add_argument("--n-tasks", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data_dir = V3 / "outputs/curriculum_v3_1/filtered"
    if not data_dir.is_dir() or not list(data_dir.glob("stage*.jsonl")):
        data_dir = V3 / "outputs/curriculum_v3_1"
    stage_files = {
        1: data_dir / "stage1_1call_atomic.jsonl",
        2: data_dir / "stage2_2call_dependency.jsonl",
    }
    out_path = Path(args.out) if args.out else \
        V3 / "outputs/curriculum_v3_1/smoke_test_reward_dispatch_summary.json"

    failures = []
    report = {"reward_policy_configured": args.reward_policy}

    # ── 1. Resolver check (exact DP-worker resolver) ─────────────────────────
    os.environ.pop("ALLOW_STRICT_REWARD_FALLBACK", None)
    config = {"reward": {"train_policy": args.reward_policy}}
    try:
        fn, info = vllm_dp_pool.resolve_reward_info(config)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] HARD FAIL: resolver raised for configured policy "
              f"'{args.reward_policy}': {exc}")
        report["resolver_error"] = str(exc)
        _write(out_path, report)
        return 1

    report.update({
        "reward_policy_resolved": info["resolved_policy"],
        "resolved_reward_fn": f"{info['reward_fn_module']}.{info['reward_fn_name']}",
        "fallback_used": info["fallback_used"],
    })
    print(f"[smoke] configured={args.reward_policy}")
    print(f"[smoke] resolved_reward_fn={report['resolved_reward_fn']}")
    print(f"[smoke] fallback_used={str(info['fallback_used']).lower()}")

    if info["fallback_used"]:
        failures.append("intended reward fell back to strict")
    if info["reward_fn_module"] == "reward":
        failures.append(f"policy '{args.reward_policy}' resolved to the STRICT reward")

    # ── 2. Unknown policy must hard-fail ─────────────────────────────────────
    try:
        vllm_dp_pool.resolve_reward_info(
            {"reward": {"train_policy": "definitely_unknown_policy_xyz"}})
        failures.append("unknown policy did NOT raise (silent fallback still present)")
        report["unknown_policy_raises"] = False
    except ValueError:
        report["unknown_policy_raises"] = True
        print("[smoke] unknown policy correctly raises ValueError (no silent fallback)")

    # ── 3. Score trajectory variants on real stage1 + stage2 tasks ──────────
    all_rewards = []
    per_variant = {}
    for stage, sf in stage_files.items():
        if not sf.is_file():
            failures.append(f"stage{stage} file missing: {sf}")
            continue
        os.environ["TRAIN_STAGE"] = str(stage)
        tasks = load_stage_tasks(sf, args.n_tasks)
        for task in tasks:
            for vname, traj in build_variants(task).items():
                try:
                    rinfo = fn(traj, task, None)
                except Exception as exc:  # noqa: BLE001
                    failures.append(
                        f"reward raised on stage{stage}/{vname}: "
                        f"{type(exc).__name__}: {exc}")
                    continue
                r = float(rinfo["episode_reward"])
                all_rewards.append(r)
                per_variant.setdefault(f"stage{stage}:{vname}", []).append(r)
    os.environ.pop("TRAIN_STAGE", None)

    unique = sorted({round(r, 6) for r in all_rewards})
    fractional = [r for r in unique if 0.0 < r < 1.0]
    only_binary = set(unique) <= {0.0, 1.0}

    report.update({
        "n_scored": len(all_rewards),
        "reward_values_unique": unique,
        "n_unique_reward_values": len(unique),
        "fractional_rewards_present": bool(fractional),
        "all_rewards_only_0_or_1": only_binary,
        "per_variant_mean": {k: round(sum(v) / len(v), 4)
                             for k, v in sorted(per_variant.items()) if v},
    })
    print(f"[smoke] scored {len(all_rewards)} trajectories; "
          f"{len(unique)} unique reward values")
    for k, v in sorted(per_variant.items()):
        print(f"[smoke]   {k:28s} mean={sum(v)/len(v):.4f} "
              f"min={min(v):.4f} max={max(v):.4f}")
    print(f"[smoke] fractional_rewards_present={bool(fractional)}  "
          f"only_binary={only_binary}")

    if only_binary:
        failures.append("graded reward produced ONLY 0/1 values — "
                        "matches the strict-fallback failure signature")
    if len(unique) < 3:
        failures.append(f"expected >2 unique reward values, got {len(unique)}")

    report["failures"] = failures
    report["status"] = "PASS" if not failures else "FAIL"
    _write(out_path, report)
    print(f"[smoke] summary -> {out_path}")
    if failures:
        print("[smoke] HARD FAIL:")
        for f in failures:
            print(f"[smoke]   - {f}")
        return 1
    print("[smoke] PASS — reward dispatch verified end-to-end")
    return 0


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    sys.exit(main())
