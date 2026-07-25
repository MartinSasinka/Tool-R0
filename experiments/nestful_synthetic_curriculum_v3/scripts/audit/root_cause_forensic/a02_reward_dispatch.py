"""A02 — Reward dispatch audit.

Proves (or refutes) from raw artifacts that all Round-1 arms trained with the
same reward function despite distinct configured arm policies:
  1. train_log.jsonl header `reward_dispatch` + per-row policy fields;
  2. console.log `[override] reward.train_policy` vs `[v3/run.py] training reward =`
     vs dp_worker resolved lines;
  3. hash-matched completions across arms must have IDENTICAL episode rewards
     if (and only if, up to coincidence) the same reward fn scored them;
  4. logged episode-reward values compared against the arm's INTENDED
     terminal scalars (A1/A2/A3/A4 are near-discrete; v3_2_dense is continuous).
"""
from __future__ import annotations

import io
import re
from collections import Counter
from typing import Any, Dict, List

from common import (ARMS, INTENDED_EPSILON, INTENDED_TERMINAL_SCALARS,
                    load_train_log, run_dir, write_json)


def _console_lines(arm: str) -> Dict[str, Any]:
    p = run_dir(arm) / "logs" / "console.log"
    out = {"override_line": None, "training_reward_line": None, "worker_resolved": []}
    if not p.is_file():
        return out
    with io.open(p, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if "[override] reward.train_policy" in line and out["override_line"] is None:
                out["override_line"] = line.strip()
            elif "training reward =" in line and out["training_reward_line"] is None:
                out["training_reward_line"] = line.strip()
            elif "resolved_policy=" in line and len(out["worker_resolved"]) < 3:
                m = re.search(r"reward\.train_policy=(\S+).*resolved_policy=(\S+)", line)
                if m:
                    out["worker_resolved"].append(
                        {"config_policy": m.group(1), "resolved_policy": m.group(2)})
    return out


def _nearest_intended_distance(r: float, arm: str) -> float:
    scal = INTENDED_TERMINAL_SCALARS.get(arm)
    if not scal:
        return float("nan")
    eps = INTENDED_EPSILON[arm]
    best = min(abs(r - s) for s in scal.values())
    return max(0.0, best - eps)  # process tie-break can move reward by <= eps


def main() -> Dict[str, Any]:
    per_arm: Dict[str, Any] = {}
    logs = {}
    for arm in ARMS:
        header, groups = load_train_log(arm)
        logs[arm] = groups
        policies = Counter()
        rewards: List[float] = []
        for g in groups:
            policies[g.get("reward_policy_resolved") or g.get("reward_train_policy")] += 1
            rewards.extend(float(x) for x in g["episode_rewards"])
        uniq = sorted(set(round(r, 6) for r in rewards))
        # near-discrete check: how many logged rewards sit within eps of the
        # arm's intended terminal scalars (should be ~100% if intended reward ran)
        if arm in INTENDED_TERMINAL_SCALARS:
            close = sum(1 for r in rewards if _nearest_intended_distance(r, arm) < 1e-6)
            frac_close = close / len(rewards) if rewards else None
        else:
            frac_close = None
        per_arm[arm] = {
            "header_reward_dispatch": (header or {}).get("reward_dispatch"),
            "row_policy_counts": dict(policies),
            "n_groups": len(groups),
            "n_rollouts": len(rewards),
            "n_unique_episode_rewards": len(uniq),
            "unique_episode_rewards_sample": uniq[:12],
            "frac_rewards_explainable_by_intended_scalars": frac_close,
            "console": _console_lines(arm),
        }

    # cross-arm hash-matched reward identity
    cross = []
    for i, a in enumerate(ARMS):
        for b in ARMS[i + 1:]:
            ga = {g["task_id"]: g for g in logs[a]}
            gb = {g["task_id"]: g for g in logs[b]}
            matched = 0
            identical = 0
            max_diff = 0.0
            for tid in set(ga) & set(gb):
                ha = list(ga[tid].get("completion_hashes") or [])
                hb = list(gb[tid].get("completion_hashes") or [])
                ea = [float(x) for x in ga[tid]["episode_rewards"]]
                eb = [float(x) for x in gb[tid]["episode_rewards"]]
                for ix, h in enumerate(ha):
                    if h in hb:
                        jx = hb.index(h)
                        matched += 1
                        d = abs(ea[ix] - eb[jx])
                        max_diff = max(max_diff, d)
                        if d < 1e-9:
                            identical += 1
            cross.append({"arm_a": a, "arm_b": b, "hash_matched": matched,
                          "identical_rewards": identical, "max_abs_diff": max_diff})

    all_resolved = {arm: list(per_arm[arm]["row_policy_counts"].keys()) for arm in ARMS}
    same_fn_everywhere = all(
        v == ["execution_aware_v3_2_dense"] for v in all_resolved.values())
    payload = {
        "per_arm": per_arm,
        "cross_arm_hash_matched_rewards": cross,
        "verdict": {
            "all_arms_resolved_to_v3_2_dense": same_fn_everywhere,
            "hash_matched_rewards_all_identical": all(
                c["hash_matched"] == c["identical_rewards"] for c in cross),
            "conclusion": (
                "CONFIRMED: every Round-1 arm trained with execution_aware_v3_2_dense; "
                "the reward-only ablation never varied the reward."
                if same_fn_everywhere else "dispatch differs across arms"),
        },
    }
    write_json("a02_reward_dispatch.json", payload)
    return payload


if __name__ == "__main__":
    r = main()
    print(r["verdict"])
    for c in r["cross_arm_hash_matched_rewards"]:
        print(c)
