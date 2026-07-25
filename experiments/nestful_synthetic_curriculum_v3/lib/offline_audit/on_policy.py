from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from lib.offline_audit import ARMS, SYNTHETIC_SUCCESS_REWARD, WIN_REWARD
from lib.offline_audit.grpo_math import group_returns_and_advantages, rollout_scalar_advantage
from lib.offline_audit.paths import load_train_groups, train_log_path


def _is_success_reward(r: float) -> bool:
    return r >= SYNTHETIC_SUCCESS_REWARD


def _group_metrics(g: Dict[str, Any]) -> Dict[str, Any]:
    ep = [float(x) for x in g["episode_rewards"]]
    tr = [[float(x) for x in s] for s in g["turn_rewards"]]
    _, gstats = group_returns_and_advantages(tr, ep)
    n = len(ep)
    uniq = len(set(ep))
    dead = bool(g.get("dead_group"))
    mixed = bool(g.get("group_mixed"))
    all_succ = all(_is_success_reward(x) for x in ep)
    all_fail = all(not _is_success_reward(x) for x in ep)
    succ_counts = Counter(int(_is_success_reward(x)) for x in ep)
    n_success = succ_counts.get(1, 0)
    advs = [rollout_scalar_advantage(gstats, i) for i in range(n)]
    nonzero_adv_group = any(abs(a) > 1e-9 for a in advs)
    # proxy executable-wrong: high reward but not success
    exec_wrong = [
        i
        for i, r in enumerate(ep)
        if SYNTHETIC_SUCCESS_REWARD > r >= 0.35 and not _is_success_reward(r)
    ]
    success_idx = [i for i, r in enumerate(ep) if _is_success_reward(r)]
    succ_pos_adv = sum(1 for i in success_idx if advs[i] > 0)
    succ_neg_adv = sum(1 for i in success_idx if advs[i] < 0)
    ew_pos = sum(1 for i in exec_wrong if advs[i] > 0)
    n_rollouts = n
    parse_n = int(g.get("parse_error_count") or 0)
    no_call = int(g.get("no_tool_call_count") or 0)
    wrong_tool = int(g.get("wrong_tool_count") or 0)
    wrong_arg = int(g.get("wrong_arg_count") or 0)
    exec_fail = int(g.get("execfail_total") or 0)
    return {
        "mean_reward": sum(ep) / n if n else 0.0,
        "std_reward": (sum((x - sum(ep) / n) ** 2 for x in ep) / n) ** 0.5 if n else 0.0,
        "min_reward": min(ep) if ep else None,
        "max_reward": max(ep) if ep else None,
        "reward_range": (max(ep) - min(ep)) if ep else 0.0,
        "unique_rewards": uniq,
        "dead_group": dead,
        "mixed_group": mixed,
        "all_success": all_succ,
        "all_failure": all_fail,
        "n_success_rollouts": n_success,
        "nonzero_advantage_group": nonzero_adv_group,
        "terminal_success_rate": n_success / n if n else 0.0,
        "strict_gold_trace_pass": float(g.get("strict_gold_trace_pass") or 0.0),
        "parse_errors": parse_n,
        "no_call": no_call,
        "wrong_tool": wrong_tool,
        "wrong_arg": wrong_arg,
        "exec_fail": exec_fail,
        "n_rollouts": n_rollouts,
        "succ_pos_adv": succ_pos_adv,
        "succ_neg_adv": succ_neg_adv,
        "exec_wrong_rollouts": len(exec_wrong),
        "exec_wrong_pos_adv": ew_pos,
        "advs": advs,
        "ep": ep,
    }


def on_policy(runs_root: Path, seed: str, reports_dir: Path) -> Dict[str, Any]:
    by_arm: Dict[str, Any] = {}
    rows_csv: List[Dict[str, Any]] = []
    for arm in ARMS:
        groups = load_train_groups(train_log_path(runs_root, arm, seed))
        if not groups:
            continue
        gm = [_group_metrics(g) for g in groups]
        n_g = len(gm)
        n_roll = sum(x["n_rollouts"] for x in gm)
        dead_rate = sum(1 for x in gm if x["dead_group"]) / n_g
        mixed_rate = sum(1 for x in gm if x["mixed_group"]) / n_g
        eff_signal = sum(1 for x in gm if x["nonzero_advantage_group"]) / n_g
        term_succ = sum(x["n_success_rollouts"] for x in gm) / n_roll if n_roll else 0.0
        succ_neg = sum(x["succ_neg_adv"] for x in gm)
        succ_pos = sum(x["succ_pos_adv"] for x in gm)
        succ_total = sum(x["n_success_rollouts"] for x in gm)
        ew_pos = sum(x["exec_wrong_pos_adv"] for x in gm)
        ew_total = sum(x["exec_wrong_rollouts"] for x in gm)
        all_ep = [r for x in gm for r in x["ep"]]
        all_adv = [a for x in gm for a in x["advs"]]
        row = {
            "arm": arm,
            "n_groups": n_g,
            "n_rollouts": n_roll,
            "mean_reward": sum(all_ep) / len(all_ep) if all_ep else None,
            "std_reward": (
                (sum((x - sum(all_ep) / len(all_ep)) ** 2 for x in all_ep) / len(all_ep)) ** 0.5
                if all_ep
                else None
            ),
            "dead_group_rate": dead_rate,
            "mixed_group_rate": mixed_rate,
            "effective_signal_group_rate": eff_signal,
            "synthetic_terminal_success_rate": term_succ,
            "mean_strict_gold_trace_pass": sum(x["strict_gold_trace_pass"] for x in gm) / n_g,
            "success_negative_advantage_rate": (succ_neg / succ_total if succ_total else None),
            "success_positive_advantage_rate": (succ_pos / succ_total if succ_total else None),
            "executable_wrong_positive_advantage_rate": (ew_pos / ew_total if ew_total else None),
            "parse_error_rollouts_total": sum(x["parse_errors"] for x in gm),
            "exec_fail_rollouts_total": sum(x["exec_fail"] for x in gm),
            "advantage_mean_abs": sum(abs(a) for a in all_adv) / len(all_adv) if all_adv else None,
            "advantage_positive_rate": sum(1 for a in all_adv if a > 0) / len(all_adv) if all_adv else None,
        }
        by_arm[arm] = row
        rows_csv.append(row)

    json_path = reports_dir / "on_policy_metrics_by_arm.json"
    json_path.write_text(json.dumps(by_arm, indent=2), encoding="utf-8")
    csv_path = reports_dir / "on_policy_metrics_by_arm.csv"
    if rows_csv:
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows_csv[0].keys()))
            w.writeheader()
            w.writerows(rows_csv)

    md = [
        "# On-policy metrics by arm",
        "",
        "`synthetic_terminal_success_rate` is a reward-threshold PROXY "
        "(episode_reward >= 0.90 == v3_2_dense `fully_correct`, i.e. gold-trace "
        "match + final-answer pass). It is NOT the path-invariant terminal "
        "success check (`tool_final_answer_pass`).",
        "",
    ]
    for arm, r in by_arm.items():
        md.append(f"## {arm}")
        md.append(f"- reward-threshold success proxy (rollout): **{r['synthetic_terminal_success_rate']:.4f}**")
        md.append(f"- dead groups: **{r['dead_group_rate']:.4f}**")
        md.append(f"- mixed groups: **{r['mixed_group_rate']:.4f}**")
        md.append(f"- success w/ negative advantage: **{r['success_negative_advantage_rate']}**")
        md.append(f"- executable-wrong w/ positive advantage: **{r['executable_wrong_positive_advantage_rate']}**")
        md.append("")
    (reports_dir / "ON_POLICY_METRICS.md").write_text("\n".join(md), encoding="utf-8")
    return by_arm
