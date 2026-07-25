from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List

from lib.offline_audit import ARMS
from lib.offline_audit.grpo_math import group_returns_and_advantages
from lib.offline_audit.paths import load_train_groups, train_log_path


def credit_audit(runs_root: Path, seed: str, reports_dir: Path) -> Dict[str, Any]:
    focus = ["A0_R0_CURRENT", "A2_R3_OUTCOME_FIRST", "A4_GATED_VERIFIABLE"]
    metrics: Dict[str, Any] = {}
    examples: List[Dict[str, Any]] = []
    for arm in focus:
        groups = load_train_groups(train_log_path(runs_root, arm, seed))
        good_neg = 0
        bad_pos = 0
        total_turns = 0
        by_pos: Dict[int, List[float]] = {}
        for g in groups:
            ep = [float(x) for x in g["episode_rewards"]]
            tr = [[float(x) for x in s] for s in g["turn_rewards"]]
            _, gs = group_returns_and_advantages(tr, ep)
            for ci, (seq, advs) in enumerate(zip(tr, gs.advantages)):
                for t, (rt, adv) in enumerate(zip(seq, advs)):
                    total_turns += 1
                    by_pos.setdefault(t, []).append(float(adv))
                    if rt >= 0.7 and adv < 0:
                        good_neg += 1
                    if rt <= 0.3 and adv > 0:
                        bad_pos += 1
                    if len(examples) < 30 and ((rt >= 0.7 and adv < 0) or (rt <= 0.3 and adv > 0)):
                        examples.append(
                            {
                                "arm": arm,
                                "task_id": g.get("task_id"),
                                "rollout_index": ci,
                                "turn_index": t,
                                "turn_reward": rt,
                                "advantage": adv,
                                "dead_group": g.get("dead_group"),
                            }
                        )
        metrics[arm] = {
            "local_good_negative_advantage_rate": good_neg / total_turns if total_turns else None,
            "local_bad_positive_advantage_rate": bad_pos / total_turns if total_turns else None,
            "mean_abs_advantage_by_turn": {
                str(k): sum(abs(x) for x in v) / len(v) for k, v in sorted(by_pos.items())
            },
        }

    csv_path = reports_dir / "credit_assignment_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "good_neg_rate", "bad_pos_rate"])
        for arm, m in metrics.items():
            w.writerow([arm, m["local_good_negative_advantage_rate"], m["local_bad_positive_advantage_rate"]])

    ex_path = reports_dir / "credit_examples.jsonl"
    with open(ex_path, "w", encoding="utf-8") as fh:
        for ex in examples[:30]:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")

    md = [
        "# Credit assignment audit",
        "",
        "Local turn quality uses deterministic heuristics: good if r_t>=0.7, bad if r_t<=0.3.",
        "Per-call predicates are not in train_log.",
        "",
    ]
    for arm, m in metrics.items():
        md.append(f"## {arm}")
        md.append(f"- good turn, negative adv rate: {m['local_good_negative_advantage_rate']}")
        md.append(f"- bad turn, positive adv rate: {m['local_bad_positive_advantage_rate']}")
        md.append("")
    (reports_dir / "CREDIT_ASSIGNMENT_AUDIT.md").write_text("\n".join(md), encoding="utf-8")
    return {"metrics": metrics, "n_examples": len(examples)}
