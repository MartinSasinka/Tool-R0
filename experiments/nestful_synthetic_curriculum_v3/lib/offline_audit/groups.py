from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from lib.offline_audit import ARMS
from lib.offline_audit.paths import load_train_groups, train_log_path

EXPECTED_ROLLOUTS = 8


def groups_inventory(runs_root: Path, seed: str, reports_dir: Path) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"per_arm": {}, "rows": []}
    csv_path = reports_dir / "group_inventory.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "arm",
                "task_id",
                "task_idx",
                "epoch",
                "n_rollouts",
                "complete_8",
                "duplicate_task_in_log",
                "missing_rollout_indices",
            ]
        )
        for arm in ARMS:
            groups = load_train_groups(train_log_path(runs_root, arm, seed))
            tid_counts = Counter(g.get("task_id") for g in groups)
            complete = 0
            incomplete = 0
            dup_tasks = 0
            for g in groups:
                tid = g.get("task_id")
                n = len(g.get("episode_rewards") or [])
                complete_flag = n == EXPECTED_ROLLOUTS
                if complete_flag:
                    complete += 1
                else:
                    incomplete += 1
                if tid_counts[tid] > 1:
                    dup_tasks += 1
                w.writerow(
                    [
                        arm,
                        tid,
                        g.get("task_idx"),
                        g.get("epoch"),
                        n,
                        complete_flag,
                        tid_counts[tid] > 1,
                        "",
                    ]
                )
                summary["rows"].append({"arm": arm, "task_id": tid, "n_rollouts": n})
            summary["per_arm"][arm] = {
                "n_groups": len(groups),
                "complete_8_8": complete,
                "incomplete": incomplete,
                "duplicate_task_rows": dup_tasks,
                "ambiguous_identity": dup_tasks > 0,
            }

    (reports_dir / "GROUP_RECONSTRUCTION.md").write_text(
        "\n".join(
            [
                "# Group reconstruction",
                "",
                "Group identity: `(arm, task_id)` from train_log rows.",
                f"Expected rollouts per group: **{EXPECTED_ROLLOUTS}**.",
                "",
            ]
            + [
                f"- **{arm}**: {d['n_groups']} groups, "
                f"complete {d['complete_8_8']}, incomplete {d['incomplete']}, "
                f"dup task rows {d['duplicate_task_rows']}"
                for arm, d in summary["per_arm"].items()
            ]
        ),
        encoding="utf-8",
    )
    json_path = reports_dir / "group_reconstruction.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
