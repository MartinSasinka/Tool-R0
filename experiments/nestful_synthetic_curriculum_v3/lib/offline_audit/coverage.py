from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from lib.offline_audit import ARMS
from lib.offline_audit.paths import load_train_groups, train_log_path

FIELD_SCHEMA = {
    "raw_model_outputs": "MISSING",
    "parsed_calls": "MISSING",
    "actual_observations": "MISSING",
    "actual_executor_outcomes": "MISSING",
    "terminal_class_per_rollout": "MISSING",
    "failure_class_per_rollout": "MISSING",
    "episode_rewards": "AVAILABLE_EXACT",
    "turn_rewards": "AVAILABLE_EXACT",
    "returns_G_t": "RECONSTRUCTABLE",
    "normalized_advantages": "RECONSTRUCTABLE",
    "token_masks": "MISSING",
    "group_task_id": "AVAILABLE_EXACT",
    "rollout_index": "RECONSTRUCTABLE",
    "optimizer_step": "AGGREGATE_ONLY",
    "gradient_norm": "MISSING",
    "kl": "AVAILABLE_EXACT",
    "clipping_statistics": "AVAILABLE_EXACT",
    "learning_rate": "AVAILABLE_EXACT",
    "checkpoint_adapter_state": "AVAILABLE_EXACT",
    "completion_hashes": "AVAILABLE_EXACT",
    "strict_gold_trace_pass": "AGGREGATE_ONLY",
    "exec_failure_counts": "AVAILABLE_EXACT",
    "predicted_num_calls": "AVAILABLE_EXACT",
}


def coverage(runs_root: Path, seed: str, reports_dir: Path) -> Dict[str, Any]:
    per_arm: Dict[str, Any] = {}
    for arm in ARMS:
        groups = load_train_groups(train_log_path(runs_root, arm, seed))
        sample = groups[0] if groups else {}
        per_arm[arm] = {
            "n_groups": len(groups),
            "fields": dict(FIELD_SCHEMA),
            "train_log_sample_keys": sorted(sample.keys()) if sample else [],
        }
    payload = {
        "per_arm": per_arm,
        "global_note": (
            "Trajectory payloads for registry re-scoring are not stored in train_log; "
            "counterfactual cross-arm re-score is PARTIAL (hash-matched logged rewards only)."
        ),
    }
    (reports_dir / "data_coverage.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    lines = ["# Data coverage", "", "| field | status |", "|---|---|"]
    for k, v in FIELD_SCHEMA.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")
    lines.append("## Per arm group counts")
    for arm, d in per_arm.items():
        lines.append(f"- **{arm}**: {d['n_groups']} groups")
    (reports_dir / "DATA_COVERAGE.md").write_text("\n".join(lines), encoding="utf-8")
    return payload
