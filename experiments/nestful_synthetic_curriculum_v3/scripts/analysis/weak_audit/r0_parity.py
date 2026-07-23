"""Verify R0 (execution_aware_v3_2_dense) parity vs logged training rewards."""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

_V3 = Path(__file__).resolve().parents[3]
if str(_V3) not in sys.path:
    sys.path.insert(0, str(_V3))

from weak_audit.io_utils import sha256_file, write_json
from weak_audit.paths import AuditPaths


@dataclass
class ParityReport:
    gate_passed: bool
    reward_label: str
    summary: Dict[str, Any]
    limitations: List[str]


def _load_train_groups(path: Path) -> List[dict]:
    groups: List[dict] = []
    if not path.is_file():
        return groups
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("episode_rewards"):
                groups.append(row)
    return groups


def _train_log_checks(groups: List[dict]) -> dict:
    n = 0
    raw_mismatch = 0
    mean_mismatch = 0
    policy_mismatch = 0
    max_abs_mean_err = 0.0
    for g in groups:
        n += 1
        logged = [float(x) for x in g["episode_rewards"]]
        raw = [float(x) for x in g.get("raw_episode_rewards") or logged]
        if logged != raw:
            raw_mismatch += 1
        mean_logged = sum(logged) / len(logged)
        mean_field = float(g.get("mean_reward", mean_logged))
        err = abs(mean_field - mean_logged)
        max_abs_mean_err = max(max_abs_mean_err, err)
        if err > 1e-6:
            mean_mismatch += 1
        pol = g.get("reward_train_policy") or g.get("reward_policy_resolved")
        if pol and pol != "execution_aware_v3_2_dense":
            policy_mismatch += 1
    return {
        "n_groups": n,
        "raw_episode_match_rate": 1.0 - (raw_mismatch / n if n else 0),
        "mean_reward_match_rate": 1.0 - (mean_mismatch / n if n else 0),
        "max_abs_mean_error": max_abs_mean_err,
        "policy_mismatch_groups": policy_mismatch,
    }


def _eval_recompute_sample(
    paths: AuditPaths,
    tasks: Dict[str, dict],
    *,
    max_rows: int = 500,
) -> dict:
    """Recompute R0 on eval trajectories (no logged R0 on eval to compare)."""
    from lib.reward_v3_2_dense import episode_turn_reward_seq  # noqa: WPS433
    from scripts.analysis.pure_stage3_diag_utils import traj_from_dict  # noqa: WPS433

    os.environ.setdefault("TRAIN_STAGE", "3")
    diffs: List[float] = []
    n = 0
    for arm_dir in (paths.eval_c0, paths.eval_e1, paths.eval_e2):
        p = arm_dir / "final_eval_trajectories.jsonl"
        if not p.is_file():
            continue
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                if n >= max_rows:
                    break
                row = json.loads(line)
                sid = row.get("sample_id") or (row.get("_traj") or {}).get("task_id")
                task = tasks.get(sid)
                if not task:
                    continue
                rw = episode_turn_reward_seq(traj_from_dict(row["_traj"]), task)
                r0 = float(rw.get("episode_reward") or 0.0)
                strict = float((row.get("_traj") or {}).get("reward_train_strict") or 0.0)
                diffs.append(abs(r0 - strict))
                n += 1
    return {
        "n_eval_rows_sampled": n,
        "mean_abs_diff_vs_reward_train_strict": (sum(diffs) / len(diffs) if diffs else None),
        "max_abs_diff_vs_reward_train_strict": (max(diffs) if diffs else None),
        "note": "eval stores reward_train_strict only; this is NOT training R0",
    }


def build_r0_parity_report(paths: AuditPaths) -> ParityReport:
    from lib.reward_v3_2_dense import episode_turn_reward_seq  # noqa: WPS433,F401
    _SCRIPTS = paths.v3 / "scripts"
    _MINIMAL = paths.repo / "experiments/nestful_mtgrpo_minimal"
    if str(_MINIMAL) not in sys.path:
        sys.path.insert(0, str(_MINIMAL))
    if str(_SCRIPTS) not in sys.path:
        sys.path.append(str(_SCRIPTS))
    from scripts.analysis.pure_stage3_diag_utils import load_tasks  # noqa: WPS433

    limitations: List[str] = []
    train_logs = [
        paths.run_dir / "epoch_1" / "train" / "train_log.jsonl",
        paths.run_dir / "epoch_2" / "train" / "train_log.jsonl",
    ]
    all_groups: List[dict] = []
    for tl in train_logs:
        if tl.is_file():
            all_groups.extend(_load_train_groups(tl))
        else:
            limitations.append(f"missing train log: {tl}")

    train_stats = _train_log_checks(all_groups)
    limitations.append(
        "Train rollout trajectories are not persisted; full logged-vs-recomputed "
        "R0 parity on every completion is impossible from artifacts alone."
    )

    tasks = load_tasks(paths.nestful_test)
    eval_stats = _eval_recompute_sample(paths, tasks)

    manifest_policy = None
    if paths.run_manifest.is_file():
        manifest = json.loads(paths.run_manifest.read_text(encoding="utf-8"))
        manifest_policy = (manifest.get("hyperparameters") or {}).get("reward_policy")

    gate_passed = (
        train_stats.get("raw_episode_match_rate", 0) == 1.0
        and train_stats.get("mean_reward_match_rate", 0) == 1.0
        and train_stats.get("policy_mismatch_groups", 1) == 0
        and manifest_policy == "execution_aware_v3_2_dense"
    )

    # Eval packets use offline recomputed R0 — not logged per-completion train reward.
    reward_label = (
        "execution_aware_v3_2_dense_recomputed_eval"
        if gate_passed
        else "counterfactual_reward_v3_2_dense"
    )
    if not gate_passed:
        limitations.append(
            "Packets must treat reward fields as counterfactual until trajectory-level "
            "train parity is verified."
        )
    else:
        limitations.append(
            "Train log scalars are internally consistent and policy matches manifest, "
            "but eval packet rewards are still recomputed on saved eval trajectories "
            "(not logged train-time scalars)."
        )

    summary = {
        "generated_at": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
        "run_id": paths.run_dir.name,
        "manifest_reward_policy": manifest_policy,
        "reward_module_sha256": sha256_file(
            paths.v3 / "lib" / "reward_v3_2_dense.py"
        ),
        "train_log_stats": train_stats,
        "eval_recompute_sample": eval_stats,
        "gate_passed": gate_passed,
        "reward_label_for_packets": reward_label,
        "ideal_gate_not_met_reason": (
            None
            if gate_passed
            else "trajectory-level logged R0 unavailable for direct diff"
        ),
    }
    return ParityReport(
        gate_passed=gate_passed,
        reward_label=reward_label,
        summary=summary,
        limitations=limitations,
    )


def render_parity_md(report: ParityReport) -> str:
    s = report.summary
    lines = [
        "# R0 reward parity report",
        "",
        f"**Gate passed:** {report.gate_passed}",
        f"**Packet reward label:** `{report.reward_label}`",
        "",
        "## Train log checks",
        "",
    ]
    for k, v in (s.get("train_log_stats") or {}).items():
        lines.append(f"- {k}: {v}")
    lines += ["", "## Eval sample (vs reward_train_strict — not training R0)", ""]
    for k, v in (s.get("eval_recompute_sample") or {}).items():
        lines.append(f"- {k}: {v}")
    lines += ["", "## Limitations", ""]
    for lim in report.limitations:
        lines.append(f"- {lim}")
    return "\n".join(lines)


def write_parity_outputs(paths: AuditPaths, report: ParityReport) -> None:
    paths.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(paths.out_dir / "r0_parity_report.json", report.summary)
    (paths.out_dir / "R0_PARITY.md").write_text(
        render_parity_md(report), encoding="utf-8"
    )
