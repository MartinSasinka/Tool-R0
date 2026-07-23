"""Build compact case packets from eval trajectories."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Set

from weak_audit.compression import preserve_observation
from weak_audit.io_utils import sha256_file
from weak_audit.paths import AuditPaths

from lib.reward_v3_2_dense import episode_turn_reward_seq  # noqa: E402
from motif_lib import extract_motifs  # noqa: E402
from scripts.analysis.pure_stage3_diag_utils import (  # noqa: E402
    first_divergence_turn,
    load_tasks,
    load_traj_rows,
    predicted_calls,
    relevant_tools,
    tool_at,
    traj_from_dict,
)
from scripts.analysis.two_phase_root_cause_analysis import (  # noqa: E402
    classify_failure,
    official_win,
)


def _final_answer(row: dict) -> Any:
    traj = row.get("_traj") or {}
    ans = traj.get("pred_answer")
    if ans is not None:
        return ans
    for t in reversed(traj.get("turns") or []):
        if t.get("is_terminal") and t.get("model_text"):
            return t.get("model_text")
    return None


def _arm_block(row: dict, task: dict) -> dict:
    os.environ.setdefault("TRAIN_STAGE", "3")
    traj = traj_from_dict(row["_traj"])
    rw = episode_turn_reward_seq(traj, task)
    diag = rw.get("diagnostics") or {}
    executable = (row.get("_traj") or {}).get("executable")
    if executable is None:
        executable = diag.get("execution_score", 0) > 0
    return {
        "calls": predicted_calls(row),
        "observations": [
            preserve_observation(o) for o in _observations_raw(row)
        ],
        "final_answer": _final_answer(row),
        "official_win": official_win(row) == 1.0,
        "failure_class": classify_failure(row)[0],
        "executable": bool(executable) if executable is not None else None,
        "reward_total": float(rw.get("episode_reward") or 0.0),
        "reward_components": {
            k: diag.get(k)
            for k in (
                "reward_class", "quality_score", "format_score",
                "call_count_progress", "per_call_tool_score",
                "per_call_argument_score", "reference_score",
                "execution_score", "final_answer_score",
                "too_few_calls", "fully_correct",
            )
            if k in diag
        },
    }


def _observations_raw(row: dict) -> List[Any]:
    turns = (row.get("_traj") or {}).get("turns") or []
    return [
        t.get("observation")
        for t in turns
        if t.get("parsed_call") and t.get("fail_reason") is None
    ]


def build_task_meta(
    ids: List[str],
    arms: Dict[str, Dict[str, dict]],
    tasks: Dict[str, dict],
    r0: Dict[str, dict],
) -> Dict[str, dict]:
    meta: Dict[str, dict] = {}
    for sid in ids:
        c0, e2 = arms["C0"][sid], arms["E2"][sid]
        task = tasks[sid]
        w0 = official_win(c0) == 1.0
        w1 = official_win(arms["E1"][sid]) == 1.0
        w2 = official_win(e2) == 1.0
        gold = task.get("gold_calls") or []
        div = first_divergence_turn(predicted_calls(c0), predicted_calls(e2))
        rc0 = r0[sid]["C0"]
        re2 = r0[sid]["E2"]
        meta[sid] = {
            "w0": w0, "w1": w1, "w2": w2,
            "gold_call_bucket": str(len(gold)) if len(gold) <= 5 else "6+",
            "motif": extract_motifs(task).get("motif_type"),
            "first_divergence_turn": div,
            "c0_failure": classify_failure(c0)[0],
            "e2_failure": classify_failure(e2)[0],
            "reward_mismatch": w0 and not w2 and re2 > rc0,
            "e2_executable_wrong": classify_failure(e2)[0] == "executable trajectory ending wrong result",
        }
    return meta


def score_r0_all(
    ids: List[str],
    arms: Dict[str, Dict[str, dict]],
    tasks: Dict[str, dict],
) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for sid in ids:
        task = tasks[sid]
        row: Dict[str, Any] = {}
        cls: Dict[str, str] = {}
        for arm in ("C0", "E1", "E2"):
            rw = episode_turn_reward_seq(
                traj_from_dict(arms[arm][sid]["_traj"]), task
            )
            row[arm] = float(rw.get("episode_reward") or 0.0)
            cls[arm] = (rw.get("diagnostics") or {}).get("reward_class", "")
        row["class_C0"] = cls["C0"]
        row["class_E2"] = cls["E2"]
        out[sid] = row
    return out


def build_packet(
    task_id: str,
    cohorts: List[str],
    tasks: Dict[str, dict],
    arms: Dict[str, dict],
    paths: AuditPaths,
    input_hashes: Dict[str, str],
) -> dict:
    task = tasks[task_id]
    gold = task.get("gold_calls") or []
    c0 = arms["C0"][task_id]
    e1 = arms["E1"][task_id]
    e2 = arms["E2"][task_id]
    c0_calls, e2_calls = predicted_calls(c0), predicted_calls(e2)
    gold_n = len(gold)
    block_c0 = _arm_block(c0, task)
    block_e1 = _arm_block(e1, task)
    block_e2 = _arm_block(e2, task)
    r0_c0 = block_c0["reward_total"]
    r0_e2 = block_e2["reward_total"]
    w0 = official_win(c0) == 1.0
    w2 = official_win(e2) == 1.0
    return {
        "task_id": task_id,
        "cohorts": cohorts,
        "question": task.get("question", ""),
        "expected_outcome": task.get("gold_answer"),
        "gold_metadata": {
            "gold_call_count": gold_n,
            "motif": extract_motifs(task).get("motif_type"),
            "gold_calls": gold,
        },
        "relevant_tools": relevant_tools(
            task, c0_calls, predicted_calls(e1), e2_calls, gold
        ),
        "C0": block_c0,
        "E1": block_e1,
        "E2": block_e2,
        "deterministic_flags": {
            "first_divergence_turn": first_divergence_turn(c0_calls, e2_calls),
            "shorter_than_gold_C0": len(c0_calls) < gold_n,
            "shorter_than_gold_E2": len(e2_calls) < gold_n,
            "taxonomy_too_few_C0": classify_failure(c0)[0] == "too few calls",
            "taxonomy_too_few_E2": classify_failure(e2)[0] == "too few calls",
            "official_path_valid_C0": w0,
            "official_path_valid_E2": w2,
            "reward_prefers_E2_over_C0": w0 and not w2 and r0_e2 > r0_c0,
            "tool_1_same": (
                (c0_calls[0].get("name") if c0_calls else None)
                == (e2_calls[0].get("name") if e2_calls else None)
            ) if c0_calls or e2_calls else None,
            "tool_2_changed": (
                tool_at(c0_calls, 1) != tool_at(e2_calls, 1)
                if gold_n >= 2 else None
            ),
        },
        "provenance": {
            "train_run_id": paths.run_dir.name,
            "eval_task_set_sha256": input_hashes.get("nestful_test", ""),
            "reward_label": "execution_aware_v3_2_dense_recomputed_eval",
            "reward_note": (
                "reward_total/components recomputed offline on saved eval trajectories; "
                "not logged train-time scalars (see R0_PARITY.md)"
            ),
            "source_files": [
                str(paths.nestful_test),
                str(paths.eval_c0 / "final_eval_trajectories.jsonl"),
                str(paths.eval_e1 / "final_eval_trajectories.jsonl"),
                str(paths.eval_e2 / "final_eval_trajectories.jsonl"),
            ],
        },
    }


def load_eval_bundle(paths: AuditPaths) -> tuple:
    tasks = load_tasks(paths.nestful_test)
    arms = {
        "C0": load_traj_rows(paths.eval_c0),
        "E1": load_traj_rows(paths.eval_e1),
        "E2": load_traj_rows(paths.eval_e2),
    }
    ids = sorted(set.intersection(*(set(v) for v in arms.values())) & set(tasks))
    hashes = {
        "nestful_test": sha256_file(paths.nestful_test),
        "eval_C0": sha256_file(paths.eval_c0 / "final_eval_trajectories.jsonl"),
        "eval_E1": sha256_file(paths.eval_e1 / "final_eval_trajectories.jsonl"),
        "eval_E2": sha256_file(paths.eval_e2 / "final_eval_trajectories.jsonl"),
    }
    return tasks, arms, ids, hashes
