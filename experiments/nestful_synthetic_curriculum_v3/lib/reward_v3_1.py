"""execution_aware_v3_1_stepwise reward — stage-aware prefix curriculum scoring."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError:
    yaml = None


@dataclass
class RewardResult:
    reward: float
    diagnostics: Dict[str, Any]


STAGE_FROM_EPOCH = {
    1: "stage1",
    2: "stage2",
    3: "stage3",
    4: "stage4",
}

STAGE_EXACT_CALLS = {
    "stage1": 1,
    "stage2": 2,
    "stage3": 3,
    "stage1_1call_atomic": 1,
    "stage2_2call_dependency": 2,
    "stage3_3call_composition": 3,
}

DEFAULT_CONFIG: Dict[str, Any] = {
    "reward": {
        "name": "execution_aware_v3_1_stepwise",
        "stage1": {
            "weights": {
                "next_action_validity": 0.30,
                "executable_step": 0.25,
                "argument_validity": 0.20,
                "tool_selection": 0.15,
                "final_answer_from_observation": 0.10,
            }
        },
        "stage2": {
            "weights": {
                "next_action_validity": 0.25,
                "executable_step": 0.20,
                "valid_reference": 0.25,
                "continuation_correctness": 0.20,
                "final_answer_if_terminal": 0.10,
            }
        },
        "stage3": {
            "weights": {
                "executable_trajectory": 0.20,
                "valid_references": 0.25,
                "motif_trace_consistency": 0.20,
                "tool_use_completeness": 0.20,
                "final_answer_if_terminal": 0.15,
            }
        },
        "stage4": {
            "weights": {
                "executable_trajectory": 0.20,
                "valid_references": 0.20,
                "motif_trace_consistency": 0.20,
                "tool_use_completeness": 0.25,
                "final_answer": 0.15,
            }
        },
        "caps": {
            "parse_error": 0.0,
            "clipped": 0.0,
            "no_tool_call": 0.0,
            "invalid_reference": 0.1,
            "premature_final_nonterminal": 0.0,
            "too_few_calls": 0.1,
            "severe_short_trace": 0.1,
            "not_executable": 0.2,
        },
        "floors": {
            "executable_complete_prefix": 0.75,
            "executable_complete_prefix_with_valid_refs": 0.85,
        },
    }
}


def load_reward_config(path: Optional[Path] = None) -> Dict[str, Any]:
    if path is None:
        path = Path(__file__).resolve().parents[1] / "configs/reward_v3_1_stepwise.yaml"
    if yaml and path.is_file():
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    return DEFAULT_CONFIG


def detect_stage(task: Dict[str, Any], train_stage: Optional[int] = None) -> str:
    raw = task.get("stage") or task.get("train_stage") or ""
    if isinstance(raw, int):
        return STAGE_FROM_EPOCH.get(raw, f"stage{raw}")
    if raw.startswith("stage1"):
        return "stage1"
    if raw.startswith("stage2"):
        return "stage2"
    if raw.startswith("stage3"):
        return "stage3"
    if raw.startswith("stage4"):
        return "stage4"
    if train_stage is not None:
        return STAGE_FROM_EPOCH.get(int(train_stage), "stage1")
    n = int(task.get("num_calls") or len(task.get("gold_calls") or []))
    if n == 1:
        return "stage1"
    if n == 2:
        return "stage2"
    if n == 3:
        return "stage3"
    return "stage4"


def expected_calls(stage: str, task: Dict[str, Any]) -> int:
    if stage in STAGE_EXACT_CALLS:
        return STAGE_EXACT_CALLS[stage]
    return int(task.get("num_calls") or len(task.get("gold_calls") or []))


def _count_pred_calls(trajectory) -> int:
    return len([t for t in trajectory.turns if getattr(t, "parsed_call", None)])


def _has_final_answer(trajectory) -> bool:
    for t in reversed(trajectory.turns):
        if getattr(t, "final_answer", None):
            return True
    return False


def _predicates(trajectory, task: Dict[str, Any]) -> Dict[str, Any]:
    try:
        import sys
        from pathlib import Path as P
        exp = P(__file__).resolve().parents[2]
        if str(exp) not in sys.path:
            sys.path.insert(0, str(exp))
        from nestful_core import rewards as R
        return {
            "final_pass": R.tool_final_answer_pass(trajectory, task),
            "executable": float(R.is_executable_trajectory(trajectory)),
            "refs": R.valid_references_fraction(trajectory),
            "completeness": R.tool_use_completeness(trajectory, task),
            "gold_prog": R.gold_trace_progress(trajectory, task, None),
            "parse_err": R.has_parse_error(trajectory),
            "clipped": bool(trajectory.clipped_any),
            "no_tool": R.has_no_tool_call(trajectory),
            "invalid_ref": R.has_invalid_reference(trajectory),
            "few": R.too_few_calls(trajectory, task),
        }
    except Exception:
        n_pred = _count_pred_calls(trajectory)
        gold_n = len(task.get("gold_calls") or [])
        return {
            "final_pass": _has_final_answer(trajectory) and n_pred >= gold_n,
            "executable": 1.0 if n_pred >= 1 else 0.0,
            "refs": 1.0 if n_pred >= 2 else 0.0,
            "completeness": min(1.0, n_pred / max(gold_n, 1)),
            "gold_prog": min(1.0, n_pred / max(gold_n, 1)),
            "parse_err": False,
            "clipped": bool(getattr(trajectory, "clipped_any", False)),
            "no_tool": n_pred == 0,
            "invalid_ref": False,
            "few": n_pred < gold_n,
        }


def execution_aware_v3_1_stepwise(
    trajectory,
    task: Dict[str, Any],
    gold_observations: Optional[List[Any]] = None,
    *,
    train_stage: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None,
) -> RewardResult:
    cfg_root = config or load_reward_config()
    cfg = cfg_root.get("reward", cfg_root)
    caps = cfg.get("caps", DEFAULT_CONFIG["reward"]["caps"])
    floors = cfg.get("floors", DEFAULT_CONFIG["reward"]["floors"])

    stage = detect_stage(task, train_stage)
    stage_key = stage if stage in cfg else f"{stage}"
    weights = cfg.get(stage_key, cfg.get("stage1", {})).get("weights", {})

    pred = _predicates(trajectory, task)
    n_pred = _count_pred_calls(trajectory)
    gold_n = expected_calls(stage, task)
    terminal = bool(task.get("terminal_stage", True))
    has_final = _has_final_answer(trajectory)

    too_few = n_pred < gold_n or pred["few"]
    premature_final = (not terminal) and has_final

    if stage == "stage1":
        R_val = (
            weights.get("next_action_validity", 0.3) * (1.0 if n_pred >= 1 else 0.0)
            + weights.get("executable_step", 0.25) * pred["executable"]
            + weights.get("argument_validity", 0.2) * pred["executable"]
            + weights.get("tool_selection", 0.15) * (1.0 if n_pred == 1 else 0.0)
            + weights.get("final_answer_from_observation", 0.1) * (1.0 if pred["final_pass"] else 0.0)
        )
    elif stage == "stage2":
        R_val = (
            weights.get("next_action_validity", 0.25) * min(1.0, n_pred / max(gold_n, 1))
            + weights.get("executable_step", 0.2) * pred["executable"]
            + weights.get("valid_reference", 0.25) * (pred["refs"] or 0.0)
            + weights.get("continuation_correctness", 0.2) * pred["completeness"]
            + weights.get("final_answer_if_terminal", 0.1) * (1.0 if pred["final_pass"] and terminal else 0.0)
        )
    elif stage == "stage3":
        from lib.reward_motif import motif_trace_consistency
        motif_cons = motif_trace_consistency(trajectory, task)
        R_val = (
            weights.get("executable_trajectory", 0.2) * pred["executable"]
            + weights.get("valid_references", 0.25) * (pred["refs"] or 0.0)
            + weights.get("motif_trace_consistency", 0.2) * motif_cons
            + weights.get("tool_use_completeness", 0.2) * pred["completeness"]
            + weights.get("final_answer_if_terminal", 0.15) * (1.0 if pred["final_pass"] and terminal else 0.0)
        )
    else:
        from lib.reward_motif import motif_trace_consistency
        motif_cons = motif_trace_consistency(trajectory, task)
        R_val = (
            weights.get("executable_trajectory", 0.2) * pred["executable"]
            + weights.get("valid_references", 0.2) * (pred["refs"] or 0.0)
            + weights.get("motif_trace_consistency", 0.2) * motif_cons
            + weights.get("tool_use_completeness", 0.25) * pred["completeness"]
            + weights.get("final_answer", 0.15) * (1.0 if pred["final_pass"] else 0.0)
        )

    R_val = max(0.0, min(1.0, R_val))
    cap_applied = None

    if pred["parse_err"]:
        R_val = caps["parse_error"]; cap_applied = "parse_error"
    elif pred["clipped"]:
        R_val = caps["clipped"]; cap_applied = "clipped"
    elif pred["no_tool"]:
        R_val = caps["no_tool_call"]; cap_applied = "no_tool_call"
    elif premature_final:
        R_val = caps["premature_final_nonterminal"]; cap_applied = "premature_final_nonterminal"
    elif pred["invalid_ref"]:
        R_val = min(R_val, caps["invalid_reference"]); cap_applied = "invalid_reference"
    elif too_few:
        R_val = min(R_val, caps["too_few_calls"]); cap_applied = "too_few_calls"
    elif not pred["executable"] and R_val > caps["not_executable"]:
        R_val = caps["not_executable"]; cap_applied = "not_executable"
    elif n_pred < gold_n - 1 and R_val > caps["severe_short_trace"]:
        R_val = caps["severe_short_trace"]; cap_applied = "severe_short_trace"

    complete = n_pred >= gold_n and pred["executable"] >= 0.99
    if complete and (pred["refs"] or 0) >= 0.99 and stage in ("stage2", "stage3", "stage4"):
        if R_val < floors["executable_complete_prefix_with_valid_refs"]:
            R_val = floors["executable_complete_prefix_with_valid_refs"]
            cap_applied = cap_applied or "floor_valid_refs"
    elif complete and R_val < floors["executable_complete_prefix"]:
        R_val = floors["executable_complete_prefix"]
        cap_applied = cap_applied or "floor_executable_prefix"

    diag = {
        "reward_type": "execution_aware_v3_1_stepwise",
        "reward": R_val,
        "stage": stage,
        "n_pred_calls": n_pred,
        "gold_n_calls": gold_n,
        "too_few_calls": too_few,
        "premature_final": premature_final,
        "cap_applied": cap_applied,
    }
    return RewardResult(R_val, diag)
