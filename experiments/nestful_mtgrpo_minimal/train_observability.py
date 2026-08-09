"""Adapter between the GRPO trainer and the Phase-P logging schemas.

The schemas live in ``targeted_tool_data.observability`` so they can be tested
without torch. This module owns the import shim and the trainer-specific row
building, and it degrades to a no-op logger when the factory package is not on
the path (the RunPod image ships the trainer alone).

Enable with ``logging.observability_dir`` in the config or the
``TTDF_TRAIN_LOG_DIR`` environment variable.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_FACTORY_SRC = (Path(__file__).resolve().parents[1]
                / "targeted_tool_data_factory" / "src")

try:
    if str(_FACTORY_SRC) not in sys.path and _FACTORY_SRC.is_dir():
        sys.path.insert(0, str(_FACTORY_SRC))
    from targeted_tool_data.observability import (  # noqa: E402
        TrainRunLogger, chat_template_hash, sha256_obj, sha256_text)
    OBSERVABILITY_AVAILABLE = True
except Exception:  # noqa: BLE001
    TrainRunLogger = None  # type: ignore[assignment]
    OBSERVABILITY_AVAILABLE = False

    def sha256_text(text: str) -> str:  # type: ignore[misc]
        import hashlib

        return hashlib.sha256((text or "").encode("utf-8")).hexdigest()

    def sha256_obj(obj: Any) -> str:  # type: ignore[misc]
        import json

        return sha256_text(json.dumps(obj, sort_keys=True, default=str))

    def chat_template_hash(tokenizer: Any) -> str:  # type: ignore[misc]
        return sha256_text(str(getattr(tokenizer, "chat_template", "") or ""))


class NullTrainLogger:
    """Same surface, writes nothing."""

    enabled = False

    def write_manifest(self, **_kw: Any) -> None:
        return None

    def log_rollout(self, _row: Dict[str, Any]) -> None:
        return None

    def log_group(self, _row: Dict[str, Any]) -> None:
        return None

    def log_step(self, _row: Dict[str, Any]) -> None:
        return None

    def save_sampler_state(self, _state: Dict[str, Any],
                           _checkpoint_dir: Optional[Path] = None) -> List[Path]:
        return []

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


def make_train_logger(config: Dict[str, Any], *, run_id: str,
                      default_dir: Optional[str] = None) -> Any:
    """Returns a TrainRunLogger, or a null logger when logging is off."""
    log_cfg = (config.get("logging") or {}) if isinstance(config, dict) else {}
    out_dir = (os.environ.get("TTDF_TRAIN_LOG_DIR")
               or log_cfg.get("observability_dir") or default_dir)
    if not out_dir or not OBSERVABILITY_AVAILABLE:
        return NullTrainLogger()
    logger = TrainRunLogger(
        out_dir=Path(out_dir), run_id=run_id,
        repo_root=Path(__file__).resolve().parents[2],
        gzip_rollouts=bool(log_cfg.get("gzip_rollouts", False)),
        keep_response_text=bool(log_cfg.get("keep_response_text", True)))
    logger.enabled = True
    return logger


# ── row builders ──────────────────────────────────────────────────────────
def _prov(task: Dict[str, Any]) -> Dict[str, Any]:
    p = task.get("provenance")
    return p if isinstance(p, dict) else {}


def group_row(*, run_id: str, task: Dict[str, Any], rec: Dict[str, Any],
              episodes: Sequence[Any], gstats: Any, global_step: int,
              epoch: int, batch_id: str, accepted: bool,
              rejection_reason: str = "",
              sampler_weight: float = 0.0,
              sampler_weight_components: Optional[Dict[str, float]] = None,
              refill_round: int = 0) -> Dict[str, Any]:
    prov = _prov(task)
    totals = [float(getattr(e, "reward", 0.0)) for e in episodes]
    terminal = [1.0 if float(getattr(e, "reward", 0.0)) >= 1.0 else 0.0
                for e in episodes]
    process = [float(getattr(e, "process_reward", 0.0) or 0.0) for e in episodes]
    parse_ok = [not bool(getattr(getattr(e, "trajectory", None),
                                 "zero_tool_calls", False)) for e in episodes]
    exec_ok = [bool(getattr(getattr(e, "trajectory", None),
                            "executed_ok", True)) for e in episodes]
    n = max(len(totals), 1)
    mean = sum(totals) / n
    var = sum((x - mean) ** 2 for x in totals) / n
    return {
        "run_id": run_id,
        "global_step": global_step,
        "epoch": epoch,
        "batch_id": batch_id,
        "group_id": f"{run_id}:{epoch}:{rec.get('task_idx')}",
        "prompt_id": task.get("task_id") or task.get("sample_id"),
        "generation_cell": prov.get("generation_cell_id", ""),
        "semantic_program_family": prov.get("semantic_program_family", ""),
        "difficulty_signature": prov.get("difficulty_signature", {}),
        "group_size": len(totals),
        "terminal_rewards": terminal,
        "process_rewards": process,
        "total_rewards": totals,
        "reward_mean": mean,
        "reward_std": var ** 0.5,
        "terminal_success_rate": sum(terminal) / n,
        "process_std_within_terminal_class": _within_class_std(terminal, process),
        "parse_success_rate": sum(1 for x in parse_ok if x) / n,
        "executable_rate": sum(1 for x in exec_ok if x) / n,
        "group_class": classify_group_row(terminal, process, totals),
        "accepted": accepted,
        "rejection_reason": rejection_reason,
        "sampler_weight": sampler_weight,
        "sampler_weight_components": sampler_weight_components or {},
        "refill_round": refill_round,
        "n_unique_total_rewards": len({round(x, 6) for x in totals}),
        "dead_group": bool(rec.get("dead_group")),
        "reward_std_between_completion": rec.get("reward_std_between_completion"),
        "position_artifact_detected": rec.get("position_artifact_detected"),
    }


def rollout_rows(*, run_id: str, task: Dict[str, Any], rec: Dict[str, Any],
                 episodes: Sequence[Any], ep_r_seqs: Sequence[Sequence[float]],
                 gstats: Any, ep_diags: Sequence[Dict[str, Any]],
                 global_step: int, epoch: int, batch_id: str,
                 prompt_hash: str = "", kl: Optional[float] = None
                 ) -> List[Dict[str, Any]]:
    prov = _prov(task)
    totals = [float(getattr(e, "reward", 0.0)) for e in episodes]
    n = max(len(totals), 1)
    mean = sum(totals) / n
    std = (sum((x - mean) ** 2 for x in totals) / n) ** 0.5
    rows: List[Dict[str, Any]] = []
    for ri, ep in enumerate(episodes):
        traj = getattr(ep, "trajectory", None)
        diag = ep_diags[ri] if ri < len(ep_diags) else {}
        adv = list(gstats.advantages[ri]) if gstats is not None else []
        text = getattr(ep, "completion_text", "") or ""
        rows.append({
            "run_id": run_id,
            "global_step": global_step,
            "epoch": epoch,
            "batch_id": batch_id,
            "group_id": f"{run_id}:{epoch}:{rec.get('task_idx')}",
            "prompt_id": task.get("task_id") or task.get("sample_id"),
            "rollout_id": ri,
            "generation_cell": prov.get("generation_cell_id", ""),
            "semantic_program_family": prov.get("semantic_program_family", ""),
            "difficulty_signature": prov.get("difficulty_signature", {}),
            "prompt_hash": prompt_hash,
            "response_hash": sha256_text(text),
            "response_text": text,
            "parsed_calls": getattr(traj, "predicted_calls", None)
                            or diag.get("parsed_calls"),
            "terminal_reward": float(getattr(ep, "reward", 0.0)),
            "process_reward": float(getattr(ep, "process_reward", 0.0) or 0.0),
            "process_components": diag.get("reward_components") or {},
            "total_reward": float(getattr(ep, "reward", 0.0)),
            "group_reward_mean": mean,
            "group_reward_std": std,
            "raw_advantage": float(getattr(ep, "reward", 0.0)) - mean,
            "normalized_advantage": adv,
            "parse_valid": not bool(getattr(traj, "zero_tool_calls", False)),
            "executable": bool(getattr(traj, "executed_ok", True)),
            "execution_error": diag.get("execution_error"),
            "final_answer_correct": bool(diag.get("final_answer_correct", False)),
            "solution_equivalent": bool(diag.get("solution_equivalent", False)),
            "stop_reason": ("length" if getattr(traj, "clipped_any", False)
                            else diag.get("stop_reason") or "stop"),
            "n_generated_tokens": int(getattr(traj, "n_generated_tokens", 0) or 0),
            "n_tool_calls": int(getattr(traj, "num_tool_calls", 0) or 0),
            "kl": kl,
            "turn_rewards": list(ep_r_seqs[ri]) if ri < len(ep_r_seqs) else [],
        })
    return rows


def step_row(*, run_id: str, global_step: int, epoch: int,
             candidate_prompt_count: int, group_rows: Sequence[Dict[str, Any]],
             reward_component_means: Optional[Dict[str, float]] = None,
             kl: Optional[float] = None, loss: Optional[float] = None,
             gradient_norm: Optional[float] = None,
             sampler_entropy: Optional[float] = None,
             refill_rounds: int = 1,
             optimizer_step_executed: bool = True) -> Dict[str, Any]:
    n = max(len(group_rows), 1)
    accepted = [g for g in group_rows if g.get("accepted")]
    dead = [g for g in group_rows if g.get("dead_group")]
    all_correct = [g for g in group_rows
                   if g.get("group_class") == "ALL_CORRECT"]
    all_fail_flat = [g for g in group_rows
                     if g.get("group_class") == "ALL_FAIL_NO_PROGRESS"]
    all_fail_var = [g for g in group_rows
                    if g.get("group_class") == "ALL_FAIL_WITH_PROCESS_VARIANCE"]
    rollouts = sum(int(g.get("group_size") or 0) for g in group_rows)
    used = sum(int(g.get("group_size") or 0) for g in accepted)
    return {
        "run_id": run_id,
        "global_step": global_step,
        "epoch": epoch,
        "candidate_prompt_count": candidate_prompt_count,
        "accepted_effective_groups": len(accepted),
        "rejected_all_correct": len(all_correct),
        "rejected_all_fail_no_progress": len(all_fail_flat),
        "retained_all_fail_with_progress": len(all_fail_var),
        "dead_group_rate_before_filtering": round(len(dead) / n, 4),
        "effective_group_rate_after_filtering": round(len(accepted) / n, 4),
        "rollout_utilization": round(used / max(rollouts, 1), 4),
        "reward_component_means": reward_component_means or {},
        "kl": kl,
        "loss": loss,
        "gradient_norm": gradient_norm,
        "sampler_distribution_entropy": sampler_entropy,
        "cell_coverage": _cell_coverage(group_rows),
        "refill_rounds": refill_rounds,
        "optimizer_step_executed": optimizer_step_executed,
    }


def _cell_coverage(group_rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for g in group_rows:
        cell = g.get("generation_cell") or "unknown"
        out[cell] = out.get(cell, 0) + 1
    return out


def _within_class_std(terminal: Sequence[float], process: Sequence[float]) -> float:
    if not process:
        return 0.0
    groups = [[p for t, p in zip(terminal, process) if t >= 0.5],
              [p for t, p in zip(terminal, process) if t < 0.5]]
    best = 0.0
    for g in groups:
        if len(g) < 2:
            continue
        m = sum(g) / len(g)
        best = max(best, (sum((x - m) ** 2 for x in g) / len(g)) ** 0.5)
    return best


def classify_group_row(terminal: Sequence[float], process: Sequence[float],
                       totals: Sequence[float], *, eps: float = 1e-3) -> str:
    if len(totals) < 2:
        return "INVALID_GROUP"
    n = len(terminal) or 1
    sr = sum(terminal) / n
    process_varies = _within_class_std(terminal, process) > eps
    if sr >= 1.0 - 1e-9:
        return "ALL_CORRECT"
    if sr <= 1e-9:
        return ("ALL_FAIL_WITH_PROCESS_VARIANCE" if process_varies
                else "ALL_FAIL_NO_PROGRESS")
    if process_varies:
        return "MIXED_BOTH"
    return "MIXED_TERMINAL"
