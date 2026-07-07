"""execution_aware_v3_1_stepwise reward — stage-aware, discriminative curriculum scoring.

Fixed version (post-audit). Design goals:

  * Distinguish (monotonically, by reward band):
      parse error / no tool call        -> 0.0
      premature final on prefix task    -> 0.0
      wrong tool                        -> <= 0.35
      correct tool, wrong args          -> <= 0.60
      executable but wrong final answer -> <= 0.75
      fully correct trajectory          -> >= 0.90
  * tool_selection = exact gold tool-name match (NOT call count).
  * argument_validity = normalized gold argument-VALUE match (NOT just executable).
  * executable is a separate component from correctness.
  * NO silent exception swallowing: if predicate computation fails the reward is
    0.0 with a ``predicates_error`` diagnostic and a loud log line.
  * Stage must be inferable from task metadata, the ``train_stage`` argument, or
    the ``TRAIN_STAGE`` env var — otherwise a hard error is raised.
  * Emits per-generated-turn ``turn_scores`` so the trainer's r_seq comes from
    THIS reward (previously r_seq silently came from execution_aware_v2).
"""
from __future__ import annotations

import os
import re
import sys
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


class RewardError(RuntimeError):
    """Hard error: the reward cannot be computed safely (do not fake a value)."""


STAGE_FROM_EPOCH = {1: "stage1", 2: "stage2", 3: "stage3", 4: "stage4"}

STAGE_EXACT_CALLS = {
    "stage1": 1,
    "stage2": 2,
    "stage3": 3,
    "stage1_1call_atomic": 1,
    "stage2_2call_dependency": 2,
    "stage3_3call_composition": 3,
}

# Reference-argument pattern, e.g. "$var_1$" or "$var_1.result$".
_VAR_REF_RE = re.compile(r"^\$([A-Za-z_]\w*)(?:\.[^$]*)?\$$")

DEFAULT_CONFIG: Dict[str, Any] = {
    "reward": {
        "name": "execution_aware_v3_1_stepwise",
        "stage1": {
            "weights": {
                "format_valid": 0.20,
                "tool_name_match": 0.20,
                "argument_value_match": 0.25,
                "executable_step": 0.15,
                "final_answer_match": 0.20,
            }
        },
        "stage_multi": {  # stage2 / stage3 / stage4
            "weights": {
                "format_valid": 0.10,
                "tool_sequence_match": 0.20,
                "argument_value_match": 0.15,
                "valid_references": 0.15,
                "dependency_use": 0.10,
                "executable_trajectory": 0.10,
                "expected_num_calls": 0.10,
                "final_answer_match": 0.10,
            }
        },
        "caps": {
            "parse_error": 0.0,
            "clipped": 0.0,
            "no_tool_call": 0.0,
            "premature_final_nonterminal": 0.0,
            "invalid_reference": 0.10,
            "wrong_tool": 0.35,
            "correct_tool_wrong_args": 0.60,
            "executable_wrong_final": 0.75,
            "too_few_calls": 0.30,
            "too_many_calls": 0.70,
        },
        "floors": {
            "correct_tool_args_final": 0.90,
        },
    }
}


def load_reward_config(path: Optional[Path] = None) -> Dict[str, Any]:
    if path is None:
        path = Path(__file__).resolve().parents[1] / "configs/reward_v3_1_stepwise.yaml"
    if yaml and path.is_file():
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and "reward" in loaded:
                return loaded
        except Exception as exc:  # config parse failure must be visible, not silent
            print(f"[reward_v3_1] WARNING: failed to parse {path}: {exc}; "
                  f"using built-in defaults", flush=True)
    return DEFAULT_CONFIG


def detect_stage(task: Dict[str, Any], train_stage: Optional[int] = None) -> str:
    """Infer curriculum stage. Raises RewardError when nothing is available."""
    raw = task.get("stage") or task.get("train_stage") or ""
    if isinstance(raw, int):
        return STAGE_FROM_EPOCH.get(raw, f"stage{raw}")
    if isinstance(raw, str):
        for s in ("stage1", "stage2", "stage3", "stage4"):
            if raw.startswith(s):
                return s
    if train_stage is not None:
        return STAGE_FROM_EPOCH.get(int(train_stage), "stage4")
    env_stage = os.environ.get("TRAIN_STAGE", "").strip()
    if env_stage:
        try:
            return STAGE_FROM_EPOCH.get(int(env_stage), "stage4")
        except ValueError:
            pass
    n = task.get("num_calls") or len(task.get("gold_calls") or [])
    if n:
        n = int(n)
        return STAGE_FROM_EPOCH.get(n, "stage4")
    raise RewardError(
        "execution_aware_v3_1_stepwise: cannot infer stage — task has no "
        "stage/num_calls/gold_calls metadata and TRAIN_STAGE env is unset. "
        "Check normalize_task metadata preservation and launcher exports."
    )


def expected_calls(stage: str, task: Dict[str, Any]) -> int:
    n = task.get("num_calls") or len(task.get("gold_calls") or [])
    if n:
        return int(n)
    if stage in STAGE_EXACT_CALLS:
        return STAGE_EXACT_CALLS[stage]
    return 1


# ─────────────────────────────────────────────────────────────────────────────
#  Value / call matching helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_ref(value: Any) -> bool:
    return isinstance(value, str) and bool(_VAR_REF_RE.match(value.strip()))


def _values_match(pred: Any, gold: Any) -> bool:
    """Normalized value equality: int/float tolerance, stripped strings,
    element-wise lists, key-wise dicts."""
    if isinstance(gold, bool) or isinstance(pred, bool):
        return pred is gold or pred == gold
    if isinstance(gold, (int, float)) and isinstance(pred, (int, float)):
        try:
            return abs(float(pred) - float(gold)) <= 1e-6 * max(1.0, abs(float(gold)))
        except (TypeError, ValueError, OverflowError):
            return False
    if isinstance(gold, str) and isinstance(pred, str):
        if pred.strip() == gold.strip():
            return True
        # numeric strings ("33" vs "33.0")
        try:
            return abs(float(pred) - float(gold)) <= 1e-6 * max(1.0, abs(float(gold)))
        except (TypeError, ValueError):
            return False
    if isinstance(gold, (int, float)) and isinstance(pred, str):
        try:
            return abs(float(pred) - float(gold)) <= 1e-6 * max(1.0, abs(float(gold)))
        except (TypeError, ValueError):
            return False
    if isinstance(gold, str) and isinstance(pred, (int, float)):
        try:
            return abs(float(pred) - float(gold)) <= 1e-6 * max(1.0, abs(float(gold)))
        except (TypeError, ValueError):
            return False
    if isinstance(gold, list) and isinstance(pred, list):
        return len(gold) == len(pred) and all(
            _values_match(p, g) for p, g in zip(pred, gold))
    if isinstance(gold, dict) and isinstance(pred, dict):
        return set(gold.keys()) == set(pred.keys()) and all(
            _values_match(pred[k], gold[k]) for k in gold)
    return pred == gold


def _arg_match_fraction(pred_args: Dict[str, Any], gold_args: Dict[str, Any]) -> float:
    """Fraction of gold arguments matched by value (refs match refs).

    A gold reference arg is matched when the prediction is also a reference
    (labels are model-chosen, so any well-formed ref counts here; whether the
    reference resolves is scored by valid_references / dependency_use).
    A literal gold arg passed as a resolved literal where gold used a ref gets
    half credit (value plausible, dependency not exercised).
    """
    if not gold_args:
        return 1.0 if not pred_args else 0.5
    score = 0.0
    for k, gv in gold_args.items():
        if k not in pred_args:
            continue
        pv = pred_args[k]
        if _is_ref(gv):
            if _is_ref(pv):
                score += 1.0
            else:
                score += 0.5  # literal where gold used a reference
        else:
            if _values_match(pv, gv):
                score += 1.0
    frac = score / len(gold_args)
    # Extra unexpected keys dilute the match.
    extra = set(pred_args.keys()) - set(gold_args.keys())
    if extra:
        frac *= len(gold_args) / (len(gold_args) + len(extra))
    return max(0.0, min(1.0, frac))


def _emitted_calls(trajectory) -> List[Any]:
    return [t for t in trajectory.turns if getattr(t, "parsed_call", None)]


def _per_call_analysis(trajectory, gold_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compare each emitted call to the gold call at the same position."""
    out: List[Dict[str, Any]] = []
    for i, turn in enumerate(_emitted_calls(trajectory)):
        call = turn.parsed_call or {}
        gold = gold_calls[i] if i < len(gold_calls) else None
        name_ok = bool(gold) and (call.get("name") or "") == (gold.get("name") or "")
        gold_args = (gold.get("arguments") or {}) if gold else {}
        pred_args = call.get("arguments") or {}
        keys_ok = bool(gold) and set(pred_args.keys()) == set(gold_args.keys())
        val_frac = _arg_match_fraction(pred_args, gold_args) if gold else 0.0
        exec_clean = getattr(turn, "fail_reason", None) is None
        out.append({
            "position": i,
            "name_ok": name_ok,
            "keys_ok": keys_ok,
            "val_frac": val_frac,
            "exec_clean": exec_clean,
            "beyond_gold": gold is None,
        })
    return out


def _has_final_answer(trajectory) -> bool:
    for t in reversed(trajectory.turns):
        if getattr(t, "final_answer", None):
            return True
        if getattr(t, "is_terminal", False):
            return True
    return False


def _predicates(trajectory, task: Dict[str, Any]) -> Dict[str, Any]:
    """nestful_core predicates. HARD-FAILS (raises) instead of faking values."""
    exp = Path(__file__).resolve().parents[2]
    if str(exp) not in sys.path:
        sys.path.insert(0, str(exp))
    from nestful_core import rewards as R
    refs = R.valid_references_fraction(trajectory)
    return {
        "final_pass": bool(R.tool_final_answer_pass(trajectory, task)),
        "executable_frac": float(R.executable_fraction(trajectory)),
        "is_executable": bool(R.is_executable_trajectory(trajectory)),
        "refs": refs,  # None = model used no references
        "parse_err": bool(R.has_parse_error(trajectory)),
        "clipped": bool(getattr(trajectory, "clipped_any", False)),
        "no_tool": bool(R.has_no_tool_call(trajectory)),
        "invalid_ref": bool(R.has_invalid_reference(trajectory)),
        "n_success": int(R.num_successful_calls(trajectory)),
    }


def _dependency_use_fraction(trajectory, gold_calls: List[Dict[str, Any]]) -> Optional[float]:
    """Of the gold args that are references, what fraction did the model also
    pass as references (at the same call position / arg key)?  None when the
    gold trace has no reference args (component gets full credit)."""
    gold_ref_slots = []
    for i, g in enumerate(gold_calls):
        for k, v in (g.get("arguments") or {}).items():
            if _is_ref(v):
                gold_ref_slots.append((i, k))
    if not gold_ref_slots:
        return None
    emitted = _emitted_calls(trajectory)
    used = 0
    for (i, k) in gold_ref_slots:
        if i < len(emitted):
            pv = (emitted[i].parsed_call or {}).get("arguments", {}).get(k)
            if _is_ref(pv):
                used += 1
    return used / len(gold_ref_slots)


def _turn_scores(trajectory, calls_info: List[Dict[str, Any]],
                 final_ok: bool, terminal: bool) -> List[float]:
    """Per-GENERATED-turn quality in [0,1], aligned 1:1 with trajectory.turns.

    Used as the trainer's r_seq so turn-level credit comes from THIS reward.
    """
    scores: List[float] = []
    call_idx = 0
    for t in trajectory.turns:
        if getattr(t, "parsed_call", None):
            info = calls_info[call_idx] if call_idx < len(calls_info) else None
            call_idx += 1
            if info is None or info["beyond_gold"]:
                scores.append(0.0)
                continue
            s = (0.35 * (1.0 if info["name_ok"] else 0.0)
                 + 0.35 * info["val_frac"]
                 + 0.15 * (1.0 if info["keys_ok"] else 0.0)
                 + 0.15 * (1.0 if info["exec_clean"] else 0.0))
            scores.append(max(0.0, min(1.0, s)))
        elif getattr(t, "is_terminal", False) or getattr(t, "final_answer", None):
            scores.append(1.0 if (final_ok and terminal) else 0.0)
        else:
            scores.append(0.0)  # parse fail / clipped / other
    return scores


# ─────────────────────────────────────────────────────────────────────────────
#  Main reward
# ─────────────────────────────────────────────────────────────────────────────

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
    caps = {**DEFAULT_CONFIG["reward"]["caps"], **(cfg.get("caps") or {})}
    floors = {**DEFAULT_CONFIG["reward"]["floors"], **(cfg.get("floors") or {})}

    stage = detect_stage(task, train_stage)  # raises RewardError when unknown
    gold_calls = list(task.get("gold_calls") or [])
    gold_n = expected_calls(stage, task)
    terminal = bool(task.get("terminal_stage", True))

    try:
        pred = _predicates(trajectory, task)
    except Exception as exc:  # noqa: BLE001 — fail the reward WITH a diagnostic
        msg = f"{type(exc).__name__}: {exc}"
        print(f"[reward_v3_1] PREDICATES_ERROR task={task.get('task_id')}: {msg}",
              flush=True)
        return RewardResult(0.0, {
            "reward_type": "execution_aware_v3_1_stepwise",
            "reward_total": 0.0,
            "predicates_error": msg,
            "reward_cap_reason": "predicates_error",
            "cap_applied": "predicates_error",
            "stage": stage,
            "turn_scores": [0.0] * len(getattr(trajectory, "turns", [])),
        })

    calls_info = _per_call_analysis(trajectory, gold_calls)
    n_pred = len(calls_info)
    has_final = _has_final_answer(trajectory)

    matchable = min(n_pred, gold_n) or 0
    name_fracs = [1.0 if c["name_ok"] else 0.0 for c in calls_info[:gold_n]]
    val_fracs = [c["val_frac"] for c in calls_info[:gold_n]]
    keys_oks = [c["keys_ok"] for c in calls_info[:gold_n]]

    # Coverage-weighted: missing gold positions count as 0.
    tool_match = (sum(name_fracs) / gold_n) if gold_n else 0.0
    arg_match = (sum(val_fracs) / gold_n) if gold_n else 0.0
    args_full = bool(matchable == gold_n and all(keys_oks) and
                     all(v >= 0.999 for v in val_fracs)) if gold_n else False
    tools_full = bool(matchable == gold_n and all(f >= 0.999 for f in name_fracs)) if gold_n else False

    format_ok = 1.0 if (not pred["parse_err"] and not pred["clipped"]) else 0.0
    final_ok = bool(pred["final_pass"])
    refs = pred["refs"]
    gold_has_refs = any(_is_ref(v) for g in gold_calls
                        for v in (g.get("arguments") or {}).values())
    if refs is None:
        refs_frac = 0.0 if gold_has_refs else 1.0
    else:
        refs_frac = float(refs)
    dep_use = _dependency_use_fraction(trajectory, gold_calls)
    dep_frac = 1.0 if dep_use is None else dep_use

    too_few = n_pred < gold_n
    too_many = n_pred > gold_n
    premature_final = (not terminal) and has_final
    wrong_tool = bool(calls_info) and not calls_info[0]["name_ok"] if stage == "stage1" \
        else (matchable > 0 and not tools_full)

    if gold_n:
        num_calls_score = max(0.0, 1.0 - abs(n_pred - gold_n) / gold_n)
    else:
        num_calls_score = 0.0

    if stage == "stage1":
        w = {**DEFAULT_CONFIG["reward"]["stage1"]["weights"],
             **((cfg.get("stage1") or {}).get("weights") or {})}
        R_val = (
            w["format_valid"] * format_ok
            + w["tool_name_match"] * tool_match
            + w["argument_value_match"] * arg_match
            + w["executable_step"] * pred["executable_frac"]
            + w["final_answer_match"] * (1.0 if final_ok else 0.0)
        )
    else:
        w = {**DEFAULT_CONFIG["reward"]["stage_multi"]["weights"],
             **((cfg.get("stage_multi") or cfg.get(stage) or {}).get("weights") or {})}
        # On non-terminal (prefix) tasks the correct behavior is NOT emitting a
        # final answer; give the final component for stopping correctly.
        final_component = (1.0 if final_ok else 0.0) if terminal \
            else (1.0 if not has_final else 0.0)
        R_val = (
            w["format_valid"] * format_ok
            + w["tool_sequence_match"] * tool_match
            + w["argument_value_match"] * arg_match
            + w["valid_references"] * refs_frac
            + w["dependency_use"] * dep_frac
            + w["executable_trajectory"] * pred["executable_frac"]
            + w["expected_num_calls"] * num_calls_score
            + w["final_answer_match"] * final_component
        )

    R_val = max(0.0, min(1.0, R_val))
    cap_reason = None
    floor_reason = None

    # ── Hard caps (band enforcement) ─────────────────────────────────────────
    if pred["parse_err"]:
        R_val, cap_reason = caps["parse_error"], "parse_error"
    elif pred["clipped"]:
        R_val, cap_reason = caps["clipped"], "clipped"
    elif pred["no_tool"]:
        R_val, cap_reason = caps["no_tool_call"], "no_tool_call"
    elif premature_final:
        R_val, cap_reason = caps["premature_final_nonterminal"], "premature_final_nonterminal"
    elif pred["invalid_ref"]:
        if R_val > caps["invalid_reference"]:
            R_val, cap_reason = caps["invalid_reference"], "invalid_reference"
    elif too_few:
        if R_val > caps["too_few_calls"]:
            R_val, cap_reason = min(R_val, caps["too_few_calls"]), "too_few_calls"
    elif wrong_tool:
        if R_val > caps["wrong_tool"]:
            R_val, cap_reason = min(R_val, caps["wrong_tool"]), "wrong_tool"
    elif tools_full and not args_full:
        if R_val > caps["correct_tool_wrong_args"]:
            R_val, cap_reason = min(R_val, caps["correct_tool_wrong_args"]), "correct_tool_wrong_args"
    elif too_many:
        if R_val > caps["too_many_calls"]:
            R_val, cap_reason = min(R_val, caps["too_many_calls"]), "too_many_calls"
    elif tools_full and args_full and pred["is_executable"] and terminal and not final_ok:
        if R_val > caps["executable_wrong_final"]:
            R_val, cap_reason = min(R_val, caps["executable_wrong_final"]), "executable_wrong_final"

    # ── Floor: fully correct trajectory must land in the top band ───────────
    fully_correct = (tools_full and args_full and pred["is_executable"]
                     and not too_few and not too_many
                     and (final_ok if terminal else not has_final))
    if fully_correct and cap_reason is None and R_val < floors["correct_tool_args_final"]:
        R_val = floors["correct_tool_args_final"]
        floor_reason = "correct_tool_args_final"

    turn_scores = _turn_scores(trajectory, calls_info, final_ok, terminal)

    diag = {
        "reward_type": "execution_aware_v3_1_stepwise",
        "reward": R_val,
        "reward_total": R_val,
        "reward_format": format_ok,
        "reward_tool_match": tool_match,
        "reward_arg_match": arg_match,
        "reward_executable": pred["executable_frac"],
        "reward_final_answer": 1.0 if final_ok else 0.0,
        "reward_valid_refs": refs_frac,
        "reward_dependency_use": dep_frac,
        "reward_num_calls": num_calls_score,
        "reward_premature_final": bool(premature_final),
        "reward_cap_reason": cap_reason,
        "reward_floor_reason": floor_reason,
        "cap_applied": cap_reason,          # backwards-compatible key
        "stage": stage,
        "reward_seen_stage": task.get("stage") or stage,
        "reward_seen_num_calls": task.get("num_calls"),
        "reward_seen_motif_type": task.get("motif_type"),
        "reward_seen_terminal_stage": task.get("terminal_stage", True),
        "n_pred_calls": n_pred,
        "gold_n_calls": gold_n,
        "predicted_num_calls": n_pred,
        "too_few_calls": bool(too_few),
        "too_many_calls": bool(too_many),
        "wrong_tool": bool(wrong_tool),
        "wrong_args": bool(tools_full and not args_full),
        "parse_error": bool(pred["parse_err"]),
        "no_tool_call": bool(pred["no_tool"]),
        "invalid_reference": bool(pred["invalid_ref"]),
        "premature_final": bool(premature_final),
        "fully_correct": bool(fully_correct),
        "predicates_error": None,
        "turn_scores": turn_scores,
    }
    return RewardResult(R_val, diag)


# ─────────────────────────────────────────────────────────────────────────────
#  Trainer adapter — SAME reward for episode_reward AND r_seq
# ─────────────────────────────────────────────────────────────────────────────

def _env_train_stage() -> Optional[int]:
    v = os.environ.get("TRAIN_STAGE", "").strip()
    if not v:
        return None
    try:
        return int(v) or None
    except ValueError:
        return None


def episode_turn_reward_seq(trajectory, task: Dict[str, Any],
                            gold_observations=None) -> Dict[str, Any]:
    """grpo_train-compatible adapter: {'r_seq', 'episode_reward', 'diagnostics'}.

    r_seq comes from THIS reward's per-turn scores (previously it silently came
    from execution_aware_v2, mixing two different reward definitions).
    """
    res = execution_aware_v3_1_stepwise(
        trajectory, task, gold_observations, train_stage=_env_train_stage())
    r_seq = res.diagnostics.get("turn_scores") or [0.0] * len(trajectory.turns)
    return {
        "r_seq": [float(x) for x in r_seq],
        "episode_reward": float(res.reward),
        "diagnostics": res.diagnostics,
    }


episode_turn_reward_seq.reward_policy = "execution_aware_v3_1_stepwise"  # type: ignore[attr-defined]
