#!/usr/bin/env python3
"""Next-step diagnostics for pure Stage-3 overnight run (C0/E1/E2 test).

Implements roadmap items 1–3, 5–6 from the post-stop analysis plan:
  1. Discordant task audit (gained/lost C0→E2)
  2. Counterfactual training-reward vs official-outcome audit
  3. Per-turn conditional tool / argument accuracy
  5. Structural Stage-3 ↔ NESTFUL similarity (nearest-neighbor)
  6. Process vs episode advantage decomposition (train logs)

Item 4 (synthetic held-out generation + eval) is documented as blocked.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_HERE = Path(__file__).resolve().parent
_V3 = _HERE.parents[1]
_REPO = _V3.parents[1]
_MINIMAL = _REPO / "experiments/nestful_mtgrpo_minimal"
# V3 before scripts/ — scripts/lib must not shadow nestful_synthetic_curriculum_v3/lib.
sys.path.insert(0, str(_MINIMAL))
sys.path.insert(0, str(_V3))
sys.path.append(str(_V3 / "scripts"))

from lib.reward_v3_2_dense import episode_turn_reward_seq  # noqa: E402

from motif_lib import (  # noqa: E402
    default_test_path,
    extract_motifs,
    extract_references_from_value,
    load_jsonl,
    load_task_row,
)
from rollout import Trajectory, Turn  # noqa: E402
from scripts.analysis.two_phase_root_cause_analysis import (  # noqa: E402
    classify_failure,
    official_win,
)

DEFAULT_RUN = _V3 / "outputs/runs/pure_stage3_2ep_20260719_221918"
DEFAULT_STAGE3 = _V3 / "data/training_ready_v5/filtered/stage3_train_ready.jsonl"
OUT = _V3 / "reports/pure_stage3_offline_analysis"
ARM_DIRS = {"C0": "C0_test", "E1": "S3_E1_test", "E2": "S3_E2_test"}
GAMMA = 1.0
LAMBDA_EP = 1.0
_EPS = 1e-9


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)


def _write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_traj_rows(eval_dir: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    with open(eval_dir / "final_eval_trajectories.jsonl", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                out[r["sample_id"]] = r
    return out


def load_tasks(path: Path) -> Dict[str, dict]:
    return {load_task_row(r)["task_id"]: load_task_row(r) for r in load_jsonl(path)}


def traj_from_dict(d: dict) -> Trajectory:
    turns: List[Turn] = []
    for t in d.get("turns") or []:
        turns.append(Turn(
            turn_idx=int(t.get("turn_idx") or 0),
            model_text=str(t.get("model_text") or ""),
            parsed_call=t.get("parsed_call"),
            observation=t.get("observation"),
            fail_reason=t.get("fail_reason"),
            is_terminal=bool(t.get("is_terminal")),
            prompt_tokens=int(t.get("prompt_tokens") or 0),
            completion_tokens=int(t.get("completion_tokens") or 0),
            clipped_completion=bool(t.get("clipped_completion")),
            teacher_forced=bool(t.get("teacher_forced")),
        ))
    return Trajectory(
        task_id=str(d.get("task_id") or ""),
        stage=int(d.get("stage") or 3),
        gold_num_turns=int(d.get("gold_num_turns") or 0),
        turns=turns,
        stop_reason=d.get("stop_reason"),
        executor_mode=str(d.get("executor_mode") or "gold_replay"),
        clipped_any=bool(d.get("clipped_any")),
        prompt_overflow=bool(d.get("prompt_overflow")),
    )


def predicted_calls(row: dict) -> List[dict]:
    turns = (row.get("_traj") or {}).get("turns") or []
    return [t.get("parsed_call") for t in turns if t.get("parsed_call")]


def observations(row: dict) -> List[Any]:
    turns = (row.get("_traj") or {}).get("turns") or []
    return [
        t.get("observation")
        for t in turns
        if t.get("parsed_call") and t.get("fail_reason") is None
    ]


def _call_tool(call: Optional[dict]) -> str:
    return str((call or {}).get("name") or "")


def _call_keys(call: Optional[dict]) -> Tuple[str, ...]:
    args = (call or {}).get("arguments") or {}
    return tuple(sorted(args.keys())) if isinstance(args, dict) else tuple()


def _arg_match_fraction(pred_args: dict, gold_args: dict) -> float:
    if not gold_args:
        return 1.0 if not pred_args else 0.5
    score = 0.0
    for k, gv in gold_args.items():
        if k not in pred_args:
            continue
        pv = pred_args[k]
        if isinstance(gv, str) and gv.strip().startswith("$var"):
            score += 1.0 if isinstance(pv, str) and pv.strip().startswith("$var") else 0.5
        elif pv == gv:
            score += 1.0
        else:
            try:
                if abs(float(pv) - float(gv)) <= 1e-6 * max(1.0, abs(float(gv))):
                    score += 1.0
            except (TypeError, ValueError):
                pass
    return score / len(gold_args)


def longest_common_prefix(calls_a: List[dict], calls_b: List[dict]) -> int:
    n = 0
    for a, b in zip(calls_a, calls_b):
        if _call_tool(a) != _call_tool(b) or _call_keys(a) != _call_keys(b):
            break
        n += 1
    return n


def first_divergence_turn(c0_calls: List[dict], e2_calls: List[dict]) -> Optional[int]:
    """1-based call index where C0 and E2 first differ; None if identical prefix+len."""
    for i, (a, b) in enumerate(zip(c0_calls, e2_calls), start=1):
        if _call_tool(a) != _call_tool(b):
            return i
        pa = (a.get("arguments") or {}) if isinstance(a.get("arguments"), dict) else {}
        pb = (b.get("arguments") or {}) if isinstance(b.get("arguments"), dict) else {}
        if set(pa.keys()) != set(pb.keys()):
            return i
        if pa != pb:
            return i
    if len(c0_calls) != len(e2_calls):
        return min(len(c0_calls), len(e2_calls)) + 1
    return None


def classify_change_type(
    c0_row: dict,
    e2_row: dict,
    gold_calls: List[dict],
    task: dict,
) -> str:
    c0_calls = predicted_calls(c0_row)
    e2_calls = predicted_calls(e2_row)
    gold_n = len(gold_calls)
    c0_win = official_win(c0_row) == 1.0
    e2_win = official_win(e2_row) == 1.0

    if e2_row.get("alternative_valid_solution_pass") and e2_win:
        return "jiná, ale validní cesta"
    if classify_failure(e2_row)[0] == "too few calls":
        return "předčasné ukončení"
    if len(e2_calls) < gold_n and not e2_win:
        return "kratší než gold trace (metrika, ne nutně chyba)"

    div = first_divergence_turn(c0_calls, e2_calls)
    if div is None and c0_calls == e2_calls:
        if c0_win and not e2_win:
            if (c0_row.get("_traj") or {}).get("executable") and (e2_row.get("_traj") or {}).get("executable"):
                return "vykonatelná cesta se špatným výsledkem"
            return "pouze jiná finální odpověď"
        return "beze změny callů"

    idx = (div or 1) - 1
    if idx < len(c0_calls) and idx < len(e2_calls):
        a, b = c0_calls[idx], e2_calls[idx]
        if _call_tool(a) != _call_tool(b):
            return "změněný pozdější tool" if idx >= 1 else "změněný první tool"
        pa = a.get("arguments") or {}
        pb = b.get("arguments") or {}
        if set(pa.keys()) != set(pb.keys()):
            return "stejný tool, jiné keys"
        if pa != pb:
            gold_args = (gold_calls[idx].get("arguments") or {}) if idx < gold_n else {}
            if _uses_observation_wrong(b, idx, e2_row, gold_args):
                return "nesprávné použití observation"
            return "stejný tool, jiné values"

    if (e2_row.get("_traj") or {}).get("executable") and not e2_win:
        return "vykonatelná cesta se špatným výsledkem"
    return "jiná, ale validní cesta" if e2_win else "other"


def _uses_observation_wrong(
    call: dict,
    call_idx: int,
    row: dict,
    gold_args: dict,
) -> bool:
    """Heuristic: gold expects $var ref but pred uses literal inconsistent with prior obs."""
    obs = observations(row)
    args = call.get("arguments") or {}
    if not isinstance(args, dict):
        return False
    for k, gv in gold_args.items():
        if not (isinstance(gv, str) and gv.strip().startswith("$var")):
            continue
        pv = args.get(k)
        if isinstance(pv, str) and pv.strip().startswith("$var"):
            continue
        refs = extract_references_from_value(gv)
        if not refs:
            continue
        ref_idx = refs[0][0]
        if ref_idx < 1 or ref_idx > len(obs):
            continue
        prior = obs[ref_idx - 1]
        if pv != prior:
            try:
                if abs(float(pv) - float(prior)) > 1e-6:
                    return True
            except (TypeError, ValueError):
                return True
    return False


def score_training_reward(row: dict, task: dict) -> dict:
    os.environ.setdefault("TRAIN_STAGE", "3")
    traj = traj_from_dict(row["_traj"])
    out = episode_turn_reward_seq(traj, task)
    diag = out.get("diagnostics") or {}
    return {
        "episode_reward": float(out["episode_reward"]),
        "failure_class": diag.get("failure_class") or diag.get("cap_applied"),
        "quality_score": diag.get("quality_score"),
        "reward_policy": "execution_aware_v3_2_dense",
    }


def per_turn_stats(row: dict, gold_calls: List[dict]) -> dict:
    calls = predicted_calls(row)
    obs = observations(row)
    gold_n = len(gold_calls)
    out: dict = {}
    for t in range(3):
        pos = t
        gold = gold_calls[pos] if pos < gold_n else None
        pred = calls[pos] if pos < len(calls) else None
        prefix_ok = all(
            pos < len(calls)
            and pos < gold_n
            and _call_tool(calls[i]) == _call_tool(gold_calls[i])
            for i in range(pos)
        ) if pos > 0 else True
        if gold is None:
            continue
        tool_ok = bool(pred) and _call_tool(pred) == _call_tool(gold)
        keys_ok = bool(pred) and _call_keys(pred) == _call_keys(gold)
        val_frac = (
            _arg_match_fraction(pred.get("arguments") or {}, gold.get("arguments") or {})
            if pred else 0.0
        )
        obs_ok = None
        if pred and gold:
            ga = gold.get("arguments") or {}
            pa = pred.get("arguments") or {}
            ref_slots = [
                k for k, v in ga.items()
                if isinstance(v, str) and v.strip().startswith("$var")
            ]
            if ref_slots:
                obs_ok = all(
                    isinstance(pa.get(k), str) and pa.get(k, "").strip().startswith("$var")
                    for k in ref_slots
                )
        out[f"turn{pos+1}"] = {
            "tool_ok": tool_ok,
            "keys_ok": keys_ok,
            "val_frac": val_frac,
            "obs_ref_ok": obs_ok,
            "prefix_ok_for_cond": prefix_ok,
            "conditional_tool_ok": tool_ok if (pos == 0 or prefix_ok) else None,
        }
    traj = row.get("_traj") or {}
    out["terminal_outcome_ok"] = official_win(row) == 1.0
    out["executable"] = bool(traj.get("executable"))
    out["executable_wrong"] = (
        bool(traj.get("executable"))
        and official_win(row) != 1.0
        and classify_failure(row)[0] == "executable trajectory ending wrong result"
    )
    return out


def structural_signature(task: dict) -> dict:
    m = extract_motifs(task)
    return {
        "task_id": task["task_id"],
        "motif_type": m["motif_type"],
        "num_calls": m["num_calls"],
        "tool_sequence": m["tool_sequence"],
        "tool_sequence_trigram": m["tool_sequence_trigram"],
        "tool_family": m["tool_family"],
        "linear_chain": m["linear_chain"],
        "fan_in": m["fan_in"],
        "fan_out": m["fan_out"],
        "num_references": m["reference_pattern"]["num_references"],
        "difficulty_score": m["difficulty_score"],
    }


def similarity_score(nestful_sig: dict, stage3_sig: dict) -> float:
    score = 0.0
    if nestful_sig["motif_type"] == stage3_sig["motif_type"]:
        score += 0.20
    if nestful_sig["num_calls"] == stage3_sig["num_calls"]:
        score += 0.15
    na = set(nestful_sig["tool_sequence"].split("->")) - {""}
    nb = set(stage3_sig["tool_sequence"].split("->")) - {""}
    if na | nb:
        score += 0.25 * len(na & nb) / len(na | nb)
    if nestful_sig["tool_sequence_trigram"] and nestful_sig["tool_sequence_trigram"] == stage3_sig["tool_sequence_trigram"]:
        score += 0.20
    elif nestful_sig.get("tool_sequence") == stage3_sig.get("tool_sequence"):
        score += 0.15
    fa = set(nestful_sig["tool_family"].split(",")) - {""}
    fb = set(stage3_sig["tool_family"].split(",")) - {""}
    if fa | fb:
        score += 0.10 * len(fa & fb) / len(fa | fb)
    if nestful_sig["linear_chain"] == stage3_sig["linear_chain"]:
        score += 0.05
    if nestful_sig["num_references"] == stage3_sig["num_references"]:
        score += 0.05
    return round(min(1.0, score), 4)


def nearest_stage3(nestful_task: dict, stage3_sigs: List[dict]) -> Tuple[Optional[str], float]:
    ns = structural_signature(nestful_task)
    best_id, best_sc = None, -1.0
    for ss in stage3_sigs:
        sc = similarity_score(ns, ss)
        if sc > best_sc:
            best_id, best_sc = ss["task_id"], sc
    return best_id, best_sc


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def _turn_returns(r_seq: List[float], episode_reward: float) -> List[float]:
    T = len(r_seq) - 1
    out: List[float] = []
    for t in range(len(r_seq)):
        disc = sum((GAMMA ** (k - t)) * r_seq[k] for k in range(t, len(r_seq)))
        disc += LAMBDA_EP * (GAMMA ** (T - t + 1)) * episode_reward
        out.append(disc)
    return out


def _decompose_returns(r_seq: List[float], episode_reward: float) -> Tuple[List[float], List[float]]:
    T = len(r_seq) - 1
    proc: List[float] = []
    ep: List[float] = []
    for t in range(len(r_seq)):
        p = sum((GAMMA ** (k - t)) * r_seq[k] for k in range(t, len(r_seq)))
        e = LAMBDA_EP * (GAMMA ** (T - t + 1)) * episode_reward
        proc.append(p)
        ep.append(e)
    return proc, ep


def credit_decomposition(run_dir: Path) -> dict:
    from grpo_train import _turn_returns as trainer_turn_returns  # noqa: WPS433
    from group_stats import compute_group_stats  # noqa: WPS433

    rows: List[dict] = []
    for epoch_label, rel in (("E1", "epoch_1/train/train_log.jsonl"), ("E2", "epoch_2/train/train_log.jsonl")):
        path = run_dir / rel
        if not path.is_file():
            continue
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if not rec.get("turn_rewards") or not rec.get("episode_rewards"):
                    continue
                ep_rewards = [float(x) for x in rec["episode_rewards"]]
                turn_rewards = [[float(x) for x in seq] for seq in rec["turn_rewards"]]
                ep_returns = [
                    trainer_turn_returns(seq, R, GAMMA, LAMBDA_EP)
                    for seq, R in zip(turn_rewards, ep_rewards)
                ]
                gstats = compute_group_stats(ep_returns, ep_rewards)
                for ci, (seq, R, Gs, advs) in enumerate(
                    zip(turn_rewards, ep_rewards, ep_returns, gstats.advantages)
                ):
                    proc_parts, ep_parts = _decompose_returns(seq, R)
                    for t in range(len(seq)):
                        rows.append({
                            "epoch": epoch_label,
                            "task_id": rec.get("task_id"),
                            "completion_id": ci,
                            "turn_index": t,
                            "turn_reward": seq[t],
                            "G_t": Gs[t],
                            "P_t": proc_parts[t],
                            "E_t": ep_parts[t],
                            "advantage": advs[t] if t < len(advs) else 0.0,
                            "episode_reward": R,
                            "dead_group": bool(rec.get("dead_group")),
                        })

    by_turn: Dict[int, dict] = {}
    for t in range(3):
        subset = [r for r in rows if r["turn_index"] == t]
        if not subset:
            continue
        proc_vals = [r["P_t"] for r in subset]
        ep_vals = [r["E_t"] for r in subset]
        advs = [r["advantage"] for r in subset]
        local_good = [r for r in subset if r["turn_reward"] >= 0.7]
        local_bad = [r for r in subset if r["turn_reward"] <= 0.3]
        by_turn[t + 1] = {
            "n": len(subset),
            "mean_abs_advantage": _mean([abs(a) for a in advs]),
            "mean_abs_P_t": _mean([abs(x) for x in proc_vals]),
            "mean_abs_E_t": _mean([abs(x) for x in ep_vals]),
            "var_P_share": (
                _std(proc_vals) ** 2
                / max(_EPS, _std(proc_vals) ** 2 + _std(ep_vals) ** 2)
            ),
            "sign_mismatch_rate": _mean([
                1.0 if (p - _mean(proc_vals)) * (e - _mean(ep_vals)) < 0 else 0.0
                for p, e in zip(proc_vals, ep_vals)
            ]),
            "local_good_negative_adv": sum(
                1 for r in local_good if (r["advantage"] or 0) < 0
            ),
            "local_bad_positive_adv": sum(
                1 for r in local_bad if (r["advantage"] or 0) > 0
            ),
        }

    r2_num = 0.0
    r2_den = 0.0
    g0_means: List[float] = []
    ep_means: List[float] = []
    for epoch_label in ("E1", "E2"):
        path = run_dir / f"epoch_{1 if epoch_label=='E1' else 2}/train/train_log.jsonl"
        if not path.is_file():
            continue
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if not rec.get("turn_rewards") or not rec.get("episode_rewards"):
                    continue
                for seq, R in zip(rec["turn_rewards"], rec["episode_rewards"]):
                    Gs = _turn_returns([float(x) for x in seq], float(R))
                    if Gs:
                        g0_means.append(Gs[0])
                        ep_means.append(float(R))
    if g0_means:
        mg, me = _mean(g0_means), _mean(ep_means)
        r2_num = sum((g - mg) * (e - me) for g, e in zip(g0_means, ep_means))
        r2_den = math.sqrt(
            sum((g - mg) ** 2 for g in g0_means) * sum((e - me) ** 2 for e in ep_means)
        )
    r2 = (r2_num / r2_den) ** 2 if r2_den > _EPS else None

    return {
        "generated_at": _now(),
        "note": (
            "R²(G₀~R_ep) is correlation of first-turn return vs episode reward, "
            "NOT fraction of advantage variance from episode term. "
            "Per-turn table uses P_t (discounted future turn rewards) vs E_t "
            "(discounted terminal episode bonus) with trainer-normalized advantages."
        ),
        "r2_G0_vs_R_episode": r2,
        "by_turn": by_turn,
        "n_turn_rows": len(rows),
    }


def run_all(run_dir: Path, stage3_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    nestful_tasks = load_tasks(default_test_path())
    stage3_tasks = load_tasks(stage3_path)
    stage3_sigs = [structural_signature(t) for t in stage3_tasks.values()]

    arms = {a: load_traj_rows(run_dir / "eval" / d) for a, d in ARM_DIRS.items()}
    ids = sorted(set.intersection(*(set(v) for v in arms.values())))
    print(f"[diag] n={len(ids)} tasks, scoring training reward on 3 arms…")

    rewards: Dict[str, Dict[str, dict]] = {a: {} for a in arms}
    for i, sid in enumerate(ids):
        task = nestful_tasks[sid]
        for arm, rows in arms.items():
            rewards[arm][sid] = score_training_reward(rows[sid], task)
        if (i + 1) % 200 == 0:
            print(f"  reward {i+1}/{len(ids)}")

    discordant: List[dict] = []
    for sid in ids:
        w0 = official_win(arms["C0"][sid]) == 1.0
        w2 = official_win(arms["E2"][sid]) == 1.0
        if w0 == w2:
            continue
        task = nestful_tasks[sid]
        gold = task["gold_calls"]
        c0, e2 = arms["C0"][sid], arms["E2"][sid]
        c0_calls, e2_calls = predicted_calls(c0), predicted_calls(e2)
        div = first_divergence_turn(c0_calls, e2_calls)
        discordant.append({
            "task_id": sid,
            "transition": "gained" if (not w0 and w2) else "lost",
            "gold_motif": extract_motifs(task).get("motif_type"),
            "gold_call_count": len(gold),
            "C0_calls": len(c0_calls),
            "E1_calls": len(predicted_calls(arms["E1"][sid])),
            "E2_calls": len(e2_calls),
            "longest_common_prefix": longest_common_prefix(c0_calls, e2_calls),
            "first_changed_turn": div,
            "change_type": classify_change_type(c0, e2, gold, task),
            "C0_failure": classify_failure(c0)[0],
            "E2_failure": classify_failure(e2)[0],
            "C0_actual_observations": observations(c0),
            "E2_actual_observations": observations(e2),
            "C0_final_outcome": "win" if w0 else "loss",
            "E2_final_outcome": "win" if w2 else "loss",
            "C0_under_calling_metric": len(c0_calls) < len(gold),
            "E2_under_calling_metric": len(e2_calls) < len(gold),
            "C0_too_few_taxonomy": classify_failure(c0)[0] == "too few calls",
            "E2_too_few_taxonomy": classify_failure(e2)[0] == "too few calls",
            "R_train_C0": rewards["C0"][sid]["episode_reward"],
            "R_train_E1": rewards["E1"][sid]["episode_reward"],
            "R_train_E2": rewards["E2"][sid]["episode_reward"],
            "R_class_C0": rewards["C0"][sid]["failure_class"],
            "R_class_E2": rewards["E2"][sid]["failure_class"],
        })

    _write_jsonl(out_dir / "PURE_STAGE3_DISCORDANT_AUDIT.jsonl", discordant)

    lost = [r for r in discordant if r["transition"] == "lost"]
    gained = [r for r in discordant if r["transition"] == "gained"]
    change_ctr = Counter(r["change_type"] for r in discordant)
    div_lost = Counter(r["first_changed_turn"] for r in lost if r["first_changed_turn"])

    misalign = Counter()
    for r in lost:
        rc0, re2 = r["R_train_C0"], r["R_train_E2"]
        if re2 > rc0:
            misalign["R_train(E2) > R_train(C0)"] += 1
        elif abs(re2 - rc0) < 1e-9:
            misalign["R_train(E2) = R_train(C0)"] += 1
        else:
            misalign["R_train(E2) < R_train(C0)"] += 1

    reward_summary = {
        "generated_at": _now(),
        "policy": "execution_aware_v3_2_dense",
        "n_tasks": len(ids),
        "all_tasks": {
            arm: {
                "mean_R_train": _mean([rewards[arm][s]["episode_reward"] for s in ids]),
                "mean_R_when_official_win": _mean([
                    rewards[arm][s]["episode_reward"]
                    for s in ids if official_win(arms[arm][s]) == 1.0
                ]),
                "mean_R_when_official_loss": _mean([
                    rewards[arm][s]["episode_reward"]
                    for s in ids if official_win(arms[arm][s]) != 1.0
                ]),
            }
            for arm in arms
        },
        "C0_win_E2_loss_n": len(lost),
        "C0_win_E2_loss_reward_ordering": dict(misalign),
        "E2_loss_higher_than_C0_win_rate": (
            misalign["R_train(E2) > R_train(C0)"] / len(lost) if lost else None
        ),
        "executable_wrong_final_mean_reward": {
            arm: _mean([
                rewards[arm][s]["episode_reward"]
                for s in ids
                if classify_failure(arms[arm][s])[0] == "executable trajectory ending wrong result"
            ])
            for arm in arms
        },
        "fully_correct_mean_reward": {
            arm: _mean([
                rewards[arm][s]["episode_reward"]
                for s in ids
                if rewards[arm][s]["failure_class"] == "fully_correct"
            ])
            for arm in arms
        },
        "official_win_vs_train_class_crosstab_E2": {
            f"official_win={ow}|class={cls}": n
            for (ow, cls), n in Counter(
                (official_win(arms["E2"][s]) == 1.0, rewards["E2"][s]["failure_class"])
                for s in ids
            ).items()
        },
    }
    _write_json(out_dir / "PURE_STAGE3_REWARD_COUNTERFACTUAL.json", reward_summary)

    per_turn: Dict[str, dict] = {}
    for arm, rows in arms.items():
        cond = {f"turn{t}_tool": [] for t in (1, 2, 3)}
        cond.update({f"turn{t}_arg": [] for t in (2, 3)})
        obs_refs: List[float] = []
        terminal_exec: List[bool] = []
        for sid in ids:
            st = per_turn_stats(rows[sid], nestful_tasks[sid]["gold_calls"])
            for t in (1, 2, 3):
                key = f"turn{t}"
                if key not in st:
                    continue
                if t == 1:
                    cond["turn1_tool"].append(1.0 if st[key]["tool_ok"] else 0.0)
                else:
                    if st[key]["conditional_tool_ok"] is not None:
                        cond[f"turn{t}_tool"].append(1.0 if st[key]["conditional_tool_ok"] else 0.0)
                if t >= 2 and st[key]["prefix_ok_for_cond"]:
                    cond[f"turn{t}_arg"].append(st[key]["val_frac"])
                if st[key]["obs_ref_ok"] is not None and st[key]["prefix_ok_for_cond"]:
                    obs_refs.append(1.0 if st[key]["obs_ref_ok"] else 0.0)
            if st["executable"]:
                terminal_exec.append(st["terminal_outcome_ok"])
        per_turn[arm] = {
            "first_tool_acc": _mean(cond["turn1_tool"]),
            "turn2_tool_given_turn1": _mean(cond["turn2_tool"]),
            "turn3_tool_given_prefix12": _mean(cond["turn3_tool"]),
            "turn2_arg_given_turn1": _mean(cond["turn2_arg"]),
            "turn3_arg_given_prefix12": _mean(cond["turn3_arg"]),
            "observation_ref_turn23": _mean(obs_refs),
            "terminal_outcome_given_executable": _mean([1.0 if x else 0.0 for x in terminal_exec]),
            "n_obs_ref_samples": len(obs_refs),
            "n_executable": len(terminal_exec),
        }
    per_turn_json = {
        "generated_at": _now(),
        "n": len(ids),
        "arms": per_turn,
        "under_calling_metric_rate": {
            arm: _mean([
                1.0 if (arms[arm][s].get("_traj") or {}).get("num_tool_calls", 0)
                < arms[arm][s].get("num_gold_calls", 0)
                else 0.0
                for s in ids
            ])
            for arm in arms
        },
        "too_few_calls_taxonomy_rate": {
            arm: _mean([
                1.0 if classify_failure(arms[arm][s])[0] == "too few calls" else 0.0
                for s in ids
            ])
            for arm in arms
        },
    }
    _write_json(out_dir / "PURE_STAGE3_PER_TURN_ACCURACY.json", per_turn_json)

    sim_rows: List[dict] = []
    gained_sim: List[float] = []
    lost_sim: List[float] = []
    stable_win_sim: List[float] = []
    stable_loss_sim: List[float] = []
    for sid in ids:
        task = nestful_tasks[sid]
        nn_id, nn_sc = nearest_stage3(task, stage3_sigs)
        w0 = official_win(arms["C0"][sid]) == 1.0
        w2 = official_win(arms["E2"][sid]) == 1.0
        row = {
            "task_id": sid,
            "nearest_stage3_id": nn_id,
            "similarity_score": nn_sc,
            "C0_win": w0,
            "E2_win": w2,
            "transition": (
                "gained" if (not w0 and w2) else
                "lost" if (w0 and not w2) else
                "stable_win" if (w0 and w2) else "stable_loss"
            ),
        }
        sim_rows.append(row)
        if row["transition"] == "gained":
            gained_sim.append(nn_sc)
        elif row["transition"] == "lost":
            lost_sim.append(nn_sc)
        elif row["transition"] == "stable_win":
            stable_win_sim.append(nn_sc)
        elif row["transition"] == "stable_loss":
            stable_loss_sim.append(nn_sc)

    struct_summary = {
        "generated_at": _now(),
        "stage3_train_n": len(stage3_tasks),
        "mean_similarity": {
            "gained_C0_to_E2": _mean(gained_sim),
            "lost_C0_to_E2": _mean(lost_sim),
            "stable_win": _mean(stable_win_sim),
            "stable_loss": _mean(stable_loss_sim),
            "all_test": _mean([r["similarity_score"] for r in sim_rows]),
        },
        "n": {
            "gained": len(gained_sim),
            "lost": len(lost_sim),
        },
        "interpretation_hint": (
            "If gained tasks have higher nearest-neighbor similarity to Stage-3 train "
            "than lost tasks, transfer gap is measurable (model improves on train-like tasks)."
        ),
    }
    _write_json(out_dir / "PURE_STAGE3_STRUCTURAL_SIMILARITY.json", struct_summary)
    _write_jsonl(out_dir / "PURE_STAGE3_STRUCTURAL_SIMILARITY.jsonl", sim_rows)

    credit = credit_decomposition(run_dir)
    _write_json(out_dir / "PURE_STAGE3_CREDIT_DECOMPOSITION.json", credit)

    _write_csv(out_dir / "PURE_STAGE3_DISCORDANT_AUDIT.csv", discordant, [
        "task_id", "transition", "gold_motif", "gold_call_count",
        "C0_calls", "E1_calls", "E2_calls", "longest_common_prefix",
        "first_changed_turn", "change_type", "C0_failure", "E2_failure",
        "C0_final_outcome", "E2_final_outcome",
        "R_train_C0", "R_train_E2", "R_class_C0", "R_class_E2",
    ])

    md_disc = _format_discordant_md(discordant, gained, lost, change_ctr, div_lost)
    md_rew = _format_reward_md(reward_summary, misalign, lost)
    md_turn = _format_per_turn_md(per_turn, per_turn_json)
    md_struct = _format_struct_md(struct_summary)
    md_credit = _format_credit_md(credit)
    md_roadmap = _format_roadmap_md()

    _write_md(out_dir / "PURE_STAGE3_DISCORDANT_AUDIT.md", md_disc)
    _write_md(out_dir / "PURE_STAGE3_REWARD_COUNTERFACTUAL.md", md_rew)
    _write_md(out_dir / "PURE_STAGE3_PER_TURN_ACCURACY.md", md_turn)
    _write_md(out_dir / "PURE_STAGE3_STRUCTURAL_SIMILARITY.md", md_struct)
    _write_md(out_dir / "PURE_STAGE3_CREDIT_DECOMPOSITION.md", md_credit)
    _write_md(out_dir / "PURE_STAGE3_NEXT_STEPS_ROADMAP.md", md_roadmap)

    print(f"[diag] wrote reports under {out_dir}")


def _write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _pct(x: Optional[float]) -> str:
    if x is None:
        return "—"
    return f"{100.0 * x:.2f}%"


def _format_discordant_md(
    all_rows: List[dict],
    gained: List[dict],
    lost: List[dict],
    change_ctr: Counter,
    div_lost: Counter,
) -> str:
    lines = [
        "# Discordant task audit (C0↔E2)",
        "",
        f"**Generated:** {_now()}",
        f"**Discordant tasks:** {len(all_rows)} ({len(gained)} gained, {len(lost)} lost)",
        "",
        "## Change type (all discordant)",
        "",
        "| change_type | count | share |",
        "|-------------|------:|------:|",
    ]
    for k, v in change_ctr.most_common():
        lines.append(f"| {k} | {v} | {100*v/len(all_rows):.1f}% |")
    lines += [
        "",
        "## C0 win → E2 loss — first divergence turn",
        "",
        "| first_changed_turn | count |",
        "|-------------------:|------:|",
    ]
    for k in sorted(div_lost):
        lines.append(f"| {k} | {div_lost[k]} |")
    lines += [
        "",
        "## Under-calling metric vs taxonomy (all test, reminder)",
        "",
        "Metric `pred_calls < gold_calls` is **not** premature stop. "
        "See per-turn report for taxonomy `too few calls` rate (~0.8%).",
        "",
        f"- Discordant lost with under-calling metric: "
        f"{sum(1 for r in lost if r['E2_under_calling_metric'])}/{len(lost)}",
        f"- Discordant lost with too_few taxonomy: "
        f"{sum(1 for r in lost if r['E2_too_few_taxonomy'])}/{len(lost)}",
        "",
        "Full rows: `PURE_STAGE3_DISCORDANT_AUDIT.jsonl`",
    ]
    return "\n".join(lines)


def _format_reward_md(summary: dict, misalign: Counter, lost: List[dict]) -> str:
    lines = [
        "# Counterfactual training reward audit",
        "",
        f"**Policy:** `{summary['policy']}` applied to saved NESTFUL trajectories",
        f"**Generated:** {summary['generated_at']}",
        "",
        "## Mean R_train by official outcome",
        "",
        "| Arm | mean R (all) | R | official win | R | official loss |",
        "|-----|-------------:|---|-------------:|---|--------------:|",
    ]
    for arm in ("C0", "E1", "E2"):
        a = summary["all_tasks"][arm]
        lines.append(
            f"| {arm} | {a['mean_R_train']:.4f} | | "
            f"{a['mean_R_when_official_win']:.4f} | | "
            f"{a['mean_R_when_official_loss']:.4f} |"
        )
    lines += [
        "",
        f"## C0 win → E2 loss (n={summary['C0_win_E2_loss_n']})",
        "",
        "| Ordering | count | share |",
        "|----------|------:|------:|",
    ]
    n = summary["C0_win_E2_loss_n"] or 1
    for k in ("R_train(E2) > R_train(C0)", "R_train(E2) = R_train(C0)", "R_train(E2) < R_train(C0)"):
        v = misalign.get(k, 0)
        lines.append(f"| {k} | {v} | {100*v/n:.1f}% |")
    ew = summary["executable_wrong_final_mean_reward"]
    fc = summary["fully_correct_mean_reward"]
    lines += [
        "",
        "## Terminal class mean reward (E2)",
        "",
        f"- executable_wrong_final: **{ew['E2']:.4f}** (C0: {ew['C0']:.4f})",
        f"- fully_correct band: **{fc['E2']:.4f}** (C0: {fc['C0']:.4f})",
        "",
        "**Note:** eval `reward_train_strict` is strict gold-trace reward, not this policy.",
    ]
    return "\n".join(lines)


def _format_per_turn_md(per_turn: Dict[str, dict], per_turn_json: dict) -> str:
    lines = [
        "# Per-turn conditional accuracy (NESTFUL test)",
        "",
        f"**Generated:** {_now()}",
        "",
        "| Metrika | C0 | E1 | E2 |",
        "|---------|---:|---:|---:|",
    ]
    metrics = [
        ("správný první tool", "first_tool_acc"),
        ("správný 2. tool při správném 1.", "turn2_tool_given_turn1"),
        ("správný 3. tool při správném prefixu 1–2", "turn3_tool_given_prefix12"),
        ("správná reference na observation (turn 2–3)", "observation_ref_turn23"),
        ("správný terminal outcome při executable", "terminal_outcome_given_executable"),
    ]
    for label, key in metrics:
        lines.append(
            "| "
            + label
            + " | "
            + " | ".join(_pct(per_turn[a].get(key)) for a in ("C0", "E1", "E2"))
            + " |"
        )
    uc = per_turn_json.get("under_calling_metric_rate", {})
    tf = per_turn_json.get("too_few_calls_taxonomy_rate", {})
    lines += [
        "",
        "## Under-calling vs premature stop",
        "",
        "| Metrika | C0 | E1 | E2 |",
        "|---------|---:|---:|---:|",
        f"| pred_calls < gold_calls (eval metric) | {_pct(uc.get('C0'))} | {_pct(uc.get('E1'))} | {_pct(uc.get('E2'))} |",
        f"| taxonomy: too few calls | {_pct(tf.get('C0'))} | {_pct(tf.get('E1'))} | {_pct(tf.get('E2'))} |",
        "",
        "Rozdíl ~60 pp vs ~0.8 pp potvrzuje, že under-calling metrika **nesmí** řídit SFT.",
    ]
    return "\n".join(lines)


def _format_struct_md(summary: dict) -> str:
    m = summary["mean_similarity"]
    lines = [
        "# Structural Stage-3 ↔ NESTFUL similarity",
        "",
        f"**Stage-3 train tasks:** {summary['stage3_train_n']}",
        f"**Generated:** {summary['generated_at']}",
        "",
        "| Cohort | mean nearest-neighbor similarity | n |",
        "|--------|--------------------------------:|--:|",
        f"| gained (C0 loss → E2 win) | {m['gained_C0_to_E2']:.4f} | {summary['n']['gained']} |",
        f"| lost (C0 win → E2 loss) | {m['lost_C0_to_E2']:.4f} | {summary['n']['lost']} |",
        f"| stable win | {m['stable_win']:.4f} | — |",
        f"| stable loss | {m['stable_loss']:.4f} | — |",
        f"| all test | {m['all_test']:.4f} | — |",
        "",
        summary["interpretation_hint"],
    ]
    return "\n".join(lines)


def _format_credit_md(credit: dict) -> str:
    lines = [
        "# Credit assignment decomposition (train logs)",
        "",
        f"**Generated:** {credit['generated_at']}",
        "",
        f"R²(G₀ ~ R_episode) = **{credit['r2_G0_vs_R_episode']:.4f}** "
        "(correlation only; see note below)",
        "",
        credit["note"],
        "",
        "| Turn | n | mean |A| | mean |P_t| | mean |E_t| | var share P | sign mismatch | good turn, neg adv | bad turn, pos adv |",
        "|------|--:|-----------:|-----------:|-----------:|------------:|--------------:|-------------------:|------------------:|",
    ]
    for t, row in sorted(credit.get("by_turn", {}).items()):
        lines.append(
            f"| {t} | {row['n']} | {row['mean_abs_advantage']:.4f} | "
            f"{row['mean_abs_P_t']:.4f} | {row['mean_abs_E_t']:.4f} | "
            f"{row['var_P_share']:.3f} | {row['sign_mismatch_rate']:.3f} | "
            f"{row['local_good_negative_adv']} | {row['local_bad_positive_adv']} |"
        )
    return "\n".join(lines)


def _format_roadmap_md() -> str:
    return "\n".join([
        "# Next steps roadmap (post pure Stage-3 stop)",
        "",
        f"**Updated:** {_now()}",
        "",
        "## Completed offline analyses",
        "",
        "1. Discordant audit — `PURE_STAGE3_DISCORDANT_AUDIT.*`",
        "2. Counterfactual R_train — `PURE_STAGE3_REWARD_COUNTERFACTUAL.*`",
        "3. Per-turn accuracy — `PURE_STAGE3_PER_TURN_ACCURACY.*`",
        "5. Structural similarity — `PURE_STAGE3_STRUCTURAL_SIMILARITY.*`",
        "6. Credit decomposition — `PURE_STAGE3_CREDIT_DECOMPOSITION.*`",
        "",
        "## Blocked / not run",
        "",
        "4. **Synthetic held-out (200–300 new Stage-3 tasks)** — requires fresh generation "
        "with question/template/tool-combo dedup, then C0/E1/E2 eval at temp=0 and train-config rollouts.",
        "",
        "## Decision tree (unchanged)",
        "",
        "- Reward misalignment (E2 loss often R_train > C0 win) → **B1 outcome-first** ablation",
        "- Held-out↑, NESTFUL↓ → new data + SFT pilot",
        "- Held-out flat + correct reward → LR/KL ablation (deferred)",
        "",
        "## Do not run yet",
        "",
        "- Third epoch same recipe",
        "- LR increase",
        "- SFT against 60% under-calling metric",
        "- Large Stage 4/5 dataset before held-out + reward audit",
    ])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    ap.add_argument("--stage3-path", type=Path, default=DEFAULT_STAGE3)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    args = ap.parse_args()
    run_all(args.run_dir, args.stage3_path, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
