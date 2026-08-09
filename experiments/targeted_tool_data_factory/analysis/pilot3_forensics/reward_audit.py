"""Reward / dead-group audit from existing logs and optional rollouts."""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .io import read_jsonl


def extract_train_log_aggregates(log_path: Path) -> Dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    out: Dict[str, Any] = {"source": str(log_path), "aggregates": {}, "raw_matches": {}}
    patterns = {
        "dead_group_rate": r"epoch/dead_group_rate\s+([0-9.]+)",
        "mean_reward": r"epoch/mean_reward\s+([0-9.]+)",
        "mean_reward_dense": r"epoch/mean_reward_dense\s+([0-9.]+)",
        "mean_unique_rewards": r"epoch/mean_unique_rewards\s+([0-9.]+)",
        "tasks_seen": r"epoch/tasks_seen\s+([0-9]+)",
        "fallback_used": r"epoch/fallback_used\s+([0-9.]+)",
        "kl_beta": r"(?:kl_beta|training\.kl_beta)[=:\s]+([0-9.]+)",
        "run_id": r"(pilot3_D1_[A-Za-z0-9_]+)",
        "reward_policy": r"(A4_GATED_VERIFIABLE|reward_ablation_[A-Za-z0-9_]+)",
    }
    for k, pat in patterns.items():
        ms = re.findall(pat, text, flags=re.IGNORECASE)
        if ms:
            out["raw_matches"][k] = ms
            val = ms[-1]
            try:
                out["aggregates"][k] = float(val) if "." in val else (val if not val.replace(".", "").isdigit() else int(val))
                if k in ("dead_group_rate", "mean_reward", "mean_reward_dense", "mean_unique_rewards", "kl_beta", "fallback_used"):
                    out["aggregates"][k] = float(val)
                elif k in ("tasks_seen",):
                    out["aggregates"][k] = int(float(val))
                else:
                    out["aggregates"][k] = val
            except ValueError:
                out["aggregates"][k] = val
    return out


def _group_key(row: Dict[str, Any]) -> str:
    for k in ("group_id", "prompt_id", "sample_id", "task_id"):
        if row.get(k) is not None:
            return str(row[k])
    return short_fallback(row)


def short_fallback(row: Dict[str, Any]) -> str:
    return str(row.get("id") or row.get("uid") or "unknown")


def classify_group(rewards: Sequence[float], terminal_ok: Optional[Sequence[bool]] = None) -> str:
    if not rewards:
        return "EMPTY"
    uniq = len(set(round(r, 6) for r in rewards))
    mn, mx = min(rewards), max(rewards)
    span = mx - mn
    all_success = terminal_ok is not None and all(terminal_ok) and len(terminal_ok) == len(rewards)
    all_fail = terminal_ok is not None and (not any(terminal_ok)) and len(terminal_ok) == len(rewards)
    if span < 1e-9:
        if all_success:
            return "DEAD_ALL_SUCCESS"
        if all_fail:
            return "DEAD_ALL_FAIL"
        return "DEAD_EQUAL_PARTIAL"
    # mixed
    if terminal_ok is not None and len(set(terminal_ok)) > 1:
        return "MIXED_TERMINAL"
    if terminal_ok is not None and len(set(terminal_ok)) == 1 and uniq > 1:
        return "MIXED_PROCESS_ONLY"
    if uniq > 1:
        return "MIXED_BOTH"
    return "MIXED_BOTH"


def audit_rollouts(rollout_path: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    rows = read_jsonl(rollout_path)
    if not rows:
        return [], [], {"status": "empty"}

    # detect reward field
    reward_keys = ["reward", "terminal_reward", "episode_reward", "r", "total_reward"]
    sample = rows[0]
    rkey = next((k for k in reward_keys if k in sample), None)
    if rkey is None:
        # nested
        for k in sample:
            if isinstance(sample[k], dict) and any(x in sample[k] for x in reward_keys):
                rkey = k
                break

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[_group_key(r)].append(r)

    group_rows = []
    for gid, items in groups.items():
        rewards = []
        terminals = []
        for it in items:
            if rkey and isinstance(it.get(rkey), (int, float)):
                rewards.append(float(it[rkey]))
            elif rkey and isinstance(it.get(rkey), dict):
                inner = it[rkey]
                for rk in reward_keys:
                    if rk in inner and isinstance(inner[rk], (int, float)):
                        rewards.append(float(inner[rk]))
                        break
            elif isinstance(it.get("reward"), (int, float)):
                rewards.append(float(it["reward"]))
            win = it.get("official_win", it.get("official_success", it.get("success")))
            if win is not None:
                terminals.append(bool(win))
        if not rewards:
            continue
        gtype = classify_group(rewards, terminals or None)
        meta = items[0]
        group_rows.append({
            "group_id": gid,
            "group_size": len(items),
            "reward_mean": sum(rewards) / len(rewards),
            "reward_std": (sum((x - sum(rewards)/len(rewards))**2 for x in rewards) / len(rewards)) ** 0.5,
            "reward_min": min(rewards),
            "reward_max": max(rewards),
            "n_unique_rewards": len(set(round(x, 6) for x in rewards)),
            "group_type": gtype,
            "dead_group": int(gtype.startswith("DEAD_")),
            "generation_cell": meta.get("generation_cell") or (meta.get("provenance") or {}).get("generation_cell_id") if isinstance(meta.get("provenance"), dict) else "",
            "call_count": meta.get("num_calls") or meta.get("call_count") or "",
            "motif": meta.get("motif_type") or meta.get("motif") or "",
            "track": meta.get("track") or (meta.get("provenance") or {}).get("track") if isinstance(meta.get("provenance"), dict) else "",
            "target_failure_mode": meta.get("target_failure_mode") or "",
        })

    by_cell_counter: Dict[str, Counter] = defaultdict(Counter)
    for g in group_rows:
        cell = str(g.get("generation_cell") or "UNKNOWN")
        by_cell_counter[cell][g["group_type"]] += 1
        by_cell_counter[cell]["n"] += 1
        by_cell_counter[cell]["dead"] += g["dead_group"]

    by_cell = []
    for cell, c in sorted(by_cell_counter.items()):
        n = c["n"] or 1
        by_cell.append({
            "generation_cell": cell,
            "n_groups": c["n"],
            "dead_rate": c["dead"] / n,
            "DEAD_ALL_SUCCESS": c.get("DEAD_ALL_SUCCESS", 0),
            "DEAD_ALL_FAIL": c.get("DEAD_ALL_FAIL", 0),
            "DEAD_EQUAL_PARTIAL": c.get("DEAD_EQUAL_PARTIAL", 0),
            "MIXED_TERMINAL": c.get("MIXED_TERMINAL", 0),
            "MIXED_PROCESS_ONLY": c.get("MIXED_PROCESS_ONLY", 0),
            "MIXED_BOTH": c.get("MIXED_BOTH", 0),
            "effective_group_rate": 1.0 - c["dead"] / n,
        })

    summary = {
        "status": "per_rollout",
        "n_rollouts": len(rows),
        "n_groups": len(group_rows),
        "dead_group_rate": sum(g["dead_group"] for g in group_rows) / max(1, len(group_rows)),
        "group_type_counts": dict(Counter(g["group_type"] for g in group_rows)),
        "reward_field": rkey,
    }
    return group_rows, by_cell, summary


def run_reward_audit(
    *,
    train_log: Optional[Path],
    rollout_log: Optional[Path],
    rollout_is_pilot3: bool,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "per_rollout_available": False,
        "aggregates_only": True,
        "train_log_aggregates": {},
        "groups": [],
        "groups_by_cell": [],
        "missing_observability": [],
    }
    if train_log and train_log.is_file():
        result["train_log_aggregates"] = extract_train_log_aggregates(train_log)

    if rollout_log and rollout_log.is_file() and rollout_is_pilot3:
        groups, by_cell, summary = audit_rollouts(rollout_log)
        result["per_rollout_available"] = True
        result["aggregates_only"] = False
        result["groups"] = groups
        result["groups_by_cell"] = by_cell
        result["rollout_summary"] = summary
    else:
        result["missing_observability"] = [
            "dead-group rate can be verified only as a train-log aggregate",
            "cannot determine all-success vs all-fail share for D1",
            "cannot identify problematic generation cells for D1 reward groups",
            "cannot evaluate reward ranking alignment on D1 rollouts",
            "this is critical missing observability for reward-bottleneck claims",
        ]
        if rollout_log and rollout_log.is_file() and not rollout_is_pilot3:
            result["note"] = (
                f"Found non-Pilot3 rollout artifact at {rollout_log}; "
                "not used as D1 evidence."
            )
    return result


def reward_md(audit: Dict[str, Any]) -> str:
    lines = ["# REWARD_AUDIT", ""]
    agg = (audit.get("train_log_aggregates") or {}).get("aggregates") or {}
    lines.append("## Train-log aggregates")
    lines.append("")
    if agg:
        for k, v in sorted(agg.items()):
            lines.append(f"- `{k}`: `{v}`")
    else:
        lines.append("- *(none extracted)*")
    lines.append("")
    if audit.get("per_rollout_available"):
        s = audit.get("rollout_summary") or {}
        lines += [
            "## Per-rollout groups",
            "",
            f"- n_groups: {s.get('n_groups')}",
            f"- dead_group_rate: {s.get('dead_group_rate')}",
            f"- types: `{s.get('group_type_counts')}`",
            "",
        ]
    else:
        lines += ["## Per-rollout groups", "", "- **Not available for Pilot3 D1.**", ""]
    return "\n".join(lines) + "\n"


def missing_observability_md(audit: Dict[str, Any]) -> str:
    lines = [
        "# MISSING_OBSERVABILITY",
        "",
        "Pilot3 D1 training did not leave locally recoverable per-rollout reward groups.",
        "",
        "## Consequences",
        "",
    ]
    for m in audit.get("missing_observability") or []:
        lines.append(f"- {m}")
    lines += [
        "",
        "## Future logging requirements (do not run now)",
        "",
        "- Persist each GRPO group: prompt_id, sample_id, generation_cell, rollout_id, step",
        "- Persist terminal reward T, process reward P, total R, epsilon",
        "- Persist parse_valid / executable / official_win per rollout",
        "- Persist unique reward count and group dead flag at write time",
        "- Write `train_rollouts.jsonl` under the run directory before checkpointing",
        "",
    ]
    return "\n".join(lines) + "\n"
