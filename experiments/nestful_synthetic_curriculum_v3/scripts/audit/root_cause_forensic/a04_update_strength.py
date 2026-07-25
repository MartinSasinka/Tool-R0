"""A04 — Optimizer / update-strength audit from raw train logs.

Covers: real optimizer steps, groups with non-zero signal, KL, LR schedule,
clipping, loss stats, missing instrumentation (grad norms), and a scale
reference against the pure_stage3 2-epoch run (which measurably CHANGED
official NESTFUL behavior, -1.14pp, so its update magnitude is a useful
yardstick for "strong enough to matter").
"""
from __future__ import annotations

import statistics as st
from pathlib import Path
from typing import Any, Dict, List

from common import (ARMS, PURE_S3, group_advantages, load_jsonl,
                    load_train_log, write_json)


def _summarize(groups: List[Dict[str, Any]]) -> Dict[str, Any]:
    steps = sum(1 for g in groups if g.get("optimizer_step_executed"))
    lrs = sorted({float(g.get("learning_rate")) for g in groups if g.get("learning_rate") is not None})
    kls = [float(g.get("kl") or 0.0) for g in groups]
    losses = [float(g.get("loss")) for g in groups if g.get("loss") is not None]
    clip = [float(g.get("clipped_rate") or 0.0) for g in groups]
    dead = sum(1 for g in groups if g.get("dead_group"))
    dead_corr = sum(1 for g in groups if g.get("dead_group_corrected"))
    pos_artifact = sum(1 for g in groups if g.get("position_artifact_detected"))
    contributing = [int(g.get("contributing_turns") or 0) for g in groups]

    nonzero_signal = 0
    total_adv_abs = []
    for g in groups:
        _, gs = group_advantages(g["turn_rewards"], g["episode_rewards"])
        advs = [a for row in gs.advantages for a in row]
        if any(abs(a) > 1e-9 for a in advs):
            nonzero_signal += 1
        total_adv_abs.extend(abs(a) for a in advs)

    sample_keys = sorted(groups[0].keys()) if groups else []
    return {
        "n_groups": len(groups),
        "optimizer_steps_executed": steps,
        "learning_rates_seen": lrs,
        "kl_mean": st.mean(kls) if kls else None,
        "kl_max": max(kls) if kls else None,
        "loss_mean": st.mean(losses) if losses else None,
        "clipped_rate_mean": st.mean(clip) if clip else None,
        "dead_group_count": dead,
        "dead_group_corrected_count": dead_corr,
        "position_artifact_count": pos_artifact,
        "groups_with_nonzero_advantage": nonzero_signal,
        "mean_abs_advantage": st.mean(total_adv_abs) if total_adv_abs else None,
        "contributing_turns_total": sum(contributing),
        "has_grad_norm_field": any("grad" in k.lower() for k in sample_keys),
        "logged_fields": sample_keys,
    }


def _pure_s3_logs() -> List[Path]:
    out = []
    for ep in ("epoch_1", "epoch_2"):
        p = PURE_S3 / ep / "train" / "train_log.jsonl"
        if p.is_file():
            out.append(p)
    return out


def main() -> Dict[str, Any]:
    per_arm = {}
    for arm in ARMS:
        _, groups = load_train_log(arm)
        per_arm[arm] = _summarize(groups)

    pure = {}
    for p in _pure_s3_logs():
        rows = load_jsonl(p)
        groups = [r for r in rows if r.get("episode_rewards") and r.get("turn_rewards")]
        pure[p.parent.parent.name] = _summarize(groups)

    payload = {
        "round1_arms": per_arm,
        "pure_stage3_2ep_reference": pure,
        "notes": [
            "grad-norm per step is NOT logged in train_log.jsonl (missing instrumentation).",
            "pure_stage3 2-epoch run is the update-strength yardstick: it produced a "
            "measurable official-eval change (-1.14pp on n=1661).",
        ],
    }
    write_json("a04_update_strength.json", payload)
    return payload


if __name__ == "__main__":
    r = main()
    for arm, s in r["round1_arms"].items():
        print(arm, "steps", s["optimizer_steps_executed"], "kl_mean", s["kl_mean"],
              "signal_groups", s["groups_with_nonzero_advantage"], "/", s["n_groups"])
    for k, s in r["pure_stage3_2ep_reference"].items():
        print("pureS3", k, "steps", s["optimizer_steps_executed"], "groups", s["n_groups"])
