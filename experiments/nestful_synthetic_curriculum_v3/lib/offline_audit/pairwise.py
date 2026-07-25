from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from lib.offline_audit import ARMS
from lib.offline_audit.grpo_math import group_returns_and_advantages, rollout_scalar_advantage
from lib.offline_audit.paths import load_train_groups, train_log_path
from lib.offline_audit.stats_util import (
    cosine,
    pearson,
    sign_agreement,
    spearman,
    top_bottom_agreement,
)


def _groups_by_task(runs_root: Path, arm: str, seed: str) -> Dict[str, Dict]:
    return {g["task_id"]: g for g in load_train_groups(train_log_path(runs_root, arm, seed))}


def _hash_matched_pairs(
    g_a: Dict, g_b: Dict
) -> List[Tuple[float, float, float, float]]:
    """Returns (reward_a, reward_b, adv_a, adv_b) per matched completion hash."""
    h_b = list(g_b.get("completion_hashes") or [])
    e_b = [float(x) for x in g_b["episode_rewards"]]
    tr_b = [[float(x) for x in s] for s in g_b["turn_rewards"]]
    _, gs_b = group_returns_and_advantages(tr_b, e_b)

    h_a = list(g_a.get("completion_hashes") or [])
    e_a = [float(x) for x in g_a["episode_rewards"]]
    tr_a = [[float(x) for x in s] for s in g_a["turn_rewards"]]
    _, gs_a = group_returns_and_advantages(tr_a, e_a)

    pairs = []
    for i, ha in enumerate(h_a):
        if ha not in h_b:
            continue
        j = h_b.index(ha)
        pairs.append(
            (
                e_a[i],
                e_b[j],
                rollout_scalar_advantage(gs_a, i),
                rollout_scalar_advantage(gs_b, j),
            )
        )
    return pairs


def counterfactual_and_pairwise(
    runs_root: Path,
    seed: str,
    reports_dir: Path,
    canonical_arm: str,
) -> Dict[str, Any]:
    status = {
        "mode": "PARTIAL",
        "reason": (
            "Stored training artifacts do not include parsed trajectories required for "
            "frozen registry re-scoring. Cross-arm comparison uses hash-matched rollouts "
            "and per-group logged episode_rewards with recomputed GRPO advantages."
        ),
        "canonical_arm": canonical_arm,
    }
    arms_data = {a: _groups_by_task(runs_root, a, seed) for a in ARMS}
    if canonical_arm not in arms_data:
        status["error"] = f"canonical arm missing: {canonical_arm}"
        return status

    reward_rows: List[Dict[str, Any]] = []
    adv_rows: List[Dict[str, Any]] = []
    pair_rows: List[Dict[str, Any]] = []

    compare_pairs = [
        ("A0_R0_CURRENT", "A2_R3_OUTCOME_FIRST"),
        ("A0_R0_CURRENT", "A4_GATED_VERIFIABLE"),
        ("A2_R3_OUTCOME_FIRST", "A4_GATED_VERIFIABLE"),
        ("A0_R0_CURRENT", "A1_OUTCOME_ONLY"),
    ]

    for a1, a2 in compare_pairs:
        if a1 not in arms_data or a2 not in arms_data:
            continue
        shared_tasks = set(arms_data[a1]) & set(arms_data[a2])
        hash_rewards_a: List[float] = []
        hash_rewards_b: List[float] = []
        hash_adv_a: List[float] = []
        hash_adv_b: List[float] = []
        group_pearson: List[float] = []
        top_agree = 0
        bot_agree = 0
        n_groups = 0
        inv_count = 0
        inv_total = 0
        for tid in shared_tasks:
            ga = arms_data[a1][tid]
            gb = arms_data[a2][tid]
            for ra, rb, aa, ab in _hash_matched_pairs(ga, gb):
                hash_rewards_a.append(ra)
                hash_rewards_b.append(rb)
                hash_adv_a.append(aa)
                hash_adv_b.append(ab)
            ea = [float(x) for x in ga["episode_rewards"]]
            eb = [float(x) for x in gb["episode_rewards"]]
            p = pearson(ea, eb)
            if p is not None:
                group_pearson.append(p)
            n_groups += 1
            top_ok, bot_ok = top_bottom_agreement(ea, eb)
            if top_ok:
                top_agree += 1
            if bot_ok:
                bot_agree += 1
            ra_idx = sorted(range(len(ea)), key=lambda i: ea[i])
            rb_idx = sorted(range(len(eb)), key=lambda i: eb[i])
            for i in range(len(ea)):
                for j in range(i + 1, len(ea)):
                    inv_total += 1
                    da = ea[i] - ea[j]
                    db = eb[i] - eb[j]
                    if da == 0 or db == 0:
                        continue
                    if (da > 0) != (db > 0):
                        inv_count += 1

        pr_hash = pearson(hash_rewards_a, hash_rewards_b)
        sp_hash = spearman(hash_rewards_a, hash_rewards_b)
        pr_adv = pearson(hash_adv_a, hash_adv_b)
        cos_adv = cosine(hash_adv_a, hash_adv_b)
        sgn = sign_agreement(hash_adv_a, hash_adv_b)
        flips = sum(
            1
            for x, y in zip(hash_adv_a, hash_adv_b)
            if (x >= 0) != (y >= 0) and abs(x) > 1e-9 and abs(y) > 1e-9
        )
        effectively_equiv = (
            cos_adv is not None
            and cos_adv >= 0.95
            and (sgn or 0) >= 0.95
            and n_groups
            and top_agree / n_groups >= 0.90
        )
        row = {
            "arm_a": a1,
            "arm_b": a2,
            "n_hash_matched_rollouts": len(hash_rewards_a),
            "reward_pearson_hash_matched": pr_hash,
            "reward_spearman_hash_matched": sp_hash,
            "reward_pearson_group_vectors_mean": (
                sum(group_pearson) / len(group_pearson) if group_pearson else None
            ),
            "advantage_pearson_hash_matched": pr_adv,
            "advantage_cosine_hash_matched": cos_adv,
            "advantage_sign_agreement_hash_matched": sgn,
            "advantage_sign_flips_hash_matched": flips,
            "top_rollout_agreement_rate": top_agree / n_groups if n_groups else None,
            "bottom_rollout_agreement_rate": bot_agree / n_groups if n_groups else None,
            "ranking_inversion_rate_group_vectors": inv_count / inv_total if inv_total else None,
            "diagnostic_effectively_equivalent": effectively_equiv,
            "note": "group_vector metrics compare different rollouts (same task_id only)",
        }
        pair_rows.append(row)
        reward_rows.append({k: v for k, v in row.items() if "reward" in k or "arm" in k or "n_hash" in k})
        adv_rows.append({k: v for k, v in row.items() if "advantage" in k or "arm" in k or "top_" in k or "sign" in k})

    payload = {"status": status, "pairs": pair_rows}
    (reports_dir / "counterfactual_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    if pair_rows:
        with open(reports_dir / "pairwise_signal_similarity.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(pair_rows[0].keys()))
            w.writeheader()
            w.writerows(pair_rows)
        with open(reports_dir / "counterfactual_reward_matrix.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(reward_rows[0].keys()))
            w.writeheader()
            w.writerows(reward_rows)
        with open(reports_dir / "counterfactual_advantage_matrix.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(adv_rows[0].keys()))
            w.writeheader()
            w.writerows(adv_rows)

    md = [
        "# Pairwise signal similarity",
        "",
        f"**Mode: {status['mode']}** — {status['reason']}",
        "",
    ]
    for r in pair_rows:
        md.append(f"## {r['arm_a']} vs {r['arm_b']}")
        md.append(f"- hash-matched rollouts: {r['n_hash_matched_rollouts']}")
        md.append(f"- advantage cosine (hash-matched): {r['advantage_cosine_hash_matched']}")
        md.append(f"- sign agreement: {r['advantage_sign_agreement_hash_matched']}")
        md.append(f"- effectively equivalent (heuristic): {r['diagnostic_effectively_equivalent']}")
        md.append("")
    (reports_dir / "PAIRWISE_SIGNAL_SIMILARITY.md").write_text("\n".join(md), encoding="utf-8")
    return payload
