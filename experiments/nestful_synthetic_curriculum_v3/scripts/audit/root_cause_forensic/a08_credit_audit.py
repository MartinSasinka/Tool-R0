"""A08 — Turn-level credit assignment audit from raw Round-1 train logs.

Recomputes G_t and per-position advantages with the ACTUAL trainer math
(group_stats.compute_group_stats + _turn_returns formula) and tests the
claims behind the previous audit's CREDIT_ASSIGNMENT_SUSPECT verdict:

  1. "good turn (r_t>=0.7) with negative advantage" — is it a bug, or the
     expected consequence of episode-outcome-dominated returns? We decompose:
     among good-turn-negative-advantage cases, what fraction sit in episodes
     whose EPISODE reward is below the group mean (outcome-driven, correct
     GRPO behavior)?
  2. "executable-wrong with positive advantage" — decompose by group type:
     in all-failure groups a positive advantage on the least-bad rollout is
     GRPO working as designed (relative ranking), not a credit bug.
  3. double-counting: v3_2_dense feeds process quality into BOTH r_seq
     (turn scores) and the episode reward; quantify how much of G_t at t=0
     is episode-reward vs turn-reward mass.
  4. position artifacts: stored flags + recomputed dead/alive parity.
"""
from __future__ import annotations

import statistics as st
from typing import Any, Dict

from common import ARMS, group_advantages, load_train_log, write_json

SUCCESS_R = 0.90  # v3_2_dense fully_correct band lower bound
GOOD_T = 0.7
BAD_T = 0.3


def main() -> Dict[str, Any]:
    per_arm: Dict[str, Any] = {}
    for arm in ARMS:
        _, groups = load_train_log(arm)
        good_neg = 0
        good_total = 0
        good_neg_outcome_driven = 0
        bad_pos = 0
        bad_total = 0
        ew_pos = ew_total = 0
        ew_pos_allfail = 0
        allfail_groups = 0
        mixed_groups = 0
        g0_ep_share = []
        stored_dead = recomputed_dead = 0
        for g in groups:
            ep = [float(x) for x in g["episode_rewards"]]
            tr = [[float(x) for x in s] for s in g["turn_rewards"]]
            _, gs = group_advantages(tr, ep)
            ep_mean = sum(ep) / len(ep)
            all_fail = all(r < SUCCESS_R for r in ep)
            if all_fail:
                allfail_groups += 1
            if g.get("group_mixed"):
                mixed_groups += 1
            if g.get("dead_group"):
                stored_dead += 1
            if gs.dead_corrected:
                recomputed_dead += 1
            for e_idx, (seq, R) in enumerate(zip(tr, ep)):
                advs = gs.advantages[e_idx]
                # G_0 mass decomposition (gamma=lambda=1): G_0 = sum(r_seq) + R
                tot = sum(seq) + R
                if tot > 0:
                    g0_ep_share.append(R / tot)
                is_ew = SUCCESS_R > R >= 0.35
                scalar_adv = advs[-1] if advs else 0.0
                if is_ew:
                    ew_total += 1
                    if scalar_adv > 0:
                        ew_pos += 1
                        if all_fail:
                            ew_pos_allfail += 1
                for t, r_t in enumerate(seq):
                    a_t = advs[t] if t < len(advs) else 0.0
                    if r_t >= GOOD_T:
                        good_total += 1
                        if a_t < 0:
                            good_neg += 1
                            if R < ep_mean:
                                good_neg_outcome_driven += 1
                    elif r_t <= BAD_T:
                        bad_total += 1
                        if a_t > 0:
                            bad_pos += 1
        per_arm[arm] = {
            "n_groups": len(groups),
            "good_turn_negative_adv_rate": good_neg / good_total if good_total else None,
            "good_turn_negative_adv_outcome_driven_frac": (
                good_neg_outcome_driven / good_neg if good_neg else None),
            "bad_turn_positive_adv_rate": bad_pos / bad_total if bad_total else None,
            "execwrong_positive_adv_rate": ew_pos / ew_total if ew_total else None,
            "execwrong_positive_adv_in_allfail_group_frac": (
                ew_pos_allfail / ew_pos if ew_pos else None),
            "allfail_group_rate": allfail_groups / len(groups) if groups else None,
            "mixed_group_rate": mixed_groups / len(groups) if groups else None,
            "G0_episode_reward_share_mean": st.mean(g0_ep_share) if g0_ep_share else None,
            "stored_dead_groups": stored_dead,
            "recomputed_dead_groups": recomputed_dead,
            "dead_flag_parity": stored_dead == recomputed_dead,
        }

    payload = {
        "per_arm": per_arm,
        "interpretation": {
            "good_turn_negative_advantage": (
                "With sparse-to-terminal credit (G_t includes the full episode reward at "
                "every t), any locally-valid turn inside a below-group-mean episode gets "
                "negative advantage BY DESIGN of outcome-based GRPO. It is only evidence "
                "of a credit bug if it happens in episodes at/above group mean."),
            "execwrong_positive_advantage": (
                "In all-failure groups GRPO must rank the least-bad rollout positive; "
                "positive advantage on executable-wrong is expected there."),
            "untestable": (
                "Matched-prefix / first-divergence credit audit is "
                "UNTESTABLE_WITH_CURRENT_LOGS: per-turn parsed calls and predicates are "
                "not persisted for training rollouts. Needs trajectory logging in the "
                "rollout worker (see NEXT_DECISION.md canary spec)."),
        },
    }
    write_json("a08_credit_audit.json", payload)
    return payload


if __name__ == "__main__":
    r = main()
    for arm, s in r["per_arm"].items():
        print(arm, {k: (round(v, 4) if isinstance(v, float) else v)
                    for k, v in s.items() if k != "n_groups"})
