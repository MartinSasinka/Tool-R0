"""A07 — Counterfactual reward re-score on STORED eval trajectories (non-causal).

Train-side rollout trajectories were NOT stored (only completion hashes), so a
true train-side counterfactual (same rollout group, different reward, different
advantages) is impossible with current logs. What IS possible: the n=500 eval
trajectories at temp 0 are stored with full turns for C0 and every arm. We
re-score each stored trajectory with every arm's INTENDED reward definition to
quantify how different the never-dispatched rewards would have been.

Limitations (stated in output): eval trajectories are 1 rollout/task at temp 0
(no GRPO groups -> no advantages); the classes are reconstructed from stored
`_traj` fields (parse_valid, clipped_any, per-turn fail_reason, official_win),
not by re-running the executor. This is a reward-definition comparison, not a
causal training comparison.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional

from common import (ARMS, INTENDED_EPSILON, INTENDED_TERMINAL_SCALARS,
                    c0_eval_dir, eval_dir, eval_ids_500, load_jsonl, write_json)

# v3_2_dense band midpoints per unified class, for a coarse A0 proxy ordering
V32_PROXY = {
    "official_success": 0.95,        # fully_correct band [0.90, 1.00]
    "executable_wrong_result": 0.70,  # executable_wrong_final [0.60, 0.80]
    "executable_partial": 0.45,      # partial/too_many mid
    "execution_failure": 0.25,       # wrong_tool/too_few mid
    "parse_or_no_call": 0.01,
}


def _unified_class(row: Dict[str, Any]) -> Optional[str]:
    """Reconstruct the unified 5-class terminal taxonomy from stored eval fields.
    Mirrors lib/reward_ablation_registry.unified_terminal_class."""
    tr = row.get("_traj") or {}
    if tr.get("official_win"):
        return "official_success"
    if not tr.get("parse_valid", True) or tr.get("clipped_any") or not tr.get("num_tool_calls"):
        return "parse_or_no_call"
    turns = tr.get("turns") or []
    call_turns = [t for t in turns if t.get("parsed_call")]
    if not call_turns:
        return "parse_or_no_call"
    ok = sum(1 for t in call_turns if not t.get("fail_reason"))
    frac = ok / len(call_turns)
    if tr.get("executable") and frac >= 0.999:
        return "executable_wrong_result"
    if frac > 0.0:
        return "executable_partial"
    return "execution_failure"


def _rows(arm: Optional[str]):
    d = c0_eval_dir() if arm is None else eval_dir(arm)
    return load_jsonl(d / "final_eval_trajectories.jsonl")


def main() -> Dict[str, Any]:
    ids = set(eval_ids_500())
    out: Dict[str, Any] = {"policies_compared": ["A0_proxy_v3_2_mid"] + ARMS[1:]}
    per_source = {}
    for label, arm in [("C0", None)] + [(a, a) for a in ARMS]:
        rows = [r for r in _rows(arm) if str(r.get("sample_id")) in ids]
        classes = Counter()
        rescored: Dict[str, List[float]] = {a: [] for a in ARMS[1:]}
        proxy_a0: List[float] = []
        skipped = 0
        for r in rows:
            cls = _unified_class(r)
            if cls is None:
                skipped += 1
                continue
            classes[cls] += 1
            proxy_a0.append(V32_PROXY[cls])
            for a in ARMS[1:]:
                rescored[a].append(INTENDED_TERMINAL_SCALARS[a][cls])
        n = sum(classes.values())
        per_source[label] = {
            "n_rescored": n,
            "skipped": skipped,
            "class_distribution": dict(classes),
            "mean_reward_by_policy": {
                "A0_proxy_v3_2_mid": sum(proxy_a0) / n if n else None,
                **{a: sum(v) / n if n else None for a, v in rescored.items()},
            },
        }
        # separation between success and executable_wrong (the design target
        # of the ablation): gap per policy
        gaps = {"A0_proxy_v3_2_mid": V32_PROXY["official_success"] - V32_PROXY["executable_wrong_result"]}
        for a in ARMS[1:]:
            s = INTENDED_TERMINAL_SCALARS[a]
            gaps[a] = s["official_success"] - s["executable_wrong_result"]
        per_source[label]["success_vs_execwrong_gap_by_policy"] = gaps

    out["per_source"] = per_source
    out["caveats"] = [
        "NON-CAUSAL: eval trajectories are temp-0 single rollouts; no GRPO groups, "
        "no advantages can be derived.",
        "Train-side counterfactual is UNTESTABLE_WITH_CURRENT_LOGS: rollout "
        "trajectories were not persisted during Round-1 training (only hashes).",
        "Classes reconstructed from stored _traj predicates, not re-execution.",
    ]
    write_json("a07_counterfactual_rescore.json", out)
    return out


if __name__ == "__main__":
    r = main()
    for src, d in r["per_source"].items():
        print(src, d["class_distribution"], {k: round(v, 4) for k, v in
                                             d["mean_reward_by_policy"].items() if v})
