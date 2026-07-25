"""A05 — Evaluation audit on the fixed NESTFUL n=500 subset.

Verifies from raw eval artifacts:
  - 500/500 paired IDs across C0 and all arms;
  - official win-rate recomputation from per-task results;
  - stored paired_vs_c0.json gained/lost counts;
  - A0 vs A4 direct pairing (claimed 21/21);
  - per-call-count buckets (2 vs 3 vs 4+);
  - shared-C0 eval set anomaly (manifest says 500-row subset, output has 1861 rows);
  - behavioral similarity A0 vs A4 (identical predicted call sequences);
  - failure taxonomy decomposition of gained/lost tasks.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

from common import (ARMS, R1_ROOT, c0_eval_dir, eval_dir, eval_ids_500,
                    load_json, load_jsonl, write_json)


def _win_map_from_task_results(arm: str) -> Dict[str, Dict[str, Any]]:
    rows = load_jsonl(eval_dir(arm) / "task_results.jsonl")
    return {str(r["task_id"]): r for r in rows}


def _c0_maps():
    rows = load_jsonl(c0_eval_dir() / "final_eval_trajectories.jsonl")
    win = {}
    extra = {}
    for r in rows:
        sid = str(r.get("sample_id"))
        tr = r.get("_traj") or {}
        win[sid] = 1.0 if tr.get("official_win") else 0.0
        extra[sid] = {
            "num_gold_calls": r.get("num_gold_calls"),
            "num_tool_calls": tr.get("num_tool_calls"),
            "stop_reason": tr.get("stop_reason"),
            "final_answer_pass": r.get("final_answer_pass"),
            "alternative_valid_solution_pass": r.get("alternative_valid_solution_pass"),
            "solution_equivalent_pass": r.get("solution_equivalent_pass"),
        }
    return win, extra


def _bucket(n: Optional[int]) -> str:
    if n is None:
        return "?"
    if n <= 2:
        return "2"
    if n == 3:
        return "3"
    return "4+"


def _preds_map(dir_path) -> Dict[str, Any]:
    p = dir_path / "final_eval_predictions.partial.jsonl"
    if not p.is_file():
        return {}
    return {str(r["sample_id"]): r.get("predicted_calls")
            for r in load_jsonl(p)}


def main() -> Dict[str, Any]:
    ids = eval_ids_500()
    ids_set = set(ids)
    c0_win, c0_extra = _c0_maps()

    c0_cov = sum(1 for i in ids if i in c0_win)
    c0_win_500 = sum(c0_win[i] for i in ids if i in c0_win) / max(1, c0_cov)

    # C0 eval-set anomaly
    c0_manifest = load_json(c0_eval_dir() / "eval_manifest.json")
    c0_metrics = load_json(c0_eval_dir() / "metrics_official.json")
    subset_file = (R1_ROOT / "shared_C0_eval_500" / "shared_C0_eval_500"
                   / "eval_subset_500.jsonl")
    subset_rows = len(load_jsonl(subset_file)) if subset_file.is_file() else None

    per_arm = {}
    win_maps = {}
    for arm in ARMS:
        tr = _win_map_from_task_results(arm)
        win_maps[arm] = {k: float(v["win"]) for k, v in tr.items()}
        metrics = load_json(eval_dir(arm) / "metrics_official.json")
        paired_stored = load_json(eval_dir(arm) / "paired_vs_c0.json")
        common = [i for i in ids if i in tr and i in c0_win]
        wr = sum(win_maps[arm][i] for i in common) / len(common)
        gained = [i for i in common if win_maps[arm][i] > c0_win[i]]
        lost = [i for i in common if win_maps[arm][i] < c0_win[i]]
        # bucket win rates
        buckets = defaultdict(lambda: {"n": 0, "arm_wins": 0.0, "c0_wins": 0.0})
        tax = Counter()
        for i in common:
            b = _bucket(tr[i].get("num_gold_calls"))
            buckets[b]["n"] += 1
            buckets[b]["arm_wins"] += win_maps[arm][i]
            buckets[b]["c0_wins"] += c0_win[i]
            tax[tr[i].get("taxonomy")] += 1
        bucket_out = {b: {"n": v["n"],
                          "arm_win_rate": v["arm_wins"] / v["n"],
                          "c0_win_rate": v["c0_wins"] / v["n"]}
                      for b, v in sorted(buckets.items())}
        per_arm[arm] = {
            "task_results_rows": len(tr),
            "ids_match_500": set(tr.keys()) == ids_set,
            "metrics_official_win_rate": metrics.get("win_rate"),
            "recomputed_win_rate_500": wr,
            "win_rate_matches_metrics": abs(wr - float(metrics.get("win_rate", -1))) < 5e-4,
            "n_paired_with_c0": len(common),
            "recomputed_gained": len(gained),
            "recomputed_lost": len(lost),
            "stored_paired_gained": paired_stored.get("n_gained"),
            "stored_paired_lost": paired_stored.get("n_regressed"),
            "paired_matches_stored": (len(gained) == paired_stored.get("n_gained")
                                      and len(lost) == paired_stored.get("n_regressed")),
            "win_by_gold_call_bucket": bucket_out,
            "taxonomy_counts": dict(tax),
        }

    # A0 vs A4 direct pairing + behavior diff
    a0, a4 = win_maps["A0_R0_CURRENT"], win_maps["A4_GATED_VERIFIABLE"]
    common04 = [i for i in ids if i in a0 and i in a4]
    a0_gain = [i for i in common04 if a0[i] > a4[i]]
    a4_gain = [i for i in common04 if a4[i] > a0[i]]

    preds_a0 = _preds_map(eval_dir("A0_R0_CURRENT"))
    preds_a4 = _preds_map(eval_dir("A4_GATED_VERIFIABLE"))
    preds_c0 = _preds_map(c0_eval_dir())
    same_calls_04 = same_calls_a0c0 = n_cmp = 0
    for i in common04:
        pa, pb = preds_a0.get(i), preds_a4.get(i)
        if pa is None or pb is None:
            continue
        n_cmp += 1
        if json.dumps(pa, sort_keys=True) == json.dumps(pb, sort_keys=True):
            same_calls_04 += 1
        pc = preds_c0.get(i)
        if pc is not None and json.dumps(pa, sort_keys=True) == json.dumps(pc, sort_keys=True):
            same_calls_a0c0 += 1

    # C0 taxonomy-ish decomposition for the 500 (undercalling etc.)
    c0_under = sum(1 for i in ids
                   if i in c0_extra and (c0_extra[i]["num_tool_calls"] or 0)
                   < (c0_extra[i]["num_gold_calls"] or 0))
    c0_shorter_valid = sum(1 for i in ids if i in c0_extra
                           and c0_extra[i]["alternative_valid_solution_pass"])

    payload = {
        "eval_ids": {"n_ids": len(ids), "unique": len(ids_set) == len(ids)},
        "c0": {
            "eval_rows_total": len(c0_win),
            "manifest_eval_set": c0_manifest.get("eval_set"),
            "manifest_checkpoint": c0_manifest.get("checkpoint"),
            "metrics_official_num_examples": c0_metrics.get("num_examples"),
            "metrics_official_win_rate": c0_metrics.get("win_rate"),
            "local_subset_file_rows": subset_rows,
            "coverage_of_500_ids": c0_cov,
            "win_rate_on_500": c0_win_500,
            "undercalling_rate_500": c0_under / len(ids),
            "alternative_valid_solution_pass_500": c0_shorter_valid,
        },
        "per_arm": per_arm,
        "a0_vs_a4": {
            "n_common": len(common04),
            "a0_only_wins": len(a0_gain),
            "a4_only_wins": len(a4_gain),
            "identical_predicted_call_sequences": same_calls_04,
            "n_pred_compared": n_cmp,
            "identical_call_fraction": same_calls_04 / n_cmp if n_cmp else None,
            "a0_identical_to_c0_calls": same_calls_a0c0,
        },
        "verdict": {
            "pairing_500_ok": all(per_arm[a]["ids_match_500"] for a in ARMS),
            "win_rates_reproduce": all(per_arm[a]["win_rate_matches_metrics"] for a in ARMS),
            "paired_counts_reproduce": all(per_arm[a]["paired_matches_stored"] for a in ARMS),
            "c0_eval_set_anomaly": (subset_rows != 500 or len(c0_win) != 500),
        },
    }
    write_json("a05_eval_audit.json", payload)
    return payload


if __name__ == "__main__":
    r = main()
    print("C0 on 500:", r["c0"]["win_rate_on_500"], "cov", r["c0"]["coverage_of_500_ids"])
    for arm, s in r["per_arm"].items():
        print(arm, s["recomputed_win_rate_500"], "gained", s["recomputed_gained"],
              "lost", s["recomputed_lost"], "stored ok", s["paired_matches_stored"])
    print("A0 vs A4:", r["a0_vs_a4"])
    print(r["verdict"])
