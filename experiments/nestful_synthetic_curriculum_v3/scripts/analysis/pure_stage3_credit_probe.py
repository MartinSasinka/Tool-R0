#!/usr/bin/env python3
"""Post-hoc credit-assignment probe from pure-Stage3 train_log.jsonl.

Does NOT change training. Recomputes G_t and per-position advantages from
logged ``turn_rewards`` + ``episode_rewards`` using the same formulas as
``grpo_train._turn_returns`` and ``group_stats.compute_group_stats``.

Per-call predicates (name_ok / keys_ok / …) are not in train_log; those
fields are emitted as null with an explanatory note.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_V3 = os.path.normpath(os.path.join(_HERE, "..", ".."))
_MINIMAL = os.path.normpath(os.path.join(_V3, "..", "nestful_mtgrpo_minimal"))
sys.path.insert(0, _MINIMAL)

from grpo_train import _turn_returns  # noqa: E402
from group_stats import compute_group_stats  # noqa: E402

GAMMA = 1.0
LAMBDA = 1.0


def _load_groups(train_log: str, *, epoch_label: str) -> List[dict]:
    groups = []
    with open(train_log, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not r.get("episode_rewards") or not r.get("turn_rewards"):
                continue
            groups.append({**r, "epoch_label": epoch_label})
    return groups


def probe_group(rec: dict) -> List[dict]:
    ep_rewards = [float(x) for x in rec["episode_rewards"]]
    turn_rewards = [[float(x) for x in seq] for seq in rec["turn_rewards"]]
    ep_returns = [
        _turn_returns(seq, R, GAMMA, LAMBDA)
        for seq, R in zip(turn_rewards, ep_rewards)
    ]
    gstats = compute_group_stats(ep_returns, ep_rewards)
    rows = []
    for ci, (seq, R, Gs, advs) in enumerate(
            zip(turn_rewards, ep_rewards, ep_returns, gstats.advantages)):
        for t, (rt, Gt, adv) in enumerate(zip(seq, Gs, advs)):
            rows.append({
                "task_id": rec.get("task_id"),
                "epoch_label": rec.get("epoch_label"),
                "completion_id": ci,
                "trajectory_length": len(seq),
                "turn_index": t,
                "first_error_turn": rec.get("first_error_turn_mean"),
                "name_ok": None,
                "keys_ok": None,
                "val_frac": None,
                "exec_clean": None,
                "turn_score_r_t": rt,
                "episode_reward": R,
                "return_G_t": Gt,
                "position_mean": (gstats.position_means[t]
                                  if t < len(gstats.position_means) else None),
                "position_std": (gstats.position_stds[t]
                                 if t < len(gstats.position_stds) else None),
                "normalized_advantage": adv,
                "failure_class": None,
                "dead_group": bool(rec.get("dead_group")),
                "note": "predicates unavailable in train_log (post-hoc)",
            })
    return rows


def summarize(rows: List[dict]) -> dict:
    if not rows:
        return {"n": 0}
    # Heuristic: "local good" = r_t >= 0.7; "local bad" = r_t <= 0.3
    good_neg = sum(1 for r in rows if (r["turn_score_r_t"] or 0) >= 0.7
                   and (r["normalized_advantage"] or 0) < 0)
    bad_pos = sum(1 for r in rows if (r["turn_score_r_t"] or 0) <= 0.3
                  and (r["normalized_advantage"] or 0) > 0)
    g0 = [r for r in rows if r["turn_index"] == 0]
    # corr G_0 vs trajectory length
    if len(g0) >= 3:
        xs = [r["trajectory_length"] for r in g0]
        ys = [r["return_G_t"] for r in g0]
        mx = sum(xs) / len(xs)
        my = sum(ys) / len(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        denx = sum((x - mx) ** 2 for x in xs) ** 0.5
        deny = sum((y - my) ** 2 for y in ys) ** 0.5
        corr = (num / (denx * deny)) if denx and deny else None
    else:
        corr = None

    # variance of G_0 explained by episode reward (simple R^2 of linear fit)
    if len(g0) >= 3:
        xs = [r["episode_reward"] for r in g0]
        ys = [r["return_G_t"] for r in g0]
        mx = sum(xs) / len(xs)
        my = sum(ys) / len(ys)
        # slope
        den = sum((x - mx) ** 2 for x in xs)
        if den > 0:
            b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
            a = my - b * mx
            ss_tot = sum((y - my) ** 2 for y in ys)
            ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
            r2 = 1.0 - ss_res / ss_tot if ss_tot else None
        else:
            r2 = None
    else:
        r2 = None

    by_epoch: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_epoch[str(r.get("epoch_label"))].append(r)

    return {
        "n_rows": len(rows),
        "n_groups_approx": len({(r["task_id"], r["epoch_label"]) for r in rows}),
        "frac_local_good_negative_adv": good_neg / len(rows),
        "frac_local_bad_positive_adv": bad_pos / len(rows),
        "corr_G0_vs_traj_length": corr,
        "r2_G0_explained_by_episode_reward": r2,
        "by_epoch": {
            k: {
                "n": len(v),
                "mean_abs_adv": sum(abs(r["normalized_advantage"] or 0) for r in v) / len(v),
                "mean_r_t": sum((r["turn_score_r_t"] or 0) for r in v) / len(v),
            }
            for k, v in by_epoch.items()
        },
        "predicates_note": (
            "name_ok/keys_ok/val_frac/exec_clean/failure_class are null — "
            "train_log does not store per-completion reward diagnostics."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epoch1-log", required=True)
    ap.add_argument("--epoch2-log", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--n-groups", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    g1 = _load_groups(args.epoch1_log, epoch_label="epoch1")
    g2 = _load_groups(args.epoch2_log, epoch_label="epoch2")
    # Stratify: prefer mixed / non-dead, then fill
    def pick(groups: List[dict], n: int) -> List[dict]:
        mixed = [g for g in groups if g.get("group_mixed") and not g.get("dead_group")]
        other = [g for g in groups if g not in mixed]
        rng = random.Random(args.seed)
        rng.shuffle(mixed)
        rng.shuffle(other)
        return (mixed + other)[:n]

    half = max(1, args.n_groups // 2)
    selected = pick(g1, half) + pick(g2, args.n_groups - half)

    out_rows: List[dict] = []
    for rec in selected:
        out_rows.extend(probe_group(rec))

    os.makedirs(os.path.dirname(os.path.abspath(args.out_jsonl)) or ".", exist_ok=True)
    with open(args.out_jsonl, "w", encoding="utf-8") as fh:
        for r in out_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = summarize(out_rows)
    summary["n_groups_selected"] = len(selected)
    md = [
        "# Pure Stage 3 Credit Probe Summary",
        "",
        f"Groups sampled: {summary['n_groups_selected']} "
        f"(target {args.n_groups})",
        f"Turn rows: {summary['n_rows']}",
        "",
        f"- Local-good (r≥0.7) with **negative** advantage: "
        f"{summary['frac_local_good_negative_adv']:.4f}",
        f"- Local-bad (r≤0.3) with **positive** advantage: "
        f"{summary['frac_local_bad_positive_adv']:.4f}",
        f"- corr(G₀, traj_length): {summary['corr_G0_vs_traj_length']}",
        f"- R²(G₀ ~ episode_reward): {summary['r2_G0_explained_by_episode_reward']}",
        "",
        "## By epoch",
        "",
    ]
    for k, v in summary.get("by_epoch", {}).items():
        md.append(f"- **{k}**: n={v['n']} mean|adv|={v['mean_abs_adv']:.4f} "
                  f"mean r_t={v['mean_r_t']:.4f}")
    md += ["", summary["predicates_note"], ""]
    with open(args.out_md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))
    # also dump summary json beside md
    with open(args.out_md.replace(".md", ".json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(f"[credit-probe] wrote {len(out_rows)} rows -> {args.out_jsonl}")
    print(f"[credit-probe] summary -> {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
