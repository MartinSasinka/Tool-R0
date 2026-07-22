#!/usr/bin/env python3
"""Offline (no-GPU) forensic analysis of the pure-Stage3 2-epoch run.

Consumes only artifacts that exist locally after the RunPod loss:
  - epoch_{1,2}/train/train_log.jsonl   (8-rollout groups, dense rewards)
  - data/training_ready_v5/filtered/stage3_train_ready.jsonl (motif/gold)
  - eval/C0_dev/final_eval_trajectories.jsonl (baseline, 200 dev tasks)
  - checkpoints/{S3_E1,S3_E2}/adapter_model.safetensors (LoRA deltas)

It CANNOT do the paired C0/E1/E2 NESTFUL analysis: the E1 dev eval OOM-crashed
and E2 / test evals never ran (see run FAILED marker). Those sections are
reported as BLOCKED with the reason.

Emits report files under reports/pure_stage3_offline_analysis/.
Replicates the trainer's exact returns/advantage math:
  G_t = sum_{k>=t} gamma^(k-t) r_k + lambda * gamma^(T-t+1) * R_episode
  adv[e][t] = (G-pos_mean_t)/(pos_std_t+1e-8) if pos_std_t>eps else 0
"""
from __future__ import annotations

import glob
import json
import math
import os
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
V3 = os.path.normpath(os.path.join(HERE, "..", ".."))
RUN = os.path.join(
    V3, "outputs", "runs", "pure_stage3_2ep_20260719_221918"
)
DATASET = os.path.join(
    V3, "data", "training_ready_v5", "filtered", "stage3_train_ready.jsonl"
)
OUT = os.path.join(V3, "reports", "pure_stage3_offline_analysis")
os.makedirs(OUT, exist_ok=True)

GAMMA = 1.0
LAMBDA = 1.0
EPS_STD = 1e-6
WIN = 0.99


# ───────────────────────── math replicated from trainer ─────────────────────
def turn_returns(r_seq, R, gamma=GAMMA, lam=LAMBDA):
    T = len(r_seq) - 1
    out = []
    for t in range(len(r_seq)):
        disc = 0.0
        for k in range(t, len(r_seq)):
            disc += (gamma ** (k - t)) * r_seq[k]
        disc += lam * (gamma ** (T - t + 1)) * R
        out.append(disc)
    return out


def position_advantages(ep_returns):
    """Position-wise (per turn index) standardized advantages."""
    max_len = max((len(g) for g in ep_returns), default=0)
    pos_mean, pos_std = [], []
    for t in range(max_len):
        vals = [g[t] for g in ep_returns if t < len(g)]
        m = sum(vals) / len(vals) if vals else 0.0
        if len(vals) >= 2:
            s = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5
        else:
            s = 0.0
        pos_mean.append(m)
        pos_std.append(s)
    advs = []
    for g in ep_returns:
        row = []
        for t, val in enumerate(g):
            s = pos_std[t] if t < len(pos_std) else 0.0
            row.append((val - pos_mean[t]) / (s + 1e-8) if s > EPS_STD else 0.0)
        advs.append(row)
    return advs, pos_mean, pos_std


def spearman(xs, ys):
    n = len(xs)
    if n < 3:
        return None

    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(xs), rank(ys)
    return pearson(rx, ry)


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx > 0 and dy > 0 else None


# ───────────────────────── loaders ──────────────────────────────────────────
def load_train_log(path, label):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "episode_rewards" not in r or "turn_rewards" not in r:
                continue
            r["_epoch"] = label
            rows.append(r)
    return rows


def load_dataset_meta(path):
    meta = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            sid = d.get("sample_id")
            gold = d.get("gold_calls") or []
            meta[sid] = {
                "motif": d.get("motif_type"),
                "num_calls": d.get("num_calls"),
                "gold_first_tool": gold[0]["name"] if gold else None,
            }
    return meta


# ───────────────────────── reward alignment (sec 4 + 5) ─────────────────────
def credit_schemes(turn_rewards, R):
    """Return dict scheme -> per-episode per-turn returns."""
    T = len(turn_rewards) - 1
    A0 = turn_returns(turn_rewards, R, 1.0, 1.0)          # current
    A1 = turn_returns(turn_rewards, 0.0, 1.0, 1.0)        # no episode reward
    A2 = list(turn_rewards)                               # local r_t only
    # A3: local + terminal outcome band (win/loss) discounted to turn
    outcome = 1.0 if R >= WIN else 0.0
    beta = 0.5
    A3 = [turn_rewards[t] + beta * (1.0 ** (T - t)) * outcome
          for t in range(len(turn_rewards))]
    return {"A0_current": A0, "A1_no_episode": A1,
            "A2_local": A2, "A3_local_plus_outcome": A3}


def analyze_reward_alignment(groups, meta):
    scheme_stats = {k: {"dead_pos": 0, "total_pos": 0, "abs_adv": 0.0,
                        "good_neg": 0, "bad_pos": 0, "local_pairs": 0}
                    for k in ["A0_current", "A1_no_episode",
                              "A2_local", "A3_local_plus_outcome"]}
    g0_len_x, g0_len_y = [], []
    r2_num = []
    too_few_gap = []      # (too_few_reward, full_reward) within mixed groups
    win_vs_loss_reward = {"win": [], "loss": []}
    pair_ordering = {"correct": 0, "reversed": 0, "tie": 0}
    per_epoch = defaultdict(lambda: {"groups": 0, "mean_reward": [],
                                     "dead": 0, "mixed": 0})

    for g in groups:
        ep = g["_epoch"]
        ep_rewards = [float(x) for x in g["episode_rewards"]]
        turn_r = [[float(x) for x in seq] for seq in g["turn_rewards"]]
        pred_calls = g.get("predicted_num_calls") or []
        gold_calls = g.get("gold_num_calls")
        per_epoch[ep]["groups"] += 1
        per_epoch[ep]["mean_reward"].append(sum(ep_rewards) / len(ep_rewards))
        if g.get("dead_group"):
            per_epoch[ep]["dead"] += 1
        if g.get("group_mixed"):
            per_epoch[ep]["mixed"] += 1

        # win vs loss reward
        for R in ep_rewards:
            win_vs_loss_reward["win" if R >= WIN else "loss"].append(R)

        # pairwise ordering: within group, does higher reward == more correct?
        # proxy correctness = (pred_calls == gold) and reward; we use reward
        # rank vs #correct-calls proxy = pred matches gold count.
        if gold_calls and len(pred_calls) == len(ep_rewards):
            for i in range(len(ep_rewards)):
                for j in range(i + 1, len(ep_rewards)):
                    ci = 1 if pred_calls[i] == gold_calls else 0
                    cj = 1 if pred_calls[j] == gold_calls else 0
                    if ci == cj:
                        continue
                    ri, rj = ep_rewards[i], ep_rewards[j]
                    better_reward = i if ri > rj else (j if rj > ri else None)
                    better_correct = i if ci > cj else j
                    if better_reward is None:
                        pair_ordering["tie"] += 1
                    elif better_reward == better_correct:
                        pair_ordering["correct"] += 1
                    else:
                        pair_ordering["reversed"] += 1

        # too-few gap
        if gold_calls and len(pred_calls) == len(ep_rewards):
            few = [ep_rewards[i] for i in range(len(pred_calls))
                   if pred_calls[i] < gold_calls]
            full = [ep_rewards[i] for i in range(len(pred_calls))
                    if pred_calls[i] >= gold_calls]
            if few and full:
                too_few_gap.append(
                    (sum(few) / len(few), sum(full) / len(full)))

        # credit schemes
        schemes = {k: [] for k in scheme_stats}
        for seq, R in zip(turn_r, ep_rewards):
            cs = credit_schemes(seq, R)
            for k in schemes:
                schemes[k].append(cs[k])
        for k, ep_returns in schemes.items():
            advs, pmean, pstd = position_advantages(ep_returns)
            for t, s in enumerate(pstd):
                scheme_stats[k]["total_pos"] += 1
                if s <= EPS_STD:
                    scheme_stats[k]["dead_pos"] += 1
            for e, row in enumerate(advs):
                for t, a in enumerate(row):
                    scheme_stats[k]["abs_adv"] += abs(a)
                    rt = turn_r[e][t] if t < len(turn_r[e]) else 0.0
                    if rt >= 0.7 and a < 0:
                        scheme_stats[k]["good_neg"] += 1
                    if rt <= 0.3 and a > 0:
                        scheme_stats[k]["bad_pos"] += 1

        # horizon bias (A0 G_0 vs traj length)
        A0 = schemes["A0_current"]
        for seq_ret, seq in zip(A0, turn_r):
            g0_len_x.append(len(seq))
            g0_len_y.append(seq_ret[0])
        # R^2 of G_0 explained by episode reward (per group linear proxy)
        xs = ep_rewards
        ys = [ret[0] for ret in A0]
        p = pearson(xs, ys)
        if p is not None:
            r2_num.append(p * p)

    # finalize
    for k, s in scheme_stats.items():
        s["dead_pos_frac"] = s["dead_pos"] / s["total_pos"] if s["total_pos"] else None
        tot_adv = s["total_pos"]  # approx denom for abs adv per position group
    horizon = pearson(g0_len_x, g0_len_y)
    horizon_sp = spearman(g0_len_x, g0_len_y)
    return {
        "scheme_stats": scheme_stats,
        "horizon_pearson_G0_vs_len": horizon,
        "horizon_spearman_G0_vs_len": horizon_sp,
        "mean_r2_G0_by_episode_reward": (sum(r2_num) / len(r2_num)
                                         if r2_num else None),
        "pair_ordering_pred_call_count_proxy": pair_ordering,
        "too_few_vs_full_reward_gap": {
            "n_groups": len(too_few_gap),
            "mean_too_few_reward": (sum(a for a, _ in too_few_gap) / len(too_few_gap)
                                    if too_few_gap else None),
            "mean_full_reward": (sum(b for _, b in too_few_gap) / len(too_few_gap)
                                 if too_few_gap else None),
        },
        "mean_reward_win": (sum(win_vs_loss_reward["win"]) / len(win_vs_loss_reward["win"])
                            if win_vs_loss_reward["win"] else None),
        "mean_reward_loss": (sum(win_vs_loss_reward["loss"]) / len(win_vs_loss_reward["loss"])
                             if win_vs_loss_reward["loss"] else None),
        "n_win_rollouts": len(win_vs_loss_reward["win"]),
        "n_loss_rollouts": len(win_vs_loss_reward["loss"]),
        "per_epoch": {k: {"groups": v["groups"],
                          "mean_reward": sum(v["mean_reward"]) / len(v["mean_reward"]),
                          "dead": v["dead"], "mixed": v["mixed"]}
                      for k, v in per_epoch.items()},
    }


# ───────────────────────── train failure shift (sec 2/3 proxy) ──────────────
FAIL_KEYS = ["wrong_tool_count", "wrong_arg_count", "invalid_ref_count",
             "premature_final_count", "too_few_calls_count",
             "parse_error_count", "no_tool_call_count", "execfail_total"]


def analyze_failure_shift(groups, meta):
    by_epoch = defaultdict(lambda: {k: 0 for k in FAIL_KEYS})
    by_epoch_n = defaultdict(int)          # total rollouts
    win_by_epoch = defaultdict(list)
    firsterr_by_epoch = defaultdict(list)
    bucket = defaultdict(lambda: defaultdict(lambda: {"win": [], "n": 0}))
    motif = defaultdict(lambda: defaultdict(lambda: {"win": [], "n": 0}))
    for g in groups:
        ep = g["_epoch"]
        n = len(g["episode_rewards"])
        by_epoch_n[ep] += n
        for k in FAIL_KEYS:
            by_epoch[ep][k] += int(g.get(k, 0) or 0)
        win_by_epoch[ep].append(float(g.get("win_rate", 0.0) or 0.0))
        if g.get("first_error_turn_mean") is not None:
            firsterr_by_epoch[ep].append(float(g["first_error_turn_mean"]))
        m = meta.get(g.get("task_id"), {})
        gc = g.get("gold_num_calls") or m.get("num_calls")
        mo = m.get("motif")
        wr = float(g.get("win_rate", 0.0) or 0.0)
        if gc is not None:
            bucket[ep][gc]["win"].append(wr)
            bucket[ep][gc]["n"] += 1
        if mo is not None:
            motif[ep][mo]["win"].append(wr)
            motif[ep][mo]["n"] += 1

    def rate_table(d, dn):
        return {ep: {k: (d[ep][k] / dn[ep]) for k in FAIL_KEYS}
                for ep in d}

    return {
        "per_rollout_failure_rate": rate_table(by_epoch, by_epoch_n),
        "abs_failure_counts": {ep: dict(by_epoch[ep]) for ep in by_epoch},
        "n_rollouts": dict(by_epoch_n),
        "mean_group_win_rate": {ep: sum(v) / len(v) for ep, v in win_by_epoch.items()},
        "mean_first_error_turn": {ep: sum(v) / len(v) for ep, v in firsterr_by_epoch.items()},
        "win_by_gold_calls": {
            ep: {gc: {"mean_win": sum(x["win"]) / len(x["win"]), "n": x["n"]}
                 for gc, x in sorted(bucket[ep].items())}
            for ep in bucket},
        "win_by_motif": {
            ep: {mo: {"mean_win": sum(x["win"]) / len(x["win"]), "n": x["n"]}
                 for mo, x in motif[ep].items()}
            for ep in motif},
    }


# ───────────────────────── checkpoint delta (sec 7 offline) ─────────────────
def analyze_checkpoints():
    try:
        from safetensors import safe_open
    except Exception as e:  # noqa
        return {"error": f"safetensors unavailable: {e}"}
    import torch
    e1 = os.path.join(RUN, "checkpoints", "S3_E1", "adapter_model.safetensors")
    e2 = os.path.join(RUN, "checkpoints", "S3_E2", "adapter_model.safetensors")
    if not (os.path.exists(e1) and os.path.exists(e2)):
        return {"error": "adapter files missing"}

    def load(path):
        d = {}
        with safe_open(path, framework="pt") as f:
            for k in f.keys():
                d[k] = f.get_tensor(k).float()
        return d

    a1, a2 = load(e1), load(e2)
    keys = sorted(set(a1) & set(a2))
    n1 = math.sqrt(sum(float(a1[k].pow(2).sum()) for k in keys))
    n2 = math.sqrt(sum(float(a2[k].pow(2).sum()) for k in keys))
    dnorm = math.sqrt(sum(float((a2[k] - a1[k]).pow(2).sum()) for k in keys))
    # cosine of flattened deltas (E1 = delta from base since base=0 for lora_B)
    dot = sum(float((a1[k] * a2[k]).sum()) for k in keys)
    cos = dot / (n1 * n2) if n1 > 0 and n2 > 0 else None
    # relative movement E1->E2 vs E1 magnitude
    rel = dnorm / n1 if n1 > 0 else None
    return {
        "adapter_norm_E1": n1,
        "adapter_norm_E2": n2,
        "delta_norm_E1_to_E2": dnorm,
        "cosine_E1_E2": cos,
        "rel_move_E1_to_E2_over_E1": rel,
        "n_tensors": len(keys),
        "note": "C0 baseline adapter norm = 0 (no LoRA). E1/E2 norms are deltas from base.",
    }


# ───────────────────────── C0 dev first-error (sec 3 baseline) ──────────────
def classify_traj(rec):
    t = rec.get("_traj", {})
    win = int(rec.get("_traj", {}).get("official_win", 0)) if False else None
    # official_win lives at top-level _traj
    win = t.get("official_win")
    stop = t.get("stop_reason")
    parse_valid = t.get("parse_valid")
    executable = t.get("executable")
    turns = t.get("turns", [])
    num_calls = t.get("num_tool_calls")
    gold_calls = rec.get("num_gold_calls")
    # first error turn / type
    first_err_turn = None
    first_err_type = None
    for tn in turns:
        if tn.get("fail_reason"):
            first_err_turn = tn.get("turn_idx")
            first_err_type = tn.get("fail_reason")
            break
    if win == 1 or win is True:
        cls = "success"
    elif not parse_valid:
        cls = "parse_or_format"
    elif num_calls == 0:
        cls = "no_tool_call"
    elif gold_calls and num_calls is not None and num_calls < gold_calls:
        cls = "too_few_calls"
    elif executable is False:
        cls = "non_executable"
    elif executable and (win == 0 or win is False):
        cls = "executable_wrong_result"
    else:
        cls = "other_non_win"
    return {
        "task_id": rec.get("sample_id"),
        "gold_calls": gold_calls,
        "win": win,
        "stop_reason": stop,
        "parse_valid": parse_valid,
        "executable": executable,
        "num_calls": num_calls,
        "first_error_turn": first_err_turn,
        "first_error_type": first_err_type,
        "class": cls,
    }


def analyze_c0_dev():
    p = os.path.join(RUN, "eval", "C0_dev", "final_eval_trajectories.jsonl")
    if not os.path.exists(p):
        return {"error": "C0_dev trajectories missing"}
    rows = []
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(classify_traj(json.loads(line)))
    n = len(rows)
    cls_counts = Counter(r["class"] for r in rows)
    firsterr = Counter(r["first_error_type"] for r in rows if r["first_error_type"])
    by_bucket = defaultdict(lambda: {"n": 0, "win": 0})
    for r in rows:
        gc = r["gold_calls"]
        by_bucket[gc]["n"] += 1
        if r["win"] in (1, True):
            by_bucket[gc]["win"] += 1
    return {
        "n": n,
        "win_rate": sum(1 for r in rows if r["win"] in (1, True)) / n if n else None,
        "class_distribution": dict(cls_counts),
        "first_error_type_distribution": dict(firsterr),
        "win_by_gold_calls": {gc: {"n": v["n"],
                                   "win_rate": v["win"] / v["n"] if v["n"] else None}
                              for gc, v in sorted(by_bucket.items(),
                                                  key=lambda kv: (kv[0] is None, kv[0]))},
        "rows": rows,
    }


# ───────────────────────── main ─────────────────────────────────────────────
def main():
    meta = load_dataset_meta(DATASET)
    g1 = load_train_log(os.path.join(RUN, "epoch_1", "train", "train_log.jsonl"), "epoch1")
    g2 = load_train_log(os.path.join(RUN, "epoch_2", "train", "train_log.jsonl"), "epoch2")
    groups = g1 + g2

    reward = analyze_reward_alignment(groups, meta)
    shift = analyze_failure_shift(groups, meta)
    ckpt = analyze_checkpoints()
    c0 = analyze_c0_dev()
    c0_rows = c0.pop("rows", []) if isinstance(c0, dict) else []

    result = {
        "meta": {
            "n_groups_epoch1": len(g1),
            "n_groups_epoch2": len(g2),
            "dataset_tasks": len(meta),
            "run_dir": RUN,
        },
        "reward_alignment": reward,
        "failure_shift": shift,
        "checkpoint_delta": ckpt,
        "c0_dev_baseline": c0,
    }
    with open(os.path.join(OUT, "analysis.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    with open(os.path.join(OUT, "c0_dev_task_level.jsonl"), "w", encoding="utf-8") as fh:
        for r in c0_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
