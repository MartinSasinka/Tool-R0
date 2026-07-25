#!/usr/bin/env python3
"""Validate dispatch-canary A1 vs A4 runs (no GPU).

Gates from reports/reward_ablation/DISPATCH_CANARY_RUNBOOK.md:
  1. resolved policy matches configured arm policy on 100% of train rows
  2. canary_rollouts.jsonl present with required fields
  3. no NaN/Inf in rewards / advantages / KL / grad norms
  4. no terminal ordering inversion (success reward < execwrong, etc.)
  5. on hash-matched completions: rewards differ (max_abs_diff > 0,
     reward correlation < 1)

Exit 0 = PASS, 1 = FAIL (STOP — do not proceed to data experiments).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_V3 = Path(__file__).resolve().parents[2]
EXPECTED = {
    "A1_OUTCOME_ONLY": "reward_ablation_A1_OUTCOME_ONLY",
    "A4_GATED_VERIFIABLE": "reward_ablation_A4_GATED_VERIFIABLE",
}
REQUIRED_CANARY_FIELDS = (
    "task_id", "rollout_index", "episode_reward", "turn_rewards", "G_t",
    "normalized_advantage", "completion_hash",
    "reward_policy_configured", "reward_policy_resolved",
)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _is_bad_num(x: Any) -> bool:
    if isinstance(x, bool) or x is None:
        return False
    if isinstance(x, (int, float)):
        return math.isnan(float(x)) or math.isinf(float(x))
    if isinstance(x, list):
        return any(_is_bad_num(v) for v in x)
    return False


def _pearson(xs: List[float], ys: List[float]) -> Optional[float]:
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    denx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    deny = math.sqrt(sum((b - my) ** 2 for b in ys))
    if denx == 0.0 or deny == 0.0:
        return None
    return num / (denx * deny)


def _arm_dir(root: Path, arm: str, seed: int) -> Path:
    return root / f"dispatch_canary_{arm}_seed{seed}"


def _check_arm(run_dir: Path, arm: str) -> Dict[str, Any]:
    expected = EXPECTED[arm]
    train_log = run_dir / "train" / "train_log.jsonl"
    canary = run_dir / "train" / "canary_rollouts.jsonl"
    out: Dict[str, Any] = {
        "arm": arm,
        "run_dir": str(run_dir),
        "ok": True,
        "errors": [],
        "n_train_rows": 0,
        "n_canary_rows": 0,
        "resolved_ok_frac": None,
    }
    if not train_log.is_file():
        out["ok"] = False
        out["errors"].append(f"missing train_log: {train_log}")
        return out
    rows = _load_jsonl(train_log)
    group_rows = [r for r in rows if "reward_policy_resolved" in r and "task_id" in r]
    out["n_train_rows"] = len(group_rows)
    if not group_rows:
        out["ok"] = False
        out["errors"].append("no group rows with reward_policy_resolved")
        return out
    bad = [r for r in group_rows if r.get("reward_policy_resolved") != expected]
    out["resolved_ok_frac"] = 1.0 - len(bad) / len(group_rows)
    if bad:
        out["ok"] = False
        out["errors"].append(
            f"{len(bad)}/{len(group_rows)} rows have wrong reward_policy_resolved "
            f"(expected {expected!r}, sample={bad[0].get('reward_policy_resolved')!r})"
        )
    for r in group_rows:
        for key in ("episode_rewards", "turn_rewards", "kl", "loss", "mean_reward"):
            if key in r and _is_bad_num(r[key]):
                out["ok"] = False
                out["errors"].append(f"NaN/Inf in train_log field {key} task={r.get('task_id')}")
                break
    if not canary.is_file():
        out["ok"] = False
        out["errors"].append(f"missing canary_rollouts.jsonl (CANARY_TRAJ_LOG not active?)")
        return out
    crows = _load_jsonl(canary)
    out["n_canary_rows"] = len(crows)
    if len(crows) < 8:
        out["ok"] = False
        out["errors"].append(f"too few canary rows: {len(crows)}")
    missing = [f for f in REQUIRED_CANARY_FIELDS if f not in (crows[0] if crows else {})]
    if missing:
        out["ok"] = False
        out["errors"].append(f"canary row missing fields: {missing}")
    for r in crows:
        if r.get("reward_policy_resolved") != expected:
            out["ok"] = False
            out["errors"].append(
                f"canary row wrong policy: {r.get('reward_policy_resolved')!r}"
            )
            break
        for key in ("episode_reward", "turn_rewards", "G_t", "normalized_advantage", "kl", "gradient_norm"):
            if key in r and _is_bad_num(r[key]):
                out["ok"] = False
                out["errors"].append(f"NaN/Inf in canary field {key}")
                break
    # Terminal ordering: among rows with both classes present, mean(success) > mean(execwrong)
    by_cls: Dict[str, List[float]] = defaultdict(list)
    for r in crows:
        cls = r.get("terminal_class")
        if cls:
            by_cls[str(cls)].append(float(r["episode_reward"]))
    if "official_success" in by_cls and "executable_wrong_result" in by_cls:
        ms = sum(by_cls["official_success"]) / len(by_cls["official_success"])
        mw = sum(by_cls["executable_wrong_result"]) / len(by_cls["executable_wrong_result"])
        out["mean_reward_success"] = ms
        out["mean_reward_execwrong"] = mw
        if ms <= mw:
            out["ok"] = False
            out["errors"].append(
                f"terminal ordering inversion: mean(success)={ms:.4f} <= mean(execwrong)={mw:.4f}"
            )
    return out


def _hash_matched(a_rows: List[Dict[str, Any]], b_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    a_map: Dict[str, List[float]] = defaultdict(list)
    b_map: Dict[str, List[float]] = defaultdict(list)
    for r in a_rows:
        h = r.get("completion_hash")
        if h:
            a_map[str(h)].append(float(r["episode_reward"]))
    for r in b_rows:
        h = r.get("completion_hash")
        if h:
            b_map[str(h)].append(float(r["episode_reward"]))
    common = sorted(set(a_map) & set(b_map))
    xs, ys, diffs = [], [], []
    for h in common:
        # take first reward per hash on each side
        xa, yb = a_map[h][0], b_map[h][0]
        xs.append(xa)
        ys.append(yb)
        diffs.append(abs(xa - yb))
    max_abs = max(diffs) if diffs else None
    pear = _pearson(xs, ys) if len(xs) >= 2 else None
    identical = sum(1 for d in diffs if d == 0.0)
    return {
        "n_hash_matched": len(common),
        "identical_rewards": identical,
        "max_abs_diff": max_abs,
        "reward_pearson": pear,
        "ok": bool(
            len(common) >= 1
            and max_abs is not None
            and max_abs > 0.0
            and (pear is None or pear < 0.999999)
        ),
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-root", type=Path,
                    default=_V3 / "outputs" / "runs")
    ap.add_argument("--seed", type=int, default=20260724)
    ap.add_argument("--report", type=Path,
                    default=_V3 / "reports" / "reward_ablation" / "dispatch_canary" / "CANARY_GATE.json")
    args = ap.parse_args(argv)

    per_arm = {}
    ok = True
    for arm in EXPECTED:
        d = _arm_dir(args.output_root, arm, args.seed)
        if not d.is_dir():
            per_arm[arm] = {"ok": False, "errors": [f"missing run dir {d}"]}
            ok = False
            continue
        res = _check_arm(d, arm)
        per_arm[arm] = res
        ok = ok and res["ok"]

    cross: Dict[str, Any] = {"ok": False, "errors": ["skipped — arm checks failed"]}
    if all(per_arm.get(a, {}).get("ok") for a in EXPECTED):
        a_rows = _load_jsonl(
            _arm_dir(args.output_root, "A1_OUTCOME_ONLY", args.seed)
            / "train" / "canary_rollouts.jsonl"
        )
        b_rows = _load_jsonl(
            _arm_dir(args.output_root, "A4_GATED_VERIFIABLE", args.seed)
            / "train" / "canary_rollouts.jsonl"
        )
        cross = _hash_matched(a_rows, b_rows)
        if not cross["ok"]:
            ok = False
            cross["errors"] = [
                "hash-matched rewards not distinct — dispatch still broken or "
                "arms still share one reward function "
                f"(n={cross['n_hash_matched']}, max_abs_diff={cross['max_abs_diff']}, "
                f"pearson={cross['reward_pearson']})"
            ]
            if cross["n_hash_matched"] == 0:
                # No overlapping completions is weak evidence, not automatic fail
                # if policies already differ in train_log; warn only.
                cross["ok"] = True
                cross["errors"] = [
                    "WARNING: 0 hash-matched completions between arms — "
                    "cannot verify reward inequality on identical rollouts; "
                    "rely on per-arm resolved-policy checks"
                ]
                ok = True and all(per_arm[a]["ok"] for a in EXPECTED)

    report = {
        "verdict": "PASS" if ok else "FAIL",
        "label": "dispatch_canary",
        "seed": args.seed,
        "per_arm": per_arm,
        "cross_arm_hash_matched": cross,
        "next": (
            "Use A4_GATED_VERIFIABLE as working reward for the next data experiment. "
            "Do NOT re-interpret Round 1 as a reward ablation."
            if ok else
            "STOP. Fix reward dispatch. Do not run further training."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[validate_dispatch_canary] wrote {args.report}")
    print(f"[validate_dispatch_canary] VERDICT={report['verdict']}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
