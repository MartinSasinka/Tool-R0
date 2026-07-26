#!/usr/bin/env python3
"""Scrape Phase-1 canary train logs into a compact diagnostics JSON.

Reads the C1 run directory produced by ``run_reward_ablation`` / 
``run_phase1_train.py`` and summarises dead/mixed groups, gradient norms, KL,
LoRA adapter delta (when present), executability and failure taxonomy.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def _mean(vals: List[float]) -> Optional[float]:
    return round(sum(vals) / len(vals), 6) if vals else None


def collect(run_dir: Path) -> Dict[str, Any]:
    train_log = None
    for cand in (run_dir / "train" / "train_log.jsonl",
                 run_dir / "train_log.jsonl",
                 run_dir / "logs" / "train_log.jsonl"):
        if cand.is_file():
            train_log = cand
            break
    # Fall back to any *train*.jsonl under the run.
    if train_log is None:
        hits = sorted(run_dir.rglob("*train*.jsonl"))
        train_log = hits[0] if hits else None

    rows = _read_jsonl(train_log) if train_log else []
    canary = None
    for cand in (run_dir / "train" / "canary_rollouts.jsonl",
                 run_dir / "canary_rollouts.jsonl"):
        if cand.is_file():
            canary = cand
            break

    dead = mixed = groups = 0
    grad_norms: List[float] = []
    kls: List[float] = []
    rewards: List[float] = []
    for r in rows:
        if "dead_group" in r or "dead" in r:
            groups += 1
            if r.get("dead_group") or r.get("dead"):
                dead += 1
            elif r.get("unique_rewards", 0) and int(r.get("unique_rewards") or 0) > 1:
                mixed += 1
            elif r.get("reward_std", 0) and float(r.get("reward_std") or 0) > 1e-6:
                mixed += 1
        if r.get("grad_norm") is not None:
            try:
                grad_norms.append(float(r["grad_norm"]))
            except (TypeError, ValueError):
                pass
        for key in ("kl", "kl_mean", "approx_kl", "mean_kl"):
            if r.get(key) is not None:
                try:
                    kls.append(float(r[key]))
                    break
                except (TypeError, ValueError):
                    pass
        if r.get("mean_reward") is not None:
            try:
                rewards.append(float(r["mean_reward"]))
            except (TypeError, ValueError):
                pass

    failure_tax: Counter = Counter()
    exec_ok = exec_n = 0
    if canary:
        for r in _read_jsonl(canary):
            fc = (r.get("failure_class") or r.get("diagnostics", {}).get("terminal_class")
                  or r.get("terminal_class") or "unknown")
            failure_tax[str(fc)] += 1
            exec_n += 1
            if r.get("executable") or (r.get("diagnostics") or {}).get("is_executable"):
                exec_ok += 1

    # LoRA delta: compare adapter weight norms if a simple summary exists.
    lora_delta = None
    for cand in (run_dir / "train" / "adapter_delta.json",
                 run_dir / "adapter_delta.json",
                 run_dir / "diagnostics" / "lora_delta.json"):
        if cand.is_file():
            lora_delta = json.loads(cand.read_text(encoding="utf-8"))
            break
    if lora_delta is None:
        # Presence of a final adapter is itself the LoRA delta vs C0.
        final = None
        for cand in (run_dir / "checkpoints" / "final",
                     run_dir / "final"):
            if (cand / "adapter_config.json").is_file():
                final = cand
                break
        lora_delta = {
            "final_adapter": str(final) if final else None,
            "present": final is not None,
            "note": "full weight-delta matrix not logged; adapter presence recorded",
        }

    return {
        "run_dir": str(run_dir),
        "train_log": str(train_log) if train_log else None,
        "canary_rollouts": str(canary) if canary else None,
        "n_log_rows": len(rows),
        "groups_logged": groups,
        "dead_groups": dead,
        "mixed_groups": mixed,
        "dead_group_rate": round(dead / groups, 6) if groups else None,
        "mixed_group_rate": round(mixed / groups, 6) if groups else None,
        "grad_norm_mean": _mean(grad_norms),
        "grad_norm_max": max(grad_norms) if grad_norms else None,
        "kl_mean": _mean(kls),
        "mean_reward_mean": _mean(rewards),
        "optimizer_steps_logged": sum(1 for r in rows if r.get("grad_norm") is not None
                                      or r.get("step") is not None),
        "executability_rate": round(exec_ok / exec_n, 6) if exec_n else None,
        "failure_taxonomy": dict(failure_tax),
        "lora_delta": lora_delta,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        print(f"[diag] DRY RUN — would scrape {args.run_dir} -> {args.out}")
        return 0

    report = collect(args.run_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(json.dumps({k: report[k] for k in
                      ("dead_group_rate", "mixed_group_rate", "grad_norm_mean",
                       "kl_mean", "executability_rate",
                       "optimizer_steps_logged")}, indent=2))
    print(f"[diag] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
