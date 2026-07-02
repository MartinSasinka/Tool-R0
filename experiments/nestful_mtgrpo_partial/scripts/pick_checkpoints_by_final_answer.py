#!/usr/bin/env python3
"""Pick best curriculum checkpoint per train stage by rollout_eval final_answer_pass.

Reads:  outputs/curriculum/stage_*/epoch_*/eval/metrics.json
Prints: ranked table + recommended CKPTS lines for run_checkpoint_evals.sh

Usage (from nestful_mtgrpo_partial/):
    python scripts/pick_checkpoints_by_final_answer.py
"""
from __future__ import annotations

import json
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CURR = os.path.join(ROOT, "outputs", "curriculum")


def main() -> int:
    rows = []
    for path in sorted(glob.glob(os.path.join(CURR, "stage_*/epoch_*/eval/metrics.json"))):
        rel = path.replace("\\", "/")
        stage = int(rel.split("stage_")[1].split("/")[0])
        epoch = int(rel.split("epoch_")[1].split("/")[0])
        with open(path, encoding="utf-8") as fh:
            m = json.load(fh)
        ckpt = f"outputs/curriculum/stage_{stage}/checkpoints/adapter_epoch_{epoch}"
        ckpt_exists = os.path.isfile(os.path.join(ROOT, ckpt, "adapter_config.json"))
        rows.append({
            "stage": stage,
            "epoch": epoch,
            "final_answer_pass": float(m.get("final_answer_pass", 0)),
            "strict_gold_trace_pass": float(m.get("strict_gold_trace_pass", 0)),
            "num_tasks": int(m.get("num_tasks", 0)),
            "wandb_eval": f"eval-stage{stage + 1}-e{epoch}",
            "checkpoint": ckpt,
            "exists": ckpt_exists,
        })

    if not rows:
        print("No metrics.json found under outputs/curriculum/", file=sys.stderr)
        return 1

    print("=== All rollout_eval results (sorted by stage, epoch) ===")
    print(f"{'stage':>5} {'ep':>3} {'final':>7} {'strict':>7} {'n':>5}  {'W&B':<16} {'ckpt ok'}")
    for r in rows:
        ok = "yes" if r["exists"] else "MISSING"
        print(f"{r['stage']:>5} {r['epoch']:>3} {r['final_answer_pass']:>7.3f} "
              f"{r['strict_gold_trace_pass']:>7.3f} {r['num_tasks']:>5}  "
              f"{r['wandb_eval']:<16} {ok}")

    best: dict[int, dict] = {}
    for r in rows:
        s = r["stage"]
        if s not in best or r["final_answer_pass"] > best[s]["final_answer_pass"]:
            best[s] = r

    print("\n=== BEST per stage (by final_answer_pass) ===")
    for s in sorted(best):
        r = best[s]
        strict_best = max((x for x in rows if x["stage"] == s),
                          key=lambda x: x["strict_gold_trace_pass"])
        note = ""
        if strict_best["epoch"] != r["epoch"]:
            note = (f"  (strict-best would be e{strict_best['epoch']} "
                    f"final={strict_best['final_answer_pass']:.3f} "
                    f"strict={strict_best['strict_gold_trace_pass']:.3f})")
        print(f"  stage {s}: epoch {r['epoch']}  final={r['final_answer_pass']:.3f}  "
              f"strict={r['strict_gold_trace_pass']:.3f}  -> {r['checkpoint']}{note}")

    g = max(rows, key=lambda x: x["final_answer_pass"])
    print(f"\nGlobal best final_answer: stage {g['stage']} epoch {g['epoch']} "
          f"= {g['final_answer_pass']:.3f} ({g['wandb_eval']})")

    print("\n=== Suggested run_checkpoint_evals.sh CKPTS lines ===")
    for s in sorted(best):
        r = best[s]
        label = f"partial_s{s}_e{r['epoch']}"
        print(f'  "{label}:{r["checkpoint"]}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
