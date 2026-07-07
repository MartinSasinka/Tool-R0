#!/usr/bin/env python3
"""Build FIXED_REWARD_PILOT_REPORT.md for a fixed-stack smoke pilot run.

Reads the artifacts written by run_curriculum.sh / grpo_train.py:
  <run_dir>/global_best_react_win.json
  <run_dir>/stage_<N>/stage_manifest.json
  <run_dir>/stage_<N>/stage_gate_report.json
  <run_dir>/stage_<N>/epoch_<E>/train_summary.json
  <run_dir>/best_react_win_adapter/best_meta.json

and writes:
  <run_dir>/FIXED_REWARD_PILOT_REPORT.md

Usage:
  python build_fixed_reward_pilot_report.py --latest \
      --curriculum-version v3_1 --reward-policy execution_aware_v3_1_stepwise
  python build_fixed_reward_pilot_report.py --run-dir outputs/runs/<run_id>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
V3_ROOT = os.path.dirname(HERE)
RUNS_ROOT = os.path.join(V3_ROOT, "outputs", "runs")


def _load_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _load_jsonl(path: str):
    rows = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except ValueError:
                        pass
    except OSError:
        pass
    return rows


def find_latest_run() -> str | None:
    if not os.path.isdir(RUNS_ROOT):
        return None
    cands = [
        os.path.join(RUNS_ROOT, d)
        for d in os.listdir(RUNS_ROOT)
        if os.path.isdir(os.path.join(RUNS_ROOT, d))
    ]
    if not cands:
        return None
    return max(cands, key=os.path.getmtime)


def collect_stage(run_dir: str, n: int) -> dict | None:
    stage_dir = os.path.join(run_dir, f"stage_{n}")
    if not os.path.isdir(stage_dir):
        return None
    manifest = _load_json(os.path.join(stage_dir, "stage_manifest.json")) or {}
    gate = _load_json(os.path.join(stage_dir, "stage_gate_report.json")) or {}
    epochs = _load_jsonl(os.path.join(stage_dir, "epoch_summary.jsonl"))

    # Aggregate train_summary.json across epochs (smoke pilot: usually 1 epoch).
    summaries = []
    for name in sorted(os.listdir(stage_dir)):
        if name.startswith("epoch_"):
            s = _load_json(os.path.join(stage_dir, name, "train_summary.json"))
            if s:
                summaries.append(s)
    last = summaries[-1] if summaries else {}

    return {
        "stage": n,
        "dir": stage_dir,
        "manifest": manifest,
        "gate": gate,
        "epochs": epochs,
        "train_summaries": summaries,
        "last_summary": last,
    }


def fmt(v, digits=4):
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def decide(stages: list[dict], requested: list[int], reward_ok: bool,
           fractional: bool) -> str:
    if not stages or not reward_ok or not fractional:
        return "VALID_REWARD_SMOKE_TEST_FAILED"
    ran = [s["stage"] for s in stages]

    def gate_pass(n: int) -> bool:
        for s in stages:
            if s["stage"] == n:
                g = s["gate"]
                if g and "hard_fail" in g:
                    return not g["hard_fail"]
                return bool(s["manifest"].get("gate_pass"))
        return False

    if 1 in ran and 2 in ran and 3 in ran and gate_pass(3):
        return "FULL_STAGE123_SMOKE_COMPLETED"
    if 2 in ran and gate_pass(2) and 3 not in ran and 3 in requested:
        return "STAGE2_PASSED_STAGE3_BLOCKED"
    if 2 in ran and not gate_pass(2):
        return "STAGE2_PASSED_STAGE3_BLOCKED" if gate_pass(1) else "VALID_REWARD_SMOKE_TEST_FAILED"
    if 1 in ran and gate_pass(1) and 2 not in ran and 2 in requested:
        return "STAGE1_PASSED_STAGE2_BLOCKED"
    if 1 in ran and gate_pass(1):
        return "VALID_REWARD_SMOKE_TEST_PASSED"
    return "VALID_REWARD_SMOKE_TEST_FAILED"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--latest", action="store_true")
    ap.add_argument("--curriculum-version", default="v3_1")
    ap.add_argument("--reward-policy", default=None)
    args = ap.parse_args()

    run_dir = args.run_dir
    if run_dir is None and args.latest:
        run_dir = find_latest_run()
    if not run_dir or not os.path.isdir(run_dir):
        print(f"[report] ERROR: run dir not found (looked in {RUNS_ROOT})", file=sys.stderr)
        return 1
    run_dir = os.path.abspath(run_dir)
    run_id = os.path.basename(run_dir)

    stages = [s for s in (collect_stage(run_dir, n) for n in (1, 2, 3, 4)) if s]
    if any(s["stage"] == 4 for s in stages):
        print("[report] WARNING: stage_4 directory exists — that should NOT happen in a smoke pilot.")

    global_best = _load_json(os.path.join(run_dir, "global_best_react_win.json")) or {}
    best_meta = _load_json(os.path.join(run_dir, "best_react_win_adapter", "best_meta.json")) or {}
    requested = [int(x) for x in os.environ.get("STAGES", "1 2 3").split()]

    # Reward verification from the first stage that actually trained.
    first = stages[0]["last_summary"] if stages else {}
    configured = (args.reward_policy
                  or first.get("reward_policy_configured")
                  or first.get("reward_policy") or "n/a")
    resolved = first.get("reward_policy_resolved") or first.get("resolved_reward_policy")
    fn_module = first.get("reward_fn_module")
    fn_name = first.get("reward_fn_name")
    fallback = first.get("fallback_used")
    fractional = bool(first.get("fractional_rewards_present"))
    n_unique = first.get("n_unique_reward_values")
    dead_50 = first.get("dead_group_rate_first_50")
    reward_ok = bool(resolved) and (resolved == configured) and not fallback

    lines: list[str] = []
    w = lines.append
    w(f"# FIXED REWARD PILOT REPORT — {run_id}")
    w("")
    w(f"Curriculum: {args.curriculum_version} | generated from run artifacts in `{run_dir}`")
    w("")
    w("## A. Run configuration")
    w("")
    w("| field | value |")
    w("|---|---|")
    w(f"| run id | {run_id} |")
    w(f"| model | {first.get('model', 'n/a')} |")
    w(f"| reward policy configured | {configured} |")
    w(f"| reward policy resolved | {fmt(resolved)} |")
    w(f"| fallback used | {fmt(fallback)} |")
    w(f"| stages requested | {requested} |")
    w(f"| stages actually run | {[s['stage'] for s in stages]} |")
    w(f"| num generations | {stages[0]['manifest'].get('num_generations') if stages else 'n/a'} |")
    w(f"| train temperature | {os.environ.get('TRAIN_TEMPERATURE', 'n/a')} |")
    w(f"| regression guard | {os.environ.get('REGRESSION_GUARD', 'n/a')} |")
    w(f"| baseline dev Win | {fmt(global_best.get('baseline_win'))} |")
    w(f"| global best dev Win | {fmt(global_best.get('react_win_rate'))} |")
    w("")
    w("## B. Reward verification")
    w("")
    w("| metric | value |")
    w("|---|---|")
    w(f"| fractional rewards present | {fmt(fractional)} |")
    w(f"| reward values unique count | {fmt(n_unique)} |")
    w(f"| first 50 groups dead rate | {fmt(dead_50)} |")
    w(f"| reward fn module | {fmt(fn_module)} |")
    w(f"| reward fn name | {fmt(fn_name)} |")
    w("")
    w("## C. Stage results")
    w("")
    w("| stage | epochs | dev Win | delta vs baseline | steps | dead_group_rate | position_artifact_rate | gates |")
    w("|---|---|---|---|---|---|---|---|")
    base = global_best.get("baseline_win")
    for s in stages:
        ls = s["last_summary"]
        man = s["manifest"]
        dev = man.get("best_react_win_stage")
        delta = (dev - base) if (dev is not None and base is not None) else None
        gate = s["gate"]
        status = ("PASS" if gate and not gate.get("hard_fail")
                  else "FAIL" if gate else fmt(man.get("gate_pass")))
        w(f"| {s['stage']} | {len(s['train_summaries'])} | {fmt(dev)} | {fmt(delta)} "
          f"| {fmt(ls.get('steps'))} | {fmt(ls.get('dead_group_rate'))} "
          f"| {fmt(ls.get('position_artifact_group_rate'))} | {status} |")
    if not stages:
        w("| (no stage completed) | | | | | | | |")
    w("")
    w("## D. Failure diagnostics (last epoch per stage)")
    w("")
    w("| stage | no_tool_call_rate | too_few_calls_rate | avg_predicted_calls | contributing_turns |")
    w("|---|---|---|---|---|")
    for s in stages:
        ls = s["last_summary"]
        w(f"| {s['stage']} | {fmt(ls.get('no_tool_call_rate'))} | {fmt(ls.get('too_few_calls_rate'))} "
          f"| {fmt(ls.get('avg_predicted_calls'))} | {fmt(ls.get('contributing_turns_total'))} |")
    w("")
    if best_meta:
        w("### Best adapter")
        w("")
        w("```json")
        w(json.dumps(best_meta, indent=2)[:2000])
        w("```")
        w("")
    decision = decide(stages, requested, reward_ok, fractional)
    w("## E. Decision")
    w("")
    w(f"**{decision}**")
    w("")
    w("## F. Paper-safe interpretation")
    w("")
    if reward_ok:
        w(f"- The intended reward (`{configured}`) was verified to run in the DP rollout "
          f"workers (resolved to `{fn_module}.{fn_name}`, fallback_used=false).")
    else:
        w("- Reward dispatch was NOT verified for this run; no claims about the intended "
          "reward can be made from these results.")
    steps_total = sum(int(s["last_summary"].get("steps") or 0) for s in stages)
    if steps_total > 0:
        w(f"- GRPO produced a nonzero learning signal ({steps_total} optimizer steps across stages).")
    else:
        w("- GRPO produced ZERO optimizer steps — no learning signal; results are diagnostic only.")
    if base is not None and stages:
        devs = [s["manifest"].get("best_react_win_stage") for s in stages
                if s["manifest"].get("best_react_win_stage") is not None]
        if devs:
            best_dev = max(devs)
            if best_dev > base + 1e-9:
                w(f"- Dev ReAct Win improved over baseline ({fmt(best_dev)} vs {fmt(base)}) on the "
                  f"{os.environ.get('VAL_SUBSET_SIZE', '200')}-sample dev subset — smoke-level evidence only.")
            elif best_dev < base - 1e-9:
                w(f"- Dev ReAct Win regressed vs baseline ({fmt(best_dev)} vs {fmt(base)}).")
            else:
                w("- Dev ReAct Win was inconclusive (no change vs baseline).")
        else:
            w("- Dev Win was not evaluated; performance is inconclusive.")
    else:
        w("- No baseline dev Win recorded; performance comparison is inconclusive.")
    w("- This run does NOT use the NESTFUL test split. No test performance is claimed.")
    w("- No final transfer claim is made; full dev gates must pass in a longer run first.")
    w("")

    out_path = os.path.join(run_dir, "FIXED_REWARD_PILOT_REPORT.md")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[report] wrote {out_path}")
    print(f"[report] decision: {decision}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
