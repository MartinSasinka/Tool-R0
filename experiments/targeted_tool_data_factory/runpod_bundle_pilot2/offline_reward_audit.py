#!/usr/bin/env python3
"""Offline reward audit + variant rescoring over stored Pilot2 probe rollouts.

No new inference. Rebuilds episode rewards for five variants from the fields
already persisted in ``rollouts.jsonl``, then classifies within-group pairs into:

  terminal_class_inversion   — better terminal class, lower reward (HARD FAIL)
  clear_dominance_inversion  — gold-free Pareto dominance, lower reward
  success_success_disagreement — both official_success; differ only on gold-
                                 aware axes (prefix / gold arg errors). Reported,
                                 but NOT counted as a real inversion.
  incomparable_pair          — neither dominates on the gold-free axes

Hard gate: 0 terminal-class inversions. The safest passing variant is written
to ``SELECTED_REWARD_VARIANT.json`` for the Phase-1 canary.

Usage:
  python offline_reward_audit.py --probe-dir .../signal_probe
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

BUNDLE = Path(__file__).resolve().parent
sys.path.insert(0, str(BUNDLE))
from signal_probe_lib import (  # noqa: E402
    DEAD_EPS, build_group, extract_task_meta, phase_summary,
)

# Terminal scalars / epsilons mirrored from reward_ablation_registry.py so this
# audit stays offline-capable without importing the full v3 reward stack. Kept
# in sync by test_phase1_next.py::test_terminal_scalars_match_registry.
TERMINAL_RANK = {
    "official_success": 0,
    "executable_wrong_result": 1,
    "executable_partial": 2,
    "execution_failure": 3,
    "parse_or_no_call": 4,
}
TERMINAL_SCALARS = {
    "A4": {
        "official_success": 0.97,
        "executable_wrong_result": 0.2,
        "executable_partial": 0.1575,
        "execution_failure": 0.115,
        "parse_or_no_call": 0.02,
    },
    "A1": {
        "official_success": 0.96,
        "executable_wrong_result": 0.53,
        "executable_partial": 0.34,
        "execution_failure": 0.15,
        "parse_or_no_call": 0.02,
    },
}
A4_EPS_DEFAULT = 0.02

VARIANT_SPECS = (
    {"id": "A4_current", "family": "A4", "epsilon": A4_EPS_DEFAULT,
     "success_flat": False,
     "train_policy": "reward_ablation_A4_GATED_VERIFIABLE",
     "description": "A4 gated verifiable as probed (eps=0.02, process on success)"},
    {"id": "A4_success_flat", "family": "A4", "epsilon": A4_EPS_DEFAULT,
     "success_flat": True,
     "train_policy": "reward_ablation_A4_GATED_VERIFIABLE",
     "description": "A4 with process tie-break zeroed on official_success"},
    {"id": "A4_eps_0.01", "family": "A4", "epsilon": 0.01,
     "success_flat": False,
     "train_policy": "reward_ablation_A4_GATED_VERIFIABLE",
     "description": "A4 with epsilon=0.01"},
    {"id": "A4_eps_0.005", "family": "A4", "epsilon": 0.005,
     "success_flat": False,
     "train_policy": "reward_ablation_A4_GATED_VERIFIABLE",
     "description": "A4 with epsilon=0.005"},
    {"id": "A1_outcome_only", "family": "A1", "epsilon": 0.0,
     "success_flat": False,
     "train_policy": "reward_ablation_A1_OUTCOME_ONLY",
     "description": "A1 outcome-only reference (no process)"},
)


# ───────────────────────────────────────────────────────────── IO helpers ──

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_probe_rollouts(probe_dir: Path) -> List[Dict[str, Any]]:
    merged = probe_dir / "rollouts.jsonl"
    if merged.is_file():
        return read_jsonl(merged)
    by_key: Dict[str, Dict[str, Any]] = {}
    for shard in sorted(probe_dir.glob("shard_*.jsonl")):
        for rec in read_jsonl(shard):
            key = str(rec.get("cache_key")
                      or f"{rec.get('phase')}|{rec.get('task_id')}|{rec.get('rollout_idx')}")
            by_key[key] = rec
    return list(by_key.values())


# ─────────────────────────────────────────────────────────── rescoring ──

def rescore(rec: Dict[str, Any], variant: Dict[str, Any]) -> Dict[str, Any]:
    """Rebuild episode reward for one variant from stored terminal + process."""
    terminal = str(rec.get("terminal_outcome") or "parse_or_no_call")
    if terminal not in TERMINAL_RANK:
        terminal = "parse_or_no_call"
    scalars = TERMINAL_SCALARS[variant["family"]]
    terminal_score = float(scalars[terminal])
    process = float(rec.get("process_reward") or 0.0)
    if variant["family"] == "A1":
        process = 0.0
    if variant.get("success_flat") and terminal == "official_success":
        process = 0.0
    eps = float(variant["epsilon"])
    total = round(terminal_score + eps * process, 6)
    return {
        "terminal_class": terminal,
        "terminal_score": terminal_score,
        "process_score": process,
        "epsilon": eps,
        "episode_reward": total,
        "success": terminal == "official_success",
    }


# ──────────────────────────────────────────────────── pair classification ──

def gold_free_quality(rec: Dict[str, Any]) -> Tuple[float, ...]:
    """Axes that A4's verifiable process is allowed to care about.

    Deliberately excludes gold-prefix length and gold argument-error counts —
    those are the source of the probe's false "inversions" inside the success
    band.
    """
    return (
        1.0 if rec.get("success") or rec.get("terminal_outcome") == "official_success"
        else 0.0,
        float(rec.get("n_successful_calls") or 0),
        float(rec.get("executable_frac") or 0.0),
        1.0 if not rec.get("parse_error") and not rec.get("clipped") else 0.0,
        -float(TERMINAL_RANK.get(str(rec.get("terminal_outcome")), 4)),
    )


def gold_aware_quality(rec: Dict[str, Any]) -> Tuple[float, ...]:
    return (
        float(rec.get("correct_prefix_len") or 0),
        -float((rec.get("arg_key_errors") or 0)
               + (rec.get("arg_type_errors") or 0)
               + (rec.get("arg_value_errors") or 0)),
    )


def _dominates(a: Tuple[float, ...], b: Tuple[float, ...]) -> bool:
    return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))


def classify_pair(a: Dict[str, Any], b: Dict[str, Any], *,
                  reward_a: float, reward_b: float
                  ) -> Dict[str, Any]:
    """Classify one ordered pair (a vs b) under the rescored rewards."""
    ta = str(a.get("terminal_outcome") or "parse_or_no_call")
    tb = str(b.get("terminal_outcome") or "parse_or_no_call")
    ra, rb = TERMINAL_RANK.get(ta, 4), TERMINAL_RANK.get(tb, 4)

    # Terminal-class inversion: better class must not get a lower reward.
    if ra != rb:
        better, worse = (a, b) if ra < rb else (b, a)
        r_better = reward_a if ra < rb else reward_b
        r_worse = reward_b if ra < rb else reward_a
        if r_better + DEAD_EPS < r_worse:
            return {
                "kind": "terminal_class_inversion",
                "better_rollout": better.get("rollout_idx"),
                "worse_rollout": worse.get("rollout_idx"),
                "better_reward": r_better,
                "worse_reward": r_worse,
                "better_terminal": better.get("terminal_outcome"),
                "worse_terminal": worse.get("terminal_outcome"),
            }

    both_success = (ta == "official_success" and tb == "official_success")
    gf_a, gf_b = gold_free_quality(a), gold_free_quality(b)
    ga_a, ga_b = gold_aware_quality(a), gold_aware_quality(b)

    if both_success and gf_a == gf_b:
        # Same gold-free quality: any reward disagreement is just gold-prefix /
        # gold-arg noise and must NOT gate training.
        if _dominates(ga_a, ga_b) or _dominates(ga_b, ga_a) or abs(reward_a - reward_b) > DEAD_EPS:
            better_ga = a if _dominates(ga_a, ga_b) else b
            worse_ga = b if better_ga is a else a
            return {
                "kind": "success_success_disagreement",
                "better_rollout": better_ga.get("rollout_idx"),
                "worse_rollout": worse_ga.get("rollout_idx"),
                "better_reward": reward_a if better_ga is a else reward_b,
                "worse_reward": reward_b if better_ga is a else reward_a,
                "note": "gold-prefix / gold-arg disagreement inside two successes",
            }
        return {"kind": "incomparable_pair"}

    if _dominates(gf_a, gf_b):
        if reward_a + DEAD_EPS < reward_b:
            return {
                "kind": "clear_dominance_inversion",
                "better_rollout": a.get("rollout_idx"),
                "worse_rollout": b.get("rollout_idx"),
                "better_reward": reward_a,
                "worse_reward": reward_b,
                "better_terminal": ta,
                "worse_terminal": tb,
            }
        return {"kind": "incomparable_pair"}  # concordant or tied
    if _dominates(gf_b, gf_a):
        if reward_b + DEAD_EPS < reward_a:
            return {
                "kind": "clear_dominance_inversion",
                "better_rollout": b.get("rollout_idx"),
                "worse_rollout": a.get("rollout_idx"),
                "better_reward": reward_b,
                "worse_reward": reward_a,
                "better_terminal": tb,
                "worse_terminal": ta,
            }
        return {"kind": "incomparable_pair"}
    return {"kind": "incomparable_pair"}


# ───────────────────────────────────────────────────── per-variant report ──

def decision_records(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deepest evidence per (task, rollout_idx): P3 beats P2."""
    best: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for rec in records:
        key = (str(rec.get("task_id")), int(rec.get("rollout_idx") or 0))
        prev = best.get(key)
        if prev is None or (prev.get("phase") != "P3" and rec.get("phase") == "P3"):
            best[key] = rec
        elif prev.get("phase") == rec.get("phase"):
            best[key] = rec  # last write wins inside a phase (dedupe)
    return list(best.values())


def evaluate_variant(records: Sequence[Dict[str, Any]],
                     variant: Dict[str, Any],
                     meta_by_id: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    scored: List[Dict[str, Any]] = []
    for rec in records:
        s = rescore(rec, variant)
        row = dict(rec)
        row["episode_reward"] = s["episode_reward"]
        row["process_reward"] = s["process_score"]
        row["reward_terminal_score"] = s["terminal_score"]
        row["reward_epsilon"] = s["epsilon"]
        row["success"] = s["success"]
        row["terminal_outcome"] = s["terminal_class"]
        scored.append(row)

    # Prefer P3 evidence for group metrics when present.
    by_task_phase: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for rec in scored:
        by_task_phase[(str(rec.get("phase")), str(rec.get("task_id")))].append(rec)

    decision: Dict[str, List[Dict[str, Any]]] = {}
    for (phase, tid), recs in by_task_phase.items():
        if tid not in decision or phase == "P3":
            decision[tid] = recs

    groups = []
    for tid, recs in sorted(decision.items()):
        meta = meta_by_id.get(tid) or {
            "task_id": tid,
            "track": recs[0].get("track"),
            "call_count": recs[0].get("call_count"),
            "motif": recs[0].get("motif"),
            "answer_type": recs[0].get("answer_type"),
            "generation_cell": recs[0].get("generation_cell"),
        }
        # Tag the group with the phase of the evidence used.
        phase = "P3" if any(r.get("phase") == "P3" for r in recs) else "P2"
        groups.append(build_group(meta, recs, phase=phase))

    summary = phase_summary(groups)

    # Pair audit on decision evidence only.
    counts = Counter()
    examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for tid, recs in decision.items():
        ordered = sorted(recs, key=lambda r: int(r.get("rollout_idx") or 0))
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                a, b = ordered[i], ordered[j]
                kind_rec = classify_pair(
                    a, b,
                    reward_a=float(a["episode_reward"]),
                    reward_b=float(b["episode_reward"]),
                )
                kind = kind_rec["kind"]
                counts[kind] += 1
                if kind != "incomparable_pair" and len(examples[kind]) < 8:
                    examples[kind].append({"task_id": tid, **kind_rec})

    # Reconstruct fidelity for A4_current against the stored probe rewards.
    fidelity = None
    if variant["id"] == "A4_current":
        deltas = []
        for rec, row in zip(records, scored):
            if rec.get("cache_key") != row.get("cache_key"):
                continue
            if rec.get("episode_reward") is None:
                continue
            # Only compare when the stored row was scored with A4.
            if "A4" not in str(rec.get("reward_policy") or ""):
                continue
            deltas.append(abs(float(rec["episode_reward"]) - float(row["episode_reward"])))
        fidelity = {
            "n_compared": len(deltas),
            "max_abs_delta": max(deltas) if deltas else None,
            "mean_abs_delta": (sum(deltas) / len(deltas)) if deltas else None,
            "within_1e-6": all(d <= 1e-6 for d in deltas) if deltas else None,
        }

    return {
        "variant_id": variant["id"],
        "description": variant["description"],
        "train_policy": variant["train_policy"],
        "epsilon": variant["epsilon"],
        "success_flat": bool(variant.get("success_flat")),
        "family": variant["family"],
        "summary": summary,
        "pair_counts": dict(counts),
        "terminal_class_inversions": int(counts.get("terminal_class_inversion", 0)),
        "clear_dominance_inversions": int(counts.get("clear_dominance_inversion", 0)),
        "success_success_disagreements": int(counts.get("success_success_disagreement", 0)),
        "incomparable_pairs": int(counts.get("incomparable_pair", 0)),
        "examples": {k: v for k, v in examples.items()},
        "reconstruction_fidelity": fidelity,
        "hard_gate_pass": int(counts.get("terminal_class_inversion", 0)) == 0,
        "dead_group_rate": summary.get("dead_group_rate"),
        "terminal_mixed_rate": summary.get("terminal_mixed_rate"),
        "process_only_mixed_rate": summary.get("process_only_mixed_rate"),
        "mean_reward_range": summary.get("mean_reward_range"),
    }


def select_safest(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Pick the safest variant under the hard gate + signal heuristics."""
    passing = [r for r in results if r["hard_gate_pass"]]
    if not passing:
        return {
            "selected": None,
            "reason": "no variant has 0 terminal-class inversions",
            "hard_gate": "FAIL",
        }

    def _key(r: Dict[str, Any]) -> Tuple:
        # Lower clear-dominance inversions is safer; then prefer more mixed
        # signal and less dead; then prefer A4 over A1; then the probed
        # default (A4_current) over epsilon tweaks / success-flat, since the
        # refined taxonomy already cleared the success-band false alarms.
        signal = ((r.get("terminal_mixed_rate") or 0.0)
                  + (r.get("process_only_mixed_rate") or 0.0))
        family_pref = 0 if r["family"] == "A4" else 1
        # Explicit preference order among otherwise-tied A4 flavours.
        flavour_rank = {
            "A4_current": 0,
            "A4_success_flat": 1,
            "A4_eps_0.01": 2,
            "A4_eps_0.005": 3,
            "A1_outcome_only": 4,
        }.get(r["variant_id"], 9)
        return (
            r["clear_dominance_inversions"],
            -(signal),
            r.get("dead_group_rate") or 1.0,
            family_pref,
            flavour_rank,
        )

    best = sorted(passing, key=_key)[0]
    reason = (
        f"{best['variant_id']}: 0 terminal-class inversions, "
        f"{best['clear_dominance_inversions']} clear-dominance inversions, "
        f"dead={best['dead_group_rate']}, "
        f"terminal_mixed={best['terminal_mixed_rate']}, "
        f"process_only_mixed={best['process_only_mixed_rate']}"
    )
    return {
        "selected": best["variant_id"],
        "train_policy": best["train_policy"],
        "epsilon": best["epsilon"],
        "success_flat": best["success_flat"],
        "family": best["family"],
        "description": best["description"],
        "reason": reason,
        "hard_gate": "PASS",
        "metrics": {
            "terminal_class_inversions": best["terminal_class_inversions"],
            "clear_dominance_inversions": best["clear_dominance_inversions"],
            "success_success_disagreements": best["success_success_disagreements"],
            "dead_group_rate": best["dead_group_rate"],
            "terminal_mixed_rate": best["terminal_mixed_rate"],
            "process_only_mixed_rate": best["process_only_mixed_rate"],
            "mean_reward_range": best["mean_reward_range"],
        },
    }


def render_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Offline reward audit — Pilot2 probe rollouts",
        "",
        f"**Selected variant: `{report['selection']['selected']}`**",
        "",
        f"Hard gate (0 terminal-class inversions): "
        f"**{report['selection']['hard_gate']}**",
        "",
        report["selection"].get("reason") or "",
        "",
        "Gold-prefix disagreement inside two valid successes is classified as "
        "`success_success_disagreement` and is **not** a gate failure.",
        "",
        "## Variants",
        "",
        "| variant | term. inv. | clear-dom. inv. | success-success | dead | "
        "terminal-mixed | process-only-mixed | mean reward range | gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in report["variants"]:
        lines.append(
            f"| `{r['variant_id']}` | {r['terminal_class_inversions']} | "
            f"{r['clear_dominance_inversions']} | "
            f"{r['success_success_disagreements']} | "
            f"{_pct(r['dead_group_rate'])} | {_pct(r['terminal_mixed_rate'])} | "
            f"{_pct(r['process_only_mixed_rate'])} | {r['mean_reward_range']} | "
            f"{'PASS' if r['hard_gate_pass'] else 'FAIL'} |"
        )
    lines += ["", "## Selection", "",
              f"- selected: `{report['selection']['selected']}`",
              f"- train_policy: `{report['selection'].get('train_policy')}`",
              f"- success_flat: {report['selection'].get('success_flat')}",
              f"- epsilon: {report['selection'].get('epsilon')}",
              f"- reason: {report['selection'].get('reason')}",
              ""]
    a4 = next((v for v in report["variants"] if v["variant_id"] == "A4_current"), None)
    if a4 and a4.get("reconstruction_fidelity"):
        f = a4["reconstruction_fidelity"]
        lines += ["## A4_current reconstruction fidelity", "",
                  f"- compared: {f.get('n_compared')}",
                  f"- max |Δ|: {f.get('max_abs_delta')}",
                  f"- within 1e-6: {f.get('within_1e-6')}",
                  ""]
    return "\n".join(lines) + "\n"


def _pct(v: Optional[float]) -> str:
    return "—" if v is None else f"{100.0 * float(v):.1f}%"


# ─────────────────────────────────────────────────────────────────── main ──

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe-dir", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=None,
                    help="frozen train_grpo_pilot2.jsonl (for task metadata)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="defaults to <probe-dir>/offline_reward_audit")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    probe_dir = args.probe_dir
    out_dir = args.out_dir or (probe_dir / "offline_reward_audit")
    data = args.data or (BUNDLE / "data" / "train_grpo_pilot2.jsonl")

    records = load_probe_rollouts(probe_dir)
    if not records:
        print(f"[audit] ABORT: no rollouts in {probe_dir}", file=sys.stderr)
        return 2

    meta_by_id: Dict[str, Dict[str, Any]] = {}
    if data.is_file():
        for row in read_jsonl(data):
            m = extract_task_meta(row)
            meta_by_id[m["task_id"]] = m

    decision = decision_records(records)
    print(f"[audit] loaded {len(records)} rollouts "
          f"({len(decision)} decision-evidence rows after P3 preference)")

    if args.dry_run:
        print(f"[audit] DRY RUN — would rescore {len(VARIANT_SPECS)} variants "
              f"into {out_dir}")
        return 0

    results = [evaluate_variant(decision, v, meta_by_id) for v in VARIANT_SPECS]
    selection = select_safest(results)
    report = {
        "probe_dir": str(probe_dir),
        "n_rollouts_loaded": len(records),
        "n_decision_rows": len(decision),
        "variants": results,
        "selection": selection,
        "hard_gate": "0 terminal_class_inversions",
        "pair_taxonomy": [
            "terminal_class_inversion",
            "clear_dominance_inversion",
            "success_success_disagreement",
            "incomparable_pair",
        ],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "OFFLINE_REWARD_AUDIT.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "OFFLINE_REWARD_AUDIT.md").write_text(
        render_md(report), encoding="utf-8")

    selected_path = out_dir / "SELECTED_REWARD_VARIANT.json"
    selected_path.write_text(json.dumps(selection, indent=2, ensure_ascii=False),
                             encoding="utf-8")
    # Also publish next to the probe root so the canary can find it easily.
    (probe_dir / "SELECTED_REWARD_VARIANT.json").write_text(
        json.dumps(selection, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[audit] selected={selection.get('selected')} "
          f"hard_gate={selection.get('hard_gate')}")
    print(f"[audit] reason: {selection.get('reason')}")
    for r in results:
        print(f"[audit]   {r['variant_id']}: term_inv={r['terminal_class_inversions']} "
              f"clear_dom={r['clear_dominance_inversions']} "
              f"ss={r['success_success_disagreements']} "
              f"dead={r['dead_group_rate']} gate="
              f"{'PASS' if r['hard_gate_pass'] else 'FAIL'}")
    print(f"[audit] report -> {out_dir / 'OFFLINE_REWARD_AUDIT.md'}")

    if selection.get("hard_gate") != "PASS" or not selection.get("selected"):
        print("[audit] ABORT: no safe reward variant", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
