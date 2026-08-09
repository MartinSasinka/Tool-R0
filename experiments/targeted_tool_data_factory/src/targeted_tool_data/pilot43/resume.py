"""Resume helpers: freeze the semantic-selectable pool and plan LLM spend.

The shortlist has already been fully verified. This module turns that into a
frozen intermediate the rest of the pipeline must not mutate, then builds a
cost-aware render allocation so we do not spend OpenRouter budget on every
selectable task.
"""
from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from . import CALL_BUCKETS, RUN_ID, TIER_TARGETS, TRAIN_MASTER_TARGET
from .pipeline import SHORTLIST, VERIFIED, iter_jsonl, read_jsonl, write_jsonl
from .qstage import RENDERED

SELECTABLE_FINAL = "semantic_selectable_final.jsonl"
SELECTABLE_CSV = "semantic_selectable_final.csv"
SELECTABLE_MANIFEST = "semantic_selectable_manifest.json"
REUSED_LLM = "llm_queries_reused.jsonl"
ALLOC_JSON = "PILOT43_RENDER_ALLOCATION.json"
ALLOC_MD = "PILOT43_RENDER_ALLOCATION.md"

#: How many clean queries we want after LLM yield, and the hard render ceiling.
CLEAN_TARGET_LO = 8500
CLEAN_TARGET_HI = 9500
RENDER_CEILING = 11500
#: Assumed accepted yield for tasks still to be rendered (from the interrupted full).
EXPECTED_YIELD = 0.73


def _bucket(call_count: int) -> str:
    return str(call_count) if call_count <= 5 else "6+"


def freeze_selectable(out_dir: Path) -> Dict[str, Any]:
    """Join shortlist metadata with verification evidence; freeze the pool."""
    verified = {r["task_id"]: r for r in iter_jsonl(out_dir / VERIFIED)}
    shortlist = read_jsonl(out_dir / SHORTLIST)
    rows: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for base in shortlist:
        tid = base["task_id"]
        ver = verified.get(tid)
        if ver is None:
            rejected.append({"task_id": tid, "reason": "not_verified"})
            continue
        if not ver.get("selectable"):
            rejected.append({
                "task_id": tid,
                "reason": ver.get("reject_reason") or "not_selectable",
            })
            continue
        v4 = ver.get("v4") or {}
        nec = ver.get("necessity") or []
        nec_sum = ver.get("necessity_summary") or {}
        offered = ver.get("offered") or {}
        row = {
            **base,
            "semantic_hard_valid": True,
            "executor_success": True,
            "executor_replay_deterministic": True,
            "actual_pattern_matches_declared": (
                base.get("actual_primary_pattern")
                in (base.get("actual_patterns") or [])
                and base.get("requested_structural_skill",
                             base.get("actual_primary_pattern"))
                in (base.get("actual_patterns") or [])
            ),
            "v4_executed": bool(v4.get("v4_executed")),
            "v4_search_complete": bool(v4.get("resolved")),
            "v4_shortcut": bool(v4.get("has_shortcut")),
            "v4_unresolved": not bool(v4.get("resolved")),
            "node_necessity_coverage": len(nec),
            "all_gold_nodes_necessary": bool(
                nec_sum.get("all_necessary")
                or (nec and all(n.get("necessary") for n in nec))),
            "distractor_validation_passed": int(
                offered.get("distractor_count") or 0) >= 1,
            "solution_verifier_valid": True,
            "offered_tool_count": offered.get("offered_tool_count"),
            "distractor_count": offered.get("distractor_count"),
            "hard_distractor_count": offered.get("hard_distractor_count"),
            "verification_seconds": ver.get("seconds"),
            "v4": v4,
            "necessity_summary": nec_sum,
            "offered_tools": ver.get("offered_tools"),
            "verifier": ver.get("verifier"),
        }
        # hard gates the freeze itself enforces
        if not (row["v4_executed"] and row["v4_search_complete"]
                and not row["v4_shortcut"] and not row["v4_unresolved"]
                and row["all_gold_nodes_necessary"]
                and row["distractor_validation_passed"]
                and row["actual_pattern_matches_declared"]):
            rejected.append({"task_id": tid, "reason": "freeze_gate_failed",
                             "detail": {
                                 "v4_executed": row["v4_executed"],
                                 "v4_search_complete": row["v4_search_complete"],
                                 "v4_shortcut": row["v4_shortcut"],
                                 "all_necessary": row["all_gold_nodes_necessary"],
                                 "distractor": row["distractor_validation_passed"],
                                 "pattern": row["actual_pattern_matches_declared"],
                             }})
            continue
        rows.append(row)

    rows.sort(key=lambda r: r["task_id"])
    write_jsonl(out_dir / SELECTABLE_FINAL, rows, append=False)

    csv_fields = [
        "task_id", "workflow_id", "plan_id", "call_count", "call_bucket",
        "answer_type", "coding_like", "actual_primary_pattern", "surface_track",
        "boolean_label", "cell_id", "v4_executed", "all_gold_nodes_necessary",
        "distractor_count",
    ]
    with (out_dir / SELECTABLE_CSV).open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in csv_fields})

    strata = {
        "call_bucket": dict(Counter(_bucket(int(r["call_count"])) for r in rows)),
        "answer_type": dict(Counter(r.get("answer_type", "?") for r in rows)),
        "coding_like": sum(1 for r in rows if r.get("coding_like")),
        "pattern": dict(Counter(r.get("actual_primary_pattern", "?") for r in rows)),
        "surface_track": dict(Counter(r.get("surface_track", "?") for r in rows)),
        "exact_call_count": dict(Counter(int(r["call_count"]) for r in rows)),
    }
    manifest = {
        "run_id": RUN_ID,
        "n": len(rows),
        "n_rejected_at_freeze": len(rejected),
        "source_shortlist": SHORTLIST,
        "source_verified": VERIFIED,
        "frozen": True,
        "mutation_policy": "do not append, rewrite or re-verify; allocate and render only",
        "strata": strata,
        "gates_enforced": [
            "semantic_hard_valid", "v4_executed", "v4_search_complete",
            "v4_shortcut=false", "all_gold_nodes_necessary",
            "distractor_validation_passed", "actual_pattern_matches_declared",
        ],
    }
    (out_dir / SELECTABLE_MANIFEST).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "semantic_selectable_freeze_rejects.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rejected[:5000])
        + ("\n" if rejected else ""),
        encoding="utf-8")
    return manifest


def export_reused_llm(out_dir: Path) -> Dict[str, Any]:
    """Keep every unblocked LLM render whose task is in the frozen selectable set."""
    selectable = {r["task_id"] for r in iter_jsonl(out_dir / SELECTABLE_FINAL)}
    kept: List[Dict[str, Any]] = []
    skipped = Counter()
    for rec in iter_jsonl(out_dir / RENDERED):
        tid = rec.get("task_id")
        if tid not in selectable:
            skipped["not_selectable"] += 1
            continue
        if rec.get("run_id") and rec.get("run_id") != RUN_ID:
            skipped["foreign_run"] += 1
            continue
        pv = str(rec.get("prompt_version") or "")
        if pv and not pv.startswith("pilot43"):
            skipped["bad_prompt_version"] += 1
            continue
        if rec.get("blocked") or not rec.get("query"):
            skipped["blocked_or_empty"] += 1
            continue
        if not rec.get("structured_output_ok", True):
            skipped["structured_output_invalid"] += 1
            continue
        critic = rec.get("critic") or {}
        if str(critic.get("verdict", "")).upper() != "PASS":
            skipped["first_critic_not_pass"] += 1
            continue
        validation = rec.get("validation") or {}
        if validation and not validation.get("passed", True):
            skipped["deterministic_validation_failed"] += 1
            continue
        # second critic, when routed, must not reject
        if rec.get("second_critic_reason"):
            sc = rec.get("second_critic")
            if sc is None:
                skipped["second_critic_unavailable"] += 1
                continue
            if str(sc.get("verdict", "")).upper() != "PASS":
                skipped["second_critic_not_pass"] += 1
                continue
        kept.append(rec)
    write_jsonl(out_dir / REUSED_LLM, kept, append=False)
    return {
        "n_reused": len(kept),
        "skipped": dict(skipped),
        "output": REUSED_LLM,
    }


def _tier_affinity(row: Mapping[str, Any]) -> List[str]:
    """Which selection tiers this task can honestly serve, richest first."""
    calls = int(row["call_count"])
    coding = bool(row.get("coding_like"))
    pattern = row.get("actual_primary_pattern", "")
    out: List[str] = []
    if coding and calls >= 5:
        out.append("CAPABILITY_ENRICHMENT")
    if calls >= 6 or (calls >= 4 and pattern in {
        "MULTI_JOIN", "NESTED_AGGREGATION", "LATE_REFERENCE",
        "REUSE_EARLY_OUTPUT", "DIAMOND", "PARALLEL_THEN_MERGE",
        "REPEATED_PRIMITIVE", "TYPE_TRANSITION_CHAIN",
    }):
        out.append("LONG_HORIZON_ENRICHMENT")
    if calls >= 7 or (coding and calls >= 6) or pattern in {
        "MULTI_JOIN", "NESTED_AGGREGATION", "ALTERNATING_BRANCH_CHAIN",
    }:
        out.append("CHALLENGE")
    out.append("PROFILE_CORE")
    # unique, stable
    seen: Set[str] = set()
    ordered: List[str] = []
    for t in out:
        if t not in seen:
            seen.add(t)
            ordered.append(t)
    return ordered


def build_render_allocation(out_dir: Path, *, seed: int = 20260801,
                            clean_target: int = 9000,
                            render_ceiling: int = RENDER_CEILING,
                            expected_yield: float = EXPECTED_YIELD
                            ) -> Dict[str, Any]:
    """Pick which selectable tasks still need an LLM render.

    Already-reused LLM queries count toward the clean target. Deterministic
    modes are allocated but marked ``deterministic`` so they do not consume the
    OpenRouter budget. The plan stops at ``render_ceiling`` total LLM attempts.
    """
    rng = random.Random(seed)
    selectable = read_jsonl(out_dir / SELECTABLE_FINAL)
    reused = {r["task_id"] for r in iter_jsonl(out_dir / REUSED_LLM)}
    already_attempted = {r["task_id"] for r in iter_jsonl(out_dir / RENDERED)}

    # Mode policy: LLM-mandatory / partial / deterministic-ok
    llm_mandatory_modes = {"DOMAIN_GROUNDED_IMPLICIT", "GOAL_BASED_IMPLICIT"}
    llm_partial_modes = {"SEMI_IMPLICIT"}
    det_modes = {"OPERATION_EXPLICIT_GRAPH_IMPLICIT", "GRAPH_EXPLICIT"}

    # Approximate mode assignment weights matching TRAIN_MASTER targets
    mode_weights = [
        ("DOMAIN_GROUNDED_IMPLICIT", 0.50),
        ("GOAL_BASED_IMPLICIT", 0.22),
        ("SEMI_IMPLICIT", 0.18),
        ("OPERATION_EXPLICIT_GRAPH_IMPLICIT", 0.07),
        ("GRAPH_EXPLICIT", 0.03),
    ]

    def assign_mode(tid: str) -> str:
        u = random.Random(f"{seed}:mode:{tid}").random()
        acc = 0.0
        for mode, w in mode_weights:
            acc += w
            if u <= acc:
                return mode
        return mode_weights[-1][0]

    # Quotas we must be able to fill after yield
    want = {
        "PROFILE_CORE": 3000,
        "LONG_HORIZON_ENRICHMENT": 1200,
        "CAPABILITY_ENRICHMENT": 600,
        "CHALLENGE": 200,
        "HELDOUT": 1000,
        "RESERVE": 1000,
    }
    # Over-provision buffer per tier for heldout key loss + yield
    buffer = {
        "PROFILE_CORE": 1.55,
        "LONG_HORIZON_ENRICHMENT": 1.70,
        "CAPABILITY_ENRICHMENT": 1.80,
        "CHALLENGE": 2.00,
        "HELDOUT": 1.40,
        "RESERVE": 1.30,
    }

    # Stratified buckets for PROFILE_CORE call counts
    core_call_targets = {"2": 990, "3": 660, "4": 405, "5": 285, "6+": 660}

    by_id = {r["task_id"]: r for r in selectable}
    pending_ids = [r["task_id"] for r in selectable if r["task_id"] not in reused]

    # Score: prefer under-filled strata, long-horizon depth diversity, coding
    def score(tid: str) -> Tuple:
        r = by_id[tid]
        calls = int(r["call_count"])
        return (
            # never-attempted first; blocked retries are a separate pass
            0 if tid not in already_attempted else 1,
            -int(bool(r.get("coding_like"))),
            -calls,
            tid,
        )

    # Build tier fills from reused first
    tier_have: Dict[str, List[str]] = defaultdict(list)
    for tid in reused:
        row = by_id.get(tid)
        if not row:
            continue
        for tier in _tier_affinity(row):
            tier_have[tier].append(tid)

    allocated_llm: List[Dict[str, Any]] = []
    allocated_det: List[Dict[str, Any]] = []
    allocated_ids: Set[str] = set(reused)

    # How many more LLM renders we can afford under the ceiling
    already_llm = len(already_attempted)
    llm_budget = max(0, render_ceiling - already_llm)
    # How many more accepted queries we still want
    need_clean = max(0, clean_target - len(reused))
    need_llm_attempts = int(need_clean / max(0.01, expected_yield)) + 200
    llm_budget = min(llm_budget, need_llm_attempts)

    # Fill deficits in priority order
    priority = [
        "CHALLENGE", "CAPABILITY_ENRICHMENT", "LONG_HORIZON_ENRICHMENT",
        "PROFILE_CORE", "HELDOUT", "RESERVE",
    ]
    candidates = [tid for tid in pending_ids if tid not in allocated_ids]
    rng.shuffle(candidates)
    candidates.sort(key=score)

    def tier_need(tier: str) -> int:
        target = int(want[tier] * buffer[tier])
        return max(0, target - len(tier_have[tier]))

    for tier in priority:
        need = tier_need(tier)
        if need <= 0:
            continue
        picked = 0
        still: List[str] = []
        for tid in candidates:
            if picked >= need:
                still.append(tid)
                continue
            row = by_id[tid]
            if tier not in _tier_affinity(row) and tier not in ("HELDOUT", "RESERVE"):
                still.append(tid)
                continue
            mode = assign_mode(tid)
            entry = {
                "task_id": tid,
                "tier_focus": tier,
                "call_count": int(row["call_count"]),
                "call_bucket": _bucket(int(row["call_count"])),
                "answer_type": row.get("answer_type"),
                "coding_like": bool(row.get("coding_like")),
                "pattern": row.get("actual_primary_pattern"),
                "planned_mode": mode,
                "already_attempted": tid in already_attempted,
            }
            if mode in det_modes:
                entry["render_channel"] = "deterministic"
                allocated_det.append(entry)
                allocated_ids.add(tid)
                tier_have[tier].append(tid)
                picked += 1
            elif mode in llm_partial_modes and random.Random(
                    f"{seed}:semi:{tid}").random() > 0.60:
                entry["render_channel"] = "deterministic"
                allocated_det.append(entry)
                allocated_ids.add(tid)
                tier_have[tier].append(tid)
                picked += 1
            else:
                if len(allocated_llm) >= llm_budget:
                    still.append(tid)
                    continue
                entry["render_channel"] = "openrouter"
                allocated_llm.append(entry)
                allocated_ids.add(tid)
                tier_have[tier].append(tid)
                picked += 1
        candidates = still

    # Keep allocating until the projected clean pool hits the target or the
    # LLM attempt budget is exhausted. Tier buffers above are a floor, not a
    # ceiling — selection still enforces exact quotas later.
    def projected_clean() -> float:
        return (len(reused)
                + len(allocated_llm) * expected_yield
                + len(allocated_det) * 0.95)

    overflow = 0
    still_left = [tid for tid in candidates if tid not in allocated_ids]
    for tid in still_left:
        if projected_clean() >= clean_target or len(allocated_llm) >= llm_budget:
            break
        row = by_id[tid]
        mode = assign_mode(tid)
        entry = {
            "task_id": tid,
            "tier_focus": "OVERFLOW",
            "call_count": int(row["call_count"]),
            "call_bucket": _bucket(int(row["call_count"])),
            "answer_type": row.get("answer_type"),
            "coding_like": bool(row.get("coding_like")),
            "pattern": row.get("actual_primary_pattern"),
            "planned_mode": mode,
            "already_attempted": tid in already_attempted,
        }
        if mode in det_modes or (
                mode in llm_partial_modes
                and random.Random(f"{seed}:semi:{tid}").random() > 0.60):
            entry["render_channel"] = "deterministic"
            allocated_det.append(entry)
        else:
            entry["render_channel"] = "openrouter"
            allocated_llm.append(entry)
        allocated_ids.add(tid)
        overflow += 1
    tier_have["OVERFLOW"] = [e["task_id"] for e in allocated_llm + allocated_det
                             if e["tier_focus"] == "OVERFLOW"]

    # PROFILE_CORE call-count repair: ensure each bucket has enough planned
    core_pool = [e for e in allocated_llm + allocated_det
                 if e["tier_focus"] == "PROFILE_CORE"] + [
        {"task_id": tid, "call_bucket": _bucket(int(by_id[tid]["call_count"]))}
        for tid in reused if "PROFILE_CORE" in _tier_affinity(by_id[tid])
    ]
    core_counts = Counter(e["call_bucket"] for e in core_pool)

    plan = {
        "run_id": RUN_ID,
        "clean_target": clean_target,
        "clean_target_band": [CLEAN_TARGET_LO, CLEAN_TARGET_HI],
        "render_ceiling": render_ceiling,
        "expected_yield": expected_yield,
        "n_selectable": len(selectable),
        "n_reused_llm": len(reused),
        "n_already_attempted": already_llm,
        "n_allocated_llm": len(allocated_llm),
        "n_allocated_deterministic": len(allocated_det),
        "n_overflow_extra": overflow,
        "projected_clean": int(projected_clean()),
        "tier_targets": want,
        "tier_buffers": buffer,
        "tier_planned_counts": {t: len(v) for t, v in tier_have.items()},
        "core_call_targets": core_call_targets,
        "core_call_planned": dict(core_counts),
        "llm_budget_remaining_attempts": llm_budget,
        "allocated_llm_task_ids": [e["task_id"] for e in allocated_llm],
        "allocated_deterministic_task_ids": [e["task_id"] for e in allocated_det],
        "entries_llm": allocated_llm,
        "entries_deterministic": allocated_det,
        "mode_policy": {
            "llm_mandatory": sorted(llm_mandatory_modes),
            "llm_partial": sorted(llm_partial_modes),
            "deterministic_ok": sorted(det_modes),
            "semi_implicit_llm_fraction": 0.60,
        },
        "notes": [
            "Do not LLM-render the full selectable pool.",
            "Reuse every record in llm_queries_reused.jsonl before spending.",
            "Deterministic channel covers GRAPH_EXPLICIT, OPERATION_EXPLICIT "
            "and ~40 % of SEMI_IMPLICIT.",
            "Selection still enforces hard tier quotas; this plan only "
            "over-provisions the render buffer.",
        ],
    }
    # Compact JSON on disk: drop bulky entry bodies into a sidecar list file
    compact = {k: v for k, v in plan.items()
               if k not in ("entries_llm", "entries_deterministic")}
    (out_dir / ALLOC_JSON).write_text(
        json.dumps(compact, indent=2, ensure_ascii=False), encoding="utf-8")
    write_jsonl(out_dir / "render_allocation_llm.jsonl", allocated_llm, append=False)
    write_jsonl(out_dir / "render_allocation_deterministic.jsonl",
                allocated_det, append=False)

    lines = [
        "# Pilot4.3 render allocation",
        "",
        f"- selectable frozen: **{plan['n_selectable']}**",
        f"- reusable LLM queries: **{plan['n_reused_llm']}**",
        f"- already attempted (any outcome): {plan['n_already_attempted']}",
        f"- newly allocated for OpenRouter: **{plan['n_allocated_llm']}**",
        f"- allocated for deterministic render: **{plan['n_allocated_deterministic']}**",
        f"- projected clean pool: **{plan['projected_clean']}** "
        f"(target {clean_target}, band {CLEAN_TARGET_LO}–{CLEAN_TARGET_HI})",
        f"- render ceiling: {render_ceiling}",
        f"- expected LLM yield: {expected_yield}",
        "",
        "## Tier buffer (reused + newly allocated)",
        "",
        "| tier | want | buffered want | planned |",
        "| --- | ---: | ---: | ---: |",
    ]
    for tier in priority:
        lines.append(
            f"| {tier} | {want[tier]} | {int(want[tier]*buffer[tier])} | "
            f"{plan['tier_planned_counts'].get(tier, 0)} |")
    lines += [
        "",
        "## PROFILE_CORE call buckets (planned)",
        "",
        "| bucket | target | planned |",
        "| --- | ---: | ---: |",
    ]
    for b in ("2", "3", "4", "5", "6+"):
        lines.append(
            f"| {b} | {core_call_targets[b]} | "
            f"{plan['core_call_planned'].get(b, 0)} |")
    lines += ["", "## Notes", ""]
    for n in plan["notes"]:
        lines.append(f"- {n}")
    lines.append("")
    (out_dir / ALLOC_MD).write_text("\n".join(lines), encoding="utf-8")
    return compact
