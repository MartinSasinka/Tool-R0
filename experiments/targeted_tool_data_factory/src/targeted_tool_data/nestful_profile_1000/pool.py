"""Load Pilot4.3 clean pool, eligibility, heldout/reserve leakage shields."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Set, Tuple

from ..pilot43.pipeline import VERIFIED, iter_jsonl, read_jsonl
from ..pilot43.resume import SELECTABLE_FINAL
from .quotas import (depth_bucket, join_bucket, map_motif, ref_band, tool_band)


def _bucket(n: int) -> str:
    return str(n) if n <= 5 else "6+"


def load_blocked_keys(pilot_out: Path) -> Dict[str, Set[str]]:
    """Keys that would leak into the existing Pilot4.3 heldout/reserve.

    Important: ``heldout_standard_profile`` shares workflows with train by design.
    Only family-holdout *parts* donate blocking keys (workflow_family, program_plan,
    query_template, capability_combination, actual_topology, surface). Every
    heldout/reserve ``task_id`` and ``workflow_instance_id`` is always blocked.
    """
    blocked_ids: Set[str] = set()
    workflows: Set[str] = set()
    plans: Set[str] = set()
    templates: Set[str] = set()
    topologies: Set[str] = set()
    instances: Set[str] = set()
    combos: Set[str] = set()

    part_key_fields = {
        "heldout_workflow_family.jsonl": "workflow",
        "heldout_program_plan.jsonl": "plan",
        "heldout_query_template.jsonl": "template",
        "heldout_capability_combination.jsonl": "combo",
        "heldout_actual_topology.jsonl": "topology",
        "heldout_standard_profile.jsonl": "standard",
        "heldout_surface.jsonl": "surface",
    }

    def _add_row(r: Mapping[str, Any], kind: str) -> None:
        blocked_ids.add(r["task_id"])
        instances.add(r.get("workflow_instance_id") or "")
        if kind == "workflow":
            workflows.add(r.get("workflow_id") or "")
        elif kind == "plan":
            caps = (r.get("declared") or {}).get("normalized_capability_sequence") or r.get(
                "normalized_capability_sequence")
            if caps:
                plans.add(caps)
        elif kind == "template" or kind == "standard":
            fp = (r.get("query_fingerprints") or {}).get("intent_fingerprint")
            if fp:
                templates.add(fp)
        elif kind == "combo":
            fams = r.get("capability_families") or []
            if fams:
                combos.add("+".join(fams))
        elif kind == "topology":
            pat = (r.get("declared") or {}).get("structural_pattern") or ""
            bucket = r.get("call_bucket") or ""
            if pat:
                topologies.add(f"{pat}|{bucket}")

    for name, kind in part_key_fields.items():
        path = pilot_out / name
        if path.exists():
            for r in iter_jsonl(path):
                _add_row(r, kind)

    # Reserve: never used for selection (task + instance only; no family expansion)
    if (pilot_out / "reserve_1000.jsonl").exists():
        for r in iter_jsonl(pilot_out / "reserve_1000.jsonl"):
            blocked_ids.add(r["task_id"])
            instances.add(r.get("workflow_instance_id") or "")

    man = pilot_out / "split_manifest.json"
    if man.exists():
        data = json.loads(man.read_text(encoding="utf-8"))
        for tasks in (data.get("heldout") or {}).values():
            blocked_ids.update(tasks)
        blocked_ids.update(data.get("reserve") or [])

    return {
        "task_id": blocked_ids,
        "workflow_id": {x for x in workflows if x},
        "normalized_capability_sequence": {x for x in plans if x},
        "workflow_instance_id": {x for x in instances if x},
        "intent_fingerprint": {x for x in templates if x},
        "capability_combo": {x for x in combos if x},
        "topology_key": {x for x in topologies if x},
        "surface_holdout_track": {"G_GENERAL_2"},
    }


def _critic_ok(q: Mapping[str, Any]) -> bool:
    if not q.get("passed"):
        return False
    src = q.get("query_source") or ""
    if src != "openrouter":
        return True
    critic = q.get("critic") or {}
    if not critic.get("executed"):
        return False
    if str(critic.get("verdict") or "").upper() != "PASS":
        return False
    c2 = q.get("critic2") or {}
    if c2.get("routed") or c2.get("executed"):
        # if routed, must have PASS
        if c2.get("routed") and str(c2.get("verdict") or "").upper() != "PASS":
            return False
    return True


def _ref_density(gf: Mapping[str, Any], call_count: int) -> float:
    """Nestful-like reference density in [0, 1].

    PROFILE_SAFE uses n_ref_args/n_args. Without arg bags here we use
    n_edges / max(n_nodes, 1), which is the closest exported graph ratio.
    """
    n_edges = int(gf.get("n_edges") or 0)
    n_nodes = int(gf.get("n_nodes") or call_count or 1)
    return round(min(1.0, n_edges / max(n_nodes, 1)), 4)


def build_candidates(pilot_out: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Join clean queries × selectable × verified; apply eligibility + leakage shield."""
    blocked = load_blocked_keys(pilot_out)
    queries = {r["task_id"]: r for r in iter_jsonl(pilot_out / "query_hard_valid.jsonl")}
    selectable = {r["task_id"]: r for r in iter_jsonl(pilot_out / SELECTABLE_FINAL)}
    verified = {r["task_id"]: r for r in iter_jsonl(pilot_out / VERIFIED)
                if r.get("selectable")}

    # Prefer full export when present (gold_calls already materialised)
    full: Dict[str, Dict[str, Any]] = {}
    for name in ("selected_all.jsonl", "train_master_5000.jsonl",
                 "heldout_all.jsonl", "reserve_1000.jsonl"):
        p = pilot_out / name
        if not p.exists():
            continue
        for r in iter_jsonl(p):
            full.setdefault(r["task_id"], r)

    reject_reasons: Counter = Counter()
    candidates: List[Dict[str, Any]] = []
    exact_fp_seen: Dict[str, str] = {}

    for tid, q in queries.items():
        sel = selectable.get(tid)
        ver = verified.get(tid)
        if sel is None or ver is None:
            reject_reasons["missing_semantic_or_verified"] += 1
            continue
        if not _critic_ok(q):
            reject_reasons["critic_or_query_gate"] += 1
            continue

        # Freeze / semantic gates (re-check)
        v4 = ver.get("v4") or sel.get("v4") or {}
        nec = ver.get("necessity_summary") or sel.get("necessity_summary") or {}
        if not (sel.get("v4_executed") and sel.get("v4_search_complete")
                and not sel.get("v4_shortcut") and not sel.get("v4_unresolved")):
            reject_reasons["v4_gate"] += 1
            continue
        if not sel.get("all_gold_nodes_necessary"):
            reject_reasons["necessity_gate"] += 1
            continue
        if not sel.get("distractor_validation_passed"):
            reject_reasons["distractor_gate"] += 1
            continue

        # Leakage shield
        if tid in blocked["task_id"]:
            reject_reasons["heldout_or_reserve_task"] += 1
            continue
        if sel.get("workflow_id") in blocked["workflow_id"]:
            reject_reasons["heldout_workflow_family"] += 1
            continue
        plan_key = sel.get("normalized_capability_sequence") or ""
        if plan_key and plan_key in blocked["normalized_capability_sequence"]:
            reject_reasons["heldout_program_plan"] += 1
            continue
        if sel.get("workflow_instance_id") in blocked["workflow_instance_id"]:
            reject_reasons["heldout_instance"] += 1
            continue
        intent = (q.get("fingerprints") or {}).get("intent_fingerprint") or ""
        if intent and intent in blocked["intent_fingerprint"]:
            reject_reasons["heldout_template"] += 1
            continue
        combo = "+".join(sel.get("capability_families") or [])
        if combo and combo in blocked["capability_combo"]:
            reject_reasons["heldout_capability_combo"] += 1
            continue
        topo = f"{sel.get('actual_primary_pattern')}|{sel.get('call_bucket')}"
        if topo in blocked["topology_key"]:
            reject_reasons["heldout_topology"] += 1
            continue
        track = sel.get("surface_track") or ""
        if track in blocked["surface_holdout_track"]:
            reject_reasons["surface_holdout_track"] += 1
            continue

        # Exact query duplicate within candidate pool
        exact = (q.get("fingerprints") or {}).get("exact_fingerprint") or ""
        if exact and exact in exact_fp_seen:
            reject_reasons["exact_duplicate_query"] += 1
            continue

        fr = full.get(tid)
        if fr and isinstance(fr.get("gold_calls"), list):
            call_count = len(fr["gold_calls"])
            answer_type = fr.get("answer_type") or sel.get("answer_type")
            offered_n = int(fr.get("offered_tool_count")
                            or len(fr.get("tools") or [])
                            or sel.get("offered_tool_count") or 0)
            gf = (fr.get("declared") or {}).get("graph_features") or sel.get(
                "graph_features") or {}
            primary = (fr.get("declared") or {}).get("structural_pattern") or sel.get(
                "actual_primary_pattern") or ""
            boolean_label = fr.get("boolean_label")
            coding = bool(fr.get("coding_like") if "coding_like" in fr
                          else sel.get("coding_like"))
            prim_seq = "->".join(
                c.get("primitive_id") or c.get("name") or ""
                for c in fr["gold_calls"])
            cap_fams = fr.get("capability_families") or sel.get("capability_families") or []
        else:
            call_count = int(sel["call_count"])
            answer_type = sel.get("answer_type")
            offered_n = int(sel.get("offered_tool_count")
                            or (ver.get("offered") or {}).get("offered_tool_count")
                            or 0)
            gf = sel.get("graph_features") or {}
            primary = sel.get("actual_primary_pattern") or ""
            boolean_label = sel.get("boolean_label")
            coding = bool(sel.get("coding_like"))
            prim_seq = sel.get("normalized_capability_sequence") or ""
            cap_fams = sel.get("capability_families") or []

        join_n = int(gf.get("n_join_nodes") or gf.get("n_multi_parent_nodes") or 0)
        multi_join = join_n >= 2 or primary in {
            "MULTI_JOIN", "NESTED_AGGREGATION", "TWO_STAGE_AGGREGATION"}
        depth = int(gf.get("depth") or max(0, call_count - 1))
        dens = _ref_density(gf, call_count)

        mode = q.get("actual_mode") or ""
        if not mode:
            reject_reasons["missing_actual_mode"] += 1
            continue

        feat = {
            "task_id": tid,
            "call_count": call_count,
            "call_bucket": _bucket(call_count),
            "answer_type": answer_type,
            "query_mode": mode,
            "offered_tool_count": offered_n,
            "tool_band": tool_band(offered_n),
            "depth": depth,
            "depth_bucket": depth_bucket(depth),
            "join_count": join_n,
            "join_bucket": join_bucket(join_n),
            "reference_density": dens,
            "ref_band": ref_band(dens),
            "motif": map_motif(primary, join_n, multi_join),
            "primary_pattern": primary,
            "schema_complexity": "high" if offered_n >= 8 else "medium",
            "surface_track": track if track in ("A_NATIVE", "G_GENERAL_1") else "A_NATIVE",
            "workflow_id": sel.get("workflow_id"),
            "plan_id": sel.get("plan_id"),
            "program_fingerprint": sel.get("program_fingerprint"),
            "semantic_program_id": sel.get("semantic_program_id"),
            "intent_fingerprint": intent,
            "exact_fingerprint": exact,
            "primitive_sequence": prim_seq,
            "capability_families": list(cap_fams),
            "boolean_label": boolean_label,
            "coding_like": coding,
            "query_source": q.get("query_source"),
            "has_full_export": tid in full,
            "graph_features": gf,
            "v4_resolved": bool(v4.get("resolved") or sel.get("v4_search_complete")),
            "v4_shortcut": bool(v4.get("has_shortcut") or sel.get("v4_shortcut")),
            "necessity_ok": bool(sel.get("all_gold_nodes_necessary")),
            "difficulty_band": sel.get("difficulty_band") or "medium",
        }
        if exact:
            exact_fp_seen[exact] = tid
        candidates.append(feat)

    meta = {
        "n_query_hard_valid": len(queries),
        "n_candidates": len(candidates),
        "reject_reasons": dict(reject_reasons.most_common()),
        "blocked_task_ids": len(blocked["task_id"]),
        "blocked_workflows": len(blocked["workflow_id"]),
    }
    return candidates, meta


def pool_audit(candidates: Sequence[Dict[str, Any]], meta: Mapping[str, Any],
               out_dir: Path) -> Dict[str, Any]:
    from collections import Counter

    n = len(candidates)
    calls = Counter(c["call_bucket"] for c in candidates)
    answers = Counter(c["answer_type"] for c in candidates)
    modes = Counter(c["query_mode"] for c in candidates)
    motifs = Counter(c["motif"] for c in candidates)
    tools = Counter(c["tool_band"] for c in candidates)
    surfaces = Counter(c["surface_track"] for c in candidates)
    six = Counter(c["call_count"] for c in candidates if c["call_bucket"] == "6+")
    audit = {
        "n_candidates_eligible": n,
        "meta": dict(meta),
        "call_bucket": dict(calls),
        "call_count_6plus_detail": {str(k): v for k, v in sorted(six.items())},
        "answer_type": dict(answers),
        "query_mode": dict(modes),
        "motif": dict(motifs),
        "tool_band": dict(tools),
        "surface_track": dict(surfaces),
        "n_with_full_export": sum(1 for c in candidates if c["has_full_export"]),
        "n_coding": sum(1 for c in candidates if c["coding_like"]),
        "v4_shortcut_any": sum(1 for c in candidates if c["v4_shortcut"]),
        "feasibility_vs_hard_calls": {
            b: {"available": calls.get(b, 0), "need": need,
                "ok": calls.get(b, 0) >= need}
            for b, need in (("2", 330), ("3", 220), ("4", 135), ("5", 95), ("6+", 220))
        },
    }
    (out_dir / "NESTFUL_PROFILE_1000_POOL_AUDIT.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8")
    lines = [
        "# NESTFUL_PROFILE_1000 pool audit",
        "",
        f"- eligible candidates (after gates + heldout/reserve shield): **{n}**",
        f"- query_hard_valid source size: {meta.get('n_query_hard_valid')}",
        f"- with materialised full export: {audit['n_with_full_export']}",
        "",
        "## Call-count feasibility vs hard quotas",
        "",
        "| bucket | available | need | ok |",
        "| --- | ---: | ---: | --- |",
    ]
    for b, info in audit["feasibility_vs_hard_calls"].items():
        lines.append(
            f"| {b} | {info['available']} | {info['need']} | {info['ok']} |")
    lines += [
        "",
        "## Reject reasons (not eligible)",
        "",
    ]
    for k, v in (meta.get("reject_reasons") or {}).items():
        lines.append(f"- `{k}`: {v}")
    lines += ["", "## Marginal distributions (eligible)", ""]
    for title, dist in (("call_bucket", calls), ("answer_type", answers),
                        ("query_mode", modes), ("motif", motifs)):
        lines.append(f"### {title}")
        for k, v in sorted(dist.items(), key=lambda kv: (-kv[1], str(kv[0]))):
            lines.append(f"- {k}: {v}")
        lines.append("")
    (out_dir / "NESTFUL_PROFILE_1000_POOL_AUDIT.md").write_text(
        "\n".join(lines), encoding="utf-8")
    return audit
