"""Stages 9-13 of the Pilot4.3 plan: candidates, hard validation, necessity, V4.

The pipeline is deliberately staged and file-backed. Each stage reads the previous
stage's JSONL, writes its own, and can be resumed; nothing is held in memory at
50k scale and nothing is trusted across a stage boundary -- a candidate is always
re-derived from ``(workflow, plan, track, seed)`` and its program fingerprint is
re-checked (see :mod:`.tasks`).

Stage boundaries and what each one is allowed to conclude:

``generate``
    instantiate every (plan, track, seed) cell. Structure, semantic edges,
    execution, deterministic replay and value/type agreement are already enforced
    by :func:`.build.instantiate`; this stage adds value realism and the domain
    capability claim, and records derived structure only.
``validate``
    the semantic hard gates that need the whole pool: program-level duplicates,
    primitive-sequence concentration and per-plan caps.
``shortlist``
    pick the render pool with the call-count, answer-type and coding-share targets
    already in mind, so the expensive stages are not spent on tasks that could
    never be selected.
``verify``
    counterfactuals, per-node necessity, V4 for every answer type, and offered-tool
    construction with behaviourally validated distractors. This is the expensive
    stage and the only one that can declare a task selectable.
"""
from __future__ import annotations

import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Tuple

from . import (CALL_BUCKETS, RUN_ID, SURFACE_TRACKS)
from .blueprints import Blueprint, Plan, all_blueprints, assert_full_registry
from .build import BuildError, Instance, instantiate
from .counterfactuals import (as_fact_pairs, as_programs, counterfactual_instances)
from .distractors import build_offered_tools
from .necessity import all_nodes_necessary, necessity_summary, node_necessity
from .ops import build_ops
from .tasks import (CandidateId, candidate_row, cell_id, domain_claim_satisfied,
                    program_fingerprint, rebuild, value_realism_flags)
from .v4 import V4Config, v4_gate

CANDIDATES = "semantic_candidates.jsonl"
HARD_VALID = "semantic_hard_valid.jsonl"
SHORTLIST = "query_render_shortlist.jsonl"
VERIFIED = "verified_candidates.jsonl"
NECESSITY = "per_node_necessity.jsonl"
V4_ROWS = "v4_per_task.jsonl"
LEDGER = "per_task_validation_ledger.jsonl"
REJECTS = "rejected_candidates.jsonl"


# ── small IO helpers ─────────────────────────────────────────────────────
def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]], *,
                append: bool = False) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("a" if append else "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, default=_json_default))
            fh.write("\n")
            n += 1
    return n


def _json_default(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


# ── stage 1: candidate generation ────────────────────────────────────────
@dataclass(frozen=True)
class GenSpec:
    workflow_id: str
    plan_id: str
    track: str
    seed: int


def _plan_cells(seed0: int, target: int) -> List[GenSpec]:
    """Round-robin over (plan, track) so no family dominates the pool.

    Tracks are cycled per plan rather than multiplied out, because a surface track
    only renames tools: three tracks over the same seed would be three near-copies
    of one task, which is how Pilot4.2 inflated its candidate count.
    """
    bps = all_blueprints()
    cells: List[Tuple[str, str, Tuple[str, ...]]] = []
    for bp in bps:
        tracks = tuple(t for t in bp.surface_compatibility if t in SURFACE_TRACKS)
        for plan in bp.plans:
            cells.append((bp.workflow_id, plan.plan_id, tracks or SURFACE_TRACKS))
    if not cells:
        return []
    per_cell = max(1, target // len(cells) + 1)
    specs: List[GenSpec] = []
    for i in range(per_cell):
        for j, (wid, pid, tracks) in enumerate(cells):
            track = tracks[(i + j) % len(tracks)]
            specs.append(GenSpec(wid, pid, track, seed0 + 7919 * i + 104729 * j))
            if len(specs) >= target:
                return specs
    return specs


def _generate_chunk(specs: Sequence[GenSpec]) -> Tuple[List[Dict[str, Any]],
                                                       List[Dict[str, Any]]]:
    """Worker: instantiate a chunk and apply the per-candidate hard gates."""
    from .blueprints import by_id

    rows: List[Dict[str, Any]] = []
    rejects: List[Dict[str, Any]] = []
    for spec in specs:
        try:
            bp = by_id(spec.workflow_id)
            plan = next(p for p in bp.plans if p.plan_id == spec.plan_id)
        except Exception as exc:                      # noqa: BLE001
            rejects.append({"spec": spec.__dict__, "reason": f"lookup: {exc}"})
            continue
        try:
            inst = instantiate(bp, plan, spec.seed, track=spec.track)
        except BuildError as exc:
            rejects.append({"spec": spec.__dict__, "stage": "instantiate",
                            "reason": str(exc)[:160]})
            continue
        cid = CandidateId(spec.workflow_id, spec.plan_id, spec.track, spec.seed)
        row = candidate_row(inst, cid)
        flags = value_realism_flags(inst, plan)
        if flags:
            rejects.append({"task_id": row["task_id"], "stage": "value_realism",
                            "reason": "; ".join(flags[:3])})
            continue
        if not domain_claim_satisfied(bp, row["capability_families"]):
            rejects.append({"task_id": row["task_id"], "stage": "domain_claim",
                            "reason": f"{bp.domain} not backed by "
                                      f"{row['capability_families']}"})
            continue
        if row["call_count"] != plan.call_count:
            rejects.append({"task_id": row["task_id"], "stage": "call_count",
                            "reason": "plan/program call count disagreement"})
            continue
        rows.append(row)
    return rows, rejects


def generate(out_dir: Path, *, target: int, seed: int = 20260731,
             workers: int = 0, chunk: int = 250) -> Dict[str, Any]:
    assert_full_registry()
    specs = _plan_cells(seed, target)
    chunks = [specs[i:i + chunk] for i in range(0, len(specs), chunk)]
    path = out_dir / CANDIDATES
    rej_path = out_dir / REJECTS
    if path.exists():
        path.unlink()
    if rej_path.exists():
        rej_path.unlink()
    n_ok = 0
    n_rej = 0
    t0 = time.perf_counter()
    for rows, rejects in _map_chunks(_generate_chunk, chunks, workers):
        n_ok += write_jsonl(path, rows, append=True)
        n_rej += write_jsonl(rej_path, rejects, append=True)
    return {
        "stage": "generate", "run_id": RUN_ID, "requested": target,
        "specs": len(specs), "candidates": n_ok, "rejected": n_rej,
        "seconds": round(time.perf_counter() - t0, 1),
        "workers": workers or os.cpu_count() or 1,
        "output": str(path),
    }


def _map_chunks(fn, chunks: Sequence[Sequence[Any]], workers: int):
    """Chunked map that degrades to serial execution when workers <= 1."""
    if workers == 1 or len(chunks) <= 1:
        for ch in chunks:
            yield fn(ch)
        return
    n = workers or (os.cpu_count() or 2)
    with ProcessPoolExecutor(max_workers=n) as pool:
        futures = [pool.submit(fn, ch) for ch in chunks]
        for fut in as_completed(futures):
            yield fut.result()


# ── stage 2: pool-level hard validation ──────────────────────────────────
#: Concentration ceilings from the spec. They are applied to the *hard-valid pool*
#: so the later selection has room to hit them exactly in the exported set.
MAX_EXACT_SEQUENCE_SHARE = 0.03
MAX_NORMALIZED_SEQUENCE_SHARE = 0.05
MAX_TOP10_SEQUENCE_SHARE = 0.30


def validate_pool(out_dir: Path) -> Dict[str, Any]:
    """Deduplicate and cap concentration; write ``semantic_hard_valid.jsonl``."""
    src = out_dir / CANDIDATES
    rows = read_jsonl(src)
    seen_fp: set[str] = set()
    seen_wi: set[str] = set()
    kept: List[Dict[str, Any]] = []
    rejects: List[Dict[str, Any]] = []
    for row in rows:
        if row["program_fingerprint"] in seen_fp:
            rejects.append({"task_id": row["task_id"], "stage": "duplicate",
                            "reason": "identical program and values"})
            continue
        if row["workflow_instance_id"] in seen_wi:
            rejects.append({"task_id": row["task_id"], "stage": "duplicate",
                            "reason": "identical workflow instance"})
            continue
        seen_fp.add(row["program_fingerprint"])
        seen_wi.add(row["workflow_instance_id"])
        kept.append(row)

    kept, seq_report = _cap_sequences(kept, rejects)
    write_jsonl(out_dir / HARD_VALID, kept)
    write_jsonl(out_dir / REJECTS, rejects, append=True)
    return {
        "stage": "validate", "input": len(rows), "hard_valid": len(kept),
        "rejected": len(rows) - len(kept),
        "sequence_concentration": seq_report,
        "distinct_exact_sequences": len({r["primitive_sequence"] for r in kept}),
        "distinct_normalized_sequences":
            len({r["normalized_capability_sequence"] for r in kept}),
        "output": str(out_dir / HARD_VALID),
    }


def _cap_sequences(rows: List[Dict[str, Any]],
                   rejects: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]],
                                                           Dict[str, Any]]:
    """Drop the tail of over-represented primitive sequences.

    Capping happens here rather than at selection time so that the pool the
    selector sees can satisfy the concentration gates without having to weaken any
    other quota. The cap is a share of the *kept* pool, computed iteratively.
    """
    by_exact: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_exact.setdefault(row["primitive_sequence"], []).append(row)
    total = len(rows)
    cap = max(4, int(MAX_EXACT_SEQUENCE_SHARE * total))
    kept: List[Dict[str, Any]] = []
    for seq, group in sorted(by_exact.items()):
        for row in group[:cap]:
            kept.append(row)
        for row in group[cap:]:
            rejects.append({"task_id": row["task_id"], "stage": "concentration",
                            "reason": f"exact sequence over cap ({len(group)})"})
    kept.sort(key=lambda r: r["task_id"])
    return kept, {
        "exact_cap_per_sequence": cap,
        "max_exact_share": round(max((len([r for r in kept
                                           if r["primitive_sequence"] == s])
                                      for s in {r["primitive_sequence"]
                                                for r in kept}), default=0)
                                 / max(1, len(kept)), 5),
        "max_normalized_share": _max_share(kept,
                                           "normalized_capability_sequence"),
        "top10_exact_share": _topk_share(kept, "primitive_sequence", 10),
    }


def _max_share(rows: Sequence[Dict[str, Any]], key: str) -> float:
    counts: Dict[str, int] = {}
    for row in rows:
        counts[row[key]] = counts.get(row[key], 0) + 1
    return round(max(counts.values(), default=0) / max(1, len(rows)), 5)


def _topk_share(rows: Sequence[Dict[str, Any]], key: str, k: int) -> float:
    counts: Dict[str, int] = {}
    for row in rows:
        counts[row[key]] = counts.get(row[key], 0) + 1
    top = sorted(counts.values(), reverse=True)[:k]
    return round(sum(top) / max(1, len(rows)), 5)


# ── stage 3: render shortlist ────────────────────────────────────────────
def shortlist(out_dir: Path, *, target: int, seed: int = 5150,
              profile: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Pick the pool that will be rendered and verified.

    The shortlist is stratified by call bucket, answer type, coding share and
    workflow family so that the expensive verification stage produces a pool the
    tier quotas can actually be met from. It intentionally over-provisions the
    long tail (6+ calls, coding, non-numeric answers), because those are the
    strata Pilot4.2 ran out of.
    """
    rows = read_jsonl(out_dir / HARD_VALID)
    rng = random.Random(seed)
    rng.shuffle(rows)
    want = _shortlist_targets(target)
    picked: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {k: 0 for k in want}
    per_plan: Dict[str, int] = {}
    plan_cap = max(6, int(target / max(1, len({(r['workflow_id'], r['plan_id'])
                                               for r in rows})) * 3))
    for row in rows:
        key = _stratum(row)
        if counts.get(key, 0) >= want.get(key, 0):
            continue
        pk = f"{row['workflow_id']}/{row['plan_id']}"
        if per_plan.get(pk, 0) >= plan_cap:
            continue
        counts[key] = counts.get(key, 0) + 1
        per_plan[pk] = per_plan.get(pk, 0) + 1
        picked.append(row)
    # top up with anything still unused, keeping family balance
    if len(picked) < target:
        taken = {r["task_id"] for r in picked}
        for row in rows:
            if len(picked) >= target:
                break
            if row["task_id"] in taken:
                continue
            pk = f"{row['workflow_id']}/{row['plan_id']}"
            if per_plan.get(pk, 0) >= plan_cap + 4:
                continue
            per_plan[pk] = per_plan.get(pk, 0) + 1
            picked.append(row)
    picked.sort(key=lambda r: r["task_id"])
    write_jsonl(out_dir / SHORTLIST, picked)
    return {
        "stage": "shortlist", "requested": target, "picked": len(picked),
        "call_buckets": _hist(picked, "call_bucket"),
        "answer_types": _hist(picked, "answer_type"),
        "coding_share": round(sum(1 for r in picked if r["coding_like"])
                              / max(1, len(picked)), 4),
        "workflows": len({r["workflow_id"] for r in picked}),
        "plans": len({(r["workflow_id"], r["plan_id"]) for r in picked}),
        "output": str(out_dir / SHORTLIST),
    }


def _stratum(row: Dict[str, Any]) -> str:
    return (f"{row['call_bucket']}|{row['answer_type']}|"
            f"{'coding' if row['coding_like'] else 'plain'}")


def _shortlist_targets(total: int) -> Dict[str, int]:
    """Per-stratum ceilings for the render pool.

    Shares are enrichment-oriented, not profile-matched: the profile match happens
    at selection, and it can only match downwards, so every stratum the profile
    needs must be over-supplied here.
    """
    bucket_share = {"2": 0.20, "3": 0.16, "4": 0.14, "5": 0.14, "6+": 0.36}
    answer_share = {"float": 0.34, "boolean": 0.20, "integer": 0.08,
                    "string": 0.18, "list": 0.07, "object": 0.07,
                    "category": 0.06}
    out: Dict[str, int] = {}
    for bucket, bshare in bucket_share.items():
        for atype, ashare in answer_share.items():
            for coding, cshare in (("coding", 0.45), ("plain", 0.55)):
                n = int(round(total * bshare * ashare * cshare))
                out[f"{bucket}|{atype}|{coding}"] = max(2, n)
    return out


def _hist(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for row in rows:
        out[str(row[key])] = out.get(str(row[key]), 0) + 1
    return dict(sorted(out.items()))


# ── stage 4: verification (counterfactuals, necessity, V4, distractors) ──
@dataclass(frozen=True)
class VerifyConfig:
    offered_target_min: int = 6
    offered_target_max: int = 12
    v4_depth_cap: int = 3
    min_hard_distractors: int = 1


def _verify_chunk(rows: Sequence[Dict[str, Any]]
                  ) -> List[Dict[str, Any]]:
    """Worker: run the expensive per-task gates. Never raises for one bad task."""
    cfg = VerifyConfig()
    out: List[Dict[str, Any]] = []
    for row in rows:
        t0 = time.perf_counter()
        try:
            out.append(_verify_one(row, cfg))
        except Exception as exc:                       # noqa: BLE001
            out.append({"task_id": row["task_id"], "selectable": False,
                        "reject_stage": "verify_error",
                        "reject_reason": f"{type(exc).__name__}: {exc}"[:200],
                        "seconds": round(time.perf_counter() - t0, 2)})
    return out


def _verify_one(row: Dict[str, Any], cfg: VerifyConfig) -> Dict[str, Any]:
    t0 = time.perf_counter()
    inst, bp, plan = rebuild(row)
    rng = random.Random(f"verify:{row['task_id']}")

    cf_insts, cf_meta = counterfactual_instances(
        bp, plan, answer_type=inst.answer_type, track=inst.track,
        seed=int(row["seed"]) + 31337)
    cf_progs = as_programs(cf_insts)
    cf_pairs = as_fact_pairs(cf_insts)

    target = rng.randint(cfg.offered_target_min, cfg.offered_target_max)
    offered = build_offered_tools(inst.program, inst.answer, track=inst.track,
                                  target_count=max(target, inst.call_count + 2),
                                  seed=int(row["seed"]),
                                  counterfactuals=cf_progs[:6],
                                  min_hard=cfg.min_hard_distractors)
    offered_ids = [t["primitive_id"] for t in offered["tools"]]

    nec = node_necessity(inst.program, allowed_ops=offered_ids,
                         check_alternatives=True, counterfactuals=cf_progs)
    nec_ok = all_nodes_necessary(nec)

    facts = {r.name: (inst.role_values[r.name], r.sem) for r in plan.roles}
    gate = v4_gate(facts, offered_ids, inst.answer, inst.call_count, cf_pairs,
                   cfg=V4Config(depth_cap=cfg.v4_depth_cap),
                   counterfactuals_mixed=cf_meta["mixed"])

    reasons: List[str] = []
    if cf_meta["weak"]:
        reasons.append("counterfactual set too weak to decide equivalence")
    if not nec_ok:
        reasons.append("unnecessary gold node: "
                       + ",".join(necessity_summary(nec)["unnecessary_nodes"]))
    if not gate["v4_executed"]:
        reasons.append("v4 not executed")
    if gate["has_shortcut"]:
        reasons.append("v4 shortcut: "
                       + (gate["confirmed_shortcuts"][0]["rendered"][:80]
                          if gate["confirmed_shortcuts"] else "confirmed"))
    if not gate["resolved"]:
        reasons.append("v4 unresolved")
    if offered["distractor_count"] < 1:
        reasons.append("no validated distractor")

    return {
        "task_id": row["task_id"],
        "selectable": not reasons,
        "reject_stage": "" if not reasons else "verify",
        "reject_reason": "; ".join(reasons)[:300],
        "counterfactuals": cf_meta,
        "necessity": nec,
        "necessity_summary": necessity_summary(nec),
        "v4": gate,
        "offered": {k: v for k, v in offered.items() if k != "tools"},
        "offered_tools": offered["tools"],
        "verifier": _verifier_spec(inst, gate, nec, offered_ids),
        "seconds": round(time.perf_counter() - t0, 2),
    }


def _verifier_spec(inst: Instance, gate: Dict[str, Any],
                   nec: Sequence[Dict[str, Any]],
                   offered_ids: Sequence[str]) -> Dict[str, Any]:
    """What counts as a correct solution, beyond an exact trace match."""
    alt_nodes = [r["node_id"] for r in nec if r["alternative_binding_found"]]
    return {
        "canonical_program": [
            {"node_id": nd.node_id, "primitive_id": nd.op,
             "args": {k: (f"${v.node_id}$" if hasattr(v, "node_id") else v)
                      for k, v in nd.args.items()}}
            for nd in inst.program.nodes],
        "sink": inst.program.sink,
        "minimal_valid_call_count": gate["minimal_valid_call_count"],
        "gold_call_count": inst.call_count,
        "terminal_state_verifier": {
            "type": "answer_equivalence",
            "answer": inst.answer,
            "answer_type": inst.answer_type,
            "tolerance": 1e-6 if inst.answer_type in ("float",) else 0,
        },
        "answer_grounding_verifier": {
            "type": "sink_value_matches_answer",
            "sink_node": inst.program.sink,
        },
        "accepted_solution_classes": (
            ["canonical"] if not alt_nodes else
            ["canonical", "single_node_substitution"]),
        "alternative_binding_nodes": alt_nodes,
        "strict_trace_required": not alt_nodes,
    }


def verify(out_dir: Path, *, workers: int = 0, chunk: int = 24,
           limit: int = 0, resume: bool = True) -> Dict[str, Any]:
    rows = read_jsonl(out_dir / SHORTLIST)
    if limit:
        rows = rows[:limit]
    done: set[str] = set()
    ver_path = out_dir / VERIFIED
    if resume and ver_path.exists():
        done = {r["task_id"] for r in iter_jsonl(ver_path)}
    todo = [r for r in rows if r["task_id"] not in done]
    chunks = [todo[i:i + chunk] for i in range(0, len(todo), chunk)]
    t0 = time.perf_counter()
    n = 0
    stats = {"selectable": 0, "rejected": 0}
    reasons: Dict[str, int] = {}
    for results in _map_chunks(_verify_chunk, chunks, workers):
        n += write_jsonl(ver_path, results, append=True)
        for r in results:
            if r.get("selectable"):
                stats["selectable"] += 1
            else:
                stats["rejected"] += 1
                key = (r.get("reject_reason") or "unknown").split(":")[0][:60]
                reasons[key] = reasons.get(key, 0) + 1
    return {
        "stage": "verify", "shortlist": len(rows), "already_done": len(done),
        "verified_now": n, **stats,
        "reject_reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
        "seconds": round(time.perf_counter() - t0, 1),
        "output": str(ver_path),
    }


def write_verification_artifacts(out_dir: Path) -> Dict[str, Any]:
    """Split the verified stream into the per-task artifacts the spec names."""
    ver = out_dir / VERIFIED
    nec_rows: List[Dict[str, Any]] = []
    v4_rows: List[Dict[str, Any]] = []
    ledger: List[Dict[str, Any]] = []
    n_nodes = 0
    n_necessary = 0
    for rec in iter_jsonl(ver):
        tid = rec["task_id"]
        for row in rec.get("necessity", []):
            nec_rows.append({"task_id": tid, **row})
            n_nodes += 1
            n_necessary += int(bool(row["necessary"]))
        gate = rec.get("v4")
        if gate:
            v4_rows.append({"task_id": tid, **gate})
        ledger.append({
            "task_id": tid,
            "selectable": rec.get("selectable", False),
            "reject_stage": rec.get("reject_stage", ""),
            "reject_reason": rec.get("reject_reason", ""),
            "v4_executed": bool(gate and gate.get("v4_executed")),
            "v4_resolved": bool(gate and gate.get("resolved")),
            "v4_has_shortcut": bool(gate and gate.get("has_shortcut")),
            "v4_max_depth_complete": (gate or {}).get(
                "search_space", {}).get("max_depth_complete"),
            "node_necessity_coverage": len(rec.get("necessity", [])),
            "all_nodes_necessary": (rec.get("necessity_summary", {})
                                    .get("all_necessary")),
            "counterfactuals_built": (rec.get("counterfactuals", {})
                                      .get("built")),
            "counterfactuals_mixed": (rec.get("counterfactuals", {})
                                      .get("mixed")),
            "offered_tool_count": (rec.get("offered", {})
                                   .get("offered_tool_count")),
            "distractor_count": (rec.get("offered", {}).get("distractor_count")),
            "seconds": rec.get("seconds"),
        })
    write_jsonl(out_dir / NECESSITY, nec_rows)
    write_jsonl(out_dir / V4_ROWS, v4_rows)
    write_jsonl(out_dir / LEDGER, ledger)
    return {
        "per_node_rows": len(nec_rows),
        "nodes_checked": n_nodes,
        "nodes_necessary": n_necessary,
        "v4_rows": len(v4_rows),
        "v4_executed": sum(1 for r in v4_rows if r.get("v4_executed")),
        "v4_resolved": sum(1 for r in v4_rows if r.get("resolved")),
        "v4_shortcuts": sum(1 for r in v4_rows if r.get("has_shortcut")),
        "ledger_rows": len(ledger),
        "selectable": sum(1 for r in ledger if r["selectable"]),
    }
