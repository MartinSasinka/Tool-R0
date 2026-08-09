"""Stages 14-23: query contracts, rendering, hard validation, hard-valid pool.

Mode assignment happens *before* rendering (from the TargetProfile), rendering is
routed by mode, and then the mode is classified again from the text alone. Only the
classified mode is written to the dataset and used by selection: Pilot4.2 quoted
requested labels, which is why its "5 % graph-explicit" claim was wrong.

Routing:

* ``GRAPH_EXPLICIT`` and ``OPERATION_EXPLICIT_GRAPH_IMPLICIT`` are deterministic by
  design -- they are supposed to name operations, and a writer model adds nothing.
* ``DOMAIN_GROUNDED_IMPLICIT`` and ``GOAL_BASED_IMPLICIT`` require the OpenRouter
  writer for every selected task. When no API key is available the tasks are still
  rendered deterministically so the dataset exists, but each record carries
  ``query_source="deterministic_fallback"`` and the run-level ``LLM_VALIDATED``
  status stays false. Nothing silently pretends a model wrote the text.
* ``SEMI_IMPLICIT`` is split: the writer takes at least half when available.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from . import QUERY_MODES
from .blueprints import Blueprint, Plan
from .pipeline import (SHORTLIST, VERIFIED, iter_jsonl, read_jsonl, write_jsonl)
from .determinability import NOT_STATABLE
from .program import gold_calls
from .qvalidate import (check_diversity, contract_payload, diversity_report,
                        fingerprints, validate_query)
from .queries import build_contract, render_deterministic
from .tasks import rebuild

#: written by :mod:`.orrun` -- one record per OpenRouter-written query, with its
#: critic verdicts. Deterministic renders live in their own file so the two sources
#: can never be confused for each other.
RENDERED = "llm_rendered.jsonl"
DETERMINISTIC = "deterministic_rendered.jsonl"
QUERY_VALID = "query_hard_valid.jsonl"
QUERY_REJECTS = "query_rejected.jsonl"

DETERMINISTIC_MODES = ("GRAPH_EXPLICIT", "OPERATION_EXPLICIT_GRAPH_IMPLICIT")
WRITER_MODES = ("DOMAIN_GROUNDED_IMPLICIT", "GOAL_BASED_IMPLICIT")
#: fraction of SEMI_IMPLICIT that must come from the writer when it is available
SEMI_WRITER_SHARE = 0.5


def contract_seed(task_id: str, seed: int) -> int:
    """Per-task contract seed, stable across processes.

    ``hash()`` on a str is salted per interpreter, so a contract built in the
    rendering process drew different entities and fact wording than the same
    contract rebuilt later for validation or audit -- the query was scored against
    facts it was never given. Digest the id instead.
    """
    digest = hashlib.blake2b(task_id.encode("utf-8"), digest_size=8).digest()
    return seed + int.from_bytes(digest, "big") % 9973


#: Modes in which the query may state the rule behind a computed criterion or a
#: composite answer shape. A task that needs a rule is routed here; asking for it
#: in a fully implicit mode produces a query nobody could answer, which is what
#: the first critic rejected in the smoke stage.
RULE_MODES = ("SEMI_IMPLICIT", "OPERATION_EXPLICIT_GRAPH_IMPLICIT",
              "GRAPH_EXPLICIT")


def assign_modes(rows: Sequence[Dict[str, Any]], targets: Dict[str, float],
                 seed: int = 4242,
                 needs_rule: Dict[str, bool] | None = None) -> Dict[str, str]:
    """Requested mode per task, in the profile's proportions.

    Assignment is stratified by call bucket so the mode mix does not correlate with
    task length: in Pilot4.2 the long tasks were all explicit, which made the
    "implicit share" number meaningless for exactly the tasks that mattered.

    Within a bucket, tasks that need a rule stated take the rule-bearing modes
    first and the fully implicit quota is filled from the rest. When a bucket holds
    more rule-bearing tasks than those modes can absorb, the surplus is left
    without a mode: such a task is dropped from rendering rather than asked for in
    a mode that cannot express it.
    """
    rng = random.Random(seed)
    needs_rule = needs_rule or {}
    by_bucket: Dict[str, List[str]] = {}
    for row in rows:
        by_bucket.setdefault(row["call_bucket"], []).append(row["task_id"])
    out: Dict[str, str] = {}
    modes = [m for m in QUERY_MODES if targets.get(m, 0) > 0]
    weights = [targets[m] for m in modes]
    for bucket, ids in sorted(by_bucket.items()):
        ids = sorted(ids)
        rng.shuffle(ids)
        quota = _integer_quota(len(ids), modes, weights)
        with_rule = [t for t in ids if needs_rule.get(t)]
        plain = [t for t in ids if not needs_rule.get(t)]
        for mode in modes:
            want = quota[mode]
            take: List[str] = []
            if mode in RULE_MODES:
                take, with_rule = with_rule[:want], with_rule[want:]
            room = want - len(take)
            if room > 0:
                take, plain = take + plain[:room], plain[room:]
            for task_id in take:
                out[task_id] = mode
        for task_id in plain:                   # rounding remainder
            out[task_id] = modes[0]
        for task_id in with_rule:
            out[task_id] = ""                   # no mode can express this task
    return out


def _integer_quota(n: int, keys: Sequence[str],
                   weights: Sequence[float]) -> Dict[str, int]:
    total = sum(weights) or 1.0
    raw = {k: n * w / total for k, w in zip(keys, weights)}
    quota = {k: int(v) for k, v in raw.items()}
    left = n - sum(quota.values())
    for k in sorted(raw, key=lambda k: -(raw[k] - quota[k])):
        if left <= 0:
            break
        quota[k] += 1
        left -= 1
    return quota


def _payload(inst, bp: Blueprint, plan: Plan, contract) -> Dict[str, Any]:
    calls = gold_calls(inst.program, inst.track)
    predicate_steps = sum(1 for s in plan.steps
                          if s.capability.split(".")[0] in
                          ("comparison", "boolean", "decision", "classification"))
    return contract_payload(contract, answer=inst.answer,
                            gold_capabilities=[c["capability"] for c in calls],
                            predicate_steps=predicate_steps)


def render_task(row: Dict[str, Any], mode: str, *, seed: int,
                writer: Any = None) -> Dict[str, Any]:
    """Render and validate one query; returns the record either way."""
    inst, bp, plan = rebuild(row)
    contract = build_contract(inst, bp, plan, mode=mode,
                             task_id=row["task_id"], seed=seed)
    if contract.determinability == NOT_STATABLE:
        return {"task_id": row["task_id"], "requested_mode": mode,
                "dropped": "rule_not_statable"}
    payload = _payload(inst, bp, plan, contract)
    payload["mode"] = mode

    source = "deterministic"
    renderer = ""
    attempts: List[Dict[str, Any]] = []
    query = ""
    if writer is not None and mode in WRITER_MODES + ("SEMI_IMPLICIT",):
        got = writer(contract, payload, mode)
        if got and got.get("query"):
            query = got["query"]
            source = "openrouter"
            renderer = got.get("model", "openrouter")
            attempts = got.get("attempts", [])
    if not query:
        det = render_deterministic(contract, mode, seed=seed)
        query = det["query"]
        renderer = det.get("renderer", "deterministic")
        if mode in WRITER_MODES:
            source = "deterministic_fallback"
    verdict = validate_query(query, payload)
    return {
        "task_id": row["task_id"],
        "requested_mode": mode,
        "query": query,
        "query_source": source,
        "renderer": renderer,
        "actual_mode": verdict["classification"]["actual_query_mode"],
        "passed": verdict["passed"],
        "failed_layers": verdict["failed_layers"],
        "classification": verdict["classification"],
        "layers": {k: v for k, v in verdict["layers"].items() if not v["passed"]},
        "fingerprints": fingerprints(query),
        "writer_attempts": attempts,
        "contract": {
            "target_phrase": contract.target_phrase,
            "expected_numbers": payload["expected_numbers"],
            "expected_units": payload["expected_units"],
            "entities": payload["entities"],
        },
    }


def mode_targets_from(profile: Dict[str, Any]) -> Dict[str, float]:
    targets = {m: float(v) for m, v in
               (profile.get("query_mode", {}).get("overall") or {}).items()}
    return targets or {"DOMAIN_GROUNDED_IMPLICIT": 0.50,
                       "GOAL_BASED_IMPLICIT": 0.22,
                       "SEMI_IMPLICIT": 0.18,
                       "OPERATION_EXPLICIT_GRAPH_IMPLICIT": 0.08,
                       "GRAPH_EXPLICIT": 0.02}


def selectable_rows(out_dir: Path, limit: int = 0) -> List[Dict[str, Any]]:
    """Shortlist rows that survived verification, in task-id order."""
    selectable = {r["task_id"] for r in iter_jsonl(out_dir / VERIFIED)
                  if r.get("selectable")}
    rows = [r for r in read_jsonl(out_dir / SHORTLIST)
            if r["task_id"] in selectable]
    return rows[:limit] if limit else rows


def needs_rule_map(rows: Sequence[Dict[str, Any]]) -> Dict[str, bool]:
    """Per task: does an answerable query have to state a rule? (spec 17)"""
    from .determinability import needs_rule_plan_ids

    by_plan = needs_rule_plan_ids()
    return {r["task_id"]: by_plan.get((r["workflow_id"], r["plan_id"]), False)
            for r in rows}


def render_pool(out_dir: Path, *, profile: Dict[str, Any], seed: int = 4242,
                writer: Any = None, limit: int = 0,
                resume: bool = True) -> Dict[str, Any]:
    """Render every verified-selectable task, validate, and split the pool.

    Queries written by the OpenRouter stage runner are read back from its log; the
    rest are rendered here. A writer-mode task with no LLM record is rendered
    deterministically and marked ``deterministic_fallback`` rather than dropped, so
    the dataset exists on a machine with no API key -- and the run-level
    ``LLM_VALIDATED`` status is what stays false.
    """
    rows = selectable_rows(out_dir, limit)
    targets = mode_targets_from(profile)
    modes = assign_modes(rows, targets, seed=seed,
                         needs_rule=needs_rule_map(rows))
    llm = llm_records(out_dir)

    done: set[str] = set()
    path = out_dir / DETERMINISTIC
    if path.exists():
        if resume:
            done = {r["task_id"] for r in iter_jsonl(path)}
        else:
            path.unlink()      # appending onto a previous run would double the pool
    written = unroutable = not_statable = 0
    for row in rows:
        tid = row["task_id"]
        if tid in done or tid in llm:
            continue
        if not modes[tid]:
            unroutable += 1    # needs a rule stated and no rule-bearing mode left
            continue
        rec = render_task(row, modes[tid],
                          seed=contract_seed(tid, seed), writer=writer)
        if rec.get("dropped"):
            not_statable += 1
            continue
        written += write_jsonl(path, [rec], append=True)

    report = finalise_pool(out_dir, expected=len(rows) - unroutable - not_statable,
                           rendered_now=written, mode_targets=targets)
    return {**report, "unroutable_rule_tasks": unroutable,
            "dropped_rule_not_statable": not_statable}


def build_render_tasks(out_dir: Path, *, profile: Dict[str, Any],
                       seed: int = 4242, limit: int = 0,
                       modes: Sequence[str] = ("DOMAIN_GROUNDED_IMPLICIT",
                                               "GOAL_BASED_IMPLICIT",
                                               "SEMI_IMPLICIT"),
                       tiers: Dict[str, str] | None = None,
                       mode_overrides: Mapping[str, str] | None = None,
                       ) -> List[Dict[str, Any]]:
    """Payloads for :mod:`.orrun`: writer contract, validator contract, critic view.

    The writer payload deliberately contains no program and no node ids, while the
    critic payload contains the executed oracle. Building both here, from one
    rebuild of the instance, is what keeps them consistent without letting the
    writer see the graph.

    ``mode_overrides`` (task_id → mode) wins over the profile mixer. The render
    allocation plan uses this so OpenRouter spend follows the planned channel
    rather than a second, independent mode draw.
    """
    rows = selectable_rows(out_dir, limit)
    assigned = assign_modes(rows, mode_targets_from(profile), seed=seed,
                            needs_rule=needs_rule_map(rows))
    if mode_overrides:
        assigned = {**assigned, **dict(mode_overrides)}
    tiers = tiers or {}
    out: List[Dict[str, Any]] = []
    for row in rows:
        tid = row["task_id"]
        mode = assigned.get(tid) or ""
        if mode not in modes:
            continue
        inst, bp, plan = rebuild(row)
        contract = build_contract(inst, bp, plan, mode=mode, task_id=tid,
                                  seed=contract_seed(tid, seed))
        if contract.determinability == NOT_STATABLE:
            continue           # no readable rule exists, so no answerable query
        payload = _payload(inst, bp, plan, contract)
        payload["mode"] = mode
        calls = gold_calls(inst.program, inst.track, inst.observations)
        out.append({
            "task_id": tid,
            "requested_mode": mode,
            "workflow_id": row["workflow_id"],
            "semantic_program_id": row["semantic_program_id"],
            "tier": tiers.get(tid, "PROFILE_CORE"),
            "call_count": int(row["call_count"]),
            "coding_call_count": sum(1 for c in calls if c["coding_like"]),
            "answer_type": row["answer_type"],
            "prompt_contract": {**contract.as_payload(), "mode": mode},
            "validator_contract": payload,
            "critic_context": {
                "workflow_goal": bp.natural_user_goal,
                "target": contract.target_phrase,
                "specification": list(contract.specification),
                "answer_type": inst.answer_type,
                "program": [{"node_id": c["node_id"], "tool": c["name"],
                             "arguments": c["arguments"],
                             "capability": c["capability"]} for c in calls],
                "edges": [[a, b] for a, b in inst.program.edges()],
                "node_purposes": {c["node_id"]: c["capability"] for c in calls},
                "input_facts": [{"role": f.role, "description": f.description,
                                 "value": f.value, "unit": f.unit}
                                for f in contract.facts],
                "observations": {c["node_id"]: c["observation"] for c in calls},
                "answer": inst.answer,
            },
        })
    return out


def writer_view_is_safe(view: Dict[str, Any]) -> bool:
    """No key that would disclose the program, its length or the answer.

    The writer's view is built from the contract rather than filtered out of the
    validator payload, because filtering by key name is only as good as the key
    names: ``expected_numbers`` looked harmless and was in fact an unpaired bag of
    values, which is how a writer came to describe five readings as a decimal
    count.
    """
    forbidden = {"gold_capabilities", "predicate_steps", "answer_rendered",
                 "call_count", "expected_numbers", "hidden_plan"}
    return not (forbidden & set(view))


def llm_records(out_dir: Path) -> Dict[str, Dict[str, Any]]:
    """OpenRouter-written queries, keyed by task id, in the qstage record shape.

    Only records the LLM layer did not block are returned: a blocked record means a
    critic refused it or a deterministic layer failed, and such a query must not
    reach selection through a side door.

    A missing log is not an error: on a machine with no API key every task is
    rendered deterministically and ``LLM_VALIDATED`` stays false, which is the
    documented degraded path rather than a failure to produce a dataset.
    """
    out: Dict[str, Dict[str, Any]] = {}
    if not (out_dir / RENDERED).is_file():
        return out
    for rec in iter_jsonl(out_dir / RENDERED):
        if rec.get("blocked") or not rec.get("query"):
            continue
        validation = rec.get("validation") or {}
        classification = validation.get("classification") or {}
        query = rec["query"]
        out[rec["task_id"]] = {
            "task_id": rec["task_id"],
            "requested_mode": rec.get("requested_mode", ""),
            "query": query,
            "query_source": "openrouter",
            "renderer": rec.get("model", "openrouter"),
            "actual_mode": classification.get("actual_query_mode", ""),
            "passed": bool(validation.get("passed")),
            "failed_layers": validation.get("failed_layers", []),
            "classification": classification,
            "layers": {k: v for k, v in (validation.get("layers") or {}).items()
                       if not v.get("passed")},
            "fingerprints": fingerprints(query),
            "writer_attempts": rec.get("attempts", 0),
            "critic": {"executed": rec.get("critic") is not None,
                       "verdict": (rec.get("critic") or {}).get("verdict"),
                       "findings": rec.get("critic"),
                       "rewrites": rec.get("rewrite_history", [])},
            "critic2": {"executed": rec.get("second_critic") is not None,
                        "routed": bool(rec.get("second_critic_reason")),
                        "reason": rec.get("second_critic_reason", ""),
                        "verdict": (rec.get("second_critic") or {}).get("verdict"),
                        "disagreement": bool(rec.get("disagreement"))},
        }
    return out


def finalise_pool(out_dir: Path, *, expected: int, rendered_now: int,
                  mode_targets: Dict[str, float]) -> Dict[str, Any]:
    records = list(llm_records(out_dir).values())
    records.extend(iter_jsonl(out_dir / DETERMINISTIC))
    kept: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    for rec in records:
        if rec["task_id"] in seen_ids:
            continue
        seen_ids.add(rec["task_id"])
        (kept if rec.get("passed") else rejected).append(rec)
    seen: Dict[str, str] = {}
    deduped: List[Dict[str, Any]] = []
    for rec in kept:
        fp = rec["fingerprints"]["exact_fingerprint"]
        if fp in seen:
            rejected.append({**rec, "failed_layers": ["exact_duplicate"]})
            continue
        seen[fp] = rec["task_id"]
        deduped.append(rec)
    write_jsonl(out_dir / QUERY_VALID, deduped)
    write_jsonl(out_dir / QUERY_REJECTS, rejected)
    div = diversity_report([r["query"] for r in deduped])
    layer_counts: Dict[str, int] = {}
    for rec in rejected:
        for layer in rec.get("failed_layers", []) or ["unknown"]:
            layer_counts[layer] = layer_counts.get(layer, 0) + 1
    return {
        "stage": "render", "expected": expected, "rendered_now": rendered_now,
        "rendered_total": len(kept) + len(rejected),
        "hard_valid": len(deduped), "rejected": len(rejected),
        "reject_layers": dict(sorted(layer_counts.items(), key=lambda kv: -kv[1])),
        "requested_mode_mix": _share(deduped, "requested_mode"),
        "actual_mode_mix": _share(deduped, "actual_mode"),
        "mode_targets": mode_targets,
        "query_source_mix": _share(deduped, "query_source"),
        "diversity": div,
        "diversity_gates": check_diversity(div),
        "output": str(out_dir / QUERY_VALID),
    }


def _share(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, float]:
    counts: Dict[str, int] = {}
    for row in rows:
        counts[str(row.get(key))] = counts.get(str(row.get(key)), 0) + 1
    n = max(1, len(rows))
    return {k: round(v / n, 4) for k, v in sorted(counts.items())}


def openrouter_writer(out_dir: Path, cfg_path: Path = Path("configs/pilot4_3_openrouter.yaml")):
    """Return a writer callable, or ``None`` when OpenRouter is unavailable.

    Import is lazy and failure is non-fatal by design: the dataset must still be
    produced (with the status flags set to false) on a machine without an API key.
    """
    try:
        from .orclient import OpenRouterClient, load_config      # noqa: WPS433
        from .orprompts import writer_prompt                     # noqa: WPS433
    except Exception:                                            # noqa: BLE001
        return None
    try:
        cfg = load_config(cfg_path)
        client = OpenRouterClient(cfg, out_dir)
        if not client.available():
            return None
    except Exception:                                            # noqa: BLE001
        return None

    def write(contract, payload, mode) -> Dict[str, Any] | None:
        try:
            system, user, schema, version = writer_prompt(payload, mode)
            got = client.chat("writer", [{"role": "system", "content": system},
                                         {"role": "user", "content": user}],
                              schema=schema,
                              meta={"sample_id": contract.task_id,
                                    "workflow_id": contract.workflow_id,
                                    "semantic_program_id": contract.plan_id,
                                    "prompt_version": version})
            return {"query": (got.get("content") or {}).get("query", ""),
                    "model": got.get("actual_model", ""),
                    "attempts": got.get("attempts", [])}
        except Exception:                                        # noqa: BLE001
            return None
    return write
