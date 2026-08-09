"""Offered-tool construction with behaviourally validated distractors.

A distractor is only useful if it is *schema-plausible and semantically wrong*.
Every candidate is therefore substituted into the gold program and executed: it
survives only when it changes the answer on the current instance and on
counterfactual instances of the same plan. Anything that leaves the answer intact
is an alias, not a distractor, and would make the task multi-solution by accident.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Set, Tuple

from . import semtypes as st
from .ops import Op, build_ops
from .program import (ExecError, Node, Program, ProgramError, answers_equal,
                      execute, validate_semantic_edges, validate_structure)

HARD = "hard"
MEDIUM = "medium"
EASY = "easy"


@dataclass(frozen=True)
class Distractor:
    primitive_id: str
    hardness: str
    reason: str


def _signature(op: Op) -> Tuple[Tuple[str, ...], str]:
    return (tuple(p.runtime for p in op.params),
            "number" if op.out_sem == "@preserve" else st.runtime_of(op.out_sem))


def hardness_of(gold_ops: Sequence[str], pid: str) -> str:
    """How close is ``pid`` to this gold set, by the ranking rule used above?"""
    ops = build_ops()
    if pid not in ops or pid in set(gold_ops):
        return ""
    op = ops[pid]
    families = {ops[g].family for g in gold_ops if g in ops}
    for gid in gold_ops:
        gop = ops.get(gid)
        if gop is None or _signature(op) != _signature(gop):
            continue
        if op.family == gop.family and op.capability != gop.capability:
            return HARD
        if op.family != gop.family:
            return MEDIUM
    return MEDIUM if op.family in families else EASY


def hard_aliases(gold_ops: Sequence[str],
                 rejected: Sequence[Mapping[str, str]]) -> int:
    """How many same-family siblings were dropped for computing the same answer?

    Without this count an empty hard slot is ambiguous: it can mean the search
    never reached a sibling, or that every sibling turned out to be an alias and
    offering it would have made the task multi-solution.
    """
    return sum(1 for r in rejected
               if hardness_of(gold_ops, r.get("primitive_id", "")) == HARD)


def candidate_distractors(gold_ops: Sequence[str]) -> List[Distractor]:
    """Ranked candidates: same signature first, same capability family first."""
    ops = build_ops()
    gold = set(gold_ops)
    out: List[Distractor] = []
    seen: Set[str] = set(gold)
    for gid in gold_ops:
        gop = ops[gid]
        gsig = _signature(gop)
        for pid, op in sorted(ops.items()):
            if pid in seen:
                continue
            if _signature(op) != gsig:
                continue
            if op.family == gop.family and op.capability != gop.capability:
                out.append(Distractor(pid, HARD,
                                      f"same signature and capability family as {gid}"))
                seen.add(pid)
            elif op.family != gop.family:
                out.append(Distractor(pid, MEDIUM,
                                      f"same signature as {gid}, other family"))
                seen.add(pid)
    for pid, op in sorted(ops.items()):
        if pid in seen:
            continue
        if any(op.family == ops[g].family for g in gold):
            out.append(Distractor(pid, MEDIUM, "same capability family, other signature"))
            seen.add(pid)
    for pid in sorted(ops):
        if pid not in seen:
            out.append(Distractor(pid, EASY, "unrelated capability"))
            seen.add(pid)
    return out


def _substitute(prog: Program, node_id: str, pid: str) -> Program | None:
    ops = build_ops()
    cand = ops[pid]
    node = prog.node(node_id)
    if cand.arity != len(node.args):
        return None
    values = list(node.args.values())
    nodes: List[Node] = []
    for nd in prog.nodes:
        if nd.node_id == node_id:
            nodes.append(Node(nd.node_id, pid,
                              {p.name: values[i] for i, p in enumerate(cand.params)},
                              dict(nd.arg_sems)))
        else:
            nodes.append(Node(nd.node_id, nd.op, dict(nd.args), dict(nd.arg_sems)))
    return Program(nodes=nodes, sink=prog.sink)


def _preserves(prog: Program, node_id: str, pid: str,
               gold_answer: Any) -> bool | None:
    """Did swapping ``pid`` into this slot keep the answer? ``None`` = inadmissible."""
    trial = _substitute(prog, node_id, pid)
    if trial is None:
        return None
    try:
        validate_structure(trial)
        if validate_semantic_edges(trial):
            return None
        _obs, answer = execute(trial)
    except (ProgramError, ExecError):
        return None
    return answers_equal(answer, gold_answer)


def behaviourally_wrong(prog: Program, pid: str, gold_answer: Any,
                        counterfactuals: Sequence[Program] = ()) -> bool:
    """True when this op is schema-plausible here yet never computes the target.

    An op is rejected as an *alias* only when a slot exists in which it preserves
    the answer on the current instance **and** on the counterfactual instances. On a
    boolean sink almost any substitution preserves the answer on a single input by
    luck, and treating that as aliasing rejected nearly every candidate in the smoke
    run, leaving boolean tasks with no distractors at all.
    """
    changed_somewhere = False
    for nd in prog.nodes:
        verdict = _preserves(prog, nd.node_id, pid, gold_answer)
        if verdict is None:
            continue
        if not verdict:
            changed_somewhere = True
            continue
        tested = 0
        disagreed = False
        for cf in counterfactuals:
            try:
                _o, cf_gold = execute(cf)
            except ExecError:
                continue
            cf_verdict = _preserves(cf, nd.node_id, pid, cf_gold)
            if cf_verdict is None:
                continue
            tested += 1
            if not cf_verdict:
                disagreed = True       # one disagreement settles it: not an alias
                break
        if not disagreed:
            return False               # an alias: never offer it as a distractor
        changed_somewhere = True
    return changed_somewhere


def build_offered_tools(prog: Program, gold_answer: Any, *, track: str,
                        target_count: int, seed: int,
                        counterfactuals: Sequence[Program] = (),
                        min_hard: int = 2, max_examined: int = 90
                        ) -> Dict[str, Any]:
    """Gold tools plus validated distractors, shuffled to a target set size.

    ``max_examined`` bounds the behavioural validation work: every candidate costs
    one program execution per slot plus counterfactual confirmations, and the ranked
    candidate list is ~200 ops long, so an unbounded scan dominates the run time
    without improving the offered set.
    """
    ops = build_ops()
    rng = random.Random(seed)
    gold_ids = list(dict.fromkeys(nd.op for nd in prog.nodes))
    chosen: List[Distractor] = []
    rejected: List[Dict[str, str]] = []
    hard_count = 0
    examined = 0
    for cand in candidate_distractors(gold_ids):
        if len(gold_ids) + len(chosen) >= target_count or examined >= max_examined:
            break
        if cand.hardness == EASY and hard_count < min_hard:
            continue
        examined += 1
        if not behaviourally_wrong(prog, cand.primitive_id, gold_answer,
                                   counterfactuals):
            rejected.append({"primitive_id": cand.primitive_id,
                             "reason": "answer-preserving alias"})
            continue
        chosen.append(cand)
        if cand.hardness == HARD:
            hard_count += 1
    tools: List[Dict[str, Any]] = []
    for pid in gold_ids:
        tools.append(_tool_schema(ops[pid], track, is_distractor=False))
    for cand in chosen:
        tools.append({**_tool_schema(ops[cand.primitive_id], track,
                                     is_distractor=True),
                      "distractor_hardness": cand.hardness,
                      "distractor_reason": cand.reason})
    rng.shuffle(tools)
    names = [t["name"] for t in tools]
    if len(set(names)) != len(names):
        raise ValueError("duplicate tool name in offered set")
    return {
        "tools": tools,
        "offered_tool_count": len(tools),
        "gold_tool_count": len(gold_ids),
        "distractor_count": len(chosen),
        "hard_distractor_count": sum(1 for c in chosen if c.hardness == HARD),
        "medium_distractor_count": sum(1 for c in chosen if c.hardness == MEDIUM),
        "easy_distractor_count": sum(1 for c in chosen if c.hardness == EASY),
        "rejected_distractors": rejected[:12],
        "candidates_examined": examined,
        "distractor_primitive_ids": [c.primitive_id for c in chosen],
    }


def rerender_tools(tools: Sequence[Mapping[str, Any]],
                   track: str) -> List[Dict[str, Any]]:
    """Re-derive every offered tool's surface from the registry.

    Tool names and parameter names are a rendering of the registry, not data, so
    a verified row keeps only the primitive ids and hardness labels authoritative.
    Re-deriving them at export time is what stops a surface added after
    verification from leaving the tool list and the gold calls disagreeing.
    """
    ops = build_ops()
    out: List[Dict[str, Any]] = []
    for tool in tools:
        op = ops.get(str(tool.get("primitive_id") or ""))
        if op is None:
            out.append(dict(tool))
            continue
        fresh = _tool_schema(op, track,
                             is_distractor=bool(tool.get("is_distractor")))
        out.append({**fresh, **{k: v for k, v in tool.items()
                                if k.startswith("distractor_")}})
    names = [t["name"] for t in out]
    if len(set(names)) != len(names):
        raise ValueError("duplicate tool name after re-rendering surfaces")
    return out


def _tool_schema(op: Op, track: str, *, is_distractor: bool) -> Dict[str, Any]:
    surf = op.surface(track)
    params = []
    for p, shown in zip(op.params, surf.param_names):
        params.append({"name": shown, "type": p.runtime, "required": True,
                       "description": f"{p.role.replace('_', ' ')} ({p.sem})"})
    return {
        "name": surf.name,
        "description": surf.description,
        "parameters": params,
        "output_field": surf.output_field,
        "output_type": ("number" if op.out_sem == "@preserve"
                        else st.runtime_of(op.out_sem)),
        "primitive_id": op.pid,
        "capability": op.capability,
        "surface_track": track,
        "is_distractor": is_distractor,
    }
