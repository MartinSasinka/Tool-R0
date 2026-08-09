"""Per-node necessity evidence.

Pilot4.2 stored a single boolean per task, which proves nothing. For every node
this module runs four independent experiments and records the outcome of each:

* **bypass** -- drop the node and feed its consumers one of its own inputs. If the
  resulting shorter program still executes and still produces the gold answer,
  the node did no work.
* **answer change** -- did the bypass change the answer at all?
* **target reachability** -- delete the node outright; is the sink still
  computable?
* **alternative binding** -- can another offered op replace this one and keep the
  answer? That does not make the node unnecessary, but it is recorded because it
  feeds the solution-equivalence report.

A bypass that preserves the answer is only believed when it preserves it on
*counterfactual instances of the same plan* as well. On a boolean sink, dropping
a node leaves the answer unchanged about half the time by luck, and Pilot4.2's
single-input check is exactly how such a node would have been declared
unnecessary (or, worse, how a shortcut would have been missed).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence

from .ops import build_ops
from .program import (ExecError, Node, Program, ProgramError, Ref, answers_equal,
                      execute, validate_semantic_edges, validate_structure)


def _bypass_candidates(node: Node) -> List[Any]:
    """Values that could stand in for a node's output if it were a no-op."""
    refs = [v for v in node.args.values() if isinstance(v, Ref)]
    if refs:
        return list(refs)
    return [v for v in node.args.values() if not isinstance(v, (list, dict))]


def _without(prog: Program, node_id: str, replacement: Any) -> Program:
    nodes: List[Node] = []
    for nd in prog.nodes:
        if nd.node_id == node_id:
            continue
        args = {}
        for k, v in nd.args.items():
            args[k] = replacement if isinstance(v, Ref) and v.node_id == node_id else v
        nodes.append(Node(nd.node_id, nd.op, args, dict(nd.arg_sems)))
    sink = prog.sink
    if sink == node_id:
        sink = nodes[-1].node_id if nodes else ""
    return Program(nodes=nodes, sink=sink)


def _run(prog: Program) -> Any:
    """Execute a trial program or raise; structural/semantic gates first."""
    validate_structure(prog)
    if validate_semantic_edges(prog):
        raise ProgramError("semantic edges")
    _obs, answer = execute(prog)
    return answer


def _same_arity_swap(prog: Program, node_id: str, pid: str) -> Program | None:
    ops = build_ops()
    cand = ops[pid]
    node = prog.node(node_id)
    if cand.arity != len(node.args):
        return None
    values = list(node.args.values())
    nodes = []
    for nd in prog.nodes:
        if nd.node_id == node_id:
            nodes.append(Node(nd.node_id, pid,
                              {p.name: values[i] for i, p in enumerate(cand.params)},
                              dict(nd.arg_sems)))
        else:
            nodes.append(Node(nd.node_id, nd.op, dict(nd.args), dict(nd.arg_sems)))
    return Program(nodes=nodes, sink=prog.sink)


def _alternative_binding(prog: Program, node_id: str, gold: Any,
                         allowed_ops: Sequence[str]) -> bool:
    for pid in allowed_ops:
        if pid == prog.node(node_id).op:
            continue
        trial = _same_arity_swap(prog, node_id, pid)
        if trial is None:
            continue
        try:
            answer = _run(trial)
        except (ProgramError, ExecError):
            continue
        if answers_equal(answer, gold):
            return True
    return False


def node_necessity(prog: Program, *, allowed_ops: Sequence[str] = (),
                   check_alternatives: bool = True,
                   counterfactuals: Sequence[Program] = ()) -> List[Dict[str, Any]]:
    """One evidence record per node, in program order."""
    gold = _run(prog)
    rows: List[Dict[str, Any]] = []
    for nd in prog.nodes:
        bypass_executed = False
        bypass_answer_changed: bool | None = None
        confirmed_on = 0
        tested_on = 0
        for replacement in _bypass_candidates(nd):
            trial = _without(prog, nd.node_id, replacement)
            if not trial.nodes:
                continue
            try:
                answer = _run(trial)
            except (ProgramError, ExecError):
                continue
            bypass_executed = True
            changed = not answers_equal(answer, gold)
            if changed:
                bypass_answer_changed = True if bypass_answer_changed is None \
                    else bypass_answer_changed
                continue
            # the bypass kept the answer here: only believe it if it also keeps it
            # on enough other instances of the same plan. One instance proves
            # nothing on a boolean sink, so an unconfirmable bypass counts as the
            # node doing work.
            tested_on, confirmed_on = _confirm_bypass(
                counterfactuals, nd.node_id, replacement)
            if tested_on >= _min_confirmations(counterfactuals) \
                    and confirmed_on == tested_on:
                bypass_answer_changed = False
                break
            bypass_answer_changed = True
        deleted = _without(prog, nd.node_id, None)
        target_unreachable = True
        try:
            _run(deleted)
            target_unreachable = False
        except (ProgramError, ExecError, TypeError):
            target_unreachable = True
        alt = (_alternative_binding(prog, nd.node_id, gold, allowed_ops)
               if check_alternatives and allowed_ops else False)
        necessary = not (bypass_executed and bypass_answer_changed is False)
        rows.append({
            "node_id": nd.node_id,
            "primitive_id": nd.op,
            "removal_executable": bypass_executed,
            "removal_changes_answer": (bool(bypass_answer_changed)
                                       if bypass_answer_changed is not None else None),
            "target_unreachable": target_unreachable,
            "alternative_binding_found": alt,
            "counterfactuals_tested": tested_on,
            "counterfactuals_agreeing_with_bypass": confirmed_on,
            "necessary": necessary,
        })
    return rows


def _min_confirmations(counterfactuals: Sequence[Program]) -> int:
    return max(2, (len(counterfactuals) + 1) // 2)


def _confirm_bypass(counterfactuals: Sequence[Program], node_id: str,
                    replacement: Any) -> tuple[int, int]:
    """How many counterfactual instances the answer-preserving bypass survives."""
    tested = 0
    agreeing = 0
    for cf in counterfactuals:
        try:
            cf_gold = _run(cf)
        except (ProgramError, ExecError):
            continue
        if not any(nd.node_id == node_id for nd in cf.nodes):
            continue
        # rebuild the same replacement shape against this instance's own nodes
        cf_node = cf.node(node_id)
        cf_replacement: Any
        if isinstance(replacement, Ref):
            cf_replacement = replacement
        else:
            candidates = [v for v in cf_node.args.values()
                          if not isinstance(v, (Ref, list, dict))]
            if not candidates:
                continue
            cf_replacement = candidates[0]
        trial = _without(cf, node_id, cf_replacement)
        if not trial.nodes:
            continue
        try:
            answer = _run(trial)
        except (ProgramError, ExecError):
            continue
        tested += 1
        if answers_equal(answer, cf_gold):
            agreeing += 1
    return tested, agreeing


def all_nodes_necessary(rows: Sequence[Dict[str, Any]]) -> bool:
    return bool(rows) and all(r["necessary"] for r in rows)


def necessity_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "nodes": len(rows),
        "necessary_nodes": sum(1 for r in rows if r["necessary"]),
        "unnecessary_nodes": [r["node_id"] for r in rows if not r["necessary"]],
        "alternative_bindings": [r["node_id"] for r in rows
                                 if r["alternative_binding_found"]],
        "all_necessary": all_nodes_necessary(rows),
    }
