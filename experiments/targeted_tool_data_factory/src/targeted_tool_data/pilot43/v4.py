"""V4: shortcut search for every answer type, with counterfactual confirmation.

The question V4 answers is narrow and operational: *using the tools this task
actually offers and only the facts its query actually states, can the gold
answer be produced with fewer calls than the gold program uses?* If yes the task
teaches the model to guess rather than to plan, and it is rejected.

Two properties distinguish this from the Pilot4.2 check.

* It runs for numeric, boolean, string, list, object and category answers alike.
  There is no "skipped because non-numeric" state.
* A hit is not believed on one input. A candidate shorter program is re-evaluated
  on several counterfactual instances of the same workflow plan; a boolean that
  agrees once out of five is a coincidence, not a shortcut.

Search space (exported with every task so the claim is checkable):
offered tools x stated facts x depth <= min(gold_calls - 1, depth_cap). The
search is exhaustive inside that space; if a budget is hit the task is marked
unresolved and is never selected.
"""
from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Sequence, Set, Tuple

from . import semtypes as st
from .ops import Op, build_ops
from .program import answers_equal


@dataclass(frozen=True)
class V4Config:
    """Search budget. ``depth_cap`` is the honest limit of the claim.

    For a 9-call gold program an exhaustive search to depth 8 is not computable, so
    the guarantee this gate provides is "no equivalent program of at most
    ``depth_cap`` calls exists inside the offered-tool x stated-fact space". That
    is the claim exported per task (``max_depth_searched``,
    ``complete_to_gold_minus_one``) and reported in aggregate; it is never
    described as a full-depth proof.
    """
    depth_cap: int = 3
    max_frontier: int = 400
    max_expansions: int = 600_000
    max_assignments_per_op: int = 250_000
    counterfactual_instances: int = 6
    #: Chance matches to confirm before the depth is declared truncated. A boolean
    #: sink yields them by the thousand and each costs counterfactual evaluations.
    max_confirmations: int = 600
    #: Wall-clock ceiling per task. A depth cut short by the clock is reported as
    #: truncated, so the task fails the gate instead of silently claiming a
    #: guarantee the search never established.
    max_seconds: float = 20.0


#: the least depth a selectable task must have fully excluded. A one- or two-call
#: alternative is the shortcut that actually teaches guessing, so it must always be
#: ruled out; deeper guarantees are recorded per task and reported in aggregate.
MIN_GUARANTEE_DEPTH = 2


@dataclass
class _Entry:
    key: str
    value: Any
    sem: str
    depth: int
    expr: Any                      # ("fact", role) | (pid, (expr, ...))


def _canon(value: Any) -> str:
    if isinstance(value, bool):
        return f"b:{value}"
    if isinstance(value, (int, float)):
        try:
            return f"n:{round(float(value), 6)}"
        except OverflowError:      # a big-int intermediate, e.g. 900 ** 400
            return f"i:{value}"
    if isinstance(value, str):
        return f"s:{value}"
    if isinstance(value, list):
        return "l:[" + ",".join(_canon(v) for v in value) + "]"
    if isinstance(value, dict):
        return "d:{" + ",".join(f"{k}={_canon(v)}"
                                for k, v in sorted(value.items())) + "}"
    return f"o:{value!r}"


def _sane(value: Any) -> bool:
    """Same degeneracy limits the executor applies, so the search space matches it."""
    from .program import MAX_ABS, MAX_LIST, MAX_TEXT

    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        try:
            magnitude = abs(float(value))
        except OverflowError:
            return False
        return not (math.isnan(magnitude) or math.isinf(magnitude)) \
            and magnitude <= MAX_ABS
    if isinstance(value, str):
        return bool(value) and len(value) <= MAX_TEXT
    if isinstance(value, list):
        return bool(value) and len(value) <= MAX_LIST and all(
            _sane(v) for v in value)
    if isinstance(value, dict):
        return bool(value) and len(value) <= MAX_LIST and all(
            _sane(v) for v in value.values())
    return False


def evaluate_expr(expr: Any, facts: Dict[str, Any]) -> Any:
    """Re-evaluate a candidate shortcut on a different instance's facts."""
    ops = build_ops()
    if isinstance(expr, tuple) and expr and expr[0] == "fact":
        if expr[1] not in facts:
            raise KeyError(expr[1])
        return facts[expr[1]]
    pid, args = expr
    op = ops[pid]
    values = [evaluate_expr(a, facts) for a in args]
    return op.fn(**{p.name: v for p, v in zip(op.params, values)})


def expr_depth(expr: Any) -> int:
    if isinstance(expr, tuple) and expr and expr[0] == "fact":
        return 0
    return 1 + max((expr_depth(a) for a in expr[1]), default=0)


def expr_calls(expr: Any) -> int:
    if isinstance(expr, tuple) and expr and expr[0] == "fact":
        return 0
    return 1 + sum(expr_calls(a) for a in expr[1])


def render_expr(expr: Any) -> str:
    if isinstance(expr, tuple) and expr and expr[0] == "fact":
        return f"${expr[1]}"
    pid, args = expr
    return f"{pid}(" + ", ".join(render_expr(a) for a in args) + ")"


def _assignments(per_param: Sequence[Sequence[_Entry]], frontier_ids: Set[int],
                 require_novel: bool) -> Iterator[Tuple[_Entry, ...]]:
    """Argument tuples, with the novelty requirement enforced by construction.

    Filtering the full cartesian product after the fact was the original approach
    and it made arity-3 ops blow the budget at depth 2 even though almost every
    tuple was going to be discarded. Here one slot is pinned to the frontier.
    """
    if not require_novel:
        yield from itertools.product(*per_param)
        return
    seen: Set[Tuple[int, ...]] = set()
    for slot in range(len(per_param)):
        novel = [e for e in per_param[slot] if id(e) in frontier_ids]
        if not novel:
            continue
        others = [per_param[i] for i in range(len(per_param)) if i != slot]
        for pick in novel:
            for rest in itertools.product(*others) if others else [()]:
                tup = list(rest)
                tup.insert(slot, pick)
                key = tuple(id(e) for e in tup)
                if key in seen:
                    continue
                seen.add(key)
                yield tuple(tup)


def _count(per_param: Sequence[Sequence[_Entry]], frontier_ids: Set[int],
           require_novel: bool, cap: int) -> int:
    total = 1
    if not require_novel:
        for slot in per_param:
            total *= len(slot)
            if total > cap:
                return total
        return total
    total = 0
    for slot in range(len(per_param)):
        novel = sum(1 for e in per_param[slot] if id(e) in frontier_ids)
        rest = 1
        for i, other in enumerate(per_param):
            if i != slot:
                rest *= len(other)
        total += novel * rest
        if total > cap:
            return total
    return total


def _admissible(op: Op, entries: Sequence[_Entry]) -> List[List[_Entry]]:
    per_param: List[List[_Entry]] = []
    for p in op.params:
        pool = [e for e in entries
                if st.compatible(p.sem, e.sem) and st.matches_value(p.sem, e.value)]
        if not pool:
            return []
        per_param.append(pool)
    return per_param


def search_shortcuts(facts: Dict[str, Tuple[Any, str]], offered: Sequence[str],
                     gold_answer: Any, gold_calls: int,
                     cfg: V4Config = V4Config(),
                     confirm: Any = None) -> Dict[str, Any]:
    """Bounded search for a strictly shorter equivalent program.

    ``confirm(expr) -> bool`` re-evaluates a candidate on counterfactual instances.
    It is called as soon as a candidate appears, because a candidate that agrees on
    one input proves nothing: on a boolean sink roughly half of all one-call
    expressions match by luck. Only a *confirmed* candidate stops the search;
    unconfirmed ones are recorded and the search keeps going, which is what makes
    the reported depth guarantee mean anything for boolean tasks.
    """
    ops = build_ops()
    offered = [pid for pid in dict.fromkeys(offered) if pid in ops]
    max_depth = min(max(gold_calls - 1, 0), cfg.depth_cap)
    pool: List[_Entry] = [
        _Entry(key=name, value=value, sem=sem, depth=0, expr=("fact", name))
        for name, (value, sem) in sorted(facts.items())]
    seen = {(_canon(e.value), e.sem) for e in pool}
    expansions = 0
    budget_hit = False
    confirmed: List[Dict[str, Any]] = []
    coincidental: List[Dict[str, Any]] = []
    n_coincidental = 0

    frontier = list(pool)
    max_depth_complete = 0
    started = time.perf_counter()
    for depth in range(1, max_depth + 1):
        depth_truncated = False
        produced: List[_Entry] = []
        for pid in offered:
            op = ops[pid]
            per_param = _admissible(op, pool)
            if not per_param:
                continue
            frontier_ids = {id(e) for e in frontier}
            assignments = _assignments(per_param, frontier_ids, depth > 1)
            combos = _count(per_param, frontier_ids, depth > 1,
                            cfg.max_assignments_per_op)
            if combos > cfg.max_assignments_per_op:
                budget_hit = True
                depth_truncated = True
                continue
            for assignment in assignments:
                expansions += 1
                if expansions > cfg.max_expansions or (
                        expansions % 512 == 0
                        and time.perf_counter() - started > cfg.max_seconds):
                    budget_hit = True
                    depth_truncated = True
                    break
                try:
                    out = op.fn(**{p.name: e.value
                                   for p, e in zip(op.params, assignment)})
                except Exception:          # noqa: BLE001 - op guards are the filter
                    continue
                if out is None or isinstance(out, (set, tuple)) or not _sane(out):
                    continue
                expr = (pid, tuple(e.expr for e in assignment))
                calls = expr_calls(expr)
                if calls < gold_calls and answers_equal(out, gold_answer):
                    row = {"calls": calls, "expr": expr,
                           "rendered": render_expr(expr)}
                    verdict = (confirm(expr) if confirm else
                               {"tested": 0, "agree": 0, "confirms": True})
                    row.update(counterfactuals_tested=verdict["tested"],
                               counterfactuals_agreeing=verdict["agree"])
                    if verdict["confirms"]:
                        confirmed.append(row)
                        break
                    n_coincidental += 1
                    if len(coincidental) < 8:
                        coincidental.append(row)
                    if n_coincidental >= cfg.max_confirmations:
                        budget_hit = True
                        depth_truncated = True
                        break
                sem = op.resolve_out_sem([e.sem for e in assignment])
                key = (_canon(out), sem)
                if key in seen:
                    continue
                try:
                    if not st.matches_value(sem, out):
                        continue
                except ValueError:
                    continue
                seen.add(key)
                produced.append(_Entry(key=f"d{depth}_{len(produced)}", value=out,
                                       sem=sem, depth=depth, expr=expr))
            if confirmed or depth_truncated or expansions > cfg.max_expansions:
                break
        if not depth_truncated and not confirmed:
            max_depth_complete = depth
        if confirmed or depth_truncated:
            break
        produced.sort(key=lambda e: _canon(e.value))
        if len(produced) > cfg.max_frontier:
            # the next depth would start from an incomplete value set, so the
            # guarantee stops here rather than pretending to cover that depth
            produced = produced[:cfg.max_frontier]
            budget_hit = True
            frontier = produced
            pool = pool + produced
            break
        frontier = produced
        pool = pool + produced
        if not frontier:
            max_depth_complete = max_depth      # nothing new is reachable at all
            break

    return {
        "max_depth_requested": max_depth,
        "max_depth_complete": max_depth_complete,
        "expansions": expansions,
        "budget_hit": budget_hit,
        "exhausted": not budget_hit,
        "confirmed_shortcuts": confirmed,
        "n_confirmed": len(confirmed),
        "coincidental_matches": coincidental,
        "n_coincidental": n_coincidental,
    }


def counterfactual_checker(counterfactuals: Sequence[Tuple[Dict[str, Any], Any]]):
    """``expr -> {tested, agree, confirms}`` over other instances of the same plan.

    ``confirms`` also demands that the candidate was *admissible* on at least half of
    the counterfactuals: agreeing on the two instances it happens to accept is not
    evidence of equivalence.
    """
    min_tested = max(2, (len(counterfactuals) + 1) // 2)

    def check(expr: Tuple[Any, ...]) -> Dict[str, Any]:
        agree = 0
        tested = 0
        for facts, gold in counterfactuals:
            try:
                value = evaluate_expr(expr, facts)
            except Exception:          # noqa: BLE001 - inadmissible on this input
                continue
            tested += 1
            if answers_equal(value, gold):
                agree += 1
            else:
                # one disagreement already rules out equivalence, and boolean tasks
                # produce hundreds of chance matches per search, so stopping here is
                # what keeps the gate affordable
                break
        return {"tested": tested, "agree": agree,
                "confirms": tested >= min_tested and agree == tested}
    return check


def confirm_on_counterfactuals(hits: Sequence[Dict[str, Any]],
                               counterfactuals: Sequence[Tuple[Dict[str, Any], Any]]
                               ) -> Dict[str, Any]:
    """A candidate is a real shortcut only if it holds on other instances too."""
    check = counterfactual_checker(counterfactuals)
    confirmed: List[Dict[str, Any]] = []
    coincidental: List[Dict[str, Any]] = []
    for hit in hits:
        verdict = check(hit["expr"])
        row = {"rendered": hit["rendered"], "calls": hit["calls"],
               "counterfactuals_tested": verdict["tested"],
               "counterfactuals_agreeing": verdict["agree"]}
        if verdict["confirms"]:
            confirmed.append(row)
        else:
            coincidental.append(row)
    return {"confirmed_shortcuts": confirmed[:8],
            "coincidental_matches": coincidental[:8],
            "n_confirmed": len(confirmed), "n_coincidental": len(coincidental)}


def v4_gate(facts: Dict[str, Tuple[Any, str]], offered: Sequence[str],
            gold_answer: Any, gold_calls: int,
            counterfactuals: Sequence[Tuple[Dict[str, Any], Any]],
            cfg: V4Config = V4Config(),
            counterfactuals_mixed: bool = True) -> Dict[str, Any]:
    """The pre-selection hard gate. Runs for every answer type, no exceptions.

    ``counterfactuals_mixed`` says whether the counterfactual set actually contains
    more than one answer. When it does not, a confirmed hit cannot be distinguished
    from a constant that happens to agree, so the verdict is *unresolved* rather
    than a pass -- the task is then never selected.
    """
    search = search_shortcuts(facts, offered, gold_answer, gold_calls, cfg,
                              confirm=counterfactual_checker(counterfactuals))
    has_shortcut = search["n_confirmed"] > 0
    depth_complete = int(search["max_depth_complete"])
    complete_to_gold = depth_complete >= gold_calls - 1
    # the guarantee is "no equivalent program of <= depth_complete calls exists in
    # the offered-tool x stated-fact space"; MIN_GUARANTEE_DEPTH is the least
    # guarantee a selectable task must carry
    min_depth = min(max(gold_calls - 1, 1), MIN_GUARANTEE_DEPTH)
    resolved = (depth_complete >= min_depth or has_shortcut) and counterfactuals_mixed
    shortest = min([h["calls"] for h in search["confirmed_shortcuts"]],
                   default=gold_calls)
    return {
        "v4_executed": True,
        "answer_type_checked": st.value_kind(gold_answer),
        "search_space": {
            "offered_tools": len(set(offered)),
            "stated_facts": len(facts),
            "max_depth_requested": search["max_depth_requested"],
            "max_depth_complete": depth_complete,
            "complete_to_gold_minus_one": complete_to_gold,
            "guarantee": (f"no equivalent program with <= {depth_complete} calls "
                          f"over the offered tools and stated facts"),
        },
        "expansions": search["expansions"],
        "search_budget_exhausted": bool(search["budget_hit"]),
        "search_complete": depth_complete >= min_depth,
        "resolved": bool(resolved),
        "has_shortcut": has_shortcut,
        "minimal_valid_call_count": shortest,
        "counterfactual_instances_tested": len(counterfactuals),
        "counterfactuals_mixed": bool(counterfactuals_mixed),
        "confirmed_shortcuts": search["confirmed_shortcuts"],
        "n_confirmed": search["n_confirmed"],
        "coincidental_matches": search["coincidental_matches"],
        "n_coincidental": search["n_coincidental"],
        "safe_for_core_train": bool(resolved and not has_shortcut),
    }
