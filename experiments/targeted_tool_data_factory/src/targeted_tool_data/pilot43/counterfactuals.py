"""Counterfactual instances of one plan, deliberately spanning different answers.

Both the V4 shortcut search and the node-necessity check ask "does this shorter
program / this bypass still give the gold answer?", and both are worthless if
every counterfactual instance happens to answer the same way. The smoke run made
that concrete: on a boolean task whose sampled instances were all True,
``is_at_least(budget, budget)`` looked like a confirmed shortcut and a genuinely
load-bearing tax step looked unnecessary.

So the counterfactual set is *constructed* to disagree: boolean plans get
instances calibrated to True and to False, categorical plans get every band, and
numeric plans simply get different seeds. A set that still fails to produce two
distinct answers is reported as weak, and the caller treats the corresponding V4
verdict as unresolved rather than as a pass.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from .blueprints import Blueprint, Plan
from .build import BuildError, Instance, instantiate
from .program import Program, answers_equal

CATEGORY_BANDS = ("low", "medium", "high")

#: How many instances a low-entropy answer needs. A candidate shortcut that matches
#: a boolean on k independent instances survives by luck with probability ~2^-k, and
#: the V4 search tests hundreds of candidates per task, so k=6 lets roughly a dozen
#: false shortcuts through per task (the smoke run showed exactly that). 20 puts the
#: expected number of false confirmations per task far below one.
LOW_ENTROPY_N = 20
DEFAULT_N = 8


def counterfactual_instances(bp: Blueprint, plan: Plan, *, answer_type: str,
                             track: str, seed: int, n: int | None = None
                             ) -> Tuple[List[Instance], Dict[str, Any]]:
    """Instances of one plan spread across answer outcomes.

    ``n`` defaults to :data:`LOW_ENTROPY_N` for boolean and categorical answers and
    to :data:`DEFAULT_N` otherwise.
    """
    if n is None:
        n = LOW_ENTROPY_N if answer_type in ("boolean", "category") else DEFAULT_N
    out: List[Instance] = []
    attempts = 0
    plans_wanted: List[Dict[str, Any]] = []
    if answer_type == "boolean":
        for i in range(n):
            plans_wanted.append({"want_bool": i % 2 == 0,
                                 "near_boundary": i >= n // 2})
    elif answer_type == "category":
        for i in range(n):
            plans_wanted.append({"want_category": CATEGORY_BANDS[i % 3]})
    else:
        plans_wanted = [{} for _ in range(n)]

    for i, kwargs in enumerate(plans_wanted):
        for bump in range(4):
            attempts += 1
            trial_seed = seed + 7919 * (i + 1) + 104729 * bump
            try:
                out.append(instantiate(bp, plan, trial_seed, track=track, **kwargs))
                break
            except BuildError:
                continue
    answers = [i.answer for i in out]
    distinct = _distinct_answers(answers)
    low_entropy = answer_type in ("boolean", "category")
    minority = _minority_share(answers)
    # A usable set needs enough instances *and* both outcomes reasonably represented:
    # 19 True and 1 False would still let a constant-True expression look confirmed.
    strong = (distinct >= 2 and
              (not low_entropy or (len(out) >= LOW_ENTROPY_N // 2 and
                                   minority >= 0.25)))
    meta = {
        "requested": len(plans_wanted),
        "built": len(out),
        "attempts": attempts,
        "distinct_answers": distinct,
        "answer_type": answer_type,
        "minority_share": round(minority, 4),
        "low_entropy_answer": low_entropy,
        "mixed": bool(strong),
        "weak": not strong,
    }
    return out, meta


def _minority_share(answers: Sequence[Any]) -> float:
    if not answers:
        return 0.0
    groups: List[List[Any]] = []
    for a in answers:
        for g in groups:
            if answers_equal(a, g[0]):
                g.append(a)
                break
        else:
            groups.append([a])
    return min(len(g) for g in groups) / len(answers)


def _distinct_answers(answers: Sequence[Any]) -> int:
    seen: List[Any] = []
    for a in answers:
        if not any(answers_equal(a, s) for s in seen):
            seen.append(a)
    return len(seen)


def as_fact_pairs(instances: Sequence[Instance]) -> List[Tuple[Dict[str, Any], Any]]:
    """(role values, gold answer) pairs, the form the V4 confirmation wants."""
    return [({k: v for k, v in inst.role_values.items()}, inst.answer)
            for inst in instances]


def as_programs(instances: Sequence[Instance]) -> List[Program]:
    return [inst.program for inst in instances]
