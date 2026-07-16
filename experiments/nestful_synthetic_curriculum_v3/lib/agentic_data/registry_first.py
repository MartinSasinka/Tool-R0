"""Registry-first candidate generation for the agentic pipeline.

The executable gold trace is built deterministically from the v5 tool registry
(``synthetic_gen_v5._build_chain``) — semantically compatible, always runnable.
An LLM pass (``challenger.question_polish_messages``) only rewrites the natural-
language question; it never authors calls, references or answers.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple

from ..synthetic_gen_v5 import (
    DiversityConfig,
    _UsageBalancer,
    _answer_type,
    _build_chain,
    execute_gold_calls,
)
from .schema import STAGES


def generate_registry_skeletons(
        stage: str, motif: str, n: int, rng: random.Random,
        balancer: _UsageBalancer,
        cfg: DiversityConfig | None = None,
) -> List[Dict[str, Any]]:
    """Build ``n`` executable skeleton candidates (no question text yet)."""
    cfg = cfg or DiversityConfig()
    lo, hi = STAGES[stage]
    out: List[Dict[str, Any]] = []
    for _ in range(n):
        n_calls = lo if lo == hi else rng.randrange(lo, hi + 1)
        for _attempt in range(cfg.max_pick_attempts):
            try:
                calls, observations, phrases = _build_chain(
                    rng, n_calls, motif, balancer)
                # Belt-and-suspenders: generation must only emit runnable chains.
                replay_obs = execute_gold_calls(calls)
                if replay_obs[-1] != observations[-1]:
                    raise RuntimeError("replay mismatch")
                break
            except (RuntimeError, KeyError, ZeroDivisionError, ValueError,
                    ArithmeticError):
                continue
        else:
            continue
        out.append({
            "gold_calls": calls,
            "tool_names": [c["name"] for c in calls],
            "motif_type": motif,
            "answer_type": _answer_type(observations[-1]),
            "observations": observations,
            "gold_answer": observations[-1],
            "_phrases": phrases,
            "_n_calls": n_calls,
        })
    return out


def attach_polished_questions(
        skeletons: List[Dict[str, Any]], parsed: Any,
) -> List[Dict[str, Any]]:
    """Merge LLM-polished questions onto registry skeletons by index."""
    if not skeletons:
        return []
    questions: List[str] = []
    if isinstance(parsed, dict):
        rows = parsed.get("candidates") or parsed.get("questions")
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, str):
                    questions.append(row.strip())
                elif isinstance(row, dict) and isinstance(row.get("question"), str):
                    questions.append(row["question"].strip())
    elif isinstance(parsed, list):
        for row in parsed:
            if isinstance(row, str):
                questions.append(row.strip())
            elif isinstance(row, dict) and isinstance(row.get("question"), str):
                questions.append(row["question"].strip())
    cands: List[Dict[str, Any]] = []
    for i, sk in enumerate(skeletons):
        if i >= len(questions) or not questions[i]:
            continue
        cands.append({
            "question": questions[i],
            "tool_names": sk["tool_names"],
            "gold_calls": sk["gold_calls"],
            "motif_type": sk["motif_type"],
            "answer_type": sk["answer_type"],
            "rationale": "registry-first deterministic trace",
            "_registry_first": True,
        })
    return cands
