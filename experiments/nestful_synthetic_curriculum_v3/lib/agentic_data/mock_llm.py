"""Offline mock LLM backend — validates the whole agentic loop without API cost.

The mock challenger composes real executable candidates with the deterministic
chain builder (plus injected flaws to exercise every rejection path); the mock
weak solver under-calls, the mock strong solver returns the full gold trace,
the mock judge approves. Used by --mock smoke tests and unit tests ONLY; mock
outputs must never be shipped as a dataset (source field would still say so —
the builder refuses to write filtered/ files in mock mode unless asked).
"""
from __future__ import annotations

import json
import random
import re
from typing import Any, Dict, List

from ..synthetic_gen_v5 import DiversityConfig, _UsageBalancer, _build_chain, _question_from_phrases
from .schema import STAGES

_TASK_RE = re.compile(r"TASK:\n(.*)\Z", re.DOTALL)
_NCAND_RE = re.compile(r"propose (\d+) DIVERSE")
_CALLS_RE = re.compile(r"calls per task: (?:exactly (\d+)|between (\d+) and (\d+))")


class MockLLM:
    """Callable (role, messages) -> completion text, with candidate memory."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)
        self.gold_by_question: Dict[str, List[Dict[str, Any]]] = {}
        self.n_generated = 0
        self.balancer = _UsageBalancer(DiversityConfig())

    # ------------------------------------------------------------ challenger
    def _make_candidate(self, n_calls: int, motif: str) -> Dict[str, Any]:
        for _attempt in range(10):
            try:
                calls, observations, phrases = _build_chain(
                    self.rng, n_calls, motif, self.balancer)
                break
            except RuntimeError:
                continue
        else:
            # every attempt dead-ended — fall back to a trivial 1-call chain
            calls, observations, phrases = _build_chain(
                self.rng, 1, motif, self.balancer)
        question = _question_from_phrases(self.rng, phrases, len(calls))
        self.n_generated += 1
        flaw = self.n_generated % 9
        if flaw == 7:      # exercise non_executable_gold_trace
            calls = [dict(c, arguments={"bogus_param": 1}) for c in calls[:1]] \
                + calls[1:]
        elif flaw == 8:    # exercise invalid_schema (call count off)
            calls = calls[:1]
        else:
            self.gold_by_question[" ".join(question.lower().split())] = calls
        return {
            "question": question,
            "tool_names": [c["name"] for c in calls],
            "gold_calls": calls,
            "motif_type": motif,
            "answer_type": "scalar",
            "rationale": "chain of dependent computations",
        }

    def _challenger(self, user: str) -> str:
        n = int(_NCAND_RE.search(user).group(1))
        m = _CALLS_RE.search(user)
        if m.group(1):
            lo = hi = int(m.group(1))
        else:
            lo, hi = int(m.group(2)), int(m.group(3))
        motif = "long_chain"
        for stage_motif in ("argument_binding", "reference_reuse",
                            "distractor_heavy", "long_chain"):
            if stage_motif in user:
                motif = stage_motif
                break
        cands = [self._make_candidate(self.rng.randrange(lo, hi + 1), motif)
                 for _ in range(n)]
        return json.dumps({"candidates": cands})

    # ------------------------------------------------------------ solvers
    def _gold_for(self, user: str) -> List[Dict[str, Any]]:
        m = _TASK_RE.search(user)
        key = " ".join((m.group(1) if m else "").strip().lower().split())
        return self.gold_by_question.get(key, [])

    def _weak(self, user: str) -> str:
        gold = self._gold_for(user)
        # weak solver under-calls (this is the dominant real failure mode);
        # every 6th weak call solves the task to exercise weak_solver_passed
        if gold and self.rng.randrange(6) == 0:
            return json.dumps({"calls": gold, "final_answer": None})
        return json.dumps({"calls": gold[:1], "final_answer": None})

    def _strong(self, user: str) -> str:
        gold = self._gold_for(user)
        # strong succeeds most of the time; occasionally fails (too_hard path)
        if gold and self.rng.randrange(12) == 0:
            return json.dumps({"calls": gold[:-1], "final_answer": None})
        return json.dumps({"calls": gold, "final_answer": None})

    # ------------------------------------------------------------ dispatch
    def __call__(self, role: str, messages: List[Dict[str, str]]) -> str:
        user = messages[-1]["content"]
        if role == "challenger":
            return self._challenger(user)
        if role == "weak_solver":
            return self._weak(user)
        if role == "strong_solver":
            return self._strong(user)
        if role == "judge":
            return json.dumps({"nestful_like": True, "ambiguous": False,
                               "natural": True, "quality_score": 0.9,
                               "issues": []})
        raise ValueError(f"unknown role {role}")
