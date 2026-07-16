"""Unit tests for multi-turn solver-gap (no GPU, stub generate_fn)."""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
V3_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, V3_ROOT)

from lib.agentic_data.multiturn_solver import (  # noqa: E402
    reset_solver_context_cache, run_solver_episode, solver_gap_mode,
    solve_weak_multiturn)
from lib.agentic_data.quality import solver_gap_verdict  # noqa: E402

os.environ["AGENTIC_ACCEPTANCE_POLICY"] = "solver_gap"

FAILURES: list[str] = []


def check(name: str, cond: bool, detail=None) -> None:
    if cond:
        print(f"[PASS] {name}")
    else:
        print(f"[FAIL] {name} — {detail}")
        FAILURES.append(name)


GOLD_CALLS = [
    {"name": "triangle_area", "arguments": {"base": 8, "height": 5},
     "label": "$var1"},
    {"name": "percentage_of",
     "arguments": {"part": "$var1.result$", "whole": 50}, "label": "$var2"},
]
OFFERED = [{"name": c["name"], "description": "d",
            "parameters": {"type": "object", "properties": {
                k: {"type": "number"} for k in (c.get("arguments") or {})},
                "required": list((c.get("arguments") or {}))}}
           for c in GOLD_CALLS]
GOLD_OBS = [20.0, 40.0]
STAGE = "stage2_2call_agentic_openrouter"


def _fmt(call):
    return ("<tool_call_answer>[" + json.dumps(
        {"name": call["name"], "arguments": call.get("arguments") or {}},
        ensure_ascii=False) + "]</tool_call_answer>")


def _stub_gen(behavior: str):
    state = {"turn": 0}

    def generate_fn(messages, max_new_tokens):  # noqa: ARG001
        i = state["turn"]
        state["turn"] += 1
        if behavior == "too_few":
            text = _fmt(GOLD_CALLS[0]) if i == 0 else "<tool_call_answer>[]</tool_call_answer>"
        elif behavior == "perfect":
            text = _fmt(GOLD_CALLS[i]) if i < len(GOLD_CALLS) else "<tool_call_answer>[]</tool_call_answer>"
        else:
            text = "not a tool call"
        return {"text": text, "prompt_tokens": 0, "completion_tokens": 0,
                "clipped": False, "prompt_overflow": False}

    return generate_fn


reset_solver_context_cache()
weak = run_solver_episode(
    "q", OFFERED, GOLD_CALLS, GOLD_OBS, 40.0, stage=STAGE,
    generate_fn=_stub_gen("too_few"), seed=1)
check("MT solver-gap: too_few episode has solver_mode multiturn",
      weak.get("solver_mode") == "multiturn", weak)
check("MT solver-gap: too_few scores below weak max (<=0.5)",
      weak["score"] <= 0.5, weak)
check("MT solver-gap: too_few has 1 pred call",
      weak.get("n_calls") == 1, weak)

strong = run_solver_episode(
    "q", OFFERED, GOLD_CALLS, GOLD_OBS, 40.0, stage=STAGE,
    generate_fn=_stub_gen("perfect"), seed=2)
check("MT solver-gap: perfect episode scores as win (>=0.999)",
      strong["score"] >= 0.999, strong)

ok, why = solver_gap_verdict(weak, strong)
check("MT solver-gap: too_few weak + perfect strong passes gap",
      ok is True and why is None, (ok, why))

bad_ok, bad_why = solver_gap_verdict(strong, strong)
check("MT solver-gap: both strong fails as too_easy",
      bad_ok is False and bad_why == "too_easy_both_solvers_pass",
      (bad_ok, bad_why))

os.environ["AGENTIC_SOLVER_GAP_MODE"] = "single_shot"
check("solver_gap_mode respects AGENTIC_SOLVER_GAP_MODE=single_shot",
      solver_gap_mode() == "single_shot")
os.environ.pop("AGENTIC_SOLVER_GAP_MODE", None)

_os_backend = os.environ.get("WEAK_SOLVER_BACKEND")
os.environ["WEAK_SOLVER_BACKEND"] = "local"
check("solver_gap_mode defaults to multiturn when local",
      solver_gap_mode() == "multiturn")

import lib.agentic_data.local_llm as _llm  # noqa: E402

class _StubSolver:
    def generate(self, messages, *, temperature, max_tokens, seed=None):
        turn = sum(1 for m in messages if m.get("role") == "assistant")
        if turn < 1:
            return _fmt(GOLD_CALLS[0])
        return "<tool_call_answer>[]</tool_call_answer>"

_old = _llm.get_local_weak_solver
_llm.get_local_weak_solver = lambda: _StubSolver()
try:
    reset_solver_context_cache()
    w = solve_weak_multiturn(
        "q", OFFERED, GOLD_CALLS, GOLD_OBS, 40.0, stage=STAGE, seed=3)
    check("solve_weak_multiturn end-to-end via stub local solver",
          w.get("solver_mode") == "multiturn" and w.get("n_calls") == 1, w)
finally:
    _llm.get_local_weak_solver = _old
    if _os_backend is None:
        os.environ.pop("WEAK_SOLVER_BACKEND", None)
    else:
        os.environ["WEAK_SOLVER_BACKEND"] = _os_backend

if __name__ == "__main__":
    if FAILURES:
        print(f"\n{len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    print("\nALL TESTS PASSED")
