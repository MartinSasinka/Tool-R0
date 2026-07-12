"""Offline regression tests for the 2026-07-11 hardening audit (no network,
no GPU): hard trace-structure validation, semantic compatibility, offered-
tool-count scaling, and the multi-rollout GRPO-signal probe.

The two malformed candidates below are LITERAL reproductions of real
accepted pilot rows (agentic_v4_stage2_000007 / _000008) and the three
semantically-unnatural rows (_000003 / _000006 / _000009) that motivated this
module — see docstrings in trace_validation.py / semantics.py.

Run:  python experiments/nestful_synthetic_curriculum_v3/tests/test_agentic_hardening.py
"""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
V3_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, V3_ROOT)
sys.path.insert(0, os.path.join(V3_ROOT, "scripts", "data"))

from lib.nestful_like_generator import TOOLS  # noqa: E402
from lib.agentic_data.trace_validation import (  # noqa: E402
    hard_trace_errors, label_errors, reference_errors)
from lib.agentic_data.semantics import semantic_errors  # noqa: E402
from lib.agentic_data.orchestrator import _offered_schemas  # noqa: E402
import random  # noqa: E402
from lib.agentic_data.rollout_signal import (  # noqa: E402
    summarize_rollouts, run_rollouts, probe_rollout_signal, target_is_local)
from lib.agentic_data.training_reward import (  # noqa: E402
    build_task_dict, score_with_training_reward, configured_reward_policy)

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}{(' — ' + str(detail)) if detail and not cond else ''}")
    if not cond:
        FAILURES.append(name)


# ============================================================ trace validation
# LITERAL reproduction of agentic_v4_stage2_000007: mean_of_values -> $var1,
# percentage_of -> $var1 (label reused instead of $var2).
MALFORMED_000007 = {
    "gold_calls": [
        {"name": "mean_of_values", "arguments": {"values": [8, 12, 15, 3, 7]},
         "label": "$var1"},
        {"name": "percentage_of",
         "arguments": {"part": "$var1.output_0$", "whole": 20},
         "label": "$var1"},
    ],
    "tool_names": ["mean_of_values", "percentage_of"],
}

# LITERAL reproduction of agentic_v4_stage2_000008: rectangle_perimeter ->
# $var1, scale_dimension -> $var1 (same duplicate-label defect).
MALFORMED_000008 = {
    "gold_calls": [
        {"name": "rectangle_perimeter", "arguments": {"length": 8, "width": 5},
         "label": "$var1"},
        {"name": "scale_dimension",
         "arguments": {"dimension": "$var1.result$", "scale_factor": 1.5},
         "label": "$var1"},
    ],
    "tool_names": ["rectangle_perimeter", "scale_dimension"],
}

WELL_FORMED = {
    "gold_calls": [
        {"name": "triangle_area", "arguments": {"base": 8, "height": 5},
         "label": "$var1"},
        {"name": "percentage_of",
         "arguments": {"part": "$var1.result$", "whole": 50}, "label": "$var2"},
    ],
    "tool_names": ["triangle_area", "percentage_of"],
}

for name, cand in (("000007", MALFORMED_000007), ("000008", MALFORMED_000008)):
    errs = hard_trace_errors(cand, TOOLS, (2, 2))
    check(f"hard_trace_errors flags duplicate-label pilot row {name}",
         len(errs) > 0, errs)
    check(f"label_errors reports the reuse for {name}",
         any("reused" in e for e in label_errors(cand["gold_calls"])),
         label_errors(cand["gold_calls"]))

check("hard_trace_errors accepts a well-formed sequential trace",
     hard_trace_errors(WELL_FORMED, TOOLS, (2, 2)) == [],
     hard_trace_errors(WELL_FORMED, TOOLS, (2, 2)))

# non-sequential (first label wrong)
BAD_SEQUENCE = {
    "gold_calls": [
        {"name": "triangle_area", "arguments": {"base": 8, "height": 5},
         "label": "$var2"},
        {"name": "percentage_of",
         "arguments": {"part": "$var2.result$", "whole": 50}, "label": "$var1"},
    ],
    "tool_names": ["triangle_area", "percentage_of"],
}
check("hard_trace_errors flags a non-sequential first label",
     len(hard_trace_errors(BAD_SEQUENCE, TOOLS, (2, 2))) > 0)

# forward reference (call 1 references a label that doesn't exist yet)
FORWARD_REF = {
    "gold_calls": [
        {"name": "triangle_area",
         "arguments": {"base": "$var2.result$", "height": 5}, "label": "$var1"},
        {"name": "percentage_of",
         "arguments": {"part": "$var1.result$", "whole": 50}, "label": "$var2"},
    ],
    "tool_names": ["triangle_area", "percentage_of"],
}
check("reference_errors flags a forward/unknown reference",
     len(reference_errors(FORWARD_REF["gold_calls"], TOOLS)) > 0)

# wrong output field (producer outputs .result$, reference asks .output_0$)
WRONG_FIELD = {
    "gold_calls": [
        {"name": "triangle_area", "arguments": {"base": 8, "height": 5},
         "label": "$var1"},
        {"name": "percentage_of",
         "arguments": {"part": "$var1.output_0$", "whole": 50},
         "label": "$var2"},
    ],
}
check("reference_errors flags a wrong output-key reference",
     len(reference_errors(WRONG_FIELD["gold_calls"], TOOLS)) > 0)

# call-count / tool_names-length mismatch
COUNT_MISMATCH = {
    "gold_calls": WELL_FORMED["gold_calls"],
    "tool_names": ["triangle_area"],   # declares 1, has 2
}
check("call_count_errors flags tool_names/gold_calls length mismatch",
     len(hard_trace_errors(COUNT_MISMATCH, TOOLS, (2, 2))) > 0)


# ============================================================ semantic checks
# LITERAL reproduction of the 3 unnatural pilot rows.
FAHRENHEIT_AS_MONEY = {  # agentic_v4_stage2_000003
    "gold_calls": [
        {"name": "celsius_to_fahrenheit", "arguments": {"celsius": 100},
         "label": "$var1"},
        {"name": "add_sales_tax",
         "arguments": {"net_price": "$var1.value$", "tax_rate_percent": 15},
         "label": "$var2"},
    ],
}
FAHRENHEIT_AS_RATE = {  # agentic_v4_stage2_000006
    "gold_calls": [
        {"name": "celsius_to_fahrenheit", "arguments": {"celsius": 25},
         "label": "$var1"},
        {"name": "calculate_simple_interest",
         "arguments": {"principal_amount": 1000,
                       "annual_rate_percent": "$var1.value$", "years": 2},
         "label": "$var2"},
    ],
}
FUEL_AS_PRINCIPAL = {  # agentic_v4_stage2_000009
    "gold_calls": [
        {"name": "fuel_needed_liters",
         "arguments": {"distance_km": 450, "consumption_per_100km": 8.5},
         "label": "$var1"},
        {"name": "calculate_simple_interest",
         "arguments": {"principal_amount": "$var1.value$",
                       "annual_rate_percent": 5, "years": 3}, "label": "$var2"},
    ],
}
for label, cand in (("Fahrenheit->net_price (000003)", FAHRENHEIT_AS_MONEY),
                    ("Fahrenheit->annual_rate_percent (000006)", FAHRENHEIT_AS_RATE),
                    ("fuel_liters->principal_amount (000009)", FUEL_AS_PRINCIPAL)):
    errs = semantic_errors(cand["gold_calls"], TOOLS)
    check(f"semantic_errors rejects {label}", len(errs) > 0, errs)

# these must STAY accepted: generic (unit-agnostic) slots on either side
AREA_AS_PART = {"gold_calls": WELL_FORMED["gold_calls"]}          # area -> part
SUM_AS_TOTAL_AMOUNT = {  # agentic_v4_stage2_000002: generic producer -> money
    "gold_calls": [
        {"name": "sum_of_values", "arguments": {"values": [10, 20, 30, 40, 50]},
         "label": "$var1"},
        {"name": "split_bill_evenly",
         "arguments": {"total_amount": "$var1.output_0$", "num_people": 4},
         "label": "$var2"},
    ],
}
for label, cand in (("area used as generic 'part' (000001)", AREA_AS_PART),
                    ("generic sum used as money total (000002)",
                     SUM_AS_TOTAL_AMOUNT)):
    errs = semantic_errors(cand["gold_calls"], TOOLS)
    check(f"semantic_errors ACCEPTS {label}", errs == [], errs)

# same-family binding must be allowed (perimeter/length -> distance_km)
PERIMETER_AS_DISTANCE = {
    "gold_calls": [
        {"name": "rectangle_perimeter", "arguments": {"length": 12, "width": 8},
         "label": "$var1"},
        {"name": "travel_time_hours",
         "arguments": {"distance_km": "$var1.result$", "speed_kmh": 4},
         "label": "$var2"},
    ],
}
check("semantic_errors ACCEPTS same-family length->distance binding (000010)",
     semantic_errors(PERIMETER_AS_DISTANCE["gold_calls"], TOOLS) == [],
     semantic_errors(PERIMETER_AS_DISTANCE["gold_calls"], TOOLS))


# ============================================================ offered-tool scaling
rng = random.Random(7)
counts_2call = [len(_offered_schemas(rng, ["triangle_area", "percentage_of"],
                                     "long_chain", 2)) for _ in range(20)]
check("2-call offered-tool count stays within the reduced NESTFUL-scale range",
     all(6 <= c <= 11 for c in counts_2call), counts_2call)
check("2-call offered-tool count is well below the old flat 16-26 range",
     max(counts_2call) < 16, counts_2call)

counts_distractor = [len(_offered_schemas(rng, ["triangle_area", "percentage_of"],
                                          "distractor_heavy", 2))
                    for _ in range(20)]
check("distractor_heavy still offers more tools than a plain motif",
     sum(counts_distractor) / len(counts_distractor)
     > sum(counts_2call) / len(counts_2call))


# ============================================================ rollout signal
def _score(score, status, n_calls):
    return {"score": score, "status": status, "n_calls": n_calls}


ALL_WIN = [_score(1.0, "win", 2) for _ in range(8)]
summ = summarize_rollouts(ALL_WIN, n_gold_calls=2)
check("rollout signal: all-win task is NOT grpo-signal-positive (too easy)",
     summ["grpo_signal_positive"] is False, summ)
check("rollout signal: all-win has unique_rewards == 1",
     summ["unique_rewards"] == 1, summ)

ALL_PARSE_ERROR = [_score(0.0, "parse_error", 0) for _ in range(8)]
summ = summarize_rollouts(ALL_PARSE_ERROR, n_gold_calls=2)
check("rollout signal: parse-error-only is NOT grpo-signal-positive",
     summ["grpo_signal_positive"] is False, summ)
check("rollout signal: parse-error-only has has_valid_trace == False",
     summ["has_valid_trace"] is False, summ)
check("rollout signal: parse-error-only has all_degenerate == True",
     summ["all_degenerate"] is True, summ)

MIXED = [_score(1.0, "win", 2), _score(0.5, "correct_prefix_then_stop", 1),
        _score(0.0, "parse_error", 0), _score(0.5, "correct_prefix_then_stop", 1),
        _score(1.0, "win", 2), _score(0.0, "wrong_tool", 1),
        _score(0.5, "partial_prefix", 2), _score(0.0, "parse_error", 0)]
summ = summarize_rollouts(MIXED, n_gold_calls=2)
check("rollout signal: mixed rollouts ARE grpo-signal-positive",
     summ["grpo_signal_positive"] is True, summ)
check("rollout signal: mixed rollouts have unique_rewards >= 2",
     summ["unique_rewards"] >= 2, summ)
check("rollout signal: mixed rollouts have reward_variance > 0",
     summ["reward_variance"] > 0, summ)

# parse-error-heavy but ONE valid partial trace -> still signal-positive
ONE_VALID_AMONG_DEGENERATE = ([_score(0.0, "parse_error", 0) for _ in range(7)]
                             + [_score(0.5, "correct_prefix_then_stop", 1)])
summ = summarize_rollouts(ONE_VALID_AMONG_DEGENERATE, n_gold_calls=2)
check("rollout signal: one valid trace among 7 parse errors has_valid_trace True",
     summ["has_valid_trace"] is True, summ)
check("rollout signal: predicted_call_distribution recorded",
     summ["predicted_call_distribution"].get("0") == 7, summ)
check("rollout signal: failure_type_distribution recorded",
     summ["failure_type_distribution"].get("parse_error") == 7, summ)

_old_backend_probe = os.environ.get("WEAK_SOLVER_BACKEND")
os.environ.pop("WEAK_SOLVER_BACKEND", None)
try:
    check("probe_rollout_signal skips cleanly when WEAK_SOLVER_BACKEND != local",
         probe_rollout_signal("q", [], [], [], 1.0,
                              stage="stage2_2call_agentic_openrouter")
         .get("skipped") is True)
    check("target_is_local() reflects WEAK_SOLVER_BACKEND env var",
         target_is_local() is False)
finally:
    if _old_backend_probe is None:
        os.environ.pop("WEAK_SOLVER_BACKEND", None)
    else:
        os.environ["WEAK_SOLVER_BACKEND"] = _old_backend_probe


# ============================================================ training reward dispatch
check("default agentic rollout reward policy is v3.2 dense",
     configured_reward_policy() == "execution_aware_v3_2_dense")

_task = build_task_dict(
    gold_calls=WELL_FORMED["gold_calls"], gold_answer=40.0,
    stage="stage2_2call_agentic_openrouter", question="q")
_gold_obs = [20.0, 40.0]
_win = score_with_training_reward(
    WELL_FORMED["gold_calls"], 40.0, _task, _gold_obs,
    parsed={"calls": WELL_FORMED["gold_calls"], "final_answer": 40.0})
check("training reward scores a correct 2-call trace as fully_correct",
     _win["reward_class"] == "fully_correct" and _win["episode_reward"] >= 0.9,
     _win)
_too_few = score_with_training_reward(
    [WELL_FORMED["gold_calls"][0]], None, _task, _gold_obs,
    parsed={"calls": [WELL_FORMED["gold_calls"][0]], "final_answer": None})
check("training reward scores a 1-call prefix differently than a full win",
     _too_few["episode_reward"] < _win["episode_reward"]
     and _too_few["reward_class"] == "too_few_calls", (_too_few, _win))
_parse = score_with_training_reward(None, None, _task, _gold_obs, parsed=None)
check("training reward scores unparseable output as parse_error",
     _parse["reward_class"] == "parse_error" and _parse["episode_reward"] == 0.0,
     _parse)


# ---- end-to-end run_rollouts() against a stubbed local weak solver (no GPU)
class _StubMultiTurnSolver:
    """Emit gold tool calls one turn at a time (MT-GRPO format)."""

    def __init__(self, gold_calls: List[Dict[str, Any]]) -> None:
        self._gold = gold_calls

    def generate(self, messages, *, temperature, max_tokens, seed=None) -> str:
        turn = sum(1 for m in messages if m.get("role") == "assistant")
        if turn < len(self._gold):
            c = self._gold[turn]
            payload = json.dumps(
                {"name": c["name"], "arguments": c.get("arguments") or {}},
                ensure_ascii=False)
            return f"<tool_call_answer>[{payload}]</tool_call_answer>"
        return "<tool_call_answer>[]</tool_call_answer>"


import lib.agentic_data.local_llm as _local_llm_mod  # noqa: E402

_old_backend = os.environ.get("WEAK_SOLVER_BACKEND")
os.environ["WEAK_SOLVER_BACKEND"] = "local"
_old_get_solver = _local_llm_mod.get_local_weak_solver
_local_llm_mod.get_local_weak_solver = lambda: _StubMultiTurnSolver(
    WELL_FORMED["gold_calls"])
try:
    check("target_is_local() reflects WEAK_SOLVER_BACKEND=local", target_is_local())
    gold_calls = WELL_FORMED["gold_calls"]
    offered = [{"name": c["name"], "description": "d", "parameters": {
        k: {"type": "number"} for k in (c.get("arguments") or {})}}
               for c in gold_calls]
    scored = run_rollouts("q", offered, gold_calls, [20.0, 40.0], 40.0,
                          stage="stage2_2call_agentic_openrouter", n=4, seed=1)
    check("run_rollouts against a stubbed local solver returns n scores",
         len(scored) == 4, scored)
    check("run_rollouts multi-turn scores perfect gold trace with 2 pred calls",
         all(s.get("n_calls") == 2 for s in scored), scored)
    check("run_rollouts multi-turn gives consistent rewards on identical rollouts",
         len({round(s["episode_reward"], 6) for s in scored}) == 1, scored)
    full = probe_rollout_signal("q", offered, gold_calls, [20.0, 40.0], 40.0,
                                  stage="stage2_2call_agentic_openrouter",
                                  n=4, seed=1)
    check("probe_rollout_signal runs end-to-end against the stub (not skipped)",
         full.get("skipped") is False, full)
    check("probe_rollout_signal uses multiturn rollout mode by default",
         full.get("rollout_mode") == "multiturn", full)
finally:
    _local_llm_mod.get_local_weak_solver = _old_get_solver
    if _old_backend is None:
        os.environ.pop("WEAK_SOLVER_BACKEND", None)
    else:
        os.environ["WEAK_SOLVER_BACKEND"] = _old_backend


if __name__ == "__main__":
    if FAILURES:
        print(f"\n{len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    print("\nALL TESTS PASSED")
