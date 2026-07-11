"""Offline unit tests for the agentic data pipeline (no network, no GPU).

Run:  python experiments/nestful_synthetic_curriculum_v3/tests/test_agentic_data.py
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
V3_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, V3_ROOT)
sys.path.insert(0, os.path.join(V3_ROOT, "scripts", "data"))

from openrouter_client import extract_json  # noqa: E402
from lib.agentic_data.quality import solver_gap_verdict  # noqa: E402
from lib.agentic_data.schema import candidate_schema_errors, question_leak_errors  # noqa: E402
from lib.agentic_data.solvers import score_prediction  # noqa: E402
from lib.agentic_data.verifier import deterministic_verify  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}{(' — ' + str(detail)) if detail and not cond else ''}")
    if not cond:
        FAILURES.append(name)


# --------------------------------------------------------- extract_json
check("extract_json plain", extract_json('{"a": 1}') == {"a": 1})
check("extract_json fenced",
      extract_json('text\n```json\n{"a": 1}\n```\nmore') == {"a": 1})
check("extract_json trailing comma + prose",
      extract_json('Sure! {"a": [1, 2,], "b": 3,} done')["b"] == 3)
try:
    extract_json("no json here")
    check("extract_json raises on garbage", False)
except ValueError:
    check("extract_json raises on garbage", True)

# --------------------------------------------------------- gold trace verify
GOLD = [
    {"name": "rectangle_area", "arguments": {"length": 10, "width": 4},
     "label": "$var1"},
    {"name": "apply_discount",
     "arguments": {"price": "$var1.result$", "discount_percent": 10},
     "label": "$var2"},
]
v = deterministic_verify({"question": "Compute the area of a 10 by 4 rectangle, "
                                      "then apply a 10% discount to that value "
                                      "as if it were a price in dollars today.",
                          "gold_calls": GOLD})
check("verify executable 2-call chain", v["ok"], v)
check("verify gold answer computed", v["gold_answer"] == 36.0, v["gold_answer"])
GOLD_OBS = v["observations"]

bad = deterministic_verify({"question": "x " * 30,
                            "gold_calls": [{"name": "nope", "arguments": {}}]})
check("verify rejects unknown tool", bad["reason"] == "non_executable_gold_trace")

leak = deterministic_verify({"question": "In this stage of the curriculum, "
                                         "compute the area of a 10 by 4 "
                                         "rectangle and then stop working.",
                             "gold_calls": GOLD[:1]})
check("verify rejects metadata leakage", leak["reason"] == "metadata_leakage")

noref = deterministic_verify(
    {"question": "Compute the area of a 10 by 4 rectangle and separately "
                 "apply a 10% discount to a price of 40 dollars please now.",
     "gold_calls": [GOLD[0],
                    {"name": "apply_discount",
                     "arguments": {"price": 40, "discount_percent": 10},
                     "label": "$var2"}]})
check("verify rejects chains without $var refs", noref["reason"] == "invalid_schema")

# --------------------------------------------------------- schema gates
errs = candidate_schema_errors(
    {"question": "word " * 20, "gold_calls": GOLD,
     "tool_names": ["rectangle_area", "apply_discount"],
     "motif_type": "long_chain"}, "stage2_2call_agentic_openrouter")
check("schema accepts valid candidate", errs == [], errs)
errs = candidate_schema_errors(
    {"question": "word " * 20, "gold_calls": GOLD,
     "tool_names": ["rectangle_area"], "motif_type": "long_chain"},
    "stage3_3call_agentic_openrouter")
check("schema rejects wrong call count", any("call count" in e for e in errs))
check("leak check catches $var", question_leak_errors("use $var1.result$ now") != [])

# --------------------------------------------------------- solver scoring
s = score_prediction(GOLD, None, GOLD, GOLD_OBS, 36.0)
check("score: gold prediction wins", s["score"] == 1.0 and s["status"] == "win", s)
s = score_prediction(GOLD[:1], None, GOLD, GOLD_OBS, 36.0)
check("score: clean stop after correct prefix in 0.5-0.8",
      0.5 <= s["score"] <= 0.8 and s["status"] == "correct_prefix_then_stop", s)
s = score_prediction(None, None, GOLD, GOLD_OBS, 36.0)
check("score: parse error = 0", s["score"] == 0.0 and s["status"] == "parse_error")
s = score_prediction([], 36.0, GOLD, GOLD_OBS, 36.0)
check("score: direct answer capped at 0.2", s["score"] == 0.2, s)
s = score_prediction([{"name": "circle_area", "arguments": {"radius": 3},
                       "label": "$var1"}], None, GOLD, GOLD_OBS, 36.0)
check("score: wrong tool low", s["score"] <= 0.4, s)
s = score_prediction([GOLD[0], {"name": "apply_discount",
                                "arguments": {"price": "$var1.result$",
                                              "discount_percent": 20},
                                "label": "$var2"}], None, GOLD, GOLD_OBS, 36.0)
check("score: executable wrong final in 0.5-0.8 (correct prefix)",
      0.5 <= s["score"] <= 0.8, s)

# --------------------------------------------------------- gap policy
ok, why = solver_gap_verdict({"score": 0.5, "status": "under_call"},
                             {"score": 1.0, "status": "win"})
check("gap: weak-fail strong-pass accepted", ok, why)
ok, why = solver_gap_verdict({"score": 1.0, "status": "win"}, None)
check("gap: weak passed rejected", not ok and why == "weak_solver_passed")
ok, why = solver_gap_verdict({"score": 0.1, "status": "wrong_tool"},
                             {"score": 0.2, "status": "wrong_args"})
check("gap: both fail rejected", not ok and why == "too_hard_both_solvers_fail")
ok, why = solver_gap_verdict({"score": 0.5, "status": "under_call"},
                             {"score": 0.7, "status": "partial_prefix"})
check("gap: strong below 0.8 rejected", not ok, why)

# --------------------------------------------------------- new failure taxonomy
s = score_prediction(
    [{"name": "apply_discount",
      "arguments": {"price": "$var9.result$", "discount_percent": 10},
      "label": "$var1"}], None, GOLD, GOLD_OBS, 36.0)
check("score: unresolved reference -> invalid_reference",
      s["status"] == "invalid_reference" and s["score"] == 0.15, s)
s = score_prediction([{"name": "circle_area", "arguments": {"r": 3},
                       "label": "$var1"}], 36.0, GOLD, GOLD_OBS, 36.0)
check("score: right answer, broken trace -> correct_answer_wrong_trace",
      s["status"] == "correct_answer_wrong_trace" and s["score"] == 0.4, s)

# --------------------------------------------------------- strong exact-win policy
ok, why = solver_gap_verdict({"score": 0.5, "status": "under_call"},
                             {"score": 0.85, "status": "partial_prefix"})
check("gap: partial strong (0.85) rejected under exact_win policy",
      not ok and why == "strong_solver_failed", why)

# --------------------------------------------------------- diversity tracker
from lib.agentic_data.quality import DiversityTracker  # noqa: E402
dt = DiversityTracker(max_same_weak_score=0.4, max_same_failure_type=0.4,
                      enforce_after=5)
for _ in range(5):
    dt.add(0.5, "correct_prefix_then_stop")
check("diversity: cap blocks dominant weak-score bucket",
      dt.verdict(0.5, "wrong_args") == "diversity_cap_weak_score")
check("diversity: different bucket+type passes",
      dt.verdict(0.2, "wrong_args") is None)
dt2 = DiversityTracker(max_same_weak_score=0.9, max_same_failure_type=0.4,
                       enforce_after=5)
for _ in range(5):
    dt2.add(0.5, "correct_prefix_then_stop")
check("diversity: cap blocks dominant failure type",
      dt2.verdict(0.3, "correct_prefix_then_stop")
      == "diversity_cap_failure_type")
check("diversity: not enforced during warmup",
      DiversityTracker(enforce_after=50).verdict(0.5, "x") is None)

# resume: caps on NEW rows only — legacy seed must not block 0.50 bucket
dt_resume = DiversityTracker(resume_mode=True, max_same_weak_score=0.40,
                             enforce_after=1)
for _ in range(200):
    dt_resume.seed_reference_from_rows([{
        "solver_gap": {"weak_score": 0.5, "weak_status": "partial_prefix"}}])
check("diversity resume: seed does not count toward enforcement n",
      dt_resume.n == 0 and dt_resume.n_seed == 200)
check("diversity resume: 0.50 bucket allowed in new set after homogeneous seed",
      dt_resume.verdict(0.5, "wrong_args") is None)
for _ in range(4):
    dt_resume.add(0.5, "wrong_args")   # 4/4 = 100% of NEW -> at cap boundary
check("diversity resume: cap still applies within NEW rows",
      dt_resume.verdict(0.5, "wrong_tool") == "diversity_cap_weak_score")

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURES: {FAILURES}")
    sys.exit(1)
print("ALL TESTS PASSED")
