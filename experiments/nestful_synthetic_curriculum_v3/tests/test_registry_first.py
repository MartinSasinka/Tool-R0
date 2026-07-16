#!/usr/bin/env python3
"""Registry-first agentic generation smoke tests."""
from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.agentic_data.registry_first import (  # noqa: E402
    attach_polished_questions,
    generate_registry_skeletons,
)
from lib.agentic_data.semantics import semantic_errors  # noqa: E402
from lib.agentic_data.trace_validation import hard_trace_errors  # noqa: E402
from lib.agentic_data.exec_bridge import TOOLS  # noqa: E402
from lib.agentic_data.schema import STAGES  # noqa: E402
from lib.synthetic_gen_v5 import DiversityConfig, _UsageBalancer, execute_gold_calls  # noqa: E402

FAILURES: list[str] = []


def check(name: str, cond: bool, detail=None) -> None:
    if not cond:
        FAILURES.append(f"{name}: {detail}")


rng = random.Random(99)
balancer = _UsageBalancer(DiversityConfig())
stage = "stage2_2call_agentic_openrouter"
skeletons = generate_registry_skeletons(
    stage, "long_chain", 3, rng, balancer)
check("registry_first: produced skeletons", len(skeletons) >= 1, len(skeletons))
for sk in skeletons:
    calls = sk["gold_calls"]
    check("registry_first: replay executes",
          execute_gold_calls(calls)[-1] == sk["gold_answer"])
    check("registry_first: no semantic errors",
          not semantic_errors(calls, TOOLS), semantic_errors(calls, TOOLS))
    check("registry_first: trace labels valid",
          not hard_trace_errors({"gold_calls": calls}, TOOLS, STAGES[stage]))

polished = attach_polished_questions(skeletons, {
    "candidates": [{"index": i, "question": f"Natural question {i}?"}
                   for i in range(len(skeletons))]})
check("registry_first: polish merge", len(polished) == len(skeletons), polished)

if FAILURES:
    print(f"{len(FAILURES)} FAILURES: {FAILURES}")
    sys.exit(1)
print("ALL REGISTRY_FIRST TESTS PASSED")
