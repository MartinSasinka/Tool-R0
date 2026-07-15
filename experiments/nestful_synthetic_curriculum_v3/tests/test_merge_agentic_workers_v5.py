"""v5 counterpart of test_merge_agentic_workers.py — same merge-logic
coverage, but exercises merge_agentic_workers_v5.py (hashing helpers from
lib.agentic_data.exec_bridge / the v5 synthetic_tools registry, and
agentic_v5_<short>_NNNNNN sample_id renumbering).

Covers:
  1. cross-worker duplicate (same question_hash) is dropped, first worker wins;
  2. cross-worker duplicate (same trace_hash, different question) is dropped;
  3. distinct rows across workers are all kept;
  4. sample_id is renumbered sequentially agentic_v5_<short>_NNNNNN with no
     collisions, regardless of the original (colliding) worker-local ids;
  5. --max-rows-per-stage truncates AFTER dedup, keeping worker order.

Run:  python experiments/nestful_synthetic_curriculum_v3/tests/test_merge_agentic_workers_v5.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
V3_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, V3_ROOT)
sys.path.insert(0, os.path.join(V3_ROOT, "scripts", "data"))

from lib.agentic_data.schema import STAGE_FILES  # noqa: E402
from merge_agentic_workers_v5 import merge_stage  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}{(' — ' + str(detail)) if detail and not cond else ''}")
    if not cond:
        FAILURES.append(name)


STAGE = "stage2_2call_agentic_openrouter"


def _row(sample_id: str, question: str, calls):
    return {
        "sample_id": sample_id,
        "question": question,
        "tools": [],
        "gold_calls": calls,
        "gold_answer": 1,
    }


CALLS_A = [{"name": "toolA", "arguments": {"x": 1}, "label": "$var1"},
          {"name": "toolB", "arguments": {"y": "$var1$"}, "label": "$var2"}]
CALLS_B = [{"name": "toolC", "arguments": {"x": 2}, "label": "$var1"},
          {"name": "toolD", "arguments": {"y": "$var1$"}, "label": "$var2"}]
CALLS_C = [{"name": "toolE", "arguments": {"x": 3}, "label": "$var1"},
          {"name": "toolF", "arguments": {"y": "$var1$"}, "label": "$var2"}]

tmp = tempfile.mkdtemp(prefix="merge_agentic_test_")
try:
    gpu0 = os.path.join(tmp, "gpu0")
    gpu1 = os.path.join(tmp, "gpu1")
    os.makedirs(os.path.join(gpu0, "filtered"))
    os.makedirs(os.path.join(gpu1, "filtered"))

    # gpu0: two rows, both worker-local ids start at 000001 (collides with gpu1)
    rows0 = [
        _row("agentic_v5_stage2_000001", "What is the total of A and B?", CALLS_A),
        _row("agentic_v5_stage2_000002", "How much fuel is left after the trip?",
            CALLS_B),
    ]
    # gpu1: one exact question duplicate of gpu0 row 1 (different id),
    # one trace duplicate (different question text, same calls as CALLS_B),
    # one genuinely new row.
    rows1 = [
        _row("agentic_v5_stage2_000001", "What is the total of A and B?", CALLS_C),
        _row("agentic_v5_stage2_000002",
            "How much fuel remains once the trip finishes?", CALLS_B),
        _row("agentic_v5_stage2_000003", "What is the average speed?", CALLS_C),
    ]

    with open(os.path.join(gpu0, "filtered", STAGE_FILES[STAGE]), "w",
             encoding="utf-8") as fh:
        for r in rows0:
            fh.write(json.dumps(r) + "\n")
    with open(os.path.join(gpu1, "filtered", STAGE_FILES[STAGE]), "w",
             encoding="utf-8") as fh:
        for r in rows1:
            fh.write(json.dumps(r) + "\n")

    merged, report = merge_stage(STAGE, [gpu0, gpu1])

    check("question-hash duplicate dropped (gpu1 row 1)",
         report["total_dropped_as_duplicate"] >= 1, report)
    check("trace-hash duplicate dropped (gpu1 row 2, CALLS_B reused)",
         report["total_dropped_as_duplicate"] == 2, report)
    check("distinct rows kept: gpu0 x2 + gpu1's 1 new row = 3",
         len(merged) == 3, len(merged))
    check("first-worker-wins: kept row for dup question is gpu0's CALLS_A",
         merged[0]["gold_calls"] == CALLS_A, merged[0])

    ids = [r["sample_id"] for r in merged]
    short_ids_ok = all(sid.startswith("agentic_v5_stage2_") for sid in ids)
    check("renumbered ids all use agentic_v5_stage2_ prefix", short_ids_ok, ids)
    check("renumbered ids are sequential 000001..N with no collisions",
         ids == [f"agentic_v5_stage2_{i + 1:06d}" for i in range(len(merged))],
         ids)
    check("no duplicate sample_ids after renumber",
         len(set(ids)) == len(ids), ids)

    check("per-worker report totals sum correctly",
         report["total_loaded"] == 5 and report["total_kept"] == 3, report)

    merged_capped, report_capped = merge_stage(STAGE, [gpu0, gpu1], max_rows=2)
    check("--max-rows-per-stage truncates after dedup", len(merged_capped) == 2,
         len(merged_capped))
    check("truncation keeps worker order (gpu0's two rows first)",
         merged_capped[0]["gold_calls"] == CALLS_A
         and merged_capped[1]["gold_calls"] == CALLS_B,
         merged_capped)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print()
if FAILURES:
    print(f"FAILED: {len(FAILURES)} — {FAILURES}")
    sys.exit(1)
print("ALL TESTS PASSED")
