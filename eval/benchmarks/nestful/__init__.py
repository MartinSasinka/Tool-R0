"""NESTFUL benchmark — nested API call evaluation.

Public surface:
    run                 -- mode dispatcher (structural / execute / multiturn)
    load_tasks          -- HF dataset loader returning normalized task dicts
    build_user_content  -- prompt builder shared by every mode

Helpers re-exported for tests and external scripts:
    execute_call_sequence  -- pure-Python NESTFUL interpreter
    PRIMITIVES             -- math primitive dispatch table
    IBMFunctionRegistry    -- lazy loader for IBM/NESTFUL Python funcs
    evaluate_with_llm      -- OpenAI judge fallback (off by default)
    run_multiturn_tasks    -- batched interactive loop
"""

from eval.benchmarks.nestful.executor import (
    PRIMITIVES,
    CallTrace,
    ExecutionResult,
    execute_call_sequence,
    execute_one,
    is_known_primitive,
    resolve_variables,
)
from eval.benchmarks.nestful.ibm_loader import (
    IBMFunctionRegistry,
    call_with_signature_match,
)
from eval.benchmarks.nestful.judge import (
    JudgeResult,
    evaluate_with_llm,
    get_cache,
)
from eval.benchmarks.nestful.loader import build_user_content, load_tasks
from eval.benchmarks.nestful.multiturn import MultiTurnResult, run_multiturn_tasks
from eval.benchmarks.nestful.runner import run

__all__ = [
    "run",
    "load_tasks",
    "build_user_content",
    "PRIMITIVES",
    "CallTrace",
    "ExecutionResult",
    "execute_call_sequence",
    "execute_one",
    "is_known_primitive",
    "resolve_variables",
    "IBMFunctionRegistry",
    "call_with_signature_match",
    "JudgeResult",
    "evaluate_with_llm",
    "get_cache",
    "MultiTurnResult",
    "run_multiturn_tasks",
]
