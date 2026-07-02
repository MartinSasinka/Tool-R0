"""
Multi-turn interactive evaluation loop for NESTFUL.

Instead of asking the model for the entire call sequence at once (the
``structural`` and ``execute`` modes), this driver runs an agentic
back-and-forth:

1. Send ``[system, user(question + tools)]`` to the model.
2. Parse the *first* tool call from the model's reply.
3. Execute it locally via :func:`executor.execute_one` (math primitives +
   variable resolver).
4. Append the assistant turn and a synthetic
   ``<tool_response>{"name": ..., "result": ...}</tool_response>`` user
   turn that carries the executed result back into context.
5. Repeat until the model emits no more tool calls, an explicit
   ``<final_answer>X</final_answer>``, the step limit is hit, or the
   prompt no longer fits inside ``max_model_len``.

The driver batches *all currently-active conversations* per turn, so vLLM
sees a healthy batch on each call rather than a stream of size-1 prompts.
This keeps wall-clock cost roughly proportional to the *deepest* task
chain, not the number of tasks.
"""

from __future__ import annotations

import dataclasses
import json
import re
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from eval.benchmarks.nestful.executor import (
    CallTrace,
    coerce_numeric,
    execute_one,
)

if TYPE_CHECKING:
    from eval.benchmarks.nestful.ibm_loader import IBMFunctionRegistry
from eval.benchmarks.nestful.loader import build_user_content
from eval.model_adapter import (
    TOOL_R0_SYSTEM_PROMPT,
    _apply_chat_template,
    generate,
)
from eval.parse_utils import parse_tool_calls


_FINAL_ANSWER_RE = re.compile(
    r"<final_answer>(.*?)</final_answer>", re.DOTALL | re.IGNORECASE
)
_NUMBER_IN_TEXT_RE = re.compile(r"-?\d+(?:\.\d+)?")


@dataclasses.dataclass
class MultiTurnResult:
    task_id: str
    final_text: str
    final_value: Any
    stopped: str
    num_steps: int
    traces: List[CallTrace]
    raw_completions: List[str]
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "final_text": self.final_text,
            "final_value": self.final_value,
            "stopped": self.stopped,
            "num_steps": self.num_steps,
            "traces": [t.to_dict() for t in self.traces],
            "raw_completions": self.raw_completions,
            "error": self.error,
        }


def _extract_numeric(text: str) -> Any:
    """Last resort: pull the final number out of free-text."""
    if not text:
        return None
    matches = _NUMBER_IN_TEXT_RE.findall(text)
    if not matches:
        return text.strip()
    return coerce_numeric(matches[-1])


def _format_tool_response(call: Dict[str, Any], result: Any) -> str:
    """Build the user-side tool response message injected after each call."""
    payload = {"name": call.get("name", ""), "result": result}
    return f"<tool_response>{json.dumps(payload, default=str, ensure_ascii=False)}</tool_response>"


class _TaskState:
    """Per-task mutable state during the batched loop."""

    def __init__(
        self,
        task: Dict[str, Any],
        system_prompt: str,
        *,
        ibm_registry: "Optional[IBMFunctionRegistry]" = None,
    ):
        self.task = task
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_user_content(task)},
        ]
        self.by_label: Dict[str, Any] = {}
        self.indexed: List[Any] = []
        self.traces: List[CallTrace] = []
        self.completions: List[str] = []
        self.done: bool = False
        self.stop_reason: Optional[str] = None
        self.final_text: str = ""
        self.final_value: Any = None
        self.error: Optional[str] = None
        self._ibm_registry = ibm_registry

    def advance(self, completion: str) -> None:
        """Process one model turn and update state in place."""
        self.completions.append(completion)

        m = _FINAL_ANSWER_RE.search(completion)
        if m:
            text = m.group(1).strip()
            self.final_text = text
            self.final_value = coerce_numeric(text)
            self.done = True
            self.stop_reason = "explicit_final"
            return

        calls, _ = parse_tool_calls(completion)
        if not calls:
            self.final_text = completion.strip()
            self.final_value = self.indexed[-1] if self.indexed else _extract_numeric(self.final_text)
            self.done = True
            self.stop_reason = "no_more_calls"
            return

        call = calls[0]
        idx = len(self.traces)
        trace = execute_one(
            call,
            self.by_label,
            self.indexed,
            index=idx,
            ibm_registry=self._ibm_registry,
        )
        self.traces.append(trace)

        if trace.error is not None:
            self.done = True
            self.stop_reason = "execution_error"
            self.error = trace.error
            self.final_value = self.indexed[-1] if self.indexed else None
            return

        self.by_label[trace.label] = trace.result
        self.indexed.append(trace.result)
        self.final_value = trace.result

        self.messages.append({"role": "assistant", "content": completion})
        self.messages.append({
            "role": "user",
            "content": _format_tool_response(call, trace.result),
        })

    def to_result(self) -> MultiTurnResult:
        return MultiTurnResult(
            task_id=self.task["task_id"],
            final_text=self.final_text,
            final_value=self.final_value,
            stopped=self.stop_reason or "step_limit",
            num_steps=len(self.completions),
            traces=self.traces,
            raw_completions=self.completions,
            error=self.error,
        )


def _build_prompt(state: _TaskState, model_cfg: Dict[str, Any]) -> str:
    """Apply the tokenizer chat template to the conversation so far."""
    backend = model_cfg.get("backend", "vllm")
    if backend == "vllm":
        return _apply_chat_template(state.messages, model_cfg)
    if backend == "dummy":
        return json.dumps(state.messages, ensure_ascii=False)
    raise NotImplementedError(
        f"Multi-turn NESTFUL is currently only implemented for backends "
        f"'vllm' and 'dummy'. Got {backend!r}."
    )


def _approx_token_count(text: str) -> int:
    """Cheap upper bound on token count without re-tokenising the prompt."""
    return max(1, len(text) // 3)


def run_multiturn_tasks(
    tasks: List[Dict[str, Any]],
    model_cfg: Dict[str, Any],
    *,
    max_steps: int = 10,
    batch_size: int = 8,
    system_prompt: Optional[str] = None,
    ibm_registry: "Optional[IBMFunctionRegistry]" = None,
) -> List[MultiTurnResult]:
    """Run NESTFUL multi-turn evaluation over a list of tasks.

    Active tasks are batched together per turn so vLLM sees one large
    batched call rather than many tiny ones; finished tasks drop out of
    the batch on the next turn.

    Pass *ibm_registry* to enable dispatch into the IBM/NESTFUL Python
    implementations when our math primitive set doesn't recognise a call.
    """
    sys_prompt = system_prompt or model_cfg.get("system_prompt", TOOL_R0_SYSTEM_PROMPT)
    states: List[_TaskState] = [
        _TaskState(t, sys_prompt, ibm_registry=ibm_registry) for t in tasks
    ]
    max_model_len = int(model_cfg.get("max_model_len", 4096))
    safety_margin = int(model_cfg.get("max_new_tokens", 1024))

    for step in range(max_steps):
        active: List[_TaskState] = [s for s in states if not s.done]
        if not active:
            break

        prompts: List[str] = []
        skipped_for_context: List[_TaskState] = []
        for s in active:
            try:
                p = _build_prompt(s, model_cfg)
            except NotImplementedError:
                raise
            except Exception as exc:
                s.done = True
                s.stop_reason = "prompt_build_error"
                s.error = f"{type(exc).__name__}:{exc}"
                continue
            if _approx_token_count(p) + safety_margin > max_model_len:
                s.done = True
                s.stop_reason = "context_limit"
                continue
            prompts.append(p)
            skipped_for_context.append(s)

        if not prompts:
            continue

        completions = generate(prompts, model_cfg, batch_size=batch_size)
        for s, completion in zip(skipped_for_context, completions):
            try:
                s.advance(completion)
            except Exception as exc:
                s.done = True
                s.stop_reason = "advance_error"
                s.error = f"{type(exc).__name__}:{exc}"

    for s in states:
        if not s.done:
            s.done = True
            s.stop_reason = "step_limit"
            if s.final_value is None and s.indexed:
                s.final_value = s.indexed[-1]

    return [s.to_result() for s in states]
