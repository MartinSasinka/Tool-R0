"""Shared tool-agent interaction budgets and episode loop helpers (P43).

Semantics
---------
``max_turns`` historically meant **assistant generations** (not tool calls).
Training used ``max_turns = gold_n`` (no final ``[]`` turn); eval used
``gold_n + max_extra_turns_eval`` (+1), which accidentally reserved a final turn.

This module makes the split explicit:

* ``max_tool_calls`` — how many tool actions may be **executed**
* ``max_assistant_turns`` — how many assistant generations are allowed
  (``max_tool_calls + 1`` when ``reserve_final_answer_turn`` is true)

After the tool budget is exhausted the runner still performs one assistant
generation (FINAL_RESPONSE_MODE). New tool calls in that turn are **not**
executed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from parser import parse_tool_call
from prompt import build_messages, format_tool_response
from rollout import Trajectory, Turn


@dataclass
class InteractionBudget:
    gold_calls: int
    max_tool_calls: int
    max_assistant_turns: int
    tool_call_slack: int
    reserve_final_answer_turn: bool
    mode: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "gold_calls": self.gold_calls,
            "max_tool_calls": self.max_tool_calls,
            "max_assistant_turns": self.max_assistant_turns,
            "tool_call_slack": self.tool_call_slack,
            "reserve_final_answer_turn": self.reserve_final_answer_turn,
            "mode": self.mode,
        }


@dataclass
class InteractionMeta:
    tool_calls_emitted: int = 0
    tool_calls_executed: int = 0
    assistant_turns: int = 0
    tool_budget: int = 0
    assistant_turn_budget: int = 0
    final_response_turn_attempted: bool = False
    final_answer_present: bool = False
    final_turn_tool_attempt: bool = False
    stop_reason: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "tool_calls_emitted": self.tool_calls_emitted,
            "tool_calls_executed": self.tool_calls_executed,
            "assistant_turns": self.assistant_turns,
            "tool_budget": self.tool_budget,
            "assistant_turn_budget": self.assistant_turn_budget,
            "final_response_turn_attempted": self.final_response_turn_attempted,
            "final_answer_present": self.final_answer_present,
            "final_turn_tool_attempt": self.final_turn_tool_attempt,
            "stop_reason": self.stop_reason,
        }


def derive_interaction_budget(
    gold_call_count: int,
    config: Optional[Dict[str, Any]] = None,
    *,
    mode: str = "train",
) -> InteractionBudget:
    """Derive tool vs assistant budgets.

    Tool slack preserves existing knobs:
    * train: ``train.max_extra_turns_train`` (default 0) — extra **tools**, not final
    * eval/smoke: tool slack 0 by default (``max_extra_turns_eval`` is the
      final-answer reserve, matching historical eval ``gold_n + 1`` assistant turns)

    Hard safety: tool budget capped at ``gold_n + 4`` (legacy).
    """
    config = config or {}
    gen = config.get("generation") or {}
    train = config.get("train") or config.get("training") or {}
    inter = config.get("interaction") or {}

    gold_n = max(0, int(gold_call_count))
    reserve = bool(inter.get("reserve_final_answer_turn", True))
    if "reserve_final_answer_turn" in gen:
        reserve = bool(gen.get("reserve_final_answer_turn"))

    eval_like = mode in ("eval", "smoke")
    if eval_like:
        tool_slack = int(inter.get("tool_call_slack", 0) or 0)
        # Historical max_extra_turns_eval=1 == final-answer opportunity.
        final_extra = int(gen.get("max_extra_turns_eval", 1) or 0) if reserve else 0
    else:
        tool_slack = int(
            inter.get("tool_call_slack",
                      train.get("max_extra_turns_train", 0)) or 0)
        final_extra = 1 if reserve else 0

    max_tools = gold_n + max(0, tool_slack)
    max_tools = max(0, min(max_tools, gold_n + 4))
    max_assistant = max_tools + max(0, final_extra)
    if max_assistant < 1:
        max_assistant = 1

    if reserve and max_assistant <= max_tools:
        raise RuntimeError(
            f"[interaction] HARD ERROR: reserve_final_answer_turn=true but "
            f"assistant_turn_budget ({max_assistant}) <= tool_call_budget "
            f"({max_tools}) for gold_calls={gold_n} mode={mode}")

    return InteractionBudget(
        gold_calls=gold_n,
        max_tool_calls=max_tools,
        max_assistant_turns=max_assistant,
        tool_call_slack=tool_slack,
        reserve_final_answer_turn=reserve,
        mode=mode,
    )


def log_budget_mapping(config: Optional[Dict[str, Any]] = None,
                       *, mode: str = "train",
                       gold_range: range = range(2, 11)) -> List[Dict[str, Any]]:
    rows = []
    for g in gold_range:
        b = derive_interaction_budget(g, config, mode=mode)
        rows.append(b.as_dict())
    return rows


def assert_p43_interaction_config(config: Dict[str, Any]) -> None:
    """Fail-fast at P43 training startup."""
    inter = config.get("interaction") or {}
    # Default True when block missing — but explicit false is a hard error for P43.
    if inter.get("reserve_final_answer_turn") is False:
        raise RuntimeError(
            "[interaction] P43 requires interaction.reserve_final_answer_turn=true")
    # Validate budgets for all P43 call lengths.
    for g in range(2, 11):
        derive_interaction_budget(g, config, mode="train")
        derive_interaction_budget(g, config, mode="eval")


def attach_budget_fields(traj: Trajectory, budget: InteractionBudget,
                         meta: InteractionMeta) -> None:
    traj.tool_budget = budget.max_tool_calls  # type: ignore[attr-defined]
    traj.assistant_turn_budget = budget.max_assistant_turns  # type: ignore[attr-defined]
    traj.interaction_meta = meta.as_dict()  # type: ignore[attr-defined]


GenerateFn = Callable[[List[Dict[str, str]], int], Dict[str, Any]]
OnEncoded = Optional[Callable[[Turn, List[int], List[int]], None]]


def run_tool_agent_loop(
    *,
    task: Dict[str, Any],
    config: Dict[str, Any],
    executor,
    traj: Trajectory,
    history: List[Dict[str, str]],
    generate_fn: GenerateFn,
    max_new_tokens: int,
    budget: InteractionBudget,
    n_forced: int = 0,
    lenient_parse: bool = False,
    eval_hardening: bool = False,
    encode_fn: Optional[Callable[[List[Dict[str, str]], str],
                                 Tuple[List[int], List[int]]]] = None,
    on_generated_turn: Optional[Callable[[Turn], None]] = None,
) -> InteractionMeta:
    """Run assistant↔tool loop with separated tool / final-answer budgets.

    ``encode_fn(messages, text) -> (prompt_ids, completion_ids)`` is optional;
    when provided, ``on_generated_turn`` is invoked after each generated Turn
    (including those that stop early) so trainers can collect log-prob tokens.
    """
    meta = InteractionMeta(
        tool_budget=budget.max_tool_calls,
        assistant_turn_budget=budget.max_assistant_turns,
    )
    remaining = max(1, budget.max_assistant_turns - n_forced)

    for step in range(remaining):
        turn_idx = n_forced + step
        messages = build_messages(task, history, eval_hardening=eval_hardening)
        g = generate_fn(messages, max_new_tokens)
        meta.assistant_turns += 1

        if g.get("prompt_overflow"):
            traj.prompt_overflow = True
            traj.clipped_any = True
            traj.stop_reason = "prompt_overflow"
            meta.stop_reason = "prompt_overflow"
            if eval_hardening:
                # Eval diagnostics record the overflow attempt; train omits it
                # so r_seq stays aligned with turn_token_ids.
                turn = Turn(turn_idx, g.get("text") or "",
                            fail_reason="prompt_overflow")
                traj.turns.append(turn)
            break

        text = g.get("text") or ""
        clipped = bool(g.get("clipped", False))
        p_ids: List[int] = []
        c_ids: List[int] = []
        if encode_fn is not None:
            p_ids, c_ids = encode_fn(messages, text)
            turn = Turn(turn_idx, text, prompt_tokens=len(p_ids),
                        completion_tokens=len(c_ids), clipped_completion=clipped)
        else:
            turn = Turn(
                turn_idx, text,
                prompt_tokens=int(g.get("prompt_tokens") or 0),
                completion_tokens=int(g.get("completion_tokens") or 0),
                clipped_completion=clipped,
            )

        history.append({"role": "assistant", "content": text})

        if clipped:
            traj.clipped_any = True
            turn.fail_reason = "clipped_completion"
            traj.turns.append(turn)
            if on_generated_turn is not None:
                on_generated_turn(turn)
            traj.stop_reason = "clipped"
            meta.stop_reason = "clipped"
            break

        # Final-response phase: tool budget already exhausted before this turn.
        in_final_phase = meta.tool_calls_executed >= budget.max_tool_calls
        if in_final_phase:
            meta.final_response_turn_attempted = True

        pr = parse_tool_call(text, lenient=lenient_parse)
        if pr.is_terminal:
            turn.is_terminal = True
            traj.turns.append(turn)
            if on_generated_turn is not None:
                on_generated_turn(turn)
            traj.stop_reason = "terminal_answer"
            meta.stop_reason = "terminal_answer"
            meta.final_answer_present = True
            if in_final_phase:
                # Still a valid final after tools.
                pass
            break

        if not pr.ok:
            turn.fail_reason = f"parse:{pr.reason}"
            traj.turns.append(turn)
            if on_generated_turn is not None:
                on_generated_turn(turn)
            if eval_hardening:
                history.append({
                    "role": "user",
                    "content": "[tool error: could not parse a single valid tool call]",
                })
            traj.stop_reason = "parse_fail"
            meta.stop_reason = "parse_fail"
            break

        # Model emitted a tool call.
        call = pr.call
        turn.parsed_call = call
        meta.tool_calls_emitted += 1

        if in_final_phase:
            # Do NOT execute — budget exhausted; this was the final opportunity.
            turn.fail_reason = "final_turn_tool_attempt"
            meta.final_turn_tool_attempt = True
            traj.turns.append(turn)
            if on_generated_turn is not None:
                on_generated_turn(turn)
            traj.stop_reason = "final_turn_tool_attempt"
            meta.stop_reason = "final_turn_tool_attempt"
            break

        if meta.tool_calls_executed >= budget.max_tool_calls:
            # Should be unreachable (in_final_phase covers it); belt-and-suspenders.
            turn.fail_reason = "max_tool_calls"
            meta.final_turn_tool_attempt = True
            meta.final_response_turn_attempted = True
            traj.turns.append(turn)
            if on_generated_turn is not None:
                on_generated_turn(turn)
            traj.stop_reason = "max_tool_calls"
            meta.stop_reason = "max_tool_calls"
            break

        res = executor.execute(call)
        turn.observation = res.observation
        if res.error is not None:
            turn.fail_reason = f"exec:{res.error}"
            traj.turns.append(turn)
            if on_generated_turn is not None:
                on_generated_turn(turn)
            if eval_hardening:
                history.append({
                    "role": "user",
                    "content": format_tool_response(
                        call, f"[error: {res.error}]"),
                })
            traj.stop_reason = "executor_error"
            meta.stop_reason = "executor_error"
            break

        meta.tool_calls_executed += 1
        traj.final_observation = res.observation
        traj.turns.append(turn)
        if on_generated_turn is not None:
            on_generated_turn(turn)
        history.append({
            "role": "user",
            "content": format_tool_response(call, res.observation),
        })
        # Loop continues → if tools exhausted, next iteration is final phase.

    if traj.stop_reason is None:
        # Ran out of assistant turns without terminal.
        if (meta.tool_calls_executed >= budget.max_tool_calls
                and not meta.final_response_turn_attempted):
            # Should not happen if assistant_budget = tool_budget + 1.
            traj.stop_reason = "max_assistant_turns"
            meta.stop_reason = "max_assistant_turns"
        elif meta.final_response_turn_attempted:
            traj.stop_reason = "max_assistant_turns"
            meta.stop_reason = "max_assistant_turns"
        else:
            traj.stop_reason = "max_assistant_turns"
            meta.stop_reason = "max_assistant_turns"

    # Normalize legacy alias used in older logs.
    if traj.stop_reason == "terminal":
        traj.stop_reason = "terminal_answer"
        meta.stop_reason = "terminal_answer"

    meta.final_answer_present = bool(
        meta.final_answer_present
        or any(getattr(t, "is_terminal", False) for t in traj.turns)
    )
    attach_budget_fields(traj, budget, meta)
    return meta
