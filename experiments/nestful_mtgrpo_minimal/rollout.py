"""Online multi-turn rollout (no gold prefix).

This file is a minimal standalone reimplementation inspired by the original
project rollout (curricullum/train/evaluate_nestful_stage.py::rollout_task).
The model only ever sees its OWN previous turns + real executor observations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from parser import parse_tool_call
from prompt import build_messages, format_tool_response
from executor import ToolExecutor


@dataclass
class Turn:
    turn_idx: int
    model_text: str
    parsed_call: Optional[Dict[str, Any]] = None
    observation: Any = None
    fail_reason: Optional[str] = None
    is_terminal: bool = False
    # token diagnostics
    prompt_tokens: int = 0
    completion_tokens: int = 0
    clipped_completion: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_idx": self.turn_idx,
            "model_text": self.model_text,
            "parsed_call": self.parsed_call,
            "observation": self.observation,
            "fail_reason": self.fail_reason,
            "is_terminal": self.is_terminal,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "clipped_completion": self.clipped_completion,
        }


@dataclass
class Trajectory:
    task_id: str
    stage: int
    gold_num_turns: int
    turns: List[Turn] = field(default_factory=list)
    final_observation: Any = None
    stop_reason: Optional[str] = None
    executor_mode: str = "gold_replay"
    # token budget rollup
    clipped_any: bool = False
    prompt_overflow: bool = False

    @property
    def predicted_calls(self) -> List[Dict[str, Any]]:
        return [t.parsed_call for t in self.turns if t.parsed_call is not None]

    @property
    def num_tool_calls(self) -> int:
        return len(self.predicted_calls)

    @property
    def observations(self) -> List[Any]:
        return [
            t.observation
            for t in self.turns
            if t.parsed_call is not None and t.fail_reason is None
        ]

    @property
    def zero_tool_calls(self) -> bool:
        return self.num_tool_calls == 0

    @property
    def executor_error(self) -> bool:
        return any(
            t.fail_reason and t.fail_reason.startswith("exec:")
            for t in self.turns
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "stage": self.stage,
            "gold_num_turns": self.gold_num_turns,
            "stop_reason": self.stop_reason,
            "executor_mode": self.executor_mode,
            "num_tool_calls": self.num_tool_calls,
            "clipped_any": self.clipped_any,
            "prompt_overflow": self.prompt_overflow,
            "turns": [t.to_dict() for t in self.turns],
        }


# ---------------------------------------------------------------------------
#  Generation helper (transformers.generate; vLLM is opt-in elsewhere)
# ---------------------------------------------------------------------------

def generate_once(
    model,
    tokenizer,
    messages: List[Dict[str, str]],
    max_new_tokens: int,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_prompt_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """Single greedy/sampled generation. Returns text + token diagnostics."""
    import torch  # local import so non-train modes don't hard-require torch early

    prompt_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    )
    # transformers >=5.x returns a BatchEncoding instead of a plain tensor.
    if hasattr(prompt_ids, "input_ids"):
        prompt_ids = prompt_ids.input_ids
    prompt_len = int(prompt_ids.shape[1])
    prompt_overflow = bool(max_prompt_tokens and prompt_len > max_prompt_tokens)

    device = getattr(model, "device", None) or next(model.parameters()).device
    prompt_ids = prompt_ids.to(device)
    attn = torch.ones_like(prompt_ids)

    do_sample = temperature is not None and temperature > 0
    with torch.no_grad():
        out = model.generate(
            input_ids=prompt_ids,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    gen_ids = out[0][prompt_len:]
    completion_len = int(gen_ids.shape[0])
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    clipped = completion_len >= max_new_tokens
    return {
        "text": text,
        "prompt_tokens": prompt_len,
        "completion_tokens": completion_len,
        "clipped": clipped,
        "prompt_overflow": prompt_overflow,
    }


def get_stage_token_budget(config: Dict[str, Any], stage: int, mode: str) -> Dict[str, int]:
    """Resolve stage-aware token budget for a given run mode.

    Returns: {max_prompt_tokens, max_new_tokens, max_model_length, vllm_max_model_length}

    Stage defaults come from `token_budget.stage_defaults[str(stage)]`. The
    completion budget is mode-specific when a `generation.max_new_tokens_<mode>`
    key is present (e.g. smoke is intentionally smaller); otherwise it falls back
    to the stage default. The prompt budget and model-length come from the stage
    default. This is what both rollout and training use.
    """
    sd = (config.get("token_budget", {}) or {}).get("stage_defaults", {}) or {}
    base = sd.get(str(stage), {}) or {}
    gen = config.get("generation", {})

    max_prompt = int(base.get("max_prompt_tokens") or gen.get("max_prompt_tokens") or 4096)
    mode_key = {
        "smoke": "max_new_tokens_smoke",
        "train": "max_new_tokens_train",
        "eval": "max_new_tokens_eval",
    }.get(mode)
    max_new = int(
        (gen.get(mode_key) if mode_key else None)
        or base.get("max_new_tokens")
        or gen.get("max_new_tokens")
        or 2048
    )
    vllm = int(
        base.get("vllm_max_model_length")
        or gen.get("max_model_length")
        or (max_prompt + max_new)
    )
    return {
        "max_prompt_tokens": max_prompt,
        "max_new_tokens": max_new,
        "max_model_length": vllm,
        "vllm_max_model_length": vllm,
    }


def run_episode(
    model,
    tokenizer,
    task: Dict[str, Any],
    config: Dict[str, Any],
    *,
    registry=None,
    max_turns: Optional[int] = None,
    mode: str = "eval",
    is_eval: Optional[bool] = None,
    generate_fn=None,
) -> Trajectory:
    """Run one online multi-turn episode.

    Args:
        mode: one of "smoke" | "eval" | "train"; selects the stage-aware token
              budget and whether extra eval turns are allowed.
        is_eval: deprecated override; if set, takes precedence for the extra-turn
                 decision (kept for backward compatibility).
        generate_fn: optional callable(messages, max_new_tokens)->dict for testing
                     or vLLM; defaults to transformers generate_once.
    """
    gen_cfg = config.get("generation", {})
    exec_cfg = config.get("executor", {})

    gold_num_turns = int(task.get("num_calls") or len(task.get("gold_calls", [])))
    eval_like = is_eval if is_eval is not None else (mode == "eval")
    if max_turns is None:
        extra = int(gen_cfg.get("max_extra_turns_eval", 1)) if eval_like else 0
        max_turns = gold_num_turns + extra
    # Hard safety cap.
    max_turns = max(1, min(max_turns, gold_num_turns + 4))

    budget = get_stage_token_budget(config, gold_num_turns, mode)
    max_new_tokens = budget["max_new_tokens"]
    max_prompt_tokens = budget["max_prompt_tokens"]
    temperature = float(gen_cfg.get("temperature", 0.7))
    top_p = float(gen_cfg.get("top_p", 0.95))

    executor = ToolExecutor(
        task,
        registry=registry,
        mode=exec_cfg.get("mode", "auto"),
        ibm_call_timeout=float(exec_cfg.get("ibm_call_timeout", 30.0)),
    )

    traj = Trajectory(
        task_id=task["task_id"],
        stage=gold_num_turns,
        gold_num_turns=gold_num_turns,
        executor_mode=executor.mode,
    )

    history: List[Dict[str, str]] = []

    for turn_idx in range(max_turns):
        messages = build_messages(task, history, eval_hardening=eval_like)
        if generate_fn is not None:
            g = generate_fn(messages, max_new_tokens)
        else:
            g = generate_once(
                model, tokenizer, messages, max_new_tokens,
                temperature=temperature, top_p=top_p,
                max_prompt_tokens=max_prompt_tokens,
            )

        turn = Turn(
            turn_idx=turn_idx,
            model_text=g["text"],
            prompt_tokens=g.get("prompt_tokens", 0),
            completion_tokens=g.get("completion_tokens", 0),
            clipped_completion=bool(g.get("clipped", False)),
        )
        if g.get("prompt_overflow"):
            # The accumulated multi-turn history (or a runaway tool observation)
            # pushed the prompt past the context window. Generation was skipped
            # and returned empty text — there is nothing to parse, so end the
            # episode cleanly instead of looping on empty turns.
            traj.prompt_overflow = True
            turn.fail_reason = "prompt_overflow"
            traj.turns.append(turn)
            traj.stop_reason = "prompt_overflow"
            break

        history.append({"role": "assistant", "content": g["text"]})

        if turn.clipped_completion:
            traj.clipped_any = True
            turn.fail_reason = "clipped_completion"
            traj.turns.append(turn)
            traj.stop_reason = "clipped"
            break

        pr = parse_tool_call(g["text"], lenient=eval_like)
        if pr.is_terminal:
            turn.is_terminal = True
            traj.turns.append(turn)
            traj.stop_reason = "terminal"
            break
        if not pr.ok:
            turn.fail_reason = f"parse:{pr.reason}"
            traj.turns.append(turn)
            history.append({
                "role": "user",
                "content": "[tool error: could not parse a single valid tool call]",
            })
            traj.stop_reason = "parse_fail"
            break

        call = pr.call
        turn.parsed_call = call
        exec_res = executor.execute(call)
        turn.observation = exec_res.observation

        if exec_res.error is not None:
            turn.fail_reason = f"exec:{exec_res.error}"
            traj.turns.append(turn)
            history.append({
                "role": "user",
                "content": format_tool_response(call, f"[error: {exec_res.error}]"),
            })
            traj.stop_reason = "executor_error"
            break

        traj.final_observation = exec_res.observation
        traj.turns.append(turn)
        history.append({
            "role": "user",
            "content": format_tool_response(call, exec_res.observation),
        })

    if traj.stop_reason is None:
        traj.stop_reason = "max_turns"
    return traj
