#!/usr/bin/env python3
"""Per-epoch NESTFUL evaluation using HF inference (no vLLM).

Loads all NESTFUL tasks for a given call depth (n_calls), runs multi-turn
rollouts using the same Tool-R0 format used during training, and reports
exec_pass_rate, tool_call_acc, partial_score, and parse_fail_rate.

Designed to run as a subprocess after each training epoch, so GPU memory
is fully freed when it exits.

CLI example:
    python curricullum/train/evaluate_nestful_stage.py \
        --base_model Qwen/Qwen3-4B-Instruct-2507 \
        --adapter_path curricullum/checkpoints/.../stage1_epoch1 \
        --nestful_path eval/data/NESTFUL-main/data_v2/nestful_data.jsonl \
        --call_dist_path helper_calculations/output/nestful_call_distribution.json \
        --n_calls 2 \
        --batch_size 8 \
        --output_json curricullum/training/results/stage1_epoch1_val.json

For baseline (no adapter):
    python ... --no_adapter
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Reduce CUDA fragmentation OOMs during long batched generation.
# Must be set before torch initialises CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# Match CUDA device ids to `nvidia-smi` ordering so CUDA_VISIBLE_DEVICES shard
# assignment targets the intended physical GPUs (not the display GPU).
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from curricullum.train.prepare_dataset_toolr0 import build_task_tools
from nestful_evaluation.run import (
    IBMFunctionRegistry,
    TOOL_R0_SYSTEM_PROMPT,
    _format_tool_response,
    _matches_gold,
    _normalize_call,
    build_user_content,
    execute_one,
    parse_tool_calls,
)

# Max number of turns before we forcibly stop (safety valve)
_MAX_TURNS_SAFETY = 12


def _log(msg: str) -> None:
    print(f"[eval] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_nestful_tasks(
    nestful_path: str,
    call_dist_path: str,
    n_calls: int,
) -> List[Dict[str, Any]]:
    """Return all NESTFUL tasks that have exactly n_calls gold tool calls."""
    with open(call_dist_path, encoding="utf-8") as f:
        dist = json.load(f)

    sample_ids: List[str] = dist.get("sample_ids_by_num_gold_calls", {}).get(str(n_calls), [])
    if not sample_ids:
        _log(f"WARNING: no tasks with exactly {n_calls} calls in distribution file")
        return []

    id_set = set(sample_ids)
    tasks: List[Dict[str, Any]] = []
    with open(nestful_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("sample_id") in id_set:
                tasks.append(row)

    _log(f"loaded {len(tasks)} NESTFUL tasks with n_calls={n_calls}")
    return tasks


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(
    base_model: str,
    adapter_path: Optional[str],
    device_map: str = "cuda:0",
) -> Tuple[Any, Any]:
    """Load base model + optional LoRA adapter.

    Eval always runs on a single GPU (cuda:0 = first visible after CUDA_VISIBLE_DEVICES).
    Using eager attention: compatible with all GPU generations and avoids SDPA
    flash-kernel issues when device_map distributes layers across mixed devices.
    """
    _log(f"loading base model: {base_model}")
    # SDPA on Ampere+ (A100) uses the memory-efficient/flash kernel and does NOT
    # materialise the full [B, H, S, S] fp32 attention matrix like eager does —
    # eager OOMs at batch>8. Pinning device_map=cuda:0 avoids the earlier failure
    # where device_map=auto spilled layers onto the tiny 4GB DGX Display GPU.
    attn_impl = "flash_attention_2" if _flash_available() else "sdpa"
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )
    _log(f"attn_implementation={attn_impl}")

    if adapter_path:
        _log(f"loading LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
        model = model.merge_and_unload()
        _log("adapter merged")

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def _flash_available() -> bool:
    try:
        import flash_attn  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Multi-turn rollout for a single task
# ---------------------------------------------------------------------------

def _build_initial_messages(task: Dict[str, Any]) -> List[Dict[str, str]]:
    tools = build_task_tools(task.get("tools") or [])
    return [
        {"role": "system", "content": TOOL_R0_SYSTEM_PROMPT},
        {"role": "user", "content": build_user_content({"question": task["input"], "tools": tools})},
    ]


def _apply_template(tokenizer, messages: List[Dict[str, str]], enable_thinking: bool) -> str:
    """Apply chat template with optional thinking-mode suppression.

    Qwen3 generates 600–900 token <think> blocks by default, which accounts for
    ~80% of eval time. Setting enable_thinking=False makes generation 5–8x faster
    while still producing correct <tool_call_answer> output.
    """
    kwargs: Dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    # Qwen3 / HF transformers ≥ 4.51 support enable_thinking kwarg
    try:
        return tokenizer.apply_chat_template(messages, **kwargs, enable_thinking=enable_thinking)
    except TypeError:
        # Older tokenizer that doesn't know enable_thinking — fall back silently
        return tokenizer.apply_chat_template(messages, **kwargs)


def _generate_one(
    model,
    tokenizer,
    messages: List[Dict[str, str]],
    max_new_tokens: int,
) -> str:
    """Single-task generation (kept for compatibility). Prefer _generate_batch."""
    completions, _ = _generate_batch(model, tokenizer, [messages], max_new_tokens)
    return completions[0]


def _generate_batch(
    model,
    tokenizer,
    messages_list: List[List[Dict[str, str]]],
    max_new_tokens: int,
    enable_thinking: bool = False,
) -> Tuple[List[str], List[bool]]:
    """Greedy generation for a batch of conversations.

    OOM-resilient: if CUDA runs out of memory, the batch is split in half and
    retried recursively (after freeing cache), so a long run never crashes on a
    transiently large batch. Returns (completions, hit_max) preserving input order.
    """
    if not messages_list:
        return [], []

    try:
        return _generate_batch_once(model, tokenizer, messages_list, max_new_tokens, enable_thinking)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(messages_list) == 1:
            _log("  WARNING: OOM on a single-item batch; emitting empty completion")
            return [""], [False]
        mid = len(messages_list) // 2
        _log(f"  OOM at batch={len(messages_list)} → splitting into {mid}+{len(messages_list)-mid}")
        left_c, left_h = _generate_batch(model, tokenizer, messages_list[:mid], max_new_tokens, enable_thinking)
        right_c, right_h = _generate_batch(model, tokenizer, messages_list[mid:], max_new_tokens, enable_thinking)
        return left_c + right_c, left_h + right_h


def _generate_batch_once(
    model,
    tokenizer,
    messages_list: List[List[Dict[str, str]]],
    max_new_tokens: int,
    enable_thinking: bool = False,
) -> Tuple[List[str], List[bool]]:
    """Single batched generation pass (no OOM handling)."""
    texts = [
        _apply_template(tokenizer, msgs, enable_thinking)
        for msgs in messages_list
    ]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=False)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    completions: List[str] = []
    hit_max: List[bool] = []
    eos_id = tokenizer.eos_token_id
    for j in range(len(texts)):
        generated = out_ids[j, prompt_len:]
        last_tok = generated[-1].item() if len(generated) > 0 else -1
        hit_max.append(len(generated) >= max_new_tokens and last_tok != eos_id)
        completions.append(tokenizer.decode(generated, skip_special_tokens=True))
    return completions, hit_max


def _is_terminal(completion: str) -> bool:
    """True if the model signalled end of chain with an empty tool_call_answer."""
    calls, _ = parse_tool_calls(completion)
    return isinstance(calls, list) and len(calls) == 0



def rollout_task(
    task: Dict[str, Any],
    model,
    tokenizer,
    ibm: IBMFunctionRegistry,
    max_new_tokens: int,
) -> Dict[str, Any]:
    """Execute a full multi-turn rollout for one NESTFUL task.

    Returns a dict with:
        correct_calls: int  (number of turns where predicted call matched gold)
        total_calls: int    (number of gold calls in task)
        exec_pass: bool     (all calls correct AND final answer matches gold)
        parse_fail: bool    (any turn failed to produce parseable tool call)
        turns_completed: int
        predicted_calls: list
    """
    gold_calls: List[Dict[str, Any]] = task.get("output") or []
    total_calls = len(gold_calls)
    messages = _build_initial_messages(task)

    by_label: Dict[str, Any] = {}
    indexed: List[Any] = []

    correct_calls = 0
    parse_fail = False
    predicted_calls: List[Optional[Dict[str, Any]]] = []
    final_result: Any = None

    for turn_idx in range(min(total_calls + 2, _MAX_TURNS_SAFETY)):
        completion = _generate_one(model, tokenizer, messages, max_new_tokens)
        messages.append({"role": "assistant", "content": completion})

        if _is_terminal(completion):
            break

        raw_calls, _ = parse_tool_calls(completion)
        if not raw_calls:
            # Only flag as a true parse failure if we were still expecting gold calls.
            # Extra turns after the chain is done (terminal-signal turns) are fine to skip.
            if turn_idx < total_calls:
                parse_fail = True
            predicted_calls.append(None)
            # Feed a stub so conversation can continue
            messages.append({
                "role": "user",
                "content": "[tool error: could not parse tool call]",
            })
            continue

        pred_call = _normalize_call(raw_calls[0])
        predicted_calls.append(pred_call)

        # Compare to gold call for this turn
        if turn_idx < total_calls:
            gold_call = _normalize_call(gold_calls[turn_idx])
            if (
                (pred_call.get("name") or "") == (gold_call.get("name") or "")
                and set((pred_call.get("arguments") or {}).keys())
                   == set((gold_call.get("arguments") or {}).keys())
            ):
                correct_calls += 1

        # execute_one already uses SIGALRM internally (_run_with_alarm) so no extra wrapper needed
        trace = execute_one(
            pred_call,
            by_label,
            indexed,
            index=turn_idx,
            ibm_registry=ibm,
        )
        if trace.error:
            result_val = f"[error: {trace.error}]"
        else:
            result_val = trace.result
            final_result = result_val

        by_label[trace.label] = result_val
        indexed.append(result_val)

        messages.append({
            "role": "user",
            "content": _format_tool_response(
                {"name": pred_call.get("name", ""), "arguments": pred_call.get("arguments") or {}},
                result_val,
            ),
        })

        if turn_idx >= total_calls - 1:
            # Gold chain exhausted — one more turn to get terminal signal
            pass

    gold_answer = task.get("gold_answer")
    exec_pass = (
        not parse_fail
        and correct_calls == total_calls
        and _matches_gold(final_result, gold_answer)
    )

    return {
        "sample_id": task.get("sample_id"),
        "correct_calls": correct_calls,
        "total_calls": total_calls,
        "exec_pass": exec_pass,
        "parse_fail": parse_fail,
        "turns_completed": len(predicted_calls),
        "predicted_calls": [p for p in predicted_calls if p],
        "gold_answer": gold_answer,
        "final_result": str(final_result)[:500] if final_result is not None else None,
    }


# ---------------------------------------------------------------------------
# Batched multi-turn evaluation
# ---------------------------------------------------------------------------

def evaluate(
    tasks: List[Dict[str, Any]],
    model,
    tokenizer,
    ibm: IBMFunctionRegistry,
    max_new_tokens: int,
    save_failures: int = 20,
    batch_size: int = 8,
    enable_thinking: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Evaluate tasks using batched generation for ~batch_size× speedup.

    All tasks advance together turn-by-turn. In each turn we batch `batch_size`
    active conversations into one model.generate() call. IBM execution for each
    task stays sequential (it is fast and cannot be trivially parallelised).
    """
    n = len(tasks)
    if n == 0:
        return {"total_tasks": 0, "exec_pass_rate": 0.0, "tool_call_acc": 0.0,
                "partial_score": 0.0, "parse_fail_rate": 0.0,
                "avg_turns_completed": 0.0, "clipped_frac": 0.0}, []

    gold_calls_list   = [task.get("output") or [] for task in tasks]
    total_calls_list  = [len(gc) for gc in gold_calls_list]
    max_turns_global  = min(max(total_calls_list) + 2, _MAX_TURNS_SAFETY)

    messages_list     = [_build_initial_messages(task) for task in tasks]
    by_label_list     = [{} for _ in range(n)]
    indexed_list      = [[] for _ in range(n)]
    correct_calls_list = [0] * n
    parse_fail_list   = [False] * n
    predicted_calls_list: List[List] = [[] for _ in range(n)]
    final_result_list: List[Any] = [None] * n
    done_list         = [False] * n

    clipped_gens = 0
    total_gens   = 0
    t0 = time.time()

    def _conv_len(i: int) -> int:
        # Cheap proxy for token length: total chars across messages.
        return sum(len(m.get("content", "")) for m in messages_list[i])

    for turn_idx in range(max_turns_global):
        active = [i for i in range(n) if not done_list[i]]
        if not active:
            break

        n_done_before = sum(done_list)
        t_turn = time.time()

        # Sort by current conversation length so each batch packs similar-length
        # sequences together → less padding, less wasted compute on the longest item.
        active.sort(key=_conv_len)

        # Process active tasks in sub-batches
        n_batches = (len(active) + batch_size - 1) // batch_size
        for b_idx, b_start in enumerate(range(0, len(active), batch_size)):
            b_indices = active[b_start: b_start + batch_size]
            b_msgs    = [messages_list[i] for i in b_indices]

            completions, hit_max = _generate_batch(model, tokenizer, b_msgs, max_new_tokens, enable_thinking)
            clipped_gens += sum(hit_max)
            total_gens   += len(hit_max)

            # Intra-turn progress for long batches
            if n_batches >= 4 and (b_idx + 1) % max(1, n_batches // 4) == 0:
                _log(
                    f"  turn {turn_idx+1}/{max_turns_global}"
                    f"  batch {b_idx+1}/{n_batches}"
                    f"  elapsed={time.time()-t0:.0f}s"
                )

            for k, i in enumerate(b_indices):
                completion = completions[k]
                messages_list[i].append({"role": "assistant", "content": completion})

                if _is_terminal(completion):
                    done_list[i] = True
                    continue

                raw_calls, _ = parse_tool_calls(completion)
                if not raw_calls:
                    if turn_idx < total_calls_list[i]:
                        parse_fail_list[i] = True
                    predicted_calls_list[i].append(None)
                    messages_list[i].append({
                        "role": "user",
                        "content": "[tool error: could not parse tool call]",
                    })
                    continue

                pred_call = _normalize_call(raw_calls[0])
                predicted_calls_list[i].append(pred_call)

                if turn_idx < total_calls_list[i]:
                    gold_call = _normalize_call(gold_calls_list[i][turn_idx])
                    if (
                        (pred_call.get("name") or "") == (gold_call.get("name") or "")
                        and set((pred_call.get("arguments") or {}).keys())
                           == set((gold_call.get("arguments") or {}).keys())
                    ):
                        correct_calls_list[i] += 1

                trace = execute_one(
                    pred_call, by_label_list[i], indexed_list[i],
                    index=turn_idx, ibm_registry=ibm,
                )
                result_val = f"[error: {trace.error}]" if trace.error else trace.result
                if not trace.error:
                    final_result_list[i] = result_val

                by_label_list[i][trace.label] = result_val
                indexed_list[i].append(result_val)
                messages_list[i].append({
                    "role": "user",
                    "content": _format_tool_response(
                        {"name": pred_call.get("name", ""), "arguments": pred_call.get("arguments") or {}},
                        result_val,
                    ),
                })

                if turn_idx >= total_calls_list[i]:
                    done_list[i] = True

        # Per-turn summary line
        n_done_after = sum(done_list)
        newly_done   = n_done_after - n_done_before
        t_turn_s     = time.time() - t_turn
        _log(
            f"  turn {turn_idx+1}/{max_turns_global}"
            f"  active={len(active)}  done={n_done_after}/{n}"
            f"  (+{newly_done} finished this turn)"
            f"  turn_time={t_turn_s:.0f}s  total={time.time()-t0:.0f}s"
        )

    # ── Compute metrics ──────────────────────────────────────────────────
    exec_pass_count = 0
    tool_acc_sum    = 0.0
    partial_sum     = 0.0
    parse_fail_count = 0
    turns_sum       = 0.0
    failures: List[Dict[str, Any]] = []

    for i in range(n):
        gold_answer = tasks[i].get("gold_answer")
        exec_pass = (
            not parse_fail_list[i]
            and correct_calls_list[i] == total_calls_list[i]
            and _matches_gold(final_result_list[i], gold_answer)
        )
        if exec_pass:
            exec_pass_count += 1
        if parse_fail_list[i]:
            parse_fail_count += 1
        tc = total_calls_list[i]
        cc = correct_calls_list[i]
        tool_acc_sum += cc / tc if tc > 0 else 0.0
        partial_sum  += cc / tc if tc > 0 else 0.0
        turns_sum    += len(predicted_calls_list[i])

        if not exec_pass and len(failures) < save_failures:
            pred_calls_clean = [p for p in predicted_calls_list[i] if p]
            failures.append({
                "sample_id": tasks[i].get("sample_id"),
                "input": tasks[i].get("input", "")[:300],
                "gold_calls": [c.get("name") for c in gold_calls_list[i]],
                "pred_calls": [c.get("name") for c in pred_calls_clean],
                "gold_answer": str(gold_answer)[:100],
                "final_result": str(final_result_list[i])[:500] if final_result_list[i] is not None else None,
                "parse_fail": parse_fail_list[i],
            })

        completed = i + 1
        if completed % 50 == 0 or completed == n:
            elapsed = time.time() - t0
            _log(f"  {completed}/{n}  exec_pass={exec_pass_count/completed:.3f}  elapsed={elapsed:.0f}s")

    clipped_frac = clipped_gens / total_gens if total_gens else 0.0
    if clipped_frac > 0.05:
        _log(
            f"  WARNING: {clipped_frac:.1%} of generations hit max_new_tokens={max_new_tokens}"
            f" — responses may be truncated; consider increasing --max_new_tokens"
        )
    else:
        _log(f"  context OK: {clipped_frac:.1%} of generations clipped at {max_new_tokens} tokens")

    metrics = {
        "total_tasks":        n,
        "exec_pass_rate":     exec_pass_count / n,
        "tool_call_acc":      tool_acc_sum / n,
        "partial_score":      partial_sum / n,
        "parse_fail_rate":    parse_fail_count / n,
        "avg_turns_completed": turns_sum / n,
        "clipped_frac":       round(clipped_frac, 4),
        # Raw counts — required for correct aggregation when sharding across GPUs.
        "exec_pass_count":    exec_pass_count,
        "parse_fail_count":   parse_fail_count,
        "tool_acc_sum":       tool_acc_sum,
        "partial_sum":        partial_sum,
        "turns_sum":          turns_sum,
        "clipped_gens":       clipped_gens,
        "total_gens":         total_gens,
    }
    return metrics, failures


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate adapter on NESTFUL n_calls group")
    ap.add_argument("--base_model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--adapter_path", default=None, help="LoRA adapter dir (omit or --no_adapter for baseline)")
    ap.add_argument("--no_adapter", action="store_true", help="Evaluate base model without any adapter")
    ap.add_argument("--nestful_path", required=True)
    ap.add_argument("--call_dist_path", required=True)
    ap.add_argument("--n_calls", type=int, required=True)
    ap.add_argument("--batch_size", type=int, default=8, help="Conversations processed per model.generate() call (higher = faster, more VRAM)")
    ap.add_argument("--enable_thinking", action="store_true", default=False,
                    help="Allow Qwen3 <think> blocks (accurate but 5-8x slower; default: off)")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--max_tasks", type=int, default=None, help="Limit number of tasks evaluated (smoke test)")
    ap.add_argument("--shard_id", type=int, default=0, help="This shard's index (data-parallel eval across GPUs)")
    ap.add_argument("--num_shards", type=int, default=1, help="Total number of shards; tasks are strided across them")
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--save_failures", type=int, default=20)
    ap.add_argument("--nestful_repo_dir", default=os.environ.get("NESTFUL_REPO_DIR", "nestful_repo"))
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    tasks = load_nestful_tasks(args.nestful_path, args.call_dist_path, args.n_calls)
    if not tasks:
        _log(f"No tasks found for n_calls={args.n_calls}. Exiting.")
        result = {
            "n_calls": args.n_calls,
            "total_tasks": 0,
            "exec_pass_rate": 0.0,
            "tool_call_acc": 0.0,
            "partial_score": 0.0,
            "parse_fail_rate": 0.0,
            "avg_turns_completed": 0.0,
            "error": "no_tasks",
        }
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
        return

    adapter = None if args.no_adapter else args.adapter_path
    model, tokenizer = load_model_and_tokenizer(args.base_model, adapter)

    # IBM registry
    os.environ.setdefault("NESTFUL_REPO_DIR", args.nestful_repo_dir)
    from curricullum.data.exec_trajectory import get_ibm_registry
    ibm = get_ibm_registry()
    if ibm is None:
        _log("WARNING: IBM registry unavailable — exec_pass_rate will be 0")

    if args.max_tasks and args.max_tasks < len(tasks):
        tasks = tasks[: args.max_tasks]
        _log(f"smoke_test: limited to {args.max_tasks} tasks")

    if args.num_shards > 1:
        before = len(tasks)
        tasks = tasks[args.shard_id :: args.num_shards]
        _log(f"shard {args.shard_id}/{args.num_shards}: {len(tasks)}/{before} tasks")

    _log(
        f"evaluating {len(tasks)} tasks  n_calls={args.n_calls}"
        f"  max_new_tokens={args.max_new_tokens}"
        f"  thinking={'on' if args.enable_thinking else 'OFF'}"
    )
    t0 = time.time()
    metrics, failures = evaluate(
        tasks, model, tokenizer, ibm, args.max_new_tokens, args.save_failures,
        batch_size=args.batch_size,
        enable_thinking=args.enable_thinking,
    )
    elapsed = time.time() - t0

    metrics["n_calls"] = args.n_calls
    metrics["elapsed_s"] = round(elapsed, 1)
    metrics["adapter_path"] = adapter or "base"

    _log(
        f"DONE  exec_pass={metrics['exec_pass_rate']:.3f}"
        f"  tool_acc={metrics['tool_call_acc']:.3f}"
        f"  partial={metrics['partial_score']:.3f}"
        f"  parse_fail={metrics['parse_fail_rate']:.3f}"
        f"  elapsed={elapsed:.0f}s"
    )

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    _log(f"metrics -> {out_path}")

    if failures:
        fail_path = out_path.with_suffix(".failures.jsonl")
        fail_path.write_text(
            "\n".join(json.dumps(f, ensure_ascii=False) for f in failures) + "\n",
            encoding="utf-8",
        )
        _log(f"failures ({len(failures)}) -> {fail_path}")


if __name__ == "__main__":
    main()
