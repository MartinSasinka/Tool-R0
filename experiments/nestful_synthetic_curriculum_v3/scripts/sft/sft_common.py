"""Shared helpers for the Stage2 continuation SFT experiment.

IMPORTANT — this module does NOT define a new dataset. It only provides
serialization helpers that turn the EXISTING, already-filtered GRPO Stage2
curriculum file

    outputs/curriculum_v3_1/filtered/stage2_2call_dependency.jsonl

into an SFT training *view* (a different representation of the SAME rows:
"Stage2 continuation SFT serialization"). No new tasks are generated here;
no re-sampling, re-filtering, or curriculum regeneration happens in this
package. See build_stage2_sft_dataset.py for the hard-fail integrity checks
that enforce this.

Reuses (imports, does not reimplement) the exact prompt/tool-normalization
code the GRPO pipeline uses at train/eval time, so the SFT input distribution
matches what free ReAct rollout.run_episode() actually feeds the model:
  - data.normalize_task            (tool-schema + gold-call normalization)
  - prompt.SYSTEM_PROMPT / build_user_content / format_tool_response
  - rollout._format_forced_call_text (identical formatting used by the GRPO
    teacher-forced-prefix mechanism for a synthesized "gold" assistant turn)
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent              # .../scripts/sft
SCRIPTS = HERE.parent                                # .../scripts
V3 = SCRIPTS.parent                                  # .../nestful_synthetic_curriculum_v3
EXPERIMENTS = V3.parent                              # .../experiments
MINIMAL = EXPERIMENTS / "nestful_mtgrpo_minimal"

# IMPORTANT: only MINIMAL (nestful_mtgrpo_minimal) goes on sys.path here.
# V3 also has its own unrelated top-level `run.py` (the curriculum
# orchestrator); if V3 were inserted too, `from run import ...` in
# downstream scripts (train_stage2_continuation_sft.py,
# continuation_conditioned_eval.py) would silently shadow
# nestful_mtgrpo_minimal/run.py (which defines build_registry /
# load_model_and_tokenizer) with the wrong module. V3/EXPERIMENTS are only
# used here as Path objects for constants below, never as import roots.
if str(MINIMAL) not in sys.path:
    sys.path.insert(0, str(MINIMAL))

from data import normalize_task  # noqa: E402  (nestful_mtgrpo_minimal/data.py)
from prompt import (  # noqa: E402  (nestful_mtgrpo_minimal/prompt.py)
    SYSTEM_PROMPT,
    build_user_content,
    format_tool_response,
)
from rollout import _format_forced_call_text  # noqa: E402

# ---------------------------------------------------------------------------
#  Paths / constants
# ---------------------------------------------------------------------------
DEFAULT_SOURCE_STAGE2 = str(
    V3 / "outputs" / "curriculum_v3_1" / "filtered" / "stage2_2call_dependency.jsonl"
)
DEFAULT_MANIFEST = str(V3 / "outputs" / "curriculum_v3_1" / "curriculum_v3_1_manifest.json")
DEFAULT_OUT_DIR = str(V3 / "outputs" / "sft" / "stage2_continuation")
STAGE_KEY = "stage2_2call_dependency"
BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"

TERMINAL_TEXT = "<tool_call_answer>[]</tool_call_answer>"

# ---------------------------------------------------------------------------
#  Generic file helpers
# ---------------------------------------------------------------------------

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl_raw(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_expected_stage_count(manifest_path: str, stage_key: str = STAGE_KEY) -> int:
    """Read the GRPO run's expected row count for `stage_key` from the v3.1
    curriculum manifest. Used to hard-fail build_stage2_sft_dataset.py if the
    source file has silently drifted from what the GRPO run actually trained
    Stage2 on."""
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    stages = manifest.get("stages") or {}
    if stage_key not in stages:
        raise KeyError(
            f"manifest {manifest_path} has no 'stages.{stage_key}' entry — "
            f"available keys: {sorted(stages.keys())}"
        )
    return int(stages[stage_key])


# ---------------------------------------------------------------------------
#  Metadata-leak guard — user-visible prompt must never mention internal
#  curriculum bookkeeping fields (motif/stage/cluster/placeholder tokens).
# ---------------------------------------------------------------------------
_LEAK_SUBSTRINGS = (
    "motif", "cluster=", "prefix_of_motif", "target_full_motif",
    "source_failure_cluster", "trajectory_id", "sample_id",
    "add B", "multiply by B", "{prefix}", "prefix=",
)


def question_leak_hit(question: str) -> Optional[str]:
    q = (question or "")
    ql = q.lower()
    for pat in _LEAK_SUBSTRINGS:
        if pat.lower() in ql:
            return pat
    return None


# ---------------------------------------------------------------------------
#  Task construction (reuses data.normalize_task so tools/gold_calls are in
#  EXACTLY the shape rollout.run_episode / prompt.build_user_content expect —
#  NOT the richer raw curriculum tool-schema dict, which the GRPO pipeline
#  never shows the model directly).
# ---------------------------------------------------------------------------

def normalize_stage2_row(row: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """normalize_task() + attach fields data.normalize_task drops on the floor.

    data.normalize_task()'s `_METADATA_FIELDS` allowlist does NOT include
    `answer_type` (only stage/motif/trajectory provenance), so it is
    re-attached here directly from the raw row. Also attaches the raw row's
    precomputed gold observations (data.normalize_task does not carry
    `observations` through, since GRPO recomputes them via
    reward.compute_gold_observations against the live executor; the SFT
    dataset builder uses the precomputed ones directly so it never needs the
    IBM executable-functions dir)."""
    task = normalize_task(row, idx)
    obs = row.get("observations")
    task["_gold_observations"] = list(obs) if isinstance(obs, list) else None
    task["_answer_type"] = row.get("answer_type")
    return task


# ---------------------------------------------------------------------------
#  Continuation-mode message / text builders
# ---------------------------------------------------------------------------

def build_input_prefix_messages(task: Dict[str, Any]) -> List[Dict[str, str]]:
    """system + user(question+tools) + assistant(gold call 1) + user(gold obs 1).

    Identical in spirit to rollout.build_teacher_forced_prefix(n_forced=1):
    the assistant turn is the exact text a correctly-behaving policy would
    have emitted for gold_calls[0], and the following user turn is the real
    (precomputed) gold observation for that call — so this is exactly the
    context free ReAct generation would have produced after its own first,
    correct turn.
    """
    gold_calls = task["gold_calls"]
    obs = task.get("_gold_observations")
    if not obs or len(obs) < 1:
        raise ValueError(f"task {task.get('task_id')} has no gold observation for call 1")
    call1, obs1 = gold_calls[0], obs[0]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_content(task)},
        {"role": "assistant", "content": _format_forced_call_text(call1)},
        {"role": "user", "content": format_tool_response(call1, obs1)},
    ]


def build_continuation_messages(task: Dict[str, Any]) -> Tuple[List[Dict[str, str]], List[int]]:
    """Full 7-message continuation-SFT conversation:

        0 system
        1 user    (question + tools)
        2 assistant (gold call 1 — GIVEN, part of the input, NOT a loss target)
        3 user    (gold observation 1 — GIVEN)
        4 assistant (gold call 2 — GENERATION TARGET)
        5 user    (gold observation 2 — real env feedback, NOT a loss target,
                    but required context for the turn-6 stop decision)
        6 assistant (terminal empty finish — GENERATION TARGET)

    Returns (messages, loss_target_message_indices) = (messages, [4, 6]).
    """
    prefix = build_input_prefix_messages(task)
    gold_calls = task["gold_calls"]
    obs = task["_gold_observations"]
    if len(gold_calls) != 2 or not obs or len(obs) < 2:
        raise ValueError(
            f"task {task.get('task_id')} is not a valid 2-call continuation "
            f"example (gold_calls={len(gold_calls)}, observations={len(obs) if obs else 0})"
        )
    call2, obs2 = gold_calls[1], obs[1]
    messages = prefix + [
        {"role": "assistant", "content": _format_forced_call_text(call2)},
        {"role": "user", "content": format_tool_response(call2, obs2)},
        {"role": "assistant", "content": TERMINAL_TEXT},
    ]
    return messages, [4, 6]


def render_flat_text(messages: List[Dict[str, str]]) -> str:
    tag = {"system": "SYSTEM", "user": "USER", "assistant": "ASSISTANT"}
    parts = []
    for m in messages:
        parts.append(f"[{tag.get(m['role'], m['role'].upper())}]\n{m['content']}")
    return "\n\n".join(parts)


def build_continuation_record(task: Dict[str, Any]) -> Dict[str, Any]:
    """Build the full SFT record for one Stage2 row (continuation target_type)."""
    messages, loss_idx = build_continuation_messages(task)
    input_messages = messages[:4]
    target_messages = [messages[i] for i in loss_idx]  # assistant call2, assistant terminal
    input_text = render_flat_text(input_messages)
    target_text = "\n".join(m["content"] for m in target_messages)
    record = {
        "sample_id": task["task_id"],
        "question": task["question"],
        "tools": task["tools"],
        "gold_calls": task["gold_calls"],
        "gold_observations": task["_gold_observations"],
        "gold_answer": task["gold_answer"],
        "input_text": input_text,
        "target_text": target_text,
        "messages": messages,
        "target_type": "continuation",
        # Additive metadata (not required by the schema, kept for auditability
        # and so the trainer never has to re-derive which turns are targets).
        "loss_target_message_indices": loss_idx,
        "provenance": _provenance(task),
    }
    return record


def _provenance(task: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "stage": task.get("stage"),
        "motif_type": task.get("motif_type"),
        "target_full_motif": task.get("target_full_motif"),
        "source_failure_cluster": task.get("source_failure_cluster"),
        "trajectory_id": task.get("trajectory_id"),
        "answer_type": task.get("_answer_type"),
        # Tools actually invoked in the gold trace (2 per Stage2 record).
        "tool_names_used": sorted({c.get("name", "") for c in task["gold_calls"]}),
        # Full tool menu offered to the model in this task's prompt (includes
        # distractor tools never called) — NOT the same distribution as
        # tool_names_used; conflating the two undercounts distractor exposure.
        "tool_names_offered": sorted({t.get("name", "") for t in task.get("tools", [])}),
    }


# ---------------------------------------------------------------------------
#  Full-trace builder (scaffold for future support — NOT wired into the
#  trainer yet; see train_stage2_continuation_sft.py docstring). Kept here so
#  a future --target-type full_trace only needs trainer-side loss-mask work,
#  not a new record schema.
# ---------------------------------------------------------------------------

def build_full_trace_record(task: Dict[str, Any]) -> Dict[str, Any]:
    """target_type = 'full_trace': input is question+tools ONLY; the model must
    generate the ENTIRE trace (call1, call2, terminal) itself. All three
    assistant turns are loss targets; only the two REAL environment tool
    responses are masked context.
    """
    gold_calls = task["gold_calls"]
    obs = task["_gold_observations"]
    if len(gold_calls) != 2 or not obs or len(obs) < 2:
        raise ValueError(f"task {task.get('task_id')} is not a valid 2-call example")
    call1, call2 = gold_calls
    obs1, obs2 = obs[0], obs[1]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_content(task)},
        {"role": "assistant", "content": _format_forced_call_text(call1)},   # target
        {"role": "user", "content": format_tool_response(call1, obs1)},
        {"role": "assistant", "content": _format_forced_call_text(call2)},   # target
        {"role": "user", "content": format_tool_response(call2, obs2)},
        {"role": "assistant", "content": TERMINAL_TEXT},                    # target
    ]
    loss_idx = [2, 4, 6]
    input_messages = messages[:2]
    target_messages = [messages[i] for i in loss_idx]
    record = {
        "sample_id": task["task_id"],
        "question": task["question"],
        "tools": task["tools"],
        "gold_calls": task["gold_calls"],
        "gold_observations": task["_gold_observations"],
        "gold_answer": task["gold_answer"],
        "input_text": render_flat_text(input_messages),
        "target_text": "\n".join(m["content"] for m in target_messages),
        "messages": messages,
        "target_type": "full_trace",
        "loss_target_message_indices": loss_idx,
        "provenance": _provenance(task),
    }
    return record


# ---------------------------------------------------------------------------
#  Tokenizer helper (best-effort; dataset building must work even without a
#  local/cached tokenizer — falls back to a documented char-based estimate).
# ---------------------------------------------------------------------------

def try_load_tokenizer(base_model: str = BASE_MODEL):
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        return tok, base_model
    except Exception as exc:  # noqa: BLE001 - dataset build must not hard-depend on this
        print(f"[sft_common] WARNING: could not load tokenizer '{base_model}' "
              f"({exc!r}); falling back to an approximate char/4 token estimate.")
        return None, None


def count_tokens(tokenizer, text: str) -> int:
    if tokenizer is None:
        return max(1, round(len(text) / 4))
    return len(tokenizer.encode(text, add_special_tokens=False))
