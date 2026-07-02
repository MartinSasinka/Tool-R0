"""
LLM-as-judge fallback for NESTFUL execute / multiturn modes.

Used when the local executor cannot decide a task — typically because:

* the task uses a non-math API the executor doesn't know about, or
* execution succeeded but ``robust_value_match`` against ``gold_answer``
  is ambiguous (string answers, alternative phrasings, units).

Calls a small OpenAI-compatible chat model (default ``gpt-4o-mini``) via
the existing ``openai`` dependency. A disk-backed cache (`_judge_cache.jsonl`)
keyed on a stable SHA-1 of (task_id + sorted predicted_calls + gold_answer)
prevents paying twice for the same verdict across reruns.

Graceful degradation
--------------------
The judge is *optional*. If ``OPENAI_API_KEY`` is unset, every call
returns ``verdict="skip"`` with reason ``"no_api_key"`` and the runner
counts the task as ``skipped`` rather than ``failed``. This keeps eval
runnable on any machine (CI, smoke tests) and avoids silently penalising
non-math tasks because of a missing key.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import threading
import time
from typing import Any, Dict, List, Optional

DEFAULT_JUDGE_MODEL = os.environ.get("NESTFUL_JUDGE_MODEL", "gpt-4o-mini")
DEFAULT_CACHE_PATH = "eval/results/nestful/_judge_cache.jsonl"


_JUDGE_PROMPT = """\
You are a strict tool-use evaluator for the NESTFUL benchmark.

A user asked the question below. The model produced a sequence of API/tool
calls (potentially nested via $var_N.result$ references) intending to
solve it. The dataset's ground-truth final answer is GOLD_ANSWER.

Decide whether the predicted call sequence, if executed correctly, would
produce GOLD_ANSWER (or a value semantically equivalent to it under
reasonable rounding / units / phrasing).

QUESTION:
{question}

GOLD_ANSWER:
{gold_answer}

PREDICTED CALL SEQUENCE (JSON):
{predicted_calls}

EXECUTION TRACE (may be partial, may be empty):
{execution_trace}

Respond with ONLY a JSON object on a single line:
{{"verdict": "pass" | "fail", "reason": "<one short sentence>"}}
"""


@dataclasses.dataclass
class JudgeResult:
    verdict: str  # "pass" | "fail" | "skip" | "error"
    reason: str
    used_cache: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def _stable_key(
    task_id: str,
    predicted_calls: List[Dict[str, Any]],
    gold_answer: Any,
) -> str:
    payload = {
        "task_id": task_id,
        "calls": predicted_calls,
        "gold": gold_answer,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()


class _JudgeCache:
    """Append-only JSONL cache. Thread-safe within a process.

    File format: one JSON object per line ``{"key": ..., "verdict": ...,
    "reason": ...}``. We re-read on first access and append on writes;
    this keeps writes durable even if the process is killed.
    """

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._memo: Dict[str, Dict[str, Any]] = {}
        self._loaded = False

    def _load_unlocked(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not os.path.isfile(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    key = rec.get("key")
                    if isinstance(key, str):
                        self._memo[key] = rec
        except OSError:
            pass

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            self._load_unlocked()
            return self._memo.get(key)

    def put(self, key: str, verdict: str, reason: str) -> None:
        rec = {"key": key, "verdict": verdict, "reason": reason}
        with self._lock:
            self._load_unlocked()
            self._memo[key] = rec
            try:
                d = os.path.dirname(self.path)
                if d:
                    os.makedirs(d, exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except OSError:
                pass


_GLOBAL_CACHE: Optional[_JudgeCache] = None


def get_cache(path: str = DEFAULT_CACHE_PATH) -> _JudgeCache:
    global _GLOBAL_CACHE
    if _GLOBAL_CACHE is None or _GLOBAL_CACHE.path != path:
        _GLOBAL_CACHE = _JudgeCache(path)
    return _GLOBAL_CACHE


# ---------------------------------------------------------------------------
# OpenAI client (lazy)
# ---------------------------------------------------------------------------


_OPENAI_CLIENT = None


def _get_openai_client():
    """Build the OpenAI client lazily. Returns None if openai isn't installed."""
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is not None:
        return _OPENAI_CLIENT
    try:
        import openai
    except ImportError:
        return None
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    base_url = os.environ.get("OPENAI_API_BASE") or os.environ.get("NESTFUL_JUDGE_API_BASE")
    _OPENAI_CLIENT = openai.OpenAI(api_key=api_key, base_url=base_url) if base_url else openai.OpenAI(api_key=api_key)
    return _OPENAI_CLIENT


def _parse_verdict_payload(text: str) -> Optional[JudgeResult]:
    """Extract the first ``{"verdict": ...}`` JSON object from a model reply."""
    if not text:
        return None
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:].strip()
    start = s.find("{")
    end = s.rfind("}")
    if start < 0 or end < 0 or end <= start:
        return None
    try:
        obj = json.loads(s[start : end + 1])
    except json.JSONDecodeError:
        return None
    verdict = str(obj.get("verdict", "")).strip().lower()
    if verdict not in ("pass", "fail"):
        return None
    reason = str(obj.get("reason", "")).strip() or "no_reason_given"
    return JudgeResult(verdict=verdict, reason=reason[:280])


def evaluate_with_llm(
    *,
    task_id: str,
    question: str,
    gold_answer: Any,
    predicted_calls: List[Dict[str, Any]],
    execution_trace: Optional[List[Dict[str, Any]]] = None,
    cache_path: str = DEFAULT_CACHE_PATH,
    model: str = DEFAULT_JUDGE_MODEL,
    max_retries: int = 2,
) -> JudgeResult:
    """Send a task + predicted calls to the LLM judge.

    Returns a :class:`JudgeResult` with verdict in ``{pass, fail, skip,
    error}``. Never raises — all failures are encoded in the result so the
    runner can keep going.
    """
    cache = get_cache(cache_path)
    key = _stable_key(task_id, predicted_calls, gold_answer)
    cached = cache.get(key)
    if cached is not None:
        return JudgeResult(
            verdict=str(cached.get("verdict", "skip")),
            reason=str(cached.get("reason", "")),
            used_cache=True,
        )

    client = _get_openai_client()
    if client is None:
        return JudgeResult(verdict="skip", reason="no_api_key")

    prompt = _JUDGE_PROMPT.format(
        question=question[:2000],
        gold_answer=json.dumps(gold_answer, default=str, ensure_ascii=False)[:500],
        predicted_calls=json.dumps(predicted_calls, default=str, ensure_ascii=False)[:3000],
        execution_trace=json.dumps(execution_trace or [], default=str, ensure_ascii=False)[:2000],
    )

    last_err: Optional[str] = None
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200,
            )
            text = resp.choices[0].message.content or ""
            parsed = _parse_verdict_payload(text)
            if parsed is None:
                last_err = f"unparseable_judge_reply:{text[:120]}"
                continue
            cache.put(key, parsed.verdict, parsed.reason)
            return parsed
        except Exception as exc:  # network, rate limits, auth, etc.
            last_err = f"{type(exc).__name__}:{exc}"
            if attempt < max_retries:
                time.sleep(min(8, 2 ** attempt))

    return JudgeResult(verdict="error", reason=last_err or "unknown_judge_failure")
