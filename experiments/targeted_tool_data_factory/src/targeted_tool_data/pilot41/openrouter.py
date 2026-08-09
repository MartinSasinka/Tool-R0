"""Staged OpenRouter writer/critic for Pilot4.1 query surfaces only."""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..paraphrase import Budget, BudgetExceeded, get_api_key, key_fingerprint
from ..repro import sha256_obj, write_json
from ..util import short_hash

SCHEMA_VERSION = "ttdf.openrouter.v41"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

FORBIDDEN_MODEL_TOKENS = ("latest", "openrouter/free", "openrouter/auto", "/auto")

WRITER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["query", "entities_used", "facts_used", "units_used",
                 "target_requested", "contains_explicit_steps",
                 "contains_tool_names", "writer_confidence", "warnings"],
    "properties": {
        "query": {"type": "string"},
        "entities_used": {"type": "array", "items": {"type": "string"}},
        "facts_used": {"type": "array", "items": {"type": "string"}},
        "units_used": {"type": "array", "items": {"type": "string"}},
        "target_requested": {"type": "string"},
        "contains_explicit_steps": {"type": "boolean"},
        "contains_tool_names": {"type": "boolean"},
        "writer_confidence": {"type": "number"},
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
}

CRITIC_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "facts_preserved", "target_preserved", "units_preserved",
        "no_new_conditions", "all_program_nodes_necessary",
        "program_sufficient_for_query", "query_unambiguous", "query_natural",
        "graph_not_disclosed", "semantic_coherence", "naturalness",
        "ambiguity", "failure_reasons", "verdict",
    ],
    "properties": {
        "facts_preserved": {"type": "boolean"},
        "target_preserved": {"type": "boolean"},
        "units_preserved": {"type": "boolean"},
        "no_new_conditions": {"type": "boolean"},
        "all_program_nodes_necessary": {"type": "boolean"},
        "program_sufficient_for_query": {"type": "boolean"},
        "query_unambiguous": {"type": "boolean"},
        "query_natural": {"type": "boolean"},
        "graph_not_disclosed": {"type": "boolean"},
        "semantic_coherence": {"type": "number"},
        "naturalness": {"type": "number"},
        "ambiguity": {"type": "number"},
        "failure_reasons": {"type": "array", "items": {"type": "string"}},
        "verdict": {"type": "string", "enum": ["PASS", "REWRITE", "REJECT"]},
    },
}

WRITER_SYSTEM = (
    "You write a short natural-language user request from a SemanticContract. "
    "You must include every constant number from the contract exactly. "
    "You never invent numbers, entities, units, conditions or goals. "
    "You never name tools, stages, step numbers or dependency graphs. "
    "Prefer 1–3 sentences. Return JSON only matching the provided schema."
)

CRITIC_SYSTEM = (
    "You audit whether a generated query preserves a SemanticContract without "
    "disclosing the computation graph. "
    "PASS if facts, target, and units are preserved, no new conditions are "
    "invented, and the graph/tools/stages are not disclosed. "
    "REWRITE for awkward phrasing when facts are otherwise intact. "
    "REJECT only for missing/changed facts, invented conditions, or graph leak. "
    "Return JSON only matching the schema."
)


def _parse_json_content(content: str) -> Dict[str, Any]:
    text = (content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


def assert_pinned_model(model: str) -> str:
    m = (model or "").strip()
    if not m:
        raise ValueError("empty model slug")
    low = m.lower()
    for tok in FORBIDDEN_MODEL_TOKENS:
        if tok in low:
            raise ValueError(f"refusing unpinned/router model {model!r}")
    return m


def load_openrouter_config(path: Optional[Path] = None) -> Dict[str, Any]:
    cfg = {
        "enabled": True,
        "writer_model": os.environ.get(
            "OPENROUTER_WRITER_MODEL",
            "mistralai/mistral-small-24b-instruct-2501"),
        "critic_model": os.environ.get(
            "OPENROUTER_CRITIC_MODEL",
            "google/gemini-2.5-flash-lite"),
        "audit_model": os.environ.get(
            "OPENROUTER_AUDIT_MODEL",
            "google/gemini-2.5-flash-lite"),
        "exact_provider": None,
        "allow_fallbacks": False,
        "require_structured_outputs": True,
        "reasoning_effort": "none",
        "temperature": 0.7,
        "max_output_tokens": 1024,
        "max_retries": 2,
        "request_timeout_seconds": 120,
        "max_total_cost_usd": 20.0,
        "max_cost_per_task_usd": 0.02,
        "cache_system_prompt": True,
        "base_url": DEFAULT_BASE_URL,
    }
    if path and Path(path).is_file():
        text = Path(path).read_text(encoding="utf-8")
        if path.suffix in (".yaml", ".yml"):
            import yaml
            loaded = yaml.safe_load(text) or {}
        else:
            loaded = json.loads(text)
        cfg.update(loaded.get("openrouter") or loaded)
    for key in ("writer_model", "critic_model", "audit_model"):
        cfg[key] = assert_pinned_model(str(cfg[key]))
    if cfg.get("allow_fallbacks"):
        raise ValueError("allow_fallbacks must be false for Pilot4.1 freeze")
    return cfg


@dataclass
class OpenRouterSession:
    cfg: Dict[str, Any]
    log_path: Path
    usage_path: Path
    failures_path: Path
    mode: str = "GENERATE_NEW_LLM_OUTPUTS"  # or REPLAY_EXISTING_LLM_OUTPUTS
    replay: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    budget: Budget = field(default_factory=Budget)
    _key: Optional[str] = None

    def __post_init__(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        prior_usd = 0.0
        prior_req = 0
        prior_pt = 0
        prior_ct = 0
        if self.usage_path.is_file():
            try:
                prior = json.loads(self.usage_path.read_text(encoding="utf-8"))
                b = prior.get("budget") or {}
                prior_usd = float(b.get("usd") or 0.0)
                prior_req = int(b.get("requests") or 0)
                prior_pt = int(b.get("prompt_tokens") or 0)
                prior_ct = int(b.get("completion_tokens") or 0)
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                pass
        self.budget = Budget(
            max_requests=100_000,
            max_usd=float(self.cfg["max_total_cost_usd"]),
            requests=prior_req,
            prompt_tokens=prior_pt,
            completion_tokens=prior_ct,
            usd=prior_usd,
        )
        self._key = get_api_key()
        if self.log_path.is_file() and self.mode == "REPLAY_EXISTING_LLM_OUTPUTS":
            with self.log_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    self.replay[row["input_hash"]] = row

    @property
    def available(self) -> bool:
        return bool(self._key)

    def _append(self, path: Path, row: Dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            fh.flush()

    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int,
                       model: str) -> float:
        # conservative default if provider omits cost
        return round((prompt_tokens + completion_tokens) * 0.0000005, 8)

    def complete_json(self, *, purpose: str, model: str,
                      messages: List[Dict[str, str]],
                      schema: Dict[str, Any],
                      task_ids: Sequence[str],
                      ) -> Dict[str, Any]:
        model = assert_pinned_model(model)
        input_hash = sha256_obj({"model": model, "messages": messages,
                                 "schema": schema, "purpose": purpose})
        if self.mode == "REPLAY_EXISTING_LLM_OUTPUTS":
            hit = self.replay.get(input_hash)
            if hit is None:
                raise RuntimeError(
                    f"REPLAY miss for purpose={purpose} hash={input_hash[:12]}")
            return hit

        if not self._key:
            raise RuntimeError("OPENROUTER_API_KEY not available")
        self.budget.check()

        import httpx

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": float(self.cfg["temperature"]),
            "max_tokens": int(self.cfg["max_output_tokens"]),
            "usage": {"include": True},
            "provider": {"allow_fallbacks": False},
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": purpose,
                    "strict": True,
                    "schema": schema,
                },
            },
        }
        if self.cfg.get("exact_provider"):
            payload["provider"]["order"] = [self.cfg["exact_provider"]]
            payload["provider"]["require_parameters"] = True

        headers = {
            "Authorization": f"Bearer {self._key}",
            "Content-Type": "application/json",
            "X-Title": "tool-r0-pilot41",
        }
        # never log the key
        t0 = time.perf_counter()
        last_err = None
        parsed: Optional[Dict[str, Any]] = None
        usage: Dict[str, Any] = {}
        request_id = ""
        provider = ""
        attempt = 0
        for attempt in range(int(self.cfg["max_retries"]) + 1):
            try:
                with httpx.Client(timeout=float(self.cfg["request_timeout_seconds"])) as client:
                    resp = client.post(f"{self.cfg['base_url']}/chat/completions",
                                       json=payload, headers=headers)
                request_id = resp.headers.get("x-request-id") or short_hash(
                    [time.time(), attempt, purpose])[:16]
                if resp.status_code >= 400:
                    last_err = f"HTTP {resp.status_code}: {resp.text[:400]}"
                    self._append(self.failures_path, {
                        "purpose": purpose, "model": model,
                        "error": last_err, "retry": attempt,
                        "task_ids": list(task_ids),
                        "key_fingerprint": key_fingerprint(self._key),
                    })
                    # rate-limit / overload: longer backoff
                    delay = (8.0 if resp.status_code in (429, 503)
                             else 1.5) * (attempt + 1)
                    time.sleep(delay)
                    continue
                data = resp.json()
                content = data["choices"][0]["message"].get("content") or ""
                usage = data.get("usage") or {}
                provider = (data.get("provider")
                            or (data.get("model") or model))
                # account cost even if JSON parse fails
                prompt_tokens = int(usage.get("prompt_tokens") or 0)
                completion_tokens = int(usage.get("completion_tokens") or 0)
                cost = usage.get("cost")
                if cost is None:
                    cost = self._estimate_cost(
                        prompt_tokens, completion_tokens, model)
                cost = float(cost)
                self.budget.add(prompt_tokens, completion_tokens, cost)
                write_json(self.usage_path, {
                    "schema_version": SCHEMA_VERSION,
                    "budget": self.budget.as_dict(),
                    "mode": self.mode,
                    "writer_model": self.cfg["writer_model"],
                    "critic_model": self.cfg["critic_model"],
                })
                try:
                    parsed = _parse_json_content(content)
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    last_err = f"invalid JSON: {exc}; content={content[:240]!r}"
                    self._append(self.failures_path, {
                        "purpose": purpose, "model": model,
                        "error": last_err, "retry": attempt,
                        "task_ids": list(task_ids),
                    })
                    parsed = None
                    time.sleep(1.0 * (attempt + 1))
                    continue
                break
            except BudgetExceeded:
                raise
            except Exception as exc:  # noqa: BLE001
                last_err = str(exc)
                self._append(self.failures_path, {
                    "purpose": purpose, "model": model, "error": last_err,
                    "retry": attempt, "task_ids": list(task_ids),
                })
                time.sleep(1.5 * (attempt + 1))
        latency_ms = int((time.perf_counter() - t0) * 1000)
        if parsed is None:
            raise RuntimeError(f"OpenRouter failed: {last_err}")

        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        cached_tokens = int(usage.get("prompt_tokens_details", {}).get("cached_tokens")
                            or usage.get("cached_tokens") or 0)
        cost = float(usage.get("cost")
                     if usage.get("cost") is not None
                     else self._estimate_cost(
                         prompt_tokens, completion_tokens, model))

        row = {
            "schema_version": SCHEMA_VERSION,
            "request_id": request_id,
            "task_ids": list(task_ids),
            "purpose": purpose,
            "model": model,
            "provider": str(provider),
            "model_metadata_hash": short_hash([model, provider]),
            "prompt_template_version": "pilot41.writer.v1" if purpose.startswith("writer")
            else "pilot41.critic.v1",
            "input_hash": input_hash,
            "response_hash": sha256_obj(parsed),
            "raw_response": parsed,
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "cached_tokens": cached_tokens,
            "estimated_cost_usd": cost,
            "actual_cost_usd": cost,
            "latency_ms": latency_ms,
            "retry_count": attempt,
            "structured_output_valid": True,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "key_fingerprint": key_fingerprint(self._key) if self._key else "",
        }
        self._append(self.log_path, row)
        write_json(self.usage_path, {
            "schema_version": SCHEMA_VERSION,
            "budget": self.budget.as_dict(),
            "mode": self.mode,
            "writer_model": self.cfg["writer_model"],
            "critic_model": self.cfg["critic_model"],
        })
        try:
            self.budget.check()
        except BudgetExceeded:
            row["budget_stop"] = True
            raise
        return row

    def write_query(self, contract: Dict[str, Any], *,
                    rewrite_of: Optional[str] = None,
                    failure_reasons: Optional[List[str]] = None
                    ) -> Dict[str, Any]:
        # strip program summary from user-visible copy note
        safe = {k: v for k, v in contract.items()}
        user = {
            "instruction": "Write one natural English user request.",
            "contract": {k: safe[k] for k in safe
                         if k != "semantic_program_summary"},
            # summary is hidden context — model may use for necessity, not copy
            "hidden_program_summary_do_not_copy": safe.get(
                "semantic_program_summary"),
            "rewrite_of": rewrite_of,
            "failure_reasons": failure_reasons or [],
        }
        messages = [
            {"role": "system", "content": WRITER_SYSTEM},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ]
        return self.complete_json(
            purpose="writer" if not rewrite_of else "rewrite",
            model=self.cfg["writer_model"],
            messages=messages, schema=WRITER_SCHEMA,
            task_ids=[contract.get("task_id") or "unknown"])

    def critique(self, contract: Dict[str, Any], query: str,
                 det_findings: Dict[str, Any], *,
                 use_audit_model: bool = False) -> Dict[str, Any]:
        layers = det_findings.get("layers") or {}
        det_slim = {
            "passed": det_findings.get("passed"),
            "failed_validators": [
                k for k, v in layers.items() if not (v or {}).get("passed")],
            "warnings": [
                w for v in layers.values() for w in ((v or {}).get("warnings") or [])
            ][:12],
        }
        user = {
            "contract": {k: contract[k] for k in contract
                         if k != "semantic_program_summary"},
            "query": query,
            "deterministic_findings": det_slim,
        }
        messages = [
            {"role": "system", "content": CRITIC_SYSTEM},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ]
        model = (self.cfg["audit_model"] if use_audit_model
                 else self.cfg["critic_model"])
        return self.complete_json(
            purpose="audit" if use_audit_model else "critic",
            model=model, messages=messages, schema=CRITIC_SCHEMA,
            task_ids=[contract.get("task_id") or "unknown"])
