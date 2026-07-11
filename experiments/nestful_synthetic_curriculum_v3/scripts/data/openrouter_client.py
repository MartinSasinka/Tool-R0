"""Minimal OpenRouter chat client for the agentic data pipeline (stdlib only).

Security: the API key is read from the OPENROUTER_API_KEY environment variable
at call time and is NEVER stored on the client object, written to disk, or
included in logs/raw dumps. Raw responses are saved with redacted metadata.

Features required by the agentic build (docs/AGENTIC_DATA_GENERATION.md):
  * configurable model per role (challenger/weak/strong/judge);
  * retry with exponential backoff + jitter, Retry-After respected;
  * JSON-mode prompting (response_format=json_object) with graceful fallback
    when a model rejects it;
  * robust JSON extraction + local repair (code fences, trailing commas);
  * prompt-hash response cache (OPENROUTER_CACHE=1) to avoid duplicate cost;
  * request / spend budget guard (raises BudgetExceeded BEFORE the request);
  * cost/token accounting from response `usage` (OpenRouter usage.include);
  * `mock` backend for offline tests — a caller-supplied handler produces the
    completion text, everything else (cache, budget, raw dumps) is identical;
  * OPENROUTER_DRY_RUN=1 — no network, prints what would be sent.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Fallback per-token pricing used ONLY when the response carries no usage.cost.
# Overridable because model prices drift (USD per 1M tokens).
DEFAULT_PRICE_PROMPT_PER_M = float(os.environ.get("OPENROUTER_PRICE_PROMPT_PER_M", "0.30"))
DEFAULT_PRICE_COMPLETION_PER_M = float(os.environ.get("OPENROUTER_PRICE_COMPLETION_PER_M", "1.20"))


class BudgetExceeded(RuntimeError):
    """Raised before a request would exceed the request or spend budget."""


class OpenRouterError(RuntimeError):
    pass


class OfflineCacheMiss(OpenRouterError):
    """Raised in offline mode when a prompt is not in the cache (no API calls
    are ever made in offline mode — used for zero-cost salvage/replay)."""


def _api_error_message(resp: Any) -> Optional[str]:
    """Human-readable error when the JSON body lacks a usable completion."""
    if not isinstance(resp, dict):
        return f"non-object response: {type(resp).__name__}"
    err = resp.get("error")
    if err is not None:
        if isinstance(err, dict):
            code = err.get("code")
            msg = err.get("message") or str(err)
            return f"API error {code}: {msg}" if code is not None else str(msg)
        return str(err)
    choices = resp.get("choices")
    if not choices:
        return "response missing choices"
    return None


def weak_solver_backend() -> str:
    """openrouter (default) | local (HF weak solver on this machine)."""
    b = os.environ.get("WEAK_SOLVER_BACKEND", "openrouter")
    if b not in ("openrouter", "local"):
        raise ValueError(f"WEAK_SOLVER_BACKEND={b!r} not in (openrouter, local)")
    return b


def _local_weak_generate(messages: List[Dict[str, str]], *,
                         temperature: float, max_tokens: int,
                         seed: Optional[int]) -> str:
    import sys
    v3_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if v3_root not in sys.path:
        sys.path.insert(0, v3_root)
    from lib.agentic_data.local_llm import get_local_weak_solver
    return get_local_weak_solver().generate(
        messages, temperature=temperature, max_tokens=max_tokens, seed=seed)


def _retry_delay(attempt: int, retry_after: Optional[str] = None) -> float:
    if retry_after:
        try:
            return float(retry_after)
        except (TypeError, ValueError):
            pass
    return min(60.0, (2 ** attempt)) + random.uniform(0, 0.5)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extract_json(text: str) -> Any:
    """Extract the first JSON object/array from an LLM completion.

    Local repair only (no extra API cost): strips markdown code fences,
    scans for the first balanced {...} or [...], removes trailing commas.
    Raises ValueError when nothing parses.
    """
    if text is None:
        raise ValueError("empty completion")
    s = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", s, re.DOTALL)
    if fence:
        s = fence.group(1).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    start = None
    for i, ch in enumerate(s):
        if ch in "{[":
            start = i
            break
    if start is None:
        raise ValueError("no JSON object found in completion")
    opener, closer = s[start], {"{": "}", "[": "]"}[s[start]]
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                frag = s[start:i + 1]
                try:
                    return json.loads(frag)
                except json.JSONDecodeError:
                    repaired = re.sub(r",\s*([}\]])", r"\1", frag)
                    return json.loads(repaired)
    raise ValueError("unbalanced JSON in completion")


@dataclass
class ClientStats:
    n_requests: int = 0
    n_cache_hits: int = 0
    n_retries: int = 0
    n_json_fallbacks: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    spend_usd: float = 0.0
    by_role: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def record(self, role: str, prompt_toks: int, completion_toks: int,
               cost: float, cached: bool, *, local: bool = False) -> None:
        r = self.by_role.setdefault(role, {"requests": 0, "cache_hits": 0,
                                           "prompt_tokens": 0, "completion_tokens": 0,
                                           "spend_usd": 0.0, "local_requests": 0})
        if cached:
            self.n_cache_hits += 1
            r["cache_hits"] += 1
            return
        if local:
            r["local_requests"] = r.get("local_requests", 0) + 1
            return
        self.n_requests += 1
        self.prompt_tokens += prompt_toks
        self.completion_tokens += completion_toks
        self.spend_usd += cost
        r["requests"] += 1
        r["prompt_tokens"] += prompt_toks
        r["completion_tokens"] += completion_toks
        r["spend_usd"] += cost

    def as_dict(self) -> Dict[str, Any]:
        return {
            "n_requests": self.n_requests,
            "n_cache_hits": self.n_cache_hits,
            "n_retries": self.n_retries,
            "n_json_fallbacks": self.n_json_fallbacks,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "spend_usd": round(self.spend_usd, 6),
            "by_role": {k: {kk: (round(vv, 6) if isinstance(vv, float) else vv)
                            for kk, vv in v.items()}
                        for k, v in self.by_role.items()},
        }


class OpenRouterClient:
    """Chat client with caching, budgets and redacted raw-response dumps."""

    def __init__(
        self,
        *,
        cache_dir: Optional[str] = None,
        raw_dir: Optional[str] = None,
        max_retries: Optional[int] = None,
        max_requests: Optional[int] = None,
        max_spend_usd: Optional[float] = None,
        use_cache: Optional[bool] = None,
        save_raw: Optional[bool] = None,
        dry_run: Optional[bool] = None,
        offline: Optional[bool] = None,
        backend: str = "openrouter",           # "openrouter" | "mock"
        mock_handler: Optional[Callable[[str, List[Dict[str, str]]], str]] = None,
        timeout_s: float = 120.0,
        app_title: str = "nestful-agentic-curriculum",
    ) -> None:
        env = os.environ
        self.cache_dir = cache_dir
        self.raw_dir = raw_dir
        self.max_retries = int(env.get("OPENROUTER_MAX_RETRIES", "5")) \
            if max_retries is None else max_retries
        self.max_requests = int(env.get("OPENROUTER_MAX_REQUESTS", "1000")) \
            if max_requests is None else max_requests
        self.max_spend_usd = float(env.get("OPENROUTER_MAX_SPEND_USD", "20")) \
            if max_spend_usd is None else max_spend_usd
        self.use_cache = (env.get("OPENROUTER_CACHE", "1") == "1") \
            if use_cache is None else use_cache
        self.save_raw = (env.get("OPENROUTER_SAVE_RAW", "1") == "1") \
            if save_raw is None else save_raw
        self.dry_run = (env.get("OPENROUTER_DRY_RUN", "0") == "1") \
            if dry_run is None else dry_run
        # offline: serve ONLY from cache, raise OfflineCacheMiss otherwise.
        # Guarantees zero API spend (used by --salvage / replay).
        self.offline = (env.get("OPENROUTER_OFFLINE", "0") == "1") \
            if offline is None else offline
        if self.offline:
            self.use_cache = True
        self.backend = backend
        self.mock_handler = mock_handler
        self.timeout_s = timeout_s
        self.app_title = app_title
        self.stats = ClientStats()
        if backend == "mock" and mock_handler is None:
            raise ValueError("backend='mock' requires mock_handler")
        if backend == "openrouter" and not self.dry_run and not self.offline \
                and not env.get("OPENROUTER_API_KEY"):
            raise OpenRouterError(
                "OPENROUTER_API_KEY is not set. Export it (never hardcode it) "
                "or use OPENROUTER_DRY_RUN=1 / the mock backend.")
        for d in (self.cache_dir, self.raw_dir):
            if d:
                os.makedirs(d, exist_ok=True)

    # ------------------------------------------------------------------ utils
    def _prompt_key(self, model: str, messages: List[Dict[str, str]],
                    params: Dict[str, Any]) -> str:
        blob = json.dumps({"model": model, "messages": messages, "params": params},
                          sort_keys=True, ensure_ascii=False)
        return _sha(blob)

    def _cache_path(self, key: str) -> Optional[str]:
        return os.path.join(self.cache_dir, f"{key}.json") if self.cache_dir else None

    def _check_budget(self) -> None:
        if self.stats.n_requests >= self.max_requests:
            raise BudgetExceeded(
                f"request budget exhausted ({self.stats.n_requests}/{self.max_requests})")
        if self.stats.spend_usd >= self.max_spend_usd:
            raise BudgetExceeded(
                f"spend budget exhausted (${self.stats.spend_usd:.4f}/"
                f"${self.max_spend_usd:.2f})")

    def _save_raw(self, role: str, key: str, record: Dict[str, Any]) -> Optional[str]:
        if not (self.save_raw and self.raw_dir):
            return None
        # REDACTION: record contains only model/messages/response/usage —
        # never headers, never the API key.
        role_dir = os.path.join(self.raw_dir, role)
        os.makedirs(role_dir, exist_ok=True)
        path = os.path.join(role_dir, f"{key}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2, ensure_ascii=False)
        return path

    # ------------------------------------------------------------------ HTTP
    def _http_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        last_err: Optional[str] = None
        for attempt in range(self.max_retries + 1):
            # Key is read per-request from the environment; nothing on `self`.
            req = urllib.request.Request(
                OPENROUTER_URL, data=body, method="POST",
                headers={
                    "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                    "Content-Type": "application/json",
                    "X-Title": self.app_title,
                })
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                status = exc.code
                detail = ""
                try:
                    detail = exc.read().decode("utf-8", errors="replace")[:500]
                except Exception:  # noqa: BLE001
                    pass
                last_err = f"HTTP {status}: {detail}"
                if status == 400:
                    raise OpenRouterError(last_err)  # not retryable; caller may fall back
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                if status in (402, 401, 403):
                    raise OpenRouterError(last_err)  # auth / credit problems: stop
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_err = f"network error: {exc}"
                retry_after = None
            if attempt >= self.max_retries:
                break
            self.stats.n_retries += 1
            delay = float(retry_after) if retry_after else min(60.0, (2 ** attempt)) \
                + random.uniform(0, 0.5)
            time.sleep(delay)
        raise OpenRouterError(f"request failed after {self.max_retries + 1} attempts: {last_err}")

    # ------------------------------------------------------------------ main
    def chat(
        self,
        *,
        role: str,                         # challenger | weak_solver | strong_solver | judge
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        json_mode: bool = True,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Returns {"text", "parsed" (or None), "cached", "cost_usd", "raw_path"}."""
        params = {"temperature": temperature, "max_tokens": max_tokens,
                  "json_mode": json_mode, "seed": seed}
        key = self._prompt_key(model, messages, params)

        cache_path = self._cache_path(key)
        if self.use_cache and cache_path and os.path.isfile(cache_path):
            with open(cache_path, encoding="utf-8") as fh:
                cached = json.load(fh)
            self.stats.record(role, 0, 0, 0.0, cached=True)
            return {"text": cached["text"], "parsed": cached.get("parsed"),
                    "cached": True, "cost_usd": 0.0,
                    "raw_path": cached.get("raw_path")}

        if self.offline:
            raise OfflineCacheMiss(
                f"offline mode: prompt {key[:12]} for role={role} not in cache "
                f"({self.cache_dir}) — refusing to spend API money")

        if self.dry_run:
            print(f"[openrouter DRY RUN] role={role} model={model} "
                  f"prompt_hash={key[:12]} max_tokens={max_tokens} (not sent)")
            return {"text": "", "parsed": None, "cached": False,
                    "cost_usd": 0.0, "raw_path": None, "dry_run": True}

        use_local_weak = (role == "weak_solver"
                          and weak_solver_backend() == "local"
                          and self.backend != "mock")

        if not use_local_weak:
            self._check_budget()

        if use_local_weak:
            text = _local_weak_generate(
                messages, temperature=temperature, max_tokens=max_tokens,
                seed=seed)
            usage = {"prompt_tokens": 0, "completion_tokens": 0}
            cost = 0.0
        elif self.backend == "mock":
            text = self.mock_handler(role, messages)
            usage = {"prompt_tokens": 0, "completion_tokens": 0}
            cost = 0.0
        else:
            payload: Dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "usage": {"include": True},
            }
            if seed is not None:
                payload["seed"] = seed
            if json_mode:
                payload["response_format"] = {"type": "json_object"}

            text: Optional[str] = None
            usage: Dict[str, Any] = {}
            cost = 0.0
            last_err: Optional[str] = None
            json_fallback_used = False

            for attempt in range(self.max_retries + 1):
                try:
                    resp = self._http_request(payload)
                except OpenRouterError as exc:
                    last_err = str(exc)
                    if (json_mode and not json_fallback_used
                            and "HTTP 400" in last_err):
                        self.stats.n_json_fallbacks += 1
                        json_fallback_used = True
                        payload.pop("response_format", None)
                        continue
                    if attempt >= self.max_retries:
                        raise OpenRouterError(
                            f"request failed after {self.max_retries + 1} "
                            f"attempts: {last_err}") from exc
                    self.stats.n_retries += 1
                    time.sleep(_retry_delay(attempt))
                    continue

                api_err = _api_error_message(resp)
                if api_err:
                    last_err = api_err
                    if attempt >= self.max_retries:
                        raise OpenRouterError(
                            f"request failed after {self.max_retries + 1} "
                            f"attempts: {last_err}")
                    self.stats.n_retries += 1
                    time.sleep(_retry_delay(attempt))
                    continue

                try:
                    text = resp["choices"][0]["message"]["content"] or ""
                except (KeyError, IndexError, TypeError) as exc:
                    last_err = (f"malformed response: {exc}: "
                                f"{str(resp)[:300]}")
                    if attempt >= self.max_retries:
                        raise OpenRouterError(last_err) from exc
                    self.stats.n_retries += 1
                    time.sleep(_retry_delay(attempt))
                    continue

                usage = resp.get("usage") or {}
                cost = usage.get("cost")
                if cost is None:
                    cost = (usage.get("prompt_tokens", 0) / 1e6
                            * DEFAULT_PRICE_PROMPT_PER_M
                            + usage.get("completion_tokens", 0) / 1e6
                            * DEFAULT_PRICE_COMPLETION_PER_M)
                cost = float(cost)
                break
            else:
                raise OpenRouterError(
                    f"request failed after {self.max_retries + 1} attempts: "
                    f"{last_err or 'unknown'}")
            assert text is not None

        parsed: Optional[Any] = None
        try:
            parsed = extract_json(text)
        except ValueError:
            parsed = None

        raw_path = self._save_raw(role, key, {
            "role": role, "model": model, "prompt_hash": key,
            "messages": messages, "text": text,
            "usage": {"prompt_tokens": usage.get("prompt_tokens", 0),
                      "completion_tokens": usage.get("completion_tokens", 0),
                      "cost_usd": round(cost, 6)},
            "backend": "local_hf" if use_local_weak else self.backend,
        })
        self.stats.record(role, usage.get("prompt_tokens", 0),
                          usage.get("completion_tokens", 0), cost, cached=False,
                          local=use_local_weak)

        if self.use_cache and cache_path:
            with open(cache_path, "w", encoding="utf-8") as fh:
                json.dump({"text": text, "parsed": parsed, "raw_path": raw_path},
                          fh, ensure_ascii=False)
        return {"text": text, "parsed": parsed, "cached": False,
                "cost_usd": cost, "raw_path": raw_path}


def models_from_env() -> Dict[str, str]:
    """Role → model slug (OpenRouter) or local HF id for weak_solver."""
    env = os.environ
    ds = env.get("OPENROUTER_DEFAULT_MODEL", "deepseek/deepseek-v3.2")
    qwen235 = env.get("OPENROUTER_STRONG_DEFAULT",
                      "qwen/qwen3-235b-a22b-2507")
    local_weak = env.get("LOCAL_WEAK_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
    weak = (local_weak if weak_solver_backend() == "local"
            else env.get("OPENROUTER_WEAK_MODEL", ds))
    return {
        "challenger": env.get("OPENROUTER_CHALLENGER_MODEL", ds),
        "weak_solver": weak,
        "strong_solver": env.get("OPENROUTER_STRONG_MODEL", qwen235),
        "judge": env.get("OPENROUTER_JUDGE_MODEL", ds),
    }
