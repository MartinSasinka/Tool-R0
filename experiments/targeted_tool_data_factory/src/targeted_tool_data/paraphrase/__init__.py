"""OpenRouter paraphrasing provider (pilot2).

The LLM is used for ONE thing only: rewriting an already valid, already
executed synthetic query into more natural English. It never sees the target
benchmark, never proposes tools, arguments, constants or answers, and its
output is accepted only after a deterministic validator proves that the
program is unchanged (DECISIONS.md D06 / DESIGN.md §9).

Safety:
  * the API key is read from the repository root ``.env`` at call time and is
    never logged, never written into any artifact and never committed;
  * every request is capped by a budget guard (requests + USD);
  * every response is cached by content hash, so re-runs are free and the
    pipeline is resumable.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..util import EXPERIMENTS_ROOT, read_json, sha256_obj, write_json

REPO_ROOT = EXPERIMENTS_ROOT.parent
DEFAULT_MODEL = "mistralai/mistral-small-24b-instruct-2501"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


# ── key handling ──────────────────────────────────────────────────────────
def load_env_file(path: Optional[Path] = None) -> Dict[str, str]:
    """Minimal .env reader (no dependency on python-dotenv being present)."""
    p = path or (REPO_ROOT / ".env")
    out: Dict[str, str] = {}
    if not p.is_file():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def get_api_key() -> Optional[str]:
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key.strip()
    return load_env_file().get("OPENROUTER_API_KEY")


def key_fingerprint(key: str) -> str:
    """Stable, non-reversible identifier for reports (never the key itself)."""
    return "sha256:" + sha256_obj(key)[:12]


# ── budget guard ──────────────────────────────────────────────────────────
class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Budget:
    max_requests: int = 600
    max_usd: float = 2.0
    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    usd: float = 0.0

    def check(self) -> None:
        if self.requests >= self.max_requests:
            raise BudgetExceeded(f"request cap reached ({self.max_requests})")
        if self.usd >= self.max_usd:
            raise BudgetExceeded(f"USD cap reached ({self.max_usd})")

    def add(self, prompt_tokens: int, completion_tokens: int, usd: float) -> None:
        self.requests += 1
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.usd += usd

    def as_dict(self) -> Dict[str, Any]:
        return {"requests": self.requests, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "usd": round(self.usd, 6), "max_requests": self.max_requests,
                "max_usd": self.max_usd}


# ── prompt ────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You rewrite short arithmetic tool-use instructions into more natural "
    "English. You never solve anything, never compute, never add or remove "
    "numbers, and never change the order of the steps. You answer with JSON "
    "only."
)

USER_TEMPLATE = """Rewrite the request below as {n} alternative English phrasings.

REQUEST:
{query}

The request describes {steps} operations that must be carried out in this exact order:
{step_list}

Hard rules for every rewrite:
1. Keep every number exactly as written. Do not add any number that is not already in the request, and do not drop any.
2. Keep the operations in the same order, one mention per operation.
3. Whenever an operation uses the result of an earlier operation, say so explicitly (for example "that result", "the previous value", "the result of step 2").
4. Never state or hint at any computed value, intermediate or final.
5. One short paragraph, at most {maxlen} characters, unambiguous, plain English, no markdown, no bullet points, no step numbering unless the original had it.
6. Do not mention tools, functions, APIs or JSON.

Answer with exactly this JSON object and nothing else:
{{"paraphrases": ["<rewrite 1>", "<rewrite 2>"]}}"""


def build_prompt(query: str, step_descriptions: List[str], *, n: int = 2,
                 maxlen: int = 420) -> List[Dict[str, str]]:
    step_list = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(step_descriptions))
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(
            n=n, query=query, steps=len(step_descriptions),
            step_list=step_list, maxlen=maxlen)},
    ]


# ── client ────────────────────────────────────────────────────────────────
@dataclass
class ParaphraseClient:
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    cache_dir: Path = field(default_factory=lambda: Path("cache/openrouter"))
    budget: Budget = field(default_factory=Budget)
    temperature: float = 0.7
    max_tokens: int = 480
    timeout: float = 90.0
    retries: int = 3
    _key: Optional[str] = None
    stats: Dict[str, int] = field(default_factory=lambda: {
        "cache_hits": 0, "api_calls": 0, "errors": 0, "retries": 0})

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._key = get_api_key()

    @property
    def available(self) -> bool:
        return bool(self._key)

    def cache_key(self, messages: List[Dict[str, str]]) -> str:
        return sha256_obj({"model": self.model, "messages": messages,
                           "temperature": self.temperature,
                           "max_tokens": self.max_tokens})

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.json"

    def complete(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Cached chat completion. Returns {content, usage, cached, model}."""
        key = self.cache_key(messages)
        path = self._cache_path(key)
        if path.is_file():
            self.stats["cache_hits"] += 1
            cached = read_json(path)
            cached["cached"] = True
            return cached
        if not self._key:
            raise RuntimeError("OPENROUTER_API_KEY not available")
        self.budget.check()

        import httpx

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "usage": {"include": True},
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self._key}",
            "Content-Type": "application/json",
            "X-Title": "targeted-tool-data-factory",
        }
        last_err: Optional[str] = None
        for attempt in range(self.retries):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(f"{self.base_url}/chat/completions",
                                       json=payload, headers=headers)
                if resp.status_code in (429, 500, 502, 503, 504):
                    last_err = f"http_{resp.status_code}"
                    self.stats["retries"] += 1
                    time.sleep(2.0 * (attempt + 1))
                    continue
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:                     # noqa: BLE001
                last_err = f"{type(exc).__name__}"
                self.stats["retries"] += 1
                time.sleep(2.0 * (attempt + 1))
                continue
            usage = data.get("usage") or {}
            out = {
                "content": (data.get("choices") or [{}])[0]
                .get("message", {}).get("content", ""),
                "usage": {
                    "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                    "completion_tokens": int(usage.get("completion_tokens") or 0),
                    "cost": float(usage.get("cost") or 0.0),
                },
                "model": data.get("model", self.model),
                "cached": False,
            }
            self.budget.add(out["usage"]["prompt_tokens"],
                            out["usage"]["completion_tokens"],
                            out["usage"]["cost"])
            self.stats["api_calls"] += 1
            write_json(path, out)
            return out
        self.stats["errors"] += 1
        raise RuntimeError(f"openrouter request failed: {last_err}")


def parse_paraphrases(content: str) -> List[str]:
    """Tolerant JSON extraction; returns [] when the answer is unusable."""
    if not content:
        return []
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return []
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
    items = obj.get("paraphrases") if isinstance(obj, dict) else None
    if not isinstance(items, list):
        return []
    return [str(x).strip() for x in items if isinstance(x, str) and x.strip()]
