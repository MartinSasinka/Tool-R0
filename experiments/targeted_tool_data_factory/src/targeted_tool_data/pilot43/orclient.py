"""OpenRouter transport for the Pilot4.3 query writer, critics and rewriter.

The failure mode this module exists to prevent is a *silently mixed run*. The
Pilot4.2 logs turned out to hold records from three generator runs at once,
responses from a model OpenRouter had substituted for the pinned slug, and
prompt versions that no longer matched the freeze manifest -- none of which was
detectable after the fact. Here every one of those is a hard refusal:

* the output directory basename must be the run id,
* the prompt version must start with ``pilot43.``,
* the model that answered must be the model that was configured,
* a log file that already contains a foreign ``run_id`` is never appended to.

Model slugs are never written in Python; they come from
``configs/pilot4_3_openrouter.yaml`` so the manifest can record what was frozen.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from ..paraphrase import key_fingerprint, load_env_file
from ..repro import write_json
from . import PROMPT_VERSION_PREFIX, RUN_ID

SCHEMA_VERSION = "ttdf.pilot43.openrouter.v1"
DEFAULT_CONFIG = "pilot4_3_openrouter.yaml"

REQUEST_LOG = "openrouter_requests_pilot43.jsonl"
FAILURE_LOG = "openrouter_failures_pilot43.jsonl"
USAGE_FILE = "openrouter_usage_pilot43.json"
CACHE_DIR = "or_cache"
LOG_FILES = (REQUEST_LOG, FAILURE_LOG)

PURPOSES = ("writer", "critic", "critic2", "rewrite", "audit")
#: slugs that route to a model the freeze manifest cannot name
FORBIDDEN_MODEL_TOKENS = (":free", "/auto", "openrouter/auto", "latest", "*")
RETRY_STATUS = (408, 409, 425, 429, 500, 502, 503, 504)


class RunIsolationError(RuntimeError):
    """A write would mix this run with another run, model or prompt version."""


class BudgetExceeded(RuntimeError):
    """A USD cap was reached. ``scope`` is ``"total"`` or ``"task"``."""

    def __init__(self, message: str, *, scope: str = "total") -> None:
        super().__init__(message)
        self.scope = scope


class StructuredOutputError(RuntimeError):
    """The model never returned JSON matching the strict schema."""


class TransportError(RuntimeError):
    """The request never produced a usable HTTP response."""


class ReplayMiss(RuntimeError):
    """``replay_only`` was set and the cache has no entry for this request."""


# ── model pinning ────────────────────────────────────────────────────────
def assert_pinned_model(model: str) -> str:
    slug = (model or "").strip()
    if not slug or "/" not in slug:
        raise ValueError(f"not an OpenRouter model slug: {model!r}")
    low = slug.lower()
    for token in FORBIDDEN_MODEL_TOKENS:
        if token in low:
            raise ValueError(f"refusing unpinned/router model {model!r}")
    return slug


def _family(model: str) -> str:
    return model.split("/", 1)[0].lower()


# ── configuration ────────────────────────────────────────────────────────
def default_config_path() -> Path:
    return Path(__file__).resolve().parents[3] / "configs" / DEFAULT_CONFIG


@dataclass(frozen=True)
class OpenRouterConfig:
    """Frozen view of the YAML. Nothing here is defaulted from Python code."""

    run_id: str
    base_url: str
    models: Dict[str, str]
    prompt_versions: Dict[str, str]
    allow_fallbacks: bool
    require_structured_outputs: bool
    temperature: float
    top_p: float
    max_output_tokens: int
    max_retries: int
    backoff_seconds_base: float
    backoff_max_seconds: float
    request_timeout_seconds: float
    max_total_cost_usd: float
    max_cost_per_task_usd: float
    cache_namespace: str
    second_critic_sample_rate: float
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    def model_for(self, purpose: str) -> str:
        if purpose not in self.models:
            raise KeyError(f"no model configured for purpose {purpose!r}")
        return self.models[purpose]

    def prompt_version_for(self, purpose: str) -> str:
        if purpose not in self.prompt_versions:
            raise KeyError(f"no prompt_version configured for {purpose!r}")
        return self.prompt_versions[purpose]

    def manifest(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "models": dict(self.models),
            "prompt_versions": dict(self.prompt_versions),
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_output_tokens": self.max_output_tokens,
            "allow_fallbacks": self.allow_fallbacks,
            "cache_namespace": self.cache_namespace,
        }


def load_openrouter_config(path: Optional[Path] = None) -> OpenRouterConfig:
    """Load and hard-validate the YAML. Raises rather than defaulting."""
    import yaml

    src = Path(path) if path else default_config_path()
    if not src.is_file():
        raise FileNotFoundError(f"missing OpenRouter config: {src}")
    loaded = yaml.safe_load(src.read_text(encoding="utf-8")) or {}
    block = loaded.get("openrouter") or loaded
    return build_config(block)


def build_config(block: Mapping[str, Any]) -> OpenRouterConfig:
    run_id = str(block.get("run_id") or "")
    if run_id != RUN_ID:
        raise RunIsolationError(
            f"config run_id {run_id!r} is not the Pilot4.3 run id {RUN_ID!r}")
    if block.get("allow_fallbacks"):
        raise ValueError("allow_fallbacks must be false for the Pilot4.3 freeze")
    if not block.get("require_structured_outputs"):
        raise ValueError("require_structured_outputs must be true")

    writer = assert_pinned_model(str(block["writer_model"]))
    critic = assert_pinned_model(str(block["critic_model"]))
    second = assert_pinned_model(str(block["second_critic_model"]))
    rewrite = assert_pinned_model(str(block["rewrite_model"]))
    audit = assert_pinned_model(str(block.get("audit_model") or second))
    if _family(critic) == _family(writer):
        raise ValueError("critic_model must be a different family than writer_model")
    if _family(second) == _family(critic):
        raise ValueError(
            "second_critic_model must be a different family than critic_model")

    versions_block = block.get("prompt_version") or {}
    if not isinstance(versions_block, Mapping):
        raise ValueError("prompt_version must map purpose -> version string")
    versions = {str(k): str(v) for k, v in versions_block.items()}
    for purpose in PURPOSES:
        version = versions.get(purpose, "")
        if not version.startswith(PROMPT_VERSION_PREFIX + "."):
            raise RunIsolationError(
                f"prompt_version for {purpose!r} must start with "
                f"{PROMPT_VERSION_PREFIX}.: got {version!r}")

    return OpenRouterConfig(
        run_id=run_id,
        base_url=str(block.get("base_url") or "https://openrouter.ai/api/v1"),
        models={"writer": writer, "critic": critic, "critic2": second,
                "rewrite": rewrite, "audit": audit},
        prompt_versions=versions,
        allow_fallbacks=False,
        require_structured_outputs=True,
        temperature=float(block["temperature"]),
        top_p=float(block["top_p"]),
        max_output_tokens=int(block["max_output_tokens"]),
        max_retries=int(block["max_retries"]),
        backoff_seconds_base=float(block["backoff_seconds_base"]),
        backoff_max_seconds=float(block.get("backoff_max_seconds") or 60.0),
        request_timeout_seconds=float(block["request_timeout_seconds"]),
        max_total_cost_usd=float(block["max_total_cost_usd"]),
        max_cost_per_task_usd=float(block["max_cost_per_task_usd"]),
        cache_namespace=str(block["cache_namespace"]),
        second_critic_sample_rate=float(block.get("second_critic_sample_rate") or 0.0),
        raw=dict(block),
    )


# ── transport ────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class HttpResponse:
    status_code: int
    text: str
    headers: Dict[str, str] = field(default_factory=dict)

    def json(self) -> Dict[str, Any]:
        return json.loads(self.text)


#: (url, payload, headers, timeout) -> HttpResponse. Injectable so the tests
#: exercise every retry and isolation path without a socket.
Transport = Callable[[str, Dict[str, Any], Dict[str, str], float], HttpResponse]


def httpx_transport(url: str, payload: Dict[str, Any], headers: Dict[str, str],
                    timeout: float) -> HttpResponse:
    import httpx

    with httpx.Client(timeout=timeout) as client:
        resp = client.post(url, json=payload, headers=headers)
    return HttpResponse(status_code=resp.status_code, text=resp.text,
                        headers={k.lower(): v for k, v in resp.headers.items()})


def get_api_key() -> Optional[str]:
    key = os.environ.get("OPENROUTER_API_KEY")
    if key and key.strip():
        return key.strip()
    from_env_file = load_env_file().get("OPENROUTER_API_KEY")
    return from_env_file.strip() if from_env_file else None


# ── run isolation ────────────────────────────────────────────────────────
def _iter_log_run_ids(path: Path) -> List[str]:
    out: List[str] = []
    if not path.is_file():
        return out
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                out.append("<unparseable>")
                continue
            out.append(str(row.get("run_id") or "<missing>"))
    return out


def count_foreign_run_records(out_dir: Path, run_id: str = RUN_ID) -> int:
    """Number of log records in ``out_dir`` that belong to another run."""
    total = 0
    for name in LOG_FILES:
        total += sum(1 for rid in _iter_log_run_ids(Path(out_dir) / name)
                     if rid != run_id)
    return total


def assert_log_isolation(out_dir: Path, run_id: str = RUN_ID) -> Dict[str, Any]:
    """Refuse to touch a directory whose logs already hold a foreign run."""
    out_dir = Path(out_dir)
    if out_dir.name != run_id:
        raise RunIsolationError(
            f"output directory {out_dir.name!r} is not the run id {run_id!r}")
    seen: Dict[str, int] = {}
    for name in LOG_FILES:
        for rid in _iter_log_run_ids(out_dir / name):
            seen[rid] = seen.get(rid, 0) + 1
    foreign = {rid: n for rid, n in seen.items() if rid != run_id}
    if foreign:
        raise RunIsolationError(
            f"{out_dir} already holds records from other runs: {sorted(foreign)}")
    return {"run_id": run_id, "records": sum(seen.values()), "foreign": 0}


# ── cache ────────────────────────────────────────────────────────────────
def cache_key(namespace: str, model: str, prompt_version: str,
              messages: Sequence[Mapping[str, str]],
              schema: Mapping[str, Any], sample_id: str = "") -> str:
    """Replay identity of one request.

    ``sample_id`` participates because two tasks can share a contract word for
    word: without it they share a cache entry and therefore the same query text,
    which is an exact duplicate the diversity gate forbids.
    """
    blob = json.dumps(
        {"namespace": namespace, "model": model, "prompt_version": prompt_version,
         "messages": [dict(m) for m in messages], "schema": dict(schema),
         "sample_id": sample_id},
        sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _strip_fence(text: str) -> str:
    body = (text or "").strip()
    if body.startswith("```"):
        body = body.split("\n", 1)[-1] if "\n" in body else body
        if body.endswith("```"):
            body = body[: -3]
    return body.strip()


def parse_json_content(content: str) -> Dict[str, Any]:
    body = _strip_fence(content)
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        start, end = body.find("{"), body.rfind("}")
        if start >= 0 and end > start:
            return json.loads(body[start:end + 1])
        raise


def validate_structured(payload: Any, schema: Mapping[str, Any]) -> None:
    import jsonschema

    jsonschema.validate(payload, dict(schema))


# ── client ───────────────────────────────────────────────────────────────
@dataclass
class Totals:
    requests: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {"requests": self.requests, "cache_hits": self.cache_hits,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "cost_usd": round(self.cost_usd, 8)}


class OpenRouterClient:
    """Structured-output OpenRouter client with per-run write isolation."""

    def __init__(self, cfg: OpenRouterConfig, out_dir: Path, *,
                 transport: Optional[Transport] = None,
                 replay_only: bool = False,
                 sleep: Callable[[float], None] = time.sleep,
                 jitter: Callable[[], float] = random.random,
                 api_key: Optional[str] = None) -> None:
        self.cfg = cfg
        self.out_dir = Path(out_dir)
        self.replay_only = bool(replay_only)
        self._transport = transport or httpx_transport
        self._sleep = sleep
        self._jitter = jitter
        # constructed before any write so a foreign log is refused up front
        assert_log_isolation(self.out_dir, cfg.run_id)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = self.out_dir / CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.request_log = self.out_dir / REQUEST_LOG
        self.failure_log = self.out_dir / FAILURE_LOG
        self.usage_path = self.out_dir / USAGE_FILE
        self._key = api_key if api_key is not None else get_api_key()
        self._checked: set[Path] = set()
        self.totals = Totals()
        self.task_cost: Dict[str, float] = {}
        # a render stage runs many tasks concurrently; logs, usage and the
        # running cost must not interleave
        self._lock = threading.RLock()
        self._load_usage()

    # -- availability ----------------------------------------------------
    def available(self) -> bool:
        """False degrades the caller to deterministic rendering, never a crash."""
        return bool(self.replay_only or self._key)

    def close(self) -> None:
        """Flush the usage file. Safe to call more than once."""
        self._write_usage()

    # -- usage bookkeeping ------------------------------------------------
    def _load_usage(self) -> None:
        if not self.usage_path.is_file():
            return
        try:
            prior = json.loads(self.usage_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if str(prior.get("run_id") or self.cfg.run_id) != self.cfg.run_id:
            raise RunIsolationError(
                f"{self.usage_path} belongs to run {prior.get('run_id')!r}")
        totals = prior.get("totals") or {}
        self.totals = Totals(
            requests=int(totals.get("requests") or 0),
            cache_hits=int(totals.get("cache_hits") or 0),
            prompt_tokens=int(totals.get("prompt_tokens") or 0),
            completion_tokens=int(totals.get("completion_tokens") or 0),
            cost_usd=float(totals.get("cost_usd") or 0.0))
        self.task_cost = {str(k): float(v)
                          for k, v in (prior.get("task_cost") or {}).items()}

    def _write_usage(self) -> None:
        with self._lock:
            self._write_usage_locked()

    def _write_usage_locked(self) -> None:
        write_json(self.usage_path, {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.cfg.run_id,
            "totals": self.totals.as_dict(),
            "max_total_cost_usd": self.cfg.max_total_cost_usd,
            "max_cost_per_task_usd": self.cfg.max_cost_per_task_usd,
            "replay_only": self.replay_only,
            "config": self.cfg.manifest(),
            "task_cost": {k: round(v, 8) for k, v in sorted(self.task_cost.items())},
        })

    # -- log writing ------------------------------------------------------
    def _append(self, path: Path, row: Dict[str, Any]) -> None:
        if row.get("run_id") != self.cfg.run_id:
            raise RunIsolationError(f"refusing to log run_id {row.get('run_id')!r}")
        with self._lock:
            if path not in self._checked:
                foreign = [rid for rid in _iter_log_run_ids(path)
                           if rid != self.cfg.run_id]
                if foreign:
                    raise RunIsolationError(
                        f"{path.name} already holds run_id {sorted(set(foreign))}")
                self._checked.add(path)
            with path.open("a", encoding="utf-8", newline="\n") as fh:
                fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    def _fail(self, meta: Mapping[str, Any], purpose: str, model: str,
              attempt: int, error: str, *, http_status: Optional[int] = None,
              prompt_version: str = "") -> None:
        self._append(self.failure_log, {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.cfg.run_id,
            "sample_id": meta.get("sample_id", ""),
            "workflow_id": meta.get("workflow_id", ""),
            "semantic_program_id": meta.get("semantic_program_id", ""),
            "purpose": purpose,
            "prompt_version": prompt_version,
            "configured_model": model,
            "attempt": attempt,
            "http_status": http_status,
            "error": error[:600],
            "timestamp": _utc_now(),
        })

    # -- cache ------------------------------------------------------------
    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _cache_read(self, key: str) -> Optional[Dict[str, Any]]:
        path = self._cache_path(key)
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _cache_write(self, key: str, entry: Dict[str, Any]) -> None:
        self._cache_path(key).write_text(
            json.dumps(entry, ensure_ascii=False, default=str), encoding="utf-8")

    # -- budget -----------------------------------------------------------
    def _check_budget(self, sample_id: str) -> None:
        if self.totals.cost_usd >= self.cfg.max_total_cost_usd:
            raise BudgetExceeded(
                f"total cost cap reached: {self.totals.cost_usd:.6f} USD >= "
                f"{self.cfg.max_total_cost_usd} USD", scope="total")
        if self.task_cost.get(sample_id, 0.0) >= self.cfg.max_cost_per_task_usd:
            raise BudgetExceeded(
                f"per-task cost cap reached for {sample_id!r}", scope="task")

    # -- the request ------------------------------------------------------
    def chat(self, purpose: str, messages: Sequence[Mapping[str, str]],
             schema: Mapping[str, Any], meta: Mapping[str, Any]) -> Dict[str, Any]:
        """One structured-output completion. Returns the parsed content plus
        the log record that was written for it."""
        if purpose not in PURPOSES:
            raise ValueError(f"unknown purpose {purpose!r}")
        model = assert_pinned_model(self.cfg.model_for(purpose))
        prompt_version = str(meta.get("prompt_version")
                             or self.cfg.prompt_version_for(purpose))
        if not prompt_version.startswith(PROMPT_VERSION_PREFIX + "."):
            raise RunIsolationError(
                f"prompt_version {prompt_version!r} is not a "
                f"{PROMPT_VERSION_PREFIX} prompt")
        sample_id = str(meta.get("sample_id") or "")
        key = cache_key(self.cfg.cache_namespace, model, prompt_version,
                        messages, schema, sample_id)

        cached = self._cache_read(key)
        if cached is not None:
            return self._from_cache(cached, purpose, model, prompt_version,
                                    schema, meta, key)
        if self.replay_only:
            raise ReplayMiss(
                f"replay_only: no cache entry for {purpose} {key[:12]}")
        if not self._key:
            raise TransportError("OPENROUTER_API_KEY is not available")
        self._check_budget(sample_id)
        return self._request(purpose, model, prompt_version, messages, schema,
                             meta, key)

    def _from_cache(self, entry: Mapping[str, Any], purpose: str, model: str,
                    prompt_version: str, schema: Mapping[str, Any],
                    meta: Mapping[str, Any], key: str) -> Dict[str, Any]:
        actual = str(entry.get("actual_model") or "")
        self._assert_same_model(model, actual, purpose)
        raw_text = str(entry.get("raw_text") or "")
        parsed = parse_json_content(raw_text)
        validate_structured(parsed, schema)
        with self._lock:
            self.totals.cache_hits += 1
        record = self._record(
            purpose=purpose, prompt_version=prompt_version, model=model,
            actual_model=actual, provider=str(entry.get("provider") or ""),
            request_id=str(entry.get("request_id") or ""), meta=meta,
            latency_ms=0, usage=dict(entry.get("usage") or {}), cost_usd=0.0,
            cache_hit=True, http_status=int(entry.get("http_status") or 200),
            attempt=0, cache_key=key, raw_sha=sha256_text(raw_text))
        self._append(self.request_log, record)
        self._write_usage()
        return {"parsed": parsed, "raw_text": raw_text, "cache_hit": True,
                "raw_response_sha256": record["raw_response_sha256"],
                "actual_model": actual, "record": record, "attempts": 0,
                "cost_usd": 0.0}

    def _assert_same_model(self, configured: str, actual: str,
                           purpose: str) -> None:
        if not actual or actual.strip() != configured:
            raise RunIsolationError(
                f"{purpose}: configured model {configured!r} but the response "
                f"came from {actual!r}")

    def _payload(self, model: str, messages: Sequence[Mapping[str, str]],
                 schema: Mapping[str, Any], purpose: str) -> Dict[str, Any]:
        return {
            "model": model,
            "messages": [dict(m) for m in messages],
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "max_tokens": self.cfg.max_output_tokens,
            "usage": {"include": True},
            "provider": {"allow_fallbacks": False},
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": f"pilot43_{purpose}", "strict": True,
                                "schema": dict(schema)},
            },
        }

    def _backoff(self, attempt: int, retry_after: Optional[str]) -> float:
        if retry_after:
            try:
                return min(float(retry_after), self.cfg.backoff_max_seconds)
            except ValueError:
                pass
        base = self.cfg.backoff_seconds_base * (2 ** attempt)
        return min(base * (1.0 + self._jitter()), self.cfg.backoff_max_seconds)

    def _request(self, purpose: str, model: str, prompt_version: str,
                 messages: Sequence[Mapping[str, str]],
                 schema: Mapping[str, Any], meta: Mapping[str, Any],
                 key: str) -> Dict[str, Any]:
        url = self.cfg.base_url.rstrip("/") + "/chat/completions"
        payload = self._payload(model, messages, schema, purpose)
        headers = {"Authorization": f"Bearer {self._key}",
                   "Content-Type": "application/json",
                   "X-Title": "tool-r0-pilot43"}
        sample_id = str(meta.get("sample_id") or "")
        last_error = "no attempt was made"
        started = time.perf_counter()
        for attempt in range(self.cfg.max_retries + 1):
            try:
                resp = self._transport(url, payload, headers,
                                       self.cfg.request_timeout_seconds)
            except Exception as exc:                      # noqa: BLE001
                last_error = f"transport error: {exc}"
                self._fail(meta, purpose, model, attempt, last_error,
                           prompt_version=prompt_version)
                self._sleep(self._backoff(attempt, None))
                continue

            if resp.status_code in RETRY_STATUS:
                last_error = f"HTTP {resp.status_code}: {resp.text[:300]}"
                self._fail(meta, purpose, model, attempt, last_error,
                           http_status=resp.status_code,
                           prompt_version=prompt_version)
                self._sleep(self._backoff(attempt,
                                          resp.headers.get("retry-after")))
                continue
            if resp.status_code >= 400:
                last_error = f"HTTP {resp.status_code}: {resp.text[:300]}"
                self._fail(meta, purpose, model, attempt, last_error,
                           http_status=resp.status_code,
                           prompt_version=prompt_version)
                raise TransportError(last_error)

            try:
                data = resp.json()
            except json.JSONDecodeError as exc:
                # OpenRouter occasionally returns an empty/truncated body with
                # HTTP 200; treat as a retriable transport failure.
                last_error = (f"invalid JSON body (HTTP {resp.status_code}): "
                              f"{exc}; body={resp.text[:200]!r}")
                self._fail(meta, purpose, model, attempt, last_error,
                           http_status=resp.status_code,
                           prompt_version=prompt_version)
                self._sleep(self._backoff(attempt,
                                          resp.headers.get("retry-after")))
                continue
            actual = str(data.get("model") or "")
            # a substituted model invalidates the whole record, so this is a
            # refusal and not a retry
            self._assert_same_model(model, actual, purpose)
            raw_text = str(
                (data.get("choices") or [{}])[0].get("message", {}).get("content")
                or "")
            usage = dict(data.get("usage") or {})
            cost = _cost_of(usage)
            with self._lock:
                self.totals.requests += 1
                self.totals.prompt_tokens += int(usage.get("prompt_tokens") or 0)
                self.totals.completion_tokens += int(
                    usage.get("completion_tokens") or 0)
                self.totals.cost_usd += cost
                self.task_cost[sample_id] = self.task_cost.get(sample_id,
                                                               0.0) + cost

            try:
                parsed = parse_json_content(raw_text)
                validate_structured(parsed, schema)
            except Exception as exc:                      # noqa: BLE001
                last_error = f"structured output invalid: {exc}"
                self._fail(meta, purpose, model, attempt, last_error,
                           http_status=resp.status_code,
                           prompt_version=prompt_version)
                self._write_usage()
                self._sleep(self._backoff(attempt, None))
                continue

            latency_ms = int((time.perf_counter() - started) * 1000)
            request_id = (resp.headers.get("x-request-id")
                          or str(data.get("id") or ""))
            provider = str(data.get("provider") or "")
            self._cache_write(key, {
                "cache_key": key, "run_id": self.cfg.run_id, "purpose": purpose,
                "configured_model": model, "actual_model": actual,
                "prompt_version": prompt_version, "provider": provider,
                "request_id": request_id, "http_status": resp.status_code,
                "usage": usage, "raw_text": raw_text,
                "raw_response_sha256": sha256_text(raw_text),
                "created": _utc_now(),
            })
            record = self._record(
                purpose=purpose, prompt_version=prompt_version, model=model,
                actual_model=actual, provider=provider, request_id=request_id,
                meta=meta, latency_ms=latency_ms, usage=usage, cost_usd=cost,
                cache_hit=False, http_status=resp.status_code, attempt=attempt,
                cache_key=key, raw_sha=sha256_text(raw_text))
            self._append(self.request_log, record)
            self._write_usage()
            if self.totals.cost_usd >= self.cfg.max_total_cost_usd:
                raise BudgetExceeded(
                    f"total cost cap reached: {self.totals.cost_usd:.6f} USD >= "
                    f"{self.cfg.max_total_cost_usd} USD", scope="total")
            return {"parsed": parsed, "raw_text": raw_text, "cache_hit": False,
                    "raw_response_sha256": record["raw_response_sha256"],
                    "actual_model": actual, "record": record,
                    "attempts": attempt + 1, "cost_usd": cost}

        self._write_usage()
        raise StructuredOutputError(
            f"{purpose}: no valid response after {self.cfg.max_retries + 1} "
            f"attempts ({last_error})")

    def _record(self, *, purpose: str, prompt_version: str, model: str,
                actual_model: str, provider: str, request_id: str,
                meta: Mapping[str, Any], latency_ms: int,
                usage: Mapping[str, Any], cost_usd: float, cache_hit: bool,
                http_status: int, attempt: int, cache_key: str,
                raw_sha: str) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.cfg.run_id,
            "sample_id": str(meta.get("sample_id") or ""),
            "workflow_id": str(meta.get("workflow_id") or ""),
            "semantic_program_id": str(meta.get("semantic_program_id") or ""),
            "purpose": purpose,
            "prompt_version": prompt_version,
            "configured_model": model,
            "actual_model": actual_model,
            "provider": provider,
            "request_id": request_id,
            "latency_ms": latency_ms,
            "usage": dict(usage),
            "cost_usd": round(float(cost_usd), 8),
            "cache_hit": bool(cache_hit),
            "http_status": int(http_status),
            "attempt": int(attempt),
            "cache_key": cache_key,
            "raw_response_sha256": raw_sha,
            "key_fingerprint": key_fingerprint(self._key) if self._key else "",
            "timestamp": _utc_now(),
        }


def _cost_of(usage: Mapping[str, Any]) -> float:
    if usage.get("cost") is not None:
        return float(usage["cost"])
    tokens = int(usage.get("prompt_tokens") or 0) + \
        int(usage.get("completion_tokens") or 0)
    # conservative placeholder so an omitted cost still consumes budget
    return round(tokens * 5e-7, 8)


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
