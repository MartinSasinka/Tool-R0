"""Standalone NESTFUL multi-turn rollouts.

Downloads ``ibm-research/nestful`` from HuggingFace, ``git clone``s
``IBM/NESTFUL`` for the executable helper functions, and runs each task
``--num-rollouts`` times through a batched multi-turn loop on top of
vLLM. Tool calls are dispatched exclusively through the dataset's own
helpers (``data_v2/executable_functions/``); no primitives, no judge.

Outputs (under ``--output-dir``) match the schema in
``eval/results/nestful``:

* ``<profile>_multiturn_predictions.jsonl`` -- one row per rollout
* ``<profile>_multiturn_summary.json``     -- aggregate metrics

Run ``python run.py --help`` for the CLI surface.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import dataclasses
import importlib.util
import inspect
import json
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_REPO_URL = "https://github.com/IBM/NESTFUL.git"
DEFAULT_REPO_DIR = "nestful_repo"
HF_DATASET_ID = "ibm-research/nestful"
DEFAULT_IBM_CALL_TIMEOUT = 30.0
DEFAULT_ADVANCE_LOG_EVERY = 500

TOOL_R0_SYSTEM_PROMPT = (
    "You are a tool-calling assistant. Solve problems by building a chain of tool calls "
    "where each step feeds into the next.\n"
    "Rules:\n"
    "- Always use the provided tools. Do not solve problems mentally.\n"
    "- On every turn before the task is finished, emit exactly one "
    "<tool_call_answer>[...]</tool_call_answer> with a non-empty tool call; "
    "never reply with only <think> or plain text.\n"
    "- Plan the full call chain in <think> before emitting the first call.\n"
    "- Reference previous results as $var1.result$, $var2.result$, ... (1-indexed by call order).\n"
    "- Arguments must be arg_0, arg_1, ... (positional, in order).\n"
    "- After the final tool returns the answer, emit <tool_call_answer>[]</tool_call_answer> to end.\n"
    "Example — What is (4 × 3) ÷ 2?\n"
    "<think>\nStep 1: multiply(arg_0=4, arg_1=3) → 12\n"
    "Step 2: divide(arg_0=$var1.result$, arg_1=2) → 6 = final answer\n</think>\n"
    "<tool_call_answer>[{\"name\": \"multiply\", \"arguments\": {\"arg_0\": 4, \"arg_1\": 3}}]</tool_call_answer>\n"
    "After receiving the result of step 1, continue:\n"
    "<tool_call_answer>[{\"name\": \"divide\", \"arguments\": {\"arg_0\": \"$var1.result$\", \"arg_1\": 2}}]</tool_call_answer>\n"
    "After receiving the final result:\n"
    "<tool_call_answer>[]</tool_call_answer>"
)

"""  TOOL_R0_SYSTEM_PROMPT = (
    "You are a tool-calling assistant. Solve problems by building a chain of tool calls "
    "where each step feeds into the next.\n"
    "Rules:\n"
    "- Always use the provided tools. Do not solve problems mentally.\n"
    "- Plan the full call chain in <think> before emitting the first call.\n"
    "- Reference previous results as $var1.result$, $var2.result$, ... (1-indexed by call order).\n"
    "- Arguments must be arg_0, arg_1, ... (positional, in order).\n"
    "- After the final tool returns the answer, emit <tool_call_answer>[]</tool_call_answer> to end.\n"
    "Example — What is (4 × 3) ÷ 2?\n"
    "<think>\nStep 1: multiply(arg_0=4, arg_1=3) → 12\n"
    "Step 2: divide(arg_0=$var1.result$, arg_1=2) → 6 = final answer\n</think>\n"
    "<tool_call_answer>[{\"name\": \"multiply\", \"arguments\": {\"arg_0\": 4, \"arg_1\": 3}}]</tool_call_answer>\n"
    "After receiving the result of step 1, continue:\n"
    "<tool_call_answer>[{\"name\": \"divide\", \"arguments\": {\"arg_0\": \"$var1.result$\", \"arg_1\": 2}}]</tool_call_answer>\n"
    "After receiving the final result:\n"
    "<tool_call_answer>[]</tool_call_answer>"
)  """


# =====================================================================
#  Tool-call parser (vendored from eval/parse_utils.py)
# =====================================================================

_TAG_PATTERNS = [
    re.compile(r"<tool_call_answer>(.*?)</tool_call_answer>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<function_call>(.*?)</function_call>", re.DOTALL | re.IGNORECASE),
]
_UNCLOSED_TAG_PATTERNS = [
    re.compile(r"<tool_call_answer>\s*(.+)", re.DOTALL | re.IGNORECASE),
    re.compile(r"<tool_call>\s*(.+)", re.DOTALL | re.IGNORECASE),
]
_FENCED_JSON = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_TOOL_CALL_OBJ = re.compile(
    r"\{\s*\"name\"\s*:\s*\"[^\"]+\"\s*,\s*\"arguments\"\s*:\s*\{[^}]*\}\s*\}",
    re.DOTALL,
)


def _loads_relaxed(text: str) -> Optional[Any]:
    if not text or not text.strip():
        return None
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.IGNORECASE).strip()
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError, TypeError):
        pass
    s2 = s.replace("None", "null").replace("True", "true").replace("False", "false")
    try:
        return json.loads(s2)
    except (json.JSONDecodeError, TypeError):
        pass
    return None


def _normalize_call(obj: Any) -> Optional[Dict[str, Any]]:
    if isinstance(obj, list):
        if not obj:
            return None
        obj = obj[0]
    if not isinstance(obj, dict):
        return None
    if "function" in obj and isinstance(obj["function"], dict):
        fn = obj["function"]
        name = fn.get("name")
        args = fn.get("arguments")
        if isinstance(args, str):
            args = _loads_relaxed(args)
        if isinstance(name, str) and isinstance(args, dict):
            return {"name": name, "arguments": args, "label": fn.get("label", "")}
    name = obj.get("name") or obj.get("tool_name") or obj.get("tool")
    if not isinstance(name, str) or not name.strip():
        action = obj.get("Action") or obj.get("action")
        if isinstance(action, str) and action.strip():
            action_input = obj.get("Action_Input") or obj.get("action_input") or obj.get("parameters") or {}
            if isinstance(action_input, str):
                action_input = _loads_relaxed(action_input) or {}
            return {
                "name": action.strip(),
                "arguments": action_input if isinstance(action_input, dict) else {},
                "label": "",
            }
        return None
    args = obj.get("arguments") or obj.get("parameters") or obj.get("args")
    if isinstance(args, str):
        args = _loads_relaxed(args)
    label = obj.get("label", "")
    if isinstance(args, dict):
        return {"name": name.strip(), "arguments": args, "label": label}
    flat = {k: v for k, v in obj.items() if k not in ("name", "tool_name", "tool", "type", "id", "label")}
    return {"name": name.strip(), "arguments": flat, "label": label}


def _parse_call_list(obj: Any) -> Optional[List[Dict[str, Any]]]:
    if isinstance(obj, list):
        out = []
        for item in obj:
            norm = _normalize_call(item)
            if norm:
                out.append(norm)
        return out if out else None
    norm = _normalize_call(obj)
    return [norm] if norm else None


def _extract_json_candidates(text: str) -> List[str]:
    candidates: List[str] = []
    depth = 0
    start = -1
    opener = ""
    for i, ch in enumerate(text):
        if ch in ("[", "{") and depth == 0:
            start = i
            opener = ch
            depth = 1
        elif depth > 0:
            if ch == opener:
                depth += 1
            elif (opener == "[" and ch == "]") or (opener == "{" and ch == "}"):
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : i + 1])
                    start = -1
    return candidates


def _try_close_json(text: str) -> str:
    open_sq = text.count("[") - text.count("]")
    open_cu = text.count("{") - text.count("}")
    suffix = "}" * max(0, open_cu) + "]" * max(0, open_sq)
    return text + suffix


def parse_tool_calls(response: str) -> Tuple[Optional[List[Dict[str, Any]]], str]:
    if not response or not response.strip():
        return None, "empty_output"
    for pattern in _TAG_PATTERNS:
        m = pattern.search(response)
        if m:
            inner = m.group(1).strip()
            obj = _loads_relaxed(inner)
            if obj is not None:
                calls = _parse_call_list(obj)
                if calls:
                    return calls, "tag_closed"
    for pattern in _UNCLOSED_TAG_PATTERNS:
        m = pattern.search(response)
        if m:
            inner = m.group(1).strip()
            inner = _try_close_json(inner)
            obj = _loads_relaxed(inner)
            if obj is not None:
                calls = _parse_call_list(obj)
                if calls:
                    return calls, "tag_unclosed"
    for m in _FENCED_JSON.finditer(response):
        obj = _loads_relaxed(m.group(1))
        if obj is not None:
            calls = _parse_call_list(obj)
            if calls:
                return calls, "fenced_json"
    json_candidates = _extract_json_candidates(response)
    for candidate in reversed(json_candidates):
        obj = _loads_relaxed(candidate)
        if obj is not None:
            calls = _parse_call_list(obj)
            if calls:
                return calls, "json_in_text"
    matches = _TOOL_CALL_OBJ.findall(response)
    if matches:
        calls = []
        for match_str in matches:
            obj = _loads_relaxed(match_str)
            if obj:
                norm = _normalize_call(obj)
                if norm:
                    calls.append(norm)
        if calls:
            return calls, "regex_objects"
    obj = _loads_relaxed(response)
    if obj is not None:
        calls = _parse_call_list(obj)
        if calls:
            return calls, "full_response_json"
    return None, "unparseable"


# =====================================================================
#  Variable resolver + arg coercion (vendored from eval/.../executor.py)
# =====================================================================

_VAR_REF_RE = re.compile(r"^\$([A-Za-z_][\w]*)(?:\.([A-Za-z_][\w]*))?\$$")
_ARG_NUM_RE = re.compile(r"^arg_?(\d+)$", re.IGNORECASE)
_VAR_INDEX_RE = re.compile(r"^var_?(\d+)$", re.IGNORECASE)


@dataclasses.dataclass
class CallTrace:
    index: int
    name: str
    label: str
    arguments_resolved: Dict[str, Any]
    result: Any
    error: Optional[str]
    source: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def coerce_numeric(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return value
        try:
            if "." in s or "e" in s.lower():
                return float(s)
            return int(s)
        except (ValueError, TypeError):
            try:
                return float(s)
            except (ValueError, TypeError):
                return value
    return value


def _arg_sort_key(key: str) -> Tuple[int, Any]:
    m = _ARG_NUM_RE.match(key)
    if m:
        return (0, int(m.group(1)))
    return (1, key)


def values_in_order(arguments: Dict[str, Any]) -> List[Any]:
    if not arguments:
        return []
    keys = sorted(arguments.keys(), key=_arg_sort_key)
    return [arguments[k] for k in keys]


def _is_variable_ref(value: Any) -> bool:
    return isinstance(value, str) and _VAR_REF_RE.match(value.strip()) is not None


def _lookup_variable(
    name: str,
    by_label: Dict[str, Any],
    indexed: List[Any],
) -> Tuple[bool, Any]:
    if name in by_label:
        return True, by_label[name]
    m = _VAR_INDEX_RE.match(name)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(indexed):
            return True, indexed[idx]
    return False, None


def resolve_variables(
    arguments: Dict[str, Any],
    by_label: Dict[str, Any],
    indexed: List[Any],
) -> Tuple[Dict[str, Any], Optional[str]]:
    resolved: Dict[str, Any] = {}
    for key, value in arguments.items():
        if _is_variable_ref(value):
            m = _VAR_REF_RE.match(value.strip())
            assert m is not None
            var_name = m.group(1)
            ok, val = _lookup_variable(var_name, by_label, indexed)
            if not ok:
                return resolved, f"unresolved_variable:{var_name}"
            resolved[key] = val
        else:
            resolved[key] = value
    return resolved, None


_DATETIME_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?(?:Z|[+-]\d{2}:?\d{2})?$"
)


def _try_parse_datetime(s: str) -> Any:
    """Try to parse an ISO-8601-ish datetime string into a datetime object.

    IBM helpers frequently call ``.year``, ``.month`` etc. on date values.
    When the model passes the date as a plain string the attribute lookup
    raises AttributeError.  Converting to datetime lets the helper succeed.
    Returns the original string unchanged on any parse failure.
    """
    import datetime as _dt
    for fmt in (
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
    ):
        try:
            return _dt.datetime.strptime(s.strip().rstrip("Z"), fmt)
        except ValueError:
            pass
    return s


_NUM_CSV_RE = re.compile(r"^-?\d+(?:\.\d+)?(?:[,\s]+-?\d+(?:\.\d+)?)+$")


def _try_parse_num_list(s: str) -> Any:
    """Parse a comma/space-separated number string into a Python list.

    E.g. "1, 2, 3" or "1 2 3" → [1, 2, 3].  Returns the original string if
    parsing fails.  This handles IBM helpers that expect a list[int|float]
    but receive the model's plain-text CSV representation.
    """
    sep = "," if "," in s else None
    parts = s.split(sep)
    try:
        nums: List[Any] = []
        for p in parts:
            p = p.strip()
            nums.append(int(p) if p.lstrip("-").isdigit() else float(p))
        return nums
    except (ValueError, AttributeError):
        return s


def normalize_argument_value(value: Any) -> Any:
    """Coerce JSON-like, CSV-numeric and datetime strings before IBM dispatch."""
    if isinstance(value, str):
        if _is_variable_ref(value):
            return value
        s = value.strip()
        # JSON list or dict  →  real Python object
        if s and s[0] in "[{":
            parsed = _loads_relaxed(s)
            if isinstance(parsed, (list, dict)):
                return normalize_argument_value(parsed)
        # ISO-8601 datetime string  →  datetime object so .year/.month work
        if _DATETIME_RE.match(s):
            return _try_parse_datetime(s)
        # "1, 2, 3" or "1 2 3"  →  [1, 2, 3]
        if _NUM_CSV_RE.match(s):
            return _try_parse_num_list(s)
        return value
    if isinstance(value, dict):
        return {k: normalize_argument_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize_argument_value(v) for v in value]
    return value


def normalize_arguments(arguments: Dict[str, Any]) -> Dict[str, Any]:
    return {k: normalize_argument_value(v) for k, v in arguments.items()}


# =====================================================================
#  IBM function registry (vendored from eval/benchmarks/nestful/ibm_loader.py)
# =====================================================================


class IBMFunctionRegistry:
    """Lazy registry mapping NESTFUL function names to Python callables.

    Looks under ``<repo_root>/data_v2/executable_functions/`` for the
    IBM-shipped helpers. Preloads ``basic_functions.py`` (~40 small
    helpers) and lazily imports the rest via ``func_file_map.json``.
    """

    _FUNC_SUBDIR = os.path.join("data_v2", "executable_functions")
    _BASIC_FILE = "basic_functions.py"
    _MAP_FILE = "func_file_map.json"

    def __init__(self, repo_root: str) -> None:
        self._repo_root = os.path.abspath(repo_root)
        self._funcs_dir = os.path.join(self._repo_root, self._FUNC_SUBDIR)
        self._map_path = os.path.join(self._funcs_dir, self._MAP_FILE)
        self._basic_path = os.path.join(self._funcs_dir, self._BASIC_FILE)
        self._cache: Dict[str, Callable[..., Any]] = {}
        self._unavailable: Set[str] = set()
        self._import_errors: Dict[str, str] = {}
        self._lock = threading.RLock()
        self._func_map: Dict[str, str] = {}
        self._available = os.path.isdir(self._funcs_dir)
        if not self._available:
            return
        self._preload_basic_functions()
        self._load_func_map()

    @property
    def available(self) -> bool:
        return self._available

    def get(self, name: str) -> Optional[Callable[..., Any]]:
        if not name:
            return None
        with self._lock:
            if name in self._cache:
                return self._cache[name]
            if name in self._unavailable:
                return None
            if not self._available:
                self._unavailable.add(name)
                return None
            fn = self._lazy_load(name)
            if fn is None:
                self._unavailable.add(name)
                return None
            self._cache[name] = fn
            return fn

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "available": self._available,
                "repo_root": self._repo_root,
                "cached_imports": len(self._cache),
                "unavailable_funcs": len(self._unavailable),
                "func_map_entries": len(self._func_map),
                "import_error_sample": dict(list(self._import_errors.items())[:5]),
            }

    def _preload_basic_functions(self) -> None:
        if not os.path.isfile(self._basic_path):
            return
        module = self._import_file(self._basic_path, "_nestful_basic_functions")
        if module is None:
            return
        for attr_name, attr_val in vars(module).items():
            if attr_name.startswith("_"):
                continue
            if callable(attr_val) and not inspect.isclass(attr_val):
                self._cache.setdefault(attr_name, attr_val)

    def _load_func_map(self) -> None:
        if not os.path.isfile(self._map_path):
            return
        try:
            with open(self._map_path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            if isinstance(raw, dict):
                self._func_map = {str(k): str(v) for k, v in raw.items()}
        except (OSError, json.JSONDecodeError) as exc:
            self._import_errors["__func_map__"] = f"{type(exc).__name__}: {exc}"

    def _lazy_load(self, name: str) -> Optional[Callable[..., Any]]:
        rel = self._func_map.get(name)
        if not rel:
            return None
        candidate = rel if os.path.isabs(rel) else os.path.join(self._funcs_dir, rel)
        if not os.path.isfile(candidate):
            self._import_errors[name] = f"missing_file:{candidate}"
            return None
        module_name = "_nestful_ibm_" + os.path.splitext(os.path.basename(candidate))[0]
        module = self._import_file(candidate, module_name)
        if module is None:
            return None
        fn = getattr(module, name, None)
        if not callable(fn):
            self._import_errors[name] = f"name_not_in_module:{candidate}"
            return None
        return fn

    @staticmethod
    def _import_file(path: str, module_name: str) -> Optional[Any]:
        if module_name in sys.modules:
            return sys.modules[module_name]
        try:
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                return None
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(
                devnull
            ), contextlib.redirect_stderr(devnull):
                spec.loader.exec_module(module)
            return module
        except BaseException:
            sys.modules.pop(module_name, None)
            return None


class MalformedToolCallError(Exception):
    """Raised when a model's tool call cannot be mapped onto the target signature.

    Distinguishes *model* mistakes (wrong/missing argument keys) from genuine
    *runtime* failures inside the NESTFUL helper, so the summary can separate
    the two error classes.
    """


class _IBMCallTimeout(Exception):
    """Raised when an IBM helper exceeds ``ibm_call_timeout`` seconds."""


def _run_with_alarm(timeout: float, fn: Callable[[], Any]) -> Any:
    """Run ``fn`` in the main thread, aborting after ``timeout`` seconds on Linux."""
    use_alarm = timeout > 0 and hasattr(signal, "SIGALRM")
    if not use_alarm:
        return fn()

    def _handler(signum: int, frame: Any) -> None:
        raise _IBMCallTimeout()

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        return fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def _invoke_ibm_fn(
    fn: Callable[..., Any],
    arguments: Dict[str, Any],
    timeout: float,
) -> Any:
    """Call an IBM helper with stdout/stderr suppressed and an optional timeout."""
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(
        devnull
    ), contextlib.redirect_stderr(devnull):
        return _run_with_alarm(
            timeout,
            lambda: call_with_signature_match(fn, arguments),
        )


def call_with_signature_match(fn: Callable[..., Any], arg_dict: Dict[str, Any]) -> Any:
    """Bridge from NESTFUL's positional ``arg_0/arg_1`` style to named params."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        sig = None
    if sig is not None:
        accepts_var_positional = any(
            p.kind == inspect.Parameter.VAR_POSITIONAL
            for p in sig.parameters.values()
        )
        param_names = [
            p.name
            for p in sig.parameters.values()
            if p.kind
            in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
                inspect.Parameter.POSITIONAL_ONLY,
            )
        ]
        param_set = set(param_names)
        arg_keys = set(arg_dict.keys())
        if arg_keys and arg_keys.issubset(param_set):
            return fn(**arg_dict)
        required = {
            p.name
            for p in sig.parameters.values()
            if p.default is inspect.Parameter.empty
            and p.kind
            in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
                inspect.Parameter.POSITIONAL_ONLY,
            )
        }
        if required and required.issubset(arg_keys):
            return fn(**{k: v for k, v in arg_dict.items() if k in param_set})
        values = values_in_order(arg_dict)
        if not accepts_var_positional and len(values) < len(required):
            raise MalformedToolCallError(
                f"{getattr(fn, '__name__', 'function')} needs "
                f"{len(required)} positional arg(s) {sorted(required)}, "
                f"got {len(values)} from keys {sorted(arg_dict.keys())}"
            )
        return fn(*values[: len(param_names)])
    return fn(*values_in_order(arg_dict))


# =====================================================================
#  Per-call execution
# =====================================================================


def execute_one(
    call: Dict[str, Any],
    by_label: Dict[str, Any],
    indexed: List[Any],
    *,
    index: int = 0,
    ibm_registry: Optional[IBMFunctionRegistry] = None,
    ibm_call_timeout: float = DEFAULT_IBM_CALL_TIMEOUT,
) -> CallTrace:
    """Run a single tool call against the existing variable scope.

    All dispatch goes through ``ibm_registry`` (the IBM/NESTFUL Python
    helpers under ``data_v2/executable_functions/``). Names not found
    there are reported as ``unknown_function`` -- there is no separate
    primitives table, since NESTFUL ships its own ``add`` / ``subtract``
    / ``multiply`` / ``divide`` / etc. via ``basic_functions.py``.
    """
    name = (call.get("name") or "").strip()
    label = (call.get("label") or f"var_{index + 1}").strip() or f"var_{index + 1}"
    arguments_raw = call.get("arguments") or {}
    if not isinstance(arguments_raw, dict):
        return CallTrace(
            index, name, label, {}, None,
            "invalid_arguments_type", "unknown",
        )

    arguments_resolved, var_err = resolve_variables(arguments_raw, by_label, indexed)
    if var_err is not None:
        return CallTrace(
            index, name, label, arguments_resolved, None, var_err, "unknown",
        )
    arguments_resolved = normalize_arguments(arguments_resolved)

    if ibm_registry is None:
        return CallTrace(
            index, name, label, arguments_resolved, None,
            "ibm_registry_unavailable", "unknown",
        )

    try:
        ibm_fn = _run_with_alarm(
            ibm_call_timeout, lambda: ibm_registry.get(name)
        )
    except _IBMCallTimeout:
        return CallTrace(
            index, name, label, arguments_resolved, None,
            f"ibm_timeout:import:{name}:{ibm_call_timeout}s", "ibm",
        )

    if ibm_fn is None:
        return CallTrace(
            index, name, label, arguments_resolved, None,
            f"unknown_function:{name}", "unknown",
        )

    try:
        result = _invoke_ibm_fn(ibm_fn, arguments_resolved, ibm_call_timeout)
    except _IBMCallTimeout:
        return CallTrace(
            index, name, label, arguments_resolved, None,
            f"ibm_timeout:{name}:{ibm_call_timeout}s", "ibm",
        )
    except MalformedToolCallError as exc:
        return CallTrace(
            index, name, label, arguments_resolved, None,
            f"malformed_tool_call:{name}:{exc}", "model",
        )
    except BaseException as exc:
        return CallTrace(
            index, name, label, arguments_resolved, None,
            f"ibm_runtime_error:{type(exc).__name__}:{exc}", "ibm",
        )
    return CallTrace(
        index, name, label, arguments_resolved, result, None, "ibm",
    )


# =====================================================================
#  IBM repo cloner
# =====================================================================


def ensure_ibm_repo(repo_dir: str, repo_url: str = DEFAULT_REPO_URL) -> bool:
    """Clone the IBM/NESTFUL repo into ``repo_dir`` if it isn't there yet.

    Returns True if the executable_functions directory is on disk after
    this call. Failures (no git, no network) are logged and we keep
    going without IBM dispatch.
    """
    sentinel = os.path.join(repo_dir, "data_v2", "executable_functions", "func_file_map.json")
    basic = os.path.join(repo_dir, "data_v2", "executable_functions", "basic_functions.py")
    if os.path.isfile(sentinel) and os.path.isfile(basic):
        print(f"[setup] IBM repo already present at {repo_dir}")
        return True

    if os.path.isdir(repo_dir):
        print(f"[setup] {repo_dir} exists but is incomplete; will not delete automatically.")
        print(f"[setup] If you want a fresh clone, remove the directory and re-run.")
        return os.path.isfile(sentinel) and os.path.isfile(basic)

    print(f"[setup] Cloning {repo_url} into {repo_dir} (depth=1)...")
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, repo_dir],
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"[setup] WARNING: git clone failed ({exc}). Continuing without IBM funcs.")
        return False
    return os.path.isfile(sentinel) and os.path.isfile(basic)


# =====================================================================
#  Dataset loader
# =====================================================================


def load_nestful_tasks(
    max_tasks: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load the NESTFUL benchmark from HuggingFace and normalise it."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "datasets is required (pip install datasets)."
        ) from exc

    print(f"[data] Downloading {HF_DATASET_ID} from HuggingFace...")
    ds = load_dataset(HF_DATASET_ID, split="train", cache_dir=cache_dir)

    tasks: List[Dict[str, Any]] = []
    for idx, row in enumerate(ds):
        if max_tasks is not None and idx >= max_tasks:
            break

        tools_raw = row["tools"]
        if isinstance(tools_raw, str):
            tools_raw = json.loads(tools_raw)
        tools = []
        for t in tools_raw:
            properties = {}
            for param_name, param_spec in (t.get("parameters") or {}).items():
                properties[param_name] = {
                    "type": param_spec.get("type", "string"),
                    "description": param_spec.get("description", ""),
                }
            tools.append({
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": list(properties.keys()),
                },
            })

        output_raw = row["output"]
        if isinstance(output_raw, str):
            output_raw = json.loads(output_raw)
        gold_calls = []
        for step in output_raw:
            gold_calls.append({
                "name": step["name"],
                "arguments": step.get("arguments", {}),
                "label": step.get("label", ""),
            })

        gold_answer_raw = row.get("gold_answer")
        try:
            gold_answer = (
                ast.literal_eval(str(gold_answer_raw))
                if gold_answer_raw is not None
                else None
            )
        except (ValueError, SyntaxError):
            gold_answer = gold_answer_raw

        tasks.append({
            "task_id": row.get("sample_id", f"nestful_{idx}"),
            "question": row["input"],
            "tools": tools,
            "gold_calls": gold_calls,
            "gold_answer": gold_answer,
        })

    print(
        f"[data] Loaded {len(tasks)} tasks "
        f"(avg {sum(len(t['gold_calls']) for t in tasks) / max(1, len(tasks)):.1f} calls/task)"
    )
    return tasks


def build_user_content(task: Dict[str, Any]) -> str:
    tools_json = json.dumps(task["tools"], indent=2, ensure_ascii=False)
    return (
        f"User request:\n{task['question']}\n\n"
        f"Available tools (JSON):\n{tools_json}"
    )


# =====================================================================
#  Multiturn rollout state + final-answer matcher
# =====================================================================

_FINAL_ANSWER_RE = re.compile(
    r"<final_answer>(.*?)</final_answer>", re.DOTALL | re.IGNORECASE
)
# Matches <tool_call_answer>{"result": 180}</tool_call_answer> — the model
# declares a numeric/string final value but skips the name+arguments wrapper.
# Treated identically to <final_answer>: the rollout is done, value is taken
# from the "result" field.  Stops before the multi-step tool-call list variant
# by requiring no "name" key inside the tag.
_RESULT_ONLY_RE = re.compile(
    r"<tool_call_answer>\s*(\{[^{}]*\})\s*</tool_call_answer>", re.DOTALL | re.IGNORECASE
)
_NUMBER_IN_TEXT_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _extract_numeric(text: str) -> Any:
    if not text:
        return None
    matches = _NUMBER_IN_TEXT_RE.findall(text)
    if not matches:
        return text.strip()
    return coerce_numeric(matches[-1])


def _format_tool_response(call: Dict[str, Any], result: Any) -> str:
    payload = {"name": call.get("name", ""), "result": result}
    return f"<tool_response>{_json_dumps(payload)}</tool_response>"


def _call_signature(call: Dict[str, Any]) -> str:
    """Stable fingerprint of a tool call (name + arguments) for loop detection."""
    return json.dumps(
        {"name": call.get("name", ""), "arguments": call.get("arguments", {})},
        sort_keys=True,
        default=str,
        ensure_ascii=False,
    )


def _int_digit_len(n: int) -> int:
    if n == 0:
        return 1
    n = abs(n)
    digits = 0
    while n:
        n //= 10
        digits += 1
    return digits


def _safe_str(value: Any) -> str:
    """str() that survives Python 3.11+ huge-int limits (default 4300 digits)."""
    if isinstance(value, int):
        try:
            return str(value)
        except ValueError:
            return f"<int:{_int_digit_len(value)}_digits>"
    return str(value)


def _sanitize_for_json(value: Any) -> Any:
    """Recursively make nested structures JSON-safe under huge-int str limits."""
    if isinstance(value, int):
        try:
            str(value)
            return value
        except ValueError:
            return f"<int:{_int_digit_len(value)}_digits>"
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, dict):
        return {str(k): _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_for_json(v) for v in value]
    if isinstance(value, (str, bool)) or value is None:
        return value
    return str(value)


def _json_dumps(obj: Any, **kwargs: Any) -> str:
    kwargs.setdefault("ensure_ascii", False)
    return json.dumps(_sanitize_for_json(obj), **kwargs)


def _matches_gold(predicted: Any, gold_answer: Any, *, tol: float = 1e-3) -> bool:
    """Numeric compare with tolerance, otherwise string equality."""
    if predicted is None or gold_answer is None:
        return False
    if isinstance(predicted, int) and isinstance(gold_answer, int):
        return predicted == gold_answer
    pred_n = coerce_numeric(predicted)
    gold_n = coerce_numeric(gold_answer)
    if isinstance(pred_n, bool) or isinstance(gold_n, bool):
        return pred_n == gold_n
    # Exact int compare — avoids float overflow on huge combinatorial answers.
    if isinstance(pred_n, int) and isinstance(gold_n, int):
        return pred_n == gold_n
    if isinstance(pred_n, (int, float)) and isinstance(gold_n, (int, float)):
        try:
            return abs(float(pred_n) - float(gold_n)) < tol
        except (TypeError, ValueError, OverflowError):
            pass
    try:
        return _safe_str(predicted).strip() == _safe_str(gold_answer).strip()
    except ValueError:
        return False


@dataclasses.dataclass
class _RolloutState:
    """Per-(task, rollout_idx) state during the batched multi-turn loop."""

    task: Dict[str, Any]
    rollout_idx: int
    system_prompt: str
    ibm_registry: Optional[IBMFunctionRegistry]
    ibm_call_timeout: float = DEFAULT_IBM_CALL_TIMEOUT

    messages: List[Dict[str, str]] = dataclasses.field(default_factory=list)
    by_label: Dict[str, Any] = dataclasses.field(default_factory=dict)
    indexed: List[Any] = dataclasses.field(default_factory=list)
    traces: List[CallTrace] = dataclasses.field(default_factory=list)
    completions: List[str] = dataclasses.field(default_factory=list)
    seen_call_sigs: Set[str] = dataclasses.field(default_factory=set)
    done: bool = False
    stop_reason: Optional[str] = None
    final_text: str = ""
    final_value: Any = None
    error: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.messages:
            self.messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": build_user_content(self.task)},
            ]

    def advance(self, completion: str) -> None:
        """Apply one model turn to this rollout."""
        self.completions.append(completion)

        m = _FINAL_ANSWER_RE.search(completion)
        if m:
            text = m.group(1).strip()
            self.final_text = text
            self.final_value = coerce_numeric(text)
            self.done = True
            self.stop_reason = "explicit_final"
            self.messages.append({"role": "assistant", "content": completion})
            return

        calls, _ = parse_tool_calls(completion)
        if not calls:
            # Secondary check: <tool_call_answer>{"result": N}</tool_call_answer>
            # — model skipped name+arguments but explicitly declared a final value.
            # Only trigger when there is NO "name" key (otherwise it's a malformed
            # real tool call that already failed parsing above).
            rm = _RESULT_ONLY_RE.search(completion)
            if rm:
                obj = _loads_relaxed(rm.group(1))
                if (
                    isinstance(obj, dict)
                    and "result" in obj
                    and "name" not in obj
                ):
                    raw_val = obj["result"]
                    self.final_text = _safe_str(raw_val)
                    self.final_value = coerce_numeric(raw_val)
                    self.done = True
                    self.stop_reason = "result_tag_final"
                    self.messages.append({"role": "assistant", "content": completion})
                    return

            self.final_text = completion.strip()
            self.final_value = (
                self.indexed[-1] if self.indexed else _extract_numeric(self.final_text)
            )
            self.done = True
            self.stop_reason = "no_more_calls"
            self.messages.append({"role": "assistant", "content": completion})
            return

        call = calls[0]

        # Loop guard: if the model re-emits a call it has already executed in
        # this rollout, re-running it cannot make progress (same inputs ->
        # same result). Treat it as convergence and keep the value we already
        # have, instead of burning turns until the step limit.
        sig = _call_signature(call)
        if sig in self.seen_call_sigs:
            self.final_text = completion.strip()
            if self.indexed:
                self.final_value = self.indexed[-1]
            self.done = True
            self.stop_reason = "converged"
            self.messages.append({"role": "assistant", "content": completion})
            return
        self.seen_call_sigs.add(sig)

        idx = len(self.traces)
        trace = execute_one(
            call,
            self.by_label,
            self.indexed,
            index=idx,
            ibm_registry=self.ibm_registry,
            ibm_call_timeout=self.ibm_call_timeout,
        )
        self.traces.append(trace)

        if trace.error is not None:
            self.done = True
            self.stop_reason = "execution_error"
            self.error = trace.error
            self.final_value = self.indexed[-1] if self.indexed else None
            self.messages.append({"role": "assistant", "content": completion})
            return

        self.by_label[trace.label] = trace.result
        self.indexed.append(trace.result)
        self.final_value = trace.result
        self.messages.append({"role": "assistant", "content": completion})
        self.messages.append({
            "role": "user",
            "content": _format_tool_response(call, trace.result),
        })


# =====================================================================
#  vLLM driver
# =====================================================================


class _VLLMRunner:
    """Thin wrapper around vLLM that exposes a chat-style generate call."""

    def __init__(
        self,
        *,
        model: str,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        max_model_len: int,
        seed: int,
    ) -> None:
        from vllm import LLM
        from transformers import AutoTokenizer

        print(
            f"[vllm] loading {model} (TP={tensor_parallel_size}, "
            f"max_model_len={max_model_len}, gpu_mem_util={gpu_memory_utilization})"
        )
        self.llm = LLM(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            seed=seed,
            enforce_eager=True,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

    def _apply_template(self, messages: List[Dict[str, str]]) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def approx_token_count(self, text: str) -> int:
        return max(1, len(text) // 3)

    def generate(
        self,
        all_messages: List[List[Dict[str, str]]],
        *,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        seeds: Optional[List[int]] = None,
    ) -> List[str]:
        from vllm import SamplingParams

        if not all_messages:
            return []

        prompts = [self._apply_template(m) for m in all_messages]
        sps: List[SamplingParams] = []
        for i in range(len(prompts)):
            sps.append(
                SamplingParams(
                    n=1,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_new_tokens,
                    seed=(seeds[i] if seeds is not None else None),
                )
            )

        outputs = self.llm.generate(prompts, sps, use_tqdm=False)
        return [o.outputs[0].text for o in outputs]


# =====================================================================
#  Orchestration: run_multiturn_rollouts + JSONL writer
# =====================================================================


def _trace_to_dict(t: CallTrace) -> Dict[str, Any]:
    return {
        "index": t.index,
        "name": t.name,
        "label": t.label,
        "arguments_resolved": t.arguments_resolved,
        "result": t.result,
        "error": t.error,
        "source": t.source,
    }


def _classify_execution(state: _RolloutState) -> str:
    """Return one bucket per rollout for ``execution_class_breakdown``.

    All real dispatch goes through the IBM registry, so the buckets are
    a strict subset of the original eval's: ``executed_ok_ibm``,
    ``ibm_runtime_error``, ``unknown_function`` (model emitted a name
    that isn't in the IBM repo), ``unresolved_variable``, or
    ``no_calls_made``.
    """
    err = state.error or ""
    if err.startswith("malformed_tool_call"):
        return "malformed_tool_call"
    if err.startswith("ibm_runtime_error"):
        return "ibm_runtime_error"
    if err.startswith("ibm_registry_unavailable"):
        return "ibm_registry_unavailable"
    if err.startswith("unknown_function"):
        return "unknown_function"
    if err.startswith("unresolved_variable"):
        return "unresolved_variable"
    if state.error:
        return "other_error"
    if state.traces:
        return "executed_ok_ibm"
    return "no_calls_made"


def _serialize_rollout(
    state: _RolloutState,
    *,
    model_name: str,
) -> Dict[str, Any]:
    """Per-rollout JSONL row matching ``eval/results/nestful`` schema."""
    task = state.task
    stopped = state.stop_reason or "step_limit"

    predicted_final = state.final_value
    matches = _matches_gold(predicted_final, task.get("gold_answer"))
    if matches:
        verdict = "pass"
        verdict_reason = "executor_match"
        status = "completed"
        score = 1.0
    else:
        verdict = "fail"
        if predicted_final is None:
            verdict_reason = "no_final_value"
        else:
            verdict_reason = "executor_mismatch"
        status = "failed"
        score = 0.0

    predicted_calls = [
        {"name": tr.name, "arguments": tr.arguments_resolved, "label": tr.label}
        for tr in state.traces
    ]
    execution_trace = [_trace_to_dict(tr) for tr in state.traces]
    raw_completions = [c[:1500] for c in state.completions[:6]]

    error_category = state.error or stopped
    trace_source_counts: Dict[str, int] = {}
    for tr in state.traces:
        trace_source_counts[tr.source] = trace_source_counts.get(tr.source, 0) + 1

    return {
        "task_id": task["task_id"],
        "question": task["question"][:300],
        "status": status,
        "score": score,
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "stopped": stopped,
        "num_steps": len(state.completions),
        "predicted_final": predicted_final,
        "gold_answer": task.get("gold_answer"),
        "predicted_calls": predicted_calls[:20],
        "execution_trace": execution_trace[:20],
        "raw_completions": raw_completions,
        "execution_error": state.error,
        "trace_source_counts": trace_source_counts,
        "num_tool_calls": len(state.traces),
        "error_category": error_category,
        "rollout_idx": state.rollout_idx,
        "model": model_name,
        "tools": task["tools"],
        "gold_calls": task.get("gold_calls", []),
        "messages": state.messages,
    }


def _log_step_progress(
    *,
    step: int,
    max_steps: int,
    states: List[_RolloutState],
    rows: List[Dict[str, Any]],
    t_start: float,
    total_rollouts: int,
) -> Dict[str, Any]:
    """Print a one-line progress snapshot and return it for JSON dumps."""
    from collections import Counter

    done = sum(1 for s in states if s.done)
    active = total_rollouts - done
    elapsed = time.time() - t_start
    rate = done / elapsed if elapsed > 0 and done > 0 else 0.0
    eta_s = (total_rollouts - done) / rate if rate > 0 else 0.0

    written = len(rows)
    completed = sum(1 for r in rows if r.get("status") == "completed")
    failed = sum(1 for r in rows if r.get("status") == "failed")
    acc = completed / written if written else 0.0
    pct_done = 100.0 * done / total_rollouts if total_rollouts else 0.0

    stop_done = Counter(s.stop_reason for s in states if s.done)

    print(
        f"[progress] step {step}/{max_steps} | "
        f"rollouts done {done}/{total_rollouts} ({pct_done:.1f}%) | "
        f"active {active} | saved {written} | "
        f"pass {completed} fail {failed} acc {100 * acc:.1f}% | "
        f"elapsed {elapsed / 60:.1f}m eta {eta_s / 3600:.1f}h",
        flush=True,
    )
    if stop_done:
        print(
            f"[progress]   stop_reasons: {dict(stop_done.most_common(5))}",
            flush=True,
        )

    return {
        "step": step,
        "max_steps": max_steps,
        "rollouts_done": done,
        "rollouts_total": total_rollouts,
        "rollouts_active": active,
        "rollouts_saved": written,
        "passed": completed,
        "failed": failed,
        "accuracy_percent": round(100.0 * acc, 2),
        "elapsed_seconds": round(elapsed, 1),
        "eta_seconds": round(eta_s, 1),
        "rollouts_per_second": round(rate, 4),
        "stop_reason_breakdown": dict(stop_done),
        "updated_at_unix": time.time(),
    }


def run_multiturn_rollouts(
    tasks: List[Dict[str, Any]],
    runner: "_VLLMRunner",
    *,
    num_rollouts: int,
    max_steps: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    max_model_len: int,
    seed: int,
    ibm_registry: Optional[IBMFunctionRegistry],
    output_path: str,
    model_name: str,
    model_profile: Optional[str] = None,
    summary_path: Optional[str] = None,
    ibm_call_timeout: float = DEFAULT_IBM_CALL_TIMEOUT,
    advance_log_every: int = DEFAULT_ADVANCE_LOG_EVERY,
    progress_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run NESTFUL multi-turn rollouts and stream results to disk.

    Writes:
      * predictions JSONL: one line per (task, rollout) with a row schema
        that matches ``eval/results/nestful``;
      * summary JSON: matches the schema of
        ``<profile>_multiturn_summary.json`` from the existing eval.

    Returns the summary dict.
    """

    states: List[_RolloutState] = []
    for t in tasks:
        for r_idx in range(num_rollouts):
            states.append(
                _RolloutState(
                    task=t,
                    rollout_idx=r_idx,
                    system_prompt=TOOL_R0_SYSTEM_PROMPT,
                    ibm_registry=ibm_registry,
                    ibm_call_timeout=ibm_call_timeout,
                )
            )

    safety_margin = max_new_tokens
    written_ids: Set[Tuple[str, int]] = set()
    rows: List[Dict[str, Any]] = []
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    t_start = time.time()
    total_rollouts = len(states)
    print(
        f"[loop] {len(tasks)} tasks × {num_rollouts} rollouts = "
        f"{total_rollouts} total",
        flush=True,
    )
    if progress_path:
        print(f"[loop] live progress file: {progress_path}", flush=True)

    def _flush_done(fh) -> None:
        for s in states:
            key = (s.task["task_id"], s.rollout_idx)
            if s.done and key not in written_ids:
                try:
                    rec = _serialize_rollout(s, model_name=model_name)
                except Exception as exc:
                    print(
                        f"[loop] WARNING: serialize failed for "
                        f"task={key[0]} rollout={key[1]}: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    rec = {
                        "task_id": key[0],
                        "rollout_idx": key[1],
                        "error": f"serialize_error:{type(exc).__name__}:{exc}",
                        "predicted_final": _safe_str(s.final_value),
                        "gold_answer": s.task.get("gold_answer"),
                        "stop_reason": s.stop_reason,
                    }
                written_ids.add(key)
                rows.append(rec)
                fh.write(_json_dumps(rec) + "\n")
                fh.flush()

    with open(output_path, "w", encoding="utf-8") as fh:
        for step in range(max_steps):
            active = [s for s in states if not s.done]
            if not active:
                break

            print(
                f"[loop] step {step + 1}/{max_steps}  "
                f"active={len(active)}  done={len(states) - len(active)}",
                flush=True,
            )

            prompts_messages: List[List[Dict[str, str]]] = []
            kept: List[_RolloutState] = []
            seeds: List[int] = []
            for s in active:
                p = runner._apply_template(s.messages)
                if runner.approx_token_count(p) + safety_margin > max_model_len:
                    s.done = True
                    s.stop_reason = "context_limit"
                    if s.final_value is None and s.indexed:
                        s.final_value = s.indexed[-1]
                    continue
                prompts_messages.append(s.messages)
                kept.append(s)
                seeds.append(seed + 1000 * s.rollout_idx + len(s.completions))

            if not prompts_messages:
                _flush_done(fh)
                continue

            t0 = time.time()
            completions = runner.generate(
                prompts_messages,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                seeds=seeds,
            )
            print(
                f"[loop]   generated {len(completions)} completions in "
                f"{time.time() - t0:.1f}s",
                flush=True,
            )

            n_kept = len(kept)
            print(
                f"[loop]   dispatching tool calls for {n_kept} rollouts...",
                flush=True,
            )
            t_adv = time.time()
            for i, (s, completion) in enumerate(zip(kept, completions)):
                t_one = time.time()
                try:
                    s.advance(completion)
                except Exception as exc:
                    s.done = True
                    s.stop_reason = "advance_error"
                    s.error = f"{type(exc).__name__}:{exc}"
                elapsed_one = time.time() - t_one
                if elapsed_one > 5.0:
                    print(
                        f"[loop]   slow advance {i + 1}/{n_kept} took "
                        f"{elapsed_one:.1f}s  "
                        f"task={s.task.get('task_id')}  rollout={s.rollout_idx}",
                        flush=True,
                    )
                if advance_log_every > 0 and (
                    (i + 1) % advance_log_every == 0 or i + 1 == n_kept
                ):
                    print(
                        f"[loop]   advanced {i + 1}/{n_kept} in "
                        f"{time.time() - t_adv:.1f}s",
                        flush=True,
                    )

            _flush_done(fh)
            snap = _log_step_progress(
                step=step + 1,
                max_steps=max_steps,
                states=states,
                rows=rows,
                t_start=t_start,
                total_rollouts=total_rollouts,
            )
            if progress_path:
                snap["predictions_path"] = output_path
                with open(progress_path, "w", encoding="utf-8") as pf:
                    json.dump(snap, pf, indent=2, ensure_ascii=False)

        # Anything still active after max_steps -> step_limit.
        for s in states:
            if not s.done:
                s.done = True
                s.stop_reason = "step_limit"
                if s.final_value is None and s.indexed:
                    s.final_value = s.indexed[-1]
        _flush_done(fh)

    elapsed = time.time() - t_start
    summary = _build_summary(
        rows, states,
        model_name=model_name,
        model_profile=model_profile or _slug_from_model(model_name),
        num_rollouts=num_rollouts,
        num_unique_tasks=len(tasks),
        max_steps=max_steps,
        elapsed=elapsed,
        ibm_registry=ibm_registry,
    )

    if summary_path:
        os.makedirs(os.path.dirname(os.path.abspath(summary_path)) or ".",
                    exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(
                _sanitize_for_json(summary),
                fh,
                indent=2,
                ensure_ascii=False,
            )

    return summary


def _slug_from_model(model: str) -> str:
    """Derive a filesystem-safe profile name from a HuggingFace model id."""
    s = model.replace("/", "__").replace(":", "_").replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).lower() or "model"


def _build_summary(
    rows: List[Dict[str, Any]],
    states: List[_RolloutState],
    *,
    model_name: str,
    model_profile: str,
    num_rollouts: int,
    num_unique_tasks: int,
    max_steps: int,
    elapsed: float,
    ibm_registry: Optional[IBMFunctionRegistry],
) -> Dict[str, Any]:
    from collections import Counter

    total = len(rows)
    completed = sum(1 for r in rows if r["status"] == "completed")
    failed = sum(1 for r in rows if r["status"] == "failed")
    errors = sum(1 for r in rows if r["status"] not in ("completed", "failed"))
    scores = [r["score"] for r in rows]
    mean_score = sum(scores) / total if total else 0.0

    error_cats = Counter(
        r["error_category"]
        for r in rows
        if r["status"] in ("failed", "error") and r.get("error_category")
    )

    avg_tool_calls = (
        sum(r["num_tool_calls"] for r in rows) / total if total else 0.0
    )
    avg_steps = sum(r["num_steps"] for r in rows) / total if total else 0.0

    stop_counter: Counter = Counter(r["stopped"] for r in rows)
    exec_class_counter: Counter = Counter()
    for s in states:
        exec_class_counter[_classify_execution(s)] += 1

    step_limit_rate = stop_counter.get("step_limit", 0) / total if total else 0.0
    context_limit_rate = (
        stop_counter.get("context_limit", 0) / total if total else 0.0
    )

    ibm_stats = (
        ibm_registry.stats() if ibm_registry is not None
        else {"available": False, "cached_imports": 0, "unavailable_funcs": 0}
    )

    summary: Dict[str, Any] = {
        "benchmark": "nestful",
        "model_profile": model_profile,
        "model": model_name,
        "total_tasks": total,
        "completed": completed,
        "failed": failed,
        "errors": errors,
        "mean_score": round(mean_score, 4),
        "mean_score_percent": round(100.0 * mean_score, 2),
        "error_categories": dict(error_cats),
        "avg_tool_calls": round(avg_tool_calls, 2),
        "avg_steps": round(avg_steps, 2),
        "mode": "multiturn",
        "final_answer_accuracy": round(mean_score, 4),
        "final_answer_accuracy_percent": round(100.0 * mean_score, 2),
        "passed": completed,
        "step_limit_hit_rate_percent": round(100.0 * step_limit_rate, 2),
        "context_limit_hit_rate_percent": round(100.0 * context_limit_rate, 2),
        "result_tag_final_count": stop_counter.get("result_tag_final", 0),
        "stop_reason_breakdown": dict(stop_counter),
        "execution_class_breakdown": dict(exec_class_counter),
        "ibm_registry_stats": ibm_stats,
        "elapsed_seconds": round(elapsed, 2),
        "max_steps_setting": max_steps,
        "num_unique_tasks": num_unique_tasks,
        "num_rollouts_per_task": num_rollouts,
        "total_rollouts": total,
    }
    return summary


# =====================================================================
#  CLI + main
# =====================================================================


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="NESTFUL multi-turn rollouts (standalone driver).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="HF model id or local path.")
    p.add_argument("--num-rollouts", type=int, default=8,
                   help="Independent rollouts per task.")
    p.add_argument("--max-tasks", type=int, default=None,
                   help="Limit task count (useful for pilots; default: all 1861).")
    p.add_argument("--max-steps", type=int, default=10,
                   help="Max model turns per rollout.")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-new-tokens", type=int, default=2048,
                   help="Max tokens generated per model turn.")
    p.add_argument("--max-model-len", type=int, default=12288,
                   help="Max combined prompt+completion length (vLLM context window).")
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--output-dir", default="nestful_results",
                   help="Directory where predictions JSONL + summary JSON are written.")
    p.add_argument("--model-profile", default=None,
                   help=("Profile slug for output filenames "
                         "(default: derived from --model)."))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--nestful-repo-dir", default=DEFAULT_REPO_DIR,
                   help="Where to clone the IBM/NESTFUL repo for executable functions.")
    p.add_argument("--cache-dir", default=None,
                   help="HuggingFace datasets cache directory.")
    p.add_argument(
        "--ibm-call-timeout", type=float, default=DEFAULT_IBM_CALL_TIMEOUT,
        help="Seconds before an IBM helper call is aborted (guards input()/loops).",
    )
    p.add_argument(
        "--advance-log-every", type=int, default=DEFAULT_ADVANCE_LOG_EVERY,
        help="Log progress every N rollouts during tool dispatch (0=disable).",
    )
    return p


def main() -> int:
    args = build_argparser().parse_args()
    profile = args.model_profile or _slug_from_model(args.model)
    os.makedirs(args.output_dir, exist_ok=True)
    pred_path = os.path.join(
        args.output_dir, f"{profile}_multiturn_predictions.jsonl"
    )
    summary_path = os.path.join(
        args.output_dir, f"{profile}_multiturn_summary.json"
    )

    print(f"[main] model={args.model}  profile={profile}")
    print(
        f"[main] num_rollouts={args.num_rollouts}  "
        f"max_tasks={args.max_tasks}  max_steps={args.max_steps}"
    )
    print(
        f"[main] sampling: temperature={args.temperature} top_p={args.top_p} "
        f"max_new_tokens={args.max_new_tokens}"
    )
    print(
        f"[main] vllm: TP={args.tensor_parallel_size} "
        f"max_model_len={args.max_model_len}"
    )
    print(f"[main] predictions: {pred_path}")
    print(f"[main] summary:     {summary_path}")

    ok = ensure_ibm_repo(args.nestful_repo_dir)
    if not ok:
        print(
            "[main] ERROR: IBM/NESTFUL repo could not be obtained at "
            f"{args.nestful_repo_dir}. Tool dispatch is impossible without it."
        )
        return 2
    try:
        ibm_registry = IBMFunctionRegistry(args.nestful_repo_dir)
    except Exception as exc:
        print(f"[main] ERROR: failed to init IBM registry: {exc}")
        return 2
    print(f"[main] IBM registry: {ibm_registry.stats()}")

    tasks = load_nestful_tasks(max_tasks=args.max_tasks, cache_dir=args.cache_dir)
    if not tasks:
        print("[main] ERROR: no tasks loaded. Check HF dataset access.")
        return 2
    n_rollouts = len(tasks) * args.num_rollouts
    print(
        f"[main] loaded {len(tasks)} tasks → {n_rollouts} rollouts "
        f"({args.max_steps} steps max each)",
        flush=True,
    )

    progress_path = pred_path.replace(
        "_multiturn_predictions.jsonl", "_multiturn_progress.json"
    )
    print(f"[main] progress:    {progress_path}", flush=True)

    t0 = time.time()
    runner = _VLLMRunner(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        seed=args.seed,
    )
    print(f"[main] vLLM ready in {time.time() - t0:.1f}s")

    t1 = time.time()
    summary = run_multiturn_rollouts(
        tasks,
        runner,
        num_rollouts=args.num_rollouts,
        max_steps=args.max_steps,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        max_model_len=args.max_model_len,
        seed=args.seed,
        ibm_registry=ibm_registry,
        output_path=pred_path,
        model_name=args.model,
        model_profile=profile,
        summary_path=summary_path,
        ibm_call_timeout=args.ibm_call_timeout,
        advance_log_every=args.advance_log_every,
        progress_path=progress_path,
    )
    elapsed = time.time() - t1

    print()
    print("=" * 60)
    print(f"  DONE in {elapsed:.1f}s")
    print(f"  Total rollouts:        {summary['total_rollouts']}")
    print(f"  Unique tasks:          {summary['num_unique_tasks']}")
    print(
        f"  Passed (executor):     "
        f"{summary['passed']} ({summary['final_answer_accuracy_percent']}%)"
    )
    print(f"  Failed:                {summary['failed']}")
    print(f"  Avg steps:             {summary['avg_steps']}")
    print(f"  Avg tool calls:        {summary['avg_tool_calls']}")
    print(
        f"  Step-limit hit rate:   "
        f"{summary['step_limit_hit_rate_percent']}%"
    )
    print(
        f"  Context-limit hit rate:{summary['context_limit_hit_rate_percent']}%"
    )
    print(f"  Stop reasons:          {summary['stop_reason_breakdown']}")
    print(f"  Execution classes:     {summary['execution_class_breakdown']}")
    print(f"  Predictions:           {pred_path}")
    print(f"  Summary:               {summary_path}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
