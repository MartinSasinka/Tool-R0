"""Standalone tool executor for NESTFUL-style trajectories.

This file is a minimal standalone reimplementation inspired by the original
project evaluator (nestful_evaluation/run.py: IBMFunctionRegistry, execute_one,
resolve_variables, _matches_gold). It imports nothing from curricullum/ or
nestful_evaluation/.

Two executor modes:

  full         The IBM/NESTFUL Python helper library (data_v2/executable_functions/)
               is available, so ANY parse- and schema-valid predicted call can be
               genuinely executed. win_rate / solution_equivalent_pass are REAL.

  gold_replay  No IBM helper library found. The executor can only validate calls
               against the tool schema and "replay" exact gold calls by mapping
               them onto known gold observations / the final gold answer. It
               CANNOT compute observations for arbitrary non-gold calls.
               -> strict_gold_trace_pass and full_sequence_accuracy are trustworthy
                  (they only require gold-call reproduction + gold answer),
                  but win_rate / solution_equivalent_pass are LIMITED and must be
                  reported as such.

See README "Executor limitations".
"""
from __future__ import annotations

import contextlib
import importlib.util
import inspect
import json
import os
import re
import signal
import sys
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

# Candidate locations to auto-detect the IBM executable_functions dir. These are
# data-dependency paths (like a dataset path), discovered at runtime; no code is
# imported from them.
_AUTO_DETECT_SUBPATH = os.path.join("data_v2", "executable_functions")
_AUTO_DETECT_ROOTS = [
    "nestful_repo",
    os.path.join("eval", "data", "NESTFUL-main"),
    os.path.join("scripts", ".verify_artifacts", "nestful_repo"),
]
_DEFAULT_REPO_URL = "https://github.com/IBM/NESTFUL.git"


# =====================================================================
#  Argument / variable handling (vendored, simplified)
# =====================================================================

_VAR_REF_RE = re.compile(r"^\$([A-Za-z_][\w]*)(?:\.([A-Za-z_][\w]*))?\$$")
_ARG_NUM_RE = re.compile(r"^arg_?(\d+)$", re.IGNORECASE)
_VAR_INDEX_RE = re.compile(r"^var_?(\d+)$", re.IGNORECASE)
_DATETIME_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?(?:Z|[+-]\d{2}:?\d{2})?$"
)
_NUM_CSV_RE = re.compile(r"^-?\d+(?:\.\d+)?(?:[,\s]+-?\d+(?:\.\d+)?)+$")


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
    name: str, by_label: Dict[str, Any], indexed: List[Any]
) -> Tuple[bool, Any]:
    if name in by_label:
        return True, by_label[name]
    # labels may be stored with a leading '$' (e.g. "$var_1")
    if f"${name}" in by_label:
        return True, by_label[f"${name}"]
    m = _VAR_INDEX_RE.match(name)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(indexed):
            return True, indexed[idx]
    return False, None


def resolve_variables(
    arguments: Dict[str, Any], by_label: Dict[str, Any], indexed: List[Any]
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


def _normalize_arg_value(value: Any) -> Any:
    if isinstance(value, str):
        if _is_variable_ref(value):
            return value
        s = value.strip()
        if s and s[0] in "[{":
            try:
                parsed = json.loads(s)
                if isinstance(parsed, (list, dict)):
                    return _normalize_arg_value(parsed)
            except (json.JSONDecodeError, TypeError):
                pass
        if _NUM_CSV_RE.match(s):
            sep = "," if "," in s else None
            parts = s.split(sep)
            try:
                return [int(p) if p.strip().lstrip("-").isdigit() else float(p) for p in parts]
            except (ValueError, AttributeError):
                return value
        return value
    if isinstance(value, dict):
        return {k: _normalize_arg_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_arg_value(v) for v in value]
    return value


def normalize_arguments(arguments: Dict[str, Any]) -> Dict[str, Any]:
    return {k: _normalize_arg_value(v) for k, v in arguments.items()}


# =====================================================================
#  Gold answer matcher (vendored, simplified)
# =====================================================================

def _safe_str(value: Any) -> str:
    if isinstance(value, int):
        try:
            return str(value)
        except ValueError:
            return f"<bigint>"
    return str(value)


def matches_gold(predicted: Any, gold_answer: Any, *, tol: float = 1e-3) -> bool:
    """Numeric compare with tolerance, otherwise string equality.

    Handles numpy arrays/scalars returned by IBM executable functions — raw ``==``
    on arrays is ambiguous and must not be used in boolean context.
    """
    if predicted is None or gold_answer is None:
        return False

    try:
        import numpy as np
        if isinstance(predicted, (np.ndarray, np.generic)) or isinstance(
            gold_answer, (np.ndarray, np.generic)
        ):
            pa = np.asarray(predicted)
            ga = np.asarray(gold_answer)
            if pa.shape != ga.shape:
                return False
            if np.issubdtype(pa.dtype, np.number) and np.issubdtype(ga.dtype, np.number):
                return bool(np.allclose(pa, ga, rtol=tol, atol=tol, equal_nan=True))
            return bool(np.array_equal(pa, ga))
    except ImportError:
        pass

    if isinstance(predicted, (list, tuple)) and isinstance(gold_answer, (list, tuple)):
        if len(predicted) != len(gold_answer):
            return False
        return all(matches_gold(a, b, tol=tol) for a, b in zip(predicted, gold_answer))

    if isinstance(predicted, int) and isinstance(gold_answer, int):
        return predicted == gold_answer
    pred_n = coerce_numeric(predicted)
    gold_n = coerce_numeric(gold_answer)
    if isinstance(pred_n, bool) or isinstance(gold_n, bool):
        return pred_n == gold_n
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


# =====================================================================
#  Timeout + signature-matched invocation (vendored)
# =====================================================================

class MalformedToolCallError(Exception):
    pass


class _CallTimeout(Exception):
    pass


def _run_with_alarm(timeout: float, fn: Callable[[], Any]) -> Any:
    """Abort fn after `timeout` seconds on platforms with SIGALRM (Linux)."""
    use_alarm = timeout > 0 and hasattr(signal, "SIGALRM")
    if not use_alarm:
        return fn()

    def _handler(signum: int, frame: Any) -> None:
        raise _CallTimeout()

    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        return fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def call_with_signature_match(fn: Callable[..., Any], arg_dict: Dict[str, Any]) -> Any:
    """Bridge NESTFUL's positional arg_0/arg_1 style onto a function signature."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        sig = None
    if sig is None:
        return fn(*values_in_order(arg_dict))

    kinds = (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
        inspect.Parameter.POSITIONAL_ONLY,
    )
    accepts_var_positional = any(
        p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values()
    )
    param_names = [p.name for p in sig.parameters.values() if p.kind in kinds]
    param_set = set(param_names)
    arg_keys = set(arg_dict.keys())

    if arg_keys and arg_keys.issubset(param_set):
        return fn(**arg_dict)
    required = {
        p.name for p in sig.parameters.values()
        if p.default is inspect.Parameter.empty and p.kind in kinds
    }
    if required and required.issubset(arg_keys):
        return fn(**{k: v for k, v in arg_dict.items() if k in param_set})
    values = values_in_order(arg_dict)
    if not accepts_var_positional and len(values) < len(required):
        raise MalformedToolCallError(
            f"{getattr(fn, '__name__', 'function')} needs {len(required)} "
            f"positional arg(s) {sorted(required)}, got {len(values)} "
            f"from keys {sorted(arg_dict.keys())}"
        )
    return fn(*values[: len(param_names)])


# =====================================================================
#  Minimal IBM function registry
# =====================================================================

class IBMFunctionRegistry:
    """Lazy registry mapping NESTFUL function names to Python callables.

    Loads basic_functions.py (preloaded) and lazily imports the rest via
    func_file_map.json under `<funcs_dir>`.
    """

    _BASIC_FILE = "basic_functions.py"
    _MAP_FILE = "func_file_map.json"

    def __init__(self, funcs_dir: str) -> None:
        self._funcs_dir = os.path.abspath(funcs_dir)
        self._map_path = os.path.join(self._funcs_dir, self._MAP_FILE)
        self._basic_path = os.path.join(self._funcs_dir, self._BASIC_FILE)
        self._cache: Dict[str, Callable[..., Any]] = {}
        self._unavailable: set = set()
        self._func_map: Dict[str, str] = {}
        self._lock = threading.RLock()
        self._available = os.path.isfile(self._map_path) and os.path.isfile(self._basic_path)
        if self._available:
            self._preload_basic()
            self._load_map()

    @property
    def available(self) -> bool:
        return self._available

    def get(self, name: str) -> Optional[Callable[..., Any]]:
        if not name:
            return None
        with self._lock:
            if name in self._cache:
                return self._cache[name]
            if name in self._unavailable or not self._available:
                return None
            fn = self._lazy_load(name)
            if fn is None:
                self._unavailable.add(name)
                return None
            self._cache[name] = fn
            return fn

    def _preload_basic(self) -> None:
        module = self._import_file(self._basic_path, "_nestful_min_basic")
        if module is None:
            return
        for attr, val in vars(module).items():
            if attr.startswith("_"):
                continue
            if callable(val) and not inspect.isclass(val):
                self._cache.setdefault(attr, val)

    def _load_map(self) -> None:
        try:
            with open(self._map_path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            if isinstance(raw, dict):
                self._func_map = {str(k): str(v) for k, v in raw.items()}
        except (OSError, json.JSONDecodeError):
            pass

    def _lazy_load(self, name: str) -> Optional[Callable[..., Any]]:
        rel = self._func_map.get(name)
        if not rel:
            return None
        candidate = rel if os.path.isabs(rel) else os.path.join(self._funcs_dir, rel)
        if not os.path.isfile(candidate):
            return None
        mod_name = "_nestful_min_" + os.path.splitext(os.path.basename(candidate))[0]
        module = self._import_file(candidate, mod_name)
        if module is None:
            return None
        fn = getattr(module, name, None)
        return fn if callable(fn) else None

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


def detect_ibm_functions_dir(
    explicit: Optional[str] = None,
    repo_root: Optional[str] = None,
) -> Optional[str]:
    """Find the IBM executable_functions dir. Returns path or None.

    Order: explicit config path -> NESTFUL_REPO_DIR env -> known local roots.
    """
    def _ok(d: Optional[str]) -> bool:
        if not d:
            return False
        return os.path.isfile(os.path.join(d, "func_file_map.json")) and os.path.isfile(
            os.path.join(d, "basic_functions.py")
        )

    repo_root = repo_root or os.getcwd()

    if explicit:
        cand = explicit
        if _ok(cand):
            return os.path.abspath(cand)
        sub = os.path.join(cand, _AUTO_DETECT_SUBPATH)
        if _ok(sub):
            return os.path.abspath(sub)

    env = os.environ.get("NESTFUL_REPO_DIR")
    if env:
        sub = os.path.join(env, _AUTO_DETECT_SUBPATH)
        if _ok(sub):
            return os.path.abspath(sub)
        if _ok(env):
            return os.path.abspath(env)

    for root in _AUTO_DETECT_ROOTS:
        for base in (root, os.path.join(repo_root, root)):
            sub = os.path.join(base, _AUTO_DETECT_SUBPATH)
            if _ok(sub):
                return os.path.abspath(sub)
    return None


# =====================================================================
#  ToolExecutor — per-task, stateful across a trajectory
# =====================================================================

@dataclass
class ExecResult:
    observation: Any
    error: Optional[str]
    name: str
    label: str
    arguments_resolved: Dict[str, Any]


class ToolExecutor:
    """Executes tool calls for a single task, tracking variable scope.

    Construct once per episode; call `execute(call)` for each turn in order.
    """

    def __init__(
        self,
        task: Dict[str, Any],
        registry: Optional[IBMFunctionRegistry] = None,
        mode: str = "auto",
        ibm_call_timeout: float = 30.0,
    ) -> None:
        self.task = task
        self.registry = registry
        self.timeout = ibm_call_timeout
        self._tool_schema = {t.get("name"): t for t in task.get("tools", [])}
        # Determine effective mode.
        if mode == "full":
            self.mode = "full" if (registry and registry.available) else "gold_replay"
        elif mode == "gold_replay":
            self.mode = "gold_replay"
        else:  # auto
            self.mode = "full" if (registry and registry.available) else "gold_replay"
        # Per-episode variable scope.
        self.by_label: Dict[str, Any] = {}
        self.indexed: List[Any] = []
        self._turn = 0
        # Precompute gold scope for gold_replay mode.
        self._gold_calls = task.get("gold_calls", [])

    # --- schema checks -------------------------------------------------
    def tool_exists(self, name: str) -> bool:
        return name in self._tool_schema

    def schema_arg_keys(self, name: str) -> Optional[set]:
        t = self._tool_schema.get(name)
        if not t:
            return None
        props = (t.get("parameters") or {}).get("properties") or {}
        return set(props.keys())

    def schema_valid(self, call: Dict[str, Any]) -> bool:
        """Tool exists and argument keys are a subset of the schema keys."""
        name = call.get("name", "")
        keys = self.schema_arg_keys(name)
        if keys is None:
            return False
        arg_keys = set((call.get("arguments") or {}).keys())
        return arg_keys.issubset(keys)

    # --- execution -----------------------------------------------------
    def execute(self, call: Dict[str, Any]) -> ExecResult:
        """Execute one call against the current scope. Advances variable state."""
        idx = self._turn
        self._turn += 1
        name = (call.get("name") or "").strip()
        label = (call.get("label") or f"var_{idx + 1}").strip() or f"var_{idx + 1}"
        args_raw = call.get("arguments") or {}
        if not isinstance(args_raw, dict):
            return ExecResult(None, "invalid_arguments_type", name, label, {})

        if not self.tool_exists(name):
            return ExecResult(None, f"unknown_tool:{name}", name, label, {})

        resolved, var_err = resolve_variables(args_raw, self.by_label, self.indexed)
        if var_err is not None:
            return ExecResult(None, var_err, name, label, resolved)
        resolved = normalize_arguments(resolved)

        if self.mode == "full":
            obs, err = self._execute_full(name, resolved)
        else:
            obs, err = self._execute_gold_replay(idx, name, args_raw, resolved)

        if err is None:
            self.by_label[label] = obs
            self.indexed.append(obs)
        return ExecResult(obs, err, name, label, resolved)

    def _execute_full(
        self, name: str, resolved: Dict[str, Any]
    ) -> Tuple[Any, Optional[str]]:
        assert self.registry is not None
        try:
            fn = _run_with_alarm(self.timeout, lambda: self.registry.get(name))
        except _CallTimeout:
            return None, f"timeout:import:{name}"
        if fn is None:
            return None, f"unknown_function:{name}"
        try:
            with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(
                devnull
            ), contextlib.redirect_stderr(devnull):
                result = _run_with_alarm(
                    self.timeout, lambda: call_with_signature_match(fn, resolved)
                )
        except _CallTimeout:
            return None, f"timeout:{name}"
        except MalformedToolCallError as exc:
            return None, f"malformed_tool_call:{name}:{exc}"
        except BaseException as exc:
            return None, f"runtime_error:{type(exc).__name__}:{exc}"
        return result, None

    def _execute_gold_replay(
        self,
        idx: int,
        name: str,
        args_raw: Dict[str, Any],
        resolved: Dict[str, Any],
    ) -> Tuple[Any, Optional[str]]:
        """Replay-only execution.

        We have no IBM helpers, so we can only "execute" a call by recognizing it
        as the gold call at this position. If it matches the gold call (name +
        argument keys), we return a synthetic observation. For the LAST gold call
        we surface the gold answer so strict final-answer matching works.

        Any non-gold call cannot be executed here -> returns a sentinel error.
        This is why win_rate / solution_equivalent_pass are LIMITED in this mode.
        """
        if idx >= len(self._gold_calls):
            return None, "gold_replay:no_gold_call_at_position"
        gold = self._gold_calls[idx]
        if (gold.get("name") or "") != name:
            return None, "gold_replay:non_gold_tool_name"
        if set((gold.get("arguments") or {}).keys()) != set(args_raw.keys()):
            return None, "gold_replay:non_gold_argument_keys"
        # Synthetic observation: for the final gold call, expose the gold answer.
        if idx == len(self._gold_calls) - 1:
            return self.task.get("gold_answer"), None
        # Intermediate gold observation is unknown; return an opaque marker that
        # still lets subsequent $varN.result$ references resolve structurally.
        return {"__gold_replay_step__": idx, "name": name}, None
