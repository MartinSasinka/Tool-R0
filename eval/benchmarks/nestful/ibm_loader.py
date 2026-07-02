"""
IBM NESTFUL function registry.

Loads the executable Python implementations shipped by IBM with the
NESTFUL benchmark (`data_v2/executable_functions/`) and exposes them as
ordinary Python callables. Replaces the LLM judge fallback for the
27% of tasks whose tool calls were previously routed to GPT for
verdict — we now actually run the function.

Layout assumed under ``repo_root``::

    data_v2/
      executable_functions/
        basic_functions.py     # ~40 stable math/geometry funcs
        func_file_map.json     # { "<name>": "py_code_file_<n>.py", ... }
        py_code_file_1.py
        ...
        py_code_file_4350.py

Resolution order in :py:meth:`IBMFunctionRegistry.get`:

1. In-process cache (``self._cache``).
2. Functions preloaded from ``basic_functions.py``.
3. Lazy ``importlib`` of the file pointed to by ``func_file_map.json``.
4. ``None`` and the name is added to ``self._unavailable`` so we don't
   pay the import cost twice for a missing/broken function.

All import / attribute / syntax errors during lazy loading are
swallowed and recorded; the runner is responsible for translating a
``None`` lookup into an ``ibm_unavailable`` execution class.

Reference: https://github.com/IBM/NESTFUL/tree/main/data_v2/executable_functions
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import os
import sys
import threading
from typing import Any, Callable, Dict, List, Optional, Set


_DEFAULT_REPO_ROOT = os.environ.get("NESTFUL_REPO_DIR", "data/nestful_repo")
_FUNC_SUBDIR = os.path.join("data_v2", "executable_functions")
_BASIC_FUNCTIONS_FILE = "basic_functions.py"
_FUNC_MAP_FILE = "func_file_map.json"


class IBMFunctionRegistry:
    """Lazy registry mapping NESTFUL function names to Python callables.

    Thread-safe (uses an internal lock around the importlib cache so the
    multiturn batched loop can safely look up the same function from
    several worker threads).

    Parameters
    ----------
    repo_root:
        Path to the cloned IBM/NESTFUL repository. Defaults to
        ``data/nestful_repo`` (or ``$NESTFUL_REPO_DIR`` if set).
    require_funcs_dir:
        If True (default), raises :class:`FileNotFoundError` when the
        executable_functions folder is missing — the runner can catch
        this to fall back to "no IBM funcs" mode without crashing.
    """

    def __init__(
        self,
        repo_root: Optional[str] = None,
        *,
        require_funcs_dir: bool = True,
    ) -> None:
        self._repo_root = os.path.abspath(repo_root or _DEFAULT_REPO_ROOT)
        self._funcs_dir = os.path.join(self._repo_root, _FUNC_SUBDIR)
        self._map_path = os.path.join(self._funcs_dir, _FUNC_MAP_FILE)
        self._basic_path = os.path.join(self._funcs_dir, _BASIC_FUNCTIONS_FILE)

        self._cache: Dict[str, Callable[..., Any]] = {}
        self._unavailable: Set[str] = set()
        self._import_errors: Dict[str, str] = {}
        self._lock = threading.RLock()
        self._func_map: Dict[str, str] = {}

        self._available = os.path.isdir(self._funcs_dir)
        if not self._available:
            if require_funcs_dir:
                raise FileNotFoundError(
                    f"IBM NESTFUL functions dir not found at {self._funcs_dir}. "
                    "Run scripts/setup_nestful_funcs.sh to clone the repo, or set "
                    "NESTFUL_REPO_DIR to point at an existing checkout."
                )
            return

        self._preload_basic_functions()
        self._load_func_map()

    # ---- public API --------------------------------------------------

    @property
    def available(self) -> bool:
        """True when the IBM repo is on disk and basic functions loaded."""
        return self._available

    def get(self, name: str) -> Optional[Callable[..., Any]]:
        """Return a callable for *name* or ``None`` if unavailable.

        Subsequent lookups for the same unavailable name are O(1) — we
        don't retry imports that have already failed in this process.
        """
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
        """Snapshot of registry state for the run summary."""
        with self._lock:
            return {
                "available": self._available,
                "repo_root": self._repo_root,
                "cached_imports": len(self._cache),
                "unavailable_funcs": len(self._unavailable),
                "func_map_entries": len(self._func_map),
                "import_error_sample": dict(list(self._import_errors.items())[:5]),
            }

    # ---- internals ---------------------------------------------------

    def _preload_basic_functions(self) -> None:
        """Import basic_functions.py once and harvest top-level callables."""
        if not os.path.isfile(self._basic_path):
            return
        module = self._import_file(self._basic_path, module_name="_nestful_basic_functions")
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
            with open(self._map_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                self._func_map = {str(k): str(v) for k, v in raw.items()}
        except (OSError, json.JSONDecodeError) as exc:
            self._import_errors["__func_map__"] = f"{type(exc).__name__}: {exc}"

    def _lazy_load(self, name: str) -> Optional[Callable[..., Any]]:
        rel_path = self._func_map.get(name)
        if not rel_path:
            return None
        # Some IBM rows give a bare filename, others a relative path under
        # executable_functions/. Accept both.
        candidate = (
            rel_path
            if os.path.isabs(rel_path)
            else os.path.join(self._funcs_dir, rel_path)
        )
        if not os.path.isfile(candidate):
            self._import_errors[name] = f"missing_file:{candidate}"
            return None

        module_name = f"_nestful_ibm_{os.path.splitext(os.path.basename(candidate))[0]}"
        module = self._import_file(candidate, module_name=module_name)
        if module is None:
            return None
        fn = getattr(module, name, None)
        if not callable(fn):
            self._import_errors[name] = f"name_not_in_module:{candidate}"
            return None
        return fn

    def _import_file(self, path: str, *, module_name: str) -> Optional[Any]:
        """``importlib`` wrapper that records all failures instead of raising."""
        if module_name in sys.modules:
            return sys.modules[module_name]
        try:
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                self._import_errors[module_name] = "spec_creation_failed"
                return None
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module
        except BaseException as exc:  # IBM files occasionally raise SystemExit on import
            self._import_errors[module_name] = f"{type(exc).__name__}: {exc}"
            sys.modules.pop(module_name, None)
            return None


# ---------------------------------------------------------------------------
# Smart calling helper — used by executor.py to bridge
# NESTFUL's positional-style {"arg_0": ..., "arg_1": ...} convention to
# IBM functions whose parameters are typically named (radius, height, n, ...)
# ---------------------------------------------------------------------------


def call_with_signature_match(fn: Callable[..., Any], arg_dict: Dict[str, Any]) -> Any:
    """Invoke *fn* with values from *arg_dict*, matching its signature.

    Resolution order:

    1. If every key in *arg_dict* is also a parameter name on *fn*,
       call as kwargs (covers ``{"radius": 5}`` against
       ``def circle_area(radius)``).
    2. If every required parameter on *fn* has the same name as a key in
       *arg_dict*, call as kwargs subset (extra keys ignored).
    3. Otherwise sort *arg_dict* values by NESTFUL's positional order
       (``arg_0``, ``arg_1``, ..., then alphabetic) and call positionally.
    """
    sig = _safe_signature(fn)
    if sig is not None:
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

        values = _values_in_order(arg_dict)
        return fn(*values[: len(param_names)])

    # Couldn't introspect (builtins / C extensions): try positional fallback.
    return fn(*_values_in_order(arg_dict))


def _safe_signature(fn: Callable[..., Any]) -> Optional[inspect.Signature]:
    try:
        return inspect.signature(fn)
    except (TypeError, ValueError):
        return None


def _values_in_order(arg_dict: Dict[str, Any]) -> List[Any]:
    """Sort args by NESTFUL positional convention (arg_0, arg_1, ...)."""
    import re as _re

    pat = _re.compile(r"^arg_?(\d+)$", _re.IGNORECASE)

    def key(k: str):
        m = pat.match(k)
        if m:
            return (0, int(m.group(1)), "")
        return (1, 0, k)

    return [arg_dict[k] for k in sorted(arg_dict.keys(), key=key)]


__all__ = [
    "IBMFunctionRegistry",
    "call_with_signature_match",
]
