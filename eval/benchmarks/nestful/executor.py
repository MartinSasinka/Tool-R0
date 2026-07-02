"""
NESTFUL local execution engine.

Executes a sequence of (typically mathematical) tool calls produced by the
model and resolves NESTFUL-style variable references like ``$var_N.result$``
or ``$<label>.result$`` so the *final* numeric/string result can be compared
with ``gold_answer`` from the dataset.

This is a SAFE dispatcher — no ``eval()`` or ``exec()`` is used. Functions
the executor doesn't recognise are reported via :class:`ExecutionResult`
so the runner can fall back to an LLM judge.

Coverage rationale
------------------
The vast majority of NESTFUL tasks use a small math + numerical-utility
primitive set (``add``, ``subtract``, ``multiply``, ``divide``, ``power``,
``sqrt``, ``abs``, ``modulo``, ``round``, ``floor``, ``ceil``, ``mean``,
``min``, ``max``, ``sum``, ``factorial``, ``log``, ``percentage``).
Anything outside this set is dispatched to the IBM/NESTFUL function
registry (see :mod:`eval.benchmarks.nestful.ibm_loader`), which exposes
the dataset's own Python implementations under
``data_v2/executable_functions/``. Only when *both* layers miss does the
runner fall back to the LLM judge (and only if the user opted in via
``--nestful-use-judge``).

Argument convention
-------------------
NESTFUL tools accept positional-style keyword args (``arg_0``, ``arg_1``,
...). When dispatching we sort args by key (numeric suffix first, then
alphabetic) and pass values positionally to the primitive. This handles
both ``{"arg_0": 1, "arg_1": 2}`` and ``{"a": 1, "b": 2}`` correctly.

Variable references
-------------------
NESTFUL chains calls via ``$<name>.result$`` placeholders. ``<name>`` is
either an explicit ``label`` from the predicted call or ``var_N`` (1-based
positional). The resolver tries label first, then index. Bare ``$var_N$``
without ``.result`` is also accepted.
"""

from __future__ import annotations

import dataclasses
import math
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .ibm_loader import IBMFunctionRegistry


_VAR_REF_RE = re.compile(r"^\$([A-Za-z_][\w]*)(?:\.([A-Za-z_][\w]*))?\$$")
_ARG_NUM_RE = re.compile(r"^arg_?(\d+)$", re.IGNORECASE)
_VAR_INDEX_RE = re.compile(r"^var_?(\d+)$", re.IGNORECASE)


@dataclasses.dataclass
class CallTrace:
    """Per-call execution outcome inside a sequence.

    ``source`` records *who* produced the result so the runner can split
    summary buckets between ``executed_ok_primitive`` (our math sandbox)
    and ``executed_ok_ibm`` (a function from IBM/NESTFUL). It is also
    set on failure paths to preserve the routing trail (e.g. an
    ``ibm_runtime_error`` always has ``source="ibm"``).
    """

    index: int
    name: str
    label: str
    arguments_resolved: Dict[str, Any]
    result: Any
    error: Optional[str]
    source: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class ExecutionResult:
    """Structured outcome of executing a NESTFUL call sequence."""

    success: bool
    final_value: Any
    per_call: List[CallTrace]
    error: Optional[str]
    error_call_index: Optional[int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "final_value": self.final_value,
            "per_call": [c.to_dict() for c in self.per_call],
            "error": self.error,
            "error_call_index": self.error_call_index,
        }


# ---------------------------------------------------------------------------
# Argument coercion helpers
# ---------------------------------------------------------------------------


def coerce_numeric(value: Any) -> Any:
    """Best-effort numeric cast: ``"42"`` -> 42, ``"3.14"`` -> 3.14.

    Returns the original value unchanged when it doesn't look like a number;
    primitives can still operate on strings if they choose to.
    """
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
    """Sort positional-looking keys (``arg_0``, ``arg_1``) numerically."""
    m = _ARG_NUM_RE.match(key)
    if m:
        return (0, int(m.group(1)))
    return (1, key)


def values_in_order(arguments: Dict[str, Any]) -> List[Any]:
    """Return argument values sorted by key in NESTFUL positional order."""
    if not arguments:
        return []
    keys = sorted(arguments.keys(), key=_arg_sort_key)
    return [arguments[k] for k in keys]


# ---------------------------------------------------------------------------
# Variable resolution
# ---------------------------------------------------------------------------


def _is_variable_ref(value: Any) -> bool:
    return isinstance(value, str) and _VAR_REF_RE.match(value.strip()) is not None


def _lookup_variable(
    name: str,
    by_label: Dict[str, Any],
    indexed: List[Any],
) -> Tuple[bool, Any]:
    """Resolve a single ``var`` token to a previously-computed result."""
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
    """Resolve every ``$var.result$`` reference in arguments.

    Returns ``(resolved_args, error_message)``. The error is non-None when a
    referenced variable cannot be found (the runner reports this and may
    delegate to the LLM judge).
    """
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


# ---------------------------------------------------------------------------
# Math primitives — pure Python, no eval/exec.
# ---------------------------------------------------------------------------


def _require(args: List[Any], n: int, name: str) -> List[Any]:
    if len(args) < n:
        raise ValueError(f"{name} expected {n} args, got {len(args)}")
    return [coerce_numeric(a) for a in args[:n]]


def _add(args: List[Any], **_: Any) -> Any:
    if not args:
        raise ValueError("add requires at least 1 argument")
    nums = [coerce_numeric(a) for a in args]
    return sum(nums)


def _subtract(args: List[Any], **_: Any) -> Any:
    a, b = _require(args, 2, "subtract")
    return a - b


def _multiply(args: List[Any], **_: Any) -> Any:
    if not args:
        raise ValueError("multiply requires at least 1 argument")
    nums = [coerce_numeric(a) for a in args]
    out: Any = 1
    for n in nums:
        out = out * n
    return out


def _divide(args: List[Any], **_: Any) -> Any:
    a, b = _require(args, 2, "divide")
    if b == 0:
        raise ZeroDivisionError("division by zero")
    return a / b


def _floor_divide(args: List[Any], **_: Any) -> Any:
    a, b = _require(args, 2, "floor_divide")
    if b == 0:
        raise ZeroDivisionError("floor division by zero")
    return a // b


def _modulo(args: List[Any], **_: Any) -> Any:
    a, b = _require(args, 2, "modulo")
    if b == 0:
        raise ZeroDivisionError("modulo by zero")
    return a % b


def _power(args: List[Any], **_: Any) -> Any:
    a, b = _require(args, 2, "power")
    return a ** b


def _sqrt(args: List[Any], **_: Any) -> Any:
    (a,) = _require(args, 1, "sqrt")
    if a < 0:
        raise ValueError("sqrt of negative number")
    return math.sqrt(a)


def _abs(args: List[Any], **_: Any) -> Any:
    (a,) = _require(args, 1, "abs")
    return abs(a)


def _negate(args: List[Any], **_: Any) -> Any:
    (a,) = _require(args, 1, "negate")
    return -a


def _round(args: List[Any], **_: Any) -> Any:
    if len(args) == 1:
        (a,) = _require(args, 1, "round")
        return round(a)
    a, ndigits = _require(args, 2, "round")
    return round(a, int(ndigits))


def _floor(args: List[Any], **_: Any) -> Any:
    (a,) = _require(args, 1, "floor")
    return math.floor(a)


def _ceil(args: List[Any], **_: Any) -> Any:
    (a,) = _require(args, 1, "ceil")
    return math.ceil(a)


def _mean(args: List[Any], **_: Any) -> Any:
    nums = [coerce_numeric(a) for a in args]
    if not nums:
        raise ValueError("mean of empty input")
    if len(nums) == 1 and isinstance(nums[0], (list, tuple)):
        nums = [coerce_numeric(x) for x in nums[0]]
        if not nums:
            raise ValueError("mean of empty input")
    return sum(nums) / len(nums)


def _min(args: List[Any], **_: Any) -> Any:
    if not args:
        raise ValueError("min requires at least 1 argument")
    nums = [coerce_numeric(a) for a in args]
    if len(nums) == 1 and isinstance(nums[0], (list, tuple)):
        nums = [coerce_numeric(x) for x in nums[0]]
    return min(nums)


def _max(args: List[Any], **_: Any) -> Any:
    if not args:
        raise ValueError("max requires at least 1 argument")
    nums = [coerce_numeric(a) for a in args]
    if len(nums) == 1 and isinstance(nums[0], (list, tuple)):
        nums = [coerce_numeric(x) for x in nums[0]]
    return max(nums)


def _sum(args: List[Any], **_: Any) -> Any:
    nums = [coerce_numeric(a) for a in args]
    if len(nums) == 1 and isinstance(nums[0], (list, tuple)):
        nums = [coerce_numeric(x) for x in nums[0]]
    return sum(nums)


def _factorial(args: List[Any], **_: Any) -> Any:
    (a,) = _require(args, 1, "factorial")
    n = int(a)
    if n < 0:
        raise ValueError("factorial of negative number")
    return math.factorial(n)


def _log(args: List[Any], **_: Any) -> Any:
    if len(args) == 1:
        (a,) = _require(args, 1, "log")
        return math.log(a)
    a, base = _require(args, 2, "log")
    return math.log(a, base)


def _log10(args: List[Any], **_: Any) -> Any:
    (a,) = _require(args, 1, "log10")
    return math.log10(a)


def _exp(args: List[Any], **_: Any) -> Any:
    (a,) = _require(args, 1, "exp")
    return math.exp(a)


def _percentage(args: List[Any], **_: Any) -> Any:
    """``percentage(part, whole)`` → 100 * part / whole."""
    a, b = _require(args, 2, "percentage")
    if b == 0:
        raise ZeroDivisionError("percentage of zero whole")
    return 100.0 * a / b


def _percent_of(args: List[Any], **_: Any) -> Any:
    """``percent_of(percent, whole)`` → percent/100 * whole."""
    a, b = _require(args, 2, "percent_of")
    return (a / 100.0) * b


def _gcd(args: List[Any], **_: Any) -> Any:
    a, b = _require(args, 2, "gcd")
    return math.gcd(int(a), int(b))


def _lcm(args: List[Any], **_: Any) -> Any:
    a, b = _require(args, 2, "lcm")
    a, b = int(a), int(b)
    if a == 0 or b == 0:
        return 0
    return abs(a * b) // math.gcd(a, b)


PRIMITIVES: Dict[str, Callable[..., Any]] = {
    "add": _add,
    "sum": _sum,
    "plus": _add,
    "subtract": _subtract,
    "sub": _subtract,
    "minus": _subtract,
    "difference": _subtract,
    "multiply": _multiply,
    "mul": _multiply,
    "mult": _multiply,
    "product": _multiply,
    "divide": _divide,
    "div": _divide,
    "quotient": _divide,
    "floor_divide": _floor_divide,
    "floor_div": _floor_divide,
    "modulo": _modulo,
    "mod": _modulo,
    "remainder": _modulo,
    "power": _power,
    "pow": _power,
    "exponent": _power,
    "sqrt": _sqrt,
    "square_root": _sqrt,
    "abs": _abs,
    "absolute": _abs,
    "negate": _negate,
    "neg": _negate,
    "round": _round,
    "floor": _floor,
    "ceil": _ceil,
    "ceiling": _ceil,
    "mean": _mean,
    "average": _mean,
    "avg": _mean,
    "min": _min,
    "minimum": _min,
    "max": _max,
    "maximum": _max,
    "factorial": _factorial,
    "log": _log,
    "ln": _log,
    "log10": _log10,
    "exp": _exp,
    "percentage": _percentage,
    "percent": _percentage,
    "percent_of": _percent_of,
    "gcd": _gcd,
    "lcm": _lcm,
}


def is_known_primitive(name: str) -> bool:
    return name in PRIMITIVES


# ---------------------------------------------------------------------------
# Public execution entry points
# ---------------------------------------------------------------------------


def execute_one(
    call: Dict[str, Any],
    by_label: Dict[str, Any],
    indexed: List[Any],
    *,
    index: int = 0,
    ibm_registry: "Optional[IBMFunctionRegistry]" = None,
) -> CallTrace:
    """Execute a single call against an existing variable scope.

    Routing order:

    1. ``PRIMITIVES`` (math sandbox).
    2. *ibm_registry* if provided — the IBM/NESTFUL Python implementation
       under ``data_v2/executable_functions/``.
    3. ``unknown_function`` error.

    Always returns a :class:`CallTrace` — runtime errors are recorded in
    ``error`` rather than raised so the multiturn loop can keep going
    (or, if the user opted in, the runner can delegate to the LLM judge).
    """
    name = (call.get("name") or "").strip()
    label = (call.get("label") or f"var_{index + 1}").strip() or f"var_{index + 1}"
    arguments_raw = call.get("arguments") or {}
    if not isinstance(arguments_raw, dict):
        return CallTrace(
            index=index,
            name=name,
            label=label,
            arguments_resolved={},
            result=None,
            error="invalid_arguments_type",
            source="unknown",
        )

    arguments_resolved, var_err = resolve_variables(arguments_raw, by_label, indexed)
    if var_err is not None:
        return CallTrace(
            index=index,
            name=name,
            label=label,
            arguments_resolved=arguments_resolved,
            result=None,
            error=var_err,
            source="unknown",
        )

    primitive_fn = PRIMITIVES.get(name)
    if primitive_fn is not None:
        args_positional = values_in_order(arguments_resolved)
        try:
            result = primitive_fn(args_positional, **arguments_resolved)
        except ZeroDivisionError as exc:
            return CallTrace(
                index=index,
                name=name,
                label=label,
                arguments_resolved=arguments_resolved,
                result=None,
                error=f"primitive_error:zero_division:{exc}",
                source="primitive",
            )
        except (ValueError, TypeError, OverflowError, ArithmeticError) as exc:
            return CallTrace(
                index=index,
                name=name,
                label=label,
                arguments_resolved=arguments_resolved,
                result=None,
                error=f"primitive_error:{type(exc).__name__}:{exc}",
                source="primitive",
            )

        return CallTrace(
            index=index,
            name=name,
            label=label,
            arguments_resolved=arguments_resolved,
            result=result,
            error=None,
            source="primitive",
        )

    if ibm_registry is not None:
        ibm_fn = ibm_registry.get(name)
        if ibm_fn is not None:
            try:
                result = _call_fn_smart(ibm_fn, arguments_resolved)
            except BaseException as exc:  # IBM funcs occasionally raise SystemExit
                return CallTrace(
                    index=index,
                    name=name,
                    label=label,
                    arguments_resolved=arguments_resolved,
                    result=None,
                    error=f"ibm_runtime_error:{type(exc).__name__}:{exc}",
                    source="ibm",
                )

            return CallTrace(
                index=index,
                name=name,
                label=label,
                arguments_resolved=arguments_resolved,
                result=result,
                error=None,
                source="ibm",
            )

        # registry was queried but missed this name — record specifically
        return CallTrace(
            index=index,
            name=name,
            label=label,
            arguments_resolved=arguments_resolved,
            result=None,
            error=f"ibm_unavailable:{name}",
            source="ibm",
        )

    return CallTrace(
        index=index,
        name=name,
        label=label,
        arguments_resolved=arguments_resolved,
        result=None,
        error=f"unknown_function:{name}",
        source="unknown",
    )


def execute_call_sequence(
    calls: List[Dict[str, Any]],
    *,
    ibm_registry: "Optional[IBMFunctionRegistry]" = None,
) -> ExecutionResult:
    """Execute every call in order, threading variable scope through.

    Stops at the first error and reports it; the runner decides whether to
    fall back to the LLM judge based on the error category and on whether
    the user opted into ``--nestful-use-judge``.
    """
    if not calls:
        return ExecutionResult(
            success=False,
            final_value=None,
            per_call=[],
            error="empty_call_sequence",
            error_call_index=None,
        )

    by_label: Dict[str, Any] = {}
    indexed: List[Any] = []
    traces: List[CallTrace] = []

    for i, call in enumerate(calls):
        trace = execute_one(
            call, by_label, indexed, index=i, ibm_registry=ibm_registry
        )
        traces.append(trace)
        if trace.error is not None:
            return ExecutionResult(
                success=False,
                final_value=None,
                per_call=traces,
                error=trace.error,
                error_call_index=i,
            )
        by_label[trace.label] = trace.result
        indexed.append(trace.result)

    return ExecutionResult(
        success=True,
        final_value=indexed[-1] if indexed else None,
        per_call=traces,
        error=None,
        error_call_index=None,
    )


def _call_fn_smart(fn: Callable[..., Any], arg_dict: Dict[str, Any]) -> Any:
    """Bridge from NESTFUL's positional-style args to IBM's named params.

    Imported lazily because :mod:`ibm_loader` is optional in unit tests.
    """
    from .ibm_loader import call_with_signature_match

    return call_with_signature_match(fn, arg_dict)
