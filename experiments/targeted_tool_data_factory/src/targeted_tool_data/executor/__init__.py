"""Deterministic graph executor — the ONLY source of oracle truth.

Executes a SemanticProgram topologically, resolves node references,
produces oracle observations and the oracle final answer. Replay support
verifies bitwise-identical results (V2 hard gate).
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .. import registry as reg
from ..graph import REF, _refs_in
from ..schemas import SemanticProgram

MAX_ABS = 1e9


class ExecutionError(Exception):
    pass


def _resolve(v: Any, values: Dict[str, Any]) -> Any:
    if isinstance(v, dict) and REF in v:
        nid = v[REF]
        if nid not in values:
            raise ExecutionError(f"unresolved reference {nid}")
        return values[nid]
    if isinstance(v, list):
        return [_resolve(item, values) for item in v]
    return v


def _check_value(x: Any) -> None:
    if isinstance(x, bool):
        return
    if isinstance(x, (int, float)):
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            raise ExecutionError("NaN/Inf result")
        if abs(float(x)) > MAX_ABS:
            raise ExecutionError("magnitude overflow")
    if isinstance(x, list):
        for item in x:
            _check_value(item)


def execute(prog: SemanticProgram) -> Tuple[List[Any], Any]:
    """Returns (observations in node order, final answer)."""
    values: Dict[str, Any] = {}
    observations: List[Any] = []
    for nd in prog.nodes:
        prim = reg.get(nd.semantic_id)
        kwargs = {}
        for (pname, ptype, _sem) in prim.params:
            if pname not in nd.inputs:
                raise ExecutionError(f"{nd.node_id}: missing param {pname}")
            kwargs[pname] = _resolve(nd.inputs[pname], values)
        try:
            out = prim.fn(**kwargs)
        except (ZeroDivisionError, ValueError, OverflowError, TypeError,
                KeyError, IndexError, ArithmeticError) as exc:
            raise ExecutionError(f"{nd.node_id}:{nd.semantic_id}: {exc}") from exc
        _check_value(out)
        values[nd.node_id] = out
        observations.append(out)
    return observations, values[prog.sink]


def replay_consistent(prog: SemanticProgram, n: int = 2) -> bool:
    results = [execute(prog) for _ in range(n)]
    first = results[0]
    return all(r == first for r in results[1:])


def executor_hash() -> str:
    src = Path(__file__).read_bytes()
    return hashlib.sha256(src).hexdigest()


def node_values(prog: SemanticProgram) -> Dict[str, Any]:
    obs, _ = execute(prog)
    return {nd.node_id: obs[i] for i, nd in enumerate(prog.nodes)}


def question_constants(prog: SemanticProgram) -> List[Any]:
    """All direct constants a solver could read from the query."""
    consts: List[Any] = []
    for nd in prog.nodes:
        for v in nd.inputs.values():
            if not _refs_in(v):
                if isinstance(v, list):
                    consts.append(v)
                    consts.extend(v)
                else:
                    consts.append(v)
    return consts
