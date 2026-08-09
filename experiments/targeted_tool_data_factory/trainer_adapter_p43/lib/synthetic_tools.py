"""P43 trainer executor adapter: Pilot4.3 ``ops.build_ops()`` → synthetic TOOLS.

Point the trainer at this directory:

    SYNTHETIC_TOOLS_DIR=.../targeted_tool_data_factory/trainer_adapter_p43
    executor.mode=synthetic

This exposes every Pilot4.3 surface name (base + ops_new), including all tools
used in ``train_nestful_profile_1000.jsonl``. It does NOT use the older 190-tool
pilot2/3 ``targeted_tool_data.registry`` adapter.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Callable, Dict, List

_HERE = os.path.dirname(os.path.abspath(__file__))
_FACTORY_SRC = os.path.normpath(os.path.join(_HERE, "..", "..", "src"))
if _FACTORY_SRC not in sys.path:
    sys.path.insert(0, _FACTORY_SRC)

from targeted_tool_data.pilot43 import ops as _p43_ops  # noqa: E402
from targeted_tool_data.pilot43 import semtypes as st  # noqa: E402
from targeted_tool_data.pilot43.ops import PRESERVE  # noqa: E402
from targeted_tool_data.util import sha256_obj  # noqa: E402

ADAPTER_VERSION = "ttdf-p43-adapter-1.0.0"
REGISTRY_SOURCE = "targeted_tool_data_factory.pilot43"
REGISTRY_VERSION = f"pilot43+{ADAPTER_VERSION}"

TOOLS: Dict[str, Dict[str, Any]] = {}


def _param_json_type(sem: str) -> str:
    return st.runtime_of(sem)


def _items_schema(sem: str) -> Dict[str, Any] | None:
    if sem == st.NUMBER_LIST:
        return {"type": "number"}
    if sem == st.TEXT_LIST:
        return {"type": "string"}
    if sem == st.RECORD_LIST:
        return {"type": "object"}
    return None


def _make_fn(op: _p43_ops.Op, surface: _p43_ops.Surface) -> Callable[..., Any]:
    s2c = {sp: p.name for sp, p in zip(surface.param_names, op.params)}

    def fn(**kwargs: Any) -> Any:
        mapped: Dict[str, Any] = {}
        for surface_name, value in kwargs.items():
            canon = s2c.get(surface_name)
            if canon is None:
                raise KeyError(
                    f"unknown argument {surface_name!r} for {surface.name}")
            mapped[canon] = value
        for p in op.params:
            if p.name not in mapped:
                raise KeyError(f"missing argument {p.name!r} for {op.pid}")
        return op.fn(**mapped)

    fn.__name__ = f"p43_{op.pid}_{surface.name}"
    return fn


def _build() -> None:
    ops = _p43_ops.build_ops()
    for op in ops.values():
        out_type = (
            "number" if op.out_sem == PRESERVE else st.runtime_of(op.out_sem))
        for surf in op.surfaces:
            if surf.name in TOOLS:
                # validate_ops already enforces unique (name, arity) signatures
                continue
            params: Dict[str, Dict[str, Any]] = {}
            for p, pname in zip(op.params, surf.param_names):
                entry: Dict[str, Any] = {
                    "type": _param_json_type(p.sem),
                    "desc": f"{pname.replace('_', ' ')}.",
                    "semantic": p.sem,
                    "required": True,
                }
                items = _items_schema(p.sem)
                if items is not None:
                    entry["items"] = items
                params[pname] = entry
            TOOLS[surf.name] = {
                "name": surf.name,
                "domain": op.family,
                "family": op.pid,
                "track": surf.track,
                "description": surf.description,
                "params": params,
                "out_key": surf.output_field,
                "out_type": out_type,
                "out_semantic": op.out_sem,
                "fn": _make_fn(op, surf),
                "semantic_id": op.pid,
                "capability": op.capability,
            }


_build()
ALL_TOOL_NAMES: List[str] = sorted(TOOLS)


def tool_schema(name: str) -> Dict[str, Any]:
    """JSON-schema shaped tool description (OpenAI-style parameters object)."""
    spec = TOOLS[name]
    properties = {}
    required = []
    for pname, meta in spec["params"].items():
        prop: Dict[str, Any] = {"type": meta["type"], "description": meta["desc"]}
        if "items" in meta:
            prop["items"] = meta["items"]
        if "enum" in meta:
            prop["enum"] = meta["enum"]
        properties[pname] = prop
        if meta.get("required", True):
            required.append(pname)
    return {
        "name": name,
        "description": spec["description"],
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


def semantics_compatible(producer: str, consumer_semantic: str) -> bool:
    return True


def registry_hash() -> str:
    return sha256_obj({
        "adapter": ADAPTER_VERSION,
        "pilot43_ops": _p43_ops.registry_hash(),
        "tools": {n: sorted(TOOLS[n]["params"]) for n in ALL_TOOL_NAMES},
    })


def factory_hashes() -> Dict[str, str]:
    return {
        "pilot43_ops_hash": _p43_ops.registry_hash(),
        "adapter_registry_hash": registry_hash(),
        "adapter_version": ADAPTER_VERSION,
        "registry_source": REGISTRY_SOURCE,
        "n_tools": str(len(ALL_TOOL_NAMES)),
        "n_ops": str(len(_p43_ops.build_ops())),
    }


__all__ = [
    "TOOLS", "ALL_TOOL_NAMES", "REGISTRY_VERSION", "REGISTRY_SOURCE",
    "ADAPTER_VERSION", "tool_schema", "registry_hash",
    "semantics_compatible", "factory_hashes",
]
