"""Trainer executor adapter: exposes the targeted_tool_data_factory registry
through the exact module contract the GRPO trainer expects.

The trainer loads its executable tool registry via
``nestful_mtgrpo_minimal/synthetic_tool_registry.py``, which imports
``lib.synthetic_tools`` from ``$SYNTHETIC_TOOLS_DIR``. Pointing that variable
at ``experiments/targeted_tool_data_factory/trainer_adapter`` makes
``ToolExecutor(mode="synthetic")`` execute FACTORY primitives for real:

    SYNTHETIC_TOOLS_DIR=.../targeted_tool_data_factory/trainer_adapter

Guarantees
    * every exported surface tool name resolves to its deterministic
      primitive — a wrong argument executes for real and returns a wrong
      observation, it can never fall back to the gold value;
    * schemas are produced by the same ``render.tool_to_jsonschema`` used for
      the dataset, so embedded task schemas and the runtime registry cannot
      drift;
    * no import of, and no fallback to, the old
      ``nestful_synthetic_curriculum_v3/lib/synthetic_tools.py``.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Callable, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_FACTORY_SRC = os.path.normpath(os.path.join(_HERE, "..", "..", "src"))
if _FACTORY_SRC not in sys.path:
    sys.path.insert(0, _FACTORY_SRC)

from targeted_tool_data import GENERATOR_VERSION            # noqa: E402
from targeted_tool_data import registry as _reg             # noqa: E402
from targeted_tool_data.executor import executor_hash as _exec_hash  # noqa: E402
from targeted_tool_data.render import (render_tool as _render_tool,  # noqa: E402
                                       tool_to_jsonschema as _to_js)
from targeted_tool_data.util import sha256_obj              # noqa: E402

ADAPTER_VERSION = "ttdf-adapter-1.0.0"
REGISTRY_VERSION = f"ttdf-{GENERATOR_VERSION}+{ADAPTER_VERSION}"
REGISTRY_SOURCE = "targeted_tool_data_factory"

TOOLS: Dict[str, Dict[str, Any]] = {}
_SPEC_CACHE: Dict[str, Any] = {}


def _json_type(ptype: str) -> str:
    if ptype.startswith("enum:"):
        return "string"
    return ptype


def _make_fn(sid: str, surface_to_canonical: Dict[str, str]) -> Callable[..., Any]:
    prim = _reg.get(sid)

    def fn(**kwargs: Any) -> Any:
        mapped = {}
        for surface_name, value in kwargs.items():
            canon = surface_to_canonical.get(surface_name)
            if canon is None:
                raise KeyError(f"unknown argument {surface_name!r} for {sid}")
            mapped[canon] = value
        for (canon, ptype, _sem) in prim.params:
            if canon not in mapped:
                raise KeyError(f"missing argument {canon!r} for {sid}")
            if ptype.startswith("enum:") and str(mapped[canon]) not in ptype[5:].split(","):
                raise ValueError(f"{canon} must be one of {ptype[5:]}")
        return prim.fn(**mapped)

    fn.__name__ = f"ttdf_{sid}"
    return fn


def _build() -> None:
    import random

    rng = random.Random(0)
    for sid, track, surf in _reg.all_surfaces():
        prim = _reg.get(sid)
        if surf.name in TOOLS:                     # uniqueness is enforced by
            continue                               # the registry itself
        params: Dict[str, Dict[str, Any]] = {}
        s2c: Dict[str, str] = {}
        for (canon, ptype, semantic), pname in zip(prim.params, surf.param_names):
            entry: Dict[str, Any] = {
                "type": _json_type(ptype),
                "desc": f"{pname.replace('_', ' ')}.",
                "semantic": semantic,
                "required": True,
            }
            if ptype.startswith("enum:"):
                entry["enum"] = ptype[5:].split(",")
            if ptype == _reg.ARR:
                entry["items"] = {"type": "number"}
            params[pname] = entry
            s2c[pname] = canon
        TOOLS[surf.name] = {
            "name": surf.name,
            "domain": prim.category,
            "family": sid,
            "track": track,
            "description": surf.description,
            "params": params,
            "out_key": surf.output_field,
            "out_type": prim.out_type,
            "out_semantic": prim.out_semantic,
            "fn": _make_fn(sid, s2c),
            "semantic_id": sid,
            "surface_id": surf.surface_id,
        }
        _SPEC_CACHE[surf.name] = _to_js(
            _render_tool(sid, track, rng, surface=surf, param_style="as_surface"))


_build()
ALL_TOOL_NAMES: List[str] = sorted(TOOLS)


def tool_schema(name: str) -> Dict[str, Any]:
    """JSON-Schema tool description, byte-identical to the exported dataset."""
    return _SPEC_CACHE[name]


def semantics_compatible(producer: str, consumer_semantic: str) -> bool:
    """Kept for exec_bridge API compatibility. The factory validates typed
    composition at generation time, so runtime composition is unrestricted."""
    return True


def registry_hash() -> str:
    return sha256_obj({
        "adapter": ADAPTER_VERSION,
        "registry": _reg.registry_hash(),
        "executor": _exec_hash(),
        "tools": {n: sorted(TOOLS[n]["params"]) for n in ALL_TOOL_NAMES},
    })


def factory_hashes() -> Dict[str, str]:
    return {"registry_hash": _reg.registry_hash(),
            "executor_hash": _exec_hash(),
            "adapter_registry_hash": registry_hash(),
            "adapter_version": ADAPTER_VERSION,
            "generator_version": GENERATOR_VERSION,
            "n_tools": str(len(ALL_TOOL_NAMES))}


__all__ = ["TOOLS", "ALL_TOOL_NAMES", "REGISTRY_VERSION", "REGISTRY_SOURCE",
           "ADAPTER_VERSION", "tool_schema", "registry_hash",
           "semantics_compatible", "factory_hashes"]
