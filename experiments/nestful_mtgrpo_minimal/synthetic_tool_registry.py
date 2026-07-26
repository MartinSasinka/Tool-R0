"""Loader for the executable synthetic tool registry (executor mode="synthetic").

The authoritative registry lives in
``experiments/nestful_synthetic_curriculum_v3/lib/synthetic_tools.py`` — the SAME
module the dataset generator uses, so the generator, the dataset and the trainer
always execute identical tool implementations. This loader only puts that module
on ``sys.path`` and exposes a thin, cached accessor object.

Override the registry location with the ``SYNTHETIC_TOOLS_DIR`` environment
variable (a directory that contains ``lib/synthetic_tools.py``).

IMPORTANT: a plain ``import lib.synthetic_tools`` is NOT enough when package
``lib`` was already imported from a different tree (e.g. Stage-3 v3 vs the
factory trainer_adapter). Package ``__path__`` pins submodule discovery, so
this loader always binds the module via ``importlib`` from the concrete file
path and injects it into ``sys.modules['lib.synthetic_tools']``.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types
from typing import Any, Dict, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_V3_DIR = os.path.normpath(
    os.path.join(_HERE, "..", "nestful_synthetic_curriculum_v3"))


def load_synthetic_tools_module(tools_dir: Optional[str] = None):
    """Load ``lib/synthetic_tools.py`` from ``tools_dir`` by file path.

    Returns the module object and installs it as ``sys.modules['lib.synthetic_tools']``
    so subsequent ``from lib.synthetic_tools import …`` / ``ToolExecutor`` see
    the same registry the caller requested.
    """
    tools_dir = os.path.abspath(
        tools_dir or os.environ.get("SYNTHETIC_TOOLS_DIR", _DEFAULT_V3_DIR))
    path = os.path.join(tools_dir, "lib", "synthetic_tools.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"synthetic tools registry missing: {path}")

    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)

    name = "lib.synthetic_tools"
    for key in list(sys.modules):
        if key == name or key.startswith(name + "."):
            del sys.modules[key]

    # Ensure parent package exists. Prefer the default v3 ``lib`` on __path__
    # so ``lib.agentic_data`` stays importable even when SYNTHETIC_TOOLS_DIR
    # points at the factory adapter (which only ships synthetic_tools.py).
    if "lib" not in sys.modules:
        pkg = types.ModuleType("lib")
        paths = []
        v3_lib = os.path.join(_DEFAULT_V3_DIR, "lib")
        tools_lib = os.path.join(tools_dir, "lib")
        if os.path.isdir(v3_lib):
            paths.append(v3_lib)
        if os.path.isdir(tools_lib) and tools_lib not in paths:
            paths.append(tools_lib)
        pkg.__path__ = paths  # type: ignore[attr-defined]
        sys.modules["lib"] = pkg

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create import spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class SyntheticToolRegistry:
    """Cached view over lib.synthetic_tools. ``available`` is False when the
    module cannot be imported (registry dir missing)."""

    def __init__(self, v3_dir: Optional[str] = None) -> None:
        self.v3_dir = os.path.abspath(
            v3_dir or os.environ.get("SYNTHETIC_TOOLS_DIR", _DEFAULT_V3_DIR))
        self._mod = None
        self._error: Optional[str] = None
        try:
            self._mod = load_synthetic_tools_module(self.v3_dir)
        except Exception as exc:  # noqa: BLE001
            self._error = f"{type(exc).__name__}: {exc}"

    @property
    def available(self) -> bool:
        return self._mod is not None

    @property
    def load_error(self) -> Optional[str]:
        return self._error

    @property
    def version(self) -> Optional[str]:
        return self._mod.REGISTRY_VERSION if self._mod else None

    def registry_hash(self) -> Optional[str]:
        return self._mod.registry_hash() if self._mod else None

    def get(self, name: str) -> Optional[Dict[str, Any]]:
        """Full tool spec (schema + executable fn + semantics) or None."""
        if self._mod is None:
            return None
        return self._mod.TOOLS.get(name)

    def tool_names(self):
        return list(self._mod.ALL_TOOL_NAMES) if self._mod else []


_SINGLETON: Optional[SyntheticToolRegistry] = None


def get_synthetic_registry() -> SyntheticToolRegistry:
    """Return the process-wide registry, reloading if SYNTHETIC_TOOLS_DIR changed."""
    global _SINGLETON
    wanted = os.path.abspath(
        os.environ.get("SYNTHETIC_TOOLS_DIR", _DEFAULT_V3_DIR))
    if _SINGLETON is not None and os.path.abspath(_SINGLETON.v3_dir) != wanted:
        _SINGLETON = None
    if _SINGLETON is None:
        _SINGLETON = SyntheticToolRegistry(v3_dir=wanted)
    return _SINGLETON


def reset_synthetic_registry() -> None:
    """Drop the cached singleton (tests / registry-switch harnesses)."""
    global _SINGLETON
    _SINGLETON = None
