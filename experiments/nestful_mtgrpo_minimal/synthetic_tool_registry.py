"""Loader for the executable synthetic tool registry (executor mode="synthetic").

The authoritative registry lives in
``experiments/nestful_synthetic_curriculum_v3/lib/synthetic_tools.py`` — the SAME
module the dataset generator uses, so the generator, the dataset and the trainer
always execute identical tool implementations. This loader only puts that module
on ``sys.path`` and exposes a thin, cached accessor object.

Override the registry location with the ``SYNTHETIC_TOOLS_DIR`` environment
variable (a directory that contains ``lib/synthetic_tools.py``).
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_V3_DIR = os.path.normpath(
    os.path.join(_HERE, "..", "nestful_synthetic_curriculum_v3"))


class SyntheticToolRegistry:
    """Cached view over lib.synthetic_tools. ``available`` is False when the
    module cannot be imported (registry dir missing)."""

    def __init__(self, v3_dir: Optional[str] = None) -> None:
        self.v3_dir = v3_dir or os.environ.get("SYNTHETIC_TOOLS_DIR", _DEFAULT_V3_DIR)
        self._mod = None
        self._error: Optional[str] = None
        try:
            if self.v3_dir not in sys.path:
                sys.path.insert(0, self.v3_dir)
            from lib import synthetic_tools as _mod  # type: ignore
            self._mod = _mod
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
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = SyntheticToolRegistry()
    return _SINGLETON
