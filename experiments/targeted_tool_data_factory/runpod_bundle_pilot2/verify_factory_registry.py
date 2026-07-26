#!/usr/bin/env python3
"""Fail-fast check that SYNTHETIC_TOOLS_DIR points at the factory registry."""
from __future__ import annotations

import os
import sys
from pathlib import Path

BUNDLE = Path(__file__).resolve().parent
FACTORY = BUNDLE.parent
EXPERIMENTS = FACTORY.parent
MINIMAL = EXPERIMENTS / "nestful_mtgrpo_minimal"
DEFAULT_TOOLS = FACTORY / "trainer_adapter"


def main() -> int:
    tools = Path(os.environ.get("SYNTHETIC_TOOLS_DIR", str(DEFAULT_TOOLS))).resolve()
    sys.path.insert(0, str(MINIMAL))
    from synthetic_tool_registry import load_synthetic_tools_module  # noqa: WPS433

    mod = load_synthetic_tools_module(str(tools))
    print(
        f"[registry] {mod.REGISTRY_VERSION} hash={mod.registry_hash()[:16]}... "
        f"n_tools={len(mod.TOOLS)} source={getattr(mod, 'REGISTRY_SOURCE', '?')}"
    )
    if getattr(mod, "REGISTRY_SOURCE", "") != "targeted_tool_data_factory":
        print("[registry] ABORT: REGISTRY_SOURCE is not targeted_tool_data_factory",
              file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
