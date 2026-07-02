"""Canonical parser (re-export of nestful_mtgrpo_minimal/parser.py).

Single source of truth for both experiments. See ``nestful_core.rewards`` and
``docs/PIPELINE_V2_STRUCTURE.md`` for the canonical-parser diagnostics contract
(``strict_ok`` / ``lenient_ok`` / ``parse_recovery``).
"""
from __future__ import annotations

from . import ensure_paths

ensure_paths()

# Absolute import resolves to the top-level (minimal) parser on sys.path, NOT
# this package module (which is nestful_core.parser).
from parser import (  # noqa: E402,F401
    ParseResult,
    parse_tool_call,
    parse_tool_calls_all,
)


def parse_canonical(text: str):
    """Canonical parse exposing BOTH strict and lenient outcomes in one result.

    Returns a dict-like diagnostic so callers (training reward + eval) share one
    parser and can log strict/lenient/recovery rates consistently:

        {
          "strict_ok": bool,        # passes the strict single-tag gate
          "lenient_ok": bool,       # recoverable under lenient eval rules
          "parse_recovery": bool,   # lenient_ok AND NOT strict_ok
          "call": dict | None,      # normalized call (from whichever mode succeeded)
          "is_terminal": bool,      # [] terminal signal
          "reason": str | None,     # strict failure reason
        }
    """
    strict = parse_tool_call(text, lenient=False)
    if strict.ok:
        return {
            "strict_ok": True,
            "lenient_ok": True,
            "parse_recovery": False,
            "call": strict.call,
            "is_terminal": bool(strict.is_terminal),
            "reason": None,
        }
    lenient = parse_tool_call(text, lenient=True)
    return {
        "strict_ok": False,
        "lenient_ok": bool(lenient.ok),
        "parse_recovery": bool(lenient.ok),
        "call": lenient.call if lenient.ok else None,
        "is_terminal": bool(lenient.is_terminal),
        "reason": strict.reason,
    }
