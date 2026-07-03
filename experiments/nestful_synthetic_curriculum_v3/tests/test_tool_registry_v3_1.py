"""Tests for tool_registry_v3_1."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from tool_registry_v3_1 import (  # noqa: E402
    ALL_TOOL_NAMES,
    FAMILY_MAP,
    execute_tool,
    infer_answer_type,
    tool_pool_for_families,
)


def test_tool_count_and_families():
    assert len(ALL_TOOL_NAMES) >= 30
    assert len(set(FAMILY_MAP.values())) >= 5


def test_math_tools_execute():
    assert execute_tool("add", {"arg_0": 2, "arg_1": 3}) == 5.0
    assert execute_tool("divide_safe", {"arg_0": 10, "arg_1": 2}) == 5.0


def test_string_list_object_tools():
    assert execute_tool("concat", {"a": "a", "b": "b"}) == "ab"
    assert execute_tool("filter_greater_than", {"values": [1, 5, 3], "threshold": 2}) == [5, 3]
    assert execute_tool("make_object", {"key": "k", "value": "v"}) == {"k": "v"}


def test_non_scalar_output_types_exist():
    pool = tool_pool_for_families(["string", "list", "object", "boolean", "lookup"])
    types = {t["output_type"] for t in pool}
    assert "string" in types
    assert "list" in types or "object" in types
    assert infer_answer_type([1, 2]) == "list"
    assert infer_answer_type({"a": 1}) == "object"


def test_lookup_tools():
    rec = execute_tool("lookup_by_key", {"table": {}, "key": "alpha"})
    assert rec is not None
