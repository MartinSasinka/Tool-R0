"""Tests for tool-observation truncation in prompt.py.

A tool can return an arbitrarily large object. Without truncation it is
serialised verbatim into the next-turn <tool_response>, blows the prompt past
the context window and CRASHES vLLM. These tests verify the configurable caps
on string length, list/dict size, and total payload size.
"""
from __future__ import annotations

import json

import pytest

import prompt


@pytest.fixture(autouse=True)
def _reset_limits():
    # Each test sets its own limits; restore defaults afterwards.
    prompt.set_observation_limits(None)
    yield
    prompt.set_observation_limits(None)


def _payload(call_name, result_str):
    # Parse the JSON inside the <tool_response>...</tool_response> wrapper.
    assert result_str.startswith("<tool_response>")
    assert result_str.endswith("</tool_response>")
    inner = result_str[len("<tool_response>"):-len("</tool_response>")]
    return json.loads(inner)


def test_defaults_loaded():
    lim = prompt.set_observation_limits(None)
    assert lim["max_str_chars"] == 2000
    assert lim["max_items"] == 200
    assert lim["max_total_chars"] == 6000


def test_long_string_truncated():
    prompt.set_observation_limits(
        {"generation": {"observation_limits": {"max_str_chars": 50, "max_total_chars": 100000}}}
    )
    big = "A" * 5000
    out = prompt.format_tool_response({"name": "f"}, big)
    payload = _payload("f", out)
    assert len(payload["result"]) < 200
    assert "truncated" in payload["result"]


def test_long_list_truncated():
    prompt.set_observation_limits(
        {"generation": {"observation_limits": {"max_items": 10, "max_total_chars": 100000}}}
    )
    big = list(range(1000))
    out = prompt.format_tool_response({"name": "f"}, big)
    payload = _payload("f", out)
    # 10 kept items + 1 truncation marker string.
    assert len(payload["result"]) == 11
    assert "truncated" in str(payload["result"][-1])


def test_large_dict_truncated():
    prompt.set_observation_limits(
        {"generation": {"observation_limits": {"max_items": 5, "max_total_chars": 100000}}}
    )
    big = {f"k{i}": i for i in range(100)}
    out = prompt.format_tool_response({"name": "f"}, big)
    payload = _payload("f", out)
    assert "__truncated__" in payload["result"]
    # 5 kept keys + the truncation key.
    assert len(payload["result"]) == 6


def test_total_payload_hard_cap():
    # Even with generous per-element caps, the whole payload must stay bounded.
    prompt.set_observation_limits(
        {"generation": {"observation_limits": {"max_str_chars": 100000,
                                               "max_items": 100000,
                                               "max_total_chars": 500}}}
    )
    big = {f"key_{i}": "v" * 100 for i in range(1000)}
    out = prompt.format_tool_response({"name": "f"}, big)
    # The full <tool_response> wrapper plus a small marker; comfortably bounded.
    assert len(out) < 1200
    assert "truncated" in out


def test_nested_structure_bounded():
    prompt.set_observation_limits(
        {"generation": {"observation_limits": {"max_str_chars": 20, "max_items": 3,
                                               "max_total_chars": 100000}}}
    )
    nested = {"a": ["x" * 1000, "y" * 1000], "b": list(range(50)), "c": {"d": "z" * 1000}}
    out = prompt.format_tool_response({"name": "f"}, nested)
    payload = _payload("f", out)
    # Inner long strings truncated.
    assert all(len(s) < 60 for s in payload["result"]["a"])
    # Inner long list truncated to max_items (+marker).
    assert len(payload["result"]["b"]) == 4


def test_disabled_caps_with_zero():
    prompt.set_observation_limits(
        {"generation": {"observation_limits": {"max_str_chars": 0, "max_items": 0,
                                               "max_total_chars": 0}}}
    )
    big = "A" * 5000
    out = prompt.format_tool_response({"name": "f"}, big)
    payload = _payload("f", out)
    assert len(payload["result"]) == 5000  # untouched


def test_huge_int_still_handled():
    # Truncation must not break the existing huge-int safety path.
    prompt.set_observation_limits(None)
    out = prompt.format_tool_response({"name": "f"}, {"n": 12345})
    payload = _payload("f", out)
    assert payload["result"]["n"] == 12345
