#!/usr/bin/env python3
"""Expanded prototype tool registry for v3.1 curriculum (6 families)."""
from __future__ import annotations

from typing import Any, Dict, List, Set

MATH_TOOL_NAMES: Set[str] = {
    "add", "subtract", "multiply", "divide_safe", "sum_list",
    "mean_list", "max_list", "min_list",
}

STRING_TOOL_NAMES: Set[str] = {
    "concat", "lowercase", "uppercase", "extract_prefix", "extract_suffix",
    "contains_substring", "string_length",
}

LIST_TOOL_NAMES: Set[str] = {
    "get_item", "list_length", "filter_greater_than", "filter_equals",
    "join_list", "sort_list", "count_items",
}

OBJECT_TOOL_NAMES: Set[str] = {
    "make_object", "get_field", "merge_objects", "update_field", "nested_get",
}

BOOLEAN_TOOL_NAMES: Set[str] = {
    "greater_than", "less_than", "equals", "contains", "and_bool", "or_bool",
}

LOOKUP_TOOL_NAMES: Set[str] = {
    "lookup_by_key", "select_record", "count_records", "aggregate_field",
}

ALL_TOOL_NAMES: Set[str] = (
    MATH_TOOL_NAMES | STRING_TOOL_NAMES | LIST_TOOL_NAMES
    | OBJECT_TOOL_NAMES | BOOLEAN_TOOL_NAMES | LOOKUP_TOOL_NAMES
)

FAMILY_MAP = {
    **{n: "math" for n in MATH_TOOL_NAMES},
    **{n: "string" for n in STRING_TOOL_NAMES},
    **{n: "list" for n in LIST_TOOL_NAMES},
    **{n: "object" for n in OBJECT_TOOL_NAMES},
    **{n: "boolean" for n in BOOLEAN_TOOL_NAMES},
    **{n: "lookup" for n in LOOKUP_TOOL_NAMES},
}

OUTPUT_TYPE_MAP = {
    "add": "scalar", "subtract": "scalar", "multiply": "scalar", "divide_safe": "scalar",
    "sum_list": "scalar", "mean_list": "scalar", "max_list": "scalar", "min_list": "scalar",
    "concat": "string", "lowercase": "string", "uppercase": "string",
    "extract_prefix": "string", "extract_suffix": "string", "contains_substring": "boolean",
    "string_length": "scalar",
    "get_item": "mixed", "list_length": "scalar", "filter_greater_than": "list",
    "filter_equals": "list", "join_list": "string", "sort_list": "list", "count_items": "scalar",
    "make_object": "object", "get_field": "mixed", "merge_objects": "object",
    "update_field": "object", "nested_get": "mixed",
    "greater_than": "boolean", "less_than": "boolean", "equals": "boolean",
    "contains": "boolean", "and_bool": "boolean", "or_bool": "boolean",
    "lookup_by_key": "mixed", "select_record": "object", "count_records": "scalar",
    "aggregate_field": "scalar",
}

_DEFAULT_TABLE: Dict[str, Any] = {
    "alpha": {"id": 1, "score": 10, "name": "alpha"},
    "beta": {"id": 2, "score": 20, "name": "beta"},
    "gamma": {"id": 3, "score": 15, "name": "gamma"},
}


def _tool(name: str, desc: str, params: dict, out_type: str = "scalar", out_key: str = "result") -> dict:
    props = {k: {"type": v, "description": k} for k, v in params.items()}
    return {
        "name": name,
        "family": FAMILY_MAP.get(name, "math"),
        "description": desc,
        "schema": {"type": "object", "properties": props, "required": list(params.keys())},
        "parameters": {"type": "object", "properties": props, "required": list(params.keys())},
        "output_parameters": {out_key: {"type": out_type, "description": ""}},
        "output_type": out_type,
        "executor": "python function",
    }


MATH_TOOLS = {
    "add": _tool("add", "Add two numbers", {"arg_0": "number", "arg_1": "number"}),
    "subtract": _tool("subtract", "Subtract", {"arg_0": "number", "arg_1": "number"}),
    "multiply": _tool("multiply", "Multiply", {"arg_0": "number", "arg_1": "number"}),
    "divide_safe": _tool("divide_safe", "Safe divide", {"arg_0": "number", "arg_1": "number"}),
    "sum_list": _tool("sum_list", "Sum list", {"values": "array"}),
    "mean_list": _tool("mean_list", "Mean list", {"values": "array"}),
    "max_list": _tool("max_list", "Max list", {"values": "array"}),
    "min_list": _tool("min_list", "Min list", {"values": "array"}),
}

STRING_TOOLS = {
    "concat": _tool("concat", "Concat strings", {"a": "string", "b": "string"}, out_type="string"),
    "lowercase": _tool("lowercase", "Lowercase", {"text": "string"}, out_type="string"),
    "uppercase": _tool("uppercase", "Uppercase", {"text": "string"}, out_type="string"),
    "extract_prefix": _tool("extract_prefix", "Prefix", {"text": "string", "n": "number"}, out_type="string"),
    "extract_suffix": _tool("extract_suffix", "Suffix", {"text": "string", "n": "number"}, out_type="string"),
    "contains_substring": _tool(
        "contains_substring", "Contains substring", {"text": "string", "sub": "string"}, out_type="boolean"
    ),
    "string_length": _tool("string_length", "String length", {"text": "string"}),
}

LIST_TOOLS = {
    "get_item": _tool("get_item", "Get list item", {"values": "array", "index": "number"}, out_type="mixed"),
    "list_length": _tool("list_length", "List length", {"values": "array"}),
    "filter_greater_than": _tool(
        "filter_greater_than", "Filter list", {"values": "array", "threshold": "number"}, out_type="list"
    ),
    "filter_equals": _tool(
        "filter_equals", "Filter equals", {"values": "array", "target": "string"}, out_type="list"
    ),
    "join_list": _tool("join_list", "Join list", {"values": "array", "sep": "string"}, out_type="string"),
    "sort_list": _tool("sort_list", "Sort list", {"values": "array"}, out_type="list"),
    "count_items": _tool("count_items", "Count items", {"values": "array"}),
}

OBJECT_TOOLS = {
    "get_field": _tool("get_field", "Get field", {"obj": "object", "key": "string"}, out_type="mixed"),
    "merge_objects": _tool("merge_objects", "Merge objects", {"a": "object", "b": "object"}, out_type="object"),
    "make_object": _tool(
        "make_object", "Make object", {"key": "string", "value": "string"}, out_type="object"
    ),
    "update_field": _tool(
        "update_field", "Update field", {"obj": "object", "key": "string", "value": "string"}, out_type="object"
    ),
    "nested_get": _tool(
        "nested_get", "Nested get", {"obj": "object", "path": "string"}, out_type="mixed"
    ),
}

BOOLEAN_TOOLS = {
    "greater_than": _tool("greater_than", "Compare gt", {"a": "number", "b": "number"}, out_type="boolean"),
    "less_than": _tool("less_than", "Compare lt", {"a": "number", "b": "number"}, out_type="boolean"),
    "equals": _tool("equals", "Equality", {"a": "string", "b": "string"}, out_type="boolean"),
    "contains": _tool("contains", "Contains", {"text": "string", "sub": "string"}, out_type="boolean"),
    "and_bool": _tool("and_bool", "Logical and", {"a": "boolean", "b": "boolean"}, out_type="boolean"),
    "or_bool": _tool("or_bool", "Logical or", {"a": "boolean", "b": "boolean"}, out_type="boolean"),
}

LOOKUP_TOOLS = {
    "lookup_by_key": _tool(
        "lookup_by_key", "Lookup by key", {"table": "object", "key": "string"}, out_type="mixed"
    ),
    "select_record": _tool(
        "select_record", "Select record", {"records": "array", "index": "number"}, out_type="object"
    ),
    "count_records": _tool("count_records", "Count records", {"records": "array"}),
    "aggregate_field": _tool(
        "aggregate_field", "Aggregate field", {"records": "array", "field": "string"}
    ),
}

_ALL_DICTS = {
    **MATH_TOOLS, **STRING_TOOLS, **LIST_TOOLS,
    **OBJECT_TOOLS, **BOOLEAN_TOOLS, **LOOKUP_TOOLS,
}


def all_tool_defs(extra_distractors: bool = True) -> List[dict]:
    return list(_ALL_DICTS.values())


def tool_pool_for_families(families: List[str]) -> List[dict]:
    family_tools = {
        "math": MATH_TOOLS, "string": STRING_TOOLS, "list": LIST_TOOLS,
        "object": OBJECT_TOOLS, "boolean": BOOLEAN_TOOLS, "lookup": LOOKUP_TOOLS,
    }
    pool: List[dict] = []
    for fam in families:
        pool.extend(family_tools.get(fam, {}).values())
    if not pool:
        pool = all_tool_defs()
    seen: Set[str] = set()
    out = []
    for t in pool:
        if t["name"] not in seen:
            seen.add(t["name"])
            out.append(t)
    return out


def infer_answer_type(val: Any) -> str:
    if isinstance(val, bool):
        return "boolean"
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return "scalar"
    if isinstance(val, str):
        return "string"
    if isinstance(val, list):
        return "list"
    if isinstance(val, dict):
        return "object"
    return "scalar"


def infer_output_type(tool_name: str, result: Any = None) -> str:
    if result is not None:
        return infer_answer_type(result)
    return OUTPUT_TYPE_MAP.get(tool_name, "scalar")


def tool_family(name: str) -> str:
    return FAMILY_MAP.get(name, "other")


def execute_tool(name: str, args: Dict[str, Any]) -> Any:
    if name == "add":
        return float(args["arg_0"]) + float(args["arg_1"])
    if name == "subtract":
        return float(args["arg_0"]) - float(args["arg_1"])
    if name == "multiply":
        return float(args["arg_0"]) * float(args["arg_1"])
    if name == "divide_safe":
        d = float(args["arg_1"])
        return float(args["arg_0"]) / d if d != 0 else 0.0
    if name == "sum_list":
        vals = list(args.get("values") or [])
        return float(sum(float(v) for v in vals))
    if name == "mean_list":
        vals = list(args.get("values") or [])
        return float(sum(float(v) for v in vals)) / max(len(vals), 1)
    if name == "max_list":
        vals = list(args.get("values") or [])
        return float(max(float(v) for v in vals)) if vals else 0.0
    if name == "min_list":
        vals = list(args.get("values") or [])
        return float(min(float(v) for v in vals)) if vals else 0.0
    if name == "concat":
        return str(args["a"]) + str(args["b"])
    if name == "lowercase":
        return str(args["text"]).lower()
    if name == "uppercase":
        return str(args["text"]).upper()
    if name == "extract_prefix":
        return str(args["text"])[: int(args["n"])]
    if name == "extract_suffix":
        n = int(args["n"])
        return str(args["text"])[-n:] if n else ""
    if name == "contains_substring":
        return str(args["sub"]) in str(args["text"])
    if name == "string_length":
        return len(str(args["text"]))
    if name == "get_item":
        vals = list(args.get("values") or [])
        return vals[int(args["index"])]
    if name == "list_length":
        return len(args.get("values") or [])
    if name == "filter_greater_than":
        thr = float(args["threshold"])
        return [v for v in (args.get("values") or []) if float(v) > thr]
    if name == "filter_equals":
        tgt = str(args["target"])
        return [v for v in (args.get("values") or []) if str(v) == tgt]
    if name == "join_list":
        sep = str(args.get("sep", ","))
        return sep.join(str(v) for v in (args.get("values") or []))
    if name == "sort_list":
        return sorted(args.get("values") or [], key=lambda x: float(x) if _is_num(x) else str(x))
    if name == "count_items":
        return len(args.get("values") or [])
    if name == "get_field":
        obj = args.get("obj") or {}
        return obj.get(args["key"]) if isinstance(obj, dict) else None
    if name == "merge_objects":
        a = dict(args.get("a") or {})
        a.update(dict(args.get("b") or {}))
        return a
    if name == "make_object":
        return {str(args["key"]): args["value"]}
    if name == "update_field":
        obj = dict(args.get("obj") or {})
        obj[str(args["key"])] = args["value"]
        return obj
    if name == "nested_get":
        obj = args.get("obj") or {}
        path = str(args.get("path", "")).split(".")
        cur = obj
        for p in path:
            if isinstance(cur, dict):
                cur = cur.get(p)
            else:
                return None
        return cur
    if name == "greater_than":
        return float(args["a"]) > float(args["b"])
    if name == "less_than":
        return float(args["a"]) < float(args["b"])
    if name == "equals":
        return str(args["a"]) == str(args["b"])
    if name == "contains":
        return str(args["sub"]) in str(args["text"])
    if name == "and_bool":
        return bool(args["a"]) and bool(args["b"])
    if name == "or_bool":
        return bool(args["a"]) or bool(args["b"])
    if name == "lookup_by_key":
        table = args.get("table") or _DEFAULT_TABLE
        key = str(args["key"])
        if isinstance(table, dict) and key in table:
            return table[key]
        return _DEFAULT_TABLE.get(key)
    if name == "select_record":
        recs = list(args.get("records") or [])
        idx = int(args["index"])
        return recs[idx] if 0 <= idx < len(recs) else {}
    if name == "count_records":
        return len(args.get("records") or [])
    if name == "aggregate_field":
        recs = list(args.get("records") or [])
        field = str(args["field"])
        vals = [r.get(field) for r in recs if isinstance(r, dict) and field in r]
        nums = [float(v) for v in vals if _is_num(v)]
        return float(sum(nums)) if nums else 0.0
    raise ValueError(f"unsupported tool: {name}")


def _is_num(v: Any) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def default_lookup_table() -> Dict[str, Any]:
    return dict(_DEFAULT_TABLE)


def is_math_only_toolset(tool_names: Set[str]) -> bool:
    return bool(tool_names) and tool_names <= MATH_TOOL_NAMES
