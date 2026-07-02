#!/usr/bin/env python3
"""Prototype synthetic tool registry + local executor for v3 curriculum."""
from __future__ import annotations

from typing import Any, Dict, List, Set

MATH_TOOL_NAMES: Set[str] = {
    "add", "multiply", "subtract", "divide", "sum_list", "scale",
    "identity", "negate", "double",
}

STRING_TOOL_NAMES: Set[str] = {
    "concat", "lowercase", "uppercase", "extract_prefix", "extract_suffix",
}

LIST_TOOL_NAMES: Set[str] = {
    "get_item", "length", "filter_greater_than", "join_list", "sort_list",
}

OBJECT_TOOL_NAMES: Set[str] = {
    "get_field", "merge_objects", "make_object",
}

BOOLEAN_TOOL_NAMES: Set[str] = {
    "greater_than", "equals", "contains",
}

ALL_TOOL_NAMES: Set[str] = (
    MATH_TOOL_NAMES | STRING_TOOL_NAMES | LIST_TOOL_NAMES
    | OBJECT_TOOL_NAMES | BOOLEAN_TOOL_NAMES
)


def _tool(name: str, desc: str, params: dict, out_type: str = "number", out_key: str = "result") -> dict:
    props = {k: {"type": v, "description": k} for k, v in params.items()}
    return {
        "name": name,
        "description": desc,
        "parameters": {"type": "object", "properties": props, "required": list(params.keys())},
        "output_parameters": {out_key: {"type": out_type, "description": ""}},
    }


MATH_TOOLS = {
    "add": _tool("add", "Add two numbers", {"arg_0": "number", "arg_1": "number"}),
    "multiply": _tool("multiply", "Multiply", {"arg_0": "number", "arg_1": "number"}),
    "subtract": _tool("subtract", "Subtract", {"arg_0": "number", "arg_1": "number"}),
    "divide": _tool("divide", "Divide", {"arg_0": "number", "arg_1": "number"}),
    "sum_list": _tool("sum_list", "Sum list", {"values": "array"}),
    "scale": _tool("scale", "Scale", {"arg_0": "number", "factor": "number"}),
    "identity": _tool("identity", "Identity", {"x": "number"}),
    "negate": _tool("negate", "Negate", {"x": "number"}),
    "double": _tool("double", "Double", {"x": "number"}),
}

STRING_TOOLS = {
    "concat": _tool("concat", "Concat strings", {"a": "string", "b": "string"}, out_type="string"),
    "lowercase": _tool("lowercase", "Lowercase", {"text": "string"}, out_type="string"),
    "uppercase": _tool("uppercase", "Uppercase", {"text": "string"}, out_type="string"),
    "extract_prefix": _tool("extract_prefix", "Prefix", {"text": "string", "n": "number"}, out_type="string"),
    "extract_suffix": _tool("extract_suffix", "Suffix", {"text": "string", "n": "number"}, out_type="string"),
}

LIST_TOOLS = {
    "get_item": _tool("get_item", "Get list item", {"values": "array", "index": "number"}),
    "length": _tool("length", "List length", {"values": "array"}),
    "filter_greater_than": _tool(
        "filter_greater_than", "Filter list", {"values": "array", "threshold": "number"}, out_type="array"
    ),
    "join_list": _tool("join_list", "Join list", {"values": "array", "sep": "string"}, out_type="string"),
    "sort_list": _tool("sort_list", "Sort list", {"values": "array"}, out_type="array"),
}

OBJECT_TOOLS = {
    "get_field": _tool("get_field", "Get field", {"obj": "object", "key": "string"}),
    "merge_objects": _tool("merge_objects", "Merge objects", {"a": "object", "b": "object"}, out_type="object"),
    "make_object": _tool(
        "make_object", "Make object", {"key": "string", "value": "string"}, out_type="object"
    ),
}

BOOLEAN_TOOLS = {
    "greater_than": _tool("greater_than", "Compare", {"a": "number", "b": "number"}, out_type="boolean"),
    "equals": _tool("equals", "Equality", {"a": "string", "b": "string"}, out_type="boolean"),
    "contains": _tool("contains", "Contains", {"text": "string", "sub": "string"}, out_type="boolean"),
}


def all_tool_defs(extra_distractors: bool = True) -> List[dict]:
    tools = (
        list(MATH_TOOLS.values())
        + list(STRING_TOOLS.values())
        + list(LIST_TOOLS.values())
        + list(OBJECT_TOOLS.values())
        + list(BOOLEAN_TOOLS.values())
    )
    return tools


def tool_pool_for_families(families: List[str]) -> List[dict]:
    pool: List[dict] = []
    if "math" in families:
        pool.extend(MATH_TOOLS.values())
    if "string" in families:
        pool.extend(STRING_TOOLS.values())
    if "list" in families:
        pool.extend(LIST_TOOLS.values())
    if "object" in families:
        pool.extend(OBJECT_TOOLS.values())
    if "boolean" in families:
        pool.extend(BOOLEAN_TOOLS.values())
    if not pool:
        pool = all_tool_defs()
    seen = set()
    out = []
    for t in pool:
        n = t["name"]
        if n not in seen:
            seen.add(n)
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


def execute_tool(name: str, args: Dict[str, Any]) -> Any:
    if name == "add":
        return float(args["arg_0"]) + float(args["arg_1"])
    if name == "multiply":
        return float(args["arg_0"]) * float(args["arg_1"])
    if name == "subtract":
        return float(args["arg_0"]) - float(args["arg_1"])
    if name == "divide":
        d = float(args["arg_1"])
        if d == 0:
            raise ZeroDivisionError("divide by zero")
        return float(args["arg_0"]) / d
    if name == "sum_list":
        return float(sum(float(v) for v in (args.get("values") or [])))
    if name == "scale":
        return float(args["arg_0"]) * float(args["factor"])
    if name == "identity":
        return float(args["x"])
    if name == "negate":
        return -float(args["x"])
    if name == "double":
        return float(args["x"]) * 2
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
    if name == "get_item":
        vals = list(args.get("values") or [])
        return vals[int(args["index"])]
    if name == "length":
        return len(args.get("values") or [])
    if name == "filter_greater_than":
        thr = float(args["threshold"])
        return [v for v in (args.get("values") or []) if float(v) > thr]
    if name == "join_list":
        sep = str(args.get("sep", ","))
        return sep.join(str(v) for v in (args.get("values") or []))
    if name == "sort_list":
        return sorted(args.get("values") or [], key=lambda x: float(x))
    if name == "get_field":
        obj = args.get("obj") or {}
        return obj.get(args["key"])
    if name == "merge_objects":
        a = dict(args.get("a") or {})
        a.update(dict(args.get("b") or {}))
        return a
    if name == "make_object":
        return {str(args["key"]): args["value"]}
    if name == "greater_than":
        return float(args["a"]) > float(args["b"])
    if name == "equals":
        return str(args["a"]) == str(args["b"])
    if name == "contains":
        return str(args["sub"]) in str(args["text"])
    raise ValueError(f"unsupported tool: {name}")


def is_math_only_toolset(tool_names: Set[str]) -> bool:
    return bool(tool_names) and tool_names <= MATH_TOOL_NAMES
