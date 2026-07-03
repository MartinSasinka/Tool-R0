#!/usr/bin/env python3
"""Trace-aligned natural-language questions for v3.1 (derived from gold_calls)."""
from __future__ import annotations

import random
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from motif_lib import extract_references_from_value
from tool_registry_v3_1 import ALL_TOOL_NAMES, infer_answer_type, tool_family

LOOKUP_KEYWORDS = ("lookup", "table", "dataset", "aggregate", "records in the dataset")
RECORD_FIELD_KEYWORDS = ("record", "field", "object")
LIST_KEYWORDS = ("filter the list", "sort the list", "take item", "list from step", "list produced")
STRING_KEYWORDS = ("join the strings", "uppercase", "lowercase", "characters in the result", "convert the result")
BOOLEAN_KEYWORDS = ("logical and", "logical or", "check whether")

LOOKUP_TOOLS = {"lookup_by_key", "select_record", "count_records", "aggregate_field"}
OBJECT_TOOLS = {"make_object", "get_field", "merge_objects", "update_field", "nested_get", "select_record"}
LIST_TOOLS = {"get_item", "list_length", "filter_greater_than", "filter_equals", "join_list", "sort_list", "count_items"}
STRING_TOOLS = {"concat", "lowercase", "uppercase", "extract_prefix", "extract_suffix", "contains_substring", "string_length"}
BOOLEAN_TOOLS = {"greater_than", "less_than", "equals", "contains", "contains_substring", "and_bool", "or_bool"}

UNRESOLVED_PATTERNS = (
    re.compile(r"\badd B\b", re.I),
    re.compile(r"\bmultiply by B\b", re.I),
    re.compile(r"[{}<>]"),
    re.compile(r"\bTODO\b", re.I),
    re.compile(r"\bNone\b"),
    re.compile(r"\bnull\b", re.I),
    re.compile(r"\bprefix\b", re.I),
    re.compile(r"cluster=", re.I),
    re.compile(r"long_chain__", re.I),
    re.compile(r"synthetic_gap", re.I),
)

INCOMPLETE_PATTERNS = (
    re.compile(r"\($"),
    re.compile(r"^Separately compute\s*$", re.I),
    re.compile(r"^Evaluate two small additions\s*$", re.I),
)

STEP_COUNT_WORDS = {
    1: ("one step", "1 step", "single step"),
    2: ("two steps", "2 steps", "two step"),
    3: ("three steps", "3 steps", "three step"),
    4: ("four steps", "4 steps", "four step"),
    5: ("five steps", "5 steps", "five step"),
    6: ("six steps", "6 steps", "six step"),
}


def _refs(val: Any) -> List[Tuple[int, Optional[str]]]:
    if val is None:
        return []
    return extract_references_from_value(val)


def _ref_var(val: Any) -> Optional[int]:
    refs = _refs(val)
    return refs[0][0] if refs else None


def _has_ref(val: Any) -> bool:
    return _ref_var(val) is not None


def _result_phrase(var_idx: int, current_step: int, *, produced: bool = False, force_step: bool = False) -> str:
    if not force_step and var_idx == current_step - 1:
        return "the previous result"
    noun = "produced in" if produced else "from"
    return f"the result {noun} step {var_idx}"


def _object_phrase(var_idx: int, current_step: int) -> str:
    if var_idx == current_step - 1:
        return "the object from the previous step"
    return f"the object produced in step {var_idx}"


def _list_phrase(var_idx: int, current_step: int) -> str:
    if var_idx == current_step - 1:
        return "the list from the previous step"
    return f"the list produced in step {var_idx}"


def _format_list(vals: Any) -> str:
    if isinstance(vals, list):
        return "[" + ", ".join(str(v) for v in vals[:8]) + "]"
    return "[]"


def _format_str(val: Any) -> str:
    return f'"{val}"'


def _variant_pick(options: List[str], variant_id: int) -> str:
    return options[variant_id % max(len(options), 1)]


def _describe_binary_math(
    name: str,
    arg0_key: str,
    arg1_key: str,
    args: dict,
    step_idx: int,
    variant_id: int = 0,
) -> str:
    a0, a1 = args.get(arg0_key), args.get(arg1_key)
    r0, r1 = _ref_var(a0), _ref_var(a1)

    if name == "add":
        if r0 is not None and r1 is not None:
            if r0 == r1:
                return f"add {_result_phrase(r0, step_idx)} to itself"
            force = True
            return (
                f"add {_result_phrase(r0, step_idx, force_step=force)} "
                f"to {_result_phrase(r1, step_idx, force_step=force)}"
            )
        if r0 is not None and r1 is None:
            if r0 == step_idx - 1:
                return _variant_pick([
                    f"add {a1} to the previous result",
                    f"take the previous result and increase it by {a1}",
                    f"combine the previous result with {a1}",
                ], variant_id)
            return _variant_pick([
                f"add {a1} to {_result_phrase(r0, step_idx)}",
                f"take the result from step {r0} and increase it by {a1}",
                f"combine the result from step {r0} with {a1}",
            ], variant_id)
        if r1 is not None and r0 is None:
            return _variant_pick([
                f"add {a0} to {_result_phrase(r1, step_idx)}",
                f"take the result from step {r1} and increase it by {a0}",
                f"combine the result from step {r1} with {a0}",
            ], variant_id)
        return _variant_pick([
            f"add {a0} and {a1}",
            f"compute the sum of {a0} and {a1}",
            f"find {a0} plus {a1}",
            f"calculate {a0} + {a1}",
            f"use the addition tool to combine {a0} and {a1}",
        ], variant_id)

    if name == "subtract":
        if r0 is not None and r1 is None:
            return f"subtract {a1} from {_result_phrase(r0, step_idx)}"
        if r1 is not None and r0 is None:
            return f"subtract {_result_phrase(r1, step_idx)} from {a0}"
        if r0 is not None and r1 is not None:
            force = True
            return (
                f"subtract {_result_phrase(r1, step_idx, force_step=force)} "
                f"from {_result_phrase(r0, step_idx, force_step=force)}"
            )
        return f"subtract {a1} from {a0}"

    if name == "multiply":
        if r0 is not None and r1 is not None:
            if r0 == r1:
                if r0 == step_idx - 1:
                    return _variant_pick([
                        "multiply the previous result by itself",
                        "square the previous result",
                        "use the previous result as both inputs to multiplication",
                    ], variant_id)
                return _variant_pick([
                    f"multiply {_result_phrase(r0, step_idx)} by itself",
                    f"square the result from step {r0}",
                    f"use the result from step {r0} as both multiplication inputs",
                ], variant_id)
            force = True
            return (
                f"multiply {_result_phrase(r0, step_idx, force_step=force)} "
                f"by {_result_phrase(r1, step_idx, force_step=force)}"
            )
        if r0 is not None and r1 is None:
            if r0 == step_idx - 1:
                return f"multiply the previous result by {a1}"
            return f"multiply {_result_phrase(r0, step_idx)} by {a1}"
        if r1 is not None and r0 is None:
            return f"multiply {a0} by {_result_phrase(r1, step_idx)}"
        return f"multiply {a0} by {a1}"

    if name == "divide_safe":
        if r0 is not None and r1 is None:
            if r0 == step_idx - 1:
                return f"divide the previous result by {a1}"
            return f"divide {_result_phrase(r0, step_idx)} by {a1}"
        if r0 is not None and r1 is not None:
            return f"divide {_result_phrase(r0, step_idx)} by {_result_phrase(r1, step_idx)}"
        return f"divide {a0} by {a1}"

    return f"run {name}"


def describe_gold_call(call: dict, step_idx: int, variant_id: int = 0) -> str:
    """Deterministic natural-language instruction for one gold tool call."""
    name = call.get("name", "")
    args = call.get("arguments") or {}

    if name in ("add", "subtract", "multiply", "divide_safe"):
        return _describe_binary_math(name, "arg_0", "arg_1", args, step_idx, variant_id)

    if name == "sum_list":
        return f"sum the numbers {_format_list(args.get('values', []))}"
    if name == "mean_list":
        return f"compute the average of {_format_list(args.get('values', []))}"
    if name == "max_list":
        return f"find the maximum in {_format_list(args.get('values', []))}"
    if name == "min_list":
        return f"find the minimum in {_format_list(args.get('values', []))}"

    if name == "concat":
        return f"join the strings {_format_str(args.get('a', 'X'))} and {_format_str(args.get('b', 'Y'))}"
    if name == "lowercase":
        r = _ref_var(args.get("text"))
        if r is not None:
            return f"convert {_result_phrase(r, step_idx, produced=True)} to lowercase"
        return f"convert {_format_str(args.get('text', 'text'))} to lowercase"
    if name == "uppercase":
        r = _ref_var(args.get("text"))
        if r is not None:
            return _variant_pick([
                f"convert {_result_phrase(r, step_idx, produced=True)} to uppercase",
                f"uppercase the previous string result",
                f"take the output of step {r} and make all characters uppercase",
            ], variant_id)
        return f"convert {_format_str(args.get('text', 'text'))} to uppercase"
    if name == "extract_prefix":
        r = _ref_var(args.get("text"))
        n = args.get("n", 3)
        if r is not None:
            return f"take the first {n} characters of {_result_phrase(r, step_idx, produced=True)}"
        return f"take the first {n} characters of {_format_str(args.get('text', 'text'))}"
    if name == "extract_suffix":
        r = _ref_var(args.get("text"))
        n = args.get("n", 3)
        if r is not None:
            return f"take the last {n} characters of {_result_phrase(r, step_idx, produced=True)}"
        return f"take the last {n} characters of {_format_str(args.get('text', 'text'))}"
    if name == "string_length":
        r = _ref_var(args.get("text"))
        if r is not None:
            return f"count the number of characters in {_result_phrase(r, step_idx, produced=True)}"
        return f"count the number of characters in {_format_str(args.get('text', 'text'))}"
    if name == "contains_substring":
        r = _ref_var(args.get("text"))
        sub = args.get("sub", "part")
        if r is not None:
            return f"check whether {_result_phrase(r, step_idx, produced=True)} contains {_format_str(sub)}"
        return f"check whether {_format_str(args.get('text', 'text'))} contains {_format_str(sub)}"

    if name == "filter_greater_than":
        vals = args.get("values")
        if _has_ref(vals):
            r = _ref_var(vals)
            return f"filter {_list_phrase(r, step_idx)} to values greater than {args.get('threshold', 0)}"
        return f"filter the list {_format_list(vals)} to values greater than {args.get('threshold', 0)}"
    if name == "filter_equals":
        vals = args.get("values")
        if _has_ref(vals):
            r = _ref_var(vals)
            return f"filter {_list_phrase(r, step_idx)} to values equal to {_format_str(args.get('target', 'x'))}"
        return f"filter the list {_format_list(vals)} to values equal to {_format_str(args.get('target', 'x'))}"
    if name == "sort_list":
        r = _ref_var(args.get("values"))
        if r is not None:
            return _variant_pick([
                f"sort {_list_phrase(r, step_idx)}",
                f"order the values produced in step {r}",
                f"take the list output of step {r} and sort it",
            ], variant_id)
        return f"sort the list {_format_list(args.get('values', []))}"
    if name == "get_item":
        r = _ref_var(args.get("values"))
        idx = args.get("index", 0)
        if r is not None:
            return f"take item {idx} from {_list_phrase(r, step_idx)}"
        return f"take item {idx} from the list {_format_list(args.get('values', []))}"
    if name == "list_length":
        r = _ref_var(args.get("values"))
        if r is not None:
            return f"count how many items are in {_list_phrase(r, step_idx)}"
        return f"count how many items are in the list {_format_list(args.get('values', []))}"
    if name == "join_list":
        r = _ref_var(args.get("values"))
        sep = args.get("sep", ",")
        if r is not None:
            return f"join items from {_list_phrase(r, step_idx)} with separator {_format_str(sep)}"
        return f"join list items with separator {_format_str(sep)}"
    if name == "count_items":
        r = _ref_var(args.get("values"))
        if r is not None:
            return f"count items in {_list_phrase(r, step_idx)}"
        return "count items in the list"

    if name == "make_object":
        return (
            f"create an object with field {_format_str(args.get('key', 'key'))} "
            f"set to {_format_str(args.get('value', 'val'))}"
        )
    if name == "get_field":
        r = _ref_var(args.get("obj"))
        key = args.get("key", "key")
        if r is not None:
            return _variant_pick([
                f"read field {_format_str(key)} from {_object_phrase(r, step_idx)}",
                f"extract the {_format_str(key)} property from the previous object",
                f"use the object from step {r} and return its {_format_str(key)} field",
            ], variant_id)
        return f"read field {_format_str(key)} from the object"
    if name == "merge_objects":
        r0, r1 = _ref_var(args.get("a")), _ref_var(args.get("b"))
        if r0 is not None and r1 is not None:
            return f"merge {_object_phrase(r0, step_idx)} and {_object_phrase(r1, step_idx)}"
        return "merge the two objects"
    if name == "update_field":
        r = _ref_var(args.get("obj"))
        key, val = args.get("key", "key"), args.get("value", "val")
        if r is not None:
            return f"set field {_format_str(key)} to {_format_str(val)} on {_object_phrase(r, step_idx)}"
        return f"set field {_format_str(key)} to {_format_str(val)} on the object"
    if name == "nested_get":
        r = _ref_var(args.get("obj"))
        path = args.get("path", "path")
        if r is not None:
            return f"read nested path {_format_str(path)} from {_object_phrase(r, step_idx)}"
        return f"read nested path {_format_str(path)} from the object"

    if name == "greater_than":
        a, b = args.get("a", "A"), args.get("b", "B")
        return _variant_pick([
            f"check whether {a} is greater than {b}",
            f"compare {a} with {b} using greater-than",
            f"return whether {a} exceeds {b}",
        ], variant_id)
    if name == "less_than":
        return f"check whether {args.get('a', 'A')} is less than {args.get('b', 'B')}"
    if name == "equals":
        return f"check whether {_format_str(args.get('a', 'A'))} equals {_format_str(args.get('b', 'B'))}"
    if name == "contains":
        return (
            f"check whether {_format_str(args.get('text', 'text'))} "
            f"contains {_format_str(args.get('sub', 'part'))}"
        )
    if name == "and_bool":
        r0, r1 = _ref_var(args.get("a")), _ref_var(args.get("b"))
        if r0 is not None and r1 is not None:
            return f"apply logical AND to the results from step {r0} and step {r1}"
        return "apply logical AND to the two boolean values"
    if name == "or_bool":
        r0, r1 = _ref_var(args.get("a")), _ref_var(args.get("b"))
        if r0 is not None and r1 is not None:
            return f"apply logical OR to the results from step {r0} and step {r1}"
        return "apply logical OR to the two boolean values"

    if name == "lookup_by_key":
        return f"look up key {_format_str(args.get('key', 'key'))} in the table"
    if name == "select_record":
        return f"select record at index {args.get('index', 0)}"
    if name == "count_records":
        return "count records in the dataset"
    if name == "aggregate_field":
        return f"sum field {_format_str(args.get('field', 'score'))} across records"

    return f"run tool {name}"


def _critical_markers(step: str) -> List[str]:
    """Substrings that must appear in the composed question."""
    markers: List[str] = []
    sl = step.lower()
    if "by itself" in sl or "square" in sl:
        markers.extend(["by itself", "square"])
    m = re.search(r"by (\d+)", sl)
    if m:
        markers.append(f"by {m.group(1)}")
    for sm in re.findall(r"step (\d+)", sl):
        markers.append(f"step {sm}")
    if "previous result" in sl:
        markers.append("previous result")
    if "logical and" in sl:
        markers.append("logical and")
    if "logical or" in sl:
        markers.append("logical or")
    if "greater than" in sl and "check whether" in sl:
        markers.append("greater than")
    if "filter the list" in sl or "filter [" in sl:
        markers.append("filter")
    if "join the strings" in sl:
        markers.append("join the strings")
    if "uppercase" in sl:
        markers.append("uppercase")
    if "lowercase" in sl:
        markers.append("lowercase")
    if "characters" in sl:
        markers.append("characters")
    if "sort" in sl:
        markers.append("sort")
    if "take item" in sl:
        markers.append("take item")
    if "create an object" in sl:
        markers.append("create an object")
    if "read field" in sl:
        markers.append("read field")
    if "look up key" in sl:
        markers.append("look up")
    # Always require the primary verb from the step
    first_word = sl.split()[0] if sl.split() else ""
    if first_word in (
        "add", "subtract", "multiply", "divide", "sum", "compute", "find",
        "join", "convert", "take", "count", "filter", "sort", "create",
        "read", "set", "merge", "check", "apply", "look", "select",
        "calculate", "combine", "square", "uppercase", "order", "extract",
        "compare", "return",
    ):
        markers.append(first_word)
    if "sum of" in sl:
        markers.append("sum")
    if " plus " in sl:
        markers.append("plus")
    if "square" in sl:
        markers.append("square")
    if "combine" in sl:
        markers.append("combine")
    return markers


def _compose_question(steps: List[str], rng: random.Random, n_calls: int, variant_id: int = 0) -> str:
    if n_calls == 1:
        return _variant_pick([
            f"Using the tools, {steps[0]}. What is the result?",
            f"Please {steps[0]} and report the answer.",
            f"Task: {steps[0][0].upper()}{steps[0][1:]}. Return the outcome.",
            f"Your job is to {steps[0]}. Provide the answer.",
            f"Complete this single-step task: {steps[0]}. Return the result.",
        ], variant_id)
    if n_calls == 2:
        return _variant_pick([
            f"First, {steps[0]}. Then, {steps[1]}. Return the value from the second step.",
            f"Complete two steps: (1) {steps[0]}; (2) {steps[1]}. Give the last result.",
            f"Start by doing this: {steps[0]}. Next, {steps[1]}. Report the final value.",
            f"Perform two operations in order. Step one: {steps[0]}. Step two: {steps[1]}. Return step two's result.",
        ], variant_id)
    if n_calls == 3:
        return _variant_pick([
            f"Follow three steps: first {steps[0]}; then {steps[1]}; finally {steps[2]}. Return the last result.",
            f"Step 1: {steps[0]}. Step 2: {steps[1]}. Step 3: {steps[2]}. What is the outcome after step 3?",
            f"Complete a three-step task: {steps[0]}; next {steps[1]}; finally {steps[2]}. Give the final answer.",
        ], variant_id)
    numbered = "; ".join(f"step {i + 1}: {s}" for i, s in enumerate(steps))
    closers = [
        "Return the value from the final step.",
        "What is the result after the last operation?",
        "Report the outcome of the final tool call.",
        "Give the answer from the last step.",
        "Provide the final computed value.",
    ]
    openers = [
        f"Execute this {n_calls}-step workflow: {numbered}.",
        f"Perform these {n_calls} operations in order: {numbered}.",
        f"Work through all {n_calls} steps sequentially: {numbered}.",
        f"Complete each of the {n_calls} tool calls: {numbered}.",
        f"Carry out the following {n_calls} steps: {numbered}.",
    ]
    return f"{_variant_pick(openers, variant_id)} {_variant_pick(closers, variant_id + 3)}"


def render_question_from_gold_calls(
    gold_calls: List[dict],
    observations: Optional[List[Any]] = None,
    stage: str = "",
    motif_type: str = "",
    *,
    seed: int = 0,
    rng: Optional[random.Random] = None,
) -> str:
    """Render a question that exactly describes the gold call sequence."""
    del observations, stage, motif_type  # reserved for future context-aware wording
    rng = rng or random.Random(seed ^ 0xC0FFEE)
    variant_id = (seed ^ 0xDEADBEEF) & 0xFFFF
    steps = [describe_gold_call(c, i + 1, variant_id=(variant_id + i * 17) & 0xFFFF) for i, c in enumerate(gold_calls)]
    q = _compose_question(steps, rng, len(gold_calls), variant_id=variant_id)
    if len(q) < 25:
        q = f"{q} Use the tools to compute the exact answer."
    return q


def question_for_trajectory(rng: random.Random, calls: List[dict], *, seed: int = 0) -> str:
    return render_question_from_gold_calls(calls, seed=seed, rng=random.Random(seed ^ 0x9E3779B9))


def question_for_prefix(
    rng: random.Random,
    calls: List[dict],
    prefix_len: int,
    stage: str,
    *,
    seed: int = 0,
    motif_type: str = "",
    observations: Optional[List[Any]] = None,
) -> str:
    return render_question_from_gold_calls(
        calls,
        observations,
        stage,
        motif_type,
        seed=seed ^ (prefix_len * 7919),
        rng=random.Random(seed ^ (prefix_len * 7919) ^ 0xC0FFEE),
    )


def stage2_task_category(sample: dict) -> str:
    calls = sample.get("gold_calls") or []
    labels = sample.get("process_labels") or []
    if len(calls) < 2:
        return "independent"
    if len(labels) > 1 and labels[1] == "reference_step":
        return "reference_dependency"
    if tool_family(calls[1].get("name", "")) in ("string", "list", "object", "boolean"):
        return "transform"
    return "independent"


def is_non_scalar_answer(answer: Any) -> bool:
    return infer_answer_type(answer) not in ("scalar", "unknown")


def question_signature(sample: dict) -> str:
    calls = sample.get("gold_calls") or []
    tools = "->".join(c.get("name", "") for c in calls)
    return f"{sample.get('question', '')}|{tools}|{sample.get('num_calls', 0)}"


def is_incomplete_question(q: str) -> bool:
    q = (q or "").strip()
    if len(q) < 25:
        return True
    for pat in INCOMPLETE_PATTERNS:
        if pat.search(q):
            return True
    if q.endswith("(") or q.endswith(","):
        return True
    return False


def _mentions_any(q: str, keywords: tuple) -> bool:
    ql = q.lower()
    return any(k in ql for k in keywords)


def check_unresolved_placeholders(question: str) -> List[str]:
    errors: List[str] = []
    for pat in UNRESOLVED_PATTERNS:
        if pat.search(question or ""):
            errors.append("unresolved_placeholder")
            break
    return errors


def check_constant_reference_consistency(question: str, gold_calls: List[dict]) -> List[str]:
    errors: List[str] = []
    ql = (question or "").lower()
    for i, call in enumerate(gold_calls, start=1):
        name = call.get("name", "")
        args = call.get("arguments") or {}
        if name == "multiply":
            r0, r1 = _ref_var(args.get("arg_0")), _ref_var(args.get("arg_1"))
            a1 = args.get("arg_1")
            if r0 is not None and r1 is not None and r0 == r1:
                if not any(
                    p in ql
                    for p in (
                        "by itself",
                        "square",
                        "both inputs to multiplication",
                        "both multiplication inputs",
                    )
                ):
                    errors.append("constant_reference_mismatch:multiply_self")
            elif r0 is not None and r1 is None and isinstance(a1, (int, float)):
                if "by itself" in ql or "square" in ql:
                    errors.append("constant_reference_mismatch:multiply_literal_as_self")
                elif f"by {a1}" not in ql:
                    errors.append(f"constant_reference_mismatch:multiply_missing_literal_{a1}")
            elif r0 is not None and r1 is not None and r0 != r1:
                if f"step {r0}" not in ql and f"step {r1}" not in ql and "previous result" not in ql:
                    errors.append("constant_reference_mismatch:multiply_two_refs")
        if name == "add":
            r0, r1 = _ref_var(args.get("arg_0")), _ref_var(args.get("arg_1"))
            if r0 is not None and r1 is not None:
                if f"step {r0}" not in ql and f"step {r1}" not in ql and "previous result" not in ql:
                    errors.append("constant_reference_mismatch:add_two_refs")
            elif (r0 is not None or r1 is not None) and re.search(r"\badd B\b", ql):
                errors.append("constant_reference_mismatch:add_placeholder")
        if name in ("uppercase", "lowercase", "string_length") and _has_ref(args.get("text")):
            r = _ref_var(args.get("text"))
            if r is not None and f"step {r}" not in ql and "previous" not in ql and "output of step" not in ql:
                errors.append(f"constant_reference_mismatch:{name}_missing_step_ref")
        if name == "greater_than":
            a, b = args.get("a"), args.get("b")
            if a is not None and str(a) in ql and str(b) in ql:
                if not any(w in ql for w in ("greater than", "exceeds", "compare")):
                    errors.append("constant_reference_mismatch:greater_than_wording")
            continue
        if name in ("and_bool", "or_bool"):
            r0, r1 = _ref_var(args.get("a")), _ref_var(args.get("b"))
            if r0 is not None and r1 is not None:
                if f"step {r0}" not in ql or f"step {r1}" not in ql:
                    errors.append(f"constant_reference_mismatch:{name}_missing_steps")
    return errors


def check_step_count_consistency(question: str, gold_calls: List[dict], num_calls: Optional[int] = None) -> List[str]:
    n = num_calls if num_calls is not None else len(gold_calls)
    if n <= 0:
        return ["step_count_mismatch:empty"]
    ql = (question or "").lower()
    explicit = len(set(re.findall(r"step (\d+)", ql)))
    two_step_markers = (
        "two steps", "2 steps", "first,", "first ", "next,", " then,", "finally,",
        "(1)", "(2)", "step 1:", "step 2:", "complete two steps",
        "step one", "step two", "two operations", "operations in order",
    )
    three_step_markers = (
        "three steps", "3 steps", "step 1:", "step 2:", "step 3:", "follow three steps",
        "three-step", "a three-step task",
    )
    if n == 1:
        if explicit > 1:
            return ["step_count_mismatch:stage1_too_many_steps"]
        return []
    if n == 2:
        ok = explicit >= 1 or any(w in ql for w in two_step_markers)
        if not ok:
            return ["step_count_mismatch:stage2"]
        return []
    if n == 3:
        ok = explicit >= 2 or any(w in ql for w in three_step_markers)
        if not ok:
            return ["step_count_mismatch:stage3"]
        return []
    if n >= 4:
        ok = (
            explicit >= n - 1
            or any(w in ql for w in STEP_COUNT_WORDS.get(n, ()))
            or f"{n}-step" in ql
            or ql.count("step ") >= n - 1
        )
        if not ok:
            return [f"step_count_mismatch:stage4_n{n}"]
    return []


def check_operation_coverage(question: str, gold_calls: List[dict]) -> List[str]:
    """Question must mention enough operations to reconstruct the trace."""
    ql = (question or "").lower()
    missing = 0
    for i, call in enumerate(gold_calls):
        markers: Set[str] = set()
        for vid in range(12):
            step = describe_gold_call(call, i + 1, variant_id=vid)
            for m in _critical_markers(step):
                markers.add(m.lower())
        if not any(m in ql for m in markers if m):
            missing += 1
    if missing > 0:
        return [f"insufficient_operation_coverage:{missing}"]
    return []


def validate_question_trace_alignment(
    question: str,
    gold_calls: List[dict],
    *,
    num_calls: Optional[int] = None,
) -> List[str]:
    errors: List[str] = []
    q = question or ""
    tools = {str(c.get("name", "")) for c in gold_calls}
    fams = {tool_family(n) for n in tools if n}

    errors.extend(check_unresolved_placeholders(q))
    errors.extend(check_constant_reference_consistency(q, gold_calls))
    errors.extend(check_step_count_consistency(q, gold_calls, num_calls))
    errors.extend(check_operation_coverage(q, gold_calls))

    if is_incomplete_question(q):
        errors.append("incomplete_or_short_question")

    if _mentions_any(q, LOOKUP_KEYWORDS) and not (tools & LOOKUP_TOOLS):
        errors.append("lookup_keywords_without_lookup_tools")

    if _mentions_any(q, RECORD_FIELD_KEYWORDS) and not (tools & (OBJECT_TOOLS | LOOKUP_TOOLS)):
        errors.append("record_field_object_keywords_without_object_or_lookup_tools")

    if _mentions_any(q, LIST_KEYWORDS) and not (tools & LIST_TOOLS):
        errors.append("list_keywords_without_list_tools")

    if _mentions_any(q, STRING_KEYWORDS) and not (tools & STRING_TOOLS):
        errors.append("string_keywords_without_string_tools")

    if _mentions_any(q, BOOLEAN_KEYWORDS) and not (tools & BOOLEAN_TOOLS):
        errors.append("boolean_keywords_without_boolean_tools")

    if fams <= {"math"}:
        if _mentions_any(q, LOOKUP_KEYWORDS):
            errors.append("math_trace_with_non_math_wording")
        if _mentions_any(q, ("filter the list", "sort the list", "uppercase", "lowercase", "join the strings", "logical and", "logical or")):
            errors.append("math_trace_with_non_math_wording")
        if "create an object" in q.lower() or "read field" in q.lower():
            errors.append("math_trace_with_non_math_wording")

    return list(dict.fromkeys(errors))


def compute_tool_usage_stats(samples: List[dict]) -> dict:
    """Report offered vs used tool diversity across samples."""
    from collections import Counter

    offered: Set[str] = set()
    used: Counter = Counter()
    used_by_stage: Dict[str, Counter] = {}
    families_used: Set[str] = set()

    for sample in samples:
        stage = sample.get("stage", "unknown")
        stage_used = used_by_stage.setdefault(stage, Counter())
        for tl in sample.get("tools") or []:
            if isinstance(tl, dict) and tl.get("name"):
                offered.add(str(tl["name"]))
        for call in sample.get("gold_calls") or []:
            name = str(call.get("name", ""))
            if name:
                used[name] += 1
                stage_used[name] += 1
                families_used.add(tool_family(name))

    return {
        "offered_tool_diversity": len(offered),
        "used_tool_diversity": len(used),
        "used_tool_family_count": len(families_used),
        "used_tool_distribution": dict(used.most_common()),
        "used_tool_distribution_per_stage": {
            k: dict(v.most_common()) for k, v in used_by_stage.items()
        },
        "offered_tool_names": sorted(offered),
        "used_tool_names": sorted(used.keys()),
    }
