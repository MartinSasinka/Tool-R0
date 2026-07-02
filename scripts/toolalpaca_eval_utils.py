import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ensure_repo_on_path() -> None:
    import sys

    root = _repo_root()
    if root not in sys.path:
        sys.path.insert(0, root)


ensure_repo_on_path()

from rewards_solver import (  # noqa: E402
    compute_accuracy_score,
    extract_solver_fields,
    normalize_tool_call,
    parse_solver_tool_calls,
)


FENCED_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)
TOOL_CALL_BLOCK_RE = re.compile(
    r"<tool_call_answer>(.*?)</tool_call_answer>",
    re.IGNORECASE | re.DOTALL,
)


def canonicalize_call_list(calls: List[Dict[str, Any]]) -> str:
    return json.dumps(calls, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def exact_canonical_match(predicted_calls: List[Dict[str, Any]], gold_calls: List[Dict[str, Any]]) -> bool:
    return canonicalize_call_list(predicted_calls) == canonicalize_call_list(gold_calls)


def _loads_relaxed(text: str) -> Optional[Any]:
    if not isinstance(text, str):
        return None

    s = text.strip()
    if not s:
        return None

    try:
        return json.loads(s)
    except Exception:
        pass

    fenced = FENCED_BLOCK_RE.findall(s)
    for block in fenced:
        try:
            return json.loads(block.strip())
        except Exception:
            continue

    return None


def _parse_action_input_relaxed(action_input: Any) -> Optional[Dict[str, Any]]:
    if isinstance(action_input, dict):
        return action_input

    if not isinstance(action_input, str):
        return None

    parsed = _loads_relaxed(action_input)
    if isinstance(parsed, dict):
        return parsed

    # ToolAlpaca can contain templated placeholders like:
    # {"animeId": ${animeId from searchAnime}}
    # Replace `${...}` with JSON strings and retry.
    normalized = re.sub(
        r"\$\{[^{}]+\}",
        lambda m: json.dumps(m.group(0), ensure_ascii=False),
        action_input,
    )
    parsed = _loads_relaxed(normalized)
    if isinstance(parsed, dict):
        return parsed

    # Last-resort fallback: keep original payload as a raw string argument
    # instead of failing the whole benchmark run on one malformed example.
    raw = action_input.strip()
    return {"_raw_action_input": raw} if raw else None


def convert_action_style_call(obj: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(obj, dict):
        return None

    action = obj.get("Action")
    if not isinstance(action, str) or not action.strip():
        return None

    action_input = _parse_action_input_relaxed(obj.get("Action_Input", {}))
    if not isinstance(action_input, dict):
        return None

    return {"name": action, "arguments": action_input}


def normalize_call_with_action_fallback(obj: Any) -> Optional[Dict[str, Any]]:
    call = normalize_tool_call(obj)
    if call is not None:
        return call
    return convert_action_style_call(obj)


def parse_tool_call_list_with_action_fallback(obj: Any) -> Optional[List[Dict[str, Any]]]:
    if isinstance(obj, list):
        out: List[Dict[str, Any]] = []
        for item in obj:
            norm = normalize_call_with_action_fallback(item)
            if norm is not None:
                out.append(norm)
        return out if out else None

    norm = normalize_call_with_action_fallback(obj)
    return [norm] if norm is not None else None


def parse_model_prediction(raw_text: str) -> Tuple[Optional[List[Dict[str, Any]]], str]:
    if not raw_text or not raw_text.strip():
        return None, "empty_output"

    fields = extract_solver_fields(raw_text)
    if fields is not None:
        parsed = parse_solver_tool_calls(fields["tool_call_answer"])
        if parsed is not None:
            return parsed, "parsed_tool_call_tag"

        obj = _loads_relaxed(fields["tool_call_answer"])
        parsed = parse_tool_call_list_with_action_fallback(obj)
        if parsed is not None:
            return parsed, "parsed_tool_call_tag_with_action_fallback"

    parsed = parse_solver_tool_calls(raw_text)
    if parsed is not None:
        return parsed, "parsed_raw_output"

    block = TOOL_CALL_BLOCK_RE.search(raw_text)
    if block:
        obj = _loads_relaxed(block.group(1))
        parsed = parse_tool_call_list_with_action_fallback(obj)
        if parsed is not None:
            return parsed, "parsed_regex_tool_call_block"

    fenced = FENCED_BLOCK_RE.findall(raw_text)
    for block_text in fenced:
        obj = _loads_relaxed(block_text)
        parsed = parse_tool_call_list_with_action_fallback(obj)
        if parsed is not None:
            return parsed, "parsed_fenced_json"

    obj = _loads_relaxed(raw_text)
    parsed = parse_tool_call_list_with_action_fallback(obj)
    if parsed is not None:
        return parsed, "parsed_action_fallback"

    return None, "unparseable_prediction"


def toolalpaca_gold_to_canonical(gold_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for call in gold_calls:
        norm = convert_action_style_call(call)
        if norm is None:
            raise ValueError(f"Invalid ToolAlpaca gold call: {call!r}")
        out.append(norm)
    return out


def _schema_to_param_spec(schema: Optional[Dict[str, Any]], fallback_description: str = "") -> Dict[str, Any]:
    schema = schema or {}
    spec: Dict[str, Any] = {}
    if "type" in schema:
        spec["type"] = schema.get("type")
    if "description" in schema and schema.get("description"):
        spec["description"] = schema.get("description")
    elif fallback_description:
        spec["description"] = fallback_description
    if "enum" in schema and isinstance(schema.get("enum"), list):
        spec["enum"] = schema.get("enum")
    return spec


def build_tools_from_openapi(doc_str: str) -> List[Dict[str, Any]]:
    doc = json.loads(doc_str)
    paths = doc.get("paths", {})
    tools: List[Dict[str, Any]] = []

    for path_name, methods in paths.items():
        if not isinstance(methods, dict):
            continue

        for method_name, op in methods.items():
            if not isinstance(op, dict):
                continue

            operation_id = op.get("operationId")
            if not isinstance(operation_id, str) or not operation_id.strip():
                continue

            properties: Dict[str, Any] = {}
            required: List[str] = []

            for param in op.get("parameters", []) or []:
                if not isinstance(param, dict):
                    continue
                name = param.get("name")
                if not isinstance(name, str) or not name:
                    continue
                schema = param.get("schema", {})
                properties[name] = _schema_to_param_spec(schema, fallback_description=param.get("description", ""))
                if param.get("required") is True:
                    required.append(name)

            request_body = op.get("requestBody")
            if isinstance(request_body, dict):
                content = request_body.get("content", {})
                if isinstance(content, dict):
                    json_body = content.get("application/json", {})
                    if isinstance(json_body, dict):
                        body_schema = json_body.get("schema", {})
                        body_props = body_schema.get("properties", {}) if isinstance(body_schema, dict) else {}
                        for name, spec in body_props.items():
                            if not isinstance(name, str):
                                continue
                            if isinstance(spec, dict):
                                properties[name] = _schema_to_param_spec(spec)
                        for req_name in body_schema.get("required", []) if isinstance(body_schema, dict) else []:
                            if isinstance(req_name, str):
                                required.append(req_name)

            tool = {
                "name": operation_id,
                "description": op.get("description") or op.get("summary") or f"{method_name.upper()} {path_name}",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": sorted(set(required)),
                },
                "required": sorted(set(required)),
            }
            tools.append(tool)

    return tools


def load_toolalpaca_examples(dataset_path: str) -> List[Dict[str, Any]]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Dataset must be a JSON list.")

    if data and isinstance(data[0], dict) and "Instructions" in data[0] and "Golden_Answers" in data[0]:
        examples: List[Dict[str, Any]] = []
        for api_idx, entry in enumerate(data):
            instructions = entry.get("Instructions", [])
            gold_answers = entry.get("Golden_Answers", [])
            if not isinstance(instructions, list) or not isinstance(gold_answers, list):
                continue

            tools = build_tools_from_openapi(entry["Documentation"])
            for inst_idx, (instruction, gold) in enumerate(zip(instructions, gold_answers)):
                if not isinstance(instruction, str) or not isinstance(gold, list):
                    continue
                examples.append(
                    {
                        "example_id": f"{api_idx}:{inst_idx}",
                        "api_name": entry.get("Name", "unknown"),
                        "question": instruction,
                        "tools": tools,
                        "gold_calls": toolalpaca_gold_to_canonical(gold),
                        "source_record": {
                            "name": entry.get("Name"),
                            "category": entry.get("Category"),
                            "documentation_title": entry.get("Description"),
                        },
                    }
                )
        return examples

    required_keys = {"question", "tools", "gold_calls"}
    if data and isinstance(data[0], dict) and required_keys.issubset(data[0].keys()):
        return data

    raise ValueError(
        "Unsupported dataset format. Expected official ToolAlpaca eval JSON "
        "with Instructions/Golden_Answers or a flattened list of {question, tools, gold_calls}."
    )


def evaluate_prediction(predicted_calls: Optional[List[Dict[str, Any]]], gold_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    if predicted_calls is None:
        predicted_calls = []

    score, diagnostics = compute_accuracy_score(predicted_calls, gold_calls)
    return {
        "exact_match": exact_canonical_match(predicted_calls, gold_calls),
        "soft_score": score,
        "diagnostics": diagnostics,
    }

