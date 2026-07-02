"""
API-Bank data loader.

Downloads pre-processed evaluation data from HuggingFace
(liminghao1630/API-Bank) and converts to a format compatible
with the Tool-R0 evaluation pipeline.

Level 1 ("given-desc"): API descriptions are provided; model
generates the correct API call.  This is the standard single-turn
evaluation setting and aligns with Tool-R0's training objective.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple


HF_REPO = "liminghao1630/API-Bank"
LEVEL1_API_FILE = "test-data/level-1-api.json"


def _download_data(cache_dir: Optional[str] = None) -> str:
    """Download level-1-api.json from HuggingFace and return local path."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=HF_REPO,
        filename=LEVEL1_API_FILE,
        repo_type="dataset",
        cache_dir=cache_dir,
    )


def _parse_api_descriptions(instruction: str) -> List[Dict[str, Any]]:
    """Extract JSON API descriptions from the instruction field.

    API-Bank embeds one JSON object per line inside the instruction
    after the 'API descriptions:' marker.  We convert each to an
    OpenAI-style function schema for Tool-R0.
    """
    marker = "API descriptions:\n"
    idx = instruction.find(marker)
    if idx < 0:
        marker = "API descriptions:"
        idx = instruction.find(marker)
    if idx < 0:
        return []

    desc_block = instruction[idx + len(marker) :]
    tools: List[Dict[str, Any]] = []

    for line in desc_block.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            api_info = json.loads(line)
        except json.JSONDecodeError:
            continue

        props: Dict[str, Any] = {}
        required: List[str] = []
        for pname, pinfo in api_info.get("input_parameters", {}).items():
            props[pname] = {
                "type": "string",
                "description": pinfo.get("description", ""),
            }
            required.append(pname)

        tool = {
            "type": "function",
            "function": {
                "name": api_info["name"],
                "description": api_info.get("description", ""),
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": required,
                },
            },
        }
        tools.append(tool)
    return tools


_APIBANK_CALL_RE = re.compile(
    r"\[(\w+)\((.*?)\)\]", re.DOTALL
)

_PARAM_RE = re.compile(
    r"""(\w+)\s*=\s*(?:'([^']*)'|"([^"]*)"|(\[.*?\])|(\w+))""",
    re.DOTALL,
)


def _parse_ground_truth(expected_output: str) -> Tuple[Optional[str], Dict[str, str]]:
    """Parse the ground truth from API-Bank expected_output format.

    Format: API-Request: [ApiName(key1='value1', key2='value2', ...)]
    Returns: (api_name, param_dict)
    """
    m = _APIBANK_CALL_RE.search(expected_output)
    if not m:
        return None, {}

    api_name = m.group(1)
    params_str = m.group(2)

    param_dict: Dict[str, str] = {}
    for pm in _PARAM_RE.finditer(params_str):
        key = pm.group(1)
        value = pm.group(2) or pm.group(3) or pm.group(4) or pm.group(5) or ""
        param_dict[key] = value

    return api_name, param_dict


def _extract_conversation(input_text: str) -> str:
    """Extract the user-facing conversation from the input field.

    The input field contains lines like:
      User: ...
      AI: ...
      API-Request: [...]->result
      Generate API Request:

    We strip the trailing 'Generate API Request:' instruction since
    our model gets its own system prompt.
    """
    text = input_text.strip()
    if text.endswith("Generate API Request:"):
        text = text[: -len("Generate API Request:")].rstrip()
    elif text.endswith("Generate API Request:\n"):
        text = text[: -len("Generate API Request:\n")].rstrip()
    return text


def load_tasks(
    max_tasks: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load API-Bank Level 1 evaluation tasks.

    Returns list of dicts, each with:
        task_id:          str
        question:         str  (conversation context)
        tools:            list (OpenAI-style function schemas)
        ground_truth_name: str
        ground_truth_args: dict
        expected_output:  str  (raw API-Bank expected output)
        source_file:      str
    """
    local_path = _download_data(cache_dir)
    with open(local_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    tasks = []
    for entry in raw_data:
        gt_name, gt_args = _parse_ground_truth(entry["expected_output"])
        if gt_name is None:
            continue

        tools = _parse_api_descriptions(entry["instruction"])
        conversation = _extract_conversation(entry["input"])

        tasks.append({
            "task_id": f"apibank-lv1-{entry['id']}",
            "question": conversation,
            "tools": tools,
            "ground_truth_name": gt_name,
            "ground_truth_args": gt_args,
            "expected_output": entry["expected_output"],
            "source_file": entry.get("file", ""),
        })

    if max_tasks is not None:
        tasks = tasks[:max_tasks]

    return tasks


def build_user_content(task: Dict[str, Any]) -> str:
    """Build the user prompt from a loaded API-Bank task.

    Mirrors the approach used for BFCL: tool descriptions in a
    structured block followed by the conversation context.
    """
    tool_block_lines = ["Available tools:"]
    for tool in task["tools"]:
        fn = tool["function"]
        params = fn["parameters"]
        param_descs = []
        for pname, pinfo in params.get("properties", {}).items():
            param_descs.append(f'    - {pname} ({pinfo.get("type", "string")}): {pinfo.get("description", "")}')
        params_str = "\n".join(param_descs) if param_descs else "    (no parameters)"
        tool_block_lines.append(
            f'  {fn["name"]}: {fn["description"]}\n'
            f'  Parameters:\n{params_str}'
        )

    tools_text = "\n\n".join(tool_block_lines)

    return (
        f"{tools_text}\n\n"
        f"Based on the conversation below, generate the appropriate tool call.\n"
        f"The current year is 2023.\n\n"
        f"{task['question']}"
    )
