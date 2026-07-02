"""
BFCL dataset loader.

Downloads Berkeley Function Calling Leaderboard data from HuggingFace
and prepares tasks in a format compatible with the Tool-R0 eval pipeline.

Dataset: gorilla-llm/Berkeley-Function-Calling-Leaderboard
"""

from __future__ import annotations

import ast
import json
import os
import re
from typing import Any, Dict, List, Optional

REPO_ID = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"

CATEGORY_FILES = {
    # AST categories (ground-truth matching)
    "simple": "BFCL_v3_simple.json",
    "multiple": "BFCL_v3_multiple.json",
    "parallel": "BFCL_v3_parallel.json",
    "parallel_multiple": "BFCL_v3_parallel_multiple.json",
    "irrelevance": "BFCL_v3_irrelevance.json",
    # Executable categories (real function execution)
    "exec_simple": "BFCL_v3_exec_simple.json",
    "exec_multiple": "BFCL_v3_exec_multiple.json",
    "exec_parallel": "BFCL_v3_exec_parallel.json",
    "exec_parallel_multiple": "BFCL_v3_exec_parallel_multiple.json",
}

ANSWER_FILES = {
    "simple": "possible_answer/BFCL_v3_simple.json",
    "multiple": "possible_answer/BFCL_v3_multiple.json",
    "parallel": "possible_answer/BFCL_v3_parallel.json",
    "parallel_multiple": "possible_answer/BFCL_v3_parallel_multiple.json",
    "exec_simple": "possible_answer/BFCL_v3_exec_simple.json",
    "exec_multiple": "possible_answer/BFCL_v3_exec_multiple.json",
    "exec_parallel": "possible_answer/BFCL_v3_exec_parallel.json",
    "exec_parallel_multiple": "possible_answer/BFCL_v3_exec_parallel_multiple.json",
}


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def _download_file(filename: str, cache_dir: Optional[str] = None) -> str:
    from huggingface_hub import hf_hub_download

    kwargs: Dict[str, Any] = {
        "repo_id": REPO_ID,
        "filename": filename,
        "repo_type": "dataset",
    }
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    return hf_hub_download(**kwargs)


def _convert_function_docs(functions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert BFCL function docs to Tool-R0 training format.

    BFCL uses "type": "dict" in parameters; Tool-R0 training used "type": "object".
    """
    converted = []
    for func in functions:
        func = json.loads(json.dumps(func))  # deep copy
        params = func.get("parameters", {})
        if params.get("type") == "dict":
            params["type"] = "object"
        converted.append(func)
    return converted


def _extract_question(question_field: Any) -> str:
    """Extract user question from BFCL nested format.

    BFCL uses [[{"role": "user", "content": "..."}]] for single-turn.
    """
    if isinstance(question_field, list):
        for turn in question_field:
            if isinstance(turn, list):
                for msg in turn:
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        return msg.get("content", "")
            elif isinstance(turn, dict) and turn.get("role") == "user":
                return turn.get("content", "")
    if isinstance(question_field, str):
        return question_field
    return str(question_field)


def _parse_exec_ground_truth(gt_strings: List[str]) -> List[Dict[str, Any]]:
    """Parse exec ground truth from Python call strings to BFCL dict format.

    Input:  ['calc_binomial_probability(n=20, k=5, p=0.6)']
    Output: [{'calc_binomial_probability': {'n': 20, 'k': 5, 'p': 0.6}}]
    """
    results = []
    for call_str in gt_strings:
        call_str = call_str.strip()
        if not call_str:
            continue
        try:
            tree = ast.parse(call_str, mode="eval")
            if not isinstance(tree.body, ast.Call):
                continue
            call_node = tree.body

            if isinstance(call_node.func, ast.Attribute):
                parts = []
                node = call_node.func
                while isinstance(node, ast.Attribute):
                    parts.append(node.attr)
                    node = node.value
                if isinstance(node, ast.Name):
                    parts.append(node.id)
                func_name = ".".join(reversed(parts))
            elif isinstance(call_node.func, ast.Name):
                func_name = call_node.func.id
            else:
                continue

            args = {}
            for kw in call_node.keywords:
                try:
                    args[kw.arg] = ast.literal_eval(kw.value)
                except (ValueError, TypeError):
                    args[kw.arg] = ast.dump(kw.value)

            for i, arg_node in enumerate(call_node.args):
                try:
                    args[f"_positional_{i}"] = ast.literal_eval(arg_node)
                except (ValueError, TypeError):
                    args[f"_positional_{i}"] = ast.dump(arg_node)

            results.append({func_name: args})
        except SyntaxError:
            m = re.match(r"(\w[\w.]*)\((.*)\)$", call_str, re.DOTALL)
            if m:
                results.append({m.group(1): {"_raw": m.group(2)}})
    return results


def load_tasks(
    categories: List[str],
    max_tasks_per_category: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load BFCL tasks for given categories.

    Returns a flat list of task dicts, each containing:
        task_id, category, question, functions, ground_truth
    """
    tasks: List[Dict[str, Any]] = []

    for category in categories:
        if category not in CATEGORY_FILES:
            print(f"[bfcl] WARNING: Unknown category '{category}', skipping")
            continue

        is_exec = category.startswith("exec_")
        print(f"[bfcl] Downloading {category} data...")
        q_path = _download_file(CATEGORY_FILES[category], cache_dir)
        questions = _load_jsonl(q_path)

        answers_by_id: Dict[str, Any] = {}
        if not is_exec and category in ANSWER_FILES:
            a_path = _download_file(ANSWER_FILES[category], cache_dir)
            for entry in _load_jsonl(a_path):
                answers_by_id[entry["id"]] = entry.get("ground_truth", [])

        if max_tasks_per_category:
            questions = questions[:max_tasks_per_category]

        for entry in questions:
            task_id = entry["id"]
            question = _extract_question(entry.get("question", ""))
            functions = _convert_function_docs(entry.get("function", []))

            if is_exec:
                raw_gt = entry.get("ground_truth", [])
                ground_truth = _parse_exec_ground_truth(raw_gt)
            else:
                ground_truth = answers_by_id.get(task_id, [])

            tasks.append({
                "task_id": task_id,
                "category": category,
                "question": question,
                "functions": functions,
                "ground_truth": ground_truth,
            })

    print(f"[bfcl] Loaded {len(tasks)} tasks across {len(categories)} categories")
    return tasks


def build_user_content(task: Dict[str, Any]) -> str:
    """Build user prompt in Tool-R0 format: question + available tools JSON."""
    question = task["question"]
    tools_json = json.dumps(task["functions"], ensure_ascii=False)
    return f"User request:\n{question}\n\nAvailable tools (JSON):\n{tools_json}\n"
