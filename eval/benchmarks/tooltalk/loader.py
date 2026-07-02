"""
ToolTalk data loader.

Clones the ToolTalk repository (shallow) and loads conversation data
and OpenAI-style tool schemas.

ToolTalk (Microsoft, 2023) contains 78 multi-turn conversations
(28 easy + 50 hard) with 28 unique tools across 7 suites.

Reference:
    Farn & Shin, "ToolTalk: Evaluating Tool-Usage in a Conversation
    Setting", 2023.  https://arxiv.org/abs/2311.10775
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import Any, Dict, List, Optional, Tuple

TOOLTALK_REPO = "https://github.com/microsoft/ToolTalk.git"
TOOLTALK_DIR_NAME = "tooltalk_repo"


def _ensure_repo(cache_dir: Optional[str] = None) -> str:
    """Clone the ToolTalk repo if not already present. Returns repo path."""
    base = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "tooltalk")
    repo_path = os.path.join(base, TOOLTALK_DIR_NAME)

    if os.path.isdir(os.path.join(repo_path, "data")):
        return repo_path

    os.makedirs(base, exist_ok=True)
    print(f"[tooltalk] Cloning ToolTalk repo to {repo_path} ...")
    subprocess.check_call(
        ["git", "clone", "--depth", "1", TOOLTALK_REPO, repo_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return repo_path


def _load_tool_schemas() -> Dict[str, List[Dict[str, Any]]]:
    """Return static tool schemas (no ToolTalk import needed).

    Returns dict mapping suite name -> list of OpenAI-style function dicts.
    """
    from eval.benchmarks.tooltalk.tools import SUITES
    return SUITES


def _load_conversations(repo_path: str) -> List[Dict[str, Any]]:
    """Load all conversation JSON files from the data directory."""
    conversations = []

    for subdir in ["easy", "tooltalk"]:
        data_dir = os.path.join(repo_path, "data", subdir)
        if not os.path.isdir(data_dir):
            continue
        for fname in sorted(os.listdir(data_dir)):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(data_dir, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                conv = json.load(f)
            conv["difficulty"] = "easy" if subdir == "easy" else "hard"
            conv["source_file"] = fname
            conversations.append(conv)

    return conversations


def _extract_turns(conversation: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract evaluation turns from a conversation.

    Each evaluation turn is a point where the assistant should make
    API call(s).  We return the conversation history up to that point
    and the ground truth API calls.
    """
    turns = conversation["conversation"]
    metadata = conversation.get("metadata", {})
    session_token = metadata.get("session_token") or ""
    if not session_token and "user" in conversation:
        session_token = conversation["user"].get("session_token", "")

    eval_turns = []
    for i, turn in enumerate(turns):
        if turn["role"] == "assistant" and turn.get("apis"):
            history = turns[:i + 1]
            eval_turns.append({
                "history_up_to": history,
                "ground_truth_apis": turn["apis"],
                "turn_index": i,
                "session_token": session_token,
            })
    return eval_turns


def load_tasks(
    max_tasks: Optional[int] = None,
    cache_dir: Optional[str] = None,
    difficulty: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    """Load ToolTalk evaluation tasks.

    Args:
        max_tasks: Limit number of conversations (not turns).
        cache_dir: Where to clone the repo.
        difficulty: Filter by "easy" or "hard", or None for all.

    Returns:
        (tasks, tool_schemas) where tasks is a list of per-turn eval dicts
        and tool_schemas maps suite name -> tool definitions.
    """
    repo_path = _ensure_repo(cache_dir)
    tool_schemas = _load_tool_schemas()
    conversations = _load_conversations(repo_path)

    if difficulty:
        conversations = [c for c in conversations if c["difficulty"] == difficulty]

    if max_tasks is not None:
        conversations = conversations[:max_tasks]

    tasks = []
    for conv in conversations:
        suites_used = conv.get("suites_used", [])
        available_tools = []
        for suite_name in suites_used:
            available_tools.extend(tool_schemas.get(suite_name, []))

        eval_turns = _extract_turns(conv)
        for et in eval_turns:
            tasks.append({
                "task_id": f"tooltalk-{conv['conversation_id']}-t{et['turn_index']}",
                "conversation_id": conv["conversation_id"],
                "conversation_name": conv.get("name", ""),
                "difficulty": conv["difficulty"],
                "source_file": conv["source_file"],
                "suites_used": suites_used,
                "tools": available_tools,
                "history": et["history_up_to"],
                "ground_truth_apis": et["ground_truth_apis"],
                "turn_index": et["turn_index"],
                "session_token": et["session_token"],
                "metadata": conv.get("metadata", {}),
            })

    return tasks, tool_schemas


def build_user_content(task: Dict[str, Any]) -> str:
    """Build the user-facing prompt for a ToolTalk turn.

    Includes tool descriptions and conversation history.
    """
    lines = ["Available tools:"]
    for tool in task["tools"]:
        fn = tool["function"]
        params = fn.get("parameters", {}).get("properties", {})
        required = fn.get("required", [])
        param_descs = []
        for pname, pinfo in params.items():
            req_marker = " (required)" if pname in required else " (optional)"
            param_descs.append(
                f'    - {pname} ({pinfo.get("type", "string")}){req_marker}: '
                f'{pinfo.get("description", "")}'
            )
        params_str = "\n".join(param_descs) if param_descs else "    (no parameters)"
        lines.append(f'\n  {fn["name"]}: {fn["description"]}\n  Parameters:\n{params_str}')

    tools_text = "\n".join(lines)

    conv_lines = []
    for turn in task["history"]:
        role = turn["role"].capitalize()
        text = turn.get("text", "")
        if role == "Assistant" and turn.get("apis"):
            api_results = []
            for api in turn["apis"]:
                req = api["request"]
                resp = api.get("response", {})
                api_results.append(
                    f'[Tool call: {req["api_name"]}({json.dumps(req.get("parameters", {}))}) '
                    f'-> {json.dumps(resp)}]'
                )
            text = "\n".join(api_results) + ("\n" + text if text else "")

        if turn == task["history"][-1] and turn["role"] == "assistant":
            continue

        conv_lines.append(f"{role}: {text}")

    conversation_text = "\n".join(conv_lines)

    metadata = task.get("metadata", {})
    context_parts = []
    if metadata.get("location"):
        context_parts.append(f"Location: {metadata['location']}")
    if metadata.get("timestamp"):
        context_parts.append(f"Current time: {metadata['timestamp']}")
    if task.get("session_token"):
        context_parts.append(f"User session token: {task['session_token']}")
    context_str = "\n".join(context_parts)

    return (
        f"{tools_text}\n\n"
        f"Context:\n{context_str}\n\n"
        f"Based on the conversation below, generate the appropriate tool call(s).\n\n"
        f"{conversation_text}"
    )
