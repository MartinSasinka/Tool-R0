"""
AppWorld task loader.

Loads tasks from AppWorld benchmark and formats them for Tool-R0
single-turn evaluation. Each task gets the instruction + API docs
for relevant apps, and the model must predict all needed tool calls.

Requirements:
    pip install appworld
    appworld install
    appworld download data

Reference:
    Trivedi et al., "AppWorld: A Controllable World of Apps and People
    for Benchmarking Interactive Coding Agents", ACL 2024 (Best Resource Paper).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


def _format_tools_for_prompt(api_docs_fc: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert AppWorld function_calling docs to Tool-R0 style tool list.

    AppWorld uses names like 'spotify__show_song'. We keep that format
    so the model's predictions can be mapped back to app.api for execution.
    """
    tools = []
    for doc in api_docs_fc:
        fn = doc.get("function", doc)
        tools.append({
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "parameters": fn.get("parameters", {}),
        })
    return tools


def load_tasks(
    dataset_name: str = "train",
    max_tasks: Optional[int] = None,
    appworld_root: Optional[str] = None,
    max_difficulty: Optional[int] = None,
    max_apis: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load AppWorld tasks with metadata for single-turn evaluation.

    Two-pass approach:
      1. Open each task world to collect metadata (instruction, API docs, ground truth)
      2. Close the world (no state is kept)

    The actual execution happens later in the runner.
    """
    try:
        from appworld import AppWorld, load_task_ids
    except ImportError:
        raise ImportError(
            "AppWorld loader requires `appworld`. Install:\n"
            "  pip install appworld\n"
            "  appworld install\n"
            "  appworld download data"
        )

    import os
    if appworld_root:
        os.environ["APPWORLD_ROOT"] = appworld_root

    task_ids = load_task_ids(dataset_name)
    print(f"[appworld] Dataset '{dataset_name}': {len(task_ids)} total tasks")

    if max_tasks is not None and max_tasks < len(task_ids):
        task_ids = task_ids[:max_tasks]
        print(f"[appworld] Limited to {max_tasks} tasks")

    tasks: List[Dict[str, Any]] = []
    skipped = 0

    for idx, task_id in enumerate(task_ids):
        if idx % 50 == 0:
            print(f"[appworld] Loading task {idx + 1}/{len(task_ids)}...")

        try:
            with AppWorld(
                task_id=task_id,
                experiment_name="_tool_r0_metadata_collect",
                ground_truth_mode="full",
            ) as world:
                # Safely extract ground truth metadata
                gt = world.task.ground_truth
                difficulty = 99
                num_apis = 99
                required_apps = []
                required_apis = []
                gt_answer = None
                gt_num_api_calls = 0

                if gt is not None:
                    md = getattr(gt, "metadata", None)
                    if md is not None and isinstance(md, dict):
                        difficulty = md.get("difficulty", 99)
                        num_apis = md.get("num_apis", 99)

                    ra = getattr(gt, "required_apps", None)
                    if ra is not None:
                        try:
                            required_apps = list(ra)
                        except TypeError:
                            required_apps = []

                    rapi = getattr(gt, "required_apis", None)
                    if rapi is not None:
                        try:
                            required_apis = list(rapi)
                        except TypeError:
                            required_apis = []

                    gt_answer = getattr(gt, "answer", None)

                    gc = getattr(gt, "api_calls", None)
                    if gc is not None:
                        try:
                            gt_num_api_calls = len(list(gc))
                        except TypeError:
                            gt_num_api_calls = 0

                if max_difficulty is not None and difficulty > max_difficulty:
                    skipped += 1
                    continue
                if max_apis is not None and num_apis > max_apis:
                    skipped += 1
                    continue

                api_docs_fc = world.task.api_docs.function_calling()
                if required_apps:
                    api_docs_fc = [
                        d for d in api_docs_fc
                        if any(d.get("function", d).get("name", "").startswith(app + "__") for app in required_apps)
                    ]

                tools = _format_tools_for_prompt(api_docs_fc)

                supervisor = world.task.supervisor
                sup_dict = supervisor if isinstance(supervisor, dict) else {}
                supervisor_info = (
                    f"My name is: {sup_dict.get('first_name', '')} {sup_dict.get('last_name', '')}. "
                    f"My personal email is {sup_dict.get('email', '')} "
                    f"and phone number is {sup_dict.get('phone_number', '')}."
                )

                tasks.append({
                    "task_id": task_id,
                    "instruction": world.task.instruction,
                    "supervisor_info": supervisor_info,
                    "tools": tools,
                    "required_apps": required_apps,
                    "required_apis": required_apis,
                    "difficulty": difficulty,
                    "num_apis": num_apis,
                    "gt_answer": gt_answer,
                    "gt_num_api_calls": gt_num_api_calls,
                })

        except Exception as e:
            import traceback
            print(f"[appworld] WARN: Failed to load task {task_id}: {e}")
            traceback.print_exc()
            skipped += 1
            if skipped >= 5 and not tasks:
                print(f"[appworld] ERROR: First {skipped} tasks all failed. Showing full traceback above.")
            continue

    print(f"[appworld] Loaded {len(tasks)} tasks (skipped {skipped})")
    if tasks:
        avg_tools = sum(len(t["tools"]) for t in tasks) / len(tasks)
        avg_gt_calls = sum(t["gt_num_api_calls"] for t in tasks) / len(tasks)
        print(f"[appworld] Avg tools/task: {avg_tools:.1f}, avg GT calls/task: {avg_gt_calls:.1f}")

    return tasks


def build_user_content(task: Dict[str, Any]) -> str:
    """Build the user prompt for an AppWorld task."""
    tools_json = json.dumps(task["tools"], indent=2, ensure_ascii=False)
    return (
        f"{task['supervisor_info']}\n\n"
        f"Task:\n{task['instruction']}\n\n"
        f"Available tools (JSON):\n{tools_json}"
    )
