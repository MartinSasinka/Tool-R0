#!/usr/bin/env python3
"""
step2_verify_candidates.py

Deterministic strict verification of NESTFUL curriculum candidates.
Never crashes on malformed samples; writes rejected rows with reasons.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

# IBM execution parallelism / timeout defaults.
_DEFAULT_EXEC_WORKERS = min(8, max(1, multiprocessing.cpu_count() - 1))
_DEFAULT_EXEC_TIMEOUT = 20  # seconds per candidate before it is rejected

from context_budget import (
    DEFAULT_MAX_INPUT_CHARS,
    DEFAULT_TARGET_MAX_COMPLETION_TOKENS,
    DEFAULT_TARGET_PROMPT_TOKENS,
    DEFAULT_TOOL_MENU_MAX,
    check_context_budget,
    estimate_training_context,
    trim_tool_menu,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from curricullum.data.exec_trajectory import execute_trajectory, get_ibm_registry, verify_gold_row  # noqa: E402

REPORTS_DIR = os.path.join(SCRIPT_DIR, "reports")

DEFAULT_NESTFUL_CANDIDATES = [
    "eval/data/NESTFUL-main/data_v2/nestful_data.jsonl",
    "data_v2/nestful_data.jsonl",
]

MAX_GOLD_ANSWER_CHARS = 8192

_VAR_REF_RE = re.compile(r"^\$var_(\d+)(?:\.([A-Za-z_][\w]*))?\$$")

_CONTEXT_REASONS = {
    "input_too_long",
    "tool_menu_too_large",
    "prompt_tokens_over_budget",
    "completion_tokens_over_budget",
}

_CONTAMINATION_REASONS = {
    "copied_real_nestful_input",
    "duplicate_synthetic_input",
    "duplicate_output_sequence",
    "high_seed_overlap",
    "placeholder_input",
}

_PLACEHOLDER_PATTERNS = [
    r"\bthe generated user question here\b",
    r"\buser question must be from the specified domain\b",
    r"\bgenerate one new tool-calling task now\b",
    r"\bgenerate a new tool-calling task now\b",
    r"\bcontrol spec\b",
    r"\brules to satisfy\b",
    r"\bthen, without revealing your reasoning\b",
    r"\[the private reasoning here\]",
    r"\[.*generated user question.*\]",
    r"\[.*user question.*here.*\]",
    r"func_name1",
    r"argument1",
    r"value1",
    r"value2",
]
_PLACEHOLDER_RE = re.compile("|".join(_PLACEHOLDER_PATTERNS), re.IGNORECASE | re.DOTALL)


def resolve_nestful_path(cli_path: Optional[str]) -> str:
    if cli_path:
        if not os.path.isfile(cli_path):
            print(f"[err] nestful_path not found: {cli_path}", file=sys.stderr)
            sys.exit(1)
        return cli_path
    for p in DEFAULT_NESTFUL_CANDIDATES:
        if os.path.isfile(p):
            return p
    print(
        "[err] Could not find NESTFUL data. Tried:\n  "
        + "\n  ".join(DEFAULT_NESTFUL_CANDIDATES),
        file=sys.stderr,
    )
    sys.exit(1)


def coerce_json(s: str) -> Optional[Any]:
    try:
        return json.loads(s)
    except Exception:
        return None


def clean_json_blob(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\d", "0", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def is_json_primitive(x: Any) -> bool:
    return x is None or isinstance(x, (str, int, float, bool))


def is_placeholder_text(text: str) -> bool:
    if not text or not text.strip():
        return True
    if len(text.strip()) < 8:
        return True
    return _PLACEHOLDER_RE.search(text) is not None


def schema_allows_complex_type(spec: Any) -> bool:
    if isinstance(spec, str):
        t = spec.strip().lower()
        return any(k in t for k in ["list", "array", "object", "dict", "map", "json"])
    if isinstance(spec, dict):
        t = str(spec.get("type", "")).lower()
        if any(k in t for k in ["array", "object", "list", "dict"]):
            return True
        if "items" in spec or "properties" in spec:
            return True
    return False


def get_output_parameters(tool: Dict[str, Any]) -> Dict[str, Any]:
    out = tool.get("output_parameters")
    if isinstance(out, dict):
        return out
    out = tool.get("output_parameter")
    if isinstance(out, dict):
        return out
    return {}


def normalize_tools(tools: Any) -> Tuple[Optional[List[Dict[str, Any]]], str]:
    if isinstance(tools, str):
        tools = coerce_json(clean_json_blob(tools))
    if not isinstance(tools, list) or len(tools) == 0:
        return None, "tools_not_list_or_empty"
    out: List[Dict[str, Any]] = []
    for t in tools:
        if not isinstance(t, dict):
            return None, "tool_not_object"
        out.append(t)
    return out, "ok"


def normalize_calls(output: Any) -> Tuple[Optional[List[Dict[str, Any]]], str]:
    if isinstance(output, str):
        output = coerce_json(clean_json_blob(output))
    if not isinstance(output, list) or len(output) == 0:
        return None, "output_not_list_or_empty"
    out: List[Dict[str, Any]] = []
    for c in output:
        if not isinstance(c, dict):
            return None, "call_not_object"
        out.append(c)
    return out, "ok"


def validate_tool_schema(tools: List[Dict[str, Any]]) -> Tuple[bool, str]:
    seen: Set[str] = set()
    for t in tools:
        name = t.get("name")
        desc = t.get("description")
        params = t.get("parameters")
        if not isinstance(name, str) or not name.strip():
            return False, "bad_tool_name"
        if name in seen:
            return False, "duplicate_tool_name"
        seen.add(name)
        if not isinstance(desc, str) or len(desc.strip()) < 3:
            return False, "tool_missing_description"
        if not isinstance(params, dict) or len(params) == 0:
            return False, "tool_missing_parameters"
        out_params = get_output_parameters(t)
        if t.get("output_parameters") is not None and not isinstance(t.get("output_parameters"), dict):
            return False, "tool_bad_output_parameters"
    return True, "ok"


def get_param_spec(tool: Dict[str, Any], arg_name: str) -> Any:
    params = tool.get("parameters", {})
    if isinstance(params, dict) and "properties" in params:
        return params.get("properties", {}).get(arg_name)
    if isinstance(params, dict):
        return params.get(arg_name)
    return None


def arg_value_allowed(value: Any, param_spec: Any) -> bool:
    if isinstance(value, str) and _VAR_REF_RE.match(value.strip()):
        return True
    if isinstance(value, list):
        return schema_allows_complex_type(param_spec)
    if isinstance(value, dict):
        return False
    if is_json_primitive(value):
        return True
    return False


def validate_calls(
    calls: List[Dict[str, Any]], tools: List[Dict[str, Any]], epoch: int
) -> Tuple[bool, str]:
    toolmap = {t["name"]: t for t in tools if isinstance(t.get("name"), str)}
    for i, c in enumerate(calls, start=1):
        name = c.get("name")
        label = c.get("label")
        args = c.get("arguments")
        if not isinstance(name, str) or name not in toolmap:
            return False, "call_tool_name_unknown"
        if label != f"$var_{i}":
            return False, "invalid_label_sequence"
        if not isinstance(args, dict):
            return False, "call_args_not_object"
        tool = toolmap[name]
        for arg_name, arg_val in args.items():
            spec = get_param_spec(tool, arg_name)
            if spec is None:
                return False, "call_unknown_arg"
            if not arg_value_allowed(arg_val, spec):
                return False, "call_invalid_arg_value"
            if isinstance(arg_val, (list, dict)) and not schema_allows_complex_type(spec):
                return False, "call_nested_arguments"
    if len(calls) != epoch:
        return False, "wrong_call_count"
    return True, "ok"


def extract_references_from_value(value: Any) -> List[Tuple[int, Optional[str]]]:
    refs: List[Tuple[int, Optional[str]]] = []
    if isinstance(value, str):
        m = _VAR_REF_RE.match(value.strip())
        if m:
            refs.append((int(m.group(1)), m.group(2)))
    elif isinstance(value, list):
        for item in value:
            refs.extend(extract_references_from_value(item))
    return refs


def extract_references(calls: List[Dict[str, Any]]) -> List[Tuple[int, int, Optional[str]]]:
    """Return list of (call_index_1based, ref_var_index, field)."""
    found: List[Tuple[int, int, Optional[str]]] = []
    for i, c in enumerate(calls, start=1):
        args = c.get("arguments", {})
        if not isinstance(args, dict):
            continue
        for v in args.values():
            for ref_idx, field in extract_references_from_value(v):
                found.append((i, ref_idx, field))
    return found


def validate_references(
    calls: List[Dict[str, Any]], tools: List[Dict[str, Any]], strict_fields: bool = True
) -> Tuple[bool, str]:
    toolmap = {t["name"]: t for t in tools if isinstance(t.get("name"), str)}

    for call_idx, ref_idx, field in extract_references(calls):
        if ref_idx == call_idx:
            return False, "invalid_reference_self"
        if ref_idx > call_idx:
            return False, "invalid_reference_future"
        if ref_idx < 1 or ref_idx > len(calls):
            return False, "invalid_reference"

        ref_call = calls[ref_idx - 1]
        ref_tool = toolmap.get(ref_call.get("name", ""))
        if ref_tool is None:
            return False, "invalid_reference"

        out_params = get_output_parameters(ref_tool)
        if strict_fields:
            if field is None:
                if out_params:
                    return False, "invalid_reference_missing_field"
                field = "result"
            if out_params and field not in out_params:
                return False, "invalid_reference_field"
            if not out_params and field != "result":
                return False, "invalid_reference_field"
    return True, "ok"


def compute_dependency_depth(calls: List[Dict[str, Any]]) -> int:
    """Longest dependency chain length across calls (1 = no refs)."""
    n = len(calls)
    if n == 0:
        return 0

    refs_by_call: Dict[int, Set[int]] = defaultdict(set)
    for call_idx, ref_idx, _ in extract_references(calls):
        if 1 <= ref_idx < call_idx <= n:
            refs_by_call[call_idx].add(ref_idx)

    memo: Dict[int, int] = {}

    def depth(i: int) -> int:
        if i in memo:
            return memo[i]
        preds = refs_by_call.get(i, set())
        if not preds:
            memo[i] = 1
            return 1
        memo[i] = 1 + max(depth(p) for p in preds)
        return memo[i]

    return max(depth(i) for i in range(1, n + 1))


def call_references_var(calls: List[Dict[str, Any]], call_index_1based: int, var_index: int) -> bool:
    if call_index_1based < 1 or call_index_1based > len(calls):
        return False
    args = calls[call_index_1based - 1].get("arguments", {})
    if not isinstance(args, dict):
        return False
    for v in args.values():
        for ref_idx, _ in extract_references_from_value(v):
            if ref_idx == var_index:
                return True
    return False


def has_required_dependency(
    calls: List[Dict[str, Any]], epoch: int, mode: str
) -> Tuple[bool, str]:
    if epoch < 2:
        return True, "ok"

    refs = extract_references(calls)
    if not refs:
        return False, "missing_nested_dependency"

    if mode == "strict_chain":
        # Every consecutive pair must be linked: call i references call i-1.
        for i in range(2, epoch + 1):
            if not call_references_var(calls, i, i - 1):
                return False, f"strict_chain_missing_call{i}_to_call{i - 1}"
        # Full linear depth required.
        depth = compute_dependency_depth(calls)
        if depth < epoch:
            return False, "insufficient_dependency_depth"
        return True, "ok"

    if mode == "dag_chain":
        import math as _math
        # Multi-branch DAG: allow up to floor(epoch/2) independent "root" calls.
        # At least one non-first call must depend on some prior call.
        # The final call (var_N) MUST have a prior dependency.
        max_roots = _math.floor(epoch / 2)  # e.g. epoch=6 → max 3 roots, epoch=7 → max 3 roots
        root_count = 0
        for i in range(2, epoch + 1):
            has_dep = any(r[0] == i and r[1] < i for r in refs)
            if not has_dep:
                root_count += 1
                if root_count > max_roots:
                    return False, f"dag_chain_too_many_roots_at_call{i}"
        # Final call must have a dependency
        if not any(r[0] == epoch and r[1] < epoch for r in refs):
            return False, "dag_chain_final_call_no_dep"
        # Require meaningful nesting depth: at least (epoch - 3), min 3.
        min_depth = max(3, epoch - 3)
        depth = compute_dependency_depth(calls)
        if depth < min_depth:
            return False, f"dag_chain_insufficient_depth_{depth}_lt_{min_depth}"
        return True, "ok"

    # relaxed_nested (legacy)
    if epoch == 2 and not call_references_var(calls, 2, 1):
        return False, "epoch2_missing_call2_to_call1"

    depth = compute_dependency_depth(calls)
    if depth < 2:
        return False, "insufficient_dependency_depth"

    for i in range(2, epoch + 1):
        if not any(r[0] == i and r[1] < i for r in refs):
            return False, "independent_calls_not_nested"

    return True, "ok"


def output_fingerprint(calls: List[Dict[str, Any]]) -> str:
    norm = []
    for c in calls:
        norm.append(
            {
                "name": c.get("name"),
                "label": c.get("label"),
                "arguments": c.get("arguments"),
            }
        )
    return sha1(json.dumps(norm, sort_keys=True, ensure_ascii=False))


def fingerprints(
    inp: str, tools: List[Dict[str, Any]], calls: List[Dict[str, Any]]
) -> Dict[str, str]:
    qn = normalize_text(inp)
    ts = sha1(json.dumps(sorted([t.get("name") for t in tools]), ensure_ascii=False))
    cs = output_fingerprint(calls)
    return {
        "input_fp": sha1(qn),
        "tool_fp": ts,
        "output_fp": cs,
        "combo_fp": sha1(qn + "||" + cs),
    }


def token_overlap_ratio(a: str, b: str) -> float:
    ta = set(re.findall(r"[a-z0-9]+", normalize_text(a)))
    tb = set(re.findall(r"[a-z0-9]+", normalize_text(b)))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def load_nestful_inputs(path: str) -> Tuple[Set[str], Dict[str, str]]:
    normalized: Set[str] = set()
    id_to_input: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            inp = row.get("input", "")
            sid = row.get("sample_id", "")
            if isinstance(inp, str) and inp.strip():
                normalized.add(normalize_text(inp))
                if sid:
                    id_to_input[sid] = inp
    return normalized, id_to_input


def validate_sample(
    sample: Dict[str, Any],
    epoch: int,
    real_inputs: Set[str],
    accepted_inputs: Set[str],
    seen_output_fps: Set[str],
    seed_inputs: List[str],
    overlap_threshold: float,
    dependency_mode: str,
    target_prompt_tokens: int,
    target_max_completion_tokens: int,
    max_input_chars: int,
    max_tool_menu: int,
) -> Tuple[bool, str]:
    inp = sample.get("input")
    if not isinstance(inp, str) or not inp.strip():
        return False, "missing_input"
    if is_placeholder_text(inp):
        return False, "placeholder_input"

    tools, reason = normalize_tools(sample.get("tools"))
    if tools is None:
        return False, reason

    calls, reason = normalize_calls(sample.get("output"))
    if calls is None:
        return False, reason

    ok, reason = validate_tool_schema(tools)
    if not ok:
        return False, reason

    if len(calls) != epoch:
        return False, "wrong_call_count"

    ok, reason = validate_calls(calls, tools, epoch)
    if not ok:
        return False, reason

    ok, reason = validate_references(calls, tools, strict_fields=True)
    if not ok:
        return False, reason

    ok, reason = has_required_dependency(calls, epoch, dependency_mode)
    if not ok:
        return False, reason

    gold = sample.get("gold_answer")
    if gold is None or (isinstance(gold, str) and not gold.strip()):
        return False, "missing_gold_answer"
    gold_str = gold if isinstance(gold, str) else json.dumps(gold, ensure_ascii=False, default=str)
    if len(gold_str) > MAX_GOLD_ANSWER_CHARS:
        return False, "gold_answer_too_long"

    norm_inp = normalize_text(inp)
    if norm_inp in real_inputs:
        return False, "copied_real_nestful_input"
    if norm_inp in accepted_inputs:
        return False, "duplicate_synthetic_input"

    ofp = output_fingerprint(calls)
    if ofp in seen_output_fps:
        return False, "duplicate_output_sequence"

    for seed_inp in seed_inputs:
        if token_overlap_ratio(inp, seed_inp) >= overlap_threshold:
            return False, "high_seed_overlap"

    ok_budget, budget_reason, _est = check_context_budget(
        inp,
        tools,
        calls,
        target_prompt_tokens=target_prompt_tokens,
        target_max_completion_tokens=target_max_completion_tokens,
        max_input_chars=max_input_chars,
        max_tool_menu=max_tool_menu,
    )
    if not ok_budget:
        return False, budget_reason

    return True, "ok"


def fail_if_use_executor(use_executor: bool, executable_functions_path: Optional[str]) -> None:
    if not use_executor:
        return
    ibm = get_ibm_registry()
    if ibm is None:
        msg = "IBM registry unavailable — clone NESTFUL repo (nestful_repo) or set NESTFUL_REPO_DIR"
        if executable_functions_path:
            msg += f" (executable_functions_path={executable_functions_path})"
        print(f"[err] {msg}", file=sys.stderr)
        sys.exit(1)


def _format_exec_gold(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        return str(value)


def verify_execution(sample: Dict[str, Any], ibm_registry) -> Tuple[bool, str, Optional[Any]]:
    """Run gold trajectory; return (ok, reason, executed_final)."""
    calls = sample.get("output")
    if isinstance(calls, str):
        calls = coerce_json(clean_json_blob(calls))
    if not isinstance(calls, list):
        return False, "exec_bad_output", None

    gold = sample.get("gold_answer")
    ok, exec_final, err = verify_gold_row(calls, gold, ibm_registry=ibm_registry)
    if ok:
        return True, "ok", exec_final

    if err == "executor_mismatch" and exec_final is not None:
        return True, "ok_gold_corrected", exec_final

    if err:
        return False, f"exec_{err}", exec_final
    return False, "exec_failed", exec_final


##############################################################################
# Multiprocessing worker functions (module-level for pickling on Windows)
##############################################################################

_MP_IBM_REGISTRY = None  # set by _mp_init_worker in each child process


def _mp_init_worker(nestful_repo_dir: str) -> None:
    """Pool initializer: create an IBM registry in each worker process."""
    global _MP_IBM_REGISTRY  # noqa: PLW0603
    _MP_IBM_REGISTRY = get_ibm_registry(nestful_repo_dir)


def _mp_exec_sample(sample_json: str) -> Tuple[bool, str, Optional[Any]]:
    """Worker payload: deserialize the sample dict and run IBM execution."""
    global _MP_IBM_REGISTRY  # noqa: PLW0603
    if _MP_IBM_REGISTRY is None:
        return (False, "ibm_unavailable", None)
    try:
        sample: Dict[str, Any] = json.loads(sample_json)
        return verify_execution(sample, _MP_IBM_REGISTRY)
    except Exception as exc:
        return (False, f"exec_error_{type(exc).__name__}", None)


def write_reports(
    epoch: int,
    summary: Dict[str, Any],
    accepted_previews: List[Dict[str, Any]],
    rejected_previews: List[Dict[str, Any]],
) -> None:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    txt_path = os.path.join(REPORTS_DIR, f"step2_epoch{epoch}_report.txt")
    json_path = os.path.join(REPORTS_DIR, f"step2_epoch{epoch}_summary.json")

    lines = [
        f"Step2 verification report — epoch {epoch}",
        "",
        f"Input candidates: {summary.get('input_count')}",
        f"Verified: {summary.get('verified_count')}",
        f"Rejected: {summary.get('rejected_count')}",
        f"Dependency mode: {summary.get('dependency_mode')}",
        f"Contamination rejections: {summary.get('contamination_rejections')}",
        f"Context budget rejections: {summary.get('context_rejections')}",
        f"Prompt tokens est (verified median): {summary.get('prompt_tokens_median')}",
        f"Completion tokens est (verified median): {summary.get('completion_tokens_median')}",
        f"Duplicate rejections: {summary.get('duplicate_rejections')}",
        "",
        "Rejection reasons:",
    ]
    for k, v in summary.get("rejection_reasons", {}).items():
        lines.append(f"  {k}: {v}")
    lines.extend(["", "Output length distribution (verified):"])
    for k, v in summary.get("output_length_distribution", {}).items():
        lines.append(f"  len={k}: {v}")
    lines.extend(["", "Dependency depth distribution (verified):"])
    for k, v in summary.get("dependency_depth_distribution", {}).items():
        lines.append(f"  depth={k}: {v}")
    lines.extend(["", "Top tool names (verified):"])
    for name, cnt in summary.get("top_tool_names", [])[:15]:
        lines.append(f"  {name}: {cnt}")
    lines.extend(["", "Accepted previews (up to 5):"])
    for i, ex in enumerate(accepted_previews[:5], 1):
        lines.append(f"--- accepted {i} ---")
        lines.append(json.dumps(ex, ensure_ascii=False)[:800])
    lines.extend(["", "Rejected previews (up to 5):"])
    for i, ex in enumerate(rejected_previews[:5], 1):
        lines.append(f"--- rejected {i} ({ex.get('reason')}) ---")
        preview = ex.get("sample", ex)
        if isinstance(preview, dict):
            meta = preview.get("meta") or {}
            lines.append(
                json.dumps(
                    {
                        "candidate_id": meta.get("candidate_id"),
                        "raw_id": meta.get("raw_id"),
                        "input": preview.get("input", "")[:200],
                    },
                    ensure_ascii=False,
                )[:800]
            )
        else:
            lines.append(str(preview)[:800])

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[verify] reports -> {txt_path}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify NESTFUL curriculum candidates")
    ap.add_argument("--in_json", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--rejected_json", required=True)
    ap.add_argument("--epoch", type=int, choices=list(range(1, 8)), required=True)
    ap.add_argument("--nestful_path", default=None)
    ap.add_argument("--seed_overlap_threshold", type=float, default=0.85)
    ap.add_argument(
        "--dependency_mode",
        choices=["strict_chain", "dag_chain", "relaxed_nested"],
        default="strict_chain",
    )
    ap.add_argument("--use_executor", action="store_true", default=False)
    ap.add_argument(
        "--no_executor",
        action="store_true",
        help="Disable IBM execution verification (not recommended for tool_r0)",
    )
    ap.add_argument(
        "--executable_functions_path",
        default="eval/data/NESTFUL-main/data_v2/executable_functions",
    )
    ap.add_argument("--target_prompt_tokens", type=int, default=DEFAULT_TARGET_PROMPT_TOKENS)
    ap.add_argument(
        "--target_max_completion_tokens",
        type=int,
        default=DEFAULT_TARGET_MAX_COMPLETION_TOKENS,
    )
    ap.add_argument("--max_input_chars", type=int, default=DEFAULT_MAX_INPUT_CHARS)
    ap.add_argument("--max_tool_menu", type=int, default=DEFAULT_TOOL_MENU_MAX)
    ap.add_argument(
        "--exec_workers",
        type=int,
        default=_DEFAULT_EXEC_WORKERS,
        help=f"Parallel IBM executor threads (default {_DEFAULT_EXEC_WORKERS})",
    )
    ap.add_argument(
        "--exec_timeout",
        type=float,
        default=_DEFAULT_EXEC_TIMEOUT,
        help=f"Hard timeout per IBM exec call in seconds (default {_DEFAULT_EXEC_TIMEOUT})",
    )
    args = ap.parse_args()

    use_executor = args.use_executor and not args.no_executor
    fail_if_use_executor(use_executor, args.executable_functions_path)
    ibm_registry = get_ibm_registry() if use_executor else None

    nestful_path = resolve_nestful_path(args.nestful_path)
    real_inputs, id_to_input = load_nestful_inputs(nestful_path)

    with open(args.in_json, "r", encoding="utf-8") as f:
        candidates = json.load(f)
    if not isinstance(candidates, list):
        print("[err] input JSON must be a list", file=sys.stderr)
        sys.exit(1)

    verified: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    rejection_reasons: Counter = Counter()
    accepted_inputs: Set[str] = set()
    seen_output_fps: Set[str] = set()
    tool_counter: Counter = Counter()
    out_len_counter: Counter = Counter()
    depth_counter: Counter = Counter()
    prompt_token_estimates: List[int] = []
    completion_token_estimates: List[int] = []
    accepted_previews: List[Dict[str, Any]] = []
    rejected_previews: List[Dict[str, Any]] = []

    # -----------------------------------------------------------------------
    # Phase 1 — fast structural validation (sequential, no IBM, ~0.01 s/item)
    # Maintains correct deduplication order.
    # -----------------------------------------------------------------------
    phase1_pass: List[Dict[str, Any]] = []  # list of dict: sample + bookkeeping fields

    t0 = time.monotonic()
    for sample in candidates:
        if not isinstance(sample, dict):
            rejection_reasons["not_object"] += 1
            row = {"sample": sample, "reason": "not_object", "epoch": args.epoch}
            rejected.append(row)
            if len(rejected_previews) < 5:
                rejected_previews.append(row)
            continue

        meta = sample.get("meta") or {}
        dep_mode = meta.get("dependency_mode", args.dependency_mode)
        seed_ids = meta.get("seed_ids") or []
        seed_inputs = [id_to_input[sid] for sid in seed_ids if sid in id_to_input]
        candidate_id = meta.get("candidate_id")
        raw_id = meta.get("raw_id")

        ok, reason = validate_sample(
            sample,
            args.epoch,
            real_inputs,
            accepted_inputs,
            seen_output_fps,
            seed_inputs,
            args.seed_overlap_threshold,
            dep_mode,
            args.target_prompt_tokens,
            args.target_max_completion_tokens,
            args.max_input_chars,
            args.max_tool_menu,
        )
        if not ok:
            rejection_reasons[reason] += 1
            row = {
                "sample": sample,
                "reason": reason,
                "epoch": args.epoch,
                "candidate_id": candidate_id,
                "raw_id": raw_id,
            }
            rejected.append(row)
            if len(rejected_previews) < 5:
                rejected_previews.append(row)
            continue

        # Pre-reserve dedup slots so subsequent Phase-1 iterations see them.
        norm_inp_pre = normalize_text(sample["input"])
        calls_pre, _ = normalize_calls(sample["output"])
        accepted_inputs.add(norm_inp_pre)
        if calls_pre is not None:
            seen_output_fps.add(output_fingerprint(calls_pre))

        phase1_pass.append(
            {
                "sample": sample,
                "meta": meta,
                "candidate_id": candidate_id,
                "raw_id": raw_id,
                "dep_mode": dep_mode,
            }
        )

    t1 = time.monotonic()
    print(
        f"[verify] phase1 structural: {len(phase1_pass)} pass / "
        f"{len(rejected)} reject out of {len(candidates)}  "
        f"({t1 - t0:.1f}s)",
        file=sys.stderr,
    )

    # -----------------------------------------------------------------------
    # Phase 2 — parallel IBM execution using multiprocessing.Pool
    #
    # Threads are GIL-bound for CPU-heavy IBM functions → use separate
    # processes for true parallelism.  Each worker gets its own IBM registry
    # via the pool initializer.  Per-task timeout is enforced via
    # apply_async().get(timeout=T); a stuck worker holds its slot but
    # remaining workers keep running.  The pool is terminated at the end so
    # any stuck processes are killed.
    # -----------------------------------------------------------------------
    exec_results: Dict[int, Tuple[bool, str, Optional[Any]]] = {}
    nestful_repo_dir = os.environ.get("NESTFUL_REPO_DIR", "nestful_repo")

    if use_executor and ibm_registry is not None:
        workers = max(1, args.exec_workers)
        timeout = args.exec_timeout
        total = len(phase1_pass)
        print(
            f"[verify] phase2 IBM exec: {total} candidates, "
            f"{workers} workers, timeout={timeout}s each (multiprocessing)",
            file=sys.stderr,
        )

        pool = multiprocessing.Pool(
            processes=workers,
            initializer=_mp_init_worker,
            initargs=(nestful_repo_dir,),
            maxtasksperchild=200,  # refresh workers periodically (memory safety)
        )
        try:
            # Submit all tasks immediately; workers start in parallel.
            ar_list: List[Tuple[Any, int]] = [
                (
                    pool.apply_async(
                        _mp_exec_sample,
                        (json.dumps(item["sample"], ensure_ascii=False),),
                    ),
                    idx,
                )
                for idx, item in enumerate(phase1_pass)
            ]

            done_count = 0
            for ar, idx in ar_list:
                try:
                    exec_results[idx] = ar.get(timeout=timeout)
                except multiprocessing.TimeoutError:
                    exec_results[idx] = (False, "exec_timeout", None)
                except Exception as exc:
                    exec_results[idx] = (False, f"exec_error_{type(exc).__name__}", None)
                done_count += 1
                if done_count % 50 == 0 or done_count == total:
                    elapsed = time.monotonic() - t1
                    rate = done_count / elapsed if elapsed > 0 else 0
                    eta = (total - done_count) / rate if rate > 0 else 0
                    print(
                        f"[verify]   exec {done_count}/{total} "
                        f"({rate:.1f}/s, ETA {eta:.0f}s)",
                        file=sys.stderr,
                    )
        finally:
            pool.terminate()  # kill any stuck worker processes
            pool.join()

        t2 = time.monotonic()
        timeouts = sum(1 for v in exec_results.values() if v[1] == "exec_timeout")
        print(
            f"[verify] phase2 done: {t2 - t1:.1f}s  timeouts={timeouts}",
            file=sys.stderr,
        )
    else:
        # No IBM exec — all phase1 pass directly.
        for idx in range(len(phase1_pass)):
            exec_results[idx] = (True, "no_exec", None)
        t2 = time.monotonic()

    # -----------------------------------------------------------------------
    # Phase 3 — sequential assembly: apply IBM results, build verified list
    # -----------------------------------------------------------------------
    for idx, item in enumerate(phase1_pass):
        sample = item["sample"]
        meta = item["meta"]
        candidate_id = item["candidate_id"]
        raw_id = item["raw_id"]

        exec_ok, exec_reason, exec_final = exec_results.get(idx, (False, "exec_missing", None))

        if use_executor and ibm_registry is not None and not exec_ok:
            rejection_reasons[exec_reason] += 1
            row = {
                "sample": sample,
                "reason": exec_reason,
                "epoch": args.epoch,
                "candidate_id": candidate_id,
                "raw_id": raw_id,
            }
            rejected.append(row)
            if len(rejected_previews) < 5:
                rejected_previews.append(row)
            continue

        if exec_final is not None and exec_reason == "ok_gold_corrected":
            sample = dict(sample)
            sample["gold_answer"] = _format_exec_gold(exec_final)

        tools, _ = normalize_tools(sample["tools"])
        calls, _ = normalize_calls(sample["output"])
        assert tools is not None and calls is not None

        depth = compute_dependency_depth(calls)
        ctx = estimate_training_context(sample["input"], tools, calls)

        # Drop bulky fields from step1 meta — raw_response bloats verified JSON (600MB+).
        slim_meta = {k: v for k, v in meta.items() if k != "raw_response"}

        out = dict(sample)
        out["tools"] = tools
        out["output"] = calls
        out["meta"] = {
            **slim_meta,
            "validation_passed": True,
            "dependency_depth": depth,
            "context_est": ctx,
            "fingerprints": fingerprints(sample["input"], tools, calls),
            "training_format": "tool_r0",
            "multiturn_ready": True,
            "ibm_exec_verified": use_executor,
        }
        verified.append(out)
        out_len_counter[str(len(calls))] += 1
        depth_counter[str(depth)] += 1
        prompt_token_estimates.append(ctx["prompt_tokens_est"])
        completion_token_estimates.append(ctx["completion_tokens_est"])
        for c in calls:
            if isinstance(c.get("name"), str):
                tool_counter[c["name"]] += 1
        if len(accepted_previews) < 5:
            accepted_previews.append(
                {
                    "candidate_id": candidate_id,
                    "raw_id": raw_id,
                    "input": sample["input"][:200],
                    "output_len": len(calls),
                    "dependency_depth": depth,
                    "gold_answer": sample.get("gold_answer"),
                }
            )

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(verified, f, ensure_ascii=False, indent=2)

    os.makedirs(os.path.dirname(args.rejected_json) or ".", exist_ok=True)
    with open(args.rejected_json, "w", encoding="utf-8") as f:
        json.dump(rejected, f, ensure_ascii=False, indent=2)

    contamination_rejections = sum(rejection_reasons[r] for r in _CONTAMINATION_REASONS if r in rejection_reasons)
    context_rejections = sum(rejection_reasons[r] for r in _CONTEXT_REASONS if r in rejection_reasons)
    duplicate_rejections = rejection_reasons.get("duplicate_synthetic_input", 0) + rejection_reasons.get(
        "duplicate_output_sequence", 0
    )

    def _median(vals: List[int]) -> Optional[float]:
        if not vals:
            return None
        s = sorted(vals)
        mid = len(s) // 2
        if len(s) % 2:
            return float(s[mid])
        return (s[mid - 1] + s[mid]) / 2.0

    summary = {
        "stage": "step2_verify_candidates",
        "epoch": args.epoch,
        "dependency_mode": args.dependency_mode,
        "use_executor": use_executor,
        "input_count": len(candidates),
        "verified_count": len(verified),
        "rejected_count": len(rejected),
        "rejection_reasons": dict(rejection_reasons),
        "contamination_rejections": contamination_rejections,
        "context_rejections": context_rejections,
        "duplicate_rejections": duplicate_rejections,
        "prompt_tokens_median": _median(prompt_token_estimates),
        "completion_tokens_median": _median(completion_token_estimates),
        "target_prompt_tokens": args.target_prompt_tokens,
        "target_max_completion_tokens": args.target_max_completion_tokens,
        "output_length_distribution": dict(out_len_counter),
        "dependency_depth_distribution": dict(depth_counter),
        "unique_tool_names": len(tool_counter),
        "top_tool_names": tool_counter.most_common(20),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "out_json": args.out_json,
        "rejected_json": args.rejected_json,
    }
    write_reports(args.epoch, summary, accepted_previews, rejected_previews)
    print(f"[verify] verified={len(verified)} rejected={len(rejected)}", file=sys.stderr)


if __name__ == "__main__":
    main()
