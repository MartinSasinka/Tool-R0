#!/usr/bin/env python3
"""
step1_gen_candidates.py

Generate NESTFUL-style curriculum candidates via OpenRouter.
Writes intermediate JSON with meta (including raw_response for every API call).
All raw responses are logged before any parse rejection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from openai import OpenAI

from context_budget import (
    DEFAULT_MAX_INPUT_CHARS,
    DEFAULT_TARGET_MAX_COMPLETION_TOKENS,
    DEFAULT_TARGET_PROMPT_TOKENS,
    DEFAULT_TOOL_MENU_MAX,
    DEFAULT_TOOL_MENU_MIN,
    check_context_budget,
    compact_tools_list,
    estimate_training_context,
    trim_tool_menu,
)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPTS_DIR = os.path.join(SCRIPT_DIR, "prompts")
REPORTS_DIR = os.path.join(SCRIPT_DIR, "reports")

# OpenRouter slug (NOT HuggingFace: deepseek-ai/DeepSeek-V4-Flash is for HF/DeepInfra only)
DEFAULT_OPENROUTER_MODEL = "deepseek/deepseek-v4-flash"
OPENROUTER_MODEL_ALIASES = {
    "deepseek-ai/deepseek-v4-flash": DEFAULT_OPENROUTER_MODEL,
    "deepseek-ai/DeepSeek-V4-Flash": DEFAULT_OPENROUTER_MODEL,
    "deepseek/deepseek-v4-flash": DEFAULT_OPENROUTER_MODEL,
}


def resolve_openrouter_model(model: str) -> str:
    """Map common HF/provider IDs to OpenRouter model slugs."""
    key = model.strip()
    mapped = OPENROUTER_MODEL_ALIASES.get(key, key)
    if mapped != key:
        print(f"[gen] OpenRouter model alias: {key} -> {mapped}", file=sys.stderr)
    return mapped

DEFAULT_NESTFUL_CANDIDATES = [
    "eval/data/NESTFUL-main/data_v2/nestful_data.jsonl",
    "data_v2/nestful_data.jsonl",
]

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
    i_obj, j_obj = s.find("{"), s.rfind("}")
    i_arr, j_arr = s.find("["), s.rfind("]")
    if i_obj != -1 and j_obj != -1 and j_obj > i_obj:
        if i_arr == -1 or i_obj < i_arr:
            return s[i_obj : j_obj + 1].strip()
    if i_arr != -1 and j_arr != -1 and j_arr > i_arr:
        return s[i_arr : j_arr + 1].strip()
    return s.strip()


def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\d", "0", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def tool_schema_fingerprint(tool: Dict[str, Any]) -> str:
    return sha1(json.dumps(tool, sort_keys=True, ensure_ascii=False))


def load_nestful_corpus(
    path: str,
) -> Tuple[
    List[Dict[str, Any]],
    Dict[str, Dict[str, Any]],
    Set[str],
    List[Dict[str, Any]],
    Dict[str, List[str]],
]:
    rows: List[Dict[str, Any]] = []
    normalized_inputs: Set[str] = set()
    by_output_len: Dict[str, List[str]] = {}
    variants: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            tools_raw = row.get("tools")
            if isinstance(tools_raw, str):
                tools_raw = json.loads(tools_raw)
            output_raw = row.get("output")
            if isinstance(output_raw, str):
                output_raw = json.loads(output_raw)

            row["_tools_parsed"] = tools_raw if isinstance(tools_raw, list) else []
            row["_output_parsed"] = output_raw if isinstance(output_raw, list) else []
            rows.append(row)

            inp = row.get("input", "")
            if isinstance(inp, str) and inp.strip():
                normalized_inputs.add(normalize_text(inp))

            sid = row.get("sample_id", str(len(rows)))
            olen = len(row["_output_parsed"])
            by_output_len.setdefault(str(olen), []).append(sid)

            for t in row["_tools_parsed"]:
                if not isinstance(t, dict):
                    continue
                name = t.get("name")
                if not isinstance(name, str) or not name:
                    continue
                fp = tool_schema_fingerprint(t)
                bucket = variants[name].setdefault(fp, {"schema": t, "count": 0})
                bucket["count"] += 1

    catalog: Dict[str, Dict[str, Any]] = {}
    conflict_report: List[Dict[str, Any]] = []
    for name, fps in variants.items():
        canonical_fp = max(fps.items(), key=lambda x: x[1]["count"])[0]
        catalog[name] = fps[canonical_fp]["schema"]
        if len(fps) > 1:
            alts = []
            for fp, info in fps.items():
                if fp == canonical_fp:
                    continue
                alts.append(
                    {
                        "schema_fingerprint": fp,
                        "occurrences": info["count"],
                        "schema": info["schema"],
                    }
                )
            conflict_report.append(
                {
                    "tool_name": name,
                    "variant_count": len(fps),
                    "canonical_schema_fingerprint": canonical_fp,
                    "canonical_occurrences": fps[canonical_fp]["count"],
                    "selected_canonical_schema": fps[canonical_fp]["schema"],
                    "rejected_alternative_schemas": alts,
                }
            )

    return rows, catalog, normalized_inputs, conflict_report, by_output_len


def write_tool_schema_conflicts_report(conflicts: List[Dict[str, Any]], nestful_path: str) -> str:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    out_path = os.path.join(REPORTS_DIR, "tool_schema_conflicts.json")
    payload = {
        "nestful_path": nestful_path,
        "conflict_count": len(conflicts),
        "conflicts": conflicts,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_path


def read_prompt_file(name: str) -> str:
    path = os.path.join(PROMPTS_DIR, name)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def truncate_json(obj: Any, max_chars: int = 1200) -> str:
    s = json.dumps(obj, ensure_ascii=False)
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3] + "..."


def sample_tools_for_prompt(
    catalog: Dict[str, Dict[str, Any]],
    rng: random.Random,
    menu_size: int = 6,
) -> List[Dict[str, Any]]:
    names = list(catalog.keys())
    if len(names) <= menu_size:
        picked = names
    else:
        picked = rng.sample(names, menu_size)
    return compact_tools_list([catalog[n] for n in picked])


def build_seed_block(
    rows: List[Dict[str, Any]],
    seed_ids: List[str],
    seed_mode: str,
) -> str:
    if seed_mode == "schema_only":
        return (
            "\nReference style (do not copy any real benchmark task):\n"
            "- Use labels $var_1, $var_2, ... sequentially.\n"
            "- Use arg_0, arg_1 when the tool schema defines them.\n"
            "- Chain outputs with $var_1.result$ or $var_1.output_0$ matching output_parameters.\n"
            "- gold_answer = numeric/string result after executing the call chain.\n"
        )
    id_map = {r.get("sample_id"): r for r in rows}
    examples = []
    for sid in seed_ids:
        r = id_map.get(sid)
        if not r:
            continue
        ex = {
            "sample_id": sid,
            "input": r.get("input", "")[:500],
            "output": r.get("_output_parsed", []),
        }
        if seed_mode == "fewshot_debug":
            ex["tools"] = r.get("_tools_parsed", [])[:5]
            ex["gold_answer"] = r.get("gold_answer")
        examples.append(ex)
    if not examples:
        return ""
    label = "Style reference only — do NOT copy:" if seed_mode == "fewshot" else "Debug examples (do NOT copy):"
    return f"\n{label}\n{truncate_json(examples, 2000 if seed_mode == 'fewshot_debug' else 800)}\n"


def build_user_prompt(
    epoch: int,
    tool_menu: List[Dict[str, Any]],
    seed_ids: List[str],
    rows: List[Dict[str, Any]],
    seed_mode: str,
) -> str:
    epoch_file = f"epoch_{epoch}call.md"
    template = read_prompt_file(epoch_file)
    allowed = sorted({t["name"] for t in tool_menu if isinstance(t.get("name"), str)})
    body = template.format(
        epoch=epoch,
        num_calls=epoch,
        allowed_tool_names=", ".join(allowed),
        tool_schemas_json=json.dumps(tool_menu, ensure_ascii=False, separators=(",", ":")),
    )
    body += build_seed_block(rows, seed_ids, seed_mode)
    return body


def parse_model_objects(raw: str) -> Tuple[List[Dict[str, Any]], str]:
    cleaned = clean_json_blob(raw)
    data = coerce_json(cleaned)
    if data is None:
        return [], "json_parse_failed"
    if isinstance(data, dict):
        return [data], "ok"
    if isinstance(data, list):
        objs = [x for x in data if isinstance(x, dict)]
        if not objs:
            return [], "empty_object_list"
        return objs, "ok"
    return [], "unexpected_json_type"


def extract_candidate_fields(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    inp = obj.get("input")
    tools = obj.get("tools")
    output = obj.get("output")
    gold = obj.get("gold_answer")
    if not isinstance(inp, str) or not inp.strip():
        return None
    if tools is None or output is None or gold is None:
        return None
    if isinstance(tools, str):
        tools = coerce_json(tools)
    if isinstance(output, str):
        output = coerce_json(output)
    if not isinstance(tools, list) or not isinstance(output, list):
        return None
    return {
        "input": inp.strip(),
        "tools": tools,
        "output": output,
        "gold_answer": gold,
    }


def tool_names_in_sample(tools: List[Any], output: List[Any]) -> Set[str]:
    names: Set[str] = set()
    for t in tools:
        if isinstance(t, dict) and isinstance(t.get("name"), str):
            names.add(t["name"])
    for c in output:
        if isinstance(c, dict) and isinstance(c.get("name"), str):
            names.add(c["name"])
    return names


def make_raw_id(epoch: int, request_idx: int, slot: int) -> str:
    return f"raw-e{epoch}-req{request_idx:06d}-slot{slot:02d}"


def make_candidate_id(epoch: int, raw_id: str, input_text: str) -> str:
    return f"cand-e{epoch}-{sha1(raw_id + '||' + normalize_text(input_text))[:12]}"


def call_openrouter_one(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout: float = 180.0,
) -> str:
    last_err: Optional[Exception] = None
    text = ""
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            text = (resp.choices[0].message.content or "").strip()
            break
        except Exception as e:
            last_err = e
            time.sleep(1.0 * (attempt + 1))
    if last_err and not text:
        print(f"[gen] API error after retries: {last_err}", file=sys.stderr)
    return text


def call_openrouter_batch(
    client: OpenAI,
    model: str,
    messages_list: List[List[Dict[str, str]]],
    temperature: float,
    top_p: float,
    max_tokens: int,
    parallel_workers: int = 1,
    timeout: float = 180.0,
) -> List[str]:
    if not messages_list:
        return []
    workers = max(1, min(parallel_workers, len(messages_list)))
    if workers == 1:
        return [
            call_openrouter_one(
                client, model, m, temperature, top_p, max_tokens, timeout=timeout
            )
            for m in messages_list
        ]

    texts: List[str] = [""] * len(messages_list)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                call_openrouter_one,
                client,
                model,
                messages_list[i],
                temperature,
                top_p,
                max_tokens,
                timeout,
            ): i
            for i in range(len(messages_list))
        }
        for fut in as_completed(futures):
            idx = futures[fut]
            texts[idx] = fut.result()
    return texts


def write_reports(
    epoch: int,
    summary: Dict[str, Any],
    accepted_previews: List[Dict[str, Any]],
    rejected_previews: List[Dict[str, Any]],
) -> None:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    txt_path = os.path.join(REPORTS_DIR, f"step1_epoch{epoch}_report.txt")
    json_path = os.path.join(REPORTS_DIR, f"step1_epoch{epoch}_summary.json")

    lines = [
        f"Step1 generation report — epoch {epoch}",
        f"Generated at: {summary.get('created_at')}",
        "",
        f"OpenRouter requests: {summary.get('api_requests')}",
        f"Generation attempts (tasks): {summary.get('api_attempts_tasks')}",
        f"Raw responses logged: {summary.get('raw_responses_logged')}",
        f"Parsed candidates written: {summary.get('written_candidates')}",
        f"Parse failures: {summary.get('parse_failures')}",
        f"Unknown tool rejects (post-parse): {summary.get('unknown_tool_rejects')}",
        f"Context budget rejects (post-parse): {summary.get('budget_rejects')}",
        f"Target prompt tokens: {summary.get('target_prompt_tokens')}",
        f"Target max completion tokens: {summary.get('target_max_completion_tokens')}",
        "",
        "Output length distribution (written candidates):",
    ]
    for k, v in summary.get("output_length_distribution", {}).items():
        lines.append(f"  len={k}: {v}")
    lines.extend(["", "Top tool names:"])
    for name, cnt in summary.get("top_tool_names", [])[:15]:
        lines.append(f"  {name}: {cnt}")
    lines.extend(
        [
            "",
            f"Tool schema conflicts: {summary.get('schema_conflict_count', 0)} "
            f"(see {summary.get('schema_conflicts_report', 'tool_schema_conflicts.json')})",
        ]
    )
    lines.extend(["", "Accepted previews (up to 5):"])
    for i, ex in enumerate(accepted_previews[:5], 1):
        lines.append(f"--- accepted {i} ---")
        lines.append(truncate_json(ex, 800))
    lines.extend(["", "Rejected previews (up to 5):"])
    for i, ex in enumerate(rejected_previews[:5], 1):
        lines.append(f"--- rejected {i} ({ex.get('reason')}) ---")
        lines.append(truncate_json(ex, 800))

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[gen] reports -> {txt_path}", file=sys.stderr)


def fail_if_use_executor(use_executor: bool, executable_functions_path: Optional[str]) -> None:
    if not use_executor:
        return
    msg = "Execution validation is not implemented yet."
    if executable_functions_path:
        msg += f" (executable_functions_path={executable_functions_path})"
    print(f"[err] {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate NESTFUL curriculum candidates via OpenRouter")
    ap.add_argument("--nestful_path", default=None)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--epoch", type=int, choices=list(range(1, 8)), required=True)
    ap.add_argument("--n_generate", type=int, default=900)
    ap.add_argument("--max_generate", type=int, default=1400)
    ap.add_argument("--batch_size", type=int, default=12)
    ap.add_argument("--parallel_workers", type=int, default=12, help="Concurrent OpenRouter calls per batch")
    ap.add_argument("--tool_menu_min", type=int, default=DEFAULT_TOOL_MENU_MIN)
    ap.add_argument("--tool_menu_max", type=int, default=DEFAULT_TOOL_MENU_MAX)
    ap.add_argument("--target_prompt_tokens", type=int, default=DEFAULT_TARGET_PROMPT_TOKENS)
    ap.add_argument(
        "--target_max_completion_tokens",
        type=int,
        default=DEFAULT_TARGET_MAX_COMPLETION_TOKENS,
    )
    ap.add_argument("--max_input_chars", type=int, default=DEFAULT_MAX_INPUT_CHARS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default=DEFAULT_OPENROUTER_MODEL)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_tokens", type=int, default=2048, help="OpenRouter completion cap (not GRPO training budget)")
    ap.add_argument("--api_timeout", type=float, default=180.0, help="Seconds per OpenRouter request")
    ap.add_argument("--n_seed_examples", type=int, default=0)
    ap.add_argument(
        "--seed_mode",
        choices=["schema_only", "fewshot", "fewshot_debug"],
        default="schema_only",
    )
    ap.add_argument(
        "--dependency_mode",
        choices=["strict_chain", "dag_chain", "relaxed_nested"],
        default="strict_chain",
        help="Stored in meta for step2; strict_chain=linear, dag_chain=DAG for epochs 6-7",
    )
    ap.add_argument("--allow_new_tools", action="store_true")
    ap.add_argument("--use_executor", action="store_true")
    ap.add_argument(
        "--executable_functions_path",
        default="eval/data/NESTFUL-main/data_v2/executable_functions",
        help="Future hook for execution validation (not implemented)",
    )
    args = ap.parse_args()
    args.model = resolve_openrouter_model(args.model)

    fail_if_use_executor(args.use_executor, args.executable_functions_path)

    if "OPENROUTER_API_KEY" not in os.environ:
        print("[err] OPENROUTER_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    rng = random.Random(args.seed)
    nestful_path = resolve_nestful_path(args.nestful_path)
    rows, catalog, _, schema_conflicts, by_len = load_nestful_corpus(nestful_path)
    conflicts_path = write_tool_schema_conflicts_report(schema_conflicts, nestful_path)
    if schema_conflicts:
        print(
            f"[gen] WARNING: {len(schema_conflicts)} tool schema conflicts; "
            f"canonical schemas by frequency -> {conflicts_path}",
            file=sys.stderr,
        )

    epoch_key = str(args.epoch)
    pool_ids = by_len.get(epoch_key, [])
    if not pool_ids:
        print(
            f"[gen] WARNING: no real examples with {args.epoch} calls; using full corpus for seeds",
            file=sys.stderr,
        )
        pool_ids = [r.get("sample_id") for r in rows if r.get("sample_id")]

    system_prompt = read_prompt_file("system_prompt.md")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )

    candidates: List[Dict[str, Any]] = []
    parse_failures = 0
    unknown_tool_rejects = 0
    budget_rejects = 0
    api_requests = 0
    accepted_previews: List[Dict[str, Any]] = []
    rejected_previews: List[Dict[str, Any]] = []
    tool_name_counter: Counter = Counter()
    output_len_counter: Counter = Counter()

    attempts = 0
    request_idx = 0
    iterator: Any
    max_batches = (args.max_generate + args.batch_size - 1) // args.batch_size
    batch_range = range(max_batches)
    if tqdm is not None:
        iterator = tqdm(batch_range, desc=f"epoch{args.epoch} gen", file=sys.stderr)
    else:
        iterator = batch_range

    raw_log_path = args.out_json.replace(".json", "_raw_responses.jsonl")
    os.makedirs(os.path.dirname(raw_log_path) or ".", exist_ok=True)
    raw_log_file = open(raw_log_path, "w", encoding="utf-8")

    t0 = time.time()
    try:
        for _ in iterator:
            if len(candidates) >= args.n_generate or attempts >= args.max_generate:
                break

            batch_n = min(args.batch_size, args.n_generate - len(candidates), args.max_generate - attempts)
            if batch_n <= 0:
                break

            messages_list: List[List[Dict[str, str]]] = []
            batch_meta: List[Dict[str, Any]] = []

            for _ in range(batch_n):
                seed_ids: List[str] = []
                if args.n_seed_examples > 0 and args.seed_mode != "schema_only":
                    seed_ids = rng.sample(pool_ids, min(args.n_seed_examples, len(pool_ids)))
                tool_menu_size = rng.randint(args.tool_menu_min, args.tool_menu_max)
                tool_menu = sample_tools_for_prompt(catalog, rng, menu_size=tool_menu_size)
                user_prompt = build_user_prompt(args.epoch, tool_menu, seed_ids, rows, args.seed_mode)
                messages_list.append(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ]
                )
                batch_meta.append({"seed_ids": seed_ids, "tool_menu_names": [t["name"] for t in tool_menu]})

            raw_texts = call_openrouter_batch(
                client,
                args.model,
                messages_list,
                args.temperature,
                args.top_p,
                args.max_tokens,
                parallel_workers=args.parallel_workers,
                timeout=args.api_timeout,
            )
            api_requests += len(messages_list)
            attempts += batch_n

            now = datetime.now(timezone.utc).isoformat()
            for slot, (raw, meta) in enumerate(zip(raw_texts, batch_meta)):
                request_idx += 1
                raw_id = make_raw_id(args.epoch, request_idx, slot)
                raw_entry = {
                    "raw_id": raw_id,
                    "epoch": args.epoch,
                    "model": args.model,
                    "raw_response": raw,
                    "seed_ids": meta["seed_ids"],
                    "seed_mode": args.seed_mode,
                    "created_at": now,
                }
                raw_log_file.write(json.dumps(raw_entry, ensure_ascii=False) + "\n")
                raw_log_file.flush()

                objs, parse_reason = parse_model_objects(raw)
                if parse_reason != "ok":
                    parse_failures += 1
                    if len(rejected_previews) < 5:
                        rejected_previews.append({"reason": parse_reason, "raw_id": raw_id, "raw": raw[:500]})
                    continue

                for obj in objs:
                    fields = extract_candidate_fields(obj)
                    if fields is None:
                        parse_failures += 1
                        if len(rejected_previews) < 5:
                            rejected_previews.append(
                                {"reason": "missing_fields", "raw_id": raw_id, "raw": raw[:500]}
                            )
                        continue

                    names = tool_names_in_sample(fields["tools"], fields["output"])
                    if not args.allow_new_tools and any(n not in catalog for n in names):
                        unknown_tool_rejects += 1
                        if len(rejected_previews) < 5:
                            rejected_previews.append(
                                {
                                    "reason": "unknown_tool",
                                    "raw_id": raw_id,
                                    "names": sorted(names),
                                    "input": fields["input"][:200],
                                }
                            )
                        continue

                    fields["tools"] = trim_tool_menu(
                        fields["tools"],
                        fields["output"],
                        catalog=catalog,
                        rng=rng,
                        min_menu=args.tool_menu_min,
                        max_menu=args.tool_menu_max,
                    )
                    ok_budget, budget_reason, est = check_context_budget(
                        fields["input"],
                        fields["tools"],
                        fields["output"],
                        target_prompt_tokens=args.target_prompt_tokens,
                        target_max_completion_tokens=args.target_max_completion_tokens,
                        max_input_chars=args.max_input_chars,
                        max_tool_menu=args.tool_menu_max,
                    )
                    if not ok_budget:
                        budget_rejects += 1
                        if len(rejected_previews) < 5:
                            rejected_previews.append(
                                {
                                    "reason": budget_reason,
                                    "raw_id": raw_id,
                                    "est": est,
                                    "input": fields["input"][:120],
                                }
                            )
                        continue

                    candidate_id = make_candidate_id(args.epoch, raw_id, fields["input"])
                    record = {
                        **fields,
                        "meta": {
                            "candidate_id": candidate_id,
                            "raw_id": raw_id,
                            "epoch": args.epoch,
                            "model": args.model,
                            "raw_response": raw,
                            "seed_ids": meta["seed_ids"],
                            "seed_mode": args.seed_mode,
                            "created_at": now,
                            "dependency_mode": args.dependency_mode,
                            "context_est": est,
                        },
                    }
                    candidates.append(record)
                    output_len_counter[str(len(fields["output"]))] += 1
                    for n in names:
                        tool_name_counter[n] += 1
                    if len(accepted_previews) < 5:
                        accepted_previews.append(
                            {
                                "candidate_id": candidate_id,
                                "raw_id": raw_id,
                                "input": fields["input"][:200],
                                "output_len": len(fields["output"]),
                            }
                        )
                    if len(candidates) >= args.n_generate:
                        break

            if (request_idx % 50) == 0:
                print(
                    f"[gen] requests={api_requests} attempts={attempts} candidates={len(candidates)}",
                    file=sys.stderr,
                )
    finally:
        raw_log_file.close()

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(candidates, f, ensure_ascii=False, indent=2)

    summary = {
        "stage": "step1_gen_candidates",
        "epoch": args.epoch,
        "nestful_path": nestful_path,
        "model": args.model,
        "seed_mode": args.seed_mode,
        "dependency_mode": args.dependency_mode,
        "parallel_workers": args.parallel_workers,
        "allow_new_tools": args.allow_new_tools,
        "api_requests": api_requests,
        "api_attempts_tasks": attempts,
        "raw_responses_logged": api_requests,
        "raw_log_path": raw_log_path,
        "parse_failures": parse_failures,
        "unknown_tool_rejects": unknown_tool_rejects,
        "budget_rejects": budget_rejects,
        "target_prompt_tokens": args.target_prompt_tokens,
        "target_max_completion_tokens": args.target_max_completion_tokens,
        "tool_menu_min": args.tool_menu_min,
        "tool_menu_max": args.tool_menu_max,
        "max_input_chars": args.max_input_chars,
        "written_candidates": len(candidates),
        "output_length_distribution": dict(output_len_counter),
        "unique_tool_names": len(tool_name_counter),
        "top_tool_names": tool_name_counter.most_common(20),
        "schema_conflict_count": len(schema_conflicts),
        "schema_conflicts_report": conflicts_path,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - t0,
        "out_json": args.out_json,
    }
    write_reports(args.epoch, summary, accepted_previews, rejected_previews)
    print(f"[gen] wrote {len(candidates)} candidates -> {args.out_json}", file=sys.stderr)
    print(f"[gen] raw responses -> {raw_log_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
