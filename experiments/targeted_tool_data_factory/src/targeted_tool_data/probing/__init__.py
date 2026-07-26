"""Cheap student-in-the-loop cascade P0–P3 for Qwen3-4B (DESIGN.md §17).

P0: structural difficulty, no model (always runs; explicitly a heuristic,
    never an oracle).
P1: 1 greedy rollout on a limited pool.
P2: 4 rollouts on P1 survivors relevant to selection.
P3: 8 rollouts on borderline/final candidates.

If no local student is reachable: status NOT_RUN_LOCAL + exact command.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from .. import registry as reg
from ..providers import BaseProvider
from ..util import is_reference

NOT_RUN_CMD = (
    "targeted-data probe --config configs/pilot_local.yaml --target nestful "
    "--provider openai_compatible_local --base-url http://127.0.0.1:1234/v1 "
    "--model qwen3-4b-instruct-2507 --resume"
)


# ── P0: structural difficulty (heuristic, model-free) ─────────────────────
def p0_structural(rec: Dict[str, Any]) -> float:
    d = 0.0
    d += min(rec["call_count"], 8) / 8 * 0.35
    d += min(rec["dependency_depth"], 6) / 6 * 0.15
    d += min(rec["offered_tool_count"], 18) / 18 * 0.10
    sims = rec.get("distractor_similarity") or {}
    d += (sims.get("signature", 0.0)) * 0.20
    d += (1.0 if rec.get("numeric_string_args") else 0.0) * 0.10
    d += rec.get("reference_arg_share", 0.0) * 0.10
    return round(min(d, 1.0), 4)


# ── rollout evaluation ────────────────────────────────────────────────────
_PROMPT = """You are given tools (JSON) and a task. Reply ONLY with a JSON \
array of tool calls, each {{"name": ..., "arguments": {{...}}, "label": "$var1"}}. \
Use "$varN.output_0$" style references to pass outputs between calls.

TOOLS:
{tools}

TASK: {query}
"""


def build_probe_prompt(rec: Dict[str, Any]) -> str:
    tools = [{"name": t["name"], "description": t["description"],
              "parameters": {p["name"]: p["type"] for p in t["params"]}}
             for t in rec["offered_tools"]]
    return _PROMPT.format(tools=json.dumps(tools, ensure_ascii=False), query=rec["query"])


def parse_calls(text: str) -> Optional[List[Dict[str, Any]]]:
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return None
    try:
        calls = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(calls, list) or not calls:
        return None
    out = []
    for c in calls:
        if not isinstance(c, dict) or "name" not in c:
            return None
        out.append({"name": c["name"], "arguments": c.get("arguments") or {},
                    "label": c.get("label", f"$var{len(out) + 1}")})
    return out


def execute_predicted(calls: List[Dict[str, Any]], rec: Dict[str, Any],
                      tol: float = 1e-6) -> Dict[str, Any]:
    """Executes predicted calls with OUR deterministic primitives (matching
    the offered specs); path-invariant success = final value equals oracle."""
    spec_by_name = {t["name"]: t for t in rec["offered_tools"]}
    values: Dict[str, Any] = {}
    executable = True
    failure = None
    last = None
    for c in calls:
        spec = spec_by_name.get(c["name"])
        if spec is None:
            return {"success": False, "executable": False, "failure_class": "unknown_tool"}
        prim = reg.get(spec["semantic_id"])
        pmap = {p["name"]: canon for p, (canon, _t, _s)
                in zip(spec["params"], prim.params)}
        kwargs = {}
        try:
            for pname, canon in pmap.items():
                v = c["arguments"].get(pname)
                if v is None:
                    raise ValueError(f"missing {pname}")
                if is_reference(v):
                    key = str(v).strip().strip("$").split(".")[0].replace("_", "")
                    if key not in values:
                        raise ValueError(f"unresolved {v}")
                    v = values[key]
                kwargs[canon] = v
            out = prim.fn(**kwargs)
        except Exception as exc:
            return {"success": False, "executable": False,
                    "failure_class": f"execution_error:{type(exc).__name__}"}
        label = str(c.get("label", "")).strip("$").replace("_", "")
        values[label or f"var{len(values) + 1}"] = out
        last = out
    gold = rec["gold_answer"]
    ok = False
    if isinstance(last, (int, float)) and isinstance(gold, (int, float)):
        ok = abs(float(last) - float(gold)) <= tol
    else:
        ok = last == gold
    if not ok:
        failure = ("too_few_calls" if len(calls) < rec["call_count"]
                   else "executable_wrong_final")
    return {"success": ok, "executable": executable,
            "failure_class": failure, "n_calls": len(calls)}


def probe_record(rec: Dict[str, Any], provider: BaseProvider, n_rollouts: int,
                 seed: int) -> Dict[str, Any]:
    gold_first = rec["canonical_calls"][0]["name"]
    prompt = build_probe_prompt(rec)
    successes = 0
    fclasses: List[str] = []
    first_hits = 0
    cont_hits = 0
    exec_hits = 0
    hashes: List[str] = []
    for i in range(n_rollouts):
        outs = provider.complete(prompt, max_tokens=768,
                                 temperature=0.0 if n_rollouts == 1 else 0.7,
                                 n=1, seed=seed + i)
        text = outs[0] if outs else ""
        hashes.append(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16])
        calls = parse_calls(text)
        if calls is None:
            fclasses.append("parse_or_no_call")
            continue
        if calls[0]["name"] == gold_first:
            first_hits += 1
            gold_second = (rec["canonical_calls"][1]["name"]
                           if len(rec["canonical_calls"]) > 1 else None)
            if gold_second and len(calls) > 1 and calls[1]["name"] == gold_second:
                cont_hits += 1
        res = execute_predicted(calls, rec)
        exec_hits += int(res["executable"])
        if res["success"]:
            successes += 1
        else:
            fclasses.append(res.get("failure_class") or "unknown")
    n = max(n_rollouts, 1)
    ent = 0.0
    if fclasses:
        c = Counter(fclasses)
        tot = sum(c.values())
        import math
        ent = -sum((v / tot) * math.log2(v / tot) for v in c.values())
    return {
        "rollouts": n_rollouts,
        "success_count": successes,
        "failure_classes": sorted(set(fclasses)),
        "failure_entropy": round(ent, 4),
        "first_tool_accuracy": round(first_hits / n, 4),
        "continuation_accuracy": round(cont_hits / n, 4),
        "executability": round(exec_hits / n, 4),
        "completion_hashes": hashes,
    }


def informative(success_count: int, rollouts: int,
                band: Tuple[float, float] = (0.125, 0.875)) -> bool:
    if rollouts == 0:
        return False
    rate = success_count / rollouts
    return band[0] <= rate <= band[1]
