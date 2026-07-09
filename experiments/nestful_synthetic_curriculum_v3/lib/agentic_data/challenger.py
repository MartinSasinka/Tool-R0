"""Challenger agent: proposes NESTFUL-like task candidates as structured JSON.

The challenger does NOT invent tools and does NOT compute answers. It composes
questions + gold-call plans over the deterministic executable tool registry
(lib/nestful_like_generator.TOOLS — written from scratch, aggregate NESTFUL
style only). The executor is the source of truth for observations/answers.
"""
from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Optional

from ..nestful_like_generator import TOOLS
from .schema import MOTIFS, STAGES

_MOTIF_GOALS = {
    "long_chain": "a strict linear chain: every call after the first consumes "
                  "the previous call's result",
    "argument_binding": "calls mixing literal arguments from the question text "
                        "with exactly one reference argument per dependent call",
    "reference_reuse": "one later call re-uses the result of the FIRST call "
                       "(not only the immediately preceding one)",
    "distractor_heavy": "a normal chain, but name entities/quantities so that "
                        "several offered distractor tools look plausible",
}


def tool_catalog_text(tool_names: Optional[List[str]] = None) -> str:
    """Compact registry documentation embedded in the challenger prompt."""
    lines = []
    for name in sorted(tool_names or TOOLS.keys()):
        t = TOOLS[name]
        params = ", ".join(f"{p}:{typ}" for p, (typ, _d) in t["params"].items())
        lines.append(f"- {name}({params}) -> {t['out_key']}:{t['out_type']} | "
                     f"{t['description']}")
    return "\n".join(lines)


def challenger_messages(*, stage: str, motif: str, n_candidates: int,
                        feedback_block: str, rng: random.Random
                        ) -> List[Dict[str, str]]:
    lo, hi = STAGES[stage]
    call_target = f"exactly {lo}" if lo == hi else f"between {lo} and {hi}"
    # subset the catalog per batch for diversity + prompt-size control
    subset = rng.sample(sorted(TOOLS.keys()), k=min(18, len(TOOLS)))
    system = (
        "You are a CHALLENGER agent creating training tasks for a tool-use "
        "model, in the style of the NESTFUL benchmark (nested, executable "
        "function calls where later calls consume earlier outputs).\n"
        "You must output STRICT JSON only — no markdown, no commentary.\n"
        "Rules:\n"
        "- Use ONLY tools from the provided registry, with EXACTLY the listed "
        "parameter names.\n"
        "- To pass a previous result into a later call, set the argument value "
        "to the string \"$varN.<output_key>$\" where N is the 1-based index of "
        "the earlier call and <output_key> is that tool's output key.\n"
        "- Every literal argument value must be stated in the question text in "
        "natural language (never as $var$ syntax).\n"
        "- The question must be a single concise paragraph (25-60 words), "
        "concrete and unambiguous, phrased so the solver must perform ALL "
        "steps (continuation pressure: 'then', 'use that result', 'finally').\n"
        "- Do NOT compute or state the final numeric answer anywhere.\n"
        "- Do NOT mention datasets, stages, motifs or tools' internal names "
        "in the question text.\n"
        "- Optional field `rationale`: at most ONE short sentence (no "
        "step-by-step reasoning)."
    )
    user = (
        f"TOOL REGISTRY (subset for this batch):\n{tool_catalog_text(subset)}\n\n"
        f"TASK: propose {n_candidates} DIVERSE task candidates.\n"
        f"- calls per task: {call_target}\n"
        f"- target dependency pattern: {_MOTIF_GOALS[motif]}\n"
        f"- vary domains, scenarios, quantities and sentence structure across "
        f"candidates\n"
        f"{feedback_block}\n"
        "OUTPUT (strict JSON): {\"candidates\": [{\n"
        "  \"question\": \"...\",\n"
        "  \"tool_names\": [\"tools you actually call, in order\"],\n"
        "  \"gold_calls\": [{\"name\": \"...\", \"arguments\": {...}, "
        "\"label\": \"$var1\"}],\n"
        f"  \"motif_type\": \"{motif}\",\n"
        "  \"answer_type\": \"scalar|string|boolean\",\n"
        "  \"rationale\": \"one short sentence\"\n"
        "}]}"
    )
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


def parse_candidates(parsed: Any) -> List[Dict[str, Any]]:
    """Extract the candidate list from the challenger's parsed JSON."""
    if parsed is None:
        return []
    if isinstance(parsed, list):
        return [c for c in parsed if isinstance(c, dict)]
    if isinstance(parsed, dict):
        cands = parsed.get("candidates")
        if isinstance(cands, list):
            return [c for c in cands if isinstance(c, dict)]
        if "question" in parsed:                     # single-candidate object
            return [parsed]
    return []


def normalize_candidate(cand: Dict[str, Any]) -> Dict[str, Any]:
    """Best-effort normalization before validation (labels, arg parsing)."""
    calls = cand.get("gold_calls") or []
    for i, c in enumerate(calls):
        if isinstance(c, dict):
            label = str(c.get("label") or f"$var{i + 1}")
            if not label.startswith("$"):
                label = "$" + label
            c["label"] = label
            args = c.get("arguments")
            if isinstance(args, str):
                try:
                    c["arguments"] = json.loads(args)
                except (json.JSONDecodeError, TypeError):
                    pass
    if "tool_names" not in cand and isinstance(cand.get("tools"), list):
        cand["tool_names"] = cand["tools"]
    cand.setdefault("motif_type", MOTIFS[0])
    return cand
