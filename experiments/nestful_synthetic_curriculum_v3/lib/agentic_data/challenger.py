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
    "fan_in": "a genuine FAN-IN: two EARLIER, INDEPENDENT calls each produce a "
             "result, and a LATER call consumes BOTH of them as separate "
             "arguments (needs >= 3 total calls)",
}

# Semantic families the executable tools carry (spec: JSON-type compatibility
# is not enough — a Fahrenheit value is not a price). Told to the challenger
# so it self-avoids nonsensical bindings before the (harder) code-level gate
# rejects them for free but wastes a generation round.
_SEMANTIC_HINT = (
    "- Respect REAL-WORLD units when chaining a result into a later call: a "
    "temperature is never a price, a percentage, a mass or a distance; a "
    "fuel/volume quantity is never money; a duration is never a price. The "
    "ONLY arguments that may freely accept ANY earlier numeric result are "
    "explicitly unit-agnostic math slots (e.g. `part`, `whole`, `value`, "
    "`numerator`, `denominator`, `minuend`, `subtrahend`, `dimension`, "
    "`threshold`, the `values` list of a statistics tool). If you want to "
    "reinterpret a real quantity as a generic number, route it through one "
    "of those unit-agnostic slots, not into a domain-specific slot like "
    "`net_price`, `principal_amount` or `annual_rate_percent`."
)

# Discourage the "First X. Then Y." boilerplate that dominated the pilot —
# vary sentence structure like real NESTFUL questions do.
_PHRASING_HINT = (
    "- Vary sentence structure across candidates: at most one of the "
    "{n_candidates} candidates may use an explicit 'First, ... Then, ...' "
    "template. Prefer embedded/relative clauses, conditional phrasing, or "
    "a plain narrative that implies order without labeling it as steps "
    "(e.g. 'A store had X and sold Y; split what's left across Z shelves.')."
)

# Encourage output-type diversity: the registry has string/boolean/count
# outputs, not only scalars — most Stage 2 pilot rows ended in a plain number.
_OUTPUT_TYPE_HINT = (
    "- Do not let every candidate end in a plain number. Where it fits "
    "naturally, end the chain on a tool that returns a STRING "
    "(format_as_currency, repeat_word), a BOOLEAN (is_above_threshold), or "
    "a COUNT (character_count, units_per_box, remaining_stock) so answer "
    "types are diverse across the batch."
)


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
        "- Argument keys must match the registry parameter names EXACTLY — "
        "any unknown or missing key makes the task invalid.\n"
        "- Every literal argument value must be stated in the question text in "
        "natural language (never as $var$ syntax).\n"
        "- Avoid trivial single-operation arithmetic; every call after the "
        "first should consume an earlier result.\n"
        "- The question must be a single concise paragraph (25-60 words), "
        "concrete and unambiguous, phrased so the solver must perform ALL "
        "steps (continuation pressure: 'then', 'use that result', 'finally').\n"
        "- Do NOT compute or state the final numeric answer anywhere.\n"
        "- Do NOT mention datasets, stages, motifs or tools' internal names "
        "in the question text.\n"
        + _SEMANTIC_HINT + "\n"
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
        f"{_PHRASING_HINT.format(n_candidates=n_candidates)}\n"
        f"{_OUTPUT_TYPE_HINT}\n"
        f"{feedback_block}\n"
        "OUTPUT (strict JSON): {\"candidates\": [{\n"
        "  \"question\": \"...\",\n"
        "  \"tool_names\": [\"tools you actually call, in order\"],\n"
        "  \"gold_calls\": [{\"name\": \"...\", \"arguments\": {...}, "
        "\"label\": \"$var1\"}],\n"
        f"  \"motif_type\": \"{motif}\",\n"
        "  \"answer_type\": \"scalar|string|boolean|list\",\n"
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


def _key_variants(key: str) -> str:
    return key.lower().replace("_", "").replace("-", "")


def repair_candidate(cand: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic repair of common challenger mistakes BEFORE the executor
    gate (spec 7B) — zero API cost. Only unambiguous repairs are applied:
      * argument keys that differ only by case/underscores are renamed to the
        registry schema (e.g. `principalAmount` -> `principal_amount`);
      * a single unknown key is mapped to the single missing schema key.
    Repairs are recorded in cand['_repairs'] for provenance.
    """
    repairs: List[str] = []
    calls = cand.get("gold_calls") or []
    for i, call in enumerate(calls):
        if not isinstance(call, dict):
            continue
        name = call.get("name")
        args = call.get("arguments")
        if name not in TOOLS or not isinstance(args, dict):
            continue
        expected = set(TOOLS[name]["params"].keys())
        got = set(args.keys())
        if got == expected:
            continue
        # 1) case/underscore-insensitive rename
        variant_map = {_key_variants(e): e for e in expected}
        renamed = {}
        for k, v in args.items():
            target = variant_map.get(_key_variants(k))
            renamed[target if target and target not in renamed else k] = v
        if set(renamed.keys()) == expected:
            call["arguments"] = renamed
            repairs.append(f"call{i + 1}:arg_keys_case_repair")
            continue
        # 2) single unknown key -> single missing key
        unknown = got - expected
        missing = expected - got
        if len(unknown) == 1 and len(missing) == 1:
            k_bad, k_good = next(iter(unknown)), next(iter(missing))
            args[k_good] = args.pop(k_bad)
            repairs.append(f"call{i + 1}:{k_bad}->{k_good}")
    if repairs:
        cand["_repairs"] = repairs
    return cand
