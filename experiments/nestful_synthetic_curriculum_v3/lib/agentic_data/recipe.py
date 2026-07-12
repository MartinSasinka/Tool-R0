"""Challenger recipe: versioned prompt template + batch-feedback revision.

This is the Autodata 'data scientist' state: the orchestrator analyzes each
batch (rejection reasons, weak/strong scores) and appends targeted LEARNINGS
to the recipe, exactly like the paper's challenger-prompt update step
(Agentic Self-Instruct, §3.1.1: feedback lists which questions were too easy
with weak-solver scores, which failed the strong solver, which were rejected
by the quality verifier; the challenger is asked for a different angle).
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

RECIPE_BASE_VERSION = "agentic_v1"

# Rejection reason → targeted instruction for the next challenger batch.
_FEEDBACK_RULES: Dict[str, str] = {
    "too_easy_both_solvers_pass":
        "Previous tasks were TOO EASY (the weak solver solved them). Make the "
        "chain less obvious: bury the needed values mid-sentence, use "
        "distractor tools from the same domain, and pick less common tool "
        "combinations. Do NOT spell out which tool to use at each step.",
    "weak_solver_passed":
        "The weak solver keeps succeeding. Increase implicitness: avoid "
        "phrasing that mirrors tool names, require the model to infer the "
        "correct order of dependent calls.",
    "too_hard_both_solvers_fail":
        "Previous tasks were TOO HARD or ambiguous (even the strong solver "
        "failed). Keep every needed literal value stated explicitly once, "
        "make each step depend on exactly one previous result, avoid "
        "underspecified wording.",
    "strong_solver_failed":
        "The strong solver failed — questions were ambiguous about argument "
        "binding. State units and quantities clearly; each argument must be "
        "either a stated literal or exactly one previous result.",
    "non_executable_gold_trace":
        "Some gold traces did not execute. Use ONLY tools from the registry "
        "with EXACTLY the listed parameter names; reference a previous result "
        "as \"$varN.<output_key>$\" using that tool's output key.",
    "wrong_gold_answer":
        "Your predicted final answers disagreed with real execution. Do not "
        "guess numeric results; the executor computes them. Focus on valid "
        "call structure.",
    "duplicate_question":
        "Questions repeated earlier batches. Vary domains, scenarios, number "
        "ranges and sentence structure.",
    "duplicate_trace":
        "Tool chains repeated earlier batches. Vary tool combinations and "
        "argument values.",
    "not_nestful_like":
        "Style drifted from NESTFUL: keep questions concise (25-60 words), "
        "concrete, single-paragraph, imperative or interrogative, with "
        "realistic everyday quantities.",
    "ambiguous_question":
        "Questions were ambiguous. Every argument of every call must be "
        "recoverable from the question text alone.",
    "unresolved_var":
        "Do not put $var...$ syntax into the question or final answer; "
        "references belong ONLY in gold_calls arguments.",
    "metadata_leakage":
        "Never mention stages, motifs, recipes or dataset names inside the "
        "question text.",
    "cot_leakage":
        "Do not include long reasoning in `rationale`; one short sentence "
        "maximum.",
    "diversity_cap_weak_score":
        "Recent accepted tasks all have the SAME difficulty level for the "
        "weak solver. Vary difficulty: mix tasks where the solver would fail "
        "immediately (wrong tool/args) with tasks where it would fail only at "
        "the last step.",
    "diversity_cap_failure_type":
        "The weak solver keeps failing in the SAME way on accepted tasks. "
        "Vary the failure mode you target: sometimes make argument binding "
        "the hard part, sometimes tool choice among similar distractors, "
        "sometimes reference reuse of an EARLY result, sometimes chain length.",
    "low_grpo_signal_prediction":
        "Tasks lacked usable GRPO training signal for the weak model: rollouts "
        "were all the same reward (too easy, all identical failures, or only "
        "parse errors). Target the SWEET SPOT: the weak model should sometimes "
        "succeed on a full trace but not always — vary WHICH step is hard "
        "(tool choice among same-domain distractors, argument binding, or "
        "reusing an earlier result). Avoid questions where every rollout fails "
        "in exactly the same way.",
    "weak_strong_gap_too_small":
        "Weak and strong solvers scored too similarly — the task does not "
        "separate them enough. Make the correct tool chain less obvious in the "
        "question (no mirrored tool names), add plausible same-domain "
        "distractors, and require inferring call order — while keeping every "
        "literal value stated once and the gold trace executable.",
    "invalid_trace_labels":
        "Gold traces had invalid structure: labels must be unique sequential "
        "$var1, $var2, ... (one per call); references must use the EXACT "
        "output_key of the referenced call (e.g. \"$var1.output_0$\"); "
        "tool_names length must match gold_calls; never reuse a label.",
    "semantic_incompatible_reference":
        "Arguments bound incompatible quantities (e.g. temperature into a money "
        "slot, area into a rate-percent slot). Match semantic types: use "
        "same-domain outputs for domain-specific slots; only generic slots "
        "(part, whole, value, values, amount) may accept cross-domain "
        "numerics. State units clearly in the question.",
}


class Recipe:
    """Versioned challenger recipe with bounded feedback memory."""

    def __init__(self, max_learnings: int = 6) -> None:
        self.iteration = 0
        self.max_learnings = max_learnings
        self.learnings: List[str] = []
        self.history: List[Dict[str, Any]] = []

    @property
    def version(self) -> str:
        return f"{RECIPE_BASE_VERSION}.{self.iteration}"

    def update_from_batch(self, rejection_counter: Counter,
                          batch_stats: Dict[str, Any]) -> None:
        """Data-scientist step: turn batch analysis into next-batch guidance."""
        self.iteration += 1
        top = [r for r, _n in rejection_counter.most_common(3)]
        new_learnings = [_FEEDBACK_RULES[r] for r in top if r in _FEEDBACK_RULES]
        # newest learnings first, bounded window so the prompt cannot grow forever
        self.learnings = (new_learnings + self.learnings)[: self.max_learnings]
        self.history.append({
            "iteration": self.iteration,
            "rejections": dict(rejection_counter),
            "batch_stats": batch_stats,
        })

    def feedback_block(self) -> str:
        if not self.learnings:
            return ""
        lines = "\n".join(f"- {ln}" for ln in self.learnings)
        return ("\nLEARNINGS FROM PREVIOUS BATCHES (address ALL of these, "
                "try a different angle than before):\n" + lines + "\n")

    def as_dict(self) -> Dict[str, Any]:
        return {"version": self.version, "iteration": self.iteration,
                "learnings": self.learnings, "history": self.history}
