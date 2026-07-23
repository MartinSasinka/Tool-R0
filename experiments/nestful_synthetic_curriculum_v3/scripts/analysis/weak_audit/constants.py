"""Shared constants for weak-model audit pipeline."""
from __future__ import annotations

SEED = 20260723
TARGET_SOFT_MAX = 230
HARD_MAX = 250
TOKEN_TARGET = 4000
TOKEN_HARD = 6000
TOKEN_OUTPUT_MAX = 180

COHORT_LIMITS = {
    "c0_win_e2_loss": None,
    "c0_loss_e2_win": 40,
    "official_win_reward_too_few": 50,
    "e2_executable_wrong_other": 35,
    "stable_win_control": 15,
    "stable_loss_control": 15,
}

COHORT_PRIORITY = [
    "c0_win_e2_loss",
    "c0_loss_e2_win",
    "official_win_reward_too_few",
    "e2_executable_wrong_other",
    "stable_win_control",
    "stable_loss_control",
]

ROOT_CAUSES = frozenset({
    "initial_tool_selection", "later_tool_selection", "argument_keys",
    "argument_values", "observation_ignored", "observation_misinterpreted",
    "wrong_output_field", "invalid_state_transition", "premature_stop",
    "valid_shorter_path", "executable_wrong_global_plan", "wrong_final_answer",
    "reward_mismatch", "evaluator_or_data_inconsistency", "unclear",
})

SHORTER_PATH_VERDICTS = frozenset({"valid", "invalid", "not_applicable", "unclear"})

RECOMMENDED_FIXES = frozenset({
    "outcome_reward", "process_reward", "terminal_classification",
    "tool_selection_data", "observation_grounding_data", "targeted_semantic_sft",
    "credit_assignment", "evaluator_or_data_fix", "no_change", "unclear",
})

REWARD_COMPONENTS = frozenset({
    "terminal_outcome", "call_count", "tool_name", "argument_keys",
    "argument_values", "executability", "episode_class", "process_total",
    "none", "unclear",
})

SYSTEM_PROMPT = """Jsi úsporný analytik multi-turn tool-use experimentu.

Dostaneš jeden diagnostický případ obsahující zadání, relevantní
tool schemas, několik trajektorií, skutečné observations, official
outcome a deterministicky vypočítané flags.

Executor a official scorer jsou autorita.

Nepřehodnocuj official win pouze podle podobnosti s gold trace.
Kratší cesta může být validní, pokud skutečně vede ke správnému
terminálnímu výsledku.

Tvým úkolem je pouze klasifikace, nikoliv dlouhé vysvětlování.

Urči:

1. první sémanticky významnou divergenci mezi trajektoriemi;
2. hlavní root cause;
3. zda je kratší cesta validní;
4. zda byla předchozí observation použita správně;
5. zda training reward správně řadí vítěznou a prohrávající cestu;
6. která reward komponenta pravděpodobně způsobuje mismatch;
7. jaký nejmenší typ intervence by případ řešil.

Povolené root_cause:

- initial_tool_selection
- later_tool_selection
- argument_keys
- argument_values
- observation_ignored
- observation_misinterpreted
- wrong_output_field
- invalid_state_transition
- premature_stop
- valid_shorter_path
- executable_wrong_global_plan
- wrong_final_answer
- reward_mismatch
- evaluator_or_data_inconsistency
- unclear

Povolené shorter_path_verdict:

- valid
- invalid
- not_applicable
- unclear

Povolené recommended_fix:

- outcome_reward
- process_reward
- terminal_classification
- tool_selection_data
- observation_grounding_data
- targeted_semantic_sft
- credit_assignment
- evaluator_or_data_fix
- no_change
- unclear

Povolené responsible_reward_component:

- terminal_outcome
- call_count
- tool_name
- argument_keys
- argument_values
- executability
- episode_class
- process_total
- none
- unclear

Používej jen informace v předloženém případu.
Nevymýšlej chybějící fakta.
Důkaz musí být krátký a odkazovat na konkrétní call nebo observation.

Vrať pouze validní JSON:

{
  "task_id": "",
  "first_divergence_turn": null,
  "root_cause": "",
  "shorter_path_verdict": "",
  "observation_used_correctly": null,
  "reward_ordering_correct": null,
  "responsible_reward_component": "",
  "recommended_fix": "",
  "confidence": 0.0,
  "evidence": ""
}
"""

REPAIR_PROMPT = (
    "Vrať stejný obsah jako validní JSON podle schématu z předchozí "
    "instrukce. Pouze JSON, bez markdown."
)
