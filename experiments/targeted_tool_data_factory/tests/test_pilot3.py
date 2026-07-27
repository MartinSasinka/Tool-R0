"""Pilot3-specific regressions (does not mutate pilot2 artefacts)."""
from __future__ import annotations

import sys
from pathlib import Path

FACTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FACTORY / "src"))

from targeted_tool_data.generation import (  # noqa: E402
    build_cells_v2, derive_call_bucket_shares,
)
from targeted_tool_data.util import load_config  # noqa: E402
from targeted_tool_data.schemas import TargetProfile  # noqa: E402


def dummy_profile() -> TargetProfile:
    return TargetProfile(
        target="t", source="s", n_rows=10, profile_version="pv",
        call_count_dist={"2": 0.33, "3": 0.22, "4": 0.14, "5": 0.09, "6+": 0.22},
        motif_dist={"linear": 0.55, "fan_in": 0.43, "mixed": 0.02},
        reference_task_rate=1.0, reference_arg_share=0.4, direct_arg_share=0.6,
        arg_type_dist={"int": 0.6, "reference": 0.4},
        numeric_string_rate=0.02,
        answer_type_dist={"float": 0.8, "string": 0.07, "int": 0.05,
                          "list": 0.05, "bool": 0.03},
        output_field_names={"output_0": 1.0},
        tools_per_task={"mean": 11},
        relevant_ratio_mean=0.25,
        tool_name_morphology={"tokens_per_name": {"1": 0.5}, "single_word_share": 0.5},
        tool_description_length={"mean": 60},
        signature_similarity_mean=0.0,
        question_length={"mean": 160},
        student_failure_profile={"win_rate_by_call_bucket": {"2": 0.45, "3": 0.62}},
    )


def test_call_bucket_boosts_raise_long_horizon_without_killing_shape():
    base = derive_call_bucket_shares(
        dummy_profile().call_count_dist, dummy_profile().student_failure_profile)
    boosted = derive_call_bucket_shares(
        dummy_profile().call_count_dist, dummy_profile().student_failure_profile,
        call_bucket_boosts={"5": 0.025, "6+": 0.035})
    assert boosted["5"] > base["5"]
    assert boosted["6+"] > base["6+"]
    assert boosted["2"] < base["2"]
    assert abs(sum(boosted.values()) - 1.0) < 1e-9


def test_pilot3_config_loads_and_cells_respect_boosts():
    cfg = load_config(FACTORY / "configs" / "pilot3_local.yaml")
    assert cfg["version"] == "pilot3"
    assert cfg["selection"]["n_selected"] == 1000
    assert cfg["selection"]["split"] == {"train": 600, "heldout": 200, "reserve": 200}
    assert cfg["thresholds"]["cell_max_share"] == 0.08
    assert cfg["generation"]["call_bucket_boosts"]["5"] == 0.025

    cells = build_cells_v2(dummy_profile(), cfg, ["adaptation", "generalization"], 0.55)
    assert cells
    g_share = sum(c.quota_weight for c in cells if c.track == "G")
    assert 0.40 <= g_share <= 0.50
    long_share = sum(c.quota_weight for c in cells if c.call_count >= 5)
    # With boosts, long-horizon weight should be meaningfully above NESTFUL raw
    # 5+6+ = 0.315 (before D07 2-call oversample).
    assert long_share > 0.30


def test_pilot2_config_untouched_by_pilot3_defaults():
    cfg2 = load_config(FACTORY / "configs" / "pilot2_local.yaml")
    assert cfg2["version"] == "pilot2"
    assert cfg2["selection"]["n_selected"] == 320
    assert "call_bucket_boosts" not in (cfg2.get("generation") or {})
