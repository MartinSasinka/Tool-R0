"""Generation cells, candidate factory, distractors."""
import json

from targeted_tool_data.distractors import (build_offered_set, name_similarity,
                                            sig_similarity)
from targeted_tool_data.generation import (build_cells, generate_pool,
                                           make_candidate, record_to_canonical)
from targeted_tool_data.profile import featurize_row
from targeted_tool_data.schemas import GenerationCell, TargetProfile

BUCKETS_CFG = {"small": [8, 9], "medium": [10, 12], "large": [13, 18]}
CONV = {"param_styles": ["semantic", "generic"],
        "label_styles": ["$var{i}", "$var_{i}"]}


def _profile():
    return TargetProfile(
        target="t", source="s", n_rows=10, profile_version="pv",
        call_count_dist={"2": 0.33, "3": 0.22, "4": 0.14, "5": 0.09, "6+": 0.22},
        motif_dist={"linear": 0.55, "fan_in": 0.43, "mixed": 0.02},
        reference_task_rate=1.0, reference_arg_share=0.4, direct_arg_share=0.6,
        arg_type_dist={"int": 0.6, "reference": 0.4},
        numeric_string_rate=0.002,
        answer_type_dist={"float": 0.8, "string": 0.07},
        output_field_names={"output_0": 1.0},
        tools_per_task={"mean": 11},
        relevant_ratio_mean=0.25,
        tool_name_morphology={"tokens_per_name": {"1": 0.5}, "single_word_share": 0.5},
        tool_description_length={"mean": 60},
        signature_similarity_mean=0.0,
        question_length={"mean": 160},
        student_failure_profile={"win_rate_by_call_bucket": {"2": 0.45, "3": 0.62}},
    )


def _cell(track="A", cc=2):
    return GenerationCell(
        generation_cell_id=f"{track}_{cc}call_linear_test_00", track=track,
        mode="adaptation" if track == "A" else "generalization",
        call_count=cc, motif="linear", target_skill="s", target_failure="f",
        hard_distractor_type="near_semantics", quota_weight=1.0)


def test_build_cells_cover_tracks_and_buckets():
    cells = build_cells(_profile(), {}, ["adaptation", "generalization"], 0.6)
    tracks = {c.track for c in cells}
    assert tracks == {"A", "G"}
    assert abs(sum(c.quota_weight for c in cells) - 1.0) < 1e-9
    # failure-driven: 2-call oversampled beyond profile 0.33
    w2 = sum(c.quota_weight for c in cells
             if c.generation_cell_id.split("_")[1] == "2call")
    assert w2 > 0.33
    # no single cell above 10 %
    assert max(c.quota_weight for c in cells) <= 0.10
    # numeric-string cells exist
    assert any(c.numeric_string for c in cells)


def test_make_candidate_deterministic():
    c = _cell()
    r1 = make_candidate(c, 0, 123, CONV, BUCKETS_CFG, "pv", "rh", "ch")
    r2 = make_candidate(c, 0, 123, CONV, BUCKETS_CFG, "pv", "rh", "ch")
    assert r1 is not None and r2 is not None
    assert r1.model_dump() == r2.model_dump()
    r3 = make_candidate(c, 1, 123, CONV, BUCKETS_CFG, "pv", "rh", "ch")
    assert r3 is None or r3.query != r1.query


def test_candidate_contract_fields():
    r = make_candidate(_cell(cc=3), 0, 99, CONV, BUCKETS_CFG, "pv", "rh", "ch")
    assert r is not None
    assert r.call_count == 3
    assert len(r.canonical_calls) == 3
    assert len(r.oracle_observations) == 3
    assert r.gold_answer == r.oracle_observations[-1]
    assert r.offered_tool_count == len(r.offered_tools)
    assert r.relevant_tool_count >= 1
    assert r.generation_cell_id
    assert r.target_skill and r.target_failure_mode
    offered = {t.name for t in r.offered_tools}
    assert {c.name for c in r.canonical_calls} <= offered


def test_g_track_uses_distinct_vocabulary():
    r = make_candidate(_cell(track="G", cc=3), 0, 7, CONV, BUCKETS_CFG,
                       "pv", "rh", "ch")
    assert r is not None
    from targeted_tool_data import registry as reg
    a_names = {s.name for p in reg.all_primitives().values() for s in p.surfaces_a}
    assert not ({t.name for t in r.offered_tools} & a_names)


def test_offered_set_and_hard_distractors():
    import random
    from targeted_tool_data.render import render_tool
    rng = random.Random(0)
    gold = [render_tool("multiply", "A", rng), render_tool("add", "A", rng)]
    offered, gold_pos, sims = build_offered_set(
        gold, "A", rng, 12, "same_signature_different_semantics")
    assert len(offered) == 12
    hard = [t for t in offered if t.is_distractor and t.distractor_type != "easy"]
    assert len(hard) >= 2
    for h in hard:
        assert h.semantic_id not in {g.semantic_id for g in gold}
        assert h.similarity_to_gold["signature"] > 0
    assert sims["signature"] > 0
    assert len(gold_pos) == len(gold)


def test_similarity_metrics():
    assert name_similarity("sum_of_values", "sum_two_numbers") > \
        name_similarity("sum_of_values", "celsius_to_fahrenheit")


def test_generate_pool_stats_and_canonical_roundtrip():
    cells = [_cell(cc=2), _cell(track="G", cc=2)]
    for c in cells:
        c.quota_weight = 0.5
    pool, stats = generate_pool(cells, 10, 42, CONV, BUCKETS_CFG, "pv", "ch")
    assert pool
    for cid, st in stats.items():
        assert st["generated"] <= st["requested"] * 1.01 + 1
    row = pool[0].model_dump()
    canon = record_to_canonical(row)
    feat = featurize_row(canon, ["2", "3", "4", "5", "6+"])
    assert feat["call_count"] == row["call_count"]
    assert feat["n_tools"] == row["offered_tool_count"]
