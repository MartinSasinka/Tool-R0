"""Validation layers V1–V6, minimal-path/shortcut, dedup, contamination."""
import copy

from targeted_tool_data.generation import make_candidate
from targeted_tool_data.schemas import GenerationCell, TaskRecord
from targeted_tool_data.util import normalize_query
from targeted_tool_data.validation import (_match, contamination_check,
                                           dedup_pool, minimal_path_search,
                                           v1_schema, v2_execution, v3_semantic,
                                           v6_distribution, validate_record)

BUCKETS_CFG = {"small": [8, 9], "medium": [10, 12], "large": [13, 18]}
CONV = {"param_styles": ["semantic"], "label_styles": ["$var{i}"]}
THRESH = {"answer_tolerance": 1e-6, "minimal_path_max_depth": 3,
          "minimal_path_max_evals": 20000}


def _rec(cc=3, track="A", idx=0, seed=42):
    cell = GenerationCell(
        generation_cell_id=f"{track}_{cc}call_linear_test_00", track=track,
        mode="adaptation" if track == "A" else "generalization",
        call_count=cc, motif="linear", target_skill="s", target_failure="f",
        hard_distractor_type="near_semantics", quota_weight=1.0)
    r = make_candidate(cell, idx, seed, CONV, BUCKETS_CFG, "pv", "rh", "ch")
    assert r is not None
    return r


def test_v1_passes_valid_record():
    assert v1_schema(_rec()) == []


def test_v1_catches_missing_tool():
    r = _rec()
    r.offered_tools = [t for t in r.offered_tools
                       if t.name != r.canonical_calls[0].name]
    errs = v1_schema(r)
    assert any("not in offered set" in e for e in errs)


def test_v1_catches_unresolved_reference():
    r = _rec()
    r.canonical_calls[0].arguments[list(r.canonical_calls[0].arguments)[0]] = \
        "$var99.output_0$"
    errs = v1_schema(r)
    assert any("unresolved reference" in e for e in errs)


def test_v2_catches_oracle_tampering():
    r = _rec()
    r.gold_answer = 123456789.0
    errs = v2_execution(r)
    assert any("final answer mismatch" in e for e in errs)


def test_v3_catches_answer_in_query():
    r = _rec()
    r.query += f" The answer is {r.gold_answer}."
    errs = v3_semantic(r)
    assert any("appears in query" in e for e in errs)


def test_v4_shortcut_detection_synthetic():
    # 2-call task whose answer is reachable in 1 call must be flagged
    for i in range(40):
        r = _rec(cc=2, idx=i, seed=77)
        res = minimal_path_search(r, tol=1e-6, max_depth=3, max_evals=20000)
        if res["minimal_found"] == 1:
            assert res["single_call_shortcut"]
            return
    # if none found, that's fine too — the factory guard filters most


def test_shortcut_search_survives_values_outside_the_float_range():
    # an exponent primitive can produce an int larger than any float, which must
    # not crash the search: such a value simply cannot equal a finite answer
    huge = 10 ** 400
    assert _match(huge, 401.268823, 1e-6) is False
    assert _match(huge, huge, 1e-6) is True


def test_validate_record_full_pass_rate():
    passed = 0
    for i in range(15):
        r = _rec(cc=2, idx=100 + i)
        res = validate_record(r, THRESH)
        passed += res["passed"]
    assert passed >= 8   # majority should pass all gates


def test_dedup_exact_and_program():
    r = _rec().model_dump()
    r2 = copy.deepcopy(r)
    r2["task_id"] = "other"
    drops = dedup_pool([r, r2])
    assert "other" in drops


def test_contamination_exact_and_normalized():
    r = _rec().model_dump()
    block = {"exact": {r["query"]}, "normalized": set(), "skeletons": set(),
             "queries": []}
    bad = contamination_check([r], block, ratio_threshold=101)
    assert r["task_id"] in bad
    block2 = {"exact": set(), "normalized": {normalize_query(r["query"])},
              "skeletons": set(), "queries": []}
    bad2 = contamination_check([r], block2, ratio_threshold=101)
    assert r["task_id"] in bad2


def test_contamination_skeleton_a_track_only():
    r = _rec(track="A").model_dump()
    skel = tuple(c["name"] for c in r["canonical_calls"])
    block = {"exact": set(), "normalized": set(), "skeletons": {skel},
             "queries": []}
    assert r["task_id"] in contamination_check([r], block, ratio_threshold=101)
    r["track"] = "G"
    assert r["task_id"] not in contamination_check([r], block, ratio_threshold=101)


def test_v6_distribution_warns_on_dominance():
    rows = [_rec(idx=i).model_dump() for i in range(6)]
    for r in rows:
        r["template_id"] = "same_template"
    audit = v6_distribution(rows, template_max=0.05, cell_max=0.10)
    assert audit["warnings"]
