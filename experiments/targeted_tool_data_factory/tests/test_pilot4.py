"""Phase-R tests for the pilot4 factory, provenance, sampler and logging."""
from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from targeted_tool_data import capability as cap
from targeted_tool_data import provenance as prov
from targeted_tool_data import query_realism as qr
from targeted_tool_data import registry as reg
from targeted_tool_data.executor import execute
from targeted_tool_data.pilot4 import compare as cmp
from targeted_tool_data.pilot4 import distractors as dis
from targeted_tool_data.pilot4 import patterns as pat
from targeted_tool_data.pilot4 import query_render as qrender
from targeted_tool_data.pilot4 import select as sel
from targeted_tool_data.pilot4 import surface_render as srender
from targeted_tool_data.pilot4 import validate as v4
from targeted_tool_data.pilot4.cells import BUCKET_CALLS, Cell, build_cells
from targeted_tool_data.pilot4.difficulty import build_signature, difficulty_band
from targeted_tool_data.pilot4.generate import build_program_for_cell, render_variant
from targeted_tool_data.pilot4.program import make_spec
from targeted_tool_data.pilot4.validators import v7_plan_leak, v8_distractor_validity
from targeted_tool_data.profile_v2 import (build_profile_v2,
                                           derive_topology_constraints,
                                           featurize, graph_features,
                                           topology_diversity)

MODULE_ROOT = Path(__file__).resolve().parents[1]
PILOT4_DIR = MODULE_ROOT / "outputs" / "pilot4_profile_safe"


def _cell(bucket="4", pattern="LINEAR_CHAIN", mode="GOAL_BASED_IMPLICIT",
          track="A_NATIVE", profile="balanced_hard") -> Cell:
    return Cell(
        cell_id=f"T_{bucket}_{pattern}_{track[0]}", mode="PROFILE_SAFE",
        track=track, query_mode=mode, call_bucket=bucket,
        pattern_family=pattern,
        capability_mix=["arithmetic.binary", "arithmetic.reduction",
                        "statistics", "rounding"],
        capability_mix_name="numeric_core",
        target_failure_skill="wrong_reference_target",
        target_skill="dependency_tracking", offered_tool_range=(10, 14),
        distractor_profile=profile, reference_profile="nestful_like",
        difficulty_band="medium", quota_weight=1.0)


def _make_record(cell: Cell, *, track=None, query_mode=None, attempts: int = 24):
    """Generation is rejection-sampled, so a fixed seed may legitimately fail.

    Walking a deterministic seed sequence keeps the test reproducible while
    still exercising a real program instead of a hand-written stub.
    """
    from targeted_tool_data.pilot4.patterns import PatternError

    for seed in range(attempts):
        try:
            spec = build_program_for_cell(cell, random.Random(seed))
        except PatternError:
            continue
        rec = _render(spec, cell, track=track, query_mode=query_mode)
        if rec is not None:
            return spec, rec
    pytest.fail(f"no candidate could be generated for {cell.cell_id}")


def _render(spec, cell: Cell, *, track=None, query_mode=None, attempts: int = 8):
    for seed in range(attempts):
        rec = render_variant(spec, cell, random.Random(seed), track=track,
                             query_mode=query_mode)
        if rec is not None:
            return rec
    return None


# ── provenance ────────────────────────────────────────────────────────────
def test_canonical_fingerprint_ignores_export_metadata():
    base = {
        "question": "What is 2 plus 3?",
        "tools": [{"name": "add", "parameters": {"properties": {"a": {"type": "number"}}}}],
        "gold_calls": [{"name": "add", "arguments": {"a": 2, "b": 3.0}, "label": "$var_1"}],
        "gold_answer": 5.0,
    }
    noisy = {**base, "split": "train", "generated_at": "2026-01-01T00:00:00Z",
             "source_path": "/tmp/x.jsonl", "export_version": "v9"}
    assert prov.row_fingerprints(base)["exact"] == prov.row_fingerprints(noisy)["exact"]


def test_canonical_fingerprint_is_sensitive_to_call_order():
    a = {"question": "q", "tools": [], "gold_calls": [
        {"name": "add", "arguments": {}, "label": "$var_1"},
        {"name": "mul", "arguments": {}, "label": "$var_2"}], "gold_answer": 1}
    b = {"question": "q", "tools": [], "gold_calls": list(reversed(a["gold_calls"])),
         "gold_answer": 1}
    assert prov.row_fingerprints(a)["exact"] != prov.row_fingerprints(b)["exact"]


def test_float_canonicalisation_matches_int_valued_floats():
    a = {"question": "q", "tools": [], "gold_calls": [
        {"name": "add", "arguments": {"a": 3.0}, "label": "$var_1"}], "gold_answer": 3}
    b = {"question": "q", "tools": [], "gold_calls": [
        {"name": "add", "arguments": {"a": 3}, "label": "$var_1"}], "gold_answer": 3.0}
    assert prov.row_fingerprints(a)["exact"] == prov.row_fingerprints(b)["exact"]


def test_byte_level_first_n_comparison(tmp_path: Path):
    parent = tmp_path / "parent.jsonl"
    lines = [json.dumps({"sample_id": f"s{i}", "v": i}) for i in range(10)]
    parent.write_text("\n".join(lines) + "\n", encoding="utf-8")
    subset = tmp_path / "subset.jsonl"
    subset.write_text("\n".join(lines[:4]) + "\n", encoding="utf-8")

    res = prov.byte_level_prefix_match(parent, subset, 4)
    assert res["exact_bytes_match"] is True

    tampered = tmp_path / "tampered.jsonl"
    tampered.write_text("\n".join(lines[1:5]) + "\n", encoding="utf-8")
    assert prov.byte_level_prefix_match(parent, tampered, 4)["exact_bytes_match"] is False


def test_equal_row_count_alone_is_not_identity(tmp_path: Path):
    def row(tag: str, i: int) -> str:
        return json.dumps({"sample_id": f"{tag}{i}", "question": f"question {tag}{i}",
                           "tools": [], "gold_answer": i,
                           "gold_calls": [{"name": tag, "arguments": {"a": i},
                                           "label": "$var_1"}]})

    parent = tmp_path / "p.jsonl"
    subset = tmp_path / "s.jsonl"
    parent.write_text("\n".join(row("a", i) for i in range(3)) + "\n", encoding="utf-8")
    subset.write_text("\n".join(row("b", i) for i in range(3)) + "\n", encoding="utf-8")
    res = prov.audit_subset(parent, subset)
    assert res["status"] == prov.STATUS_DIFFERENT
    assert res["n_canonical_matched"] == 0


# ── query realism ─────────────────────────────────────────────────────────
def test_operation_explicitness_separates_exact_from_implicit():
    explicit = qr.operation_explicitness(
        "First subtract 3 from 10, then multiply the result by 2.",
        ["subtract", "multiply"])
    assert explicit["exact_operation_coverage"] == 1.0
    implicit = qr.operation_explicitness(
        "A shipment lost part of its load; how much cargo is still on board?",
        ["subtract", "multiply"])
    assert implicit["exact_operation_coverage"] == 0.0
    assert implicit["implicit_operation_rate"] > 0.0


def test_semantic_cues_do_not_count_as_leakage():
    """"how much remains" implies subtraction without naming it."""
    audit = qr.operation_explicitness("How much remains in the budget?", ["subtract"])
    assert audit["per_operation"][0]["cue_level"] == "semantic"
    assert audit["lexical_operation_coverage"] == 0.0
    assert audit["implicit_operation_rate"] == 1.0


def test_sequence_leakage_detects_gold_order():
    ordered = qr.audit_task("First add 2 and 3, then divide the result by 4.",
                            ["add", "divide"])
    shuffled = qr.audit_task("Divide by 4 what you get after you add 2 and 3.",
                             ["add", "divide"])
    assert ordered["sequence_leakage"] > shuffled["sequence_leakage"]
    assert ordered["lcs_ratio"] == 1.0
    assert shuffled["kendall_agreement"] < 0.0


def test_procedural_cue_detection_counts_step_markers():
    cues = qr.procedural_cues(
        "Step 1: add them. Step 2: then use the result of the previous step. "
        "Finally, report it.")
    assert cues["step_number_count"] >= 2
    assert cues["explicit_intermediate_reference_count"] >= 1
    assert qr.procedural_cue_count("How many are left?") == 0


def test_query_mode_classifier_is_deterministic_and_labelled():
    q = "Step 1: subtract 4 from 9. Step 2: multiply the result by 3."
    a = qr.classify_query_mode(q, ["subtract", "multiply"])
    b = qr.classify_query_mode(q, ["subtract", "multiply"])
    assert a == b
    assert a["query_mode"] == "PROCEDURAL_EXPLICIT"
    assert a["confidence"] > 0.0
    assert a["evidence_flags"]["has_step_numbers"] is True
    assert a["evidence_flags"]["names_every_operation"] is True

    goal = qr.classify_query_mode(
        "A tank holds water for a village; how much is left for tomorrow?",
        ["subtract", "multiply"])
    assert goal["query_mode"] == "GOAL_BASED_IMPLICIT"
    assert goal["evidence_flags"]["no_operation_named"] is True


# ── graph patterns and transformations ────────────────────────────────────
def _assert_acyclic(shape) -> None:
    """A shape lists parents per node; a DAG keeps every parent index lower."""
    for i, parents in enumerate(shape):
        assert len(set(parents)) == len(parents), "duplicate parent edge"
        for parent in parents:
            assert parent < i, "an edge points forward: the graph is not a DAG"


def _reaches_all(shape) -> bool:
    """Every node must be on a path to the sink, or it is a dead gold call."""
    reachable = {len(shape) - 1}
    for i in range(len(shape) - 1, -1, -1):
        if i in reachable:
            reachable.update(shape[i])
    return len(reachable) == len(shape)


@pytest.mark.parametrize("pattern", pat.PATTERN_FAMILIES)
def test_every_pattern_builds_a_valid_dag(pattern: str):
    n = pat.MIN_CALLS[pattern]
    for n_calls in (n, n + 2):
        shape = pat.build_shape(pattern, n_calls)
        assert len(shape) == n_calls
        _assert_acyclic(shape)
        assert _reaches_all(shape), "pattern produced a dead gold call"


def test_pattern_availability_is_gated_by_call_count():
    assert pat.patterns_for(2) == ["LINEAR_CHAIN"]
    assert "NESTED_AGGREGATION" not in pat.patterns_for(6)
    assert "NESTED_AGGREGATION" in pat.patterns_for(7)
    with pytest.raises(pat.PatternError):
        pat.build_shape("MULTI_JOIN", 3)


def test_generator_covers_every_topology_that_exists_at_two_and_three_calls():
    for n_calls in (2, 3):
        built = {pat.shape_signature(pat.build_shape(p, n_calls))
                 for p in pat.patterns_for(n_calls)}
        assert len(built) == cmp.admissible_topologies(str(n_calls))
    # the three-call triangle carries the only 3-call reuse and fan-out
    assert pat.build_shape("REUSE_EARLY_OUTPUT", 3) == [[], [0], [0, 1]]


def test_cell_quotas_respect_the_derived_topology_cap():
    profile = json.loads((MODULE_ROOT / "outputs" / "pilot4_profile_safe"
                          / "target_profile_v2.json").read_text(encoding="utf-8"))
    constraints = derive_topology_constraints(profile)
    cells = build_cells(profile, constraints)
    for bucket, cons in constraints.items():
        cap = cons.get("maximum_top1_topology_share")
        group = [c for c in cells if c.call_bucket == bucket]
        if cap is None or not group:
            continue
        n_calls = max(BUCKET_CALLS[bucket])
        mass: dict = {}
        for c in group:
            sig = pat.shape_signature(pat.build_shape(c.pattern_family, n_calls))
            mass[sig] = mass.get(sig, 0.0) + c.quota_weight
        total = sum(mass.values())
        # chain-shaped patterns must not smuggle in extra mass one pattern at a time
        assert max(mass.values()) / total <= cap + 1e-9


@pytest.mark.parametrize("transform", pat.TRANSFORMATIONS)
def test_transformations_preserve_dag_and_execute(transform: str):
    rng = random.Random(hash(transform) % 10_000)
    # a base with several roots so branch-merging transforms are applicable
    shape = pat.build_shape("PARALLEL_THEN_MERGE", 4)
    out = pat.apply_transform(shape, transform, rng)   # validates DAG + no dead calls
    _assert_acyclic(out)
    assert _reaches_all(out)
    assert len(out) >= len(shape)

    result = pat.generate_program("PARALLEL_THEN_MERGE", 4, random.Random(3),
                                  answer_kind="", transformations=[transform])
    assert result.transformations == [transform]
    assert len(result.program.nodes) == len(out)
    observations, answer = execute(result.program)
    assert observations == result.observations and answer == result.answer


def test_generated_program_executes_and_is_deterministic():
    result = pat.generate_program("DIAMOND", 4, random.Random(11), answer_kind="")
    obs1, ans1 = execute(result.program)
    obs2, ans2 = execute(result.program)
    assert obs1 == obs2 and ans1 == ans2
    assert ans1 == result.answer
    assert len(result.capability_families) == len(result.program.nodes)


def test_late_reference_detection_in_graph_features():
    calls = [
        {"name": "a", "arguments": {"x": 1}, "label": "$var_1"},
        {"name": "b", "arguments": {"x": 2}, "label": "$var_2"},
        {"name": "c", "arguments": {"x": "$var_1.output_0$"}, "label": "$var_3"},
    ]
    gf = graph_features(calls)
    assert gf["n_late_references"] >= 1
    assert gf["max_reference_distance"] >= 2


def test_topology_diversity_is_measured_inside_call_buckets():
    feats = [featurize({"gold_calls": [
        {"name": "a", "arguments": {"x": 1}, "label": "$var_1"},
        {"name": "b", "arguments": {"x": "$var_1.output_0$"}, "label": "$var_2"}],
        "tools": [], "gold_answer": 1, "question": "q"}) for _ in range(3)]
    div = topology_diversity(feats)
    assert div["2"]["n_distinct_topologies"] == 1
    assert div["2"]["top1_topology_share"] == 1.0


# ── capability registry ───────────────────────────────────────────────────
def test_capability_registry_is_complete_and_valid():
    registry = cap.build_registry()
    assert cap.validate(registry) == []
    cov = cap.coverage(registry)
    assert cov["n_families_populated"] == cov["n_families_declared"]
    assert cov["primitives_outside_taxonomy"] == []


def test_behavioural_equivalence_rejects_aliases_and_accepts_differences():
    prims = reg.all_primitives()
    sid = next(iter(sorted(prims)))
    p = prims[sid]
    assert cap.behaviourally_equivalent(p, p) is True
    other = next(q for q in prims.values()
                 if q.sid != sid and cap.signatures_compatible(p, q)
                 and not cap.behaviourally_equivalent(p, q))
    assert cap.behaviourally_equivalent(p, other) is False


# ── rendering ─────────────────────────────────────────────────────────────
def test_paired_rendering_preserves_the_oracle():
    spec, a = _make_record(_cell(), track="A_NATIVE",
                           query_mode="PROCEDURAL_EXPLICIT")
    g = _render(spec, _cell(), track="G_GENERAL",
                query_mode="GOAL_BASED_IMPLICIT")
    assert g is not None
    assert a["gold_answer"] == g["gold_answer"] == spec.answer
    assert a["oracle_observations"] == g["oracle_observations"]
    assert a["semantic_program_id"] == g["semantic_program_id"]
    assert a["program_family_id"] == g["program_family_id"]
    assert a["graph_template_id"] == g["graph_template_id"]
    assert a["question"] != g["question"]
    assert {t["name"] for t in a["tools"]} != {t["name"] for t in g["tools"]}


def test_goal_based_render_leaks_less_than_procedural():
    spec, _ = _make_record(_cell(bucket="4"))
    sids = [n.semantic_id for n in spec.program.nodes]
    proc = qrender.render_query(spec, "PROCEDURAL_EXPLICIT", random.Random(2))
    goal = qrender.render_query(spec, "GOAL_BASED_IMPLICIT", random.Random(2))
    a = qr.audit_task(proc["query"], sids)
    b = qr.audit_task(goal["query"], sids)
    assert b["exact_operation_coverage"] <= a["exact_operation_coverage"]
    assert b["sequence_leakage"] <= a["sequence_leakage"]
    assert b["procedural_cue_count"] <= a["procedural_cue_count"]


def test_v7_flags_plan_leak_against_the_target_bucket():
    sids = ["subtract", "multiply"]
    question = "Step 1: subtract 3 from 10. Step 2: multiply the result by 4."
    leaky = v7_plan_leak(question, sids, "GOAL_BASED_IMPLICIT")
    assert leaky["passes_target_bucket"] is False
    assert any("step numbering" in w for w in leaky["warnings"])
    assert leaky["lexical_operation_coverage"] == 1.0

    # the same question is legitimate in the explicit bucket: V7 labels and
    # quotas plan leakage instead of discarding it
    ok = v7_plan_leak(question, sids, "PROCEDURAL_EXPLICIT")
    assert ok["passes_target_bucket"] is True
    assert ok["query_mode"] == "PROCEDURAL_EXPLICIT"


# ── distractors ───────────────────────────────────────────────────────────
def test_distractors_are_schema_semantic_and_non_equivalent():
    _spec, rec = _make_record(_cell(profile="schema_adversarial"))
    records = rec["distractors"]
    assert records
    for d in records:
        assert d["difficulty_level"] in dis.DISTRACTOR_LEVELS
        assert d["reason_incorrect"]
        assert d["distractor_primitive"] != d["target_gold_primitive"]
    hard = [d for d in records if d["difficulty_level"] in dis.HARD_LEVELS]
    assert hard, "no hard distractor was produced under a hard profile"
    assert all(d["output_type_compatible"] and d["input_types_compatible"]
               for d in hard)
    assert v8_distractor_validity(records)["passed"] is True


def test_v8_rejects_a_hidden_alias():
    """A "distractor" that is the gold primitive under another name must fail."""
    alias = [{"distractor_tool": "compute_sum", "target_gold_tool": "add",
              "distractor_primitive": "add", "target_gold_primitive": "add",
              "difficulty_level": "HARD_SEMANTIC_NEIGHBOR",
              "arity_compatible": True, "input_types_compatible": True,
              "output_type_compatible": True, "same_capability_family": True,
              "semantic_neighbor": True,
              "reason_incorrect": "claims to differ from add"}]
    res = v8_distractor_validity(alias)
    assert res["passed"] is False
    assert res["n_hidden_aliases"] == 1

    missing_reason = [{**alias[0], "distractor_primitive": "subtract",
                       "reason_incorrect": ""}]
    assert v8_distractor_validity(missing_reason)["passed"] is False


# ── difficulty signature ──────────────────────────────────────────────────
def test_difficulty_signature_has_all_four_sections():
    _spec, rec = _make_record(_cell())
    sig = rec["difficulty_signature"]
    assert set(sig) >= {"structural", "query", "surface", "environment"}
    assert sig["structural"]["call_count"] == rec["call_count"]
    assert sig["query"]["mode"] == rec["requested_query_mode"]
    assert sig["environment"]["offered_tool_count"] == rec["offered_tool_count"]
    assert difficulty_band(sig) in ("easy", "medium", "hard")


# ── validation layers ─────────────────────────────────────────────────────
def test_v1_v2_v3_pass_on_a_generated_record():
    _spec, rec = _make_record(_cell())
    assert v4.v1_schema(rec) == []
    assert v4.v2_execution(rec) == []


def test_v2_detects_a_tampered_oracle():
    _spec, rec = _make_record(_cell())
    rec["gold_answer"] = "definitely-not-the-answer"
    assert any("answer differs" in e for e in v4.v2_execution(rec))


def test_v3_rejects_an_answer_leak():
    rec = {"question": "The total is 42 already.", "gold_answer": 42,
           "gold_calls": [{"name": "a"}, {"name": "b"}],
           "oracle_observations": [1, 42], "semantic_program": {"nodes": []}}
    assert any("final answer" in e for e in v4.v3_semantic(rec))


def test_v5_dedup_and_v6_distribution():
    rows = [{"task_id": f"t{i}", "question": "same question",
             "tool_combination_hash": "tc", "query_skeleton": "s",
             "generation_cell": "c", "program_family_id": "f"} for i in range(3)]
    assert v4.v5_dedup(rows)["n_duplicates"] == 2
    assert v4.v6_distribution(rows)["passed"] is False


# ── selection and split ───────────────────────────────────────────────────
def _fake_records(n: int = 60):
    out = []
    for i in range(n):
        bucket = ["2", "3", "4", "5", "6+"][i % 5]
        out.append({
            "task_id": f"t{i}", "call_bucket": bucket, "call_count": int(bucket[0]),
            "classified_query_mode": "GOAL_BASED_IMPLICIT" if i % 2 else "SEMI_IMPLICIT",
            "requested_query_mode": "GOAL_BASED_IMPLICIT",
            "surface_track": "G_GENERAL" if i % 2 else "A_NATIVE",
            "generation_cell": f"cell{i % 4}", "program_family_id": f"fam{i}",
            "semantic_program_id": f"sp{i}", "graph_template_id": f"gt{i % 7}",
            "query_skeleton": f"sk{i % 9}", "difficulty_band": "medium",
            "tool_combination_hash": f"tc{i}", "surface_signature": f"ss{i}",
            "capability_families": ["arithmetic.binary"],
            "structural_features": {"n_joins": i % 2, "depth": 2},
            "semantic_program": {"nodes": [{"primitive_id": "add_numbers"}]},
            "schema_compatible_distractor_count": 3,
            "validation": {"V8": {"passed": True}},
        })
    return out


_FAKE_PROFILE = {
    "call_count_dist": {"2": 0.2, "3": 0.2, "4": 0.2, "5": 0.2, "6+": 0.2},
    "marginal": {"query_mode": {"GOAL_BASED_IMPLICIT": 0.6, "SEMI_IMPLICIT": 0.4}},
    "topology_diversity_by_bucket": {},
}
# the production caps (2 % family, 6 % skeleton) are unreachable on a 60-row
# pool; the caps themselves are asserted separately below
_FAKE_CONSTRAINTS = {"max_program_family_share": 0.2,
                     "max_query_skeleton_share": 0.5,
                     "min_topology_diversity_5call": 1,
                     "min_topology_diversity_6plus": 1}
_FAKE_CELLS = [{"cell_id": f"cell{i}", "quota_weight": 0.25} for i in range(4)]


def test_selection_reports_requested_achieved_and_reason():
    records = _fake_records()
    chosen, report = sel.select_records(records, _FAKE_CELLS, 20,
                                        profile=_FAKE_PROFILE, seed=1,
                                        constraints=_FAKE_CONSTRAINTS)
    assert len(chosen) == 20
    assert report["n_pool"] == len(records)
    assert report["n_selected"] == 20
    for row in report["constraint_rows"]:
        assert {"constraint", "requested_target", "achieved", "absolute_deficit",
                "relative_deficit", "met", "reason_not_met"} <= set(row)
        assert row["met"] or row["reason_not_met"], "an unmet target needs a reason"
    buckets = {r["constraint"] for r in report["constraint_rows"]}
    assert {f"call_bucket_share[{b}]" for b in ("2", "3", "4", "5", "6+")} <= buckets


def test_selection_enforces_family_and_skeleton_caps():
    records = _fake_records()
    for rec in records:                      # collapse the pool onto one family
        rec["program_family_id"] = "fam0"
    chosen, report = sel.select_records(records, _FAKE_CELLS, 20,
                                        profile=_FAKE_PROFILE, seed=1,
                                        constraints=_FAKE_CONSTRAINTS)
    # the cap is a share of the *requested* size, so a single-family pool can
    # only fill 0.2 * 20 slots and the run reports the shortfall
    assert len(chosen) == 4
    assert report["hard_constraint_rejections"]["max_program_family_share"] > 0
    assert report["all_hard_constraints_met"] is False


def test_selection_is_deterministic_and_independent_of_pool_order():
    records = _fake_records()
    a, _ = sel.select_records(records, _FAKE_CELLS, 15, profile=_FAKE_PROFILE,
                              seed=7, constraints=_FAKE_CONSTRAINTS)
    b, _ = sel.select_records(records, _FAKE_CELLS, 15, profile=_FAKE_PROFILE,
                              seed=7, constraints=_FAKE_CONSTRAINTS)
    assert [r["task_id"] for r in a] == [r["task_id"] for r in b]

    shuffled = list(records)
    random.Random(0).shuffle(shuffled)
    c, _ = sel.select_records(shuffled, _FAKE_CELLS, 15, profile=_FAKE_PROFILE,
                              seed=7, constraints=_FAKE_CONSTRAINTS)
    assert [r["task_id"] for r in c] == [r["task_id"] for r in a]


def test_split_is_family_safe():
    records = _fake_records(40)
    for i in range(0, 40, 2):                      # pair i with i+1
        records[i + 1]["program_family_id"] = records[i]["program_family_id"]
        records[i + 1]["semantic_program_id"] = records[i]["semantic_program_id"]
        records[i + 1]["graph_template_id"] = records[i]["graph_template_id"] = f"gt{i}"
        records[i]["paired_with"] = records[i + 1]["task_id"]
        records[i + 1]["paired_with"] = records[i]["task_id"]
    splits, manifest = sel.split_records(
        records, {"train": 24, "heldout": 8, "reserve": 8}, seed=3)
    seen = {}
    for name, rows in splits.items():
        for r in rows:
            prev = seen.setdefault(r["program_family_id"], name)
            assert prev == name, "a program family was split across splits"
    assert manifest["leakage"]["program_family_id"] == 0
    assert manifest["leakage"]["paired_with"] == 0
    assert manifest["leak_free"] is True
    assert manifest["sizes_achieved"] == manifest["sizes_requested"]


def test_a_shared_graph_template_is_not_counted_as_leakage():
    records = _fake_records(30)
    for rec in records:                            # one topology for everything
        rec["graph_template_id"] = "LINEAR_CHAIN@3"
    _, manifest = sel.split_records(
        records, {"train": 18, "heldout": 6, "reserve": 6}, seed=3)
    # a topology is shared by design; only program identity may not straddle splits
    assert manifest["leak_free"] is True
    assert manifest["shared_by_design"]["graph_template_id"] > 0
    assert "graph_template_id" not in manifest["leakage"]


# ── sampler ───────────────────────────────────────────────────────────────
def _obs(**kw):
    from targeted_tool_data.sampling import GroupObservation

    base = dict(global_step=0, prompt_id="p", group_size=4,
                terminal_rewards=[0, 0, 0, 0], process_rewards=[0, 0, 0, 0])
    base.update(kw)
    return GroupObservation(**base)


def test_group_classification_covers_the_taxonomy():
    from targeted_tool_data.sampling import (ALL_CORRECT, ALL_FAIL_NO_PROGRESS,
                                             ALL_FAIL_WITH_PROCESS_VARIANCE,
                                             INVALID_GROUP, MIXED_BOTH,
                                             MIXED_TERMINAL)

    assert _obs(terminal_rewards=[1, 1, 1, 1],
                process_rewards=[1, 1, 1, 1]).group_class == ALL_CORRECT
    assert _obs().group_class == ALL_FAIL_NO_PROGRESS
    assert _obs(process_rewards=[0.1, 0.4, 0.2, 0.9]).group_class == \
        ALL_FAIL_WITH_PROCESS_VARIANCE
    assert _obs(terminal_rewards=[1, 0, 0, 0],
                process_rewards=[0.5, 0.5, 0.5, 0.5]).group_class == MIXED_TERMINAL
    assert _obs(terminal_rewards=[1, 0, 0, 0],
                process_rewards=[0.5, 0.1, 0.9, 0.3]).group_class == MIXED_BOTH
    assert _obs(group_size=1, terminal_rewards=[1],
                process_rewards=[1]).group_class == INVALID_GROUP


def test_all_fail_with_process_variance_is_kept():
    from targeted_tool_data.sampling import is_effective

    assert is_effective(_obs(process_rewards=[0.1, 0.4, 0.2, 0.9])) is True
    assert is_effective(_obs()) is False


def test_sampler_state_round_trips_and_replays_deterministically():
    from targeted_tool_data.sampling import (HistoryAdaptivePromptSampler,
                                             PromptRef)

    prompts = [PromptRef(prompt_id=f"p{i}", generation_cell=f"c{i % 3}",
                         call_bucket="4", difficulty_band="medium")
               for i in range(20)]
    s1 = HistoryAdaptivePromptSampler(prompts, seed=4)
    first = [p.prompt_id for p in s1.sample_candidates(5)]
    s1.observe_group(_obs(prompt_id=first[0], generation_cell="c0",
                          terminal_rewards=[1, 0, 1, 0],
                          process_rewards=[0.4, 0.1, 0.6, 0.2]))
    state = json.loads(json.dumps(s1.state_dict()))

    s2 = HistoryAdaptivePromptSampler(prompts, seed=99)
    s2.load_state_dict(state)
    assert s2.state.prompt[first[0]].group_count == 1
    assert [p.prompt_id for p in s2.sample_candidates(5)] == \
        [p.prompt_id for p in _replay(prompts, s1, state)]


def _replay(prompts, sampler, state):
    from targeted_tool_data.sampling import HistoryAdaptivePromptSampler

    s = HistoryAdaptivePromptSampler(prompts, seed=0)
    s.load_state_dict(state)
    return s.sample_candidates(5)


def test_weight_components_are_stored_separately():
    from targeted_tool_data.sampling import (HistoryAdaptivePromptSampler,
                                             PromptRef)

    prompts = [PromptRef(prompt_id=f"p{i}", generation_cell="c0") for i in range(6)]
    s = HistoryAdaptivePromptSampler(prompts, seed=1)
    s.observe_group(_obs(prompt_id="p0", generation_cell="c0",
                         terminal_rewards=[1, 0, 1, 0],
                         process_rewards=[0.3, 0.1, 0.7, 0.2]))
    s.sample_candidates(6)
    comps = s.by_id["p0"].weight_components
    assert {"frontier_weight", "variance_weight", "staleness_weight"} <= set(comps)
    assert s.by_id["p0"].selection_weight > 0


def test_batch_refill_reaches_the_effective_target():
    from targeted_tool_data.sampling import (PromptRef, UniformPromptSampler,
                                             refill_batch)

    prompts = [PromptRef(prompt_id=f"p{i}", generation_cell="c0") for i in range(64)]
    s = UniformPromptSampler(prompts, seed=2)
    rng = random.Random(0)

    def score(prompt, step):
        # informative for half the prompts, dead for the rest
        idx = int(prompt.prompt_id[1:])
        if idx % 2:
            return _obs(prompt_id=prompt.prompt_id, global_step=step,
                        terminal_rewards=[1, 0, 1, 0],
                        process_rewards=[0.4, 0.1, 0.6, 0.2])
        return _obs(prompt_id=prompt.prompt_id, global_step=step)

    out = refill_batch(s, score, global_step=0, target_effective=8, batch_size=8)
    assert out["accepted_effective_groups"] >= 8
    assert out["target_reached"] is True
    assert out["dead_group_rate_before_filtering"] > 0.0


def test_curriculum_state_transitions():
    from targeted_tool_data.sampling import (ACTIVE, LOCKED, MASTERED, PROBING,
                                             CellCurriculumSampler, PromptRef)

    prompts = [PromptRef(prompt_id=f"p{i}", generation_cell="c_adv"
                         if i else "c_base") for i in range(6)]
    s = CellCurriculumSampler(prompts, seed=1,
                              prerequisites={"c_adv": ["c_base"]})
    assert s.state.curriculum["c_adv"] == LOCKED
    assert s.state.curriculum["c_base"] == PROBING

    for step in range(10):
        s.observe_group(_obs(prompt_id="p0", generation_cell="c_base",
                             global_step=step, terminal_rewards=[1, 1, 1, 0],
                             process_rewards=[0.5, 0.5, 0.5, 0.2]))
    assert s.state.curriculum["c_base"] == ACTIVE

    # all-correct groups are rejected as uninformative, but a cell the model
    # always solves is MASTERED, never TOO_HARD
    for step in range(10, 60):
        s.observe_group(_obs(prompt_id="p0", generation_cell="c_base",
                             global_step=step, terminal_rewards=[1, 1, 1, 1],
                             process_rewards=[0.5, 0.5, 0.5, 0.5]))
    assert s.state.curriculum["c_base"] == MASTERED
    assert s.easier_sibling("c_adv") in (None, "c_base")


def test_repeated_all_fail_without_progress_marks_a_cell_too_hard():
    from targeted_tool_data.sampling import (ACTIVE, TOO_HARD,
                                             CellCurriculumSampler, PromptRef)

    s = CellCurriculumSampler([PromptRef(prompt_id="p0", generation_cell="c")],
                              seed=1)
    for step in range(10):                       # earn ACTIVE first
        s.observe_group(_obs(prompt_id="p0", generation_cell="c",
                             global_step=step, terminal_rewards=[1, 0, 0, 0],
                             process_rewards=[0.5, 0.2, 0.1, 0.4]))
    assert s.state.curriculum["c"] == ACTIVE

    for step in range(10, 40):                   # flat zero rewards: no signal
        s.observe_group(_obs(prompt_id="p0", generation_cell="c", global_step=step))
    assert s.state.curriculum["c"] == TOO_HARD
    entry = s.state.axis["generation_cell"]["c"]
    assert entry.consecutive_all_fail_no_progress >= 12


# ── logging schemas ───────────────────────────────────────────────────────
def test_train_logging_schema_round_trips(tmp_path: Path):
    from targeted_tool_data.observability import TrainRunLogger

    logger = TrainRunLogger(out_dir=tmp_path, run_id="r1")
    logger.write_manifest(config={"training": {"kl_beta": 0.02}},
                          dataset_path=None, sample_ids=["a", "b"],
                          subset_ids=["a"], base_model="qwen")
    logger.log_rollout({"global_step": 0, "prompt_id": "a", "rollout_id": 0,
                        "total_reward": 1.0, "response_text": "x",
                        "process_components": {"exec": 0.5}})
    logger.log_group({"global_step": 0, "prompt_id": "a", "group_class": "MIXED_BOTH",
                      "reward_std": 0.4, "accepted": True})
    logger.log_step({"global_step": 0, "dead_group_rate_before_filtering": 0.5})
    logger.close()

    manifest = json.loads((tmp_path / "TRAIN_RUN_MANIFEST.json").read_text("utf-8"))
    assert manifest["dataset"]["ordered_sample_id_hash"]
    assert manifest["schema_version"]
    rollout = json.loads((tmp_path / "train_rollouts.jsonl").read_text("utf-8").strip())
    assert rollout["run_id"] == "r1" and rollout["process_components"] == {"exec": 0.5}
    group = json.loads((tmp_path / "train_groups.jsonl").read_text("utf-8").strip())
    assert group["group_class"] == "MIXED_BOTH"
    step = json.loads((tmp_path / "train_steps.jsonl").read_text("utf-8").strip())
    assert step["dead_group_rate_before_filtering"] == 0.5


def test_ordered_sample_id_hash_is_order_sensitive():
    from targeted_tool_data.observability import ordered_id_hash

    assert ordered_id_hash(["a", "b"]) != ordered_id_hash(["b", "a"])


def test_sampler_state_is_saved_next_to_the_checkpoint(tmp_path: Path):
    from targeted_tool_data.observability import TrainRunLogger
    from targeted_tool_data.sampling import (HistoryAdaptivePromptSampler,
                                             PromptRef)

    s = HistoryAdaptivePromptSampler(
        [PromptRef(prompt_id="p0", generation_cell="c0")], seed=0)
    s.observe_group(_obs(prompt_id="p0", generation_cell="c0"))
    logger = TrainRunLogger(out_dir=tmp_path, run_id="r")
    ckpt = tmp_path / "adapter_epoch_1"
    written = logger.save_sampler_state(s.state_dict(), ckpt)
    assert (ckpt / "sampler_state.json").exists()
    assert (ckpt / "sampler_cell_stats.csv").exists()
    assert any("sampler_prompt_stats" in p.name for p in written)


def test_eval_manifest_and_paired_check(tmp_path: Path):
    from targeted_tool_data.observability import EvalRunLogger, compare_eval_runs

    def make(run_id: str, backend: str, ids):
        logger = EvalRunLogger(out_dir=tmp_path / run_id, run_id=run_id)
        logger.write_manifest(model_revision="rev", adapter_path=None,
                              adapter_hash=None, merged_lora=False,
                              backend=backend, dataset_path=None,
                              sample_ids=ids, scorer_commit="abc")
        for sid in ids:
            logger.log_input({"sample_id": sid, "raw_prompt_hash": f"h{sid}",
                              "offered_tools_hash": "t"})
        logger.close()
        manifest = json.loads(
            (tmp_path / run_id / "EVAL_RUN_MANIFEST.json").read_text("utf-8"))
        inputs = [json.loads(l) for l in
                  (tmp_path / run_id / "eval_inputs.jsonl")
                  .read_text("utf-8").splitlines() if l.strip()]
        return manifest, inputs

    c0, c0_in = make("c0", "vllm", ["s1", "s2"])
    d1, d1_in = make("d1", "vllm", ["s1", "s2"])
    ok = compare_eval_runs(c0, d1, c0_in, d1_in)
    assert ok["comparable"] is True

    hf, hf_in = make("hf", "hf", ["s2", "s1"])
    bad = compare_eval_runs(c0, hf, c0_in, hf_in)
    assert bad["comparable"] is False
    assert any("backend differs" in w for w in bad["warnings"])
    assert bad["checks"]["same_task_order"] is False


# ── frozen dataset + backward compatibility ───────────────────────────────
@pytest.mark.skipif(not (PILOT4_DIR / "train.jsonl").exists(),
                    reason="pilot4 dataset has not been generated in this tree")
def test_frozen_pilot4_split_sizes_and_family_safety():
    def rows(name):
        return [json.loads(l) for l in
                (PILOT4_DIR / f"{name}.jsonl").read_text("utf-8").splitlines()
                if l.strip()]

    train, heldout, reserve = rows("train"), rows("heldout"), rows("reserve")
    assert (len(train), len(heldout), len(reserve)) == (600, 200, 200)
    fam = {}
    for name, rs in (("train", train), ("heldout", heldout), ("reserve", reserve)):
        for r in rs:
            key = r["provenance"]["semantic_program_family"]
            assert fam.setdefault(key, name) == name
    ids = [r["sample_id"] for r in train + heldout + reserve]
    assert len(set(ids)) == len(ids)


@pytest.mark.skipif(not (PILOT4_DIR / "train.jsonl").exists(),
                    reason="pilot4 dataset has not been generated in this tree")
def test_pilot4_grpo_rows_match_the_pilot3_train_contract():
    pilot3 = MODULE_ROOT / "outputs" / "selected" / "export_pilot3" / \
        "train_grpo_pilot3.jsonl"
    if not pilot3.exists():
        pytest.skip("pilot3 export not present")
    old = json.loads(pilot3.read_text("utf-8").splitlines()[0])
    new = json.loads((PILOT4_DIR / "train.jsonl").read_text("utf-8").splitlines()[0])
    required = {"sample_id", "question", "tools", "gold_calls", "gold_answer",
                "observations", "num_calls", "stage", "answer_type", "provenance"}
    assert required <= set(old)
    assert required <= set(new)
    assert isinstance(new["tools"], list) and new["tools"]
    assert {"name", "arguments", "label"} <= set(new["gold_calls"][0])
