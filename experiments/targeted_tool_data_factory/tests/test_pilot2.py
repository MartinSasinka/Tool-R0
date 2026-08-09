"""Regression tests for the pilot2 extensions.

Each test pins an invariant that a real defect violated during development, so
a regression is caught here rather than three hours into a RunPod run.
"""
from __future__ import annotations

import json
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

FACTORY = Path(__file__).resolve().parents[1]
BUNDLE = FACTORY / "runpod_bundle_pilot2"
sys.path.insert(0, str(FACTORY / "src"))

from targeted_tool_data import plausibility  # noqa: E402
from targeted_tool_data.executor import execute  # noqa: E402
from targeted_tool_data.generation import build_cells_v2, make_candidate  # noqa: E402
from targeted_tool_data.graph import build_program_v2, classify_program_motif  # noqa: E402
from targeted_tool_data.registry import (  # noqa: E402
    all_surfaces, registry_hash, validate_surface_uniqueness,
)
from targeted_tool_data.schemas import TargetProfile  # noqa: E402
from targeted_tool_data.validation import validate_record  # noqa: E402

BUCKETS_CFG = {"small": [8, 9], "medium": [10, 12], "large": [13, 18]}
CONV = {"param_styles": ["semantic"], "label_styles": ["$var{i}"]}


def dummy_profile() -> TargetProfile:
    """A NESTFUL-shaped target profile, hard-coded so the tests never depend on
    the real benchmark being present on disk."""
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


# ───────────────────────────────────────────── registry / trainer contract ────

def test_surface_names_map_to_exactly_one_signature():
    """The trainer keys tools by NAME only. Two surfaces sharing a name with
    different parameters is the schema drift that made pilot1 untrainable."""
    validate_surface_uniqueness()


def test_every_surface_is_exposed_by_the_trainer_adapter():
    sys.path.insert(0, str(FACTORY / "trainer_adapter" / "lib"))
    import synthetic_tools  # noqa: PLC0415

    exposed = set(synthetic_tools.TOOLS)
    declared = {surface.name for _sid, _track, surface in all_surfaces()}
    assert declared - exposed == set(), f"surfaces missing from adapter: {declared - exposed}"


def test_adapter_reports_the_live_registry_hash():
    sys.path.insert(0, str(FACTORY / "trainer_adapter" / "lib"))
    import synthetic_tools  # noqa: PLC0415

    hashes = synthetic_tools.factory_hashes()
    assert hashes["registry_hash"] == registry_hash()
    assert synthetic_tools.registry_hash()


def test_adapter_tool_schemas_are_json_schema_shaped():
    sys.path.insert(0, str(FACTORY / "trainer_adapter" / "lib"))
    import synthetic_tools  # noqa: PLC0415

    for name in list(synthetic_tools.TOOLS)[:40]:
        schema = synthetic_tools.tool_schema(name)
        params = schema.get("parameters") or schema
        assert params.get("type") == "object"
        assert isinstance(params.get("properties"), dict) and params["properties"]


# ──────────────────────────────────────────────────── semantic plausibility ────

def _cells(seed=20260726, **over):
    cfg = {"generation": {"engine": "v2", "hard_distractor_share": 0.8,
                          "answer_kind_shares": {"float": 0.80, "list": 0.07,
                                                 "string": 0.05, "int": 0.04,
                                                 "bool": 0.02, "numeric_string": 0.02}}}
    cfg["generation"].update(over)
    return build_cells_v2(dummy_profile(), cfg,
                          ["adaptation", "generalization"], 0.6)


def _sample_programs(n=60, seed=7):
    rng = random.Random(seed)
    cells = _cells()
    out = []
    for i in range(n):
        cell = cells[i % len(cells)]
        try:
            out.append(build_program_v2(cell, random.Random(rng.random())))
        except Exception:  # noqa: BLE001 - generation retries are expected
            continue
    assert out, "engine v2 produced no programs at all"
    return out


def test_v2_never_emits_artificial_unit_compositions():
    """Unit propagation is a structural constraint, not a post-hoc filter: the
    builder must not be able to chain a percentage into a temperature."""
    for prog in _sample_programs():
        assert plausibility.analyze(prog)["plausibility_class"] != "artificial_composition"


def test_plausibility_rejects_an_incoherent_transition():
    from targeted_tool_data.plausibility import transition_class
    from targeted_tool_data.registry import U_ABSTRACT  # noqa: PLC0415

    assert transition_class("temperature_c", "percent", None) == "artificial_composition"
    assert transition_class("any", U_ABSTRACT, None) != "artificial_composition"


# ─────────────────────────────────────────────────────────── graph motifs ────

def _refs(value) -> list[str]:
    if isinstance(value, dict) and "__ref__" in value:
        return [value["__ref__"]]
    if isinstance(value, list):
        return [r for v in value for r in _refs(v)]
    return []


def test_branch_aggregate_has_true_indegree_three_and_no_array_references():
    """The trainer cannot resolve a reference nested inside an array argument,
    so branch aggregation must use scalar three-way aggregators."""
    seen = 0
    for prog in _sample_programs(200, seed=11):
        if classify_program_motif(prog) != "branch_aggregate":
            continue
        seen += 1
        sink = prog.nodes[-1]
        n_refs = sum(len(_refs(v)) for v in sink.inputs.values())
        assert n_refs >= 3, f"branch_aggregate sink has indegree {n_refs}"
        for value in sink.inputs.values():
            assert not (isinstance(value, list) and _refs(value)), \
                "reference nested inside an array argument"
    assert seen, "no branch_aggregate program sampled"


def test_no_program_hides_a_reference_inside_an_array():
    for prog in _sample_programs(200, seed=17):
        for node in prog.nodes:
            for value in node.inputs.values():
                assert not (isinstance(value, list) and _refs(value)), \
                    f"{node.semantic_id} nests a reference in an array"


def test_every_branch_reaches_the_sink():
    """A branch that nothing consumes makes the task solvable without it, which
    breaks the whole point of the fan-in motifs."""
    for prog in _sample_programs(120, seed=13):
        produced = {n.node_id for n in prog.nodes[:-1]}
        consumed = {r for n in prog.nodes for v in n.inputs.values() for r in _refs(v)}
        assert produced <= consumed, f"dead branch: {produced - consumed}"


# ────────────────────────────────────────────────────── candidate factory ────

def test_generated_candidates_validate_and_are_deterministic():
    cells = _cells()
    made = passed = 0
    for i, cell in enumerate(cells[:24]):
        rec = make_candidate(cell, i, 20260726, CONV, BUCKETS_CFG, "test",
                             registry_hash(), "cfg", engine="v2")
        if rec is None:
            continue
        made += 1
        assert rec.gold_answer == rec.oracle_observations[-1]
        again = make_candidate(cell, i, 20260726, CONV, BUCKETS_CFG, "test",
                               registry_hash(), "cfg", engine="v2")
        assert again is not None and again.model_dump() == rec.model_dump()

        report = validate_record(rec, {})
        if report["passed"]:
            passed += 1
            continue
        # A rejection is fine, but only for a reason the ladder is allowed to
        # give: schema, oracle or reference errors must never reach this point.
        v4 = report["layers"]["V4"]
        assert not v4["passed"] and v4["reasons"], report
        for layer in ("V1", "V2", "V3"):
            assert report["layers"][layer]["passed"], report["layers"][layer]
    assert made >= 12, f"only {made} candidates built"
    assert passed / made >= 0.6, f"only {passed}/{made} candidates survived V4"


def test_candidates_carry_pilot2_provenance():
    cells = _cells()
    for i, cell in enumerate(cells):
        rec = make_candidate(cell, i, 20260726, CONV, BUCKETS_CFG, "t",
                             registry_hash(), "c", engine="v2")
        if rec is None:
            continue
        assert rec.plausibility_class in ("natural", "abstract_coherent",
                                          "artificial_composition")
        assert rec.query_source == "template"
        assert rec.sink_unit is not None
        return
    pytest.fail("no candidate built")


def test_hard_distractor_share_is_configurable_not_hardcoded():
    def share(target):
        cells = _cells(hard_distractor_share=target)
        w = sum(c.quota_weight for c in cells)
        return sum(c.quota_weight for c in cells if c.hard_distractors) / w

    assert share(0.8) == pytest.approx(0.8, abs=0.06)
    assert share(0.5) == pytest.approx(0.5, abs=0.06)
    assert share(0.5) < share(0.9)


def test_answer_kind_quotas_cover_every_requested_type():
    cells = _cells()
    kinds = {c.answer_kind for c in cells}
    for want in ("float", "int", "bool", "string", "list", "numeric_string"):
        assert want in kinds, f"no cell targets answer_kind={want}"


# ─────────────────────────────────────────────────── paraphrase validator ────

def _record():
    cells = _cells()
    for i, cell in enumerate(cells):
        rec = make_candidate(cell, i, 20260726, CONV, BUCKETS_CFG, "t",
                             registry_hash(), "c", engine="v2")
        if rec is not None:
            return rec
    pytest.skip("no candidate available")


def test_paraphrase_validator_rejects_dropped_constants():
    from targeted_tool_data.paraphrase.validate import numeric_multiset, validate_paraphrase

    rec = _record()
    consts = numeric_multiset(rec.query)
    if not consts:
        pytest.skip("template carries no numeric constant")
    stripped = rec.query.replace(str(next(iter(consts))), "some")
    ok, reasons = validate_paraphrase(rec, stripped)
    assert not ok and reasons


def test_paraphrase_validator_rejects_an_oracle_leak():
    from targeted_tool_data.paraphrase.validate import validate_paraphrase

    rec = _record()
    ok, reasons = validate_paraphrase(rec, f"{rec.query} The answer is {rec.gold_answer}.")
    assert not ok and reasons


def test_paraphrase_validator_accepts_the_original_query():
    """A validator that rejects its own input would silently force a 100 %
    template fallback and look like an API problem."""
    from targeted_tool_data.paraphrase.validate import validate_paraphrase

    rec = _record()
    ok, reasons = validate_paraphrase(rec, rec.query)
    assert ok, reasons


def test_shortlist_allocates_budget_where_selection_will_look():
    """A uniform shortlist spends most of its budget on records that will never
    be selected, which is how the paraphrase share ended up at 40 % instead of
    the requested 60-70 %."""
    from targeted_tool_data.paraphrase.validate import shortlist

    records = [{"task_id": f"t{c}_{i:03d}", "generation_cell_id": f"cell_{c}",
                "track": "A", "call_count": 2 + c, "motif": "linear",
                "answer_type": "float"}
               for c in range(4) for i in range(200)]
    cells = [{"generation_cell_id": "cell_0", "quota_weight": 0.70, "call_count": 2},
             {"generation_cell_id": "cell_1", "quota_weight": 0.10, "call_count": 3},
             {"generation_cell_id": "cell_2", "quota_weight": 0.10, "call_count": 4},
             {"generation_cell_id": "cell_3", "quota_weight": 0.10, "call_count": 5}]

    flat = shortlist(records, 400, 1)
    weighted = shortlist(records, 400, 1, cells=cells, n_select=320,
                         target_share=0.65, accept_rate=0.35)

    def per_cell(ids):
        return Counter(i.split("_")[0] for i in ids)

    assert per_cell(weighted)["t0"] > per_cell(flat)["t0"]
    # a low acceptance rate must buy MORE requests, not fewer
    cheap = shortlist(records, 400, 1, cells=cells, n_select=320,
                      target_share=0.65, accept_rate={2: 0.9, 3: 0.9, 4: 0.9, 5: 0.9})
    dear = shortlist(records, 400, 1, cells=cells, n_select=320,
                     target_share=0.65, accept_rate={2: 0.9, 3: 0.9, 4: 0.9, 5: 0.1})
    assert per_cell(dear)["t3"] > per_cell(cheap)["t3"]


def test_shortlist_is_deterministic_and_growth_is_monotone():
    """A larger shortlist must extend the smaller one, otherwise re-running with
    more budget throws away the response cache."""
    from targeted_tool_data.paraphrase.validate import shortlist

    records = [{"task_id": f"t{i:04d}", "generation_cell_id": f"cell_{i % 5}",
                "track": "A" if i % 2 else "G", "call_count": 2 + i % 4,
                "motif": "linear", "answer_type": "float"} for i in range(500)]
    cells = [{"generation_cell_id": f"cell_{c}", "quota_weight": 0.2,
              "call_count": 2 + c} for c in range(5)]
    kw = dict(cells=cells, n_select=320, target_share=0.65, accept_rate=0.35)

    assert shortlist(records, 200, 3, **kw) == shortlist(records, 200, 3, **kw)
    small = shortlist(records, 200, 3, **kw)
    large = shortlist(records, 300, 3, **kw)
    assert set(small) <= set(large), "growing the shortlist invalidated the cache"


def test_api_key_never_reaches_an_artefact():
    from targeted_tool_data.paraphrase import key_fingerprint

    fp = key_fingerprint("sk-or-v1-super-secret-value")
    assert fp.startswith("sha256:") and "secret" not in fp and len(fp) < 32


# ──────────────────────────────────────────────────────── acceptance gates ────

def _fake_selected(n=100, **over):
    base = {"track": "A", "call_count": 3, "motif": "fan_in", "answer_type": "float",
            "plausibility_class": "abstract_coherent", "query_source": "template",
            "template_id": "t", "generation_cell_id": "c", "query": "q" * 100,
            "offered_tool_count": 11, "hard_distractor_count": 1,
            "reference_arg_share": 0.4, "canonical_calls": [{"arguments": {}}],
            "minimal_valid_call_count": 3, "shortcut_check": {}}
    base.update(over)
    return [dict(base, template_id=f"t{i % 40}", generation_cell_id=f"c{i % 20}")
            for i in range(n)]


def _gate(selected, **kw):
    from targeted_tool_data.reporting.pilot2 import pilot2_gates, pilot2_metrics

    args = dict(
        pilot1_metrics=None, profile_match=[], pilot1_match=[],
        validation_summary={"replay_rate_validated": 1.0}, leakage={"leaked": False},
        thresholds={}, target_answer_dist={"float": 0.77, "string": 0.07,
                                           "list": 0.07, "int": 0.05,
                                           "bool": 0.02, "numeric_string": 0.02})
    args.update(kw)
    return pilot2_gates(pilot2_metrics(selected), **args)


def test_gates_fail_on_imperfect_replay():
    g = _gate(_fake_selected(), validation_summary={"replay_rate_validated": 0.99})
    assert g["verdict"] == "NOT_READY"
    assert any("replay" in f for f in g["fails"])


def test_gates_fail_on_split_leakage():
    g = _gate(_fake_selected(), leakage={"leaked": True})
    assert g["verdict"] == "NOT_READY"


def test_gates_fail_on_an_accepted_single_tool_shortcut():
    rows = _fake_selected()
    rows[0]["shortcut_check"] = {"single_call_shortcut": True}
    assert _gate(rows)["verdict"] == "NOT_READY"


def test_gates_fail_on_a_reference_inside_an_array_argument():
    rows = _fake_selected()
    rows[0]["canonical_calls"] = [{"arguments": {"values": ["$var1.output_0$", 2]}}]
    g = _gate(rows)
    assert g["verdict"] == "NOT_READY"
    assert any("array" in f for f in g["fails"])


def test_gates_fail_when_a_template_dominates():
    rows = _fake_selected()
    for r in rows:
        r["template_id"] = "one_template"
    assert _gate(rows)["verdict"] == "NOT_READY"


def test_float_share_is_gated_against_the_measured_target_not_a_fixed_band():
    """0.734 is 3.6 pp from the NESTFUL dev share of 0.77 and must pass; 0.60 is
    17 pp away and must not."""
    close = _gate(_fake_selected(n=1000)[:734] + _fake_selected(n=266, answer_type="int"))
    assert not any("float" in w for w in close["warns"]), close["warns"]
    far = _gate(_fake_selected(n=600) + _fake_selected(n=400, answer_type="int"))
    assert any("float" in w for w in far["warns"])


def test_a_clean_pool_is_ready():
    rows = _fake_selected(n=1000)[:770] + _fake_selected(n=230, answer_type="int")
    for i, r in enumerate(rows):
        r["query_source"] = "openrouter_paraphrase" if i % 3 else "template"
        r["template_id"] = f"t{i % 60}"
        r["generation_cell_id"] = f"c{i % 30}"
    g = _gate(rows)
    assert g["fails"] == []


# ────────────────────────────────────────────────────────── runpod bundle ────

@pytest.mark.parametrize("script", [
    "run_all_4gpu.sh", "install.sh", "run_full_nestful_test.sh",
])
def test_bundle_shell_scripts_are_syntactically_valid(script):
    path = BUNDLE / script
    assert path.is_file(), f"missing {path}"
    bash = None
    for cand in ("bash", "C:/Program Files/Git/bin/bash.exe"):
        try:
            subprocess.run([cand, "--version"], capture_output=True, check=True)
            bash = cand
            break
        except Exception:  # noqa: BLE001
            continue
    if bash is None:
        pytest.skip("bash unavailable")
    r = subprocess.run([bash, "-n", str(path)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize("script", [
    "verify_hashes.py", "check_config_parity.py", "check_canary_gates.py",
    "run_eval_all.py", "make_paired_report.py", "build_bundle.py",
])
def test_bundle_python_scripts_compile(script):
    path = BUNDLE / script
    assert path.is_file(), f"missing {path}"
    r = subprocess.run([sys.executable, "-m", "py_compile", str(path)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_config_parity_checker_fails_on_a_non_dataset_difference(tmp_path):
    base = {"label": "D0", "train_subset": "a.jsonl",
            "optimizer": {"learning_rate": 1e-6, "kl_beta": 0.1}}
    same = dict(base, label="D1", train_subset="b.jsonl")
    drift = json.loads(json.dumps(same))
    drift["optimizer"]["learning_rate"] = 2e-6

    d0 = tmp_path / "d0.json"
    d0.write_text(json.dumps(base), encoding="utf-8")
    for cfg, expected in ((same, 0), (drift, 2)):
        d1 = tmp_path / "d1.json"
        d1.write_text(json.dumps(cfg), encoding="utf-8")
        r = subprocess.run([sys.executable, str(BUNDLE / "check_config_parity.py"),
                            "--d0", str(d0), "--d1", str(d1)],
                           capture_output=True, text=True)
        assert r.returncode == expected, r.stdout + r.stderr


def test_verify_hashes_detects_a_mutated_artefact(tmp_path, monkeypatch):
    import hashlib

    payload = tmp_path / "data.jsonl"
    # the manifest hashes LF-normalised bytes, so write LF even on Windows
    payload.write_bytes(b'{"a": 1}\n')
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    manifest = tmp_path / "MANIFEST.sha256.json"
    manifest.write_text(json.dumps({"files": {"data.jsonl": {"sha256": digest}}}),
                        encoding="utf-8")

    script = tmp_path / "verify_hashes.py"
    script.write_text((BUNDLE / "verify_hashes.py").read_text(encoding="utf-8"),
                      encoding="utf-8")
    ok = subprocess.run([sys.executable, str(script), "--manifest", str(manifest)],
                        capture_output=True, text=True)
    assert ok.returncode == 0, ok.stdout + ok.stderr

    payload.write_bytes(b'{"a": 2}\n')
    bad = subprocess.run([sys.executable, str(script), "--manifest", str(manifest)],
                         capture_output=True, text=True)
    assert bad.returncode == 2


def test_mcnemar_and_bootstrap_behave_on_known_inputs():
    sys.path.insert(0, str(BUNDLE))
    import make_paired_report as mpr  # noqa: PLC0415

    assert mpr.mcnemar_exact(0, 0) == 1.0
    assert mpr.mcnemar_exact(10, 10) == pytest.approx(1.0, abs=1e-9)
    assert mpr.mcnemar_exact(0, 20) < 1e-5
    lo, hi = mpr.paired_bootstrap([0.0] * 50, [1.0] * 50)
    assert lo == pytest.approx(1.0) and hi == pytest.approx(1.0)
    lo, hi = mpr.paired_bootstrap([0.0] * 50, [0.0] * 50)
    assert lo == 0.0 and hi == 0.0
