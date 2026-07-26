"""Selection metrics, deficit matching, leakage-free splits."""
from targeted_tool_data.generation import make_candidate
from targeted_tool_data.schemas import GenerationCell
from targeted_tool_data.selection import (jsd, leakage_audit,
                                          profile_match_report, select_records,
                                          split_records, two_sample_auc,
                                          wasserstein)

BUCKETS_CFG = {"small": [8, 9], "medium": [10, 12], "large": [13, 18]}
CONV = {"param_styles": ["semantic"], "label_styles": ["$var{i}"]}


def _pool(n=40, seed=5):
    cells = []
    recs = []
    for cc, motif in [(2, "linear"), (3, "fan_in")]:
        for track in ["A", "G"]:
            cell = GenerationCell(
                generation_cell_id=f"{track}_{cc}call_{motif}_t_00", track=track,
                mode="adaptation" if track == "A" else "generalization",
                call_count=cc, motif=motif, target_skill="s", target_failure="f",
                hard_distractor_type="near_semantics", quota_weight=0.25)
            cells.append(cell.model_dump())
            got = 0
            i = 0
            while got < n // 4 and i < n:
                r = make_candidate(cell, i, seed, CONV, BUCKETS_CFG, "pv", "rh", "ch")
                i += 1
                if r:
                    recs.append(r.model_dump())
                    got += 1
    return recs, cells


def test_jsd_and_wasserstein_basics():
    assert jsd({"a": 0.5, "b": 0.5}, {"a": 0.5, "b": 0.5}) == 0.0
    assert jsd({"a": 1.0}, {"b": 1.0}) == 1.0
    assert wasserstein([1, 2, 3], [1, 2, 3]) == 0.0
    assert wasserstein([0, 0], [10, 10]) == 10.0


def test_two_sample_auc_separates_different_sets():
    feats_a = [{"call_count": 2, "depth": 2, "ref_share": 0.3,
                "numeric_string_share": 0, "n_tools": 10, "q_len": 100,
                "motif": "linear"} for _ in range(50)]
    feats_b = [{"call_count": 6, "depth": 6, "ref_share": 0.9,
                "numeric_string_share": 0, "n_tools": 3, "q_len": 400,
                "motif": "fan_in"} for _ in range(50)]
    auc = two_sample_auc(feats_a, feats_b)
    assert auc > 0.95
    auc_same = two_sample_auc(feats_a, feats_a)
    assert 0.3 <= auc_same <= 0.7


def test_select_records_respects_quotas_and_traces():
    recs, cells = _pool()
    selected, trace = select_records(recs, cells, 20, seed=1)
    assert len(selected) == 20
    assert len(trace) == 20
    from collections import Counter
    per_cell = Counter(r["generation_cell_id"] for r in selected)
    for _cid, cnt in per_cell.items():
        assert cnt <= 20 * 0.25 + 2


def test_split_no_leakage():
    recs, _cells = _pool()
    splits, audit = split_records(recs, {"train": 20, "heldout": 10,
                                         "reserve": 10}, seed=2)
    assert not audit["leaked"]
    ids = [r["task_id"] for rows in splits.values() for r in rows]
    assert len(ids) == len(set(ids)) == len(recs)


def test_leakage_audit_detects_collision():
    recs, _ = _pool(n=8)
    a, b = recs[0], dict(recs[1])
    b["semantic_program_family"] = a["semantic_program_family"]
    audit = leakage_audit({"train": [a], "heldout": [b]})
    assert audit["leaked"]


def test_profile_match_report_keys():
    recs, _ = _pool(n=8)
    from targeted_tool_data.generation import record_to_canonical
    from targeted_tool_data.profile import featurize_row
    feats = [featurize_row(record_to_canonical(r), ["2", "3", "4", "5", "6+"])
             for r in recs]
    rep = profile_match_report(feats, feats, "self")
    assert rep["jsd_call_bucket"] == 0.0
    # identical sets: AUC must not indicate separability (small-sample noise
    # can push CV-AUC below 0.5, which is equally non-separable)
    assert rep["auc_two_sample"] <= 0.8
