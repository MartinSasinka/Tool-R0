"""Tests for NESTFUL profile-preserving GRPO dynamic sampling."""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest

from targeted_tool_data.sampling.nestful_profile import (
    ALL_CORRECT, ALL_FAIL_NO_PROGRESS, ALL_FAIL_WITH_PROGRESS, CALL_BUCKETS,
    INVALID_GROUP, LOW_VARIANCE, MIXED_EFFECTIVE, NESTFUL_CALL_SHARES,
    NestfulProfileSampler, QuotaAccumulator, GroupObservation,
    classify_group, is_effective_nestful, load_profile_enrichment_refs,
    map_group_class, nestful_refill_batch, pool_of,
)

PROFILE = Path(__file__).resolve().parents[1] / "outputs" / "pilot4_3_nestful_profile_1000" / "train_nestful_profile_1000.jsonl"
ENRICH = Path(__file__).resolve().parents[1] / "outputs" / "pilot4_3_nestful_profile_1000" / "train_nestful_enrichment_500.jsonl"


def _obs(pid, rewards, terminals=None, process=None, invalid=False, bucket="2"):
    n = len(rewards)
    term = terminals if terminals is not None else [1.0 if r >= 0.99 else 0.0 for r in rewards]
    proc = process if process is not None else list(rewards)
    o = GroupObservation(
        global_step=0, prompt_id=pid, group_size=n,
        terminal_rewards=term, process_rewards=proc, total_rewards=list(rewards),
        parse_flags=[True] * n, executable_flags=[True] * n,
        call_bucket=bucket,
    )
    if invalid:
        o.group_class = INVALID_GROUP
    return o


@pytest.mark.skipif(not PROFILE.exists(), reason="profile1000 dataset missing")
def test_profile1000_exact_distribution():
    rows = [json.loads(l) for l in PROFILE.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) == 1000
    ids = [r["task_id"] for r in rows]
    assert len(set(ids)) == 1000
    fps = [r["program_fingerprint"] for r in rows]
    assert len(set(fps)) == 1000
    buckets = {}
    for r in rows:
        n = len(r["gold_calls"])
        b = "6+" if n >= 6 else str(n)
        buckets[b] = buckets.get(b, 0) + 1
    assert buckets == {"2": 330, "3": 220, "4": 135, "5": 95, "6+": 220}


@pytest.mark.skipif(not (PROFILE.exists() and ENRICH.exists()), reason="datasets missing")
def test_pool_separation():
    prof, enr = load_profile_enrichment_refs(str(PROFILE), str(ENRICH))
    assert len(prof) == 1000
    assert len(enr) == 500
    assert all(pool_of(p) == "PROFILE" for p in prof)
    assert all(pool_of(p) == "ENRICHMENT" for p in enr)
    assert not ({p.prompt_id for p in prof} & {p.prompt_id for p in enr})
    six = sum(1 for p in enr if p.call_bucket == "6+")
    assert 0.70 <= six / len(enr) <= 0.80


def test_group_size_8_classification():
    rewards = [0.0, 0.2, 0.4, 0.1, 0.9, 0.3, 0.5, 0.6]
    o = _obs("t", rewards, terminals=[0, 0, 0, 0, 1, 0, 0, 0])
    assert o.group_size == 8
    cls = classify_group(o, eps_reward=1e-6, eps_process=1e-6)
    user = map_group_class(cls, o, 1e-6)
    assert user == MIXED_EFFECTIVE
    assert is_effective_nestful(o, {"reward_variance_epsilon": 1e-6,
                                    "drop_low_variance": True})


def test_zero_variance_and_all_correct():
    o = _obs("t", [1.0] * 8, terminals=[1.0] * 8)
    cls = classify_group(o, eps_reward=1e-6, eps_process=1e-6)
    assert map_group_class(cls, o, 1e-6) == ALL_CORRECT
    assert not is_effective_nestful(o, {"reward_variance_epsilon": 1e-6})


def test_all_fail_no_progress():
    o = _obs("t", [0.0] * 8, terminals=[0.0] * 8, process=[0.0] * 8)
    cls = classify_group(o, eps_reward=1e-6, eps_process=1e-6)
    assert map_group_class(cls, o, 1e-6) == ALL_FAIL_NO_PROGRESS
    assert not is_effective_nestful(o, {"reward_variance_epsilon": 1e-6})


def test_all_fail_with_progress_kept():
    # terminal all-fail, but process/total rewards vary
    rewards = [0.10, 0.25, 0.0, 0.40, 0.15, 0.30, 0.05, 0.22]
    o = _obs("t", rewards, terminals=[0.0] * 8, process=rewards)
    cls = classify_group(o, eps_reward=1e-6, eps_process=1e-6)
    user = map_group_class(cls, o, 1e-6)
    assert user == ALL_FAIL_WITH_PROGRESS
    assert is_effective_nestful(o, {"reward_variance_epsilon": 1e-6,
                                    "drop_low_variance": True})


def test_invalid_group_exclusion():
    o = _obs("t", [0.1, 0.2], invalid=True)
    o.group_class = INVALID_GROUP
    assert not is_effective_nestful(o, {"reward_variance_epsilon": 1e-6})


def test_low_variance_drop():
    o = _obs("t", [0.5, 0.5000001] + [0.5] * 6, terminals=[0.0] * 8)
    o.reward_std = 1e-9
    o.group_class = classify_group(o, eps_reward=1e-3)
    user = map_group_class(o.group_class, o, 1e-3)
    assert user in (ALL_FAIL_NO_PROGRESS, LOW_VARIANCE, ALL_FAIL_WITH_PROGRESS)


@pytest.mark.skipif(not PROFILE.exists(), reason="profile missing")
def test_refill_within_same_call_bucket():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "nestful_mtgrpo_minimal"))
    from nestful_sampler_bridge import refill_same_bucket as refill

    prof, _ = load_profile_enrichment_refs(str(PROFILE), None)
    s = NestfulProfileSampler(prof, config={"sampler_mode": "dynamic_profile"}, seed=0)
    s.task_by_id = {p.prompt_id: {"task_id": p.prompt_id} for p in prof}
    repl = refill(s, pool="PROFILE", call_bucket="2", exclude_ids=[])
    assert repl is not None
    assert repl["_call_bucket"] == "2"
    assert repl["_pool"] == "PROFILE"


@pytest.mark.skipif(not (PROFILE.exists() and ENRICH.exists()), reason="datasets missing")
def test_no_cross_pool_refill_by_default():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "nestful_mtgrpo_minimal"))
    from nestful_sampler_bridge import refill_same_bucket

    prof, enr = load_profile_enrichment_refs(str(PROFILE), str(ENRICH))
    s = NestfulProfileSampler(
        prof, enr,
        config={"sampler_mode": "dynamic_profile_plus_enrichment",
                "allow_cross_pool_refill": False},
        seed=1)
    s.task_by_id = {p.prompt_id: {"task_id": p.prompt_id} for p in prof + enr}
    # Ask for a bucket that enrichment may have but force PROFILE pool
    repl = refill_same_bucket(s, pool="PROFILE", call_bucket="2")
    assert repl is None or repl["_pool"] == "PROFILE"


def test_cumulative_quota_convergence():
    q = QuotaAccumulator()
    # simulate 1000 effective groups with deficit picker
    import random
    rng = random.Random(0)
    stock = {b: 10_000 for b in CALL_BUCKETS}
    for _ in range(1000):
        b = q.pick_bucket(stock)
        q.observe(b)
    assert q.tv_distance() < 0.02
    for b, share in NESTFUL_CALL_SHARES.items():
        actual = q.actual[b] / q.total
        assert abs(actual - share) < 0.03


@pytest.mark.skipif(not PROFILE.exists(), reason="profile missing")
def test_history_weight_floor_and_revisit():
    prof, _ = load_profile_enrichment_refs(str(PROFILE), None)
    s = NestfulProfileSampler(
        prof[:50],
        config={"sampler_mode": "dynamic_profile", "minimum_prompt_weight": 0.05,
                "revisit_after_steps": 10,
                "bootstrap": {"enabled": False}},
        seed=2)
    p = s.profile[0]
    # simulate repeated all-correct
    for step in range(3):
        o = _obs(p.prompt_id, [1.0] * 8, terminals=[1.0] * 8, bucket=p.call_bucket)
        o.global_step = step
        s.observe_group(o)
    comps = s.weight_components(p, s.state)
    w = s._combine(comps)
    assert w >= 0.05
    assert w < 0.5  # downweighted


@pytest.mark.skipif(not PROFILE.exists(), reason="profile missing")
def test_checkpoint_resume_rng():
    prof, _ = load_profile_enrichment_refs(str(PROFILE), None)
    s1 = NestfulProfileSampler(prof[:20], config={"sampler_mode": "dynamic_profile"}, seed=7)
    # warm RNG / history, then checkpoint BEFORE the draws we compare
    s1.sample_from_bucket("PROFILE", "2", 2)
    state = s1.state_dict()
    picks1 = [p.prompt_id for p in s1.sample_from_bucket("PROFILE", "2", 3)]
    s2 = NestfulProfileSampler(prof[:20], config={"sampler_mode": "dynamic_profile"}, seed=999)
    s2.load_state_dict(state)
    picks2 = [p.prompt_id for p in s2.sample_from_bucket("PROFILE", "2", 3)]
    assert picks1 == picks2


def test_ab_config_equivalence_except_sampler():
    """A (uniform) vs online-dynamic: shared model/optim; sampler + P43 reward differ."""
    root = Path(__file__).resolve().parents[2] / "nestful_mtgrpo_minimal" / "configs"
    import yaml
    a = yaml.safe_load((root / "qwen3_p43_profile1000_uniform.yaml").read_text(encoding="utf-8"))
    b = yaml.safe_load((root / "qwen3_p43_profile1000_dynamic_online.yaml").read_text(encoding="utf-8"))
    for key in ("finetuning", "mt_grpo"):
        assert a[key] == b[key], f"mismatch in {key}"
    assert a["model"]["base_model"] == b["model"]["base_model"] == "Qwen/Qwen3-4B-Instruct-2507"
    assert a["generation"]["num_generations"] == b["generation"]["num_generations"] == 8
    assert a["generation"]["temperature"] == b["generation"]["temperature"]
    assert a["training"]["learning_rate"] == b["training"]["learning_rate"]
    assert a["training"]["kl_beta"] == b["training"]["kl_beta"]
    assert a["training"]["gradient_accumulation_steps"] == b["training"]["gradient_accumulation_steps"]
    assert a["sampler"]["mode"] == "uniform_profile"
    assert b["sampler"]["mode"] == "dynamic_profile"
    assert b["sampler"]["bootstrap"]["enabled"] is True
    assert b["reward"]["train_policy"] == "execution_aware_v2_p43"
    assert b["reward"]["p43_reward_variant"] == "A"
    assert a["experiment"]["seed"] == b["experiment"]["seed"] == 42


@pytest.mark.skipif(not PROFILE.exists(), reason="profile missing")
def test_nestful_refill_keeps_effective_and_logs_components():
    prof, _ = load_profile_enrichment_refs(str(PROFILE), None)
    s = NestfulProfileSampler(
        prof[:100],
        config={"sampler_mode": "dynamic_profile", "target_effective_groups": 8,
                "max_refill_rounds": 3, "initial_oversample_factor": 1.5,
                "max_raw_groups_per_update_factor": 3.0,
                "bootstrap": {"enabled": False}},
        seed=3)

    def score(prompt, step):
        # half effective mixed, half all-fail-no-progress
        if hash(prompt.prompt_id) % 2 == 0:
            r = [0.0, 0.2, 0.4, 0.1, 0.9, 0.3, 0.5, 0.6]
            t = [0, 0, 0, 0, 1, 0, 0, 0]
        else:
            r = [0.0] * 8
            t = [0.0] * 8
        return _obs(prompt.prompt_id, r, terminals=t, bucket=prompt.call_bucket)

    out = nestful_refill_batch(s, score, global_step=0, target_effective=8)
    assert out["accepted_effective_groups"] <= 8
    assert "group_classes" in out
    assert out["refill_rounds"] >= 1
    assert out["max_raw_groups"] == 24


@pytest.mark.skipif(not PROFILE.exists(), reason="profile missing")
def test_online_bootstrap_uniform_then_history_adaptive():
    prof, _ = load_profile_enrichment_refs(str(PROFILE), None)
    # Use tiny thresholds so we can flip within the test
    s = NestfulProfileSampler(
        prof,
        config={
            "sampler_mode": "dynamic_profile",
            "bootstrap": {
                "enabled": True,
                "min_unique_profile_prompts_seen": 5,
                "min_observed_groups_per_call_bucket": 2,
                "prefer_unseen_prompts": True,
                "unseen_weight": 2.0,
                "seen_weight": 1.0,
            },
        },
        seed=11,
    )
    assert s.in_bootstrap()
    # Bootstrap weights: unseen=2, seen=1
    p0 = s.profile[0]
    w_unseen = s._combine(s.weight_components(p0, s.state))
    assert w_unseen >= 2.0 - 1e-9

    # Observe enough groups across buckets to complete bootstrap
    # Ensure each bucket gets >=2 observations and >=5 unique prompts
    by_b = {}
    for p in s.profile:
        by_b.setdefault(p.call_bucket, []).append(p)
    step = 0
    for b in CALL_BUCKETS:
        for p in by_b.get(b, [])[:3]:
            o = _obs(p.prompt_id, [0.0, 0.3, 0.1, 0.4, 0.2, 0.5, 0.0, 0.25],
                     terminals=[0.0] * 8, bucket=b)
            o.global_step = step
            s.observe_group(o)
            step += 1
    # top up unique prompts if needed
    for p in s.profile:
        if len(s.unique_profile_prompts_seen) >= 5 and not s.in_bootstrap():
            break
        if p.prompt_id in s.unique_profile_prompts_seen:
            continue
        o = _obs(p.prompt_id, [0.1] * 8, terminals=[0.0] * 8, bucket=p.call_bucket)
        o.global_step = step
        s.observe_group(o)
        step += 1
    assert s.bootstrap_complete
    assert not s.in_bootstrap()
    assert s.bootstrap_completed_at_step is not None
    report = s.bootstrap_report()
    assert report["n_groups"] >= 5
    assert "group_class_counts" in report


def test_online_config_exists_and_has_bootstrap():
    import yaml
    root = Path(__file__).resolve().parents[2] / "nestful_mtgrpo_minimal" / "configs"
    path = root / "qwen3_p43_profile1000_dynamic_online.yaml"
    assert path.exists()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert cfg["model"]["base_model"] == "Qwen/Qwen3-4B-Instruct-2507"
    assert cfg["generation"]["num_generations"] == 8
    assert cfg["sampler"]["mode"] == "dynamic_profile"
    assert cfg["sampler"]["bootstrap"]["enabled"] is True
    assert cfg["sampler"]["target_effective_groups"] == cfg["training"]["gradient_accumulation_steps"]
    assert cfg["training"]["target_optimizer_updates"] > 0
    assert cfg["reward"]["train_policy"] == "execution_aware_v2_p43"
    assert cfg["reward"]["p43_reward_variant"] == "A"
