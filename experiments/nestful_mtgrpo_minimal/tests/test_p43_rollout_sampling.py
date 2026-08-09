"""P43 independent rollout sampling + observability/quota integration tests."""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FACTORY_SRC = ROOT.parent / "targeted_tool_data_factory" / "src"
sys.path.insert(0, str(ROOT))
if FACTORY_SRC.is_dir():
    sys.path.insert(0, str(FACTORY_SRC))

from rollout_sampling import (  # noqa: E402
    ROLLOUT_SAMPLING_VERSION,
    derive_rollout_seed,
    derive_turn_seed,
    stamp_rollout_tasks,
    sampling_source_hash,
)

PROFILE = (
    ROOT.parent / "targeted_tool_data_factory" / "outputs"
    / "pilot4_3_nestful_profile_1000" / "train_nestful_profile_1000.jsonl"
)
CFG = ROOT / "configs" / "qwen3_p43_profile1000_dynamic_online.yaml"


def test_1_eight_logical_rollout_ids():
    task = {"task_id": "tA", "num_calls": 2}
    stamped = stamp_rollout_tasks(
        task, num_generations=8, base_seed=42, global_step=0, epoch=0)
    ids = [(t["task_id"], t["_rollout_index"]) for t in stamped]
    assert len(ids) == 8
    assert len(set(ids)) == 8
    assert [i for _, i in ids] == list(range(8))


def test_2_eight_unique_rollout_seeds():
    stamped = stamp_rollout_tasks(
        {"task_id": "tB"}, num_generations=8, base_seed=42, global_step=3)
    seeds = [t["_rollout_seed"] for t in stamped]
    assert len(seeds) == 8
    assert len(set(seeds)) == 8
    assert 0 not in seeds


def test_3_reproducible_seed_sequence():
    a = stamp_rollout_tasks(
        {"task_id": "tC"}, num_generations=8, base_seed=42, global_step=7, epoch=1)
    b = stamp_rollout_tasks(
        {"task_id": "tC"}, num_generations=8, base_seed=42, global_step=7, epoch=1)
    assert [t["_rollout_seed"] for t in a] == [t["_rollout_seed"] for t in b]


def test_4_different_indices_different_streams():
    s0 = derive_rollout_seed(base_seed=42, global_step=0, task_id="t", rollout_index=0)
    s1 = derive_rollout_seed(base_seed=42, global_step=0, task_id="t", rollout_index=1)
    assert s0 != s1
    assert derive_turn_seed(s0, 0) != derive_turn_seed(s1, 0)


def test_5_world_size_does_not_collide_seeds():
    """Simulate DP round-robin placement; seeds must stay unique per logical idx."""
    stamped = stamp_rollout_tasks(
        {"task_id": "tD"}, num_generations=8, base_seed=42, global_step=0)
    for world in (1, 2, 3, 4):
        # placement: index i → worker i % world (same as rollout_many)
        by_worker = {}
        for t in stamped:
            wid = int(t["_rollout_index"]) % world
            by_worker.setdefault(wid, []).append(t["_rollout_seed"])
        all_seeds = [t["_rollout_seed"] for t in stamped]
        assert len(set(all_seeds)) == 8
        # Critical: workers 0 and 1 must NOT share identical seed pairs
        # for positions that previously collided (0,1), (3,4), (6,7)
        for a, b in ((0, 1), (3, 4), (6, 7)):
            assert stamped[a]["_rollout_seed"] != stamped[b]["_rollout_seed"]


def test_6_raw_order_survives_gather_regroup():
    """Parent reconstructs results by request_id; order must match stamp order."""
    stamped = stamp_rollout_tasks(
        {"task_id": "tE"}, num_generations=8, base_seed=1, global_step=2)
    # Simulate out-of-order worker replies
    results = {i: f"hash-{stamped[i]['_rollout_seed']}" for i in range(8)}
    shuffled_ids = [5, 1, 7, 0, 3, 2, 6, 4]
    gathered = {}
    for rid in shuffled_ids:
        gathered[rid] = results[rid]
    ordered = [gathered[i] for i in range(8)]
    assert ordered == [f"hash-{t['_rollout_seed']}" for t in stamped]


def test_7_no_structurally_forced_fixed_pairs():
    stamped = stamp_rollout_tasks(
        {"task_id": "tF"}, num_generations=8, base_seed=99, global_step=11)
    seeds = [t["_rollout_seed"] for t in stamped]
    for a, b in ((0, 1), (3, 4), (6, 7)):
        assert seeds[a] != seeds[b]


def test_8_group_size_remains_8():
    import yaml
    cfg = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    assert int(cfg["generation"]["num_generations"]) == 8
    assert int((cfg.get("sampler") or {}).get("group_size") or 8) == 8


def test_9_reward_remains_execution_aware_v2_p43():
    import yaml
    cfg = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    assert cfg["reward"]["train_policy"] == "execution_aware_v2_p43"


def test_10_dataset_remains_profile_1000():
    import yaml
    cfg = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    assert "train_nestful_profile_1000.jsonl" in str(cfg["paths"]["train_jsonl"])
    if PROFILE.exists():
        n = sum(1 for _ in PROFILE.open(encoding="utf-8") if _.strip())
        assert n == 1000


def test_11_quota_scheduler_converges_nestful():
    from targeted_tool_data.sampling.nestful_profile import (
        CALL_BUCKETS, NESTFUL_CALL_SHARES, QuotaAccumulator,
    )
    q = QuotaAccumulator()
    stock = {b: 10_000 for b in CALL_BUCKETS}
    for _ in range(1000):
        b = q.pick_bucket(stock)
        q.observe(b)
    # Expected ≈ 330, 220, 135, 95, 220
    expected = {b: int(round(1000 * s)) for b, s in NESTFUL_CALL_SHARES.items()}
    for b in CALL_BUCKETS:
        assert abs(q.actual[b] - expected[b]) <= 2, (b, q.actual[b], expected[b])
    assert q.tv_distance() < 0.01


def test_12_same_bucket_refill():
    from targeted_tool_data.sampling.nestful_profile import (
        NestfulProfileSampler, load_profile_enrichment_refs,
    )
    from nestful_sampler_bridge import refill_same_bucket

    if not PROFILE.exists():
        pytest.skip("profile dataset missing")
    prof, _ = load_profile_enrichment_refs(str(PROFILE), None)
    s = NestfulProfileSampler(prof, config={"sampler_mode": "dynamic_profile"}, seed=0)
    s.task_by_id = {p.prompt_id: {"task_id": p.prompt_id, "num_calls": int(p.call_bucket) if p.call_bucket != "6+" else 6} for p in prof}
    for bucket in ("2", "3", "4", "5", "6+"):
        repl = refill_same_bucket(s, pool="PROFILE", call_bucket=bucket,
                                  exclude_ids=[])
        assert repl is not None
        assert repl["_call_bucket"] == bucket


def test_13_manifest_extra_provenance_complete():
    from grpo_train import _build_train_manifest_extra
    import yaml
    cfg = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    extra = _build_train_manifest_extra(
        config=cfg,
        dispatch_info={
            "configured_policy": "execution_aware_v2_p43",
            "resolved_policy": "execution_aware_v2_p43",
            "reward_fn_module": "nestful_core.rewards",
            "reward_fn_name": "execution_aware_v2_p43",
            "fallback_used": False,
        },
        num_gen=8, epochs=100, grad_accum=4,
        rollout_pool=None, vllm_gen=None,
    )
    required = [
        "base_model", "dataset_path", "dataset_sha256", "reward_policy",
        "reward_variant", "sampler_policy", "group_size", "temperature",
        "top_p", "seed", "qlora", "learning_rate", "kl_beta",
        "rollout_sampling_version", "rollout_sampling_source_hash",
        "git_commit",
    ]
    for k in required:
        assert k in extra, k
        assert extra[k] not in ("", None, {}), k
    assert extra["base_model"] == "Qwen/Qwen3-4B-Instruct-2507"
    assert extra["reward_policy"] == "execution_aware_v2_p43"
    assert extra["rollout_sampling_version"] == ROLLOUT_SAMPLING_VERSION
    assert len(sampling_source_hash()) == 16


def test_14_metrics_not_aliases():
    from grpo_train import _diag_bool_success, _rollout_win_rate
    # Continuous reward must not look like strict pass
    d_bad = {"strict_gold_trace_pass": 0.247}  # mean_r mistaken into pass field
    assert _diag_bool_success(d_bad, "strict_gold_trace_success",
                              "strict_gold_trace_pass") is False
    d_ok = {"strict_gold_trace_success": True}
    assert _diag_bool_success(d_ok, "strict_gold_trace_success",
                              "strict_gold_trace_pass") is True
    # win_rate from rewards>=0.99 is distinct from mean
    rewards = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
    assert _rollout_win_rate(rewards) == 0.0
    assert abs(sum(rewards) / len(rewards) - 0.9) < 1e-9


def test_ddp_seed_invariance_across_world_size():
    """Logical (task, idx, step) seed identical regardless of GPU placement."""
    for world in (1, 3, 8):
        for idx in range(8):
            s = derive_rollout_seed(
                base_seed=42, global_step=5, task_id="inv", rollout_index=idx)
            # world size is NOT an input — invariance by construction
            assert s == derive_rollout_seed(
                base_seed=42, global_step=5, task_id="inv", rollout_index=idx)


def test_raw_bootstrap_not_stuck_on_bucket_2():
    from targeted_tool_data.sampling.nestful_profile import (
        NestfulProfileSampler, load_profile_enrichment_refs, CALL_BUCKETS,
    )
    if not PROFILE.exists():
        pytest.skip("profile dataset missing")
    prof, _ = load_profile_enrichment_refs(str(PROFILE), None)
    s = NestfulProfileSampler(
        prof, config={"sampler_mode": "dynamic_profile",
                      "bootstrap": {"enabled": True}}, seed=0)
    buckets = []
    for _ in range(12):
        b = s.pick_profile_bucket()
        s.note_raw_candidate(b, pool="PROFILE")
        buckets.append(b)
    # Must not be all "2" (pre-fix lexicographic stuck pattern)
    assert len(set(buckets)) >= 3, buckets
    assert buckets.count("2") < 12


def test_generate_fn_passes_seed_to_sampling_params():
    import types
    from unittest.mock import MagicMock, patch
    from vllm_generate import VLLMGenerator

    fake_llm = MagicMock()
    out = MagicMock()
    out.outputs = [MagicMock(text="hi", token_ids=[1, 2, 3])]
    out.prompt_token_ids = [10, 11]
    fake_llm.generate.return_value = [out]
    tok = MagicMock()
    tok.apply_chat_template.return_value = "PROMPT"
    with patch.object(VLLMGenerator, "__init__", lambda self, *a, **k: None):
        gen = VLLMGenerator.__new__(VLLMGenerator)
    gen._llm = fake_llm
    gen._tokenizer = tok
    gen._temperature = 0.7
    gen._top_p = 0.95
    gen._max_model_len = 8192
    gen._adapter_path = None
    gen._enable_lora = False
    gen._make_lora_request = lambda: None

    fake_vllm = types.ModuleType("vllm")
    captured = {}

    class SP:
        def __init__(self, **kw):
            captured.update(kw)

    fake_vllm.SamplingParams = SP
    with patch.dict(sys.modules, {"vllm": fake_vllm}):
        gen.generate_fn([{"role": "user", "content": "x"}], 32, seed=12345)
    assert captured.get("seed") == 12345
    assert captured.get("temperature") == 0.7


def test_simulated_dp_pair_pattern_broken_by_seeds():
    """Document root cause: 3 workers + shared RNG → pairs; unique seeds break it."""
    n_workers = 3
    # Without seeds, worker streams sync → positions i where i%3 equal collide
    # across wave: (0,1) both wave0 on w0/w1; with unique seeds they diverge.
    stamped = stamp_rollout_tasks(
        {"task_id": "sim"}, num_generations=8, base_seed=42, global_step=0)
    assignment = [(i, i % n_workers, stamped[i]["_rollout_seed"]) for i in range(8)]
    # Positions that share a wave and land on different workers:
    pairs = [(0, 1), (3, 4), (6, 7)]
    for a, b in pairs:
        assert assignment[a][1] != assignment[b][1] or n_workers == 1
        assert assignment[a][2] != assignment[b][2]
