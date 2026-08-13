"""Regression tests for low-risk continue350 safety and timing patches."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_unknown_configured_sampler_mode_fails_fast():
    from nestful_sampler_bridge import build_sampler_from_config

    with pytest.raises(ValueError, match="unsupported sampler mode"):
        build_sampler_from_config({"sampler": {"mode": "dynamic_typo"}})


def test_metric_sums_materialize_in_one_group_without_touching_gradients():
    import torch

    from grpo_train import _materialize_metric_sums

    x = torch.tensor(2.0, requires_grad=True)
    loss_terms = [(x * 1.5).detach(), (x * -0.25).detach()]
    logp_terms = [torch.tensor(-2.0), torch.tensor(-3.0)]
    kl_terms = []

    grad_norm = torch.tensor(1.25)
    loss_sum, logp_sum, kl_sum, grad_norm_sum = _materialize_metric_sums(
        loss_terms, logp_terms, kl_terms, [grad_norm])

    assert loss_sum == pytest.approx(2.5)
    assert logp_sum == pytest.approx(-5.0)
    assert kl_sum == 0.0
    assert grad_norm_sum == pytest.approx(1.25)
    assert x.grad is None


def test_timing_profile_reports_amdahl_ceiling():
    from grpo_train import _summarize_timing_profile

    rows = [
        {"group_s": 10.0, "rollout_s": 7.0, "learner_s": 2.0,
         "effective": True}
        for _ in range(10)
    ]
    profile = _summarize_timing_profile(rows)

    assert profile["groups"] == 10
    assert profile["effective_groups"] == 10
    assert profile["rollout_share"] == pytest.approx(0.7)
    assert profile["learner_share"] == pytest.approx(0.2)
    assert profile["other_share"] == pytest.approx(0.1)
    assert profile["perfect_overlap_speedup_ceiling"] == pytest.approx(1.25)
    assert profile["recommendation"] == "prefetch_candidate"


def test_timing_profile_prefers_batched_logprob_when_learner_is_material():
    from grpo_train import _summarize_timing_profile

    profile = _summarize_timing_profile([
        {"group_s": 10.0, "rollout_s": 5.0, "learner_s": 4.0,
         "effective": True}
    ])
    assert profile["recommendation"] == "batched_logprob_first"


def test_continue350_science_and_short_profile_are_unchanged():
    path = (
        ROOT / "configs"
        / "qwen3_p43_profile1000_dynamic_online_continue350_enrich30.yaml"
    )
    config = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert config["training"]["target_optimizer_updates"] == 350
    assert config["training"]["resume"]["expect_global_step"] == 200
    assert config["sampler"]["sampler_mode"] == (
        "dynamic_profile_plus_enrichment")
    assert config["sampler"]["profile_share"] == pytest.approx(0.70)
    assert config["sampler"]["enrichment_share"] == pytest.approx(0.30)
    assert config["logging"]["timing_profile_warmup_groups"] == 2
    assert config["logging"]["timing_profile_groups"] == 10
    assert config["logging"]["log_canary_trajectories"] is False

    path550 = (
        ROOT / "configs"
        / "qwen3_p43_profile1000_dynamic_online_continue550_enrich30.yaml"
    )
    c550 = yaml.safe_load(path550.read_text(encoding="utf-8"))
    assert c550["training"]["target_optimizer_updates"] == 550
    assert c550["training"]["resume"]["expect_global_step"] == 350
    assert c550["sampler"]["sampler_mode"] == (
        "dynamic_profile_plus_enrichment")
    assert c550["sampler"]["profile_share"] == pytest.approx(0.70)
    assert c550["sampler"]["enrichment_share"] == pytest.approx(0.30)

    path750 = (
        ROOT / "configs"
        / "qwen3_p43_profile1000_dynamic_online_continue750_enrich30.yaml"
    )
    c750 = yaml.safe_load(path750.read_text(encoding="utf-8"))
    assert c750["training"]["target_optimizer_updates"] == 750
    assert c750["training"]["resume"]["expect_global_step"] == 550
    assert c750["sampler"]["enrichment_share"] == pytest.approx(0.30)
