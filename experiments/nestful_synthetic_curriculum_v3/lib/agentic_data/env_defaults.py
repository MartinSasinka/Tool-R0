"""Central defaults for agentic data generation.

All knobs are overrideable via environment variables. Launch scripts and
single-process runs should read the same values from here (Python) or mirror
them in shell ``${VAR:-default}`` blocks — keep both in sync when changing.

Tuned for multi-GPU RunPod generation (bf16 weak solver, patient early-stop,
relaxed diversity caps). For laptop pilots: ``LOCAL_WEAK_4BIT=1``,
``MIN_ACCEPT_RATE=0.02``, ``WARMUP_BATCHES=5``.
"""
from __future__ import annotations

import os

# --- Diversity caps (per-worker during generation) ---
DIVERSITY_MAX_SAME_WEAK_SCORE = 0.55
DIVERSITY_MAX_SAME_FAILURE_TYPE = 0.55
DIVERSITY_ENFORCE_AFTER = 50
DIVERSITY_MIN_FAILURE_TYPES_PER_STAGE = 4

# --- Acceptance-rate early-stop (0 = disabled) ---
MIN_ACCEPT_RATE = 0.0
WARMUP_BATCHES = 999_999
MIN_ACCEPT_RATE_RESUME = 0.0
WARMUP_BATCHES_RESUME = 999_999
RESUME_MIN_ITERATIONS = 999_999

# --- Multi-turn solver temperatures ---
SOLVER_MT_WEAK_TEMPERATURE = 0.5
SOLVER_MT_STRONG_TEMPERATURE = 0.7

# --- Acceptance policy (rollout GRPO signal vs legacy solver-gap) ---
AGENTIC_ACCEPTANCE_POLICY = "rollout_primary"   # rollout_primary | solver_gap
AGENTIC_GEN_MODE = "registry_first"             # registry_first | llm_trace

# --- GRPO rollout probe (local weak backend only) ---
ROLLOUT_N = 8
ROLLOUT_TEMPERATURE = 1.0
ROLLOUT_TOP_P = 0.95
ROLLOUT_MAX_TOKENS = 0
ROLLOUT_REQUIRE_ACHIEVABLE_WIN = False
# Universal floor — reject micro-variance even with 2 unique reward buckets.
ROLLOUT_UNIVERSAL_MIN_REWARD_RANGE = 0.01
# Meaningful spread threshold (partial-frontier without multi-class contrast).
ROLLOUT_MIN_REWARD_RANGE = 0.05
ROLLOUT_MAX_PARSE_CLIP_RATE = 0.25
ROLLOUT_BORDERLINE_CONFIRM = True

# --- Quality-tier quotas (per-stage during generation) ---
TIER_QUOTA_ENFORCE_AFTER = 10
# Legacy global defaults (used for unknown stages / merge targets).
TIER_QUOTA_MIN_FRONTIER = 0.50
TIER_QUOTA_MAX_PARTIAL_FRONTIER = 0.35
TIER_QUOTA_MAX_EASY_ANCHOR = 0.15
# Stage 2 — more frontier anchors.
TIER_QUOTA_STAGE2_MIN_FRONTIER = 0.60
TIER_QUOTA_STAGE2_MAX_PARTIAL_FRONTIER = 0.30
TIER_QUOTA_STAGE2_MAX_EASY_ANCHOR = 0.15
# Stage 3 — allow more partial-frontier signal.
TIER_QUOTA_STAGE3_MIN_FRONTIER = 0.35
TIER_QUOTA_STAGE3_MAX_PARTIAL_FRONTIER = 0.60
TIER_QUOTA_STAGE3_MAX_EASY_ANCHOR = 0.15
# Final merged dataset targets (used by merge script stratified pick).
TIER_QUOTA_MERGE_MIN_FRONTIER = 0.50
TIER_QUOTA_MERGE_MAX_PARTIAL_FRONTIER = 0.40
TIER_QUOTA_MERGE_MAX_EASY_ANCHOR = 0.15

# --- Rollout failure-contrast cap (dominant probe failure type) ---
ROLLOUT_FAILURE_MAX_SAME_TYPE = 0.60
ROLLOUT_FAILURE_ENFORCE_AFTER = 10

# --- Local weak HF solver ---
LOCAL_WEAK_4BIT = False
LOCAL_WEAK_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
WEAK_SOLVER_BACKEND = "local"

# --- Best-of-N candidate selection (spec: pick the best candidate out of
# each challenger batch by a composite quality score, instead of accepting
# every candidate that clears the gates in generation order) ---
BEST_OF_N_ENABLED = True
CANDIDATES_PER_REQUEST = 5          # challenger candidates requested per batch
BEST_OF_N_MAX_ACCEPTS_PER_BATCH = 2  # top-K ranked survivors accepted / batch
BEST_OF_N_ACCEPT_ALL_QUALIFIED = False  # accept every grpo_ok survivor (dedup/caps still apply)
BEST_OF_N_WEIGHT_GAP = 0.5           # weight on solver-gap (strong - weak)
BEST_OF_N_WEIGHT_NOVELTY = 0.3       # weight on tool-usage novelty (inverse freq)
BEST_OF_N_WEIGHT_SIGNAL = 0.2        # weight on rollout GRPO-signal quality


def env_str(key: str, default: str) -> str:
    return os.environ.get(key, default)


def env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, str(default)))


def env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, str(default)))


def env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def local_weak_load_4bit() -> bool:
    return env_bool("LOCAL_WEAK_4BIT", LOCAL_WEAK_4BIT)
