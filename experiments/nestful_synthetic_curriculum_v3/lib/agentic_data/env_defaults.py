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

# --- GRPO rollout probe (local weak backend only) ---
ROLLOUT_N = 8
ROLLOUT_TEMPERATURE = 1.0
ROLLOUT_TOP_P = 0.95
ROLLOUT_MAX_TOKENS = 0
ROLLOUT_REQUIRE_ACHIEVABLE_WIN = False

# --- Local weak HF solver ---
LOCAL_WEAK_4BIT = False
LOCAL_WEAK_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
WEAK_SOLVER_BACKEND = "local"


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
