"""Build Pass A / Pass B annotation inputs."""
from __future__ import annotations

import random
from typing import Dict, List, Tuple

from weak_audit.compression import compress_packet, estimate_tokens
from weak_audit.constants import SEED, SYSTEM_PROMPT


def build_pass_a(packet: dict) -> dict:
    return {
        "task_id": packet["task_id"],
        "pass": "A",
        "system_prompt": SYSTEM_PROMPT,
        "case": {
            "task_id": packet["task_id"],
            "cohorts": packet["cohorts"],
            "question": packet["question"],
            "expected_outcome": packet["expected_outcome"],
            "gold_metadata": packet["gold_metadata"],
            "relevant_tools": packet["relevant_tools"],
            "deterministic_flags": packet["deterministic_flags"],
            "C0": packet["C0"],
            "E1": packet["E1"],
            "E2": packet["E2"],
        },
    }


def build_pass_b(
    packet: dict,
    *,
    seed: int = SEED,
) -> Tuple[dict, dict]:
    rng = random.Random(seed + hash(packet["task_id"]) % 100000)
    arms = ["C0", "E1", "E2"]
    labels = ["Trajectory A", "Trajectory B", "Trajectory C"]
    order = list(arms)
    rng.shuffle(order)
    mapping = {labels[i]: order[i] for i in range(3)}
    trajectories = {labels[i]: packet[order[i]] for i in range(3)}
    inp = {
        "task_id": packet["task_id"],
        "pass": "B",
        "system_prompt": SYSTEM_PROMPT.replace("C0/E1/E2", "Trajectory A/B/C"),
        "case": {
            "task_id": packet["task_id"],
            "question": packet["question"],
            "expected_outcome": packet["expected_outcome"],
            "gold_metadata": {
                "gold_call_count": packet["gold_metadata"]["gold_call_count"],
                "motif": packet["gold_metadata"]["motif"],
            },
            "relevant_tools": packet["relevant_tools"],
            "deterministic_flags": {
                k: v for k, v in packet["deterministic_flags"].items()
                if k != "reward_prefers_E2_over_C0"
            },
            "trajectories": trajectories,
        },
    }
    return inp, mapping


def prepare_inputs(
    packets: List[dict],
) -> Tuple[List[dict], List[dict], dict, List[dict], List[dict]]:
    pass_a: List[dict] = []
    pass_b: List[dict] = []
    mapping_all: Dict[str, dict] = {}
    compression_logs: List[dict] = []
    oversize: List[dict] = []

    for pkt in packets:
        compressed, log = compress_packet(pkt)
        log["task_id"] = pkt["task_id"]
        log["stage"] = "packet"
        compression_logs.append(log)
        if log["over_hard_limit"]:
            oversize.append({"task_id": pkt["task_id"], "pass": "packet", **log})
        a = build_pass_a(compressed)
        b, mapping = build_pass_b(compressed)
        mapping_all[pkt["task_id"]] = mapping
        for inp in (a, b):
            tok = estimate_tokens(inp)
            compression_logs.append({
                "task_id": pkt["task_id"],
                "pass": inp["pass"],
                "tokens_estimate": tok,
                "over_hard_limit": tok > log["hard_limit"],
            })
            if tok > log["hard_limit"]:
                oversize.append({
                    "task_id": pkt["task_id"],
                    "pass": inp["pass"],
                    "tokens_estimate": tok,
                })
        pass_a.append(a)
        pass_b.append(b)
    return pass_a, pass_b, mapping_all, compression_logs, oversize
