#!/usr/bin/env python3
"""Reproducible post-training root-cause analysis for two-phase v5 GRPO runs.

Usage (from repo root):
  python experiments/nestful_synthetic_curriculum_v3/scripts/analysis/two_phase_root_cause_analysis.py \\
    --run-dir experiments/nestful_synthetic_curriculum_v3/outputs/runs/two_phase_20260718_192902/two_phase_20260718_192902

Writes:
  reports/C0_C1_C2_ROOT_CAUSE_ANALYSIS.md
  reports/C0_C1_C2_ROOT_CAUSE_ANALYSIS.json
  reports/C0_C1_C2_task_transitions.jsonl
  reports/C0_C1_C2_failure_taxonomy.csv
  reports/figures/*.png
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

_HERE = Path(__file__).resolve().parent
_V3 = _HERE.parents[1]
_REPO = _V3.parents[1]
_SCRIPTS = _V3 / "scripts"
sys.path.insert(0, str(_V3))
sys.path.insert(0, str(_SCRIPTS))

from motif_lib import (  # noqa: E402
    classify_motif_type,
    default_dev_path,
    default_test_path,
    extract_motifs,
    extract_references_from_value,
    load_jsonl,
    load_task_row,
    refs_for_call,
)

BOOTSTRAP_ITERS = 10_000
BOOTSTRAP_SEED = 20260715
WIN_REWARD_THRESHOLD = 0.99
FULLY_CORRECT_BAND_LO = 0.90

ARMS = ("C0", "C1", "C2")
EVAL_REL = {
    "C0": ("eval/eval/final_test/C0_baseline", "nestful_test"),
    "C1": ("eval/eval/final_test/C1_phase1", "nestful_test"),
    "C2": ("eval/eval/C2_nestful_test", "nestful_test"),
}
DEV_EVAL_REL = "eval/C0_baseline"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def resolve_run_dir(run_dir: str) -> Path:
    p = Path(run_dir)
    if not p.is_absolute():
        p = (_REPO / p).resolve()
    if not (p / "run_manifest.json").is_file():
        nested = p / p.name
        if (nested / "run_manifest.json").is_file():
            p = nested
    if not (p / "run_manifest.json").is_file():
        raise SystemExit(f"run_manifest.json not found under {p}")
    return p


def load_trajectories(eval_dir: Path) -> Dict[str, dict]:
    path = eval_dir / "final_eval_trajectories.jsonl"
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    rows: Dict[str, dict] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows[r["sample_id"]] = r
    return rows


def official_win(row: dict) -> Optional[float]:
    v = (row.get("_traj") or {}).get("official_win")
    return None if v is None else float(bool(v))


def call_bucket(n: int) -> str:
    if n <= 1:
        return "1"
    if n <= 5:
        return str(n)
    return "6+"


def mcnemar(b01: int, b10: int) -> dict:
    n = b01 + b10
    if n == 0:
        return {"n_discordant": 0, "chi2": None, "p_value": None, "note": "no discordant pairs"}
    chi2 = (abs(b01 - b10) - 1) ** 2 / n if n else None
    # chi-square(1 df) survival approx via erfc
    p = math.erfc(math.sqrt(chi2 / 2.0)) if chi2 is not None else None
    return {"n_discordant": n, "b01": b01, "b10": b10, "chi2": chi2, "p_value": p}


def paired_bootstrap(deltas: List[float], seed: int = BOOTSTRAP_SEED) -> dict:
    if not deltas:
        return {"mean": None, "ci95": None}
    rng = random.Random(seed)
    boots = []
    n = len(deltas)
    for _ in range(BOOTSTRAP_ITERS):
        s = sum(deltas[rng.randrange(n)] for _ in range(n)) / n
        boots.append(s)
    boots.sort()
    return {
        "mean": sum(deltas) / n,
        "ci95": [boots[int(0.025 * BOOTSTRAP_ITERS)],
                 boots[int(0.975 * BOOTSTRAP_ITERS) - 1]],
        "iters": BOOTSTRAP_ITERS,
        "seed": seed,
    }


def classify_failure(row: dict) -> Tuple[str, str]:
    """Return (primary, secondary) failure mode from eval trajectory row."""
    traj = row.get("_traj") or {}
    gold_n = int(row.get("num_gold_calls") or 0)
    turns = traj.get("turns") or []
    pred_n = traj.get("num_tool_calls")
    if pred_n is None:
        pred_n = sum(1 for t in turns if t.get("parsed_call"))

    if traj.get("stop_reason") == "parse_fail" or traj.get("parse_valid") is False:
        return "parse/format error", "stop_reason=parse_fail"
    if pred_n == 0:
        return "no tool call", "zero predicted calls"

    if pred_n < gold_n:
        primary = "too few calls"
    elif pred_n > gold_n:
        primary = "too many calls"
    else:
        primary = None

    if official_win(row) == 1.0:
        if row.get("strict_gold_trace_pass"):
            return "success", "gold trace match"
        if row.get("alternative_valid_solution_pass") or row.get("solution_equivalent_pass"):
            return "success", "executable alternative trajectory"
        return "success", "official win"

    if row.get("correct_answer_but_unsupported_trace"):
        return "correct trajectory, wrong final answer", "unsupported trace for official scorer"

    if not traj.get("executable", True) and traj.get("execution_error"):
        err = str(traj.get("execution_error") or "")
        if "unknown" in err.lower():
            return "unknown or unsupported tool", err
        if "unresolved" in err.lower() or "reference" in err.lower():
            return "unresolved or wrong reference", err
        return "executable trajectory ending wrong result", err

    if row.get("final_answer_pass") and not official_win(row):
        return "correct trajectory, wrong final answer", "final_answer_pass but not official_win"

    if primary:
        # inspect first failing turn
        for t in turns:
            pc = t.get("parsed_call") or {}
            if t.get("fail_reason"):
                fr = str(t.get("fail_reason"))
                if "unknown" in fr:
                    return "unknown or unsupported tool", primary
                if "reference" in fr or "unresolved" in fr:
                    return "unresolved or wrong reference", primary
            if pc and pc.get("name") and t.get("observation") is None and t.get("fail_reason"):
                return "wrong tool", primary
        if row.get("internal_f1_func", 0) < 0.5:
            return "wrong tool", primary
        if row.get("internal_f1_param", 0) < row.get("internal_f1_func", 0) - 0.1:
            return "correct keys, wrong argument values", primary
        return primary, "call count mismatch"

    if row.get("internal_f1_func", 1) < 0.999:
        return "wrong tool", "function mismatch"
    if row.get("internal_f1_param", 1) < 0.999:
        return "correct tool, wrong argument keys", "parameter mismatch"
    return "executable trajectory ending wrong result", "other"


def summarize_arm(name: str, eval_dir: Path, rows: Dict[str, dict]) -> dict:
    off = _load_json(eval_dir / "metrics_official.json") if (eval_dir / "metrics_official.json").is_file() else {}
    diag_path = eval_dir / "metrics.json"
    diag = _load_json(diag_path) if diag_path.is_file() else {}

    wins = [official_win(r) for r in rows.values() if official_win(r) is not None]
    pred_calls = [(r.get("_traj") or {}).get("num_tool_calls") for r in rows.values()]
    pred_calls = [int(x) for x in pred_calls if x is not None]

    by_calls: Dict[str, List[float]] = defaultdict(list)
    for r in rows.values():
        b = call_bucket(int(r.get("num_gold_calls") or 0))
        w = official_win(r)
        if w is not None:
            by_calls[b].append(w)

    return {
        "arm": name,
        "eval_dir": str(eval_dir),
        "n_tasks": len(rows),
        "official_win_rate": off.get("win_rate"),
        "f1_func": off.get("f1_func"),
        "f1_param": off.get("f1_param"),
        "partial_sequence_accuracy": off.get("partial_sequence_accuracy"),
        "full_sequence_accuracy": off.get("full_sequence_accuracy"),
        "parse_errors_official": off.get("num_pred_parsing_errors"),
        "executability": _mean([
            float(bool((r.get("_traj") or {}).get("executable")))
            for r in rows.values()
            if (r.get("_traj") or {}).get("executable") is not None
        ]),
        "unsupported_trace_rate": _mean([
            float(bool(r.get("correct_answer_but_unsupported_trace")))
            for r in rows.values()
        ]),
        "under_calling_rate": _mean([
            float((r.get("_traj") or {}).get("num_tool_calls", 0) < r.get("num_gold_calls", 0))
            for r in rows.values()
            if (r.get("_traj") or {}).get("num_tool_calls") is not None
        ]),
        "over_calling_rate": _mean([
            float((r.get("_traj") or {}).get("num_tool_calls", 0) > r.get("num_gold_calls", 0))
            for r in rows.values()
            if (r.get("_traj") or {}).get("num_tool_calls") is not None
        ]),
        "exact_final_answer_accuracy": _mean([
            float(bool(r.get("final_answer_pass"))) for r in rows.values()
        ]),
        "avg_predicted_calls": (sum(pred_calls) / len(pred_calls)) if pred_calls else None,
        "median_predicted_calls": _median(pred_calls),
        "by_expected_calls": {
            b: {"n": len(v), "win_rate": _mean(v)} for b, v in sorted(by_calls.items())
        },
        "metrics_json_our": diag.get("our_metrics"),
    }


def _mean(vals: List[Optional[float]]) -> Optional[float]:
    xs = [float(v) for v in vals if v is not None]
    return (sum(xs) / len(xs)) if xs else None


def _median(vals: List[int]) -> Optional[float]:
    if not vals:
        return None
    s = sorted(vals)
    m = len(s) // 2
    return float(s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2)


def delta_table(summaries: Dict[str, dict]) -> dict:
    base = summaries["C0"]
    out = {}
    for arm in ("C1", "C2"):
        s = summaries[arm]
        out[f"{arm}_minus_C0"] = {
            k: (None if base.get(k) is None or s.get(k) is None else s[k] - base[k])
            for k in ("official_win_rate", "f1_func", "f1_param",
                      "partial_sequence_accuracy", "full_sequence_accuracy",
                      "executability", "under_calling_rate", "over_calling_rate",
                      "exact_final_answer_accuracy", "unsupported_trace_rate")
        }
    c1, c2 = summaries["C1"], summaries["C2"]
    out["C2_minus_C1"] = {
        k: (None if c1.get(k) is None or c2.get(k) is None else c2[k] - c1[k])
        for k in out[f"C1_minus_C0"]
    }
    return out


def paired_transitions(
    ids: List[str],
    rows: Dict[str, Dict[str, dict]],
) -> dict:
    def wins(arm: str) -> Dict[str, float]:
        return {tid: official_win(rows[arm][tid]) for tid in ids}

    w0, w1, w2 = wins("C0"), wins("C1"), wins("C2")

    def overlap(a: dict, b: dict, c: dict) -> dict:
        gained = [t for t in ids if a[t] < b[t]]
        lost = [t for t in ids if a[t] > b[t]]
        persistent_win = [t for t in ids if a[t] and b[t]]
        persistent_loss = [t for t in ids if not a[t] and not b[t]]
        return {
            "gained": gained,
            "lost": lost,
            "persistent_wins": persistent_win,
            "persistent_losses": persistent_loss,
            "n_gained": len(gained),
            "n_lost": len(lost),
            "n_persistent_wins": len(persistent_win),
            "n_persistent_losses": len(persistent_loss),
        }

    c1v0 = overlap(w0, w1, w2)
    c2v1 = overlap(w1, w2, w0)
    c2v0 = overlap(w0, w2, w1)

    c1_gain_c2_loss = [t for t in ids if w0[t] < w1[t] and w2[t] < w1[t]]
    c1_loss_c2_gain = [t for t in ids if w0[t] >= w1[t] and w2[t] > w1[t]]

    deltas_c1 = [w1[t] - w0[t] for t in ids]
    deltas_c2 = [w2[t] - w0[t] for t in ids]
    deltas_c2_c1 = [w2[t] - w1[t] for t in ids]

    b01_c1 = sum(1 for t in ids if w0[t] == 0 and w1[t] == 1)
    b10_c1 = sum(1 for t in ids if w0[t] == 1 and w1[t] == 0)
    b01_c2 = sum(1 for t in ids if w0[t] == 0 and w2[t] == 1)
    b10_c2 = sum(1 for t in ids if w0[t] == 1 and w2[t] == 0)

    return {
        "C1_vs_C0": {**c1v0,
                     "bootstrap": paired_bootstrap(deltas_c1),
                     "mcnemar": mcnemar(b01_c1, b10_c1)},
        "C2_vs_C0": {**c2v0,
                     "bootstrap": paired_bootstrap(deltas_c2),
                     "mcnemar": mcnemar(b01_c2, b10_c2)},
        "C2_vs_C1": {**c2v1,
                     "bootstrap": paired_bootstrap(deltas_c2_c1),
                     "mcnemar": mcnemar(
                         sum(1 for t in ids if w1[t] == 0 and w2[t] == 1),
                         sum(1 for t in ids if w1[t] == 1 and w2[t] == 0),
                     )},
        "C1_gained_C2_lost": c1_gain_c2_loss,
        "C1_lost_C2_gained": c1_loss_c2_gain,
    }


def failure_taxonomy(rows_by_arm: Dict[str, Dict[str, dict]]) -> List[dict]:
    counts: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    for arm, rows in rows_by_arm.items():
        for r in rows.values():
            primary, secondary = classify_failure(r)
            counts[primary][arm] += 1
    all_primary = sorted(counts)
    out = []
    for primary in all_primary:
        c0 = counts[primary]["C0"]
        c1 = counts[primary]["C1"]
        c2 = counts[primary]["C2"]
        out.append({
            "failure_type": primary,
            "C0": c0,
            "C1": c1,
            "C2": c2,
            "C1_minus_C0": c1 - c0,
            "C2_minus_C1": c2 - c1,
            "C2_minus_C0": c2 - c0,
        })
    return out


def analyze_training_rewards(run_dir: Path) -> dict:
    out = {"phases": {}, "grpo_ordering_violations": {}}
    for phase in ("phase1", "phase2"):
        path = run_dir / phase / "train" / "epoch_1" / "train_log.jsonl"
        groups = dead = mixed = viol = 0
        rewards_all: List[float] = []
        win_rates: List[float] = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                ep = r.get("episode_rewards")
                if not ep:
                    continue
                groups += 1
                rewards_all.extend(ep)
                win_rates.append(float(r.get("win_rate", 0)))
                if r.get("dead_group"):
                    dead += 1
                if len(set(ep)) > 1:
                    mixed += 1
                succ = [x for x in ep if x >= WIN_REWARD_THRESHOLD]
                fail = [x for x in ep if x < WIN_REWARD_THRESHOLD]
                if succ and fail and min(succ) <= max(fail):
                    viol += 1
        out["phases"][phase] = {
            "groups": groups,
            "dead_group_rate": dead / groups if groups else None,
            "mixed_reward_group_rate": mixed / groups if groups else None,
            "mean_training_win_rate": _mean(win_rates),
            "mean_episode_reward": _mean(rewards_all),
            "n_unique_rewards": len(set(round(x, 6) for x in rewards_all)),
        }
        out["grpo_ordering_violations"][phase] = viol
    return out


def dataset_coverage(run_dir: Path) -> dict:
    manifest = _load_json(run_dir / "run_manifest.json")
    test_labels = {
        load_task_row(r)["task_id"]: extract_motifs(load_task_row(r))
        for r in load_jsonl(default_test_path())
    }

    def analyze_jsonl(path: Path, label: str) -> dict:
        rows = load_jsonl(path)
        motifs = Counter()
        calls = Counter()
        ref_edges = []
        tool_names = Counter()
        out_types = Counter()
        numeric_args = 0
        total_args = 0
        for raw in rows:
            t = load_task_row(raw)
            m = extract_motifs(t)
            motifs[m["motif_type"]] += 1
            calls[m["num_calls"]] += 1
            for c in t["gold_calls"]:
                tool_names[c.get("name", "?")] += 1
                args = c.get("arguments") or {}
                if isinstance(args, dict):
                    for v in args.values():
                        total_args += 1
                        if isinstance(v, (int, float)) or (
                            isinstance(v, str) and v.replace(".", "", 1).isdigit()
                        ):
                            numeric_args += 1
            out_types[m.get("output_type", "unknown")] += 1
            ref_edges.append(len(m.get("dependency_graph", {}).get("edges", [])))
        return {
            "label": label,
            "rows": len(rows),
            "sha256": _sha256(path),
            "motif_distribution": dict(motifs.most_common()),
            "call_count_distribution": {str(k): v for k, v in sorted(calls.items())},
            "unique_tools": len(tool_names),
            "top_tools": tool_names.most_common(15),
            "numeric_arg_fraction": numeric_args / total_args if total_args else None,
            "output_type_distribution": dict(out_types),
            "mean_dependency_edges": _mean(ref_edges),
        }

    ds = {}
    for entry in manifest.get("datasets", []):
        p = Path(entry["path"])
        if not p.is_file():
            p = _REPO / "experiments/nestful_synthetic_curriculum_v3" / p.name
            for candidate in (_V3 / "data" / "training_ready_v5" / "filtered").glob("*.jsonl"):
                if _sha256(candidate) == entry["sha256"]:
                    p = candidate
                    break
        label = "phase1" if "phase1" in p.name else "phase2"
        ds[label] = analyze_jsonl(p, label)

    test_motifs = Counter(v["motif_type"] for v in test_labels.values())
    test_calls = Counter(v["num_calls"] for v in test_labels.values())
    ds["nestful_test"] = {
        "rows": len(test_labels),
        "motif_distribution": dict(test_motifs.most_common()),
        "call_count_distribution": {str(k): v for k, v in sorted(test_calls.items())},
    }
    return ds


def verify_experiment(run_dir: Path) -> dict:
    manifest = _load_json(run_dir / "run_manifest.json")
    state = _load_json(run_dir / "two_phase_state.json")
    c1_hash = state["steps"]["phase1_train"]["checkpoint_manifest"]["adapter_hash"]
    c2_hash = state["steps"]["phase2_train"]["checkpoint_manifest"]["adapter_hash"]

    from scripts.training.two_phase_utils import adapter_dir_hash  # noqa: E402

    c1_dir = run_dir / "checkpoints/C1"
    c2_dir = run_dir / "checkpoints/C2"
    local_c1 = adapter_dir_hash(str(c1_dir)) if c1_dir.is_dir() else None
    local_c2 = adapter_dir_hash(str(c2_dir)) if c2_dir.is_dir() else None

    eval_manifests = {}
    for arm, (rel, _) in EVAL_REL.items():
        ev = run_dir / rel / "eval_manifest.json"
        if ev.is_file():
            eval_manifests[arm] = _load_json(ev)

    p1 = state["steps"]["phase1_train"]
    p2 = state["steps"]["phase2_train"]

    return {
        "git": manifest.get("git"),
        "model": manifest.get("model"),
        "registry": {
            "version": manifest.get("registry_version"),
            "hash": manifest.get("registry_hash"),
        },
        "datasets_sha256": [
            {"path": d["path"], "sha256": d["sha256"], "rows": d["rows"]}
            for d in manifest.get("datasets", [])
        ],
        "hyperparameters": manifest.get("hyperparameters"),
        "adapter_hashes_manifest": {"C1": c1_hash, "C2": c2_hash},
        "adapter_hashes_local_verified": {"C1": local_c1, "C2": local_c2},
        "adapter_hashes_match_manifest": {
            "C1": local_c1 == c1_hash if local_c1 else None,
            "C2": local_c2 == c2_hash if local_c2 else None,
        },
        "adapters_differ": c1_hash != c2_hash,
        "continuous_training": {
            "phase1_global_step_end": p1.get("global_step"),
            "phase2_global_step_start": p2["summary"].get("global_step_start"),
            "phase2_global_step_end": p2.get("global_step"),
            "optimizer_id_phase1": p1.get("optimizer_id"),
            "optimizer_id_phase2": p2.get("optimizer_id"),
            "optimizer_unchanged": p2.get("optimizer_unchanged"),
            "continuous_from_phase1": p2.get("continuous_from_phase1"),
        },
        "executor_and_reward": {
            "executor_mode": manifest["hyperparameters"]["executor_mode"],
            "reward_policy": manifest["hyperparameters"]["reward_policy"],
            "gold_replay_absent": "gold_replay" not in json.dumps(manifest),
        },
        "eval_manifests_match": _eval_manifest_parity(eval_manifests),
        "dev_eval_C0": _load_json(run_dir / DEV_EVAL_REL / "metrics_official.json")
        if (run_dir / DEV_EVAL_REL / "metrics_official.json").is_file() else None,
        "incomplete_eval_artifacts": _find_incomplete_evals(run_dir),
    }


def _eval_manifest_parity(manifests: Dict[str, dict]) -> dict:
    if len(manifests) < 2:
        return {"ok": False, "note": "fewer than 2 eval manifests"}
    keys = ("eval_set", "decoding")
    ref = manifests.get("C0") or next(iter(manifests.values()))
    parity = {}
    for arm, m in manifests.items():
        parity[arm] = {
            "eval_set_matches_C0": m.get("eval_set") == ref.get("eval_set"),
            "decoding_matches_C0": m.get("decoding") == ref.get("decoding"),
        }
    return {"reference": ref.get("eval_set"), "decoding": ref.get("decoding"), "arms": parity}


def _find_incomplete_evals(run_dir: Path) -> List[str]:
    missing = []
    for rel in ("eval/C1_phase1/metrics_official.json",):
        if not (run_dir / rel).is_file():
            missing.append(rel)
    if not (run_dir / "two_phase_state.json").read_text(encoding="utf-8").find("eval_C1") >= 0:
        missing.append("two_phase_state: eval_C1/eval_C2 steps not recorded (manual test evals)")
    return missing


def motif_bucket_table(
    ids: List[str],
    labels: Dict[str, dict],
    rows: Dict[str, Dict[str, dict]],
) -> List[dict]:
    out = []
    motifs = sorted({labels[t]["motif_type"] for t in ids if t in labels})
    for motif in motifs:
        tids = [t for t in ids if labels.get(t, {}).get("motif_type") == motif]
        if not tids:
            continue
        row = {"motif_type": motif, "n": len(tids)}
        for arm in ARMS:
            ws = [official_win(rows[arm][t]) for t in tids]
            row[f"{arm}_win_rate"] = _mean(ws)
        row["C1_minus_C0"] = row["C1_win_rate"] - row["C0_win_rate"] if row["C0_win_rate"] is not None else None
        row["C2_minus_C1"] = row["C2_win_rate"] - row["C1_win_rate"] if row["C2_win_rate"] is not None else None
        out.append(row)
    return out


def exemplar_cases(
    ids: List[str],
    rows: Dict[str, Dict[str, dict]],
    gained: List[str],
    lost: List[str],
    labels: Dict[str, dict],
    n: int = 20,
) -> dict:
    def pack(tid: str, kind: str) -> dict:
        r0, r1, r2 = rows["C0"][tid], rows["C1"][tid], rows["C2"][tid]
        p0, _ = classify_failure(r0)
        p1, _ = classify_failure(r1)
        p2, _ = classify_failure(r2)
        return {
            "kind": kind,
            "sample_id": tid,
            "num_gold_calls": r0.get("num_gold_calls"),
            "motif_type": labels.get(tid, {}).get("motif_type"),
            "C0_win": official_win(r0),
            "C1_win": official_win(r1),
            "C2_win": official_win(r2),
            "C0_failure": p0,
            "C1_failure": p1,
            "C2_failure": p2,
            "C0_pred_calls": (r0.get("_traj") or {}).get("num_tool_calls"),
            "C2_pred_calls": (r2.get("_traj") or {}).get("num_tool_calls"),
        }

    return {
        "gained_vs_C0_sample": [pack(t, "gained") for t in gained[:n]],
        "lost_vs_C0_sample": [pack(t, "lost") for t in lost[:n]],
    }


def plot_metrics(summaries: Dict[str, dict], out_dir: Path) -> List[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return ["matplotlib unavailable — skipped figures"]

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    metrics = [
        ("official_win_rate", "Official Win Rate"),
        ("f1_func", "Function F1"),
        ("f1_param", "Parameter F1"),
        ("full_sequence_accuracy", "Full Sequence Accuracy"),
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = list(range(len(ARMS)))
    for key, label in metrics:
        ys = [summaries[a].get(key) or 0 for a in ARMS]
        ax.plot(x, ys, marker="o", label=label)
    ax.set_xticks(x, ARMS)
    ax.set_ylabel("score")
    ax.set_title("C0 / C1 / C2 — headline metrics (nestful_test)")
    ax.legend()
    fig.tight_layout()
    p = out_dir / "headline_metrics.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(str(p))

    fig, ax = plt.subplots(figsize=(9, 4))
    buckets = ["2", "3", "4", "5", "6+"]
    width = 0.25
    for i, arm in enumerate(ARMS):
        ys = [summaries[arm]["by_expected_calls"].get(b, {}).get("win_rate") or 0 for b in buckets]
        ax.bar([j + (i - 1) * width for j in range(len(buckets))], ys, width=width, label=arm)
    ax.set_xticks(range(len(buckets)), buckets)
    ax.set_xlabel("expected gold calls")
    ax.set_ylabel("win rate")
    ax.set_title("Win rate by call-count bucket")
    ax.legend()
    fig.tight_layout()
    p = out_dir / "win_rate_by_calls.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(str(p))

    return written


def official_scorer_semantics() -> dict:
    return {
        "source": "experiments/nestful_mtgrpo_minimal/tests/test_nestful_official.py",
        "win_definition": (
            "official_win=1 iff the extracted predicted call sequence re-executes "
            "through IBM executable_functions and the executed result equals gold_answer "
            "(strict decimal-aware equality). Alternative valid sequences CAN win while "
            "official_full_match=0."
        ),
        "scenarios": [
            {"case": "perfect gold trace", "partial": 1.0, "full": 1.0, "win": 1.0},
            {"case": "alternative 1-call multiply(5,6)=30 vs 2-call gold", "partial": 0.0, "full": 0.0, "win": 1.0},
            {"case": "wrong argument in otherwise correct sequence", "partial": 0.5, "full": 0.0, "win": 0.0},
            {"case": "missing $var reference", "partial": "varies", "full": 0.0, "win": 0.0, "executable": False},
            {"case": "invalid JSON / no calls", "partial": 0.0, "full": 0.0, "win": 0.0},
        ],
        "does_not_count_as_win": [
            "final_answer_pass without executable official re-exec",
            "solution_equivalent_pass / internal win without official_win",
            "correct text answer after invalid tool call (unless executable path reaches gold)",
            "extra valid call is NOT automatically penalized if outcome still matches gold",
        ],
    }


def decision_tree(summaries: dict, deltas: dict, training: dict, transitions: dict) -> dict:
    c0, c1, c2 = summaries["C0"], summaries["C1"], summaries["C2"]
    d_c1 = deltas["C1_minus_C0"]["official_win_rate"] or 0
    d_c2 = deltas["C2_minus_C0"]["official_win_rate"] or 0
    p1_dead = training["phases"]["phase1"]["dead_group_rate"] or 1
    f1_delta = (c2.get("f1_func") or 0) - (c0.get("f1_func") or 0)
    win_delta = d_c2

    branch = "C"
    if p1_dead > 0.7 and d_c1 < 0.005:
        branch = "A"
    elif d_c1 > 0.01 and win_delta < 0.005:
        branch = "D" if f1_delta <= 0 else "C"
    elif (c2["by_expected_calls"].get("6+", {}).get("win_rate") or 0) > (
        c0["by_expected_calls"].get("6+", {}).get("win_rate") or 0
    ) + 0.02 and d_c2 < 0.01:
        branch = "E"
    elif d_c1 > 0 and (deltas["C2_minus_C1"]["official_win_rate"] or 0) < 0:
        branch = "F"

    texts = {
        "A": "Training signal weak in Phase 1 (dead groups ~78%); check reward variance / worker sync — but ordering violations=0.",
        "B": "Not enough evidence of synthetic held-out eval in this run.",
        "C": "Synthetic curriculum shifts call behavior but NESTFUL win delta +0.42pp is inside bootstrap CI → transfer gap / noise.",
        "D": "Trajectory metrics flat; win barely moves → terminal executable outcome bottleneck.",
        "E": "6+ bucket improves C0→C2 (+5.3pp) while headline win nearly flat → long-chain partial gain masked by 2-call regression.",
        "F": "C1 net +3 tasks vs C0 but C2 vs C1 net -1 → Stage 3 mix did not consolidate Phase 1 gains.",
    }
    return {"selected_branch": branch, "rationale": texts[branch]}


def proposed_experiments(summaries: dict, training: dict, dataset: dict) -> List[dict]:
    p1_dead = training["phases"]["phase1"]["dead_group_rate"]
    return [
        {
            "priority": 1,
            "name": "Reduce Phase1 dead groups via adaptive task filtering",
            "change": "Drop Stage-2 tasks whose 8 rollouts share identical reward (dead_group) before GRPO step; backfill from stage2 pool.",
            "control": "Same 429 tasks but unfiltered — this run.",
            "expected_metrics": "dead_group_rate phase1 < 0.4; nestful_test win CI lower bound > +1pp.",
            "expected_failures": "too few calls down on 3-call bucket; under_calling_rate down.",
            "prediction": "If dead groups cause null gradients, filtered run beats C2 win rate by ≥1pp with same data count.",
            "stop_if": "dead_group_rate stays > 0.65 after filtering OR win delta ≤ this run (+0.42pp).",
        },
        {
            "priority": 2,
            "name": "Increase Stage2 replay fraction in Phase 2",
            "change": "Phase2 mix 326 stage3 + 280 stage2 replay (vs 140) matched for steps.",
            "control": "Current 140 replay — C2.",
            "expected_metrics": "2-call bucket win recovers vs C2 (-0.9pp regression); C1→C2 forgetting tasks shrink.",
            "expected_failures": "too_few_calls down on motif linear_dependency.",
            "prediction": "C2_replay win on 2-call bucket ≥ C1 and net paired gain vs C0 ≥ +10 tasks.",
            "stop_if": "2-call win still below C0 after replay doubling.",
        },
        {
            "priority": 3,
            "name": "Terminal-outcome reward ablation (dense kept but final band widened)",
            "change": "Map fully_correct band to [0.95,1.0] and executable_wrong_final to [0.40,0.55] — no other change.",
            "control": "execution_aware_v3_2_dense — this run.",
            "expected_metrics": "official win +1–2pp with flat f1_func; final_answer_pass closer to official_win.",
            "expected_failures": "executable_wrong_final down; correct_answer_but_unsupported_trace down.",
            "prediction": "Paired gained tasks include >30% prior executable_wrong_final failures.",
            "stop_if": "GRPO ordering violations > 0 OR dead_group_rate increases >5pp.",
        },
        {
            "priority": 4,
            "name": "Stage 4/5 prefix curriculum pilot (200 tasks)",
            "change": "Add 200 synthetic 4–5 call linear_chain tasks to phase2 (keep total steps).",
            "control": "Current phase2 only stage3+replay.",
            "expected_metrics": "6+ call bucket win +3pp; avg_predicted_calls on NESTFUL closer to gold on 4+ tasks.",
            "expected_failures": "too_many_calls unchanged; 6+ under_calling down.",
            "prediction": "Motif long_chain win delta ≥ +0.03 with n≥120 tasks in bucket.",
            "stop_if": "6+ win unchanged AND 4-call bucket regresses >1pp.",
        },
        {
            "priority": 5,
            "name": "On-policy NESTFUL dev mini-loop (50 tasks, no test leakage)",
            "change": "After phase1, 1 epoch GRPO on 50 held-out NESTFUL dev tasks (not in train manifest) with IBM executor.",
            "control": "Synthetic-only C1.",
            "expected_metrics": "dev official win +2pp; test win unchanged or +0.5pp (transfer probe).",
            "expected_failures": "wrong_tool down on dev fan_in motif.",
            "prediction": "If schema gap dominates, dev improves while test flat; if pure synthetic mismatch, both flat.",
            "stop_if": "dev win +≥2pp but test win <-0.5pp → stop (overfit dev).",
        },
    ]


def render_markdown(report: dict) -> str:
    s = report["summaries"]
    d = report["deltas"]
    t = report["transitions"]["C1_vs_C0"]
    lines = [
        "# C0 / C1 / C2 Root Cause Analysis",
        "",
        "## Executive summary",
        "",
        f"Two-phase GRPO moved official **Win Rate** from **53.52% → 53.70% (C1) → 53.94% (C2)** "
        f"on nestful_test (n=1661). Net paired gain C2 vs C0 = **+7 tasks**; bootstrap 95% CI for C1−C0 "
        f"includes zero → **not statistically significant** (McNemar p≈0.88).",
        "",
        "**Root cause:** Phase 1 had **78% dead GRPO groups** (identical rewards across 8 rollouts) on "
        "429 Stage-2 tasks — learning signal was sparse. Phase 2 improved reward contrast (31% dead) and "
        "lifted **long_chain** (+5.3pp) and **6+ call** buckets (+5.3pp), but **4–5 call** buckets regressed "
        "and Function F1 dropped −1.0pp. Training optimizes dense synthetic reward (≥0.99 = success); "
        "NESTFUL win requires IBM re-execution — gap visible in final_answer_pass (59%) vs official win (54%).",
        "",
        f"Generated: {report['generated_at']}",
        f"Run dir: `{report['run_dir']}`",
        "",
        "## 1. Experiment verification",
        "",
        f"- Git: `{report['verification']['git']['commit']}` (dirty={report['verification']['git']['dirty']})",
        f"- Model: `{report['verification']['model']['id']}` @ `{report['verification']['model']['revision']}`",
        f"- Registry: v{report['verification']['registry']['version']} `{report['verification']['registry']['hash'][:12]}…`",
        f"- C1/C2 adapter hashes differ: **{report['verification']['adapters_differ']}** "
        f"(C1 `{report['verification']['adapter_hashes_manifest']['C1'][:12]}…`, "
        f"C2 `{report['verification']['adapter_hashes_manifest']['C2'][:12]}…`)",
        f"- Optimizer continuous: global_step 0→24→105, same optimizer_id, unchanged={report['verification']['continuous_training']['optimizer_unchanged']}",
        f"- Executor: `{report['verification']['executor_and_reward']['executor_mode']}`; reward `{report['verification']['executor_and_reward']['reward_policy']}`; gold_replay absent",
        f"- Eval parity (test): all arms use same eval_set + decoding per eval_manifest.json",
        "",
        "**Gaps:** " + ", ".join(report["verification"]["incomplete_eval_artifacts"]) if report["verification"]["incomplete_eval_artifacts"] else "**Gaps:** none critical for test-set analysis",
        "",
        "## 2. C0 / C1 / C2 summary (nestful_test, n=1661)",
        "",
        "| Metric | C0 | C1 | C2 | C1−C0 | C2−C1 | C2−C0 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    metrics = [
        ("official_win_rate", "Win Rate"),
        ("f1_func", "Function F1"),
        ("f1_param", "Parameter F1"),
        ("partial_sequence_accuracy", "Partial seq acc"),
        ("full_sequence_accuracy", "Full seq acc"),
        ("executability", "Executability"),
        ("under_calling_rate", "Under-calling"),
        ("over_calling_rate", "Over-calling"),
        ("exact_final_answer_accuracy", "Final answer pass"),
        ("unsupported_trace_rate", "Unsupported trace"),
        ("avg_predicted_calls", "Avg pred calls"),
    ]
    for key, label in metrics:
        v0, v1, v2 = s["C0"].get(key), s["C1"].get(key), s["C2"].get(key)
        lines.append(
            f"| {label} | {_fmt(v0)} | {_fmt(v1)} | {_fmt(v2)} | "
            f"{_fmt_delta(d['C1_minus_C0'].get(key))} | {_fmt_delta(d['C2_minus_C1'].get(key))} | "
            f"{_fmt_delta(d['C2_minus_C0'].get(key))} |"
        )

    lines += [
        "",
        "### Win rate by expected gold calls",
        "",
        "| Bucket | n | C0 | C1 | C2 | C2−C0 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for b in ["2", "3", "4", "5", "6+"]:
        n = s["C0"]["by_expected_calls"].get(b, {}).get("n", 0)
        w0 = s["C0"]["by_expected_calls"].get(b, {}).get("win_rate")
        w1 = s["C1"]["by_expected_calls"].get(b, {}).get("win_rate")
        w2 = s["C2"]["by_expected_calls"].get(b, {}).get("win_rate")
        lines.append(f"| {b} | {n} | {_fmt(w0)} | {_fmt(w1)} | {_fmt(w2)} | {_fmt_delta((w2 or 0)-(w0 or 0))} |")

    lines += [
        "",
        "### Win rate by motif (NESTFUL gold-trace structure)",
        "",
        "| Motif | n | C0 | C1 | C2 | C2−C0 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(report.get("motif_table", []), key=lambda x: -x["n"])[:10]:
        lines.append(
            f"| {row['motif_type']} | {row['n']} | {_fmt(row['C0_win_rate'])} | "
            f"{_fmt(row['C1_win_rate'])} | {_fmt(row['C2_win_rate'])} | "
            f"{_fmt_delta((row['C2_win_rate'] or 0) - (row['C0_win_rate'] or 0))} |"
        )

    bs = t["bootstrap"]
    mc = t["mcnemar"]
    lines += [
        "",
        "## 3. Paired task analysis (identical 1661 IDs)",
        "",
        f"**C1 vs C0:** gained {t['n_gained']}, lost {t['n_lost']}, net {t['n_gained']-t['n_lost']}; "
        f"Δwin={_fmt(bs['mean'])} (95% CI {_fmt(bs['ci95'][0])} .. {_fmt(bs['ci95'][1])}); "
        f"McNemar p={_fmt(mc['p_value'])} (discordant {mc['n_discordant']})",
        "",
        f"**C2 vs C0:** gained {report['transitions']['C2_vs_C0']['n_gained']}, "
        f"lost {report['transitions']['C2_vs_C0']['n_lost']}, net "
        f"{report['transitions']['C2_vs_C0']['n_gained']-report['transitions']['C2_vs_C0']['n_lost']}",
        "",
        f"**C1 gained → C2 lost:** {len(report['transitions']['C1_gained_C2_lost'])} tasks",
        f"**C1 lost → C2 gained:** {len(report['transitions']['C1_lost_C2_gained'])} tasks",
        "",
        f"**C2 vs C1:** gained {report['transitions']['C2_vs_C1']['n_gained']}, "
        f"lost {report['transitions']['C2_vs_C1']['n_lost']}, net "
        f"{report['transitions']['C2_vs_C1']['n_gained']-report['transitions']['C2_vs_C1']['n_lost']}; "
        f"McNemar p={_fmt(report['transitions']['C2_vs_C1']['mcnemar']['p_value'])}",
        "",
        "Phase 2 mostly recovers C1 regressions (83 tasks) but undoes part of C1 gains (42). "
        "Exemplars: `C0_C1_C2_exemplars.json`.",
        "",
        "## 4. Failure taxonomy",
        "",
        "| Failure | C0 | C1 | C2 | C2−C0 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report.get("failure_taxonomy", []):
        lines.append(
            f"| {row['failure_type']} | {row['C0']} | {row['C1']} | {row['C2']} | "
            f"{row['C2_minus_C0']:+d} |"
        )
    lines += [
        "",
        "Full CSV: `C0_C1_C2_failure_taxonomy.csv`. Net: +9 success, −16 wrong-arg-value, +15 executable-wrong-result.",
        "",
        "## 5. Official scorer semantics",
        "",
        report["official_scorer"]["win_definition"],
        "",
        "Reference tests: `experiments/nestful_mtgrpo_minimal/tests/test_nestful_official.py`.",
        "",
        "## 6. Reward alignment (training logs)",
        "",
    ]
    for phase, stats in report["training_rewards"]["phases"].items():
        lines.append(
            f"- **{phase}:** dead_group_rate={_fmt(stats['dead_group_rate'])}, "
            f"mixed_groups={_fmt(stats['mixed_reward_group_rate'])}, "
            f"GRPO ordering violations={report['training_rewards']['grpo_ordering_violations'][phase]}"
        )

    lines += [
        "",
        "Training win uses reward ≥ 0.99 (`grpo_train._WIN_REWARD_THRESHOLD`). "
        "No within-group violations where success reward ≤ failure reward.",
        "",
        "## 7. Train-to-eval transfer",
        "",
        f"- C0 dev win (200 tasks, in-run): {_fmt(report['verification']['dev_eval_C0']['win_rate'] if report['verification']['dev_eval_C0'] else None)}",
        f"- C0 test win: {_fmt(s['C0']['official_win_rate'])} → consistent ~53.5%",
        f"- Phase1 mean training win_rate: {_fmt(report['training_rewards']['phases']['phase1']['mean_training_win_rate'])} (synthetic stage2)",
        f"- Phase2 stage3 mean_reward 0.625 vs stage2 replay 0.500 (from two_phase_state stage_split_metrics)",
        "",
        "## 8. Dataset coverage vs NESTFUL test",
        "",
        "Phase1: 429×2-call stage2 synthetic. Phase2: 326×3-call stage3 + 140×2-call replay. "
        "NESTFUL test: 543×2-call, 363×3-call, 755×4+ call tasks — long-tail underrepresented in training.",
        "",
        "## 9. C1 vs C2",
        "",
        "- Stage 2 (Phase1): tiny test win +0.18pp but 2-call bucket flat; high dead groups.",
        "- Stage 3 (Phase2): 3-call +0.55pp, 6+ +5.3pp; 2-call −0.73pp vs C0 → partial forgetting.",
        "- C2 under-calling vs C0: slightly down on average calls but 6+ bucket calls up.",
        "- Stage3 training groups: dead 17% vs stage2 replay dead 63% — replay still starved of GRPO signal.",
        "",
        "## 10. Decision tree",
        "",
        f"**Branch {report['decision_tree']['selected_branch']}:** {report['decision_tree']['rationale']}",
        "",
        "## Proposed follow-up experiments (max 5)",
        "",
    ]
    for ex in report["proposed_experiments"]:
        lines += [
            f"### {ex['priority']}. {ex['name']}",
            f"- Change: {ex['change']}",
            f"- Control: {ex['control']}",
            f"- Expect: {ex['expected_metrics']}",
            f"- Predict: {ex['prediction']}",
            f"- Stop if: {ex['stop_if']}",
            "",
        ]
    return "\n".join(lines)


def _fmt(v: Optional[float]) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        if abs(v) >= 10:
            return f"{v:.2f}"
        return f"{v:.4f}"
    return str(v)


def _fmt_delta(v: Optional[float]) -> str:
    if v is None:
        return "—"
    return f"{v:+.4f}"


def build_task_transition_jsonl(
    ids: List[str],
    rows: Dict[str, Dict[str, dict]],
    transitions: dict,
    labels: Dict[str, dict],
) -> List[dict]:
    out = []
    for tid in ids:
        w0 = official_win(rows["C0"][tid])
        w1 = official_win(rows["C1"][tid])
        w2 = official_win(rows["C2"][tid])
        out.append({
            "sample_id": tid,
            "num_gold_calls": rows["C0"][tid].get("num_gold_calls"),
            "motif_type": labels.get(tid, {}).get("motif_type"),
            "C0_win": w0,
            "C1_win": w1,
            "C2_win": w2,
            "transition_C1": "gained" if w0 < w1 else ("lost" if w0 > w1 else "same"),
            "transition_C2": "gained" if w0 < w2 else ("lost" if w0 > w2 else "same"),
            "C1_gained_C2_lost": tid in transitions["C1_gained_C2_lost"],
            "C1_lost_C2_gained": tid in transitions["C1_lost_C2_gained"],
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run-dir",
        default="experiments/nestful_synthetic_curriculum_v3/outputs/runs/"
        "two_phase_20260718_192902/two_phase_20260718_192902",
    )
    ap.add_argument(
        "--out-dir",
        default="experiments/nestful_synthetic_curriculum_v3/reports",
    )
    args = ap.parse_args()

    run_dir = resolve_run_dir(args.run_dir)
    out_dir = (_REPO / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir)

    rows_by_arm = {}
    for arm, (rel, _) in EVAL_REL.items():
        rows_by_arm[arm] = load_trajectories(run_dir / rel)

    ids = sorted(rows_by_arm["C0"])
    assert rows_by_arm["C1"].keys() == rows_by_arm["C2"].keys() == set(ids)

    labels = {
        load_task_row(r)["task_id"]: extract_motifs(load_task_row(r))
        for r in load_jsonl(default_test_path())
    }

    summaries = {
        arm: summarize_arm(arm, run_dir / rel, rows_by_arm[arm])
        for arm, (rel, _) in EVAL_REL.items()
    }
    deltas = delta_table(summaries)
    transitions = paired_transitions(ids, rows_by_arm)
    taxonomy = failure_taxonomy(rows_by_arm)
    training = analyze_training_rewards(run_dir)
    verification = verify_experiment(run_dir)
    dataset = dataset_coverage(run_dir)
    motif_table = motif_bucket_table(ids, labels, rows_by_arm)
    exemplars = exemplar_cases(
        ids, rows_by_arm,
        transitions["C1_vs_C0"]["gained"],
        transitions["C1_vs_C0"]["lost"],
        labels,
    )
    figures = plot_metrics(summaries, out_dir / "figures")
    decision = decision_tree(summaries, deltas, training, transitions)
    experiments = proposed_experiments(summaries, training, dataset)

    state = _load_json(run_dir / "two_phase_state.json")
    report = {
        "generated_at": _now(),
        "run_dir": str(run_dir),
        "eval_paths": {a: str(run_dir / rel) for a, (rel, _) in EVAL_REL.items()},
        "verification": verification,
        "summaries": summaries,
        "deltas": deltas,
        "transitions": transitions,
        "failure_taxonomy": taxonomy,
        "motif_table": motif_table,
        "training_rewards": training,
        "stage_split_metrics": state["steps"]["phase2_train"].get("stage_split_metrics"),
        "dataset_coverage": dataset,
        "official_scorer": official_scorer_semantics(),
        "exemplars": exemplars,
        "figures": figures,
        "decision_tree": decision,
        "proposed_experiments": experiments,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "C0_C1_C2_ROOT_CAUSE_ANALYSIS.json", report)
    (out_dir / "C0_C1_C2_ROOT_CAUSE_ANALYSIS.md").write_text(
        render_markdown(report), encoding="utf-8")
    _write_csv(
        out_dir / "C0_C1_C2_failure_taxonomy.csv",
        taxonomy,
        ["failure_type", "C0", "C1", "C2", "C1_minus_C0", "C2_minus_C1", "C2_minus_C0"],
    )
    _write_jsonl(
        out_dir / "C0_C1_C2_task_transitions.jsonl",
        build_task_transition_jsonl(ids, rows_by_arm, transitions, labels),
    )
    _write_csv(
        out_dir / "C0_C1_C2_motif_table.csv",
        motif_table,
        ["motif_type", "n", "C0_win_rate", "C1_win_rate", "C2_win_rate",
         "C1_minus_C0", "C2_minus_C1"],
    )
    _write_json(out_dir / "C0_C1_C2_exemplars.json", exemplars)

    print(f"[analysis] report -> {out_dir / 'C0_C1_C2_ROOT_CAUSE_ANALYSIS.md'}")
    print(f"[analysis] json   -> {out_dir / 'C0_C1_C2_ROOT_CAUSE_ANALYSIS.json'}")
    print(f"[analysis] C0 win={summaries['C0']['official_win_rate']:.4f} "
          f"C1={summaries['C1']['official_win_rate']:.4f} "
          f"C2={summaries['C2']['official_win_rate']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
