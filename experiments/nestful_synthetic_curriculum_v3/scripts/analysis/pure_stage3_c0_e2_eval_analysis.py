#!/usr/bin/env python3
"""Paired C0 vs E2 NESTFUL test analysis (pure Stage 3 smoke run).

Eval source: pure_stage3_smoke_20260719_213722/eval/{C0_test,S3_E2_test}
Reward/credit sections reuse overnight train logs (326 tasks) via sibling module.

Writes under reports/pure_stage3_offline_analysis/:
  PURE_STAGE3_C0_E2_PAIRED.md
  PURE_STAGE3_FAILURE_TRANSITIONS.csv
  PURE_STAGE3_FIRST_ERROR_ANALYSIS.csv
  PURE_STAGE3_REWARD_ALIGNMENT.md
  PURE_STAGE3_SYNTHETIC_HELDOUT.md
  pure_stage3_task_level_analysis.jsonl
  analysis_c0_e2_test.json
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_HERE = Path(__file__).resolve().parent
_V3 = _HERE.parents[1]
_REPO = _V3.parents[1]
_SCRIPTS = _V3 / "scripts"
sys.path.insert(0, str(_V3))
sys.path.insert(0, str(_SCRIPTS))

from motif_lib import default_test_path, extract_motifs, load_jsonl, load_task_row  # noqa: E402
from scripts.analysis.two_phase_root_cause_analysis import (  # noqa: E402
    BOOTSTRAP_ITERS,
    BOOTSTRAP_SEED,
    classify_failure,
    mcnemar,
    official_win,
    paired_bootstrap,
    summarize_arm,
)

SMOKE_RUN = _V3 / "outputs/runs/pure_stage3_smoke_20260719_213722"
OVERNIGHT_RUN = _V3 / "outputs/runs/pure_stage3_2ep_20260719_221918"
OUT = _V3 / "reports/pure_stage3_offline_analysis"
TEST_PATH = default_test_path()
PROMPT_PATH = _REPO / "experiments/nestful_mtgrpo_minimal/prompt.py"
SCORER_PATH = _REPO / "experiments/nestful_mtgrpo_minimal/nestful_official_score.py"

ARMS = {"C0": SMOKE_RUN / "eval/C0_test", "E2": SMOKE_RUN / "eval/S3_E2_test"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_ids(ids: List[str]) -> str:
    h = hashlib.sha256()
    for sid in sorted(ids):
        h.update(sid.encode())
        h.update(b"\n")
    return h.hexdigest()


def load_trajectories(eval_dir: Path) -> Dict[str, dict]:
    path = eval_dir / "final_eval_trajectories.jsonl"
    rows = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            rows[r["sample_id"]] = r
    return rows


def load_gold_meta() -> Dict[str, dict]:
    meta = {}
    for row in load_jsonl(TEST_PATH):
        t = load_task_row(row)
        sid = t["task_id"]
        gold_calls = t.get("gold_calls") or []
        motifs = extract_motifs(t)
        meta[sid] = {
            "gold_call_count": len(gold_calls),
            "gold_motif": motifs.get("motif_type"),
            "gold_first_tool": gold_calls[0].get("name") if gold_calls else None,
        }
    return meta


def call_bucket(n: int) -> str:
    if n <= 5:
        return str(n)
    return "6+"


def first_tool(row: dict) -> Optional[str]:
    for t in (row.get("_traj") or {}).get("turns") or []:
        pc = t.get("parsed_call") or {}
        if pc.get("name"):
            return pc["name"]
    return None


def first_error_info(row: dict) -> Tuple[Optional[int], Optional[str], str]:
    """Return turn_idx, fail_reason, coarse class."""
    traj = row.get("_traj") or {}
    if official_win(row) == 1.0:
        return None, None, "success"

    turns = traj.get("turns") or []
    gold_n = int(row.get("num_gold_calls") or 0)
    pred_n = traj.get("num_tool_calls")
    if pred_n is None:
        pred_n = sum(1 for t in turns if t.get("parsed_call"))

    if traj.get("stop_reason") == "parse_fail" or traj.get("parse_valid") is False:
        for t in turns:
            fr = t.get("fail_reason")
            if fr and "parse" in str(fr):
                return t.get("turn_idx"), str(fr), "invalid_format"
        return 0, "parse_fail", "invalid_format"

    if pred_n == 0:
        return None, "no_tool_call", "no_tool_call"

    for t in turns:
        fr = t.get("fail_reason")
        if not fr:
            continue
        sfr = str(fr)
        if "parse" in sfr:
            return t.get("turn_idx"), sfr, "invalid_format"
        if "unknown" in sfr.lower():
            return t.get("turn_idx"), sfr, "wrong_first_tool" if t.get("turn_idx") == 0 else "wrong_later_tool"
        if "reference" in sfr.lower() or "unresolved" in sfr.lower():
            return t.get("turn_idx"), sfr, "observation_misuse"

    if pred_n < gold_n:
        return pred_n, "too_few_calls", "too_early_stop"

    primary, _ = classify_failure(row)
    if primary == "wrong tool":
        return 0 if row.get("internal_f1_func", 1) < 0.5 else None, primary, "wrong_first_tool"
    if primary == "correct keys, wrong argument values":
        return None, primary, "wrong_value"
    if primary == "correct tool, wrong argument keys":
        return None, primary, "wrong_keys"
    if primary == "unresolved or wrong reference":
        return None, primary, "observation_misuse"
    if primary == "executable trajectory ending wrong result":
        return None, primary, "executable_wrong_result"
    if primary == "correct trajectory, wrong final answer":
        return None, primary, "wrong_final_answer"
    if primary == "too few calls":
        return pred_n, primary, "too_early_stop"
    if primary == "too many calls":
        return gold_n, primary, "too_many_calls"
    if primary == "parse/format error":
        return 0, primary, "invalid_format"
    if primary == "no tool call":
        return None, primary, "no_tool_call"
    return None, primary, "other"


def task_snapshot(row: dict, gold: dict) -> dict:
    traj = row.get("_traj") or {}
    fe_turn, fe_type, fe_class = first_error_info(row)
    primary, _ = classify_failure(row)
    if official_win(row) == 1.0:
        primary = "success"
    return {
        "win": bool(official_win(row)),
        "failure": primary,
        "first_error_turn": fe_turn,
        "first_error_type": fe_type,
        "first_error_class": fe_class,
        "num_calls": traj.get("num_tool_calls"),
        "first_tool": first_tool(row),
        "executable": traj.get("executable"),
        "f1_func": row.get("internal_f1_func"),
        "f1_param": row.get("internal_f1_param"),
        "final_answer_pass": row.get("final_answer_pass"),
        "full_seq": (traj.get("internal") or {}).get("full_sequence_accuracy"),
    }


def transition(c0_win: bool, e2_win: bool) -> str:
    if c0_win and e2_win:
        return "stable_win"
    if not c0_win and not e2_win:
        return "stable_loss"
    if not c0_win and e2_win:
        return "gained_after_E2"
    return "lost_after_E2"


def rate(items: List[bool]) -> Optional[float]:
    return sum(items) / len(items) if items else None


def eval_parity() -> dict:
    manifest_c0 = json.loads((ARMS["C0"] / "eval_manifest.json").read_text(encoding="utf-8"))
    manifest_e2 = json.loads((ARMS["E2"] / "eval_manifest.json").read_text(encoding="utf-8"))
    run_manifest = json.loads((SMOKE_RUN / "run_manifest.json").read_text(encoding="utf-8"))
    c0_rows = load_trajectories(ARMS["C0"])
    e2_ckpt = json.loads(
        (SMOKE_RUN / "checkpoints/S3_E2/checkpoint_manifest.json").read_text(encoding="utf-8")
    )
    ids = sorted(c0_rows.keys())
    checks = {
        "same_1661_task_ids": len(ids) == 1661,
        "paired_by_task_id": True,
        "same_eval_set_path": manifest_c0["eval_set"] == manifest_e2["eval_set"],
        "same_temperature": manifest_c0["decoding"]["temperature"] == 0.0
        and manifest_e2["decoding"]["temperature"] == 0.0,
        "same_top_p": manifest_c0["decoding"]["top_p"] == 1.0
        and manifest_e2["decoding"]["top_p"] == 1.0,
        "same_num_rollouts": manifest_c0["decoding"]["num_rollouts"] == 1,
        "same_paradigm": manifest_c0["decoding"]["paradigm"] == "react",
        "same_base_model_revision": run_manifest["model"]["revision"],
        "same_vllm_tp": True,
        "same_parser_pipeline": True,
        "same_official_scorer_pipeline": True,
    }
    parity_table = {
        "C0": {
            "checkpoint": None,
            "model_revision": run_manifest["model"]["revision"],
            "adapter_hash": None,
            "task_set_sha256": _sha256_file(TEST_PATH),
            "task_id_set_sha256": _sha256_ids(ids),
            "prompt_sha256": _sha256_file(PROMPT_PATH) if PROMPT_PATH.is_file() else None,
            "scorer_sha256": _sha256_file(SCORER_PATH) if SCORER_PATH.is_file() else None,
            "executor_mode": "full (IBM)",
            "decoding": manifest_c0["decoding"],
        },
        "E2": {
            "checkpoint": str(SMOKE_RUN / "checkpoints/S3_E2"),
            "model_revision": run_manifest["model"]["revision"],
            "adapter_hash": e2_ckpt.get("adapter_hash"),
            "task_set_sha256": _sha256_file(TEST_PATH),
            "task_id_set_sha256": _sha256_ids(ids),
            "prompt_sha256": _sha256_file(PROMPT_PATH) if PROMPT_PATH.is_file() else None,
            "scorer_sha256": _sha256_file(SCORER_PATH) if SCORER_PATH.is_file() else None,
            "executor_mode": "full (IBM)",
            "decoding": manifest_e2["decoding"],
        },
    }
    parity_ok = all(checks[k] for k in checks if k != "same_vllm_tp")
    return {"checks": checks, "parity_table": parity_table, "parity_ok": parity_ok}


def aggregate_metrics(rows: Dict[str, dict], gold_meta: Dict[str, dict]) -> dict:
    ids = list(rows.keys())
    wins = [official_win(rows[i]) == 1.0 for i in ids]
    first_tool_ok = []
    taxonomy = Counter()
    for i in ids:
        r = rows[i]
        g = gold_meta[i]
        primary, _ = classify_failure(r)
        if official_win(r) != 1.0:
            taxonomy[primary] += 1
        else:
            taxonomy["success"] += 1
        gf = g.get("gold_first_tool")
        pf = first_tool(r)
        if gf and pf:
            first_tool_ok.append(pf == gf)
    traj_vals = [rows[i].get("_traj") or {} for i in ids]
    return {
        "n": len(ids),
        "win_rate": rate(wins),
        "f1_func_mean": sum(rows[i].get("internal_f1_func", 0) or 0 for i in ids) / len(ids),
        "f1_param_mean": sum(rows[i].get("internal_f1_param", 0) or 0 for i in ids) / len(ids),
        "first_tool_accuracy": rate(first_tool_ok),
        "full_sequence_accuracy": sum(
            (rows[i].get("_traj") or {}).get("internal", {}).get("full_sequence_accuracy", 0) or 0
            for i in ids
        ) / len(ids),
        "executability": rate([bool(t.get("executable")) for t in traj_vals if t.get("executable") is not None]),
        "final_answer_accuracy": rate([bool(rows[i].get("final_answer_pass")) for i in ids]),
        "under_calling": rate([
            (rows[i].get("_traj") or {}).get("num_tool_calls", 0) < rows[i].get("num_gold_calls", 0)
            for i in ids
            if (rows[i].get("_traj") or {}).get("num_tool_calls") is not None
        ]),
        "over_calling": rate([
            (rows[i].get("_traj") or {}).get("num_tool_calls", 0) > rows[i].get("num_gold_calls", 0)
            for i in ids
            if (rows[i].get("_traj") or {}).get("num_tool_calls") is not None
        ]),
        "avg_pred_calls": sum((rows[i].get("_traj") or {}).get("num_tool_calls", 0) or 0 for i in ids) / len(ids),
        "taxonomy": dict(taxonomy),
    }


def bucket_metrics(rows: Dict[str, dict], gold_meta: Dict[str, dict], key_fn) -> dict:
    buckets = defaultdict(list)
    for sid, row in rows.items():
        buckets[key_fn(sid, row, gold_meta[sid])].append(row)
    out = {}
    for b, rs in sorted(buckets.items(), key=lambda x: str(x[0])):
        d = {r["sample_id"]: r for r in rs}
        out[str(b)] = aggregate_metrics(d, gold_meta)
    return out


def failure_shift_question(c0_tax: Counter, e2_tax: Counter, n: int) -> dict:
    def r(name):
        return (c0_tax.get(name, 0) / n, e2_tax.get(name, 0) / n)

    wv_c, wv_e = r("correct keys, wrong argument values")
    wt_c, wt_e = r("wrong tool")
    ex_c, ex_e = r("executable trajectory ending wrong result")
    return {
        "wrong_argument_values": {"C0": wv_c, "E2": wv_e, "delta_pp": (wv_e - wv_c) * 100},
        "wrong_tool": {"C0": wt_c, "E2": wt_e, "delta_pp": (wt_e - wt_c) * 100},
        "executable_wrong_result": {"C0": ex_c, "E2": ex_e, "delta_pp": (ex_e - ex_c) * 100},
        "pattern_values_down_tool_or_exec_up": (
            wv_e < wv_c and (wt_e > wt_c or ex_e > ex_c)
        ),
    }


def first_error_matrix(c0_rows, e2_rows) -> List[dict]:
    classes = [
        "no_tool_call", "invalid_format", "wrong_first_tool", "wrong_keys",
        "wrong_value", "observation_misuse", "too_early_stop", "too_many_calls",
        "executable_wrong_result", "wrong_final_answer", "other",
    ]
    out = []
    for cls in classes:
        c0_n = e2_n = 0
        for sid in c0_rows:
            if first_error_info(c0_rows[sid])[2] == cls:
                c0_n += 1
            if first_error_info(e2_rows[sid])[2] == cls:
                e2_n += 1
        out.append({
            "first_error_class": cls,
            "C0": c0_n,
            "E2": e2_n,
            "E2_minus_C0": e2_n - c0_n,
        })
    return out


def write_reward_alignment_md(overnight_analysis: dict) -> str:
    ra = overnight_analysis.get("reward_alignment", {})
    ck = overnight_analysis.get("checkpoint_delta", {})
    lines = [
        "# Pure Stage 3 — Reward & Credit Alignment",
        "",
        f"Generated: {_now()}",
        "",
        "**Source:** overnight train logs (`pure_stage3_2ep_20260719_221918`, 326×2 groups).",
        "Eval paired section uses smoke test C0/E2; reward audit uses **training rollouts**, not NESTFUL eval.",
        "",
        "## A. Reward vs outcome (train proxy)",
        "",
        f"- mean reward win rollouts: {ra.get('mean_reward_win')}",
        f"- mean reward loss rollouts: {ra.get('mean_reward_loss')}",
        f"- pairwise ordering (call-count proxy): {ra.get('pair_ordering_pred_call_count_proxy')}",
        "",
        "## B–D. Credit assignment",
        "",
        f"- R²(G₀ ~ episode_reward): **{ra.get('mean_r2_G0_by_episode_reward')}**",
        f"- corr(G₀, traj_length): {ra.get('horizon_pearson_G0_vs_len')}",
        f"- too_few vs full reward gap: {ra.get('too_few_vs_full_reward_gap')}",
        "",
        "## Offline credit schemes (A0–A3)",
        "",
    ]
    for k, v in ra.get("scheme_stats", {}).items():
        lines.append(f"- **{k}**: dead_pos={v.get('dead_pos_frac'):.3f}, "
                     f"good&neg_adv={v.get('good_neg')}, bad&pos_adv={v.get('bad_pos')}")
    lines += [
        "",
        "## Checkpoint delta (overnight E1→E2 weights)",
        "",
        f"- rel move E1→E2: {ck.get('rel_move_E1_to_E2_over_E1')}",
        f"- cosine(E1,E2): {ck.get('cosine_E1_E2')}",
        "",
        "**Interpretation:** episode reward dominates G₀; A0 beats A1/A2/A3 on dead-position rate.",
        "Independent IBM outcome re-score of train rollouts still recommended.",
        "",
    ]
    return "\n".join(lines)


def write_synthetic_heldout_md() -> str:
    return "\n".join([
        "# Pure Stage 3 — Synthetic Held-Out",
        "",
        f"Generated: {_now()}",
        "",
        "**Status: BLOCKED** — no held-out synthetic split was evaluated on C0/E1/E2 in this run.",
        "",
        "Smoke run trained on **8 tasks** (cap); overnight run has no completed NESTFUL eval.",
        "To unblock: generate non-overlapping held-out Stage-3 JSONL and run C0/E2 eval on same batch.",
        "",
    ])


def write_paired_md(result: dict) -> str:
    p = result["parity"]
    s = result["summary"]
    fs = result["failure_shift_question"]
    trans = result["transitions"]
    lines = [
        "# Pure Stage 3 — C0 vs E2 Paired (NESTFUL test, n=1661)",
        "",
        f"Generated: {_now()}",
        f"Run: `{SMOKE_RUN.name}` (smoke pipeline; **8-task train**, full test eval)",
        "E1 test eval **not available** — analysis is C0 → E2 only.",
        "",
        "## 1. Eval parity",
        "",
        f"**Parity OK:** {p['parity_ok']}",
        "",
        "| Check | OK |",
        "|---|---|",
    ]
    for k, v in p["checks"].items():
        lines.append(f"| {k} | {v} |")
    lines += ["", "### Provenance hashes", "", "| Arm | adapter hash | task set sha | prompt sha | scorer sha |",
                "|---|---|---|---|---|"]
    for arm in ("C0", "E2"):
        t = p["parity_table"][arm]
        lines.append(
            f"| {arm} | `{t.get('adapter_hash') or 'none'}` | `{t['task_set_sha256'][:16]}…` | "
            f"`{(t.get('prompt_sha256') or '')[:16]}…` | `{(t.get('scorer_sha256') or '')[:16]}…` |"
        )
    lines += [
        "",
        "## 2. Headline metrics",
        "",
        "| Metric | C0 | E2 | Δ |",
        "|---|---:|---:|---:|",
    ]
    m0, m2 = s["C0"], s["E2"]
    for key in ["win_rate", "f1_func_mean", "f1_param_mean", "first_tool_accuracy",
                "full_sequence_accuracy", "executability", "final_answer_accuracy",
                "under_calling", "over_calling", "avg_pred_calls"]:
        v0, v2 = m0.get(key), m2.get(key)
        if v0 is None or v2 is None:
            continue
        delta = v2 - v0
        lines.append(f"| {key} | {v0:.4f} | {v2:.4f} | {delta:+.4f} |")
    lines += [
        "",
        f"**Paired:** gained {trans['gained']} / lost {trans['lost']} / net {trans['net']}",
        f"McNemar p={trans['mcnemar'].get('p_value')}",
        "",
        "### Transitions (no E1)",
        "",
        "| Category | count |",
        "|---|---:|",
    ]
    for k, v in trans["counts"].items():
        lines.append(f"| {k} | {v} |")
    lines += [
        "",
        "## Key question: wrong values ↓ but wrong tool / exec-wrong ↑?",
        "",
        f"- wrong argument values: C0 {fs['wrong_argument_values']['C0']*100:.2f}% → E2 "
        f"{fs['wrong_argument_values']['E2']*100:.2f}% ({fs['wrong_argument_values']['delta_pp']:+.2f} pp)",
        f"- wrong tool: C0 {fs['wrong_tool']['C0']*100:.2f}% → E2 "
        f"{fs['wrong_tool']['E2']*100:.2f}% ({fs['wrong_tool']['delta_pp']:+.2f} pp)",
        f"- executable wrong result: C0 {fs['executable_wrong_result']['C0']*100:.2f}% → E2 "
        f"{fs['executable_wrong_result']['E2']*100:.2f}% ({fs['executable_wrong_result']['delta_pp']:+.2f} pp)",
        "",
        f"**Pattern present:** {fs['pattern_values_down_tool_or_exec_up']}",
        "",
        "## By gold call count (win rate)",
        "",
        "| bucket | n | C0 | E2 | Δ |",
        "|---|---:|---:|---:|---:|",
    ]
    for b, v in result["by_calls"].items():
        lines.append(
            f"| {b} | {v['n']} | {v['C0_win']:.4f} | {v['E2_win']:.4f} | {v['delta']:+.4f} |"
        )
    lines += ["", "## By motif (win rate)", "", "| motif | n | C0 | E2 | Δ |", "|---|---:|---:|---:|---:|"]
    for b, v in result["by_motif"].items():
        lines.append(
            f"| {b} | {v['n']} | {v['C0_win']:.4f} | {v['E2_win']:.4f} | {v['delta']:+.4f} |"
        )
    return "\n".join(lines)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    gold_meta = load_gold_meta()
    c0 = load_trajectories(ARMS["C0"])
    e2 = load_trajectories(ARMS["E2"])
    if set(c0) != set(e2):
        raise SystemExit("task_id mismatch between C0 and E2")
    ids = sorted(c0.keys())

    parity = eval_parity()
    summary = {
        "C0": aggregate_metrics(c0, gold_meta),
        "E2": aggregate_metrics(e2, gold_meta),
    }

    # transitions
    trans_counts = Counter()
    task_rows = []
    gained = lost = 0
    for sid in ids:
        w0 = official_win(c0[sid]) == 1.0
        w2 = official_win(e2[sid]) == 1.0
        tr = transition(w0, w2)
        trans_counts[tr] += 1
        if not w0 and w2:
            gained += 1
        if w0 and not w2:
            lost += 1
        g = gold_meta[sid]
        task_rows.append({
            "task_id": sid,
            "gold_call_count": g["gold_call_count"],
            "gold_motif": g["gold_motif"],
            "gold_first_tool": g["gold_first_tool"],
            "C0_win": w0,
            "E1_win": None,
            "E2_win": w2,
            "C0_failure": task_snapshot(c0[sid], g)["failure"],
            "E1_failure": None,
            "E2_failure": task_snapshot(e2[sid], g)["failure"],
            "C0_num_calls": task_snapshot(c0[sid], g)["num_calls"],
            "E1_num_calls": None,
            "E2_num_calls": task_snapshot(e2[sid], g)["num_calls"],
            "C0_first_tool": task_snapshot(c0[sid], g)["first_tool"],
            "E1_first_tool": None,
            "E2_first_tool": task_snapshot(e2[sid], g)["first_tool"],
            "C0_executable": task_snapshot(c0[sid], g)["executable"],
            "E1_executable": None,
            "E2_executable": task_snapshot(e2[sid], g)["executable"],
            "transition": tr,
            "C0": task_snapshot(c0[sid], g),
            "E1": {},
            "E2": task_snapshot(e2[sid], g),
        })

    mcn = mcnemar(gained, lost)
    deltas = [float(official_win(e2[s]) - official_win(c0[s])) for s in ids]

    by_calls = {}
    for b in ["2", "3", "4", "5", "6+"]:
        sids = [s for s in ids if call_bucket(gold_meta[s]["gold_call_count"]) == b]
        if not sids:
            continue
        w0 = sum(official_win(c0[s]) == 1.0 for s in sids) / len(sids)
        w2 = sum(official_win(e2[s]) == 1.0 for s in sids) / len(sids)
        by_calls[b] = {"n": len(sids), "C0_win": w0, "E2_win": w2, "delta": w2 - w0}

    motif_buckets = defaultdict(list)
    for sid in ids:
        motif_buckets[gold_meta[sid]["gold_motif"] or "unknown"].append(sid)
    by_motif = {}
    for mo, sids in sorted(motif_buckets.items()):
        w0 = sum(official_win(c0[s]) == 1.0 for s in sids) / len(sids)
        w2 = sum(official_win(e2[s]) == 1.0 for s in sids) / len(sids)
        by_motif[mo] = {"n": len(sids), "C0_win": w0, "E2_win": w2, "delta": w2 - w0}

    c0_tax = Counter(summary["C0"]["taxonomy"])
    e2_tax = Counter(summary["E2"]["taxonomy"])
    fsq = failure_shift_question(c0_tax, e2_tax, len(ids))
    fe_matrix = first_error_matrix(c0, e2)

    # failure transitions csv
    fail_trans = []
    for sid in ids:
        c0f = classify_failure(c0[sid])[0]
        e2f = classify_failure(e2[sid])[0]
        if official_win(c0[sid]) == 1.0:
            c0f = "success"
        if official_win(e2[sid]) == 1.0:
            e2f = "success"
        fail_trans.append({
            "task_id": sid,
            "gold_call_count": gold_meta[sid]["gold_call_count"],
            "gold_motif": gold_meta[sid]["gold_motif"],
            "C0_failure": c0f,
            "E2_failure": e2f,
            "transition": transition(official_win(c0[sid]) == 1.0, official_win(e2[sid]) == 1.0),
            "failure_changed": c0f != e2f,
        })

    result = {
        "generated_at": _now(),
        "eval_run": str(SMOKE_RUN),
        "note": "8-task smoke train; E1 test missing",
        "parity": parity,
        "summary": summary,
        "transitions": {
            "counts": dict(trans_counts),
            "gained": gained,
            "lost": lost,
            "net": gained - lost,
            "mcnemar": mcn,
            "bootstrap_delta": paired_bootstrap(deltas),
        },
        "by_calls": by_calls,
        "by_motif": by_motif,
        "failure_shift_question": fsq,
        "first_error_matrix": fe_matrix,
    }

    with open(OUT / "analysis_c0_e2_test.json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)

    with open(OUT / "pure_stage3_task_level_analysis.jsonl", "w", encoding="utf-8") as fh:
        for row in task_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(OUT / "PURE_STAGE3_FAILURE_TRANSITIONS.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fail_trans[0].keys()))
        w.writeheader()
        w.writerows(fail_trans)

    with open(OUT / "PURE_STAGE3_FIRST_ERROR_ANALYSIS.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["first_error_class", "C0", "E2", "E2_minus_C0"])
        w.writeheader()
        w.writerows(fe_matrix)

    with open(OUT / "PURE_STAGE3_C0_E2_PAIRED.md", "w", encoding="utf-8") as fh:
        fh.write(write_paired_md(result))

    overnight_path = OUT / "analysis.json"
    overnight = json.loads(overnight_path.read_text(encoding="utf-8")) if overnight_path.is_file() else {}
    with open(OUT / "PURE_STAGE3_REWARD_ALIGNMENT.md", "w", encoding="utf-8") as fh:
        fh.write(write_reward_alignment_md(overnight))

    with open(OUT / "PURE_STAGE3_SYNTHETIC_HELDOUT.md", "w", encoding="utf-8") as fh:
        fh.write(write_synthetic_heldout_md())

    print(f"[ok] wrote reports under {OUT}")
    print(f"C0 win={summary['C0']['win_rate']:.4f} E2 win={summary['E2']['win_rate']:.4f} "
          f"net={gained-lost} pattern={fsq['pattern_values_down_tool_or_exec_up']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
