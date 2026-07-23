#!/usr/bin/env python3
"""Build compact diagnostic case pack + offline reward variant audit.

Outputs under reports/pure_stage3_diagnostic_pack/:
  diagnostic_cases.jsonl          ~200-250 compact cases
  reward_variant_metrics.json     aggregate R0-R3 comparison
  reward_variant_per_task.jsonl   per-task pseudo-group ranks/adv
  turn2_tool_table.jsonl          turn-2 analysis rows
  turn2_confusion_counts.json     script-computed counts (for LLM clusters)
  annotation_prompt.txt           analyst prompt template
  annotation_inputs/              named + anonymized JSONL per case
  cluster_input_template.json     pre-aggregated counts for weak-model clustering
  README.md
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_HERE = Path(__file__).resolve().parent
_V3 = _HERE.parents[1]
_REPO = _V3.parents[1]
_MINIMAL = _REPO / "experiments/nestful_mtgrpo_minimal"
sys.path.insert(0, str(_MINIMAL))
sys.path.insert(0, str(_V3))
sys.path.append(str(_V3 / "scripts"))

from group_stats import compute_group_stats  # noqa: E402
from grpo_train import _turn_returns  # noqa: E402
from lib.reward_variants_offline import (  # noqa: E402
    DEFAULT_EPS_R2,
    DEFAULT_EPS_R3,
    score_variants,
    variant_to_dict,
)
from motif_lib import default_test_path, extract_motifs, load_task_row  # noqa: E402
from scripts.analysis.pure_stage3_diag_utils import (  # noqa: E402
    arm_snapshot,
    compact_value,
    first_divergence_turn,
    load_tasks,
    load_traj_rows,
    observation_shape,
    predicted_calls,
    relevant_tools,
    reward_mismatch_c0_e2,
    stratified_sample,
    tool_at,
    tool_description,
    traj_from_dict,
)
from scripts.analysis.two_phase_root_cause_analysis import classify_failure, official_win  # noqa: E402

DEFAULT_RUN = _V3 / "outputs/runs/pure_stage3_2ep_20260719_221918"
OUT = _V3 / "reports/pure_stage3_diagnostic_pack"
ARM_DIRS = {"C0": "C0_test", "E1": "S3_E1_test", "E2": "S3_E2_test"}
GAMMA = 1.0
LAMBDA_EP = 1.0
SEED = 20260723

COHORT_TARGETS = {
    "C0_win_E2_loss": None,  # all
    "C0_loss_E2_win": 35,
    "official_win_reward_too_few": 55,
    "E2_executable_wrong_other": 35,
    "stable_win_control": 20,
    "stable_loss_control": 20,
}

ANALYST_PROMPT = """Jsi analytik multi-turn tool-use experimentu.

Dostaneš jeden diagnostický případ obsahující:
- zadání;
- relevantní tool schemas;
- C0/E1/E2 trajectories;
- skutečné observations;
- official outcome;
- reward komponenty;
- deterministicky vypočítané flags.

Official scorer a executor jsou autorita. Nepřehodnocuj jejich výsledek
bez konkrétního důkazu chyby v artefaktech.

Úkol:

1. Najdi první sémanticky významnou divergenci C0 a E2.
2. Urči, zda E2 selhává na:
   - initial_tool_selection;
   - later_tool_selection;
   - argument_keys;
   - argument_values;
   - observation_ignored;
   - wrong_output_field;
   - invalid_state_transition;
   - premature_stop;
   - valid_shorter_path;
   - executable_wrong_global_plan;
   - wrong_final_answer;
   - reward_mismatch;
   - unclear.
3. Rozhodni, zda je kratší cesta:
   - validní;
   - nevalidní;
   - nerozhodnutelná.
4. Posuď, zda současný reward správně řadí C0 a E2.
5. Urči reward komponentu, která případný mismatch způsobila.
6. Navrhni nejmenší opravu:
   - outcome_reward;
   - process_reward;
   - tool_selection_data;
   - observation_grounding_data;
   - targeted_SFT;
   - credit_assignment;
   - evaluator;
   - no_change.
7. Uveď pouze krátký důkaz založený na konkrétním callu nebo observation.
8. Nepiš dlouhé reasoning vysvětlení.

Vrať pouze JSON:

{
  "task_id": "",
  "first_divergence_turn": null,
  "root_cause": "",
  "shorter_path_verdict": "",
  "observation_used_correctly": null,
  "reward_ordering_correct": null,
  "responsible_reward_component": "",
  "recommended_fix": "",
  "confidence": 0.0,
  "evidence": ""
}
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)


def _write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _gold_bucket(n: int) -> str:
    return str(n) if n <= 5 else "6+"


def build_task_meta(
    ids: List[str],
    arms: Dict[str, Dict[str, dict]],
    tasks: Dict[str, dict],
    r0_scores: Dict[str, Dict[str, float]],
) -> Dict[str, dict]:
    meta: Dict[str, dict] = {}
    for sid in ids:
        c0, e1, e2 = arms["C0"][sid], arms["E1"][sid], arms["E2"][sid]
        task = tasks[sid]
        w0 = official_win(c0) == 1.0
        w1 = official_win(e1) == 1.0
        w2 = official_win(e2) == 1.0
        motif = extract_motifs(task).get("motif_type")
        gold_n = len(task.get("gold_calls") or [])
        div = first_divergence_turn(predicted_calls(c0), predicted_calls(e2))
        r0c, r0e = r0_scores[sid]["C0"], r0_scores[sid]["E2"]
        meta[sid] = {
            "w0": w0, "w1": w1, "w2": w2,
            "gold_call_bucket": _gold_bucket(gold_n),
            "motif": motif,
            "first_divergence_turn": div,
            "failure_type": classify_failure(e2)[0],
            "reward_mismatch": reward_mismatch_c0_e2(r0c, r0e, w0, w2),
            "r0_class_e2": r0_scores[sid].get("class_E2"),
            "executable_wrong_e2": classify_failure(e2)[0] == "executable trajectory ending wrong result",
        }
    return meta


def assign_cohorts(meta: Dict[str, dict], r0_scores: Dict[str, dict]) -> Dict[str, str]:
    assigned: Dict[str, str] = {}
    pools: Dict[str, List[str]] = defaultdict(list)

    for sid, m in meta.items():
        w0, w2 = m["w0"], m["w2"]
        if w0 and not w2:
            pools["C0_win_E2_loss"].append(sid)
        elif not w0 and w2:
            pools["C0_loss_E2_win"].append(sid)
        elif w0 and w2:
            pools["stable_win_control"].append(sid)
        elif not w0 and not w2:
            pools["stable_loss_control"].append(sid)
        if w0 and r0_scores[sid].get("class_C0") == "too_few_calls":
            pools["official_win_reward_too_few"].append(sid)
        if m["executable_wrong_e2"] and not (w0 and not w2):
            pools["E2_executable_wrong_other"].append(sid)

    for sid in pools["C0_win_E2_loss"]:
        assigned[sid] = "C0_win_E2_loss"

    for cohort, target in COHORT_TARGETS.items():
        if cohort == "C0_win_E2_loss":
            continue
        candidates = [s for s in pools[cohort] if s not in assigned]
        n = target if target is not None else len(candidates)
        for sid in stratified_sample(candidates, meta, n, seed=SEED + hash(cohort) % 997):
            assigned[sid] = cohort
    return assigned


def score_all_variants(
    ids: List[str],
    arms: Dict[str, Dict[str, dict]],
    tasks: Dict[str, dict],
) -> Tuple[Dict[str, Dict[str, dict]], Dict[str, Dict[str, float]]]:
    """Return per (task, arm) variant dicts and R0 summary per task."""
    all_variants: Dict[str, Dict[str, dict]] = {}
    r0_summary: Dict[str, Dict[str, float]] = {}
    for i, sid in enumerate(ids):
        task = tasks[sid]
        r0_summary[sid] = {}
        for arm in ("C0", "E1", "E2"):
            row = arms[arm][sid]
            traj = traj_from_dict(row["_traj"])
            variants = score_variants(traj, task, row)
            all_variants.setdefault(sid, {})[arm] = {
                k: variant_to_dict(v) for k, v in variants.items()
            }
            if arm in ("C0", "E2"):
                r0_summary[sid][arm] = variants["R0"].total_reward
                r0_summary[sid][f"class_{arm}"] = variants["R0"].terminal_class
        if (i + 1) % 300 == 0:
            print(f"  variants {i+1}/{len(ids)}")
    return all_variants, r0_summary


def build_compact_case(
    sid: str,
    cohort: str,
    task: dict,
    arms: Dict[str, dict],
    variants: Dict[str, dict],
    meta: dict,
) -> dict:
    c0, e1, e2 = arms["C0"], arms["E1"], arms["E2"]
    c0_calls = predicted_calls(c0)
    e2_calls = predicted_calls(e2)
    gold = task.get("gold_calls") or []
    traj0 = c0.get("_traj") or {}
    traje = e2.get("_traj") or {}
    return {
        "task_id": sid,
        "cohort": cohort,
        "question": task.get("question", "")[:1200],
        "relevant_tools": relevant_tools(
            task, c0_calls, predicted_calls(e1), e2_calls, gold,
        ),
        "expected_outcome": compact_value(task.get("gold_answer")),
        "gold_call_count": len(gold),
        "gold_motif": meta.get("motif"),
        "C0": arm_snapshot(c0, {k: _dict_to_vs(v) for k, v in variants["C0"].items()}),
        "E1": arm_snapshot(e1, {k: _dict_to_vs(v) for k, v in variants["E1"].items()}),
        "E2": arm_snapshot(e2, {k: _dict_to_vs(v) for k, v in variants["E2"].items()}),
        "deterministic_flags": {
            "first_divergence_turn": meta.get("first_divergence_turn"),
            "shorter_than_gold": len(e2_calls) < len(gold),
            "taxonomy_too_few": classify_failure(e2)[0] == "too few calls",
            "executable": bool(traje.get("executable")),
            "reward_mismatch_R0": meta.get("reward_mismatch"),
            "R0_class_C0": variants["C0"]["R0"]["terminal_class"],
            "R0_class_E2": variants["E2"]["R0"]["terminal_class"],
            "official_win_C0": official_win(c0) == 1.0,
            "official_win_E2": official_win(e2) == 1.0,
        },
    }


class _Vs:
    def __init__(self, d: dict):
        self.terminal_class = d.get("terminal_class", "")
        self.process_score = d.get("process_score", 0.0)
        self.total_reward = d.get("total_reward", 0.0)
        self.terminal_reward = d.get("terminal_reward", 0.0)
        self.components = d.get("components", {})


def _dict_to_vs(d: dict) -> _Vs:
    return _Vs(d)


def pseudo_group_metrics(
    ids: List[str],
    all_variants: Dict[str, Dict[str, dict]],
    arms: Dict[str, Dict[str, dict]],
) -> Tuple[List[dict], dict]:
    per_task_rows: List[dict] = []
    metrics: Dict[str, defaultdict] = {v: defaultdict(float) for v in ("R0", "R1", "R2", "R3")}
    counts: Dict[str, Counter] = {v: Counter() for v in ("R0", "R1", "R2", "R3")}
    adv_e2_r0: Dict[str, float] = {}

    for sid in ids:
        official = {
            arm: official_win(arms[arm][sid]) == 1.0
            for arm in ("C0", "E1", "E2")
        }
        variant_rows: Dict[str, dict] = {}
        for vn in ("R0", "R1", "R2", "R3"):
            ep = [all_variants[sid][a][vn]["total_reward"] for a in ("C0", "E1", "E2")]
            r_seq = [[r] for r in ep]
            gstats = compute_group_stats(
                [_turn_returns(s, r, GAMMA, LAMBDA_EP) for s, r in zip(r_seq, ep)],
                ep,
            )
            ranks = sorted(range(3), key=lambda i: ep[i], reverse=True)
            rank_map = {("C0", "E1", "E2")[i]: ranks.index(i) + 1 for i in range(3)}
            adv = {
                a: (gstats.advantages[i][0] if i < len(gstats.advantages) and gstats.advantages[i] else 0.0)
                for i, a in enumerate(("C0", "E1", "E2"))
            }
            row = {
                "task_id": sid,
                "variant": vn,
                "official_win": official,
                "rewards": {"C0": ep[0], "E1": ep[1], "E2": ep[2]},
                "ranks": rank_map,
                "advantages_t0": adv,
                "dead_group": gstats.dead_corrected,
                "dead_position_t0": (
                    gstats.position_stds[0] <= 1e-9 if gstats.position_stds else True
                ),
                "reward_spread": max(ep) - min(ep),
            }
            per_task_rows.append(row)
            variant_rows[vn] = row
            metrics[vn]["dead_group_rate"] += float(gstats.dead_corrected)
            metrics[vn]["dead_position_rate"] += float(row["dead_position_t0"])
            metrics[vn]["reward_spread_sum"] += row["reward_spread"]
            if official["C0"] and not official["E2"]:
                if ep[2] > ep[0]:
                    counts[vn]["C0_win_E2_loss_wrong_order"] += 1
                elif ep[2] < ep[0]:
                    counts[vn]["C0_win_E2_loss_correct_order"] += 1
                else:
                    counts[vn]["C0_win_E2_loss_tie"] += 1
            if vn == "R0":
                adv_e2_r0[sid] = adv["E2"]

        for vn in ("R1", "R2", "R3"):
            a0 = adv_e2_r0.get(sid, 0.0)
            a1 = variant_rows[vn]["advantages_t0"]["E2"]
            if (a0 >= 0) != (a1 >= 0):
                counts[vn]["E2_adv_sign_flip_vs_R0"] += 1

    n = len(ids)
    summary = {}
    for vn in ("R0", "R1", "R2", "R3"):
        summary[vn] = {
            "dead_group_rate": round(metrics[vn]["dead_group_rate"] / n, 4),
            "dead_position_t0_rate": round(metrics[vn]["dead_position_rate"] / n, 4),
            "mean_reward_spread": round(metrics[vn]["reward_spread_sum"] / n, 4),
            **dict(counts[vn]),
        }
    return per_task_rows, summary


def build_turn2_table(
    case_ids: List[str],
    arms: Dict[str, Dict[str, dict]],
    tasks: Dict[str, dict],
) -> List[dict]:
    rows: List[dict] = []
    for sid in case_ids:
        task = tasks[sid]
        gold = task.get("gold_calls") or []
        if len(gold) < 2:
            continue
        c0_calls = predicted_calls(arms["C0"][sid])
        if not c0_calls or (c0_calls[0].get("name") or "") != (gold[0].get("name") or ""):
            continue
        obs1 = None
        turns = (arms["C0"][sid].get("_traj") or {}).get("turns") or []
        if turns and turns[0].get("observation") is not None:
            obs1 = turns[0]["observation"]
        rows.append({
            "task_id": sid,
            "gold_tool_2": gold[1].get("name"),
            "C0_tool_2": tool_at(c0_calls, 1),
            "E1_tool_2": tool_at(predicted_calls(arms["E1"][sid]), 1),
            "E2_tool_2": tool_at(predicted_calls(arms["E2"][sid]), 1),
            "tool_1": c0_calls[0].get("name"),
            "observation_1_type": observation_shape(obs1),
            "observation_1_value_shape": observation_shape(obs1),
            "observation_1_preview": compact_value(obs1),
            "gold_tool_2_description": tool_description(task, gold[1].get("name")),
            "E2_tool_2_description": tool_description(task, tool_at(predicted_calls(arms["E2"][sid]), 1)),
            "C0_tool_2_correct": tool_at(c0_calls, 1) == gold[1].get("name"),
            "E2_tool_2_correct": tool_at(predicted_calls(arms["E2"][sid]), 1) == gold[1].get("name"),
        })
    return rows


def turn2_confusion_counts(rows: List[dict]) -> dict:
    """Script-computed buckets for weak-model interpretation."""
    ctr = Counter()
    pairs: Counter = Counter()
    for r in rows:
        g = r.get("gold_tool_2") or ""
        e2 = r.get("E2_tool_2") or ""
        if r.get("E2_tool_2_correct"):
            ctr["E2_correct"] += 1
        elif e2 and g and e2.lower() == g.lower() and e2 != g:
            ctr["similar_name"] += 1
            pairs[(g, e2)] += 1
        elif e2 and g and (e2.split("_")[0] == g.split("_")[0]):
            ctr["similar_name_prefix"] += 1
            pairs[(g, e2)] += 1
        elif e2 and g:
            ctr["different_tool"] += 1
            pairs[(g, e2)] += 1
        elif not e2:
            ctr["missing_second_call"] += 1
    return {
        "n_rows": len(rows),
        "E2_wrong": sum(1 for r in rows if not r.get("E2_tool_2_correct")),
        "confusion_type_counts": dict(ctr),
        "top_tool_pairs": [{"gold": a, "E2": b, "count": c} for (a, b), c in pairs.most_common(15)],
    }


def make_annotation_inputs(case: dict, out_dir: Path) -> None:
    sid = case["task_id"]
    named = {
        "task_id": sid,
        "prompt_preamble": ANALYST_PROMPT,
        "case": case,
        "annotation_variant": "named_C0_E1_E2",
    }
    _write_json(out_dir / "named" / f"{sid}.json", named)

    arms = ["C0", "E1", "E2"]
    rng = random.Random(SEED + hash(sid) % 10000)
    labels = ["Trajectory A", "Trajectory B", "Trajectory C"]
    order = list(arms)
    rng.shuffle(order)
    mapping = {labels[i]: order[i] for i in range(3)}
    anon_case = {
        "task_id": sid,
        "question": case["question"],
        "relevant_tools": case["relevant_tools"],
        "expected_outcome": case["expected_outcome"],
        "deterministic_flags": case["deterministic_flags"],
        "trajectories": {
            labels[i]: case[order[i]]
            for i in range(3)
        },
        "note": "Official labels hidden. Do not assume A=C0.",
    }
    _write_json(out_dir / "anonymized" / f"{sid}.json", {
        "task_id": sid,
        "prompt_preamble": ANALYST_PROMPT.replace("C0/E1/E2", "Trajectory A/B/C"),
        "case": anon_case,
        "annotation_variant": "anonymized_ABC",
        "_label_map_for_evaluator_only": mapping,
    })


def cluster_input_template(case_ids: List[str], annotations_path: Optional[Path]) -> dict:
    """Pre-computed counts; LLM only interprets."""
    template = {
        "instruction": (
            "Dostaneš tabulku anotací (root_cause, recommended_fix, confidence). "
            "Seskupte do clusterů. Počty ověř proti count_fields — nepočítej znovu z raw trajektorií."
        ),
        "clusters_to_fill": [
            "later_tool_selection",
            "observation_grounding",
            "valid_shorter_path_penalized",
            "wrong_global_plan",
            "reward_terminal_mismatch",
            "argument_grounding",
        ],
        "count_fields": {},
        "annotation_rows": [],
    }
    if annotations_path and annotations_path.is_file():
        rows = [json.loads(l) for l in annotations_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        template["annotation_rows"] = rows
        template["count_fields"] = dict(Counter(r.get("root_cause", "unclear") for r in rows))
    else:
        template["note"] = "Run weak-model annotation first; populate diagnostic_annotations.jsonl"
    return template


def run(run_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    nestful_path = default_test_path()
    tasks = load_tasks(nestful_path)
    arms = {a: load_traj_rows(run_dir / "eval" / d) for a, d in ARM_DIRS.items()}
    ids = sorted(set.intersection(*(set(v) for v in arms.values())))
    print(f"[pack] loading {len(ids)} tasks…")

    print("[pack] scoring reward variants (full test)…")
    all_variants, r0_summary = score_all_variants(ids, arms, tasks)
    meta = build_task_meta(ids, arms, tasks, r0_summary)
    cohort_map = assign_cohorts(meta, r0_summary)
    case_ids = sorted(cohort_map.keys())
    print(f"[pack] selected {len(case_ids)} diagnostic cases")

    cases: List[dict] = []
    for sid in case_ids:
        case = build_compact_case(
            sid, cohort_map[sid], tasks[sid],
            {a: arms[a][sid] for a in arms},
            all_variants[sid], meta[sid],
        )
        cases.append(case)

    _write_jsonl(out_dir / "diagnostic_cases.jsonl", cases)
    _write_json(out_dir / "cohort_manifest.json", {
        "generated_at": _now(),
        "n_cases": len(cases),
        "cohort_counts": dict(Counter(cohort_map.values())),
        "targets": COHORT_TARGETS,
    })

    print("[pack] pseudo-group reward metrics…")
    per_task_rows, variant_summary = pseudo_group_metrics(ids, all_variants, arms)
    _write_jsonl(out_dir / "reward_variant_per_task.jsonl", per_task_rows)

    # Global eval metrics (all 1661)
    global_metrics = {}
    for vn in ("R0", "R1", "R2", "R3"):
        off_win_fail = 0
        loss_gt_win = 0
        too_few_win = 0
        hi_exec_wrong = 0
        n = 0
        for sid in ids:
            for arm in ("C0", "E1", "E2"):
                n += 1
                v = all_variants[sid][arm][vn]
                tc = v["terminal_class"]
                ow = official_win(arms[arm][sid]) == 1.0
                if vn == "R0":
                    if ow and tc not in ("fully_correct",) and tc != "too_few_calls":
                        if tc not in ("fully_correct",):
                            off_win_fail += 1
                    if ow and tc == "too_few_calls":
                        too_few_win += 1
                    if tc == "executable_wrong_final" and v["total_reward"] >= 0.52:
                        hi_exec_wrong += 1
                else:
                    if ow and tc != "official_success":
                        off_win_fail += 1
                    if tc == "executable_wrong_outcome" and v["total_reward"] >= 0.45:
                        hi_exec_wrong += 1
            c0w = official_win(arms["C0"][sid]) == 1.0
            e2w = official_win(arms["E2"][sid]) == 1.0
            if not c0w and e2w:
                if all_variants[sid]["E2"][vn]["total_reward"] > all_variants[sid]["C0"][vn]["total_reward"]:
                    loss_gt_win += 1
        global_metrics[vn] = {
            "official_win_labeled_failure": off_win_fail,
            "official_loss_reward_above_win": loss_gt_win,
            "valid_shorter_path_penalized_too_few": too_few_win if vn == "R0" else 0,
            "executable_wrong_high_reward": hi_exec_wrong,
            "n_trajectories": n,
        }

    _write_json(out_dir / "reward_variant_metrics.json", {
        "generated_at": _now(),
        "epsilon_R2": DEFAULT_EPS_R2,
        "epsilon_R3": DEFAULT_EPS_R3,
        "pseudo_group_per_task_n": len(ids),
        "per_variant_pseudo_group": variant_summary,
        "global_trajectory_metrics": global_metrics,
        "case_subset_C0_win_E2_loss_ordering": {
            vn: variant_summary[vn].get("C0_win_E2_loss_wrong_order", 0)
            for vn in ("R0", "R1", "R2", "R3")
        },
    })

    turn2_rows = build_turn2_table(case_ids, arms, tasks)
    _write_jsonl(out_dir / "turn2_tool_table.jsonl", turn2_rows)
    t2_counts = turn2_confusion_counts(turn2_rows)
    _write_json(out_dir / "turn2_confusion_counts.json", t2_counts)

    (out_dir / "annotation_prompt.txt").write_text(ANALYST_PROMPT, encoding="utf-8")
    ann_dir = out_dir / "annotation_inputs"
    for case in cases:
        make_annotation_inputs(case, ann_dir)

    _write_json(out_dir / "cluster_input_template.json", cluster_input_template(
        case_ids, out_dir / "diagnostic_annotations.jsonl" if (out_dir / "diagnostic_annotations.jsonl").is_file() else None
    ))

    # CSV exports
    with open(out_dir / "turn2_tool_table.csv", "w", newline="", encoding="utf-8") as fh:
        if turn2_rows:
            w = csv.DictWriter(fh, fieldnames=list(turn2_rows[0].keys()), extrasaction="ignore")
            w.writeheader()
            w.writerows(turn2_rows)

    md = _format_summary_md(len(cases), cohort_map, variant_summary, global_metrics, t2_counts)
    (out_dir / "README.md").write_text(md, encoding="utf-8")
    print(f"[pack] done -> {out_dir}")


def _format_summary_md(n_cases, cohort_map, variant_summary, global_metrics, t2_counts) -> str:
    lines = [
        "# Pure Stage-3 diagnostic pack",
        "",
        f"**Generated:** {_now()}",
        f"**Cases:** {n_cases}",
        "",
        "## Cohort counts",
        "",
        "```json",
        json.dumps(dict(Counter(cohort_map.values())), indent=2),
        "```",
        "",
        "## Reward variant summary (pseudo-group C0/E1/E2 per task, n=1661)",
        "",
        "| Variant | dead_group | C0win→E2loss wrong order | too_few on official win | exec_wrong reward≥0.52 |",
        "|---------|----------:|-------------------------:|------------------------:|------------------------:|",
    ]
    for vn in ("R0", "R1", "R2", "R3"):
        ps = variant_summary[vn]
        gm = global_metrics[vn]
        lines.append(
            f"| {vn} | {ps.get('dead_group_rate', 0):.3f} | "
            f"{ps.get('C0_win_E2_loss_wrong_order', 0)} | "
            f"{gm.get('valid_shorter_path_penalized_too_few', 0)} | "
            f"{gm.get('executable_wrong_high_reward', 0)} |"
        )
    lines += [
        "",
        "## Turn-2 confusion (case subset, script counts)",
        "",
        "```json",
        json.dumps(t2_counts, indent=2, ensure_ascii=False)[:3000],
        "```",
        "",
        "## Next: weak-model annotation",
        "",
        "1. Run `annotation_inputs/named/*.json` and `anonymized/*.json` with `annotation_prompt.txt`",
        "2. Merge to `diagnostic_annotations.jsonl`",
        "3. Re-run pack or fill `cluster_input_template.json`",
        "4. Escalate disagreements / low confidence / C0_win_E2_loss to strong model",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    args = ap.parse_args()
    run(args.run_dir, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
