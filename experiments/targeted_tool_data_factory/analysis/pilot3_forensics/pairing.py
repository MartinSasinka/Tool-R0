"""C0/D1 pairing and headline reproduction."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .io import as_bool, write_csv, write_json, write_md
from .statistics import call_bucket, mcnemar_exact, paired_bootstrap_delta


def load_traj_index(path: Path) -> Dict[str, Dict[str, Any]]:
    from .io import read_jsonl

    out: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(path):
        sid = str(row.get("sample_id"))
        traj = row.get("_traj") or {}
        out[sid] = {
            "row": row,
            "traj": traj,
            "win": bool(as_bool(traj.get("official_win"))),
            "num_gold_calls": int(row.get("num_gold_calls") or traj.get("gold_num_turns") or 0),
            "num_tool_calls": int(traj.get("num_tool_calls") or 0),
            "parse_valid": as_bool(traj.get("parse_valid")),
            "executable": as_bool(traj.get("executable")),
            "execution_error": traj.get("execution_error"),
            "stop_reason": traj.get("stop_reason"),
            "pred_answer": traj.get("pred_answer"),
            "final_answer_pass": bool(as_bool(row.get("final_answer_pass"))),
            "sol_eq": bool(as_bool(row.get("solution_equivalent_pass"))),
            "strict_gold": bool(as_bool(row.get("strict_gold_trace_pass"))),
            "alt_sol": bool(as_bool(row.get("alternative_valid_solution_pass"))),
            "unsupported_trace": bool(as_bool(row.get("correct_answer_but_unsupported_trace"))),
            "f1_func": float(row.get("internal_f1_func") or (traj.get("internal") or {}).get("f1_func") or 0.0),
            "f1_param": float(row.get("internal_f1_param") or (traj.get("internal") or {}).get("f1_param") or 0.0),
            "clipped_any": bool(as_bool(traj.get("clipped_any"))),
            "prompt_overflow": bool(as_bool(traj.get("prompt_overflow"))),
            "mismatch_reason": traj.get("mismatch_reason"),
        }
    return out


def pair_ids(c0: Dict[str, Any], d1: Dict[str, Any]) -> List[str]:
    return sorted(set(c0) & set(d1))


def outcome_label(c0_win: bool, d1_win: bool) -> str:
    if c0_win and d1_win:
        return "win_to_win"
    if (not c0_win) and (not d1_win):
        return "loss_to_loss"
    if (not c0_win) and d1_win:
        return "loss_to_win"
    return "win_to_loss"


def reproduce_headline(
    c0: Dict[str, Dict[str, Any]],
    d1: Dict[str, Dict[str, Any]],
    *,
    seed: int = 42,
    n_boot: int = 20000,
    c0_hf: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], str]:
    ids = pair_ids(c0, d1)
    wins_c0 = [c0[i]["win"] for i in ids]
    wins_d1 = [d1[i]["win"] for i in ids]
    buckets = [call_bucket(c0[i]["num_gold_calls"]) for i in ids]
    n = len(ids)
    n_c0 = sum(wins_c0)
    n_d1 = sum(wins_d1)
    gained = [i for i in ids if outcome_label(c0[i]["win"], d1[i]["win"]) == "loss_to_win"]
    lost = [i for i in ids if outcome_label(c0[i]["win"], d1[i]["win"]) == "win_to_loss"]
    both_win = [i for i in ids if outcome_label(c0[i]["win"], d1[i]["win"]) == "win_to_win"]
    both_loss = [i for i in ids if outcome_label(c0[i]["win"], d1[i]["win"]) == "loss_to_loss"]

    b, c = len(lost), len(gained)
    p_mc = mcnemar_exact(b, c)
    boot = paired_bootstrap_delta(wins_c0, wins_d1, n_boot=n_boot, seed=seed)
    boot_strat = paired_bootstrap_delta(wins_c0, wins_d1, n_boot=n_boot, seed=seed, strata=buckets)

    by_bucket: Dict[str, Any] = {}
    groups: Dict[str, List[str]] = defaultdict(list)
    for sid, bk in zip(ids, buckets):
        groups[bk].append(sid)
    for bk in ["2", "3", "4", "5", "6+"]:
        sids = groups.get(bk, [])
        if not sids:
            by_bucket[bk] = {"n": 0}
            continue
        wa = sum(c0[i]["win"] for i in sids) / len(sids)
        wb = sum(d1[i]["win"] for i in sids) / len(sids)
        by_bucket[bk] = {
            "n": len(sids),
            "wins_c0": sum(c0[i]["win"] for i in sids),
            "wins_d1": sum(d1[i]["win"] for i in sids),
            "win_rate_c0": wa,
            "win_rate_d1": wb,
            "delta_pp": 100.0 * (wb - wa),
            "gained": sum(1 for i in sids if i in gained),
            "lost": sum(1 for i in sids if i in lost),
        }

    paired_rows = []
    for sid in ids:
        paired_rows.append({
            "sample_id": sid,
            "num_gold_calls": c0[sid]["num_gold_calls"],
            "call_bucket": call_bucket(c0[sid]["num_gold_calls"]),
            "c0_win": int(c0[sid]["win"]),
            "d1_win": int(d1[sid]["win"]),
            "outcome": outcome_label(c0[sid]["win"], d1[sid]["win"]),
            "c0_executable": int(bool(c0[sid]["executable"])),
            "d1_executable": int(bool(d1[sid]["executable"])),
            "c0_final_answer_pass": int(c0[sid]["final_answer_pass"]),
            "d1_final_answer_pass": int(d1[sid]["final_answer_pass"]),
            "c0_sol_eq": int(c0[sid]["sol_eq"]),
            "d1_sol_eq": int(d1[sid]["sol_eq"]),
            "c0_strict_gold": int(c0[sid]["strict_gold"]),
            "d1_strict_gold": int(d1[sid]["strict_gold"]),
        })

    result: Dict[str, Any] = {
        "n": n,
        "wins_c0": n_c0,
        "wins_d1": n_d1,
        "win_rate_c0": n_c0 / n,
        "win_rate_d1": n_d1 / n,
        "delta_pp": 100.0 * (n_d1 - n_c0) / n,
        "relative_delta": ((n_d1 / n) / (n_c0 / n) - 1.0) if n_c0 else None,
        "win_to_win": len(both_win),
        "loss_to_loss": len(both_loss),
        "loss_to_win": len(gained),
        "win_to_loss": len(lost),
        "mcnemar": {"b_win_to_loss": b, "c_loss_to_win": c, "exact_p": p_mc},
        "bootstrap_paired": boot,
        "bootstrap_stratified_by_call_bucket": boot_strat,
        "by_call_bucket": by_bucket,
        "methodology": {
            "alignment": "sample_id",
            "official_metric": "_traj.official_win",
            "bootstrap_iterations": n_boot,
            "bootstrap_seed": seed,
            "bootstrap_ci": "percentile 2.5/97.5",
            "stratified_bootstrap": "resample within call_bucket then pool",
            "mcnemar": "exact two-sided binomial(b+c, 0.5)",
        },
        "interpretation_guards": [
            "diagnostic-500 is a balanced slice (100 tasks per call-count bucket 2/3/4/5/6+).",
            "Overall win equals the macro average across call-count buckets.",
            "It is not an estimate of naturally distributed NESTFUL official win rate.",
            "+delta_pp is a matched-engine point estimate, not a proven causal training effect.",
            "Residual LoRA inference-path confound cannot be removed from these two trajectory sets alone.",
        ],
        "gained_ids": gained,
        "lost_ids": lost,
    }

    if c0_hf is not None:
        shared_hf = sorted(set(ids) & set(c0_hf))
        if shared_hf:
            wh = sum(c0_hf[i]["win"] for i in shared_hf)
            result["three_arm"] = {
                "n": len(shared_hf),
                "wins_c0_hf": wh,
                "wins_c0_vllm": sum(c0[i]["win"] for i in shared_hf),
                "wins_d1_vllm": sum(d1[i]["win"] for i in shared_hf),
                "win_rate_c0_hf": wh / len(shared_hf),
                "win_rate_c0_vllm": sum(c0[i]["win"] for i in shared_hf) / len(shared_hf),
                "win_rate_d1_vllm": sum(d1[i]["win"] for i in shared_hf) / len(shared_hf),
                "note": "C0-HF must not be mixed into the main training-effect contrast.",
            }

    md = _headline_md(result)
    return result, paired_rows, md


def _headline_md(r: Dict[str, Any]) -> str:
    lines = [
        "# HEADLINE_REPRODUCTION",
        "",
        "## Matched-engine C0-vLLM vs D1-vLLM",
        "",
        f"- C0 wins: **{r['wins_c0']}/{r['n']}** ({100*r['win_rate_c0']:.1f}%)",
        f"- D1 wins: **{r['wins_d1']}/{r['n']}** ({100*r['win_rate_d1']:.1f}%)",
        f"- Absolute delta: **{r['delta_pp']:+.2f} pp**",
        f"- Transitions: win→win={r['win_to_win']}, loss→loss={r['loss_to_loss']}, "
        f"loss→win={r['loss_to_win']}, win→loss={r['win_to_loss']}",
        f"- McNemar exact p = `{r['mcnemar']['exact_p']:.4g}` "
        f"(b={r['mcnemar']['b_win_to_loss']}, c={r['mcnemar']['c_loss_to_win']})",
        f"- Paired bootstrap 95% CI (pp): "
        f"[{r['bootstrap_paired']['ci95_lo_pp']:.2f}, {r['bootstrap_paired']['ci95_hi_pp']:.2f}]",
        f"- Stratified (call-bucket) bootstrap 95% CI (pp): "
        f"[{r['bootstrap_stratified_by_call_bucket']['ci95_lo_pp']:.2f}, "
        f"{r['bootstrap_stratified_by_call_bucket']['ci95_hi_pp']:.2f}]",
        "",
        "## Call-count buckets",
        "",
    ]
    for bk, row in r["by_call_bucket"].items():
        if not row.get("n"):
            continue
        lines.append(
            f"- {bk}: n={row['n']} C0={100*row['win_rate_c0']:.1f}% "
            f"D1={100*row['win_rate_d1']:.1f}% Δ={row['delta_pp']:+.1f} pp "
            f"(gained={row['gained']}, lost={row['lost']})"
        )
    lines += ["", "## Interpretation guards", ""]
    for g in r["interpretation_guards"]:
        lines.append(f"- {g}")
    if "three_arm" in r:
        t = r["three_arm"]
        lines += [
            "",
            "## Three-arm (separate; not main contrast)",
            "",
            f"- C0-HF: {100*t['win_rate_c0_hf']:.1f}%",
            f"- C0-vLLM: {100*t['win_rate_c0_vllm']:.1f}%",
            f"- D1-vLLM: {100*t['win_rate_d1_vllm']:.1f}%",
            f"- {t['note']}",
        ]
    lines += [
        "",
        "## Methodology",
        "",
        f"- alignment: `{r['methodology']['alignment']}`",
        f"- bootstrap: {r['methodology']['bootstrap_iterations']} iters, seed={r['methodology']['bootstrap_seed']}",
        f"- McNemar: {r['methodology']['mcnemar']}",
        "",
    ]
    return "\n".join(lines)


def write_headline_outputs(out_dir: Path, result: Dict[str, Any], paired_rows: List[Dict[str, Any]], md: str) -> None:
    write_json(out_dir / "HEADLINE_REPRODUCTION.json", result)
    write_md(out_dir / "HEADLINE_REPRODUCTION.md", md)
    write_csv(out_dir / "PAIRED_TASK_OUTCOMES.csv", paired_rows)
