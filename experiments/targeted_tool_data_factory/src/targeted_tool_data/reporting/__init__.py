"""Report generation: PILOT_REPORT.md and COST_REPORT.md from run artifacts."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


def _pct(x: float) -> str:
    return f"{100 * x:.1f} %"


def _dist_table(rows: List[Dict[str, Any]], key) -> str:
    c = Counter(key(r) for r in rows)
    n = sum(c.values()) or 1
    lines = ["| value | count | share |", "|---|---|---|"]
    for k, v in sorted(c.items(), key=lambda x: str(x[0])):
        lines.append(f"| {k} | {v} | {_pct(v / n)} |")
    return "\n".join(lines)


def build_pilot_report(*, version: str, selected: List[Dict[str, Any]],
                       gen_stats: Dict[str, Any], validation_summary: Dict[str, Any],
                       profile_match: List[Dict[str, Any]],
                       distribution_audit: Dict[str, Any],
                       leakage: Dict[str, Any], manifest: Dict[str, Any],
                       verdict: Dict[str, Any], thresholds: Dict[str, Any],
                       examples: List[Dict[str, Any]]) -> str:
    n = len(selected)
    hard_share = sum(1 for r in selected if r["hard_distractor_count"] > 0) / max(n, 1)
    track_c = Counter(r["track"] for r in selected)
    probe_status = Counter((r.get("student_probe_result") or {}).get("status", "NONE")
                           for r in selected)

    cell_rows = []
    sel_by_cell = Counter(r["generation_cell_id"] for r in selected)
    for cell_id, st in sorted(gen_stats.get("cells", {}).items()):
        cell_rows.append(
            f"| {cell_id} | {st.get('requested', 0)} | {st.get('generated', 0)} | "
            f"{st.get('validated', 0)} | {st.get('rejected', 0)} | {sel_by_cell.get(cell_id, 0)} |")

    rej_tax = validation_summary.get("rejection_taxonomy", {})
    rej_lines = [f"| {k} | {v} |" for k, v in
                 sorted(rej_tax.items(), key=lambda x: -x[1])]

    pm_lines = ["| dataset | JSD call | JSD motif | JSD args | JSD answer | "
                "W tools | W qlen | AUC |", "|---|---|---|---|---|---|---|---|"]
    for pm in profile_match:
        pm_lines.append(
            f"| {pm['label']} | {pm['jsd_call_bucket']:.4f} | {pm['jsd_motif']:.4f} | "
            f"{pm['jsd_arg_types']:.4f} | {pm['jsd_answer_type']:.4f} | "
            f"{pm['wass_n_tools']:.2f} | {pm['wass_q_len']:.1f} | {pm['auc_two_sample']:.3f} |")

    ex_lines = []
    for e in examples[:10]:
        ex_lines.append(
            f"### {e['task_id']} ({e['track']}, {e['generation_cell_id']})\n"
            f"- query: {e['query']}\n"
            f"- calls: `{json.dumps([{'name': c['name'], 'arguments': c['arguments']} for c in e['canonical_calls']], ensure_ascii=False)}`\n"
            f"- answer: `{e['gold_answer']}` | offered {e['offered_tool_count']} "
            f"(hard distractors {e['hard_distractor_count']})\n")

    return f"""# PILOT REPORT — targeted_tool_data_factory {version}

## Verdict: **{verdict['verdict']}**

{chr(10).join('- ' + r for r in verdict['reasons'])}

## Counts
- generated candidates: {gen_stats.get('n_generated', 0)}
- validated: {validation_summary.get('n_passed', 0)}
- rejected: {validation_summary.get('n_rejected', 0)}
- selected (frozen pilot): {n}
- track mix: A={track_c.get('A', 0)}, G={track_c.get('G', 0)} \
(A share {_pct(track_c.get('A', 0) / max(n, 1))})
- tasks with >=1 hard distractor: {_pct(hard_share)} \
(threshold >= {_pct(thresholds.get('hard_distractor_min_task_share', 0.5))})
- deterministic replay rate: {_pct(validation_summary.get('replay_rate', 0.0))}
- contamination hits in candidate pool (all rejected; selected pool = 0): \
{validation_summary.get('n_contaminated', 0)}
- dedup drops: {validation_summary.get('n_deduped', 0)}
- split leakage collisions: {len(leakage.get('leakage_collisions', []))}
- split sizes: {leakage.get('split_sizes', {})}

## Rejection taxonomy
| reason | count |
|---|---|
{chr(10).join(rej_lines) if rej_lines else '| (none) | 0 |'}

## Generation cells (requested / generated / validated / rejected / selected)
| cell | req | gen | valid | rej | sel |
|---|---|---|---|---|---|
{chr(10).join(cell_rows)}

## Profile match vs NESTFUL dev (lower = closer; AUC 0.5 = indistinguishable)
{chr(10).join(pm_lines)}

## Selected distributions
### Call counts
{_dist_table(selected, lambda r: r['call_count'])}

### Motifs
{_dist_table(selected, lambda r: r['motif'])}

### Answer types
{_dist_table(selected, lambda r: r['answer_type'])}

### Templates (max share {_pct(distribution_audit.get('template_max_share', 0))}, cap {_pct(thresholds.get('template_max_share', 0.05))})
top: `{distribution_audit.get('template_top', [])[:5]}`

### Cells (max share {_pct(distribution_audit.get('cell_max_share', 0))}, cap {_pct(thresholds.get('cell_max_share', 0.10))})

### Distribution warnings
{chr(10).join('- ' + w for w in distribution_audit.get('warnings', [])) or '- none'}

## Student probe
status: `{dict(probe_status)}`

## Hashes / files
```json
{json.dumps(manifest.get('files', {}), indent=2)}
```
- config_hash: `{manifest.get('config_hash', '')}`
- profile_hash: `{manifest.get('profile_hash', '')}`
- registry_hash: `{manifest.get('registry_hash', '')}`
- executor_hash: `{manifest.get('executor_hash', '')}`
- generator_version: `{manifest.get('generator_version', '')}`

## Representative tasks (max 10)
{chr(10).join(ex_lines)}
"""


def build_cost_report(version: str, run_state: Dict[str, Any],
                      out_dir_bytes: int, llm_calls: int = 0) -> str:
    steps = run_state.get("steps", {})
    lines = ["| step | wall s | cpu s | peak python MB |", "|---|---|---|---|"]
    tw = tc = 0.0
    for name, st in steps.items():
        lines.append(f"| {name} | {st.get('wall_s', 0):.1f} | {st.get('cpu_s', 0):.1f} | "
                     f"{st.get('peak_mb', 0):.0f} |")
        tw += st.get("wall_s", 0)
        tc += st.get("cpu_s", 0)
    return f"""# COST REPORT — {version}

Local CPU-only run. No GPU, no remote API.

{chr(10).join(lines)}

- total wall time: {tw:.1f} s
- total CPU time: {tc:.1f} s
- peak python-allocated RAM (tracemalloc, per step max): \
{max((st.get('peak_mb', 0) for st in steps.values()), default=0):.0f} MB \
(process RSS is higher; tracemalloc tracks python allocations only)
- outputs disk usage: {out_dir_bytes / 1e6:.1f} MB
- local model inference: {'yes' if run_state.get('probe_model_used') else 'none'}
- LLM call count: {llm_calls}
- paid API cost: £0 (no remote endpoint enabled)
"""


def readiness_verdict(*, validation_summary: Dict[str, Any],
                      profile_match: List[Dict[str, Any]],
                      distribution_audit: Dict[str, Any],
                      leakage: Dict[str, Any], selected: List[Dict[str, Any]],
                      thresholds: Dict[str, Any],
                      probe_ran: bool) -> Dict[str, Any]:
    reasons: List[str] = []
    fails: List[str] = []
    warns: List[str] = []
    n = max(len(selected), 1)

    if validation_summary.get("replay_rate", 0) < 1.0:
        fails.append(f"deterministic replay {validation_summary.get('replay_rate')} < 100 %")
    if validation_summary.get("n_contaminated", 0) > 0:
        fails.append("target contamination present in selected pool")
    if leakage.get("leaked"):
        fails.append("split leakage collisions present")
    hard_share = sum(1 for r in selected if r["hard_distractor_count"] > 0) / n
    if hard_share < thresholds.get("hard_distractor_min_task_share", 0.5):
        fails.append(f"hard-distractor task share {hard_share:.2f} below threshold")

    new = next((p for p in profile_match if p["label"] == "new_selected"), None)
    old = next((p for p in profile_match if p["label"] == "stage3_old"), None)
    if new and old:
        closer = 0
        keys = [k for k in new if k.startswith(("jsd_", "wass_"))]
        for k in keys:
            if new[k] <= old[k]:
                closer += 1
        reasons.append(f"profile-match: new dataset closer than Stage-3 on "
                       f"{closer}/{len(keys)} metrics")
        if closer < len(keys) * 0.7:
            warns.append("new dataset not clearly closer to target than Stage-3")
        if not (new["auc_two_sample"] <= old["auc_two_sample"]):
            warns.append(f"classifier AUC new={new['auc_two_sample']} not below "
                         f"stage3={old['auc_two_sample']}")
        if new["auc_two_sample"] > thresholds.get("auc_warn", 0.75):
            warns.append(f"two-sample AUC {new['auc_two_sample']} > "
                         f"{thresholds.get('auc_warn')}")
    for w in distribution_audit.get("warnings", []):
        warns.append(w)
    if not probe_ran:
        warns.append("student probe NOT_RUN_LOCAL (structural P0 only)")

    if fails:
        verdict = "NOT_READY"
    elif warns:
        verdict = "CONDITIONAL"
    else:
        verdict = "READY_FOR_PILOT_TRAINING"
    return {"verdict": verdict,
            "reasons": reasons + [f"FAIL: {f}" for f in fails] +
                       [f"WARN: {w}" for w in warns]}
