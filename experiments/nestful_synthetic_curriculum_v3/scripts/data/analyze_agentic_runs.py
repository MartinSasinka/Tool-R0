#!/usr/bin/env python3
"""Diagnostic report over agentic v5 generation worker directories."""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE = ROOT / "data" / "agentic_workers"
DEFAULT_RUNS = [
    "agentic_v5_stage3_win1",
    "agentic_v5_stage3_loose",
    "agentic_v5_workers_stage3",
    "agentic_v5_pilot_stage2",
]

BATCH_DONE_RE = re.compile(r"BATCH (\d+) rollout done \| grpo_ok=(\d+)/(\d+)")
BATCH_SUMMARY_RE = re.compile(
    r"BATCH (\d+) \| accepted (\d+)/(\d+) \| new=(\d+) rejected=(\d+) \| "
    r"rate=([\d.]+) \| batch_rejects: ([^|]+)"
)


def resolve_run_root(path: Path) -> Path:
    if not path.exists():
        return path
    children = [c for c in path.iterdir() if c.is_dir()]
    if len(children) == 1 and children[0].name == path.name:
        return children[0]
    return path


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def classify_grpo_fail(sig: Dict[str, Any]) -> str:
    if not sig or sig.get("skipped"):
        return "skipped"
    if sig.get("grpo_sub_reason"):
        return str(sig["grpo_sub_reason"])
    if sig.get("grpo_signal_positive"):
        return "accepted_signal"
    ur = sig.get("unique_rewards", 0)
    var = sig.get("reward_variance", 0) or 0
    fsr = sig.get("full_success_rate", 0) or 0
    all_deg = sig.get("all_degenerate", False)
    hvt = sig.get("has_valid_trace", True)
    needs = sig.get("requires_achievable_win", False)
    ach = sig.get("achievable_win", False)
    statuses = sig.get("failure_type_distribution") or {}
    n = sig.get("n", 8)

    if all_deg:
        if statuses.get("parse_error", 0) == n:
            return "all_parse_fail"
        if statuses.get("no_tool_call", 0) == n:
            return "all_no_tool"
        return "all_degenerate"
    if ur < 2:
        return "all_same_reward"
    if var <= 0:
        return "variance_below_threshold"
    if fsr >= 0.999:
        return "all_correct_trivial"
    if not hvt:
        return "no_valid_trace"
    if needs and not ach:
        if fsr == 0:
            return "no_full_success"
        return "no_achievable_win_band"
    return "other_spread_fail"


def quality_tier(sig: Dict[str, Any]) -> str:
    ur = sig.get("unique_rewards") or 0
    var = sig.get("reward_variance") or 0
    fsr = sig.get("full_success_rate") or 0
    if ur < 2 or var <= 0:
        return "reject_flat"
    if fsr >= 0.999:
        return "easy_anchor"
    if 0 < fsr < 0.999:
        return "frontier"
    if ur >= 2 and var > 0 and fsr == 0:
        return "partial_frontier"
    return "other"


def parse_gpu_log(log_path: Path) -> Dict[str, Any]:
    if not log_path.exists():
        return {}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    batches: Dict[int, Dict[str, Any]] = {}
    for batch, ok, pool in BATCH_DONE_RE.findall(text):
        batches[int(batch)] = {"grpo_ok": int(ok), "pool": int(pool)}
    for m in BATCH_SUMMARY_RE.finditer(text):
        b = int(m.group(1))
        batches.setdefault(b, {})
        batches[b].update(
            accepted=int(m.group(2)),
            target=int(m.group(3)),
            cumulative_rejected=int(m.group(5)),
            rate=float(m.group(6)),
            batch_rejects=m.group(7).strip(),
        )
    grpo_ok_total = sum(b.get("grpo_ok", 0) for b in batches.values())
    pool_total = sum(b.get("pool", 0) for b in batches.values())
    return {
        "n_batches": len(batches),
        "candidates_probed": pool_total,
        "grpo_ok_total": grpo_ok_total,
        "grpo_ok_rate": round(grpo_ok_total / pool_total, 4) if pool_total else None,
        "last_batch": batches[max(batches)] if batches else None,
    }


def load_accepted_signals(gpu_dir: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    filtered = gpu_dir / "filtered"
    if not filtered.exists():
        return out
    for path in filtered.glob("*.jsonl"):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                sg = obj.get("rollout_signal") or {}
                out.append({
                    "sample_id": obj.get("sample_id"),
                    "rollout_signal": sg,
                    "tier": quality_tier(sg),
                })
    return out


def load_rejected_signals(gpu_dir: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    rej = gpu_dir / "rejected"
    if not rej.exists():
        return out
    for path in rej.glob("*.jsonl"):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                reason = obj.get("reason") or obj.get("rejection_reason") or "unknown"
                sg = obj.get("rollout_signal") or (obj.get("extra") or {}).get("rollout_signal") or {}
                out.append({
                    "reason": reason,
                    "rollout_signal": sg,
                    "sub_reason": classify_grpo_fail(sg),
                    "tier": quality_tier(sg),
                })
    return out


def summarize_signals(signals: List[Dict[str, Any]], *, accepted: bool) -> Dict[str, Any]:
    if not signals:
        return {}
    sgs = [s["rollout_signal"] for s in signals if s.get("rollout_signal")]
    if not sgs:
        return {}
    fsr = [g.get("full_success_rate", 0) or 0 for g in sgs]
    means = [g.get("reward_mean", 0) or 0 for g in sgs]
    ur = Counter(g.get("unique_rewards") for g in sgs)
    tiers = Counter(s.get("tier") for s in signals)
    return {
        "n": len(sgs),
        "full_success_rate": {
            "min": min(fsr),
            "max": max(fsr),
            "avg": round(sum(fsr) / len(fsr), 4),
            "zero": sum(1 for x in fsr if x == 0),
            "partial": sum(1 for x in fsr if 0 < x < 0.999),
            "all_win": sum(1 for x in fsr if x >= 0.999),
        },
        "reward_mean": {
            "min": round(min(means), 4),
            "max": round(max(means), 4),
            "avg": round(sum(means) / len(means), 4),
        },
        "unique_rewards_dist": dict(sorted(ur.items(), key=lambda kv: str(kv[0]))),
        "tiers": dict(tiers),
        "requires_achievable_win": Counter(g.get("requires_achievable_win") for g in sgs),
    }


def analyze_run(base: Path, run_name: str) -> Dict[str, Any]:
    root = resolve_run_root(base / run_name)
    info: Dict[str, Any] = {
        "exists": root.exists(),
        "resolved_path": str(root.relative_to(base)) if root.exists() else None,
        "workers": {},
    }
    if not root.exists():
        return info

    all_accepted: List[Dict[str, Any]] = []
    all_rejected: List[Dict[str, Any]] = []
    reject_reasons = Counter()
    grpo_sub = Counter()
    log_stats: Dict[str, Any] = {}

    for gpu_dir in sorted(root.iterdir()):
        if not gpu_dir.is_dir() or not gpu_dir.name.startswith("gpu"):
            continue
        wname = gpu_dir.name
        w: Dict[str, Any] = {}

        rp = gpu_dir / "RUN_PROGRESS.json"
        if rp.exists():
            w["progress"] = json.loads(rp.read_text(encoding="utf-8"))

        w["accepted_signals"] = load_accepted_signals(gpu_dir)
        w["rejected_signals"] = load_rejected_signals(gpu_dir)
        all_accepted.extend(w["accepted_signals"])
        all_rejected.extend(w["rejected_signals"])

        logp = root / f"{wname}.log"
        w["log"] = parse_gpu_log(logp)
        log_stats[wname] = w["log"]

        for r in w["rejected_signals"]:
            reject_reasons[r["reason"]] += 1
            grpo_sub[r["sub_reason"]] += 1

        w["accepted_count"] = len(w["accepted_signals"])
        w["rejected_with_signal_count"] = len(w["rejected_signals"])
        info["workers"][wname] = w

    info["total_accepted"] = len(all_accepted)
    info["total_rejected_with_signal"] = len(all_rejected)
    info["reject_reasons"] = dict(reject_reasons.most_common())
    info["grpo_sub_reasons"] = dict(grpo_sub.most_common())
    info["accepted_summary"] = summarize_signals(all_accepted, accepted=True)
    info["rejected_summary"] = summarize_signals(all_rejected, accepted=False)

    # aggregate RUN_PROGRESS
    prog_rows = [w.get("progress") or {} for w in info["workers"].values()]
    if prog_rows:
        info["aggregate_progress"] = {
            "accepted": sum(p.get("accepted", 0) for p in prog_rows),
            "target_total": sum(p.get("target", 0) for p in prog_rows),
            "rejected_this_run": sum(p.get("rejected_this_run", 0) for p in prog_rows),
            "accept_rate_weighted": round(
                sum(p.get("accepted", 0) for p in prog_rows)
                / max(1, sum(p.get("accepted", 0) + p.get("rejected_this_run", 0) for p in prog_rows)),
                4,
            ),
            "local_weak_requests": sum(
                (p.get("client") or {}).get("local_weak_requests", 0) for p in prog_rows
            ),
            "api_spend_usd": round(
                sum((p.get("client") or {}).get("spend_usd", 0) for p in prog_rows), 4
            ),
            "top_reject_reasons": dict(
                Counter(
                    k for p in prog_rows for k in (p.get("top_reject_reasons") or {})
                ).most_common()
            ),
        }
        merged_top = Counter()
        for p in prog_rows:
            for k, v in (p.get("top_reject_reasons") or {}).items():
                merged_top[k] += v
        info["aggregate_progress"]["top_reject_reasons"] = dict(merged_top.most_common())

    # log-derived probe stats
    probed = sum((ls or {}).get("candidates_probed", 0) for ls in log_stats.values())
    grpo_ok = sum((ls or {}).get("grpo_ok_total", 0) for ls in log_stats.values())
    info["log_probe_summary"] = {
        "candidates_probed": probed,
        "grpo_ok_total": grpo_ok,
        "grpo_ok_rate": round(grpo_ok / probed, 4) if probed else None,
        "estimated_rollouts": probed * 8,
    }

    # hypothetical win=0 rescue count from rejected pool
    if all_rejected:
        rescue = sum(
            1
            for r in all_rejected
            if r["sub_reason"] == "no_full_success"
            and (r["rollout_signal"].get("unique_rewards") or 0) >= 2
            and (r["rollout_signal"].get("reward_variance") or 0) > 0
        )
        info["would_accept_if_win0_from_rejects"] = rescue

    return info


def render_markdown(report: Dict[str, Dict[str, Any]]) -> str:
    lines = [
        "# Agentic v5 — diagnostika generování",
        "",
        "Souhrn ze čtyř worker běhů (`data/agentic_workers/`).",
        "",
        "## Executive summary",
        "",
    ]

    rows = []
    for run, info in report.items():
        if not info.get("exists"):
            rows.append((run, "—", "—", "—", "missing"))
            continue
        acc = info.get("total_accepted", 0)
        agg = info.get("aggregate_progress") or {}
        rate = agg.get("accept_rate_weighted")
        rate_s = f"{100 * rate:.1f}%" if rate is not None else "n/a"
        probed = (info.get("log_probe_summary") or {}).get("candidates_probed", 0)
        grpo_rate = (info.get("log_probe_summary") or {}).get("grpo_ok_rate")
        grpo_s = f"{100 * grpo_rate:.1f}%" if grpo_rate is not None else "n/a"
        rows.append((run, str(acc), rate_s, str(probed), grpo_s))

    lines.append("| Run | Accepted | Accept rate | Kandidátů probed (log) | grpo_ok rate |")
    lines.append("|-----|----------|-------------|------------------------|--------------|")
    for r in rows:
        lines.append(f"| `{r[0]}` | {r[1]} | {r[2]} | {r[3]} | {r[4]} |")
    lines.append("")

    lines.extend([
        "### Klíčové závěry",
        "",
        "1. **Pipeline technicky funguje** — ve všech bězích `pool=5/5`, `cheap_rejects=0`; problém je až rollout gate.",
        "2. **`win1` (ROLLOUT_REQUIRE_ACHIEVABLE_WIN=1) na Stage 3** → ~3.6% accept rate, 6/200 cíl; vyžaduje alespoň jeden full win v 8 rolloutech.",
        "3. **`loose` (win=0) na Stage 3** → ~12% accept rate, 13/100 cíl; **3–4× vyšší throughput** při stejné obtížnosti.",
        "4. **Stage 2 pilot** → 44 accepted, ~7–10% rate s win=1; model zvládá 2-call úlohy častěji než 3-call.",
        "5. **Rejected JSONL s rollout vektory** existuje jen u staršího pilotu (gpu2); novější win1/loose běhy ukládají jen accepted + logy.",
        "",
    ])

    for run, info in report.items():
        lines.append(f"## {run}")
        lines.append("")
        if not info.get("exists"):
            lines.append("*adresář chybí*")
            lines.append("")
            continue

        agg = info.get("aggregate_progress") or {}
        if agg:
            lines.append("**Agregovaný progress (RUN_PROGRESS.json):**")
            lines.append(f"- accepted: **{agg.get('accepted', 0)}** / target {agg.get('target_total', '?')}")
            lines.append(f"- rejected: {agg.get('rejected_this_run', 0)}")
            lines.append(f"- accept rate: **{100 * agg.get('accept_rate_weighted', 0):.1f}%**")
            lines.append(f"- local weak rollouts (Qwen): {agg.get('local_weak_requests', 0)}")
            lines.append(f"- OpenRouter spend: ${agg.get('api_spend_usd', 0):.4f}")
            if agg.get("top_reject_reasons"):
                lines.append(f"- top rejects: `{agg['top_reject_reasons']}`")
            lines.append("")

        lps = info.get("log_probe_summary") or {}
        if lps.get("candidates_probed"):
            lines.append("**Z GPU logů:**")
            lines.append(
                f"- {lps['candidates_probed']} kandidátů × 8 rolloutů "
                f"≈ **{lps['estimated_rollouts']} Qwen episod**"
            )
            lines.append(
                f"- grpo_ok (prošlo gate): {lps['grpo_ok_total']} "
                f"({100 * (lps['grpo_ok_rate'] or 0):.1f}%)"
            )
            lines.append("")

        acc_s = info.get("accepted_summary") or {}
        if acc_s:
            fsr = acc_s.get("full_success_rate") or {}
            lines.append("**Přijaté úlohy — rollout profil:**")
            lines.append(
                f"- full_success_rate: min={fsr.get('min')} max={fsr.get('max')} "
                f"avg={fsr.get('avg')} (0%={fsr.get('zero')}, partial={fsr.get('partial')})"
            )
            lines.append(f"- unique_rewards: `{acc_s.get('unique_rewards_dist')}`")
            lines.append(f"- quality tiers: `{acc_s.get('tiers')}`")
            lines.append("")

        rej_s = info.get("rejected_summary") or {}
        if rej_s:
            lines.append("**Odmítnuté s rollout_signal (JSONL):**")
            lines.append(f"- n={rej_s.get('n')}")
            if info.get("grpo_sub_reasons"):
                lines.append(f"- GRPO sub-důvody: `{info['grpo_sub_reasons']}`")
            fsr = rej_s.get("full_success_rate") or {}
            lines.append(
                f"- full_success: 0%={fsr.get('zero')}, partial={fsr.get('partial')}, "
                f"100%={fsr.get('all_win')}"
            )
            if info.get("would_accept_if_win0_from_rejects"):
                lines.append(
                    f"- **would pass if win=0:** {info['would_accept_if_win0_from_rejects']} "
                    f"z {rej_s.get('n')} rejectů"
                )
            lines.append("")

        for wname, w in sorted(info.get("workers", {}).items()):
            prog = w.get("progress") or {}
            lines.append(f"### Worker `{wname}`")
            if prog:
                lines.append(
                    f"- progress: {prog.get('accepted', 0)}/{prog.get('target', '?')} "
                    f"(iter {prog.get('iteration', '?')}, status={prog.get('status', '?')})"
                )
            log = w.get("log") or {}
            if log.get("last_batch"):
                lb = log["last_batch"]
                lines.append(
                    f"- last batch: grpo_ok={lb.get('grpo_ok')}/{lb.get('pool')}, "
                    f"rate={lb.get('rate')}, rejects=`{lb.get('batch_rejects', '')}`"
                )
            if w.get("accepted_count"):
                lines.append(f"- accepted: {w['accepted_count']}")
            lines.append("")

    lines.extend([
        "## Doporučení (aligned s GPT analýzou)",
        "",
        "| Akce | Priorita |",
        "|------|----------|",
        "| Zastavit dlouhé běhy na 50/GPU s `win1` | okamžitě |",
        "| Stage 3 generovat s `ROLLOUT_REQUIRE_ACHIEVABLE_WIN=0` | vysoká |",
        "| Ukládat rejected JSONL + rollout vektory u všech běhů | vysoká |",
        "| Rozdělit `low_grpo_signal_prediction` na sub-důvody v kódu | střední |",
        "| Diagnostický pilot 20–30 kandidátů offline | střední |",
        "| vLLM batched screening (4 rollout cheap → 8 rollout top-k) | později |",
        "| Nepřecházet na Stage 4 dokud Stage 3 win=0 nedá objem | vysoká |",
        "",
        "### Odhad nákladů (win1, Stage 3)",
        "",
        "Při ~3.6% accept a 8 rolloutech/kandidát:",
        "- 50 accepted ≈ **~1400 kandidátů** ≈ **~11 000 Qwen rolloutů** na GPU",
        "- 4 GPU × 50 = 200 accepted → **~44 000 rolloutů** celkem",
        "",
        "S `win=0` (~12% accept): stejný cíl **~3× levněji**.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_BASE
    runs = sys.argv[2:] if len(sys.argv) > 2 else DEFAULT_RUNS
    report = {run: analyze_run(base, run) for run in runs}

    out_json = ROOT / "reports" / "AGENTIC_V5_GENERATION_DIAGNOSTIC.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    out_md = ROOT / "reports" / "AGENTIC_V5_GENERATION_DIAGNOSTIC.md"
    out_md.write_text(render_markdown(report), encoding="utf-8")

    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
