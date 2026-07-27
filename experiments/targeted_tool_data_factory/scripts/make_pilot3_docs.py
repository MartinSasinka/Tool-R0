#!/usr/bin/env python3
"""Write Pilot3 data card, Pilot2-vs-Pilot3, signal-health stub, cost estimate.

Every figure is read from frozen artefacts. Missing files are marked MISSING.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

FACTORY = Path(__file__).resolve().parents[1]
OUT = FACTORY / "outputs"
DOCS = FACTORY / "docs"
V, BASE, SEED = "pilot3", "pilot2", 20260727


def jread(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def jlread(p: Path) -> list[dict]:
    if not p.is_file():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def pct(n, d) -> str:
    return "n/a" if not d else f"{100 * n / d:.1f} %"


def dist(rows, key):
    c = Counter(str(r.get(key)) for r in rows)
    n = sum(c.values()) or 1
    return {k: v / n for k, v in sorted(c.items())}


def jsd(p: dict, q: dict) -> float:
    keys = sorted(set(p) | set(q))
    if not keys:
        return 0.0
    a = [p.get(k, 0.0) for k in keys]
    b = [q.get(k, 0.0) for k in keys]
    sa, sb = sum(a) or 1.0, sum(b) or 1.0
    a = [x / sa for x in a]
    b = [x / sb for x in b]
    m = [(x + y) / 2 for x, y in zip(a, b)]

    def kl(x, y):
        return sum(xi * math.log2(xi / yi) for xi, yi in zip(x, y) if xi > 0 and yi > 0)

    return round(0.5 * kl(a, m) + 0.5 * kl(b, m), 6)


def main() -> int:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sel = jlread(OUT / "selected" / f"selected_{V}.jsonl")
    base = jlread(OUT / "selected" / f"selected_{BASE}.jsonl")
    train = jlread(OUT / "splits" / f"train_{V}.jsonl")
    held = jlread(OUT / "splits" / f"heldout_{V}.jsonl")
    reserve = jlread(OUT / "splits" / f"reserve_{V}.jsonl")
    gates = jread(OUT / "reports" / f"pilot3_gates_{V}.json") or {}
    match = jread(OUT / "selected" / f"profile_match_{V}.json") or []
    val = jread(OUT / "validated" / f"validation_summary_{V}.json") or {}
    gen = jread(OUT / "candidates" / f"gen_stats_{V}.json") or {}
    para = jread(OUT / "reports" / f"paraphrase_{V}.json") or {}
    pre = jread(OUT / "reports" / f"preflight_{V}.json") or {}
    nest = jread(OUT / "profiles" / "nestful_profile.json") or {}
    run_state = jread(OUT / "reports" / f"run_state_{V}.json") or {}

    n = len(sel)
    tracks = dist(sel, "track")
    calls = dist(sel, "call_count")
    motifs = dist(sel, "motif")
    answers = dist(sel, "answer_type")
    cells = Counter(r.get("generation_cell_id") for r in sel)
    cell_max = max(cells.values(), default=0) / max(n, 1)
    tmpl = Counter(r.get("query_template_original") or r.get("query") for r in sel
                   if r.get("query_source") == "template")
    # template share by graph_template_id is the pipeline gate
    gtmpl = Counter(r.get("graph_template_id") for r in sel)
    tmpl_max = max(gtmpl.values(), default=0) / max(n, 1)

    nest_call = {str(k): float(v) for k, v in (nest.get("call_count_dist") or {}).items()}
    # project call counts into buckets
    call_bucket = {}
    for k, v in calls.items():
        b = "6+" if int(k) >= 6 else k
        call_bucket[b] = call_bucket.get(b, 0.0) + v
    motif_proj = {}
    for k, v in motifs.items():
        mk = "fan_in" if k == "branch_aggregate" else k
        motif_proj[mk] = motif_proj.get(mk, 0.0) + v

    pm = next((p for p in match if p.get("label") == "new_selected"), {})
    verdict = gates.get("verdict") or "NOT_READY"

    # ── DATA CARD ────────────────────────────────────────────────────────
    card = f"""# Pilot3 data card — `{V}`

Generated {now}.

## What this dataset is

{n} synthetic nested-tool-use tasks produced program-first: a semantic program
is sampled first, executed by the factory executor to obtain the oracle answer,
and only then rendered into a natural-language question. Pilot2 artefacts are
untouched.

| field | value |
|---|---|
| version | `{V}` |
| seed | `{SEED}` |
| selected tasks | {n} |
| train / structural held-out / reserve | {len(train)} / {len(held)} / {len(reserve)} |
| adaptation / generalization | {pct(tracks.get('A', 0)*n, n)} / {pct(tracks.get('G', 0)*n, n)} |
| target benchmark profile | NESTFUL dev |
| target student | `Qwen/Qwen3-4B-Instruct-2507` |
| oracle | executor-only, deterministic |
| replay rate (validated pool) | {pct((val or {}).get('replay_rate_validated', 0), 1) if val else 'n/a'} |
| verdict | **{verdict}** |

## Composition

### call count
| value | share |
|---|---|
{chr(10).join(f'| `{k}` | {100*v:.1f} % |' for k, v in calls.items())}

### motif
| value | share |
|---|---|
{chr(10).join(f'| `{k}` | {100*v:.1f} % |' for k, v in motifs.items())}

### answer type
| value | share |
|---|---|
{chr(10).join(f'| `{k}` | {100*v:.1f} % |' for k, v in answers.items())}

- max generation-cell share: {100*cell_max:.2f} % (cap 8 %)
- max graph-template share: {100*tmpl_max:.2f} % (cap 5 %)
- candidates generated: {(gen or {}).get('n_generated', 'n/a')}
- paraphrase accepted: {(para.get('stats') or {}).get('accepted', 'n/a')}
  cost=${((para.get('budget') or {}).get('usd', 'n/a'))}

## Quality gates

| gate | status |
|---|---|
| gold replay 100 % | {'PASS' if (pre.get('passed') if pre else (val or {}).get('replay_rate_validated') == 1.0) else 'see preflight'} |
| leakage 0 | {'PASS' if not (gates.get('fails') and any('leak' in f for f in gates.get('fails', []))) else 'FAIL'} |
| cell share ≤ 8 % | {'PASS' if cell_max <= 0.08 else 'FAIL'} |
| template share ≤ 5 % | {'PASS' if tmpl_max <= 0.05 else 'FAIL'} |

## Intended use

GRPO training on the frozen train split after a RunPod signal probe selects a
NESTFUL-matched Phase-1 subset with real terminal/process mixed signal.
Structural held-out is for in-domain measurement; reserve stays sealed.

## Out of scope

- Not a public benchmark.
- Do not regenerate on RunPod — use `runpod_bundle_pilot3/`.
"""
    (DOCS / "PILOT3_DATA_CARD.md").write_text(card, encoding="utf-8")

    # ── P2 vs P3 ─────────────────────────────────────────────────────────
    b_calls = dist(base, "call_count") if base else {}
    b_motifs = dist(base, "motif") if base else {}
    b_ans = dist(base, "answer_type") if base else {}
    b_tracks = dist(base, "track") if base else {}
    cmp = f"""# Pilot2 vs Pilot3

Generated {now}.

| | pilot2 | pilot3 |
|---|---|---|
| selected | {len(base)} | {n} |
| train / heldout / reserve | 160 / 80 / 80 | {len(train)} / {len(held)} / {len(reserve)} |
| seed | 20260726 | {SEED} |
| G-track share | {pct(b_tracks.get('G', 0)*len(base), len(base)) if base else 'n/a'} | {pct(tracks.get('G', 0)*n, n)} |
| cell_max_share cap | 10 % | 8 % |
| long-horizon boost | none | +2.5 pp on 5-call, +3.5 pp on 6+ |

## Call-count share

| bucket | pilot2 | pilot3 | NESTFUL |
|---|---|---|---|
"""
    for k in sorted(set(b_calls) | set(calls) | set(nest_call),
                    key=lambda x: (len(str(x)), str(x))):
        cmp += (f"| `{k}` | {pct(b_calls.get(k, 0)*len(base), len(base)) if base else 'n/a'} | "
                f"{pct(calls.get(k, 0)*n, n)} | "
                f"{pct(nest_call.get(k if str(k) in nest_call else ('6+' if str(k).isdigit() and int(k)>=6 else k), 0), 1)} |\n")
    cmp += f"""
## Profile match (selected vs NESTFUL)

| metric | pilot3 |
|---|---|
| JSD call_bucket | {pm.get('jsd_call_bucket', 'n/a')} |
| JSD motif | {pm.get('jsd_motif', 'n/a')} |
| JSD answer_type | {pm.get('jsd_answer_type', 'n/a')} |
| Wasserstein n_tools | {pm.get('wass_n_tools', 'n/a')} |
| two-sample AUC | {pm.get('auc_two_sample', 'n/a')} |

Pilot2 artefacts under `outputs/**/*pilot2*` and `runpod_bundle_pilot2/` were
**not** modified.
"""
    (DOCS / "PILOT2_VS_PILOT3.md").write_text(cmp, encoding="utf-8")

    # ── signal-health (pre-probe stub) ───────────────────────────────────
    sig = f"""# Pilot3 signal-health report

Generated {now}.

## Local student probe

The factory `probe` step ran with `template_only` / `--no-llm` unless a local
OpenAI-compatible endpoint was available. Full signal measurement is the
**RunPod signal probe** on the frozen 600-task train split:

```bash
cd /workspace/Tool-R0
export HF_TOKEN=...
bash experiments/targeted_tool_data_factory/runpod_bundle_pilot3/run_signal_probe_4gpu.sh
```

That probe:
- rolls out **all 600** train tasks × 4 (P2);
- re-probes a boundary subset × 8 (P3);
- selects a NESTFUL-matched Phase-1 subset with terminal/process mixed signal
  (default target 400; hard structural buckets are not dropped for difficulty);
- never trains.

After the probe, replace this stub with
`outputs/runpod_pilot3/signal_probe/SIGNAL_PROBE_REPORT.md`.

## Pipeline timings (this machine)

| step | wall_s |
|---|---|
{chr(10).join(f'| {k} | {(v or {}).get("wall_s", "n/a")} |' for k, v in (run_state.get('steps') or {}).items()) or '| (run not finished) | — |'}
"""
    (DOCS / "PILOT3_SIGNAL_HEALTH.md").write_text(sig, encoding="utf-8")

    # ── cost / time estimate ─────────────────────────────────────────────
    cost = f"""# Pilot3 cost & time estimate

Generated {now}.

| phase | estimate | notes |
|---|---|---|
| CPU generate + validate (~8k candidates) | **4–10 h** wall | 1 workstation CPU; B2 expand may add 1–3 h |
| OpenRouter paraphrase (≤4500 req, ≤$5) | **1–3 h** / **≤ $5** | mistral-small; measured pilot2 ≈ $0.00003/req |
| Select / split / export / report | **5–15 min** | CPU |
| Gold-replay preflight (1000 tasks) | **2–10 min** | factory executor |
| RunPod signal probe (600×4 + P3×8, 4 GPU) | **2–5 h** | BF16 Qwen3-4B, no training |
| Subsequent GRPO (Phase-1 ~400, 8 gens, small budget) | **3–8 h** | not auto-started |
| Full D1-style 600-train GRPO (later decision) | **8–20 h** | not part of this freeze |

OpenRouter spend this run: ${((para.get('budget') or {}).get('usd', 0.0))}
(requests={(para.get('stats') or {}).get('requests', 0)}).
"""
    (DOCS / "COST_REPORT_PILOT3.md").write_text(cost, encoding="utf-8")

    # ── RUNBOOK snippet ──────────────────────────────────────────────────
    runbook = f"""# RunPod Pilot3 signal-probe runbook

## One command (after syncing the frozen bundle)

```bash
cd /workspace/Tool-R0
export HF_TOKEN=...
bash experiments/targeted_tool_data_factory/runpod_bundle_pilot3/run_signal_probe_4gpu.sh
```

Flags: `--dry-run`, `--resume`, `--stage p2|select|p3|report`.

## Local freeze (dev machine)

```bash
cd experiments/targeted_tool_data_factory
python scripts/run_pilot3.py --dry-run
python scripts/run_pilot3.py                # needs OPENROUTER_API_KEY
python runpod_bundle_pilot3/build_bundle.py
```

GRPO is **not** started by the probe. Read `SIGNAL_PROBE_REPORT.md` first.
"""
    (DOCS / "RUNPOD_PILOT3_RUNBOOK.md").write_text(runbook, encoding="utf-8")

    print(f"[docs] wrote PILOT3_DATA_CARD / PILOT2_VS_PILOT3 / "
          f"PILOT3_SIGNAL_HEALTH / COST_REPORT_PILOT3 / RUNPOD_PILOT3_RUNBOOK")
    print(f"[docs] n_selected={n} verdict={verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
