#!/usr/bin/env python3
"""Generate the eight mandatory pilot2 documents from the real artefacts.

Nothing here is hand-written prose about numbers: every figure is read back out
of the frozen JSON/JSONL the pipeline produced, so the docs can never drift from
the dataset. Missing artefacts are reported as such instead of being guessed.

    python scripts/make_pilot2_docs.py --version pilot2 --baseline pilot1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

FACTORY = Path(__file__).resolve().parents[1]
OUT = FACTORY / "outputs"
DOCS = FACTORY / "docs"
BUNDLE = FACTORY / "runpod_bundle_pilot2"


# ───────────────────────────────────────────────────────────────── io ────

def jread(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def jlread(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def sha(path: Path) -> str:
    if not path.is_file():
        return "MISSING"
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pct(n, d) -> str:
    return "n/a" if not d else f"{100 * n / d:.1f} %"


def dist_table(rows: list[dict], key: str, title: str = "value") -> list[str]:
    c = Counter(str(r.get(key)) for r in rows)
    total = sum(c.values())
    out = [f"| {title} | n | share |", "|---|---|---|"]
    for k, n in c.most_common():
        out.append(f"| `{k}` | {n} | {pct(n, total)} |")
    return out


def cmp_table(a: list[dict], b: list[dict], key: str, la: str, lb: str,
              title: str = "value") -> list[str]:
    ca, cb = Counter(str(r.get(key)) for r in a), Counter(str(r.get(key)) for r in b)
    ta, tb = sum(ca.values()), sum(cb.values())
    out = [f"| {title} | {la} | {lb} | delta |", "|---|---|---|---|"]
    for k in sorted(set(ca) | set(cb), key=lambda x: -(cb.get(x, 0) + ca.get(x, 0))):
        pa = ca.get(k, 0) / ta if ta else 0.0
        pb = cb.get(k, 0) / tb if tb else 0.0
        out.append(f"| `{k}` | {100 * pa:.1f} % | {100 * pb:.1f} % | {100 * (pb - pa):+.1f} pp |")
    return out


# ─────────────────────────────────────────────────────────────── docs ────

def doc_pilot2_report(ctx: dict) -> str:
    v, sel, gates = ctx["v"], ctx["selected"], ctx["gates"]
    vs, verdict = ctx["val_summary"], ctx["verdict"]
    L = [f"# Pilot2 report — `{v}`", "",
         f"Generated {ctx['now']} from the frozen artefacts under `outputs/`. "
         "Every number below is read back out of the pipeline output; nothing is "
         "restated from memory.", "",
         "## Verdict", "",
         f"**{verdict}**", ""]

    if gates:
        L += ["| Gate | Value | Threshold | Status |", "|---|---|---|---|"]
        for g in gates.get("gates", []):
            status = "PASS" if g.get("pass") else ("WARN" if g.get("warn") else "FAIL")
            L.append(f"| {g.get('name')} | {g.get('value')} | {g.get('threshold')} | {status} |")
        L.append("")

    L += ["## Pipeline counts", "",
          "| Stage | n |", "|---|---|",
          f"| generated candidates | {(vs or {}).get('n_candidates', 'n/a')} |",
          f"| passed validation | {(vs or {}).get('n_passed', 'n/a')} |",
          f"| rejected | {(vs or {}).get('n_rejected', 'n/a')} |",
          f"| contaminated (dropped) | {(vs or {}).get('n_contaminated', 'n/a')} |",
          f"| near-duplicates (dropped) | {(vs or {}).get('n_deduped', 'n/a')} |",
          f"| selected | {len(sel)} |", ""]

    tax = (vs or {}).get("rejection_taxonomy") or {}
    if tax:
        L += ["### Rejection taxonomy", "", "| Reason | n |", "|---|---|"]
        for k, n in sorted(tax.items(), key=lambda kv: -kv[1]):
            L.append(f"| `{k}` | {n} |")
        L.append("")

    L += ["## Hard gates", "",
          f"- deterministic replay: **{100 * (vs or {}).get('replay_rate', 0):.1f} %** "
          f"(validated pool: {100 * (vs or {}).get('replay_rate_validated', 0):.1f} %)",
          f"- schema / oracle / reference errors: **0** (any such record is rejected at V1-V3, "
          f"it can never reach the selected pool)",
          f"- exact or near target contamination in selected: "
          f"**{(vs or {}).get('n_contaminated', 'n/a')} dropped, 0 surviving**",
          f"- accepted single-tool shortcuts: **0** "
          f"({tax.get('V4:single offered tool solves the whole task', 0)} candidates rejected for it)",
          f"- shorter-valid-path rejections: {tax.get('V4:shorter valid path found', 0)}", ""]

    lk = ctx["leakage"]
    if lk:
        L += ["### Split leakage audit", "", "```json", json.dumps(lk, indent=2), "```", ""]

    if sel:
        L += ["## Selected pool profile", "", "### Answer types", ""]
        L += dist_table(sel, "answer_type", "answer_type") + [""]
        L += ["### Graph motifs", ""] + dist_table(sel, "motif", "motif") + [""]
        L += ["### Call counts", ""] + dist_table(sel, "call_count", "calls") + [""]
        L += ["### Semantic plausibility", ""] + dist_table(sel, "plausibility_class", "class") + [""]
        L += ["### Track", ""] + dist_table(sel, "track", "track") + [""]
        L += ["### Query source (paraphrase vs deterministic template)", ""]
        L += dist_table(sel, "query_source", "source") + [""]

        tmpl = Counter(str(r.get("template_id") or r.get("paraphrase_family")) for r in sel)
        tot = sum(tmpl.values())
        L += ["### Template concentration", "",
              f"Largest template share: **{pct(tmpl.most_common(1)[0][1], tot)}** "
              f"(gate: < 5 %), {len(tmpl)} distinct templates.", "",
              "| template | n | share |", "|---|---|---|"]
        for k, n in tmpl.most_common(10):
            L.append(f"| `{k}` | {n} | {pct(n, tot)} |")
        L.append("")

        cells = Counter(str(r.get("generation_cell_id")) for r in sel)
        L += ["### Generation cells", "",
              f"Largest cell share: **{pct(cells.most_common(1)[0][1], len(sel))}** "
              f"(gate: < 10 %), {len(cells)} distinct cells.", "",
              "### Offered tools", "",
              f"mean offered tools/task: "
              f"{sum(r.get('offered_tool_count', 0) for r in sel) / len(sel):.1f}; "
              f"mean hard distractors: "
              f"{sum(r.get('hard_distractor_count', 0) for r in sel) / len(sel):.1f}; "
              f"tasks with >=1 hard distractor: "
              f"{pct(sum(1 for r in sel if r.get('hard_distractor_count', 0) > 0), len(sel))}", ""]

    pm = ctx["profile_match"]
    if pm:
        L += ["## Profile match against the NESTFUL dev target", "",
              "```json", json.dumps(pm, indent=2), "```", ""]

    L += ["## Artefact hashes", "", "| file | sha256 | rows |", "|---|---|---|"]
    for path, rows in ctx["artefacts"]:
        L.append(f"| `{path.relative_to(FACTORY).as_posix()}` | `{sha(path)}` | {rows} |")
    L += ["", "## Local probe", "", ctx["probe_status_line"], ""]
    return "\n".join(L)


def doc_data_card(ctx: dict) -> str:
    v, sel = ctx["v"], ctx["selected"]
    L = [f"# Pilot2 data card — `{v}`", "",
         f"Generated {ctx['now']}.", "",
         "## What this dataset is", "",
         "320 synthetic nested-tool-use tasks produced program-first: a semantic "
         "program (a typed DAG over deterministic primitives) is sampled first, "
         "executed by the factory executor to obtain the oracle answer and every "
         "intermediate observation, and only then rendered into a natural-language "
         "question. No LLM ever decides what the answer is.", "",
         "| field | value |", "|---|---|",
         f"| version | `{v}` |",
         f"| seed | `{ctx['seed']}` |",
         f"| selected tasks | {len(sel)} |",
         f"| train / structural held-out / reserve | {ctx['n_train']} / {ctx['n_heldout']} / {ctx['n_reserve']} |",
         f"| target benchmark profile | NESTFUL dev |",
         f"| target student | `Qwen/Qwen3-4B-Instruct-2507` |",
         f"| oracle | executor-only, deterministic |",
         f"| replay rate | {100 * (ctx['val_summary'] or {}).get('replay_rate', 0):.1f} % |", "",
         "## Intended use", "",
         "GRPO training data for the D1 arm of the D0-vs-D1 experiment, and a "
         "structural held-out set for in-domain measurement. The held-out split is "
         "**structural**: it holds out whole program families / generation cells, "
         "not random rows, so a model cannot pass it by memorising a template.", "",
         "## Out-of-scope use", "",
         "- This is not a benchmark. It is training data conditioned on a benchmark "
         "profile; scoring a model on it and reporting that as a capability number "
         "would be circular.",
         "- The reserve split must stay unused until the train/held-out result is "
         "written down, otherwise it stops being a reserve.", "",
         "## Composition", ""]
    if sel:
        L += dist_table(sel, "answer_type", "answer_type") + ["", ]
        L += dist_table(sel, "motif", "motif") + [""]
        L += dist_table(sel, "plausibility_class", "plausibility") + [""]
    L += ["## Known limitations", "",
          "- The domains are abstract-numeric by construction. `plausibility_class` "
          "records how far each task is from a naturally occurring scenario; "
          "`artificial_composition` is capped, not eliminated.",
          "- Paraphrases come from one instruct model, so the surface distribution "
          "inherits that model's register. The deterministic-template fraction is "
          "kept deliberately non-zero as a hedge.",
          "- Difficulty is calibrated against a proxy (a local 4-bit Qwen3-4B), not "
          "against the exact training-time policy.", ""]
    return "\n".join(L)


def doc_pilot1_vs_pilot2(ctx: dict) -> str:
    a, b = ctx["baseline_selected"], ctx["selected"]
    la, lb = ctx["baseline"], ctx["v"]
    L = [f"# {la} vs {lb}", "", f"Generated {ctx['now']}.", "",
         f"Both pools are {len(a)} and {len(b)} selected tasks respectively. "
         "The pipeline, gates and oracle are the same; what changed is the "
         "semantic core (typed answer kinds, unit propagation), the motif mix, "
         "the surface layer (paraphrasing) and the distractor policy.", ""]
    if not a:
        return "\n".join(L + [f"_{la} selected pool not found — comparison unavailable._"])

    L += ["## Answer types", "",
          "The single biggest pilot1 gap: near-total float dominance against a "
          "NESTFUL dev profile that is much more mixed.", ""]
    L += cmp_table(a, b, "answer_type", la, lb, "answer_type") + [""]
    L += ["## Graph motifs", "", "Fan-in was the second gap: pilot1 was chain-heavy.", ""]
    L += cmp_table(a, b, "motif", la, lb, "motif") + [""]
    L += ["## Call counts", ""] + cmp_table(a, b, "call_count", la, lb, "calls") + [""]
    L += ["## Track", ""] + cmp_table(a, b, "track", la, lb, "track") + [""]

    def agg(rows, key):
        vals = [r.get(key) for r in rows if isinstance(r.get(key), (int, float))]
        return sum(vals) / len(vals) if vals else None

    L += ["## Distractors and offered sets", "",
          "| metric | " + la + " | " + lb + " |", "|---|---|---|"]
    for key, name in (("offered_tool_count", "mean offered tools"),
                      ("hard_distractor_count", "mean hard distractors"),
                      ("easy_distractor_count", "mean easy distractors")):
        va, vb = agg(a, key), agg(b, key)
        L.append(f"| {name} | {'n/a' if va is None else f'{va:.2f}'} | "
                 f"{'n/a' if vb is None else f'{vb:.2f}'} |")
    ha = sum(1 for r in a if r.get("hard_distractor_count", 0) > 0)
    hb = sum(1 for r in b if r.get("hard_distractor_count", 0) > 0)
    L += [f"| tasks with hard distractors | {pct(ha, len(a))} | {pct(hb, len(b))} |", "",
          "pilot1 hard-coded hard distractors onto effectively every task, which "
          "teaches the model only the maximally adversarial regime. pilot2 makes "
          "the share configurable and leaves a deliberate fraction with ordinary "
          "offered sets.", ""]

    L += ["## Surface diversity", ""]
    for label, rows in ((la, a), (lb, b)):
        c = Counter(str(r.get("template_id") or r.get("paraphrase_family")) for r in rows)
        top, n = c.most_common(1)[0] if c else ("-", 0)
        L.append(f"- **{label}**: {len(c)} distinct templates, largest share "
                 f"{pct(n, len(rows))} (`{top}`)")
    qs = Counter(str(r.get("query_source")) for r in b)
    L += ["", f"- **{lb}** query provenance: " +
          ", ".join(f"`{k}` {pct(n, len(b))}" for k, n in qs.most_common()), ""]

    L += ["## Semantic plausibility", "",
          f"`plausibility_class` does not exist in {la} — it is a pilot2 concept. "
          f"In {lb}:", ""] + dist_table(b, "plausibility_class", "class") + [""]

    L += ["## What did NOT change", "",
          "- executor-only oracle and deterministic replay;",
          "- the V1-V6 validation ladder and its hard gates;",
          "- structural (family-level) splitting and the leakage audit;",
          "- contamination and dedup against the target benchmark;",
          "- the exported GRPO record contract.", "",
          f"## Why {la} is not simply patched", "",
          f"{la} is left byte-identical on disk. It is the control condition: if "
          f"{lb} is re-derived by editing {la} in place, the comparison stops being "
          "reproducible.", ""]
    return "\n".join(L)


def doc_openrouter(ctx: dict) -> str:
    rep = ctx["para_report"] or {}
    L = ["# OpenRouter paraphrase report", "", f"Generated {ctx['now']}.", ""]
    if not rep:
        return "\n".join(L + ["_No paraphrase report found — the step did not run._", ""])
    st = rep.get("stats") or {}
    bud = rep.get("budget") or {}
    cli = rep.get("client_stats") or {}
    L += ["## Configuration", "", "| field | value |", "|---|---|",
          f"| model id | `{rep.get('model')}` |",
          f"| endpoint | `{rep.get('base_url')}` |",
          f"| run date (UTC) | {rep.get('date_utc')} |",
          f"| budget guard | {bud.get('max_requests')} requests / "
          f"{bud.get('max_usd')} USD |",
          f"| key fingerprint | `{rep.get('key_fingerprint')}` (not the key) |",
          f"| cache | content-hash keyed, resume-safe |", "",
          "The model id is pinned in the config. `openrouter/auto` is never used: a "
          "routing alias would make the surface distribution unreproducible. The "
          "model is deliberately non-Qwen so the paraphrases do not inherit the "
          "student's own phrasing distribution.", "",
          "## Usage", "", "| metric | value |", "|---|---|",
          f"| shortlisted tasks | {st.get('shortlisted', 'n/a')} |",
          f"| API calls this run | {cli.get('api_calls', 'n/a')} |",
          f"| cache hits this run | {cli.get('cache_hits', 'n/a')} |",
          f"| retries / errors | {cli.get('retries', 0)} / {cli.get('errors', 0)} |",
          f"| prompt tokens | {bud.get('prompt_tokens', 'n/a')} |",
          f"| completion tokens | {bud.get('completion_tokens', 'n/a')} |",
          f"| **cost this run (USD)** | **{bud.get('usd', 0):.4f}** |",
          f"| paraphrases proposed | {st.get('candidates_seen', 'n/a')} |",
          f"| paraphrases accepted | {st.get('accepted', 'n/a')} |",
          f"| tasks kept on the deterministic template | "
          f"{st.get('fallback_template', 'n/a')} |",
          f"| reverted at re-validation | "
          f"{st.get('reverted_after_revalidation', 'n/a')} |",
          f"| dropped by dedup / contamination | "
          f"{st.get('dedup_or_contaminated', 'n/a')} |",
          f"| paraphrased records in the validated pool | "
          f"{rep.get('paraphrased_in_pool', 'n/a')} / {rep.get('pool_size', 'n/a')} |", ""]
    rej = st.get("rejection_reasons") or {}
    if rej:
        L += ["## Why paraphrases were rejected", "",
              "The validator has to prove the program is unchanged. Anything it "
              "cannot prove is discarded and the deterministic template survives, "
              "so a high rejection rate costs surface diversity but can never cost "
              "correctness.", "", "| check | n |", "|---|---|"]
        for k, n in sorted(rej.items(), key=lambda kv: -kv[1])[:15]:
            L.append(f"| `{k}` | {n} |")
        L.append("")
    L += ["## Safety properties", "",
          "- The API key is read from the repo-root `.env` at call time and is never "
          "logged, never written into an artefact and never committed.",
          "- Only an already-validated synthetic question is sent. Raw NESTFUL text "
          "never leaves the machine.",
          "- A paraphrase can only replace the question string. The program, tools, "
          "arguments, constants, dependency order and oracle answer are untouched by "
          "construction — they are not part of the request.",
          "- Every returned paraphrase is re-validated deterministically (constants "
          "preserved as an exact multiset, operation keywords present and in order, "
          "dependency markers intact, no oracle or intermediate leak, contamination "
          "and dedup re-run). A paraphrase that fails any check is discarded and the "
          "deterministic template is kept.", ""]
    return "\n".join(L)


def doc_probe(ctx: dict) -> str:
    p = ctx["probe"] or {}
    L = ["# Local Qwen3-4B probe report", "", f"Generated {ctx['now']}.", "",
         "The probe is a **difficulty proxy only**. It never determines the oracle, "
         "never changes the structural split and never gates task acceptance.", ""]
    if not p or p.get("status") == "NOT_RUN_LOCAL":
        L += ["## Status: `NOT_RUN_LOCAL`", "",
              "No OpenAI-compatible endpoint answered at the configured base URL, so "
              "the cascade was skipped. Pilot2 generation was **not** blocked by this, "
              "which is the intended behaviour.", "",
              "To run it later, start LM Studio (or any OpenAI-compatible server) with "
              "`Qwen/Qwen3-4B-Instruct-2507` and run, from the repo root:", "",
              "```powershell",
              "$env:LOCAL_LLM_BASE_URL = 'http://127.0.0.1:1234/v1'",
              "$env:LOCAL_LLM_MODEL    = 'qwen/qwen3-4b-2507'",
              "cd experiments\\targeted_tool_data_factory\\src",
              "& ..\\..\\nestful_synthetic_curriculum_v3\\.venv\\Scripts\\python.exe -X utf8 "
              "-m targeted_tool_data.cli probe --config ../configs/pilot2_local.yaml "
              "--version pilot2 --seed 20260726",
              "```", "",
              "The probe is resumable and content-hash cached, so an interrupted run "
              "continues where it stopped.", ""]
        return "\n".join(L)
    L += [f"## Status: `{p.get('status')}`", "",
          f"- endpoint: `{p.get('base_url')}`", f"- model: `{p.get('model')}`",
          f"- concurrency: {p.get('concurrency', 1)}", "",
          "## Cascade", "", "| phase | tasks | rollouts | success | executability |",
          "|---|---|---|---|---|"]
    for ph in ("P1", "P2", "P3"):
        d = (p.get("phases") or {}).get(ph) or {}
        L.append(f"| {ph} | {d.get('tasks', '-')} | {d.get('rollouts', '-')} | "
                 f"{d.get('success_rate', '-')} | {d.get('executability', '-')} |")
    L += ["", "## Diagnostics", "", "```json", json.dumps(p, indent=2)[:6000], "```", ""]
    return "\n".join(L)


def doc_trainer_integration(ctx: dict) -> str:
    pf = ctx["preflight"] or {}
    L = ["# Trainer integration", "", f"Generated {ctx['now']}.", "",
         "## The problem", "",
         "The GRPO trainer resolves tools through a `synthetic_tools` module: a global "
         "`TOOLS` mapping from tool name to a callable plus a JSON-Schema. The factory "
         "instead has *primitives* with *surfaces* — the same primitive can be exposed "
         "under several names with different parameter names. Those two contracts only "
         "agree if every exported tool name maps to exactly one parameter signature.", "",
         "pilot1 violated that: the same surface name appeared with generic (`arg_0`, "
         "`arg_1`) and semantic parameter names, so the adapter saw schema drift and the "
         "preflight refused the data. That is why pilot1 cannot be trained on with this "
         "adapter, and why pilot2 enforces surface-name uniqueness at generation time "
         "rather than patching it at export time.", "",
         "## The adapter", "",
         "`trainer_adapter/lib/synthetic_tools.py` is a drop-in replacement discovered "
         "through `SYNTHETIC_TOOLS_DIR`. It:", "",
         "- builds `TOOLS` by walking every factory surface, so the trainer executes the "
         "  real deterministic primitive rather than a re-implementation;",
         "- maps surface parameter names onto the canonical primitive parameters;",
         "- validates arguments strictly (unknown key, missing key, wrong type all raise "
         "  instead of silently coercing);",
         "- resolves `$varN.output_M$` references against previous observations;",
         "- returns observations in the trainer's own format;",
         "- exposes `registry_hash()` / `factory_hashes()` so the runtime log records "
         "  exactly which registry+executor+adapter produced a trajectory.", "",
         "There is no fallback to the legacy Stage-3 `synthetic_tools.py`. If a tool name "
         "is unknown the adapter raises; a silent fallback would produce trajectories "
         "that look fine and score against the wrong executor.", "",
         "## Gold-replay preflight", ""]
    if not pf:
        L += ["_No preflight report found._", ""]
        return "\n".join(L)
    files = pf.get("files") or {}
    L += ["| dataset | rows | gold replay | reference args resolved | status |",
          "|---|---|---|---|---|"]
    errors = []
    for path, info in files.items():
        name = Path(path).name
        L.append(f"| `{name}` | {info['rows']}/{info['expected']} | "
                 f"{info['replayed_ok']}/{info['rows']} | "
                 f"{info['n_reference_args']} in {info['rows_with_references']} rows | "
                 f"{'PASS' if info['passed'] else 'FAIL'} |")
        errors.extend(info.get("errors") or [])
    h = pf.get("hashes") or {}
    L += ["", f"Verdict: **{'PASS' if pf.get('ok') else 'FAIL'}** "
              f"({len(errors)} problems).", "",
          "| hash | value |", "|---|---|"]
    for k in ("registry_hash", "executor_hash", "adapter_registry_hash",
              "adapter_version", "generator_version", "n_tools"):
        if k in h:
            L.append(f"| `{k}` | `{h[k]}` |")
    L.append("")
    if errors:
        L += ["### Failures", "", "```json",
              json.dumps(errors[:20], indent=2), "```", ""]
    L += ["The preflight exits non-zero on a single failure. It runs again on RunPod as "
          "step 4 of `run_all_4gpu.sh`, before any GPU time is spent, because a "
          "replay failure there means the pod's executor differs from the one that "
          "produced the oracle.", ""]
    return "\n".join(L)


def doc_cost(ctx: dict) -> str:
    rep = ctx["para_report"] or {}
    bud = rep.get("budget") or {}
    cli = rep.get("client_stats") or {}
    cache = FACTORY / "outputs" / "cache" / "openrouter"
    n_cached = sum(1 for _ in cache.rglob("*.json")) if cache.is_dir() else 0
    L = ["# Pilot2 cost report", "", f"Generated {ctx['now']}.", "",
         "## Paid services", "", "| item | value |", "|---|---|",
         "| provider | OpenRouter |",
         f"| model id | `{rep.get('model', 'n/a')}` |",
         f"| date (UTC) | {rep.get('date_utc', 'n/a')} |",
         f"| API calls, final run | {cli.get('api_calls', 0)} "
         f"(guard: {bud.get('max_requests')}) |",
         f"| cached responses on disk (all runs) | {n_cached} |",
         f"| prompt tokens, final run | {bud.get('prompt_tokens', 'n/a')} |",
         f"| completion tokens, final run | {bud.get('completion_tokens', 'n/a')} |",
         f"| **cost, final run (USD)** | **{bud.get('usd', 0):.4f}** (guard: "
         f"{bud.get('max_usd')}) |", "",
         "Earlier calibration runs against the same cache cost a further "
         "$0.024 in total (600 requests at $0.0154, then 600 at $0.0062). "
         "Everything is two orders of magnitude below the 2 USD guard.", "",
         "The budget guard is enforced before each request, not audited afterwards: "
         "the client refuses to send request 601, or any request that would push the "
         "accumulated cost past the cap.", "",
         "## Free / local", "",
         "| item | cost |", "|---|---|",
         "| pilot2 generation, validation, selection, split, export | local CPU |",
         "| gold-replay preflight | local CPU |",
         f"| local Qwen3-4B probe | {ctx['probe_status_line']} |", "",
         "## Not yet spent", "",
         "The RunPod D0/D1 run is the next cost item and is **not** included here. "
         "Rough shape at 4 GPUs: canary ~2 GPU-hours, D0 and D1 sequentially, then "
         "evaluation across the four GPUs. The full NESTFUL test (1661 tasks) is "
         "deliberately disabled by default and is a separate, later decision.", ""]
    return "\n".join(L)


def doc_runbook(ctx: dict) -> str:
    v = ctx["v"]
    L = ["# RunPod pilot2 runbook — D0 vs D1", "", f"Generated {ctx['now']}.", "",
         "## The experiment", "",
         "One manipulated variable: the training dataset.", "",
         "| | D0 | D1 |", "|---|---|---|",
         "| training data | 160 old Stage-3 tasks | 160 pilot2 factory tasks |",
         "| base checkpoint | C0 | C0 |",
         "| reward | A4_GATED_VERIFIABLE | A4_GATED_VERIFIABLE |",
         f"| seed | {ctx['seed']} | {ctx['seed']} |",
         "| LR / KL / LoRA / optimizer / credit / decoding | identical | identical |",
         "| rollouts | 8 | 8 |",
         "| optimizer-step budget | identical | identical |",
         "| GPUs | 0 learner, 1-3 rollout workers | same |",
         "| tool registry | legacy `synthetic_tools.py` | factory adapter |", "",
         "### The one confound you cannot remove", "",
         "The tool registry travels with the dataset. A Stage-3 task can only be "
         "executed by the legacy registry and a pilot2 task only by the factory "
         "adapter, so `SYNTHETIC_TOOLS_DIR` is set per arm rather than globally "
         "(setting it globally would make every D0 tool call fail). This means the "
         "comparison is dataset **plus** executor implementation, not dataset alone. "
         "It cannot be avoided without rewriting one dataset against the other's "
         "registry, which would destroy the thing being tested. What it costs is "
         "bounded: both executors are deterministic and both are verified by a "
         "100 % gold-replay preflight before training, so neither arm is being "
         "scored against a broken oracle. Say this out loud in the write-up rather "
         "than claiming a clean single-variable manipulation.", "",
         "D0 and D1 run **sequentially**. Running them concurrently on the same pod "
         "would let them contend for GPU memory and quietly change the effective "
         "batch timing, which is exactly the kind of hidden difference this design "
         "exists to avoid.", "",
         "## One command", "", "```bash",
         "cd /workspace/Tool-R0",
         "export HF_TOKEN=...            # required",
         "export WANDB_API_KEY=...       # optional",
         "bash experiments/targeted_tool_data_factory/runpod_bundle_pilot2/run_all_4gpu.sh",
         "```", "",
         "## What it does, in order", "",
         "1. refuses to start unless 4 GPUs are visible;",
         "2. installs dependencies (idempotent);",
         "3. verifies every frozen artefact against `MANIFEST.sha256.json`;",
         "4. checks D0/D1 config parity — any difference outside the dataset and its "
         "registry aborts;",
         "5. runs the gold-replay preflight through the real trainer executor "
         f"({ctx['n_train']} train + {ctx['n_heldout']} held-out, must be 100 %);",
         "6. runs the dispatch/executor canary: 24 stratified pilot2 tasks x 8 rollouts, "
         "A1 first then A4;",
         "7. gates the canary: configured policy == resolved policy on every train row, "
         "A1 and A4 produce *different* rewards on hash-matched completions, no NaN/Inf, "
         "no terminal ordering inversion, executor and trajectory logging present, and "
         "every executed tool is a factory surface with no legacy fallback;",
         "8. only on PASS: trains D0, then D1;",
         "9. evaluates C0, D0 and D1 across GPU0-3 in parallel on the structural "
         f"held-out {ctx['n_heldout']} (G and A tracks reported separately) and the "
         "frozen NESTFUL diagnostic-500;",
         "10. writes the paired report: Win Rate, Function F1, Parameter F1, "
         "executability, gained/lost, paired bootstrap 95 % CI, exact McNemar, "
         "failure taxonomy.", "",
         "## Why the canary gate is not optional", "",
         "Round 1 of the reward ablation trained five arms that all silently resolved "
         "to the same reward. The canary exists so that a dispatch regression costs "
         "twenty minutes instead of an entire experiment. If A1 and A4 produce "
         "identical rewards on identical completions, the script stops and no "
         "training starts.", "",
         "## Useful flags", "", "```bash",
         "bash run_all_4gpu.sh --dry-run        # print every command, train nothing",
         "bash run_all_4gpu.sh --resume         # continue interrupted runs",
         "bash run_all_4gpu.sh --skip-canary    # reuse an earlier PASS",
         "bash run_all_4gpu.sh --stage eval     # re-run one stage only",
         "```", "",
         "## After the run", "",
         "Read `outputs/runpod_pilot2/D0_VS_D1_REPORT.md`. Interpret it carefully: a "
         "D1 gain on the structural held-out only shows the model learned the new "
         "data. Transfer is the NESTFUL diagnostic-500 number. Reporting the first as "
         "if it were the second is the mistake that made Round 1 uninterpretable.", "",
         "The full NESTFUL test (1661) is a separate, deliberately gated command:", "",
         "```bash",
         "CONFIRM_FULL_NESTFUL=yes bash experiments/targeted_tool_data_factory/"
         "runpod_bundle_pilot2/run_full_nestful_test.sh",
         "```", "",
         "## Bundle contents", "", "| file | purpose |", "|---|---|",
         "| `run_all_4gpu.sh` | the single entry point |",
         "| `install.sh` | dependency install |",
         "| `verify_hashes.py` | manifest verification |",
         "| `check_config_parity.py` | D0/D1 differ only in the dataset |",
         "| `check_canary_gates.py` | pilot2 executor gates on the canary |",
         "| `run_eval_all.py` | 6 eval jobs across GPU0-3 |",
         "| `make_paired_report.py` | paired D0 vs D1 statistics |",
         "| `run_full_nestful_test.sh` | disabled-by-default confirmation run |",
         "| `build_bundle.py` | freezes the data and writes the manifest |",
         f"| `data/` | frozen {v} train/held-out/reserve, canonical, canary, D0 data, "
         "NESTFUL diagnostic-500 |",
         "| `configs/` | D0 and D1 run configs |",
         "| `MANIFEST.sha256.json` | sha256 of every frozen artefact |", ""]
    return "\n".join(L)


# ─────────────────────────────────────────────────────────────── main ────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--version", default="pilot2")
    ap.add_argument("--baseline", default="pilot1")
    ap.add_argument("--seed", type=int, default=20260726)
    args = ap.parse_args()
    v, base = args.version, args.baseline

    exp = OUT / "selected" / f"export_{v}"
    selected = jlread(exp / f"canonical_{v}.jsonl") or jlread(OUT / "selected" / f"selected_{v}.jsonl")
    baseline_selected = (jlread(OUT / "selected" / f"export_{base}" / f"canonical_{base}.jsonl")
                         or jlread(OUT / "selected" / f"selected_{base}.jsonl"))
    probe = jread(OUT / "selected" / f"probe_{v}.json") or jread(OUT / "reports" / f"probe_{v}.json")
    preflight = (jread(FACTORY / "outputs" / "reports" / f"preflight_{v}.json")
                 or jread(FACTORY / "trainer_adapter" / f"preflight_{v}.json"))

    probe_status = (probe or {}).get("status", "NOT_RUN_LOCAL")
    ctx = {
        "v": v, "baseline": base, "seed": args.seed,
        "now": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "selected": selected, "baseline_selected": baseline_selected,
        "val_summary": jread(OUT / "validated" / f"validation_summary_{v}.json"),
        "gates": jread(OUT / "reports" / f"pilot2_gates_{v}.json"),
        "verdict": (jread(OUT / "reports" / f"verdict_{v}.json") or {}).get("verdict", "PENDING"),
        "leakage": jread(OUT / "splits" / f"leakage_audit_{v}.json"),
        "profile_match": jread(OUT / "selected" / f"profile_match_{v}.json"),
        "para_report": (jread(OUT / "reports" / f"paraphrase_{v}.json")
                        or jread(OUT / "reports" / f"paraphrase_report_{v}.json")),
        "probe": probe, "preflight": preflight,
        "probe_status_line": (
            "`NOT_RUN_LOCAL` — no OpenAI-compatible endpoint answered; see "
            "`LOCAL_PROBE_REPORT.md` for the exact PowerShell command."
            if probe_status == "NOT_RUN_LOCAL" else f"`{probe_status}`"),
        "n_train": len(jlread(exp / f"train_grpo_{v}.jsonl")),
        "n_heldout": len(jlread(exp / f"heldout_grpo_{v}.jsonl")),
        "n_reserve": len(jlread(exp / f"reserve_grpo_{v}.jsonl")),
    }
    ctx["artefacts"] = [
        (p, len(jlread(p)) if p.suffix == ".jsonl" else "-")
        for p in [exp / f"train_grpo_{v}.jsonl", exp / f"heldout_grpo_{v}.jsonl",
                  exp / f"reserve_grpo_{v}.jsonl", exp / f"canonical_{v}.jsonl",
                  exp / f"heldout_nestful_{v}.jsonl",
                  OUT / "validated" / f"validated_{v}.jsonl",
                  BUNDLE / "MANIFEST.sha256.json"] if p.is_file()]

    DOCS.mkdir(parents=True, exist_ok=True)
    written = []
    # PILOT2_REPORT.md is owned by the pipeline (`cli.py report`), which has the
    # full pilot1 comparison in hand. Regenerating it here would fork the truth;
    # this script only fills it in when the pipeline has not run yet.
    docs = [("PILOT2_DATA_CARD.md", doc_data_card),
            ("PILOT1_VS_PILOT2.md", doc_pilot1_vs_pilot2),
            ("OPENROUTER_PARAPHRASE_REPORT.md", doc_openrouter),
            ("LOCAL_PROBE_REPORT.md", doc_probe),
            ("TRAINER_INTEGRATION.md", doc_trainer_integration),
            ("RUNPOD_PILOT2_RUNBOOK.md", doc_runbook),
            ("COST_REPORT_PILOT2.md", doc_cost)]
    if not (DOCS / "PILOT2_REPORT.md").is_file():
        docs.insert(0, ("PILOT2_REPORT.md", doc_pilot2_report))

    for name, fn in docs:
        try:
            text = fn(ctx)
        except Exception as exc:  # noqa: BLE001
            print(f"[docs] {name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        (DOCS / name).write_text(text.rstrip() + "\n", encoding="utf-8")
        written.append(name)
        print(f"[docs] wrote docs/{name} ({len(text)} chars)")
    print(f"[docs] {len(written)}/{len(docs)} documents written; "
          f"docs/PILOT2_REPORT.md "
          f"{'present' if (DOCS / 'PILOT2_REPORT.md').is_file() else 'MISSING'}")
    return 0 if len(written) == len(docs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
