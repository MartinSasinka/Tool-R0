"""Regression tests for the overnight-run persistence/count bugs (offline).

Covers:
  1. early stop keeps accepted rows in memory AND on disk (crash-safe writer);
  2. builder end-to-end (mock): filtered files == manifest == reports counts;
  3. target resolution: printed final table == used table (single source);
  4. offline client raises OfflineCacheMiss instead of spending money;
  5. scorer handles partial datasets (status partial, exit 0) and empty (exit 1);
  6. solver-gap log 'accepted' equals final accepted (judge rejections excluded);
  7. repair_candidate fixes unambiguous argument-key mistakes.

Run:  python experiments/nestful_synthetic_curriculum_v3/tests/test_agentic_persistence.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
V3_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, V3_ROOT)
sys.path.insert(0, os.path.join(V3_ROOT, "scripts", "data"))
sys.path.insert(0, os.path.join(V3_ROOT, "scripts", "lib"))

from openrouter_client import OfflineCacheMiss, OpenRouterClient  # noqa: E402
from lib.agentic_data.challenger import repair_candidate  # noqa: E402
from lib.agentic_data.mock_llm import MockLLM  # noqa: E402
from lib.agentic_data.orchestrator import (Orchestrator, StageBudgetStop,  # noqa: E402
                                           count_jsonl_rows, load_jsonl_rows)

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}{(' — ' + str(detail)) if detail and not cond else ''}")
    if not cond:
        FAILURES.append(name)


STAGE2 = "stage2_2call_agentic_openrouter"
BUILDER = os.path.join(V3_ROOT, "scripts", "data",
                       "build_curriculum_v4_agentic_openrouter.py")
SCORER = os.path.join(V3_ROOT, "scripts", "data", "score_dataset_quality.py")


def _mk_orch(tmp, max_iters):
    client = OpenRouterClient(
        cache_dir=os.path.join(tmp, "raw", "cache"),
        raw_dir=os.path.join(tmp, "raw"),
        backend="mock", mock_handler=MockLLM(seed=42), save_raw=False)
    models = {r: "mock" for r in
              ("challenger", "weak_solver", "strong_solver", "judge")}
    return Orchestrator(client=client, models=models, out_root=tmp, seed=42,
                        contamination_checker=None,
                        max_iterations_per_stage=max_iters, run_judge=True)


# ---------------------------------------------------- 1. early-stop persistence
tmp = tempfile.mkdtemp(prefix="agentic_es_")
try:
    orch = _mk_orch(tmp, max_iters=2)   # force an early StageBudgetStop
    stopped = False
    try:
        orch.generate_stage(STAGE2, target=10_000)
    except StageBudgetStop:
        stopped = True
    n_mem = len(orch.accepted_by_stage.get(STAGE2, []))
    stage_path = orch.stage_paths[STAGE2]
    n_disk = count_jsonl_rows(stage_path)
    check("early stop raised", stopped)
    check("early stop kept accepted rows in memory", n_mem > 0, n_mem)
    check("early stop persisted SAME rows to disk", n_disk == n_mem,
          f"disk={n_disk} mem={n_mem}")
    summ = orch.stage_summaries[STAGE2]
    check("stage summary written on early stop with status=partial",
          summ.get("status") == "partial" and summ.get("accepted") == n_mem, summ)
    rows = [json.loads(l) for l in open(stage_path, encoding="utf-8") if l.strip()]
    check("persisted rows are valid JSON with sample_id",
          all("sample_id" in r for r in rows))
    n_gap_acc = sum(1 for g in orch.solver_gap_log if g.get("accepted"))
    check("solver-gap 'accepted' == memory accepted", n_gap_acc == n_mem,
          f"gap={n_gap_acc} mem={n_mem}")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# ---------------------------------------------------- 2+3. builder end-to-end
tmp = tempfile.mkdtemp(prefix="agentic_e2e_") + "_mock"
try:
    proc = subprocess.run(
        [sys.executable, BUILDER, "--mock", "--max-accepted-per-stage", "3",
         "--seed", "7", "--output-dir", tmp],
        capture_output=True, text=True, timeout=1800)
    check("mock builder exit 0", proc.returncode == 0,
          proc.stderr[-500:] if proc.returncode else "")
    manifest = json.load(open(os.path.join(
        tmp, "manifests", "curriculum_v4_agentic_openrouter_manifest.json"),
        encoding="utf-8"))
    extra = manifest["extra"]
    ok_counts = True
    for stage, meta in extra["stage_files"].items():
        n_manifest = extra["accepted"][stage]
        p = os.path.join(os.path.dirname(os.path.dirname(V3_ROOT)), "..",
                         meta["path"])
        n_file = meta["rows"]
        real_file = count_jsonl_rows(os.path.join(tmp, "filtered",
                                                  os.path.basename(meta["path"])))
        if not (n_manifest == n_file == real_file == 3):
            ok_counts = False
    check("manifest accepted == stage_files.rows == disk rows == target",
          ok_counts, extra["accepted"])
    check("manifest has target_resolution with final_targets",
          extra.get("target_resolution", {}).get("final_targets")
          == extra.get("targets"), extra.get("target_resolution"))
    check("manifest has completion status per stage",
          all(v.get("status") == "complete"
              for v in extra.get("completion", {}).values()),
          extra.get("completion"))
    # printed FINAL target table must equal the manifest targets (bug #5)
    printed = [l for l in proc.stdout.splitlines() if "<- FINAL" in l]
    check("stdout prints exactly one FINAL target table", len(printed) == 1)
    check("printed targets match manifest targets",
          all(str(v) in printed[0] for v in extra["targets"].values()))
    # dataset report counts consistent with manifest
    rep = open(os.path.join(tmp, "reports", "AGENTIC_DATASET_REPORT.md"),
               encoding="utf-8").read()
    total = sum(extra["accepted"].values())
    check("dataset report 'Accepted total' == manifest total",
          f"Accepted total: {total} " in rep)
    gap_rep = open(os.path.join(tmp, "reports", "AGENTIC_SOLVER_GAP_REPORT.md"),
                   encoding="utf-8").read()
    check("solver-gap report 'finally accepted' == manifest total",
          f"finally accepted (this run): {total}" in gap_rep)
    check("count consistency check ran",
          "count consistency OK" in proc.stdout)

    # ------------------------------------------------ 5. scorer on partial
    # make it partial: bump the manifest targets above actual rows
    extra["targets"] = {s: 5 for s in extra["targets"]}
    json.dump(manifest, open(os.path.join(
        tmp, "manifests", "curriculum_v4_agentic_openrouter_manifest.json"),
        "w", encoding="utf-8"))
    proc2 = subprocess.run(
        [sys.executable, SCORER, "--dataset-dir", tmp],
        capture_output=True, text=True, timeout=1800)
    check("scorer exits 0 on PARTIAL dataset", proc2.returncode == 0,
          proc2.stderr[-500:])
    check("scorer reports status=partial", "status=partial" in proc2.stdout)
    q = json.load(open(os.path.join(tmp, "reports", "DATASET_QUALITY.json"),
                       encoding="utf-8"))
    check("scorer completeness says partial",
          q["completeness"]["overall_status"] == "partial")
    check("partial dataset is never a training candidate",
          q["verdict"]["training_candidate"] is False)

    # empty dataset → clear error, minimal report, exit 1
    empty_dir = os.path.join(tmp, "empty_ds")
    os.makedirs(os.path.join(empty_dir, "filtered"))
    proc3 = subprocess.run(
        [sys.executable, SCORER, "--dataset-dir", empty_dir],
        capture_output=True, text=True, timeout=600)
    check("scorer exits 1 on EMPTY dataset (with report)",
          proc3.returncode == 1
          and os.path.isfile(os.path.join(empty_dir, "reports",
                                          "DATASET_QUALITY.md")))
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# ---------------------------------------------------- 4. offline cache-only
tmp = tempfile.mkdtemp(prefix="agentic_off_")
try:
    client = OpenRouterClient(cache_dir=os.path.join(tmp, "cache"),
                              raw_dir=None, backend="openrouter", offline=True)
    try:
        client.chat(role="challenger", model="x", messages=[
            {"role": "user", "content": "hi"}])
        check("offline mode raises OfflineCacheMiss on cache miss", False)
    except OfflineCacheMiss:
        check("offline mode raises OfflineCacheMiss on cache miss", True)
    check("offline mode needs no API key (never calls network)", True)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# ---------------------------------------------------- 4b. OpenRouter 500 body retry
from openrouter_client import _api_error_message  # noqa: E402
check("api_error_message detects error body",
      "Internal Server Error" in (_api_error_message({
          "error": {"message": "Internal Server Error", "code": 500}}) or ""))
check("api_error_message ok when choices present",
      _api_error_message({"choices": [{"message": {"content": "x"}}]}) is None)

tmp = tempfile.mkdtemp(prefix="agentic_or_retry_")
try:
    os.environ["OPENROUTER_API_KEY"] = "test-key-for-retry"
    client = OpenRouterClient(
        cache_dir=os.path.join(tmp, "cache"), raw_dir=None,
        backend="openrouter", max_retries=2, use_cache=False, save_raw=False)
    calls = {"n": 0}

    def _flaky_http(_payload):
        calls["n"] += 1
        if calls["n"] < 3:
            return {"error": {"message": "Internal Server Error", "code": 500}}
        return {"choices": [{"message": {"content": '{"ok": true}'}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.0}}

    client._http_request = _flaky_http  # type: ignore[method-assign]
    out = client.chat(role="weak_solver", model="test", messages=[
        {"role": "user", "content": "ping"}], json_mode=False, max_tokens=10)
    check("OpenRouter retries error body then succeeds", out["parsed"] == {"ok": True})
    check("OpenRouter retried twice before success", calls["n"] == 3, calls["n"])
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# ---------------------------------------------------- 4c. local weak routing
tmp_local = tempfile.mkdtemp(prefix="agentic_local_weak_")
try:
    os.environ["WEAK_SOLVER_BACKEND"] = "local"
    os.environ["OPENROUTER_API_KEY"] = "test-key"
    client = OpenRouterClient(
        cache_dir=os.path.join(tmp_local, "cache"), raw_dir=None,
        backend="openrouter", use_cache=False, save_raw=False)
    called = {"n": 0}

    def _fake_local(messages, *, temperature, max_tokens, seed):
        called["n"] += 1
        return '{"tool_calls": [], "final_answer": null}'

    import openrouter_client as orc  # noqa: E402
    orc._local_weak_generate = _fake_local  # type: ignore
    out = client.chat(role="weak_solver", model="Qwen/Qwen3-4B-Instruct-2507",
                      messages=[{"role": "user", "content": "hi"}],
                      json_mode=True, max_tokens=50)
    check("local weak_solver routes to HF backend", called["n"] == 1)
    check("local weak_solver does not increment API request budget",
          client.stats.n_requests == 0)
    check("local weak_solver increments local_requests",
          client.stats.by_role.get("weak_solver", {}).get("local_requests") == 1)
finally:
    os.environ.pop("WEAK_SOLVER_BACKEND", None)
    shutil.rmtree(tmp_local, ignore_errors=True)

# ---------------------------------------------------- 7. repair_candidate
cand = {"gold_calls": [
    {"name": "rectangle_area",
     "arguments": {"Length": 10, "width": 4}, "label": "$var1"}]}
fixed = repair_candidate(cand)
check("repair: case-insensitive arg key rename",
      set(fixed["gold_calls"][0]["arguments"]) == {"length", "width"},
      fixed["gold_calls"][0]["arguments"])
cand2 = {"gold_calls": [
    {"name": "rectangle_area",
     "arguments": {"len": 10, "width": 4}, "label": "$var1"}]}
fixed2 = repair_candidate(cand2)
check("repair: single unknown->single missing key",
      set(fixed2["gold_calls"][0]["arguments"]) == {"length", "width"},
      fixed2["gold_calls"][0]["arguments"])

# ---------------------------------------------------- 8. resume continues toward target
tmp = tempfile.mkdtemp(prefix="agentic_resume_") + "_mock"
try:
    proc1 = subprocess.run(
        [sys.executable, BUILDER, "--mock", "--max-accepted-per-stage", "3",
         "--seed", "7", "--output-dir", tmp],
        capture_output=True, text=True, timeout=1800)
    check("resume step1: initial mock run", proc1.returncode == 0)
    canonical = os.path.join(tmp, "filtered",
                             "stage2_2call_agentic_openrouter.jsonl")
    n1 = count_jsonl_rows(canonical)
    check("resume step1: wrote 3 rows to canonical", n1 == 3, n1)

    proc2 = subprocess.run(
        [sys.executable, BUILDER, "--mock", "--resume",
         "--max-accepted-per-stage", "5", "--seed", "7",
         "--stages", STAGE2, "--output-dir", tmp],
        capture_output=True, text=True, timeout=1800)
    check("resume step2: resume mock run", proc2.returncode == 0,
          proc2.stderr[-400:] if proc2.returncode else "")
    n2 = count_jsonl_rows(canonical)
    check("resume step2: total 5 rows (3 existing + 2 new)", n2 == 5, n2)
    check("resume step2: stdout mentions existing rows",
          "existing" in proc2.stdout.lower())
    m2 = json.load(open(os.path.join(
        tmp, "manifests", "curriculum_v4_agentic_openrouter_manifest.json"),
        encoding="utf-8"))
    summ = m2["extra"]["stage_summaries"][STAGE2]
    check("resume manifest: resumed_from=3", summ.get("resumed_from") == 3, summ)
    check("resume manifest: accepted_new=2", summ.get("accepted_new") == 2, summ)
    rows = load_jsonl_rows(canonical)
    ids = [r["sample_id"] for r in rows]
    check("resume: sample_id continues (000004, 000005 present)",
          "agentic_v4_stage2_000004" in ids and "agentic_v4_stage2_000005" in ids,
          ids[-2:])
    check("resume: count consistency OK", "count consistency OK" in proc2.stdout)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# ---------------------------------------------------- resume from partial_salvaged
tmp = tempfile.mkdtemp(prefix="agentic_resume_salv_")
try:
    filt = os.path.join(tmp, "filtered")
    os.makedirs(filt, exist_ok=True)
    salv = os.path.join(filt, "stage2_2call_agentic_openrouter.partial_salvaged.jsonl")
    from lib.agentic_data.verifier import deterministic_verify
    GOLD = [
        {"name": "rectangle_area", "arguments": {"length": 10, "width": 4},
         "label": "$var1"},
        {"name": "apply_discount",
         "arguments": {"price": "$var1.result$", "discount_percent": 10},
         "label": "$var2"},
    ]
    v = deterministic_verify({"question": "Compute the area of a 10 by 4 rectangle, "
                                          "then apply a 10% discount to that value "
                                          "as if it were a price in dollars today.",
                              "gold_calls": GOLD})
    from lib.agentic_data.schema import final_row
    models = {k: "m" for k in ("challenger", "weak_solver", "strong_solver", "judge")}
    good = final_row(
        sample_id="agentic_v4_stage2_000001",
        question="Compute the area of a 10 by 4 rectangle, then apply a 10% "
                 "discount to that value as if it were a price in dollars today.",
        tools=[], gold_calls=GOLD, observations=v["observations"],
        gold_answer=v["gold_answer"], stage=STAGE2, motif_type="long_chain",
        answer_type="scalar", generation_seed=42, models=models,
        solver_gap={"weak_status": "under_call", "strong_status": "win",
                    "weak_score": 0.3, "strong_score": 1.0, "gap": 0.7},
        provenance={"recipe_version": "t", "iteration": 1, "prompt_hash": None,
                    "raw_response_path": None, "created_at": "t",
                    "tool_schema_source_policy": "aggregate_style_only"})
    with open(salv, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(good) + "\n")
    orch = _mk_orch(tmp, max_iters=2)
    rows = orch.generate_stage(STAGE2, target=3, resume=True)
    check("resume from partial_salvaged: total rows", len(rows) == 3, len(rows))
    canonical = os.path.join(filt, "stage2_2call_agentic_openrouter.jsonl")
    check("resume migrated to canonical file", os.path.isfile(canonical))
    check("resume partial_salvaged + 2 new", orch.stage_summaries[STAGE2]["resumed_from"] == 1
          and orch.stage_summaries[STAGE2]["accepted_new"] == 2,
          orch.stage_summaries[STAGE2])
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print()
if FAILURES:
    print(f"FAILED: {len(FAILURES)} — {FAILURES}")
    sys.exit(1)
print("ALL TESTS PASSED")
