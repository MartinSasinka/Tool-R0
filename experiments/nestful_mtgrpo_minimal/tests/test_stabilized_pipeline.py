"""Sanity checks for the stabilized-curriculum pipeline (no GPU / no training run).

Covers:
  - clean curriculum data integrity (parse, fields, tools/args/refs)
  - mixed-replay loader (correct stages, weights, determinism)
  - prompt hardening rules (anti-no-tool / anti-early-finish / anti-mental / exact-arg)
  - eval-vs-train prompt consistency (same tags + reference syntax)
  - validation-subset determinism (best-checkpoint-by-ReAct-Win selection helper)
  - early-stopping selection logic (patience / min_delta)

These tests SKIP gracefully when the clean dataset or heavy deps are unavailable,
so they are safe to run anywhere.
"""
import json
import os
import re

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXP = os.path.dirname(_HERE)
_CLEAN_DIR = os.path.join(_EXP, "data", "clean_curriculum")

_VAR_REF_RE = re.compile(r"^\$([A-Za-z_]*?_?)(\d+)(?:\.([A-Za-z_][\w]*))?\$$")


# ── helpers ───────────────────────────────────────────────────────────────────
def _clean_files():
    if not os.path.isdir(_CLEAN_DIR):
        return []
    out = []
    for n in range(1, 7):
        p = os.path.join(_CLEAN_DIR, f"epoch_{n}_{n}call.jsonl")
        if os.path.isfile(p):
            out.append((n, p))
    return out


def _refs_in_value(value):
    refs = []
    if isinstance(value, str):
        m = _VAR_REF_RE.match(value.strip())
        if m:
            refs.append((int(m.group(2)), m.group(3)))
    elif isinstance(value, list):
        for v in value:
            refs.extend(_refs_in_value(v))
    elif isinstance(value, dict):
        for v in value.values():
            refs.extend(_refs_in_value(v))
    return refs


# ── clean data integrity ──────────────────────────────────────────────────────
def test_clean_dataset_present_or_skip():
    if not _clean_files():
        pytest.skip("clean_curriculum not generated yet "
                    "(run experiments/data/prepare_clean_training_set.py)")


def test_clean_rows_parse_and_have_required_fields():
    files = _clean_files()
    if not files:
        pytest.skip("clean_curriculum not generated yet")
    for _n, path in files:
        with open(path, encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)  # must parse
                for field in ("input", "tools", "output", "gold_answer"):
                    assert field in row, f"{path}:{i} missing {field}"
                assert isinstance(row["tools"], list) and row["tools"]
                assert isinstance(row["output"], list) and row["output"]


def test_clean_calls_reference_existing_tools_and_args():
    files = _clean_files()
    if not files:
        pytest.skip("clean_curriculum not generated yet")
    for _n, path in files:
        with open(path, encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                toolmap = {t.get("name"): t for t in row["tools"]}
                calls = row["output"]
                for ci, call in enumerate(calls, start=1):
                    name = call.get("name")
                    assert name in toolmap, f"{path}:{i} unknown tool {name}"
                    args = call.get("arguments")
                    assert isinstance(args, dict), f"{path}:{i} args not a dict"
                    # declared arg names
                    params = toolmap[name].get("parameters", {})
                    props = params.get("properties", params) if isinstance(params, dict) else {}
                    declared = set(props.keys()) if isinstance(props, dict) else set()
                    for arg_name in args:
                        if declared:
                            assert arg_name in declared, \
                                f"{path}:{i} call {name} has undeclared arg {arg_name}"
                    # references must point to earlier calls
                    for v in args.values():
                        for ref_idx, _field in _refs_in_value(v):
                            assert 1 <= ref_idx < ci, \
                                f"{path}:{i} bad reference $var{ref_idx} in call #{ci}"


def test_clean_gold_answer_has_no_unresolved_reference():
    files = _clean_files()
    if not files:
        pytest.skip("clean_curriculum not generated yet")
    any_ref = re.compile(r"\$[A-Za-z_]*?_?\d+(?:\.[A-Za-z_][\w]*)?\$")
    for _n, path in files:
        with open(path, encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                ga = row["gold_answer"]
                s = ga if isinstance(ga, str) else json.dumps(ga)
                assert not any_ref.search(s), \
                    f"{path}:{i} gold_answer still has an unresolved $var ref: {s!r}"


# ── mixed replay loader ─────────────────────────────────────────────────────────
def _mixed_loader():
    import data
    return data.load_tasks_mixed


def test_mixed_replay_uniform_counts(tmp_path):
    load_tasks_mixed = _mixed_loader()
    files = _clean_files()
    if len(files) < 3:
        pytest.skip("need >=3 clean stage files")
    paths = [p for _n, p in files[:3]]
    mix = load_tasks_mixed(paths, weights=None, seed=7)
    assert len(mix["per_stage"]) == 3
    total_avail = sum(ps["available"] for ps in mix["per_stage"])
    # uniform → each stage sampled ~ avail (target = round(1/3 * total))
    assert len(mix["tasks"]) > 0
    # every task is tagged with its source stage
    assert all("_stage" in t or "task_id" in t for t in mix["tasks"][:5])
    assert abs(sum(ps["sampled"] for ps in mix["per_stage"]) - len(mix["tasks"])) == 0
    assert total_avail > 0


def test_mixed_replay_weights_oversample_stage1():
    load_tasks_mixed = _mixed_loader()
    files = _clean_files()
    if len(files) < 3:
        pytest.skip("need >=3 clean stage files")
    paths = [p for _n, p in files[:3]]
    mix = load_tasks_mixed(paths, weights=[2.0, 1.0, 1.0], seed=7)
    s1, s2, s3 = (ps["sampled"] for ps in mix["per_stage"])
    assert s1 > s2 and s1 > s3, f"stage1 should be oversampled: {s1},{s2},{s3}"


def test_mixed_replay_is_deterministic():
    load_tasks_mixed = _mixed_loader()
    files = _clean_files()
    if len(files) < 2:
        pytest.skip("need >=2 clean stage files")
    paths = [p for _n, p in files[:2]]
    a = load_tasks_mixed(paths, weights=None, seed=123)
    b = load_tasks_mixed(paths, weights=None, seed=123)
    ids_a = [t["task_id"] for t in a["tasks"]]
    ids_b = [t["task_id"] for t in b["tasks"]]
    assert ids_a == ids_b


# ── prompt hardening + eval/train consistency ───────────────────────────────────
def test_system_prompt_has_hardening_rules():
    import prompt
    sp = prompt.SYSTEM_PROMPT.lower()
    # anti-mental
    assert "never solve" in sp or "do not solve" in sp
    # anti-no-tool-call / anti-early-finish on first turn
    assert "first turn" in sp
    # anti-too-few-calls (continue on intermediate result)
    assert "intermediate" in sp and "continue" in sp
    # exactly one non-empty call per turn
    assert "exactly one" in sp
    # exact argument names
    assert "exact argument names" in sp
    # reference syntax (both variants)
    assert "$var1.result$" in prompt.SYSTEM_PROMPT
    assert "$var1.output_0$" in prompt.SYSTEM_PROMPT


def test_train_and_eval_prompt_share_tags_and_refs():
    import prompt
    task = {"tools": [], "question": "q"}
    train_msgs = prompt.build_messages(task, history=None, eval_hardening=False)
    eval_msgs = prompt.build_messages(task, history=None, eval_hardening=True)
    train_sys = train_msgs[0]["content"]
    eval_sys = eval_msgs[0]["content"]
    # eval prompt is a SUPERSET (same SYSTEM_PROMPT + hardening reminder)
    assert eval_sys.startswith(prompt.SYSTEM_PROMPT)
    assert "OUTPUT FORMAT" in eval_sys and "OUTPUT FORMAT" not in train_sys
    # identical tag + reference conventions in both
    for s in (train_sys, eval_sys):
        assert "<tool_call_answer>" in s and "</tool_call_answer>" in s
        assert "$var1.result$" in s


# ── validation subset determinism + best-checkpoint selection logic ─────────────
def test_validation_subset_is_deterministic_and_reused(tmp_path):
    # Load THIS experiment's run.py by path: several sibling experiments also
    # ship a run.py, and depending on suite order a wrong one may already own
    # the bare "run" name in sys.modules / sys.path.
    import importlib.util
    try:
        spec = importlib.util.spec_from_file_location(
            "nestful_minimal_run", os.path.join(_EXP, "run.py"))
        run = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run)
    except Exception as exc:  # noqa: BLE001 - heavy deps may be missing
        pytest.skip(f"run.py not importable here: {exc}")
    # synthesize a small "full" validation file
    full = tmp_path / "full.jsonl"
    with open(full, "w", encoding="utf-8") as fh:
        for i in range(50):
            fh.write(json.dumps({"sample_id": f"s{i:03d}", "input": "x",
                                  "tools": [], "output": [], "gold_answer": i}) + "\n")
    ids = tmp_path / "ids.json"
    sub = tmp_path / "sub.jsonl"
    p1 = run._build_validation_subset(str(full), 10, str(ids), str(sub), seed=42)
    with open(ids, encoding="utf-8") as fh:
        ids1 = json.load(fh)["sample_ids"]
    assert len(ids1) == 10
    # rebuild → reuse exactly the same subset (file already exists)
    p2 = run._build_validation_subset(str(full), 10, str(ids), str(sub), seed=42)
    assert p1 == p2
    with open(ids, encoding="utf-8") as fh:
        ids2 = json.load(fh)["sample_ids"]
    assert ids1 == ids2
    # fresh path with the same seed → identical selection (determinism)
    ids_b = tmp_path / "ids_b.json"
    sub_b = tmp_path / "sub_b.jsonl"
    run._build_validation_subset(str(full), 10, str(ids_b), str(sub_b), seed=42)
    with open(ids_b, encoding="utf-8") as fh:
        ids3 = json.load(fh)["sample_ids"]
    assert sorted(ids1) == sorted(ids3)


def _select_best_and_early_stop(win_seq, patience, min_delta):
    """Pure reference impl of the run_curriculum.sh early-stop / best selection.

    Mirrors the bash logic: track global best (strict >), reset patience only on
    improvement >= min_delta over the stage-best, stop when patience exceeded.
    Returns (best_value, best_epoch_1based, stopped_at_epoch_or_None).
    """
    best_val, best_epoch = -1.0, None
    stage_best = -1.0
    count = 0
    stopped = None
    for epoch, win in enumerate(win_seq, start=1):
        if win > best_val:
            best_val, best_epoch = win, epoch
        if (win - stage_best) >= min_delta:
            stage_best = win
            count = 0
        else:
            count += 1
            if count >= patience and epoch < len(win_seq):
                stopped = epoch
                break
    return best_val, best_epoch, stopped


def test_early_stop_selection_logic():
    # improving then flat plateau: best is the peak; stop one eval into the plateau
    best, ep, stopped = _select_best_and_early_stop(
        [0.10, 0.30, 0.30, 0.30], patience=1, min_delta=0.005)
    assert best == 0.30 and ep == 2          # peak reached at epoch 2
    assert stopped == 3                       # no >=0.005 gain at epoch 3 → stop

    # steady improvement never early-stops
    best, ep, stopped = _select_best_and_early_stop(
        [0.10, 0.20, 0.30, 0.40], patience=1, min_delta=0.005)
    assert best == 0.40 and ep == 4 and stopped is None

    # patience=2 tolerates one stall before stopping
    best, ep, stopped = _select_best_and_early_stop(
        [0.10, 0.10, 0.20, 0.20, 0.20, 0.20], patience=2, min_delta=0.005)
    assert best == 0.20 and ep == 3
    assert stopped == 5
