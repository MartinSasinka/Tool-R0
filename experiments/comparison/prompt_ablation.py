#!/usr/bin/env python3
"""Prompt-format ablation harness (subset eval; generation runs on the pod).

Compares THREE prompt variants on a fixed stratified subset of the NESTFUL test
set, across the key checkpoints:

  variants:
    current      legacy eval prompt  (minimal/prompt.py SYSTEM_PROMPT + _EVAL_HARDENING)
    train_style  nestful_core v2 prompt, role="train"  (deployment-close)
    hardened     nestful_core v2 prompt, role="eval"   (lenient-parser hardening)

  checkpoints: baseline / partial_s1_e4 / partial_s4_e1 / minimal_s4e2

Two modes:
  --dry-run (default, LOCAL): build the stratified subset, build all three prompt
            variants for every subset task, assert they construct + differ, and
            emit the subset file (+hash), a manifest, a PENDING summary and a
            report describing the pod command. No model required.
  --generate (POD): load each checkpoint, install each variant, run the canonical
            eval loop over the subset, score with the official scorer, and fill in
            `prompt_ablation_summary.csv` (ReAct Win per cell).

Outputs (experiments/comparison/):
  prompt_ablation_subset.jsonl            fixed stratified subset (+ .sha256)
  prompt_ablation_summary.csv             checkpoint x variant -> win_rate
  prompt_ablation_report.md               method + pod run command
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXPERIMENTS = os.path.dirname(_HERE)
if _EXPERIMENTS not in sys.path:
    sys.path.insert(0, _EXPERIMENTS)

from nestful_core.logging_utils import write_csv  # noqa: E402
from nestful_core import prompt as core_prompt  # noqa: E402
import prompt as legacy_prompt  # noqa: E402
from data import load_tasks  # noqa: E402

_MINIMAL = os.path.join(_EXPERIMENTS, "nestful_mtgrpo_minimal")
_PARTIAL = os.path.join(_EXPERIMENTS, "nestful_mtgrpo_partial")
_DATASET = os.path.join(_MINIMAL, "data", "NESTFUL-main", "data_v2", "nestful_data.jsonl")
_SUBSET = os.path.join(_HERE, "prompt_ablation_subset.jsonl")

VARIANTS = ("current", "train_style", "hardened")
CHECKPOINTS = ("baseline", "partial_s1_e4", "partial_s4_e1", "minimal_s4e2")


# ── prompt-variant installers ────────────────────────────────────────────────
def _install_variant(variant: str) -> None:
    """Monkeypatch the canonical ``prompt.build_messages`` for this variant."""
    if variant == "current":
        # Legacy eval prompt: SYSTEM_PROMPT + _EVAL_HARDENING when eval_hardening.
        legacy_prompt.build_messages = legacy_prompt.__dict__.get(
            "_orig_build_messages", legacy_prompt.build_messages)
        return
    role = "train" if variant == "train_style" else "eval"

    def _patched(task, history=None, eval_hardening=False):
        return core_prompt.build_messages_v2(task, history, role=role)

    legacy_prompt.build_messages = _patched


def _build_variant_messages(task, variant: str):
    if variant == "current":
        return legacy_prompt.build_messages(task, None, eval_hardening=True)
    role = "train" if variant == "train_style" else "eval"
    return core_prompt.build_messages_v2(task, None, role=role)


# ── stratified subset ────────────────────────────────────────────────────────
def build_subset(target: int = 400) -> list:
    tasks = load_tasks(_DATASET)
    by_bucket = defaultdict(list)
    for t in tasks:
        n = int(t.get("num_calls") or len(t.get("gold_calls", [])))
        bucket = "1-2" if n <= 2 else ("3-4" if n <= 4 else ("5-8" if n <= 8 else "9+"))
        by_bucket[bucket].append(t)
    # proportional stratified pick, deterministic (sorted by id).
    total = len(tasks)
    subset = []
    for bucket, items in by_bucket.items():
        items = sorted(items, key=lambda x: str(x.get("task_id")))
        k = max(1, round(target * len(items) / total))
        subset.extend(items[:k])
    subset = sorted(subset, key=lambda x: str(x.get("task_id")))
    return subset


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_subset(subset: list) -> str:
    with open(_SUBSET, "w", encoding="utf-8") as fh:
        for t in subset:
            fh.write(json.dumps({
                "task_id": t.get("task_id"),
                "question": t.get("question"),
                "tools": t.get("tools"),
                "gold_calls": t.get("gold_calls"),
                "gold_answer": t.get("gold_answer"),
                "num_calls": t.get("num_calls"),
            }, ensure_ascii=False) + "\n")
    digest = _sha256_file(_SUBSET)
    with open(_SUBSET + ".sha256", "w", encoding="utf-8") as fh:
        fh.write(digest + "\n")
    return digest


# ── dry run (local) ──────────────────────────────────────────────────────────
def run_dry(target: int) -> int:
    # keep a reference to the legacy builder so `current` restores cleanly.
    legacy_prompt.__dict__.setdefault("_orig_build_messages", legacy_prompt.build_messages)
    subset = build_subset(target)
    digest = _write_subset(subset)
    print(f"[prompt_ablation] subset n={len(subset)} sha256={digest[:12]}…")

    # validate all three variants build + differ for every subset task.
    diff_seen = False
    for t in subset:
        msgs = {v: _build_variant_messages(t, v) for v in VARIANTS}
        for v in VARIANTS:
            assert msgs[v] and msgs[v][0]["role"] == "system" and msgs[v][1]["role"] == "user", v
        sys_texts = {v: msgs[v][0]["content"] for v in VARIANTS}
        if len({sys_texts[v] for v in VARIANTS}) > 1:
            diff_seen = True
    assert diff_seen, "prompt variants are identical — ablation would be meaningless"

    sample = subset[0]
    sample_msgs = {v: _build_variant_messages(sample, v) for v in VARIANTS}
    sys_len = {v: len(sample_msgs[v][0]["content"]) for v in VARIANTS}

    # PENDING summary matrix.
    rows = []
    for ckpt in CHECKPOINTS:
        for v in VARIANTS:
            rows.append({"checkpoint": ckpt, "variant": v, "n": len(subset),
                         "react_win_rate": None, "status": "PENDING_GENERATION"})
    write_csv(os.path.join(_HERE, "prompt_ablation_summary.csv"), rows,
              fieldnames=["checkpoint", "variant", "n", "react_win_rate", "status"])

    versions = core_prompt.prompt_versions()
    report = [
        "# Prompt-format ablation",
        "",
        f"- Subset: `prompt_ablation_subset.jsonl` (n={len(subset)}, stratified by "
        f"num_calls), sha256 `{digest}`.",
        f"- Prompt versions: train=`{versions['train_prompt_version']}`, "
        f"eval=`{versions['eval_prompt_version']}`.",
        "",
        "## Variants",
        "",
        "| variant | source | system-prompt chars (sample task) |",
        "|---|---|---|",
        f"| current | legacy `SYSTEM_PROMPT + _EVAL_HARDENING` | {sys_len['current']} |",
        f"| train_style | core v2 prompt role=train | {sys_len['train_style']} |",
        f"| hardened | core v2 prompt role=eval | {sys_len['hardened']} |",
        "",
        "Local dry-run validated that all three variants build for every subset "
        "task and that the system prompts differ.",
        "",
        "## Run on the pod (fills react_win_rate)",
        "",
        "```bash",
        "USE_VLLM=1 python experiments/comparison/prompt_ablation.py --generate \\",
        "    --baseline <BASE_MODEL_DIR> \\",
        "    --adapter partial_s1_e4=<DIR> --adapter partial_s4_e1=<DIR> \\",
        "    --adapter minimal_s4e2=<DIR>",
        "```",
        "",
        "Generation installs each prompt variant via `_install_variant`, runs the "
        "canonical eval loop over the fixed subset, and scores with the official "
        "scorer. Only the prompt changes across cells; model, parser, executor, "
        "scorer and subset are held fixed.",
        "",
        "Generated by `prompt_ablation.py --dry-run`.",
    ]
    with open(os.path.join(_HERE, "prompt_ablation_report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(report) + "\n")
    print("[prompt_ablation] wrote subset, prompt_ablation_summary.csv (PENDING), report")
    return 0


# ── generation (pod) ─────────────────────────────────────────────────────────
def _load_subset() -> list:
    if not os.path.isfile(_SUBSET):
        raise SystemExit("run --dry-run first to build prompt_ablation_subset.jsonl")
    out = []
    with open(_SUBSET, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def run_generate(baseline_dir: str, adapters: dict) -> int:
    import yaml  # noqa: F401
    from nestful_core.eval_loop import run_episode, max_turns_for
    import nestful_official_score as nos
    from nestful_core import scoring

    # Lazy heavy deps — only needed on the pod.
    import torch  # noqa: F401
    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:
        from peft import PeftModel
    except Exception:  # pragma: no cover
        PeftModel = None

    cfg_path = os.path.join(_MINIMAL, "config.yaml")
    with open(cfg_path, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    subset = _load_subset()
    raw = nos.load_raw_dataset(_DATASET)
    legacy_prompt.__dict__.setdefault("_orig_build_messages", legacy_prompt.build_messages)

    ckpt_dirs = {"baseline": None, **adapters}
    rows = []
    for ckpt in CHECKPOINTS:
        if ckpt != "baseline" and ckpt not in adapters:
            print(f"[prompt_ablation] skip {ckpt} (no --adapter given)")
            continue
        tok = AutoTokenizer.from_pretrained(baseline_dir)
        model = AutoModelForCausalLM.from_pretrained(baseline_dir, torch_dtype="auto",
                                                     device_map="auto")
        if ckpt != "baseline" and PeftModel is not None:
            model = PeftModel.from_pretrained(model, adapters[ckpt])
        model.eval()
        for variant in VARIANTS:
            _install_variant(variant)
            items, n_win = [], 0
            for t in subset:
                traj = run_episode(model, tok, t, config, mode="eval",
                                   max_turns=max_turns_for(t, train=False))
                pred = [c for c in traj.predicted_calls]
                gold_row = raw.get(str(t.get("task_id")))
                if gold_row is not None:
                    items.append(nos.build_item(pred, gold_row))
            res = scoring.score_items_per_sample(items, win_rate=True)
            wr = sum(float(x.get("official_win") or 0) for x in res) / len(res) if res else 0.0
            rows.append({"checkpoint": ckpt, "variant": variant, "n": len(res),
                         "react_win_rate": round(wr, 4), "status": "DONE"})
            print(f"[prompt_ablation] {ckpt}/{variant}: win={wr:.4f}")
        del model

    write_csv(os.path.join(_HERE, "prompt_ablation_summary.csv"), rows,
              fieldnames=["checkpoint", "variant", "n", "react_win_rate", "status"])
    print("[prompt_ablation] wrote prompt_ablation_summary.csv (DONE)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true", help="run eval on the pod")
    ap.add_argument("--target", type=int, default=400, help="subset size (300-500)")
    ap.add_argument("--baseline", default="", help="base model dir (pod)")
    ap.add_argument("--adapter", action="append", default=[],
                    help="name=dir adapter (repeatable)")
    args = ap.parse_args()
    if not args.generate:
        return run_dry(args.target)
    adapters = {}
    for spec in args.adapter:
        if "=" in spec:
            k, v = spec.split("=", 1)
            adapters[k] = v
    if not args.baseline:
        raise SystemExit("--generate requires --baseline <base model dir>")
    return run_generate(args.baseline, adapters)


if __name__ == "__main__":
    raise SystemExit(main())
