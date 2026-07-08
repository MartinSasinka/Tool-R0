#!/usr/bin/env python3
"""Build the Stage2 continuation SFT view.

This is NOT a new dataset. It converts the EXISTING, already-filtered GRPO
Stage2 curriculum file

    outputs/curriculum_v3_1/filtered/stage2_2call_dependency.jsonl

into a serialized SFT training view ("Stage2 continuation SFT serialization"):
one (input_text, target_text, messages) record per Stage2 row, with an 80/20
train/val split. No new tasks are generated. No re-sampling, re-filtering, or
curriculum regeneration happens here — every row of the SFT view traces back
1:1 to a row of the source curriculum file (see `provenance` / sha256 in the
summary).

Hard failures (abort with a non-zero exit code, no partial dataset written):
  - the source file's row count does not exactly match the count the GRPO run
    expects for this stage (curriculum_v3_1_manifest.json `stages.<key>`),
  - any source row does not have exactly 2 gold tool calls.

Soft skips (excluded from the SFT view, counted + reasoned in the summary,
but do not abort the build unless ALL rows are skipped):
  - missing/short gold observations (can't build the "gold call 1 + gold
    observation 1" input prefix),
  - null gold_answer,
  - a duplicate sample_id,
  - a question that leaks internal curriculum metadata (motif/stage/cluster
    tokens, unresolved template placeholders) into the user-visible prompt.

Usage:
    python build_stage2_sft_dataset.py \
        [--source PATH] [--manifest PATH] [--out-dir DIR] \
        [--seed 42] [--train-frac 0.8] [--target-type continuation] \
        [--base-model Qwen/Qwen3-4B-Instruct-2507] [--no-tokenizer]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from sft_common import (  # noqa: E402
    BASE_MODEL,
    DEFAULT_MANIFEST,
    DEFAULT_OUT_DIR,
    DEFAULT_SOURCE_STAGE2,
    STAGE_KEY,
    build_continuation_record,
    build_full_trace_record,
    count_tokens,
    load_expected_stage_count,
    normalize_stage2_row,
    question_leak_hit,
    read_jsonl_raw,
    sha256_file,
    try_load_tokenizer,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=DEFAULT_SOURCE_STAGE2,
                    help="Existing GRPO Stage2 curriculum file (NOT regenerated).")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST,
                    help="curriculum_v3_1_manifest.json — source of the expected row count.")
    ap.add_argument("--stage-key", default=STAGE_KEY,
                    help="Key into manifest['stages'] used for the hard row-count check.")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--target-type", choices=["continuation", "full_trace"], default="continuation")
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--no-tokenizer", action="store_true",
                    help="Skip loading a tokenizer; use the char/4 length estimate instead.")
    ap.add_argument("--skip-manifest-check", action="store_true",
                    help="DANGEROUS / debug only: bypass the hard row-count gate.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    source = os.path.abspath(args.source)
    manifest = os.path.abspath(args.manifest)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isfile(source):
        print(f"[build_stage2_sft] ERROR: source file not found: {source}", file=sys.stderr)
        return 1

    source_sha256 = sha256_file(source)
    raw_rows = read_jsonl_raw(source)
    num_raw_rows = len(raw_rows)
    print(f"[build_stage2_sft] source = {source}")
    print(f"[build_stage2_sft] source sha256 = {source_sha256}")
    print(f"[build_stage2_sft] raw rows = {num_raw_rows}")

    # ---- HARD FAIL #1: row count must match what the GRPO run expects -----
    expected_count = None
    if not args.skip_manifest_check:
        if not os.path.isfile(manifest):
            print(f"[build_stage2_sft] ERROR: manifest not found: {manifest}\n"
                  f"  Refusing to build an SFT view without confirming the source "
                  f"file matches what the GRPO run trained Stage2 on. Pass "
                  f"--skip-manifest-check only for debugging.", file=sys.stderr)
            return 1
        expected_count = load_expected_stage_count(manifest, args.stage_key)
        if num_raw_rows != expected_count:
            print(
                f"[build_stage2_sft] HARD FAIL: source has {num_raw_rows} rows but "
                f"manifest['stages']['{args.stage_key}'] = {expected_count}. "
                f"The source file no longer matches what the GRPO run trained "
                f"Stage2 on — refusing to build a silently-drifted SFT view.\n"
                f"  source:   {source}\n  manifest: {manifest}",
                file=sys.stderr,
            )
            return 1
        print(f"[build_stage2_sft] manifest expected count = {expected_count} — OK (exact match)")

    # ---- HARD FAIL #2: every row must have exactly 2 gold calls ------------
    bad_call_count_rows: List[Tuple[int, str, int]] = []
    for idx, row in enumerate(raw_rows):
        gold_calls = row.get("gold_calls") or []
        if len(gold_calls) != 2:
            bad_call_count_rows.append((idx, str(row.get("sample_id", f"row_{idx}")), len(gold_calls)))
    if bad_call_count_rows:
        print(
            f"[build_stage2_sft] HARD FAIL: {len(bad_call_count_rows)} / {num_raw_rows} "
            f"row(s) do NOT have exactly 2 gold tool calls:", file=sys.stderr,
        )
        for idx, sid, n in bad_call_count_rows[:20]:
            print(f"    row {idx} sample_id={sid} num_gold_calls={n}", file=sys.stderr)
        if len(bad_call_count_rows) > 20:
            print(f"    ... and {len(bad_call_count_rows) - 20} more", file=sys.stderr)
        return 1
    print(f"[build_stage2_sft] all {num_raw_rows} rows have exactly 2 gold calls — OK")

    # ---- Build records (soft-skip on secondary validation issues) ---------
    seen_ids = set()
    valid_records: List[Dict[str, Any]] = []
    skip_reasons: Dict[str, int] = {}
    skip_examples: Dict[str, List[str]] = {}

    def _skip(reason: str, sample_id: str) -> None:
        skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
        skip_examples.setdefault(reason, [])
        if len(skip_examples[reason]) < 5:
            skip_examples[reason].append(sample_id)

    builder = build_continuation_record if args.target_type == "continuation" else build_full_trace_record

    for idx, row in enumerate(raw_rows):
        sample_id = str(row.get("sample_id", f"row_{idx}"))
        if row.get("gold_answer") is None:
            _skip("null_gold_answer", sample_id)
            continue
        leak = question_leak_hit(row.get("question", ""))
        if leak:
            _skip(f"metadata_leak_in_question:{leak}", sample_id)
            continue
        if sample_id in seen_ids:
            _skip("duplicate_sample_id", sample_id)
            continue

        task = normalize_stage2_row(row, idx)
        obs = task.get("_gold_observations")
        if not obs or len(obs) < 2:
            _skip("missing_or_short_gold_observations", sample_id)
            continue

        try:
            record = builder(task)
        except ValueError as exc:
            _skip(f"record_build_error:{exc}", sample_id)
            continue

        seen_ids.add(sample_id)
        valid_records.append(record)

    num_valid = len(valid_records)
    total_skipped = sum(skip_reasons.values())
    print(f"[build_stage2_sft] valid records = {num_valid}  (skipped = {total_skipped})")
    if num_valid == 0:
        print("[build_stage2_sft] ERROR: zero valid records produced — aborting.", file=sys.stderr)
        return 1

    # ---- Deterministic 80/20 split (seeded) --------------------------------
    valid_records.sort(key=lambda r: r["sample_id"])  # deterministic order before shuffling
    order = list(range(num_valid))
    random.Random(args.seed).shuffle(order)
    shuffled = [valid_records[i] for i in order]
    n_train = int(round(num_valid * args.train_frac))
    train_records = shuffled[:n_train]
    val_records = shuffled[n_train:]

    train_ids = {r["sample_id"] for r in train_records}
    val_ids = {r["sample_id"] for r in val_records}
    overlap = train_ids & val_ids
    no_overlap = len(overlap) == 0
    no_dupe_ids = len(seen_ids) == num_valid

    print(f"[build_stage2_sft] train = {len(train_records)}  val = {len(val_records)}")

    # ---- Length stats (tokenizer best-effort) ------------------------------
    tokenizer, tok_name = (None, None)
    if not args.no_tokenizer:
        tokenizer, tok_name = try_load_tokenizer(args.base_model)
    tokenizer_used = tok_name or "approx_chars_div_4 (tokenizer unavailable)"

    def _length_stats(records: List[Dict[str, Any]]) -> Dict[str, float]:
        if not records:
            return {"avg_input_chars": 0.0, "avg_target_chars": 0.0,
                     "avg_input_tokens": 0.0, "avg_target_tokens": 0.0}
        in_chars = [len(r["input_text"]) for r in records]
        tgt_chars = [len(r["target_text"]) for r in records]
        in_toks = [count_tokens(tokenizer, r["input_text"]) for r in records]
        tgt_toks = [count_tokens(tokenizer, r["target_text"]) for r in records]
        n = len(records)
        return {
            "avg_input_chars": sum(in_chars) / n,
            "avg_target_chars": sum(tgt_chars) / n,
            "avg_input_tokens": sum(in_toks) / n,
            "avg_target_tokens": sum(tgt_toks) / n,
        }

    length_stats = {"train": _length_stats(train_records), "val": _length_stats(val_records)}

    # ---- Distribution (train + val combined) -------------------------------
    def _bump(d: Dict[str, int], k: Any) -> None:
        k = str(k)
        d[k] = d.get(k, 0) + 1

    tool_names_used: Dict[str, int] = {}
    tool_names_offered: Dict[str, int] = {}
    motifs: Dict[str, int] = {}
    clusters: Dict[str, int] = {}
    answer_types: Dict[str, int] = {}
    for r in valid_records:
        for name in r["provenance"].get("tool_names_used", []):
            _bump(tool_names_used, name)
        for name in r["provenance"].get("tool_names_offered", []):
            _bump(tool_names_offered, name)
        _bump(motifs, r["provenance"].get("target_full_motif"))
        _bump(clusters, r["provenance"].get("source_failure_cluster"))
        _bump(answer_types, r["provenance"].get("answer_type"))

    def _sorted_desc(d: Dict[str, int]) -> Dict[str, int]:
        return dict(sorted(d.items(), key=lambda kv: (-kv[1], kv[0])))

    # ---- Write outputs ------------------------------------------------------
    train_path = os.path.join(out_dir, "train.jsonl")
    val_path = os.path.join(out_dir, "val.jsonl")
    write_jsonl(train_path, train_records)
    write_jsonl(val_path, val_records)

    validation_checks = {
        "every_example_has_exactly_2_gold_calls": True,  # hard-gated above
        "every_example_has_first_call_and_first_observation": True,  # guaranteed by builder
        "every_example_has_second_gold_call": True,  # guaranteed by builder
        "no_null_gold_answer": True,  # soft-skipped above
        "no_duplicate_sample_ids": no_dupe_ids,
        "no_train_val_overlap": no_overlap,
        "source_row_count_matches_manifest": (expected_count is None or num_raw_rows == expected_count),
    }
    all_checks_pass = all(validation_checks.values())

    summary: Dict[str, Any] = {
        "dataset_kind": "Stage2 continuation SFT serialization (derived view, NOT a new dataset)",
        "input_path": source,
        "source_sha256": source_sha256,
        "manifest_path": manifest if not args.skip_manifest_check else None,
        "manifest_stage_key": args.stage_key,
        "expected_row_count_from_manifest": expected_count,
        "num_source_rows": num_raw_rows,
        "target_type": args.target_type,
        "seed": args.seed,
        "train_frac": args.train_frac,
        "num_valid_records": num_valid,
        "num_train": len(train_records),
        "num_val": len(val_records),
        "all_have_2_calls": True,
        "skipped_total": total_skipped,
        "skip_reasons": skip_reasons,
        "skip_examples": skip_examples,
        "length_stats": length_stats,
        "tokenizer_used": tokenizer_used,
        "distribution": {
            "tool_names_used": _sorted_desc(tool_names_used),
            "tool_names_offered": _sorted_desc(tool_names_offered),
            "target_full_motif": _sorted_desc(motifs),
            "source_failure_cluster": _sorted_desc(clusters),
            "answer_type": _sorted_desc(answer_types),
        },
        "validation_checks": validation_checks,
        "all_validation_checks_pass": all_checks_pass,
        "train_path": train_path,
        "val_path": val_path,
        "design_notes": [
            "Target text omits <think>...</think> reasoning, matching the "
            "existing GRPO teacher-forced-prefix convention "
            "(rollout._format_forced_call_text) — forced/gold turns are "
            "injected as bare <tool_call_answer> tags, not model-style "
            "reasoning traces.",
            "gold_calls[1]['label'] uses the curriculum's own '$var_1'/'$var_2' "
            "underscore convention, which differs from the SYSTEM_PROMPT's own "
            "worked example ('$var1'/'$var2', no underscore). This is a "
            "pre-existing property of the curriculum data, not introduced by "
            "this script — flagged here since it may teach an inconsistent "
            "label format; do not silently 'fix' it without re-running the "
            "GRPO gold-replay checks, since replay validated the underscore form.",
            "messages[] is the FULL 7-turn conversation (including the real "
            "gold observation for call 2) so a trainer can correctly mask loss "
            "to ONLY the two generation targets (indices in "
            "loss_target_message_indices) while still giving the model the "
            "real call-2 observation as context for the turn-3 stop decision.",
        ],
    }

    with open(os.path.join(out_dir, "SFT_DATASET_SUMMARY.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    _write_markdown_summary(out_dir, summary)

    print(f"[build_stage2_sft] wrote {train_path} ({len(train_records)} rows)")
    print(f"[build_stage2_sft] wrote {val_path} ({len(val_records)} rows)")
    print(f"[build_stage2_sft] wrote {os.path.join(out_dir, 'SFT_DATASET_SUMMARY.json')}")
    print(f"[build_stage2_sft] wrote {os.path.join(out_dir, 'SFT_DATASET_SUMMARY.md')}")
    if not all_checks_pass:
        print("[build_stage2_sft] WARNING: not all validation checks passed — see summary.",
              file=sys.stderr)
        return 1
    return 0


def _write_markdown_summary(out_dir: str, s: Dict[str, Any]) -> None:
    lines: List[str] = []
    w = lines.append
    w("# Stage2 Continuation SFT — Dataset Summary")
    w("")
    w(f"**{s['dataset_kind']}**")
    w("")
    w(f"- source (existing GRPO Stage2 curriculum file, NOT regenerated): `{s['input_path']}`")
    w(f"- source sha256: `{s['source_sha256']}`")
    w(f"- manifest expected row count: {s['expected_row_count_from_manifest']}  "
      f"(source row count: {s['num_source_rows']})")
    w(f"- target_type: `{s['target_type']}`")
    w(f"- seed: {s['seed']}, train_frac: {s['train_frac']}")
    w(f"- valid records: {s['num_valid_records']}  "
      f"(train: {s['num_train']}, val: {s['num_val']})")
    w(f"- all records have exactly 2 gold calls: {s['all_have_2_calls']} (hard-gated)")
    w("")
    w("## Skipped examples")
    w("")
    if s["skipped_total"] == 0:
        w("None.")
    else:
        w(f"Total skipped: {s['skipped_total']}")
        w("")
        w("| reason | count | example sample_ids |")
        w("|---|---:|---|")
        for reason, count in sorted(s["skip_reasons"].items(), key=lambda kv: -kv[1]):
            examples = ", ".join(f"`{e}`" for e in s["skip_examples"].get(reason, []))
            w(f"| {reason} | {count} | {examples} |")
    w("")
    w("## Length statistics")
    w("")
    w("| split | avg input chars | avg target chars | avg input tokens | avg target tokens |")
    w("|---|---:|---:|---:|---:|")
    for split in ("train", "val"):
        ls = s["length_stats"][split]
        w(f"| {split} | {ls['avg_input_chars']:.1f} | {ls['avg_target_chars']:.1f} | "
          f"{ls['avg_input_tokens']:.1f} | {ls['avg_target_tokens']:.1f} |")
    w("")
    w(f"_Token lengths computed with tokenizer: `{s['tokenizer_used']}`_")
    w("")
    w("## Tool / motif distribution (train + val combined)")
    w("")
    for title, key in (("Tool names used (gold calls)", "tool_names_used"),
                       ("Tool names offered (prompt tool menu, incl. distractors)", "tool_names_offered"),
                       ("target_full_motif", "target_full_motif"),
                       ("source_failure_cluster", "source_failure_cluster"),
                       ("answer_type", "answer_type")):
        dist = s["distribution"][key]
        if not dist:
            continue
        w(f"### {title}")
        w("")
        w("| value | count |")
        w("|---|---:|")
        for k, v in dist.items():
            w(f"| {k} | {v} |")
        w("")
    w("## Validation requirements checked")
    w("")
    for k, v in s["validation_checks"].items():
        w(f"- [{'PASS' if v else 'FAIL'}] {k}")
    w("")
    w("## Design notes")
    w("")
    for note in s.get("design_notes", []):
        w(f"- {note}")
    w("")
    with open(os.path.join(out_dir, "SFT_DATASET_SUMMARY.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
