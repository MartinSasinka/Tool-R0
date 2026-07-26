#!/usr/bin/env python3
"""Freeze the pilot2 RunPod bundle.

Copies the selected pilot2 export plus the D0 comparison data into
`runpod_bundle_pilot2/data/`, derives the stratified 24-task canary subset,
writes the D0/D1 run configs, and stamps every artefact into
`MANIFEST.sha256.json`. Re-running is idempotent: identical inputs produce
identical hashes.

The bundle is DATA-FROZEN on purpose — RunPod must never regenerate anything.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

BUNDLE = Path(__file__).resolve().parent
FACTORY = BUNDLE.parent
V3 = FACTORY.parent / "nestful_synthetic_curriculum_v3"
ABL_DATA = V3 / "reports" / "reward_ablation" / "data"

SEED = 20260726
CANARY_N = 24


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stratified_canary(train_grpo: list[dict], canonical: dict[str, dict],
                      n: int = CANARY_N) -> list[dict]:
    """Round-robin over (motif, call_count, answer_type) strata so the canary
    exercises every executor path, not 24 copies of the easiest cell."""
    strata: dict[tuple, list[dict]] = defaultdict(list)
    for row in train_grpo:
        c = canonical.get(str(row["sample_id"]), {})
        key = (c.get("motif") or row.get("motif_type"),
               c.get("call_count") or row.get("num_calls"),
               c.get("answer_type") or row.get("answer_type"))
        strata[key].append(row)
    for rows in strata.values():
        rows.sort(key=lambda r: str(r["sample_id"]))

    picked: list[dict] = []
    keys = sorted(strata, key=lambda k: tuple(str(x) for x in k))
    i = 0
    while len(picked) < n and any(strata[k] for k in keys):
        k = keys[i % len(keys)]
        if strata[k]:
            picked.append(strata[k].pop(0))
        i += 1
    return picked[:n]


def run_config(label: str, train_path: Path, train_sha: str, source: str,
               tools_dir: str) -> dict:
    """Everything that defines a training arm. D0 and D1 must differ only in
    the dataset and in the tool registry the dataset is written against."""
    return {
        "label": label,
        "run_id": f"pilot2_{label}_seed{SEED}",
        "notes": source,
        "train_subset": train_path.relative_to(BUNDLE).as_posix(),
        "train_subset_sha256": train_sha,
        "train_subset_source": source,
        "base_checkpoint": "C0",
        "seed": SEED,
        "reward": {
            "arm": "A4_GATED_VERIFIABLE",
            "policy": "reward_ablation_A4_GATED_VERIFIABLE",
            "binary_only": False,
            "format_as_gate": True,
        },
        "optimizer": {
            "learning_rate": 1e-6,
            "kl_beta": 0.1,
            "max_grad_norm": 1.0,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "epochs": 1,
            "mask_clipped_from_update": True,
        },
        "lora": {"method": "qlora", "r": 16, "alpha": 32, "dropout": 0.05,
                 "target_modules": "auto", "load_in_4bit": True,
                 "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": "bfloat16"},
        "rollouts": {"num_generations": 8, "temperature": 0.7, "top_p": 0.95,
                     "paradigm": "react", "max_new_tokens_train": 2048},
        "credit_assignment": "turn_level_discounted_group_normalized",
        "hardware": {"learner_gpu": 0, "rollout_dp_gpus": [1, 2, 3],
                     "use_vllm": True, "bf16": True, "gradient_checkpointing": True},
        # Determined by the dataset: a Stage-3 task can only be executed by the
        # legacy registry, a pilot2 task only by the factory adapter. This is an
        # unavoidable confound of the comparison and is documented as such.
        "executor": {"mode": "synthetic", "synthetic_tools_dir": tools_dir},
        "eval": {"during_training": False,
                 "after_training": ["heldout80", "nestful_diagnostic_500"]},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--export", type=Path,
                    default=FACTORY / "outputs" / "selected" / "export_pilot2")
    ap.add_argument("--version", default="pilot2")
    ap.add_argument("--d0-train", type=Path, default=ABL_DATA / "train_subset_160.jsonl")
    ap.add_argument("--diagnostic-ids", type=Path,
                    default=ABL_DATA / "nestful_diagnostic_500_ids.json")
    ap.add_argument("--nestful-test", type=Path,
                    default=FACTORY.parent / "nestful_mtgrpo_minimal" / "data" / "splits" / "nestful_test.jsonl")
    args = ap.parse_args()

    v = args.version
    data = BUNDLE / "data"
    data.mkdir(parents=True, exist_ok=True)
    provenance: dict[str, str] = {}

    # ── frozen pilot2 splits ────────────────────────────────────────────────
    expected = {"train_grpo": 160, "heldout_grpo": 80, "reserve_grpo": 80,
                "train_nestful": 160, "heldout_nestful": 80, "reserve_nestful": 80,
                "canonical": 320}
    for stem, count in expected.items():
        src = args.export / f"{stem}_{v}.jsonl"
        if not src.is_file():
            print(f"[build] ABORT: missing {src}", file=sys.stderr)
            return 2
        rows = read_jsonl(src)
        if len(rows) != count:
            print(f"[build] ABORT: {src.name} has {len(rows)} rows, expected {count}",
                  file=sys.stderr)
            return 2
        dst = data / f"{stem}_{v}.jsonl"
        shutil.copyfile(src, dst)
        provenance[dst.name] = str(src)
        print(f"[build] {dst.name}: {len(rows)} rows")

    canonical = {str(r["task_id"]): r for r in read_jsonl(data / f"canonical_{v}.jsonl")}
    # canonical split views used by the paired report for the G/A track breakdown
    for stem in ("train", "heldout", "reserve"):
        ids = [str(r["sample_id"]) for r in read_jsonl(data / f"{stem}_grpo_{v}.jsonl")]
        write_jsonl(data / f"{stem}_canonical_{v}.jsonl",
                    [canonical[i] for i in ids if i in canonical])

    # ── stratified dispatch canary (24 tasks) ───────────────────────────────
    train_grpo = read_jsonl(data / f"train_grpo_{v}.jsonl")
    canary = stratified_canary(train_grpo, canonical)
    write_jsonl(data / f"canary_{v}_24.jsonl", canary)
    strata = sorted({(canonical.get(str(r["sample_id"]), {}).get("motif"),
                      canonical.get(str(r["sample_id"]), {}).get("call_count"),
                      canonical.get(str(r["sample_id"]), {}).get("answer_type"))
                     for r in canary})
    print(f"[build] canary_{v}_24.jsonl: {len(canary)} rows across {len(strata)} strata")

    # ── D0 comparison data (old Stage-3) ────────────────────────────────────
    if args.d0_train.is_file():
        rows = read_jsonl(args.d0_train)
        shutil.copyfile(args.d0_train, data / "d0_stage3_train_160.jsonl")
        provenance["d0_stage3_train_160.jsonl"] = str(args.d0_train)
        print(f"[build] d0_stage3_train_160.jsonl: {len(rows)} rows")
    else:
        print(f"[build] ABORT: D0 train subset missing: {args.d0_train}", file=sys.stderr)
        return 2

    # ── frozen NESTFUL diagnostic-500 ───────────────────────────────────────
    if args.diagnostic_ids.is_file() and args.nestful_test.is_file():
        ids = json.loads(args.diagnostic_ids.read_text(encoding="utf-8"))
        if isinstance(ids, dict):
            ids = ids.get("task_ids") or ids.get("sample_ids") or ids.get("ids") or []
        want = {str(i) for i in ids}
        rows = [r for r in read_jsonl(args.nestful_test)
                if str(r.get("sample_id")) in want]
        write_jsonl(data / "nestful_diagnostic_500.jsonl", rows)
        provenance["nestful_diagnostic_500.jsonl"] = (
            f"{args.nestful_test} filtered by {args.diagnostic_ids}")
        print(f"[build] nestful_diagnostic_500.jsonl: {len(rows)}/{len(want)} ids resolved")
        if len(rows) != len(want):
            print("[build] WARNING: diagnostic-500 id resolution incomplete", file=sys.stderr)
    else:
        print("[build] WARNING: diagnostic-500 not materialised "
              f"(ids={args.diagnostic_ids.is_file()}, test={args.nestful_test.is_file()})",
              file=sys.stderr)

    # ── D0 / D1 run configs ─────────────────────────────────────────────────
    cfg_dir = BUNDLE / "configs"
    cfg_dir.mkdir(exist_ok=True)
    d0_path = data / "d0_stage3_train_160.jsonl"
    d1_path = data / f"train_grpo_{v}.jsonl"
    (cfg_dir / "d0_stage3_old.json").write_text(
        json.dumps(run_config("D0", d0_path, sha256_file(d0_path),
                              "old Stage-3 curriculum (the data that did not transfer)",
                              "../../nestful_synthetic_curriculum_v3"),
                   indent=2) + "\n", encoding="utf-8")
    (cfg_dir / "d1_pilot2.json").write_text(
        json.dumps(run_config("D1", d1_path, sha256_file(d1_path),
                              "targeted_tool_data_factory pilot2 (program-first, executor-verified)",
                              "../trainer_adapter"),
                   indent=2) + "\n", encoding="utf-8")
    print("[build] configs/d0_stage3_old.json, configs/d1_pilot2.json")

    # ── manifest over every frozen artefact ─────────────────────────────────
    files: dict[str, dict] = {}
    for path in sorted(BUNDLE.rglob("*")):
        if not path.is_file() or path.name == "MANIFEST.sha256.json":
            continue
        rel = path.relative_to(BUNDLE).as_posix()
        if rel.startswith(("data/", "configs/")) or path.suffix in (".py", ".sh", ".txt"):
            entry = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            if path.suffix == ".jsonl":
                entry["lines"] = sum(1 for _ in path.open(encoding="utf-8"))
            if rel.split("/")[-1] in provenance:
                entry["source"] = provenance[rel.split("/")[-1]]
            files[rel] = entry

    manifest = {
        "bundle": "runpod_bundle_pilot2",
        "dataset_version": v,
        "seed": SEED,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "factory_hashes": _factory_hashes(),
        "files": files,
    }
    (BUNDLE / "MANIFEST.sha256.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[build] MANIFEST.sha256.json: {len(files)} artefacts")
    for name in (f"train_grpo_{v}.jsonl", f"heldout_grpo_{v}.jsonl",
                 f"reserve_grpo_{v}.jsonl", f"canary_{v}_24.jsonl",
                 "d0_stage3_train_160.jsonl"):
        rel = f"data/{name}"
        if rel in files:
            print(f"    {files[rel]['sha256']}  {rel}")
    return 0


def _factory_hashes() -> dict:
    sys.path.insert(0, str(FACTORY / "trainer_adapter" / "lib"))
    try:
        import synthetic_tools  # type: ignore

        return synthetic_tools.factory_hashes()
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


if __name__ == "__main__":
    raise SystemExit(main())
