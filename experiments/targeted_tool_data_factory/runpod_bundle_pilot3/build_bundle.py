#!/usr/bin/env python3
"""Freeze the pilot3 RunPod signal-probe bundle (does not touch pilot2)."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

BUNDLE = Path(__file__).resolve().parent
FACTORY = BUNDLE.parent
SEED = 20260727
VERSION = "pilot3"

EXPECTED = {
    "train_grpo": 600,
    "heldout_grpo": 200,
    "reserve_grpo": 200,
    "train_nestful": 600,
    "heldout_nestful": 200,
    "reserve_nestful": 200,
    "canonical": 1000,
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def sha256_file(path: Path) -> str:
    data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(data).hexdigest()


def write_text_lf(path: Path, text: str) -> None:
    data = text.replace("\r\n", "\n").replace("\r", "\n")
    if not data.endswith("\n"):
        data += "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data.encode("utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--export", type=Path,
                    default=FACTORY / "outputs" / "selected" / f"export_{VERSION}")
    ap.add_argument("--version", default=VERSION)
    args = ap.parse_args()
    v = args.version
    data = BUNDLE / "data"
    data.mkdir(parents=True, exist_ok=True)

    for stem, count in EXPECTED.items():
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
        print(f"[build] {dst.name}: {len(rows)} rows")

    # Shared NESTFUL profile for JSD gates on the pod.
    prof_src = FACTORY / "outputs" / "profiles" / "nestful_profile.json"
    if prof_src.is_file():
        shutil.copyfile(prof_src, data / "nestful_profile.json")
        print("[build] nestful_profile.json")

    files: dict[str, dict] = {}
    for path in sorted(BUNDLE.rglob("*")):
        if not path.is_file() or path.name == "MANIFEST.sha256.json":
            continue
        rel = path.relative_to(BUNDLE).as_posix()
        if rel.startswith(("data/", "configs/")) or path.suffix in (
                ".py", ".sh", ".txt", ".md"):
            entry = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            if path.suffix == ".jsonl":
                entry["lines"] = sum(1 for _ in path.open(encoding="utf-8"))
            files[rel] = entry

    manifest = {
        "bundle": "runpod_bundle_pilot3",
        "dataset_version": v,
        "seed": SEED,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "files": files,
        "note": "Signal-probe bundle for pilot3. Pilot2 artefacts are untouched.",
    }
    write_text_lf(BUNDLE / "MANIFEST.sha256.json",
                  json.dumps(manifest, indent=2, sort_keys=True))
    print(f"[build] MANIFEST.sha256.json: {len(files)} artefacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
