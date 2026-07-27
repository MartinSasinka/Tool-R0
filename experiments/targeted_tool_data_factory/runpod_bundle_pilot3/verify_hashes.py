#!/usr/bin/env python3
"""Verify frozen pilot3 bundle artefacts against MANIFEST.sha256.json."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

BUNDLE = Path(__file__).resolve().parent


def sha256_file(path: Path) -> str:
    data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=BUNDLE / "MANIFEST.sha256.json")
    args = ap.parse_args()
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"[verify_hashes] cannot read {args.manifest}: {exc}", file=sys.stderr)
        return 3
    files = manifest.get("files", {})
    bad: list[str] = []
    for rel, expected in sorted(files.items()):
        path = BUNDLE / rel
        if not path.is_file():
            bad.append(f"MISSING {rel}")
            continue
        actual = sha256_file(path)
        if actual != expected["sha256"]:
            bad.append(f"MISMATCH {rel}\n    expected {expected['sha256']}\n"
                       f"    actual   {actual}")
        else:
            print(f"  ok  {expected['sha256'][:16]}  {rel}")
    print(f"[verify_hashes] {len(files) - len(bad)}/{len(files)} artefacts verified")
    if bad:
        print("[verify_hashes] FAIL", *bad, sep="\n  ", file=sys.stderr)
        return 2
    print("[verify_hashes] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
