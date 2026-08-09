"""Compact view of the gate report: what failed, grouped, sizes set aside."""
from __future__ import annotations

import argparse
import json
import pathlib


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/pilot4_3_nestful_final")
    ap.add_argument("--skip-sizes", action="store_true",
                    help="hide size checks, which a scaled dry run always fails")
    args = ap.parse_args()

    path = pathlib.Path(args.out_dir) / "PILOT43_DATA_QUALITY_REPORT.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    checks = report.get("checks") or []
    failed = [c for c in checks if not c.get("passed")]
    if args.skip_sizes:
        failed = [c for c in failed if not str(c["id"]).startswith("size.")]
    print(f"{len(checks)} checks, {len(checks) - len(failed)} shown as passing, "
          f"{len(failed)} failing\n")
    for check in failed:
        print(f"[{check.get('severity', '?'):<8}] {check['id']}")
        print(f"           want: {check.get('requirement')}")
        print(f"           got : {check.get('observed')}")
        if check.get("note"):
            print(f"           note: {check['note']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
