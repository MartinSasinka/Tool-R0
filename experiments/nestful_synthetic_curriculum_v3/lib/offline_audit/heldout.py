from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Set

from lib.offline_audit.paths import REPO_TRAIN_SUBSET, STAGE3_SOURCE, sha256_file


def prepare_heldout(reports_dir: Path) -> Dict[str, Any]:
    if not STAGE3_SOURCE.is_file() or not REPO_TRAIN_SUBSET.is_file():
        payload = {"error": "missing stage3 source or train subset", "partial": True}
        (reports_dir / "heldout_stage3_166_manifest.json").write_text(json.dumps(payload, indent=2))
        return payload

    train_ids: Set[str] = set()
    with open(REPO_TRAIN_SUBSET, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                train_ids.add(json.loads(line).get("sample_id"))

    held = []
    with open(STAGE3_SOURCE, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            sid = row.get("sample_id")
            if sid not in train_ids:
                held.append(row)

    out_jsonl = reports_dir / "heldout_stage3_166.jsonl"
    with open(out_jsonl, "w", encoding="utf-8") as fh:
        for row in held:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    motifs = Counter(row.get("motif_type") for row in held)
    manifest = {
        "source": str(STAGE3_SOURCE),
        "train_subset": str(REPO_TRAIN_SUBSET),
        "heldout_count": len(held),
        "expected_count": 166,
        "disjoint": len(train_ids & {r.get("sample_id") for r in held}) == 0,
        "heldout_sha256": sha256_file(out_jsonl),
        "motif_distribution": dict(motifs),
    }
    (reports_dir / "heldout_stage3_166_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    cmd = (
        "python experiments/nestful_synthetic_curriculum_v3/scripts/ablation/run_local_heldout_eval.py "
        f"--heldout {out_jsonl} --checkpoint C0|A0|A4 --allow-model-inference"
    )
    (reports_dir / "HELDOUT_STAGE3_PLAN.md").write_text(
        "\n".join(
            [
                "# Held-out Stage-3 166 plan",
                "",
                f"- Held-out rows: **{len(held)}** (expected 166)",
                f"- Disjoint from train 160: **{manifest['disjoint']}**",
                f"- SHA-256: `{manifest['heldout_sha256']}`",
                "",
                "## Optional inference (not run in offline audit phase)",
                "",
                "```powershell",
                cmd,
                "```",
            ]
        ),
        encoding="utf-8",
    )
    return manifest
