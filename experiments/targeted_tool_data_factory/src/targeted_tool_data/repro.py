"""Reproducibility stamps attached to every new dataset and report (§22).

Every JSON/JSONL artifact produced by the pilot4 tooling carries a
``schema_version`` plus a ``repro`` block, so an artifact can always be traced
back to the commit, config and seed that produced it.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

REPRO_SCHEMA_VERSION = "ttdf.repro.v1"

_TRACKED_PACKAGES = ["pydantic", "pyyaml", "numpy", "matplotlib"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git(args: Sequence[str], cwd: Path) -> str:
    try:
        res = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True,
                             text=True, check=False, timeout=60)
        return res.stdout.strip() if res.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def git_info(repo_root: Path) -> Dict[str, Any]:
    commit = _git(["rev-parse", "HEAD"], repo_root)
    status = _git(["status", "--porcelain"], repo_root)
    dirty_files = [ln[3:] for ln in status.splitlines() if ln.strip()]
    return {
        "commit": commit,
        "dirty": bool(dirty_files),
        "n_dirty_files": len(dirty_files),
        "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"], repo_root),
    }


def dependency_versions() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for name in _TRACKED_PACKAGES:
        try:
            from importlib.metadata import version

            out[name] = version(name)
        except Exception:  # noqa: BLE001 - optional dependency
            out[name] = "not_installed"
    return out


def sha256_obj(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stamp(repo_root: Path, *, schema_version: str,
          cli_args: Optional[Sequence[str]] = None,
          seeds: Optional[Dict[str, int]] = None,
          config: Optional[Dict[str, Any]] = None,
          input_paths: Optional[Iterable[Path]] = None,
          extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build the reproducibility block embedded in every artifact."""
    inputs: Dict[str, str] = {}
    for p in input_paths or []:
        p = Path(p)
        if p.is_file():
            key = str(p)
            try:
                key = str(p.resolve().relative_to(repo_root.resolve())).replace("\\", "/")
            except ValueError:
                pass
            inputs[key] = sha256_file(p)
    return {
        "repro_schema_version": REPRO_SCHEMA_VERSION,
        "schema_version": schema_version,
        "generated_at_utc": utc_now(),
        "git": git_info(repo_root),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "dependency_versions": dependency_versions(),
        "cli_args": list(cli_args or sys.argv[1:]),
        "seeds": dict(seeds or {}),
        "config_hash": sha256_obj(config) if config is not None else "",
        "input_hashes": inputs,
        **(extra or {}),
    }


def write_json(path: Path, payload: Dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8")
    return path


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    return path


def write_csv(path: Path, rows: List[Dict[str, Any]],
              columns: Optional[Sequence[str]] = None) -> Path:
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(columns) if columns else sorted({k for r in rows for k in r})
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def write_text(path: Path, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def write_manifest(out_dir: Path, *, repro: Dict[str, Any],
                   patterns: Sequence[str] = ("*.json", "*.jsonl", "*.csv", "*.md"),
                   manifest_name: str = "MANIFEST.sha256.json") -> Path:
    """Hash every artifact in ``out_dir`` (excluding the manifest itself)."""
    out_dir = Path(out_dir)
    files: Dict[str, Any] = {}
    for pat in patterns:
        for p in sorted(out_dir.rglob(pat)):
            if p.name == manifest_name:
                continue
            rel = str(p.relative_to(out_dir)).replace("\\", "/")
            files[rel] = {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
    return write_json(out_dir / manifest_name,
                      {"schema_version": "ttdf.manifest.v1", "repro": repro,
                       "files": files})
