"""Final freeze: manifest, artifact hashes and a source snapshot.

Pilot4.2 shipped with empty ``input_hashes``, which means its "reproducible" claim
could not be checked. Here the manifest fails loudly if the inputs it is supposed to
hash are missing, and when the working tree is dirty it writes a patch plus a
per-file manifest of the modified sources, so a reproduction attempt knows it needs
the snapshot and not just the commit.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .. import repro
from . import RUN_ID, SCHEMA_VERSION, GENERATOR_VERSION, PROMPT_VERSION_PREFIX
from .blueprints import registry_hash as workflow_registry_hash
from .ops import registry_hash as primitive_registry_hash

MANIFEST = "MANIFEST.sha256.json"
FREEZE = "freeze_manifest.json"
PATCH = "SOURCE_SNAPSHOT.patch"
SOURCE_MANIFEST = "SOURCE_TREE_MANIFEST.json"

#: Inputs whose content defines the run. A missing entry is a freeze failure, not a
#: silently absent hash.
INPUT_GLOBS = (
    "src/targeted_tool_data/pilot43/**/*.py",
    "configs/pilot4_3_openrouter.yaml",
    "analysis/pilot43_independent_audit/*.py",
)
PROFILE_INPUTS = ("target_profile_v3.json",)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _git(args: Sequence[str], cwd: Path,
         env: Optional[Dict[str, str]] = None) -> str:
    try:
        res = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True,
                             check=False, timeout=300, env=env)
        # git speaks UTF-8; decoding with the console code page would corrupt
        # any non-ASCII path or diff hunk on a Windows machine.
        return res.stdout.decode("utf-8", "replace") if res.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _git_root(start: Path) -> Path:
    """The repository root. ``git status`` prints paths relative to it."""
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    return start


def _status_paths(status: str) -> List[str]:
    """Paths from ``git status --porcelain``, with renames reduced to the new name."""
    out: List[str] = []
    for line in status.splitlines():
        if not line.strip():
            continue
        rel = line[3:].strip()
        if " -> " in rel:                       # "R  old -> new"
            rel = rel.split(" -> ", 1)[1]
        out.append(rel.strip().strip('"'))
    return out


#: Suffixes a reproduction needs. Everything else is data: it is hashed in the
#: manifest, but its bytes do not belong in a source patch.
SOURCE_SUFFIXES = frozenset({".py", ".yaml", ".yml", ".toml", ".cfg", ".ini",
                             ".md", ".txt", ".json"})
#: Per-file ceiling for patch inclusion. Anything larger is a data artefact.
MAX_PATCH_FILE_BYTES = 512 * 1024


def _is_build_residue(path: Path) -> bool:
    """Compiled or cached output that says nothing about the source."""
    return ("__pycache__" in path.parts or ".pytest_cache" in path.parts
            or path.suffix in (".pyc", ".pyo"))


def _snapshot_files(root: Path, changed: Sequence[str]
                    ) -> tuple[List[str], List[Dict[str, Any]]]:
    """Split the changed paths into what the patch carries and what it omits."""
    keep: List[str] = []
    omitted: List[Dict[str, Any]] = []
    for rel in changed:
        path = root / rel
        candidates = ([path] if path.is_file()
                      else sorted(p for p in path.rglob("*") if p.is_file())
                      if path.is_dir() else [])
        for item in candidates:
            if _is_build_residue(item):
                continue
            key = str(item.relative_to(root)).replace("\\", "/")
            size = item.stat().st_size
            if item.suffix.lower() not in SOURCE_SUFFIXES:
                omitted.append({"path": key, "bytes": size, "reason": "not source"})
            elif size > MAX_PATCH_FILE_BYTES:
                omitted.append({"path": key, "bytes": size, "reason": "too large"})
            else:
                keep.append(key)
    return keep, omitted


def _full_diff(root: Path, files: Sequence[str], scratch: Path) -> str:
    """A patch covering modified *and* untracked source files.

    ``git diff HEAD`` shows nothing for a file git has never seen, and almost
    every Pilot4.3 source file is untracked, so that patch would restore a run
    that cannot execute. Staging into a throwaway index turns each untracked
    file into a new-file diff without touching the developer's real index; the
    pathspec keeps the diff to the files it was asked about, so nothing else in
    the tree can leak into the patch as a spurious deletion.
    """
    if not files:
        return ""
    # git runs with cwd=root, so a relative scratch path would resolve against
    # the repository root and quietly leave the temporary index empty -- which
    # renders every tracked file as deleted in the patch.
    scratch = scratch.resolve()
    index = scratch / ".freeze_index"
    spec = scratch / ".freeze_pathspec"
    spec.write_text("\n".join(files) + "\n", encoding="utf-8")
    env = {**os.environ, "GIT_INDEX_FILE": str(index)}
    try:
        _git(["read-tree", "HEAD"], root, env)
        if not _git(["ls-files"], root, env).strip():
            return ""      # an empty index would render the whole tree deleted
        _git(["add", "-A", f"--pathspec-from-file={spec}"], root, env)
        # Only these files were staged, so an unscoped diff against HEAD is
        # already scoped to them. ``git diff`` has no --pathspec-from-file.
        return _git(["diff", "--cached", "HEAD"], root, env)
    finally:
        index.unlink(missing_ok=True)
        spec.unlink(missing_ok=True)


def source_snapshot(out_dir: Path, repo_root: Optional[Path] = None
                    ) -> Dict[str, Any]:
    """Write the patch and per-file manifest when the tree is not clean."""
    root = _git_root(repo_root or _repo_root())
    status = _git(["status", "--porcelain"], root)
    changed = _status_paths(status)
    clean = not changed
    patched, omitted = _snapshot_files(root, changed)
    diff = _full_diff(root, patched, out_dir) if not clean else ""
    if diff:
        header = ""
    elif clean:
        header = "# working tree clean at freeze time\n"
    else:
        # never let an empty patch read as a clean tree: the run is then not
        # reproducible from this directory and the report has to say so
        header = ("# SNAPSHOT FAILED: the tree is dirty but no patch could be "
                  f"produced for {len(patched)} changed source files\n")
    (out_dir / PATCH).write_text(diff or header, encoding="utf-8")

    files: Dict[str, Dict[str, Any]] = {}
    gone: List[str] = []
    for rel in changed:
        path = root / rel
        if path.is_file():
            files[rel.replace("\\", "/")] = {
                "sha256": repro.sha256_file(path),
                "bytes": path.stat().st_size,
            }
        elif path.is_dir():                     # status collapses new directories
            for child in sorted(p for p in path.rglob("*") if p.is_file()):
                if _is_build_residue(child):
                    continue
                key = str(child.relative_to(root)).replace("\\", "/")
                files[key] = {"sha256": repro.sha256_file(child),
                              "bytes": child.stat().st_size}
        else:
            gone.append(rel)
    manifest = {
        "run_id": RUN_ID,
        "working_tree_clean": clean,
        "repo_root": str(root),
        "n_changed_paths": len(changed),
        "changed_files": files,
        "n_changed_files": len(files),
        "deleted_or_missing": gone,
        "patch": PATCH,
        "patch_bytes": len(diff.encode("utf-8")),
        "n_files_in_patch": len(patched),
        "patch_written": bool(diff) or clean,
        "omitted_from_patch": omitted,
        "n_omitted_from_patch": len(omitted),
        "reproduction_note": (
            "The working tree was clean; the commit alone reproduces this run."
            if clean else
            "The working tree was NOT clean. Reproducing this run requires the "
            f"commit plus {PATCH}; the commit alone is not sufficient. The patch "
            "carries source and configuration; data files listed under "
            "omitted_from_patch are hashed in changed_files but not embedded."),
    }
    (out_dir / SOURCE_MANIFEST).write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    return manifest


def _input_hashes(out_dir: Path, repo_root: Path) -> Dict[str, str]:
    hashes: Dict[str, str] = {}
    for pattern in INPUT_GLOBS:
        for path in sorted(repo_root.glob(pattern)):
            if path.is_file():
                key = str(path.relative_to(repo_root)).replace("\\", "/")
                hashes[key] = repro.sha256_file(path)
    for name in PROFILE_INPUTS:
        path = out_dir / name
        if path.is_file():
            hashes[f"outputs/{RUN_ID}/{name}"] = repro.sha256_file(path)
    return hashes


def _artifact_hashes(out_dir: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for path in sorted(out_dir.rglob("*")):
        if not path.is_file() or path.name in (MANIFEST, FREEZE):
            continue
        rel = str(path.relative_to(out_dir)).replace("\\", "/")
        out[rel] = {"sha256": repro.sha256_file(path),
                    "bytes": path.stat().st_size,
                    "lines": _count_lines(path) if path.suffix == ".jsonl" else None}
    return out


def _count_lines(path: Path) -> int:
    with path.open("rb") as fh:
        return sum(1 for _line in fh)


def _prompt_hashes() -> Dict[str, str]:
    try:
        from . import orprompts
    except Exception:                                  # noqa: BLE001
        return {}
    out: Dict[str, str] = {}
    for name in dir(orprompts):
        value = getattr(orprompts, name)
        if name.isupper() and isinstance(value, str) and len(value) > 40:
            out[name] = repro.sha256_obj(value)[:32]
        if name.isupper() and isinstance(value, dict):
            out[name] = repro.sha256_obj(value)[:32]
    return out


def _config_hash(repo_root: Path) -> str:
    path = repo_root / "configs" / "pilot4_3_openrouter.yaml"
    return repro.sha256_file(path) if path.is_file() else ""


def _llm_response_hashes(out_dir: Path) -> Dict[str, Any]:
    path = out_dir / "openrouter_requests_pilot43.jsonl"
    if not path.exists():
        return {"log": None, "n_records": 0, "response_hash_of_hashes": ""}
    hashes: List[str] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            digest = rec.get("response_sha256") or rec.get("raw_response_sha256")
            if digest:
                hashes.append(str(digest))
    return {"log": path.name, "n_records": len(hashes),
            "response_hash_of_hashes": repro.sha256_obj(hashes)}


def build(out_dir: Path, *, cli_args: Optional[Sequence[str]] = None,
          seeds: Optional[Dict[str, int]] = None,
          stage_reports: Optional[Dict[str, Any]] = None,
          repo_root: Optional[Path] = None) -> Dict[str, Any]:
    root = repo_root or _repo_root()
    snapshot = source_snapshot(out_dir, root)
    inputs = _input_hashes(out_dir, root)
    if not inputs:
        raise RuntimeError("refusing to freeze: no input hashes could be computed")

    ordered_ids: List[str] = []
    split_manifest = out_dir / "split_manifest.json"
    if split_manifest.exists():
        data = json.loads(split_manifest.read_text(encoding="utf-8"))
        for tier_ids in (data.get("train") or {}).values():
            ordered_ids.extend(tier_ids)
        for part_ids in (data.get("heldout") or {}).values():
            ordered_ids.extend(part_ids)
        ordered_ids.extend(data.get("reserve") or [])

    artifacts = _artifact_hashes(out_dir)
    payload = {
        "run_id": RUN_ID,
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "prompt_version_prefix": PROMPT_VERSION_PREFIX,
        "frozen_at_utc": repro.utc_now(),
        "git": repro.git_info(root),
        "python_version": sys.version.split()[0],
        "dependency_versions": repro.dependency_versions(),
        "cli_args": list(cli_args or sys.argv[1:]),
        "seeds": dict(seeds or {}),
        "input_hashes": inputs,
        "workflow_registry_hash": workflow_registry_hash(),
        "primitive_registry_hash": primitive_registry_hash(),
        "target_profile_hash": inputs.get(f"outputs/{RUN_ID}/target_profile_v3.json",
                                          ""),
        "openrouter_config_hash": _config_hash(root),
        "prompt_template_hashes": _prompt_hashes(),
        "raw_llm_response_hashes": _llm_response_hashes(out_dir),
        "ordered_sample_ids_sha256": repro.sha256_obj(ordered_ids),
        "n_ordered_sample_ids": len(ordered_ids),
        "source_snapshot": snapshot,
        "artifact_hashes": artifacts,
        "n_artifacts": len(artifacts),
        "stage_reports": stage_reports or {},
    }
    (out_dir / FREEZE).write_text(json.dumps(payload, indent=1, ensure_ascii=False),
                                  encoding="utf-8")
    (out_dir / MANIFEST).write_text(
        json.dumps({"run_id": RUN_ID, "files": artifacts}, indent=1,
                   ensure_ascii=False), encoding="utf-8")
    return payload


def verify(out_dir: Path) -> Dict[str, Any]:
    """Re-hash every artifact and report drift against the frozen manifest."""
    path = out_dir / MANIFEST
    if not path.exists():
        return {"verified": False, "reason": f"{MANIFEST} missing"}
    recorded = (json.loads(path.read_text(encoding="utf-8")).get("files") or {})
    current = _artifact_hashes(out_dir)
    changed = [k for k in recorded
               if k in current and current[k]["sha256"] != recorded[k]["sha256"]]
    missing = [k for k in recorded if k not in current]
    added = [k for k in current if k not in recorded]
    return {
        "verified": not (changed or missing),
        "n_recorded": len(recorded),
        "changed": changed,
        "missing": missing,
        "added_since_freeze": added,
    }
