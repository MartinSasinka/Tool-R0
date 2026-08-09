"""Dataset provenance auditing: is a RunPod subset really a slice of a parent export?

Answers the question "was D1 trained on the dataset we think it was" without
relying on ``sample_id`` alone. Export ids are derived from generator seeds and
attempt counters, so a regenerated parent export can carry different ids for
byte-identical tasks. Identity is therefore established at four levels:

    1. byte-level        — parent's first N lines vs the subset file
    2. multi-key         — overlap per identity key (sample_id, family, ...)
    3. canonical exact   — content fingerprint incl. offered-tool order
    4. canonical semantic— content fingerprint modulo tool order / ref syntax

Read-only. Never mutates the audited datasets.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SCHEMA_VERSION = "ttdf.provenance_audit.v1"

STATUS_BYTES = "EXACT_FIRST_300_BYTES"
STATUS_CANONICAL = "EXACT_FIRST_300_CANONICAL"
STATUS_REORDERED = "SAME_TASKS_DIFFERENT_ORDER"
STATUS_PARTIAL = "PARTIAL_SEMANTIC_MATCH"
STATUS_DIFFERENT = "DIFFERENT_PARENT_EXPORT"
STATUS_UNKNOWN = "NOT_IDENTIFIABLE"

# Fields that describe *where a row lives*, not *what the task is*.
_RUNTIME_FIELDS = {
    "split", "split_group_ids", "stage", "source", "sample_id", "task_id",
    "generation_seed", "value_seed", "config_hash", "generator_version",
    "profile_version", "registry_hash", "executor_hash", "validation",
    "student_probe_result", "timestamp", "created_at", "path",
}
_REF_RE = re.compile(r"\$var_?(\d+)(\.[A-Za-z0-9_]+)?\$")


# ── canonicalisation ──────────────────────────────────────────────────────
def _canon_float(x: Any) -> Any:
    """Stable float serialisation so 3.0 and 3 do not diverge across exports."""
    if isinstance(x, bool):
        return x
    if isinstance(x, float):
        if x == int(x) and abs(x) < 1e15:
            return int(x)
        return round(x, 9)
    return x


def _canon_value(v: Any, *, normalize_refs: bool = False) -> Any:
    if isinstance(v, dict):
        return {str(k): _canon_value(v[k], normalize_refs=normalize_refs)
                for k in sorted(v.keys())}
    if isinstance(v, list):
        return [_canon_value(x, normalize_refs=normalize_refs) for x in v]
    if isinstance(v, str):
        s = re.sub(r"\s+", " ", v).strip()
        if normalize_refs:
            s = _REF_RE.sub(lambda m: f"$var{m.group(1)}.OUT$", s)
        return s
    return _canon_float(v)


def _canon_calls(calls: Sequence[Dict[str, Any]], *, normalize_refs: bool) -> List[Dict[str, Any]]:
    """Call ORDER is semantically meaningful and is preserved."""
    out = []
    for c in calls or []:
        label = str(c.get("label") or "")
        if normalize_refs:
            label = re.sub(r"\$?var_?(\d+)\$?", r"var\1", label)
        out.append({
            "name": str(c.get("name") or ""),
            "arguments": _canon_value(c.get("arguments") or {}, normalize_refs=normalize_refs),
            "label": label,
        })
    return out


def _canon_tool(tool: Dict[str, Any]) -> Dict[str, Any]:
    params = tool.get("parameters")
    if isinstance(params, dict) and isinstance(params.get("properties"), dict):
        params = params["properties"]
    if isinstance(params, dict):
        pdesc = {str(k): str((params[k] or {}).get("type") if isinstance(params[k], dict) else "")
                 for k in sorted(params.keys())}
    elif isinstance(params, list):
        pdesc = {str(p.get("name")): str(p.get("type") or "")
                 for p in params if isinstance(p, dict)}
    else:
        pdesc = {}
    outs = tool.get("output_parameters") or {}
    okeys = sorted(str(k) for k in outs.keys()) if isinstance(outs, dict) else []
    return {"name": str(tool.get("name") or ""), "params": pdesc, "output_keys": okeys}


def canonical_payload(row: Dict[str, Any], *, order_sensitive: bool,
                      normalize_refs: bool = False) -> Dict[str, Any]:
    """Stable content view of a task row, independent of export metadata."""
    tools = [_canon_tool(t) for t in (row.get("tools") or []) if isinstance(t, dict)]
    if not order_sensitive:
        tools = sorted(tools, key=lambda t: t["name"])
    calls = row.get("gold_calls") or row.get("output") or row.get("canonical_calls") or []
    prov = row.get("provenance") or {}
    if not isinstance(prov, dict):
        prov = {}
    return {
        "question": _canon_value(row.get("question") or row.get("input") or "",
                                 normalize_refs=normalize_refs),
        "offered_tools": tools,
        "gold_calls": _canon_calls(calls, normalize_refs=normalize_refs),
        "oracle_answer": _canon_value(row.get("gold_answer"), normalize_refs=normalize_refs),
        "semantic_program": {
            "num_calls": int(row.get("num_calls") or len(calls) or 0),
            "motif": str(row.get("motif_type") or row.get("motif") or ""),
            "answer_type": str(row.get("answer_type") or ""),
            "observations": _canon_value(row.get("observations") or row.get("oracle_observations") or []),
        },
        "generation_cell": str(prov.get("generation_cell_id") or row.get("generation_cell_id") or ""),
    }


def fingerprint(payload: Dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def row_fingerprints(row: Dict[str, Any]) -> Dict[str, str]:
    return {
        "exact": fingerprint(canonical_payload(row, order_sensitive=True)),
        "order_insensitive": fingerprint(canonical_payload(row, order_sensitive=False)),
        "semantic": fingerprint(canonical_payload(row, order_sensitive=False, normalize_refs=True)),
        "question_only": hashlib.sha256(
            re.sub(r"\s+", " ", str(row.get("question") or row.get("input") or "")).strip().encode("utf-8")
        ).hexdigest(),
        "calls_only": fingerprint(_canon_calls(
            row.get("gold_calls") or row.get("output") or [], normalize_refs=False)),
    }


# ── identity keys ─────────────────────────────────────────────────────────
IDENTITY_KEYS = [
    "sample_id", "task_id", "semantic_program_id", "semantic_program_family",
    "graph_template_id", "generation_cell_id",
]


def identity_values(row: Dict[str, Any]) -> Dict[str, Optional[str]]:
    prov = row.get("provenance") or {}
    if not isinstance(prov, dict):
        prov = {}
    out: Dict[str, Optional[str]] = {}
    for key in IDENTITY_KEYS:
        val = row.get(key)
        if val is None:
            val = prov.get(key)
        if val is None and key == "semantic_program_id":
            val = prov.get("semantic_program_family")
        out[key] = str(val) if val is not None else None
    return out


# ── artifact discovery ────────────────────────────────────────────────────
@dataclass
class ArtifactInfo:
    path: Path
    sha256: str
    size_bytes: int
    n_rows: int
    schema_fields: List[str] = field(default_factory=list)
    first_sample_id: Optional[str] = None
    last_sample_id: Optional[str] = None
    mtime: Optional[str] = None
    manifest_refs: List[str] = field(default_factory=list)

    def as_dict(self, repo_root: Optional[Path] = None) -> Dict[str, Any]:
        p = str(self.path)
        if repo_root:
            try:
                p = str(self.path.resolve().relative_to(repo_root.resolve())).replace("\\", "/")
            except ValueError:
                pass
        return {
            "path": p,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "n_rows": self.n_rows,
            "schema_fields": self.schema_fields,
            "first_sample_id": self.first_sample_id,
            "last_sample_id": self.last_sample_id,
            "mtime_utc": self.mtime,
            "manifest_refs": self.manifest_refs,
        }


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_rows(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def inspect_artifact(path: Path, manifest_refs: Optional[List[str]] = None) -> ArtifactInfo:
    from datetime import datetime, timezone

    rows = _read_rows(path)
    st = path.stat()
    return ArtifactInfo(
        path=path,
        sha256=sha256_path(path),
        size_bytes=st.st_size,
        n_rows=len(rows),
        schema_fields=sorted(rows[0].keys()) if rows else [],
        first_sample_id=str(rows[0].get("sample_id")) if rows else None,
        last_sample_id=str(rows[-1].get("sample_id")) if rows else None,
        mtime=datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
        manifest_refs=list(manifest_refs or []),
    )


def discover_candidates(repo_root: Path) -> Dict[str, List[Path]]:
    """Find every plausible parent-export / subset / manifest artifact."""
    factory = repo_root / "experiments" / "targeted_tool_data_factory"
    out: Dict[str, List[Path]] = {"parent": [], "subset": [], "manifest": []}
    if not factory.is_dir():
        return out
    for p in factory.rglob("*.jsonl"):
        name = p.name.lower()
        if "train_subset" in name:
            out["subset"].append(p)
        elif name.startswith("train_grpo_") or name.startswith("grpo_train_ready_"):
            out["parent"].append(p)
    for p in factory.rglob("*.json"):
        n = p.name.lower()
        if "manifest" in n or "freeze" in n:
            out["manifest"].append(p)
    for key in out:
        out[key] = sorted(set(out[key]))
    return out


# ── byte-level comparison ─────────────────────────────────────────────────
def byte_level_prefix_match(parent: Path, subset: Path, n: int) -> Dict[str, Any]:
    """Compare the parent's first ``n`` physical lines against the subset file."""
    with parent.open("rb") as fh:
        lines: List[bytes] = []
        for i, raw in enumerate(fh):
            if i >= n:
                break
            lines.append(raw)
    prefix = b"".join(lines)
    subset_bytes = subset.read_bytes()
    exact = sha256_bytes(prefix) == sha256_bytes(subset_bytes)
    # trailing-newline tolerance is reported explicitly, never assumed silently
    normalized = (
        sha256_bytes(prefix.rstrip(b"\r\n")) == sha256_bytes(subset_bytes.rstrip(b"\r\n"))
    )
    return {
        "n_lines_taken": len(lines),
        "parent_prefix_sha256": sha256_bytes(prefix),
        "subset_sha256": sha256_bytes(subset_bytes),
        "exact_bytes_match": exact,
        "match_after_trailing_newline_normalization": normalized and not exact,
        "note": (
            "trailing-newline normalization applied and reported"
            if normalized and not exact else ""
        ),
    }


# ── multi-level audit ─────────────────────────────────────────────────────
def audit_subset(parent_path: Path, subset_path: Path) -> Dict[str, Any]:
    parent_rows = _read_rows(parent_path)
    subset_rows = _read_rows(subset_path)
    n_sub = len(subset_rows)

    byte_res = byte_level_prefix_match(parent_path, subset_path, n_sub)

    # multi-key overlap
    key_overlap: Dict[str, Any] = {}
    for key in IDENTITY_KEYS:
        pv = [identity_values(r)[key] for r in parent_rows]
        sv = [identity_values(r)[key] for r in subset_rows]
        pset = {x for x in pv if x}
        sset = {x for x in sv if x}
        prefix_set = {x for x in pv[:n_sub] if x}
        key_overlap[key] = {
            "subset_defined": sum(1 for x in sv if x),
            "overlap_with_parent_any": len(sset & pset),
            "overlap_with_parent_first_n": len(sset & prefix_set),
            "parent_unique": len(pset),
            "subset_unique": len(sset),
        }

    # fingerprint indexes over the whole parent
    parent_fp: Dict[str, Dict[str, List[int]]] = {k: {} for k in
                                                  ("exact", "order_insensitive", "semantic",
                                                   "question_only", "calls_only")}
    parent_fps: List[Dict[str, str]] = []
    for i, r in enumerate(parent_rows):
        fps = row_fingerprints(r)
        parent_fps.append(fps)
        for k, v in fps.items():
            parent_fp[k].setdefault(v, []).append(i)

    matches: List[Dict[str, Any]] = []
    unmatched: List[Dict[str, Any]] = []
    level_counts = {k: 0 for k in ("exact", "order_insensitive", "semantic",
                                   "question_only", "calls_only", "none")}
    matched_parent_positions: List[int] = []

    for j, srow in enumerate(subset_rows):
        sfp = row_fingerprints(srow)
        hit_level = None
        hit_idx = None
        for level in ("exact", "order_insensitive", "semantic", "calls_only", "question_only"):
            idxs = parent_fp[level].get(sfp[level])
            if idxs:
                hit_level = level
                hit_idx = idxs[0]
                break
        if hit_level is None:
            level_counts["none"] += 1
            diff_fields = _diff_fields(srow, parent_rows[j] if j < len(parent_rows) else {})
            unmatched.append({
                "subset_position": j,
                "subset_sample_id": str(srow.get("sample_id")),
                "differing_fields_vs_same_position_parent": diff_fields,
                "subset_fingerprints": sfp,
            })
        else:
            level_counts[hit_level] += 1
            matched_parent_positions.append(hit_idx)
        matches.append({
            "subset_position": j,
            "subset_sample_id": str(srow.get("sample_id")),
            "match_level": hit_level or "none",
            "parent_position": hit_idx if hit_idx is not None else "",
            "parent_sample_id": (str(parent_rows[hit_idx].get("sample_id"))
                                 if hit_idx is not None else ""),
            "same_sample_id": (
                str(srow.get("sample_id")) == str(parent_rows[hit_idx].get("sample_id"))
                if hit_idx is not None else False
            ),
            "same_position": (hit_idx == j) if hit_idx is not None else False,
        })

    n_matched = n_sub - level_counts["none"]
    in_prefix = sum(1 for p in matched_parent_positions if p < n_sub)
    ordered = matched_parent_positions == sorted(matched_parent_positions)
    identity_ordered = matched_parent_positions == list(range(n_sub))

    if byte_res["exact_bytes_match"]:
        status = STATUS_BYTES
    elif n_matched == n_sub and identity_ordered:
        status = STATUS_CANONICAL
    elif n_matched == n_sub:
        status = STATUS_REORDERED
    elif n_matched >= max(1, int(0.5 * n_sub)):
        status = STATUS_PARTIAL
    elif n_matched == 0:
        status = STATUS_DIFFERENT
    else:
        status = STATUS_PARTIAL

    verdict_note = _verdict_note(status, n_matched, n_sub, in_prefix,
                                 key_overlap["sample_id"]["overlap_with_parent_any"])

    return {
        "schema_version": SCHEMA_VERSION,
        "parent": str(parent_path),
        "subset": str(subset_path),
        "n_parent_rows": len(parent_rows),
        "n_subset_rows": n_sub,
        "byte_level": byte_res,
        "multi_key_overlap": key_overlap,
        "canonical_match_levels": level_counts,
        "n_canonical_matched": n_matched,
        "n_matched_inside_parent_first_n": in_prefix,
        "matched_positions_monotonic": ordered,
        "matched_positions_are_identity_prefix": identity_ordered,
        "status": status,
        "verdict_note": verdict_note,
        "_matches": matches,
        "_unmatched": unmatched,
    }


def _diff_fields(a: Dict[str, Any], b: Dict[str, Any]) -> List[str]:
    keys = (set(a.keys()) | set(b.keys())) - _RUNTIME_FIELDS
    diffs = []
    for k in sorted(keys):
        if _canon_value(a.get(k)) != _canon_value(b.get(k)):
            diffs.append(k)
    return diffs


def _verdict_note(status: str, n_matched: int, n_sub: int, in_prefix: int,
                  sample_id_overlap: int) -> str:
    if status == STATUS_BYTES:
        return ("Subset is byte-identical to the parent's first "
                f"{n_sub} lines. Provenance fully verified.")
    if status == STATUS_CANONICAL:
        return (f"All {n_sub} subset tasks canonically match the parent's first {n_sub} rows "
                f"in the same order, despite sample_id overlap of only {sample_id_overlap}. "
                "Export identifiers differ; the underlying tasks are the same. "
                "Claims that D1 trained on a different dataset are NOT supported.")
    if status == STATUS_REORDERED:
        return (f"All {n_sub} subset tasks exist in the parent export but not as an ordered "
                f"prefix ({in_prefix} of them fall inside the first {n_sub} rows). "
                "Same task population, different ordering or slice.")
    if status == STATUS_PARTIAL:
        return (f"{n_matched}/{n_sub} subset tasks matched the parent by content. "
                "The parent export is related but not the exact source; see unmatched rows.")
    if status == STATUS_DIFFERENT:
        return ("No subset task matched the parent by content. This parent export is not "
                "the source of the subset.")
    return "Insufficient information."


def audit_best_parent(subset_path: Path, parent_paths: Sequence[Path]) -> Dict[str, Any]:
    """Audit against every candidate parent, return the best-matching one."""
    _RANK = {
        STATUS_BYTES: 5, STATUS_CANONICAL: 4, STATUS_REORDERED: 3,
        STATUS_PARTIAL: 2, STATUS_DIFFERENT: 1, STATUS_UNKNOWN: 0,
    }
    results = []
    for p in parent_paths:
        try:
            results.append(audit_subset(p, subset_path))
        except (OSError, json.JSONDecodeError) as exc:
            results.append({
                "parent": str(p), "subset": str(subset_path),
                "status": STATUS_UNKNOWN, "error": str(exc),
                "n_canonical_matched": 0, "canonical_match_levels": {},
                "multi_key_overlap": {}, "byte_level": {}, "_matches": [], "_unmatched": [],
            })
    if not results:
        return {"status": STATUS_UNKNOWN, "reason": "no parent candidates"}
    best = max(results, key=lambda r: (_RANK.get(r.get("status", STATUS_UNKNOWN), 0),
                                       r.get("n_canonical_matched", 0)))
    best["alternatives"] = [
        {"parent": r["parent"], "status": r.get("status"),
         "n_canonical_matched": r.get("n_canonical_matched")}
        for r in results if r["parent"] != best["parent"]
    ]
    return best


def git_history_candidates(repo_root: Path, rel_paths: Sequence[str],
                           cache_dir: Path) -> List[Path]:
    """Materialise every historical revision of ``rel_paths`` into ``cache_dir``.

    A frozen export can be overwritten by a later regeneration run, in which
    case the working tree no longer contains the parent the checkpoint was
    actually trained on. Git still does.
    """
    import subprocess

    cache_dir.mkdir(parents=True, exist_ok=True)
    out: List[Path] = []
    seen: set = set()
    for rel in rel_paths:
        try:
            log = subprocess.run(
                ["git", "log", "--all", "--format=%H", "--", rel],
                cwd=str(repo_root), capture_output=True, text=True, check=False)
        except OSError:
            continue
        for commit in [c.strip() for c in log.stdout.splitlines() if c.strip()]:
            blob = subprocess.run(["git", "show", f"{commit}:{rel}"],
                                  cwd=str(repo_root), capture_output=True, check=False)
            if blob.returncode != 0 or not blob.stdout:
                continue
            digest = sha256_bytes(blob.stdout)
            if digest in seen:
                continue
            seen.add(digest)
            dest = cache_dir / f"{Path(rel).stem}@{commit[:8]}.jsonl"
            dest.write_bytes(blob.stdout)
            out.append(dest)
    return out


def audit_markdown(result: Dict[str, Any], artifacts: List[Dict[str, Any]]) -> str:
    lines = [
        "# PILOT3_PROVENANCE_AUDIT",
        "",
        f"**Status:** `{result.get('status')}`",
        "",
        result.get("verdict_note", ""),
        "",
        "## Compared artifacts",
        "",
        f"- parent: `{result.get('parent')}`",
        f"- subset: `{result.get('subset')}`",
        f"- parent rows: {result.get('n_parent_rows')}",
        f"- subset rows: {result.get('n_subset_rows')}",
        "",
        "## Level 1 — byte-level prefix",
        "",
    ]
    for k, v in (result.get("byte_level") or {}).items():
        lines.append(f"- {k}: `{v}`")
    lines += ["", "## Level 2 — multi-key identity overlap", "",
              "| key | subset defined | overlap (parent any) | overlap (parent first N) |",
              "|---|---:|---:|---:|"]
    for key, stats in (result.get("multi_key_overlap") or {}).items():
        lines.append(
            f"| `{key}` | {stats.get('subset_defined')} | "
            f"{stats.get('overlap_with_parent_any')} | {stats.get('overlap_with_parent_first_n')} |"
        )
    lines += ["", "## Level 3/4 — canonical content match", ""]
    for k, v in (result.get("canonical_match_levels") or {}).items():
        lines.append(f"- `{k}`: {v}")
    lines += [
        "",
        f"- matched total: {result.get('n_canonical_matched')} / {result.get('n_subset_rows')}",
        f"- matched inside parent's first N: {result.get('n_matched_inside_parent_first_n')}",
        f"- matched positions monotonic: {result.get('matched_positions_monotonic')}",
        f"- matched positions are identity prefix: {result.get('matched_positions_are_identity_prefix')}",
        "",
        "## Artifact inventory",
        "",
    ]
    for a in artifacts:
        lines.append(
            f"- `{a.get('path')}` rows={a.get('n_rows')} sha256={str(a.get('sha256'))[:16]}… "
            f"first={a.get('first_sample_id')} last={a.get('last_sample_id')}"
        )
    alts = result.get("alternatives") or []
    if alts:
        lines += ["", "## Alternative parents considered", ""]
        for a in alts:
            lines.append(f"- `{a['parent']}` → {a['status']} ({a['n_canonical_matched']} matched)")
    lines += [
        "",
        "## Interpretation rules applied",
        "",
        "- Low `sample_id` overlap alone is NOT evidence of a different training dataset.",
        "- Export ids depend on generator seed/attempt counters and change on regeneration.",
        "- Only byte-level or canonical content match is treated as identity evidence.",
        "",
    ]
    return "\n".join(lines) + "\n"


# ── orchestrator ──────────────────────────────────────────────────────────
DEFAULT_TRACKED_PARENTS = [
    "experiments/targeted_tool_data_factory/runpod_bundle_pilot3/data/train_grpo_pilot3.jsonl",
    "experiments/targeted_tool_data_factory/runpod_bundle_pilot2/data/train_grpo_pilot2.jsonl",
]


def run_provenance_audit(repo_root: Path, out_dir: Path, *,
                         subset: Optional[Path] = None,
                         parents: Optional[Sequence[Path]] = None,
                         search_git_history: bool = True,
                         cli_args: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Full Phase-A audit: discover, compare at four levels, write artifacts."""
    from . import repro

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    discovered = discover_candidates(repo_root)

    subsets = [subset] if subset else discovered["subset"]
    if not subsets:
        raise FileNotFoundError("no train_subset_*.jsonl candidate found")
    subset_path = Path(subsets[0])

    parent_paths: List[Path] = [Path(p) for p in (parents or discovered["parent"])]
    git_parents: List[Path] = []
    if search_git_history:
        git_parents = git_history_candidates(
            repo_root, DEFAULT_TRACKED_PARENTS, out_dir / "_git_revisions")
        parent_paths += git_parents

    result = audit_best_parent(subset_path, parent_paths)
    matches = result.pop("_matches", [])
    unmatched = result.pop("_unmatched", [])

    artifacts = []
    for p in [subset_path, Path(result.get("parent", subset_path))]:
        try:
            artifacts.append(inspect_artifact(Path(p)).as_dict(repo_root))
        except (OSError, json.JSONDecodeError):
            continue
    for m in discovered["manifest"][:40]:
        try:
            artifacts.append({"path": str(m), "sha256": sha256_path(m),
                              "size_bytes": m.stat().st_size, "n_rows": None,
                              "kind": "manifest"})
        except OSError:
            continue

    result["git_revision_candidates"] = [str(p) for p in git_parents]
    result["n_parent_candidates"] = len(parent_paths)
    result["resolved"] = result.get("status") in (STATUS_BYTES, STATUS_CANONICAL,
                                                  STATUS_REORDERED)
    result["retracts_previous_claim"] = result["resolved"]

    stamp = repro.stamp(repo_root, schema_version=SCHEMA_VERSION, cli_args=cli_args,
                        input_paths=[subset_path])
    payload = {"schema_version": SCHEMA_VERSION, "repro": stamp,
               "audit": result, "artifacts": artifacts}
    repro.write_json(out_dir / "PILOT3_PROVENANCE_AUDIT.json", payload)
    repro.write_text(out_dir / "PILOT3_PROVENANCE_AUDIT.md",
                     audit_markdown(result, artifacts))
    repro.write_csv(out_dir / "PILOT3_SUBSET_MATCHES.csv", matches,
                    columns=["subset_position", "subset_sample_id", "match_level",
                             "parent_position", "parent_sample_id", "same_sample_id",
                             "same_position"])
    repro.write_jsonl(out_dir / "PILOT3_UNMATCHED_SUBSET.jsonl", unmatched)
    return payload
