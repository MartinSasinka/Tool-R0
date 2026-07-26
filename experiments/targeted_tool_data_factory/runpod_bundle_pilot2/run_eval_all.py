#!/usr/bin/env python3
"""Post-training evaluation for the pilot2 experiment — GPU0-3 in parallel.

Jobs are scheduled across the four GPUs, one job per GPU at a time:

  checkpoints : any subset of C0 (base, no adapter), D0 final, D1 final
  eval sets   : pilot2 structural held-out 80 (factory executor, G/A tracks
                reported separately) and the frozen NESTFUL diagnostic-500

`--arms` selects which checkpoints to evaluate. D0 is optional: the
`--c0-vs-d1` mode of `run_all_4gpu.sh` passes `--arms C0,D1` and omits
`--d0-run`, so no D0 checkpoint is required.

Every job goes through the UNMODIFIED trainer eval path
(`nestful_synthetic_curriculum_v3/run.py --mode final_eval`) with deterministic
decoding forced by explicit overrides, so nothing can silently inherit training
settings. The held-out jobs additionally pin `executor.mode=synthetic` and
`SYNTHETIC_TOOLS_DIR` to the factory adapter.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from queue import Queue

BUNDLE = Path(__file__).resolve().parent
FACTORY = BUNDLE.parent
V3 = FACTORY.parent / "nestful_synthetic_curriculum_v3"
PARTIAL = FACTORY.parent / "nestful_mtgrpo_partial"
RUN_PY = V3 / "run.py"
ADAPTER = FACTORY / "trainer_adapter"

DECODING = [
    "--override", "generation.temperature=0.0",
    "--override", "generation.top_p=1.0",
    "--override", "data.num_eval_rollouts=1",
    "--override", "data.eval_paradigm=react",
]


def find_checkpoint(run_dir: Path) -> Path | None:
    """The published FINAL adapter of a reward-ablation run."""
    for cand in (run_dir / "checkpoints" / "final",
                 run_dir / "final",
                 run_dir / "checkpoints" / "FINAL"):
        if (cand / "adapter_config.json").is_file():
            return cand
    hits = sorted(run_dir.rglob("adapter_config.json"))
    return hits[-1].parent if hits else None


def build_job(label: str, checkpoint: Path | None, eval_set: Path,
              out_dir: Path, synthetic: bool, config: Path) -> dict:
    cmd = [sys.executable, str(RUN_PY), "--mode", "final_eval",
           "--config", str(config),
           "--override", f"experiment.output_dir={out_dir}",
           "--override", f"paths.full_nestful_jsonl={eval_set}",
           *DECODING]
    if synthetic:
        cmd += ["--override", "executor.mode=synthetic"]
    if checkpoint is not None:
        cmd += ["--checkpoint", str(checkpoint)]
    else:
        cmd += ["--override", "model.lora_adapter=null"]
    return {"label": label, "cmd": cmd, "out_dir": out_dir, "synthetic": synthetic,
            "eval_set": str(eval_set),
            "checkpoint": str(checkpoint) if checkpoint else None}


def worker(gpu: int, queue: "Queue[dict]", results: list, lock: threading.Lock,
           dry_run: bool) -> None:
    while True:
        try:
            job = queue.get_nowait()
        except Exception:  # noqa: BLE001  (queue.Empty)
            return
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        if job["synthetic"]:
            env["SYNTHETIC_TOOLS_DIR"] = str(ADAPTER)
        else:
            env.pop("SYNTHETIC_TOOLS_DIR", None)
        job["out_dir"].mkdir(parents=True, exist_ok=True)
        (job["out_dir"] / "eval_manifest.json").write_text(
            json.dumps({k: (v if not isinstance(v, Path) else str(v))
                        for k, v in job.items()}, indent=2, default=str),
            encoding="utf-8")
        print(f"[eval][gpu{gpu}] {job['label']}\n    + {' '.join(job['cmd'])}", flush=True)
        rc = 0
        if not dry_run:
            with (job["out_dir"] / "eval.log").open("w", encoding="utf-8") as log:
                rc = subprocess.run(job["cmd"], env=env, stdout=log,
                                    stderr=subprocess.STDOUT).returncode
        with lock:
            results.append({"label": job["label"], "gpu": gpu, "rc": rc,
                            "out_dir": str(job["out_dir"])})
        print(f"[eval][gpu{gpu}] {job['label']} rc={rc}", flush=True)
        queue.task_done()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--d0-run", default=None,
                    help="run id of the D0 training arm (omit for C0-vs-D1)")
    ap.add_argument("--d1-run", required=True)
    ap.add_argument("--heldout", type=Path, required=True)
    ap.add_argument("--diagnostic", type=Path, required=True)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=PARTIAL / "config.yaml")
    ap.add_argument("--arms", default="C0,D0,D1",
                    help="comma-separated arms to evaluate (subset of C0,D0,D1)")
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arms if a not in ("C0", "D0", "D1")]
    if unknown:
        print(f"[eval] ABORT: unknown arms {unknown}", file=sys.stderr)
        return 2
    if "D0" in arms and not args.d0_run:
        print("[eval] ABORT: --d0-run required when D0 is in --arms", file=sys.stderr)
        return 2
    if "D1" in arms and not args.d1_run:
        print("[eval] ABORT: --d1-run required when D1 is in --arms", file=sys.stderr)
        return 2

    ckpts: dict[str, Path | None] = {}
    for label in arms:
        if label == "C0":
            ckpts["C0"] = None
            print("[eval] C0 checkpoint: base model (no adapter)")
            continue
        run_id = args.d0_run if label == "D0" else args.d1_run
        run_dir = args.output_root / run_id
        ck = find_checkpoint(run_dir)
        if ck is None and not args.dry_run:
            print(f"[eval] ABORT: no final adapter under {run_dir}", file=sys.stderr)
            return 2
        ckpts[label] = ck
        print(f"[eval] {label} checkpoint: {ck}")

    queue: "Queue[dict]" = Queue()
    for label, ck in ckpts.items():
        queue.put(build_job(f"{label}_heldout80", ck, args.heldout,
                            args.results / "eval" / f"{label}_heldout80",
                            synthetic=True, config=args.config))
        queue.put(build_job(f"{label}_nestful500", ck, args.diagnostic,
                            args.results / "eval" / f"{label}_nestful500",
                            synthetic=False, config=args.config))

    gpus = [int(g) for g in args.gpus.split(",") if g.strip() != ""]
    results: list = []
    lock = threading.Lock()
    threads = [threading.Thread(target=worker, args=(g, queue, results, lock, args.dry_run))
               for g in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    summary = {"arms": arms, "jobs": results,
               "failed": [r["label"] for r in results if r["rc"] != 0]}
    (args.results / "eval_jobs.json").write_text(json.dumps(summary, indent=2),
                                                 encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if summary["failed"]:
        print(f"[eval] FAILED jobs: {summary['failed']}", file=sys.stderr)
        return 1
    print("[eval] all jobs finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
