"""Staged writer/critic runner for Pilot4.3 query rendering.

The failure mode this module exists to prevent is a *full run spent on a broken
prompt*. Pilot4.2 generated the whole corpus before anyone measured whether the
writer obeyed the contract, and the answer turned out to be "often not". Here
nothing advances until a 50-sample smoke stage clears explicit gates, and the
gate report is machine-readable so the decision is not a human reading a log.

The second guard is *agreement theatre*: a single critic from the writer's own
family passes almost everything. Tasks that are expensive to get wrong (long
programs, enrichment tiers, anything rewritten, anything the first critic was
not sure about) always get a second critic from a third model family, and any
PASS-versus-not disagreement blocks the task from selection instead of being
resolved by majority.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from . import (LLM_WRITER_MODES, QUERY_MODES, RUN_ID, TIER_CAPABILITY,
               TIER_CHALLENGE, TIER_PROFILE_CORE)
from . import qvalidate
from .orclient import (BudgetExceeded, OpenRouterClient, ReplayMiss,
                       RunIsolationError, StructuredOutputError, TransportError,
                       count_foreign_run_records)
from .orprompts import critic_prompt, rewrite_prompt, writer_prompt

RENDER_LOG = "llm_rendered.jsonl"
DISAGREEMENT_LOG = "critic_disagreements.jsonl"
GATE_REPORT = "stage_gate_pilot43_{stage}.json"

STAGES = ("smoke", "pilot", "full")
STAGE_SIZES = {"smoke": 50, "pilot": 300, "full": 0}
MAX_REWRITES = 1
#: one further rewrite when the two critics disagree, which spec 16 permits as a
#: resolution. An unresolved disagreement still blocks the task.
DISAGREEMENT_REWRITES = 1

SMOKE_GATES = {
    "structured_output_pass_rate": 0.95,
    "deterministic_pass_rate": 0.90,
    "critic_pass_rate": 0.85,
}

BLOCKING_VERDICTS = ("REWRITE", "REJECT")


class StageGateFailed(RuntimeError):
    """A stage gate failed; the next stage must not start."""


# ── task view ────────────────────────────────────────────────────────────
@dataclass
class RenderTask:
    """Everything the LLM layer needs about one (task, mode) pair.

    The three payloads are deliberately separate: the writer must never see the
    program, the deterministic validator needs the flat contract, and the critic
    needs the executed oracle. Passing one dict for all three is how Pilot4.2
    leaked node ids into writer prompts.
    """

    task_id: str
    requested_mode: str
    workflow_id: str = ""
    semantic_program_id: str = ""
    tier: str = TIER_PROFILE_CORE
    call_count: int = 0
    coding_call_count: int = 0
    answer_type: str = "float"
    prompt_contract: Dict[str, Any] = field(default_factory=dict)
    validator_contract: Dict[str, Any] = field(default_factory=dict)
    critic_context: Dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.task_id}|{self.requested_mode}"

    @property
    def call_bucket(self) -> str:
        return "6+" if self.call_count >= 6 else str(self.call_count)

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "RenderTask":
        known = {f for f in cls.__dataclass_fields__}          # noqa: SLF001
        return cls(**{k: v for k, v in row.items() if k in known})


# ── stratified stage selection ───────────────────────────────────────────
REQUIRED_SMOKE_STRATA: Tuple[str, ...] = tuple(
    f"mode:{m}" for m in QUERY_MODES) + (
    "calls:6+", "coding", "answer:boolean", "answer:nonscalar")
MIN_PER_STRATUM = 2


def strata_of(task: RenderTask) -> List[str]:
    out = [f"mode:{task.requested_mode}", f"calls:{task.call_bucket}"]
    if task.call_count >= 6:
        out.append("calls:6+")
    if task.coding_call_count > 0:
        out.append("coding")
    if task.answer_type == "boolean":
        out.append("answer:boolean")
    if task.answer_type in ("string", "list", "object"):
        out.append("answer:nonscalar")
    return out


def _shuffled(tasks: Sequence[RenderTask], seed: int) -> List[RenderTask]:
    """Stable pseudo-random order; the same pool always yields the same stage."""
    return sorted(tasks, key=lambda t: random.Random(f"{seed}:{t.key}").random())


def select_stage(stage: str, tasks: Sequence[RenderTask], *,
                 seed: int = 0) -> List[RenderTask]:
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}")
    if stage == "full":
        return list(tasks)
    n = STAGE_SIZES[stage]
    ordered = _shuffled(tasks, seed)
    chosen: Dict[str, RenderTask] = {}
    # cover the strata that the gates are supposed to speak about first
    for stratum in REQUIRED_SMOKE_STRATA:
        picked = 0
        for task in ordered:
            if picked >= MIN_PER_STRATUM or len(chosen) >= n:
                break
            if task.key in chosen or stratum not in strata_of(task):
                continue
            chosen[task.key] = task
            picked += 1
    # then fill round-robin over modes so no mode dominates the remainder
    by_mode: Dict[str, List[RenderTask]] = {m: [] for m in QUERY_MODES}
    for task in ordered:
        by_mode.setdefault(task.requested_mode, []).append(task)
    cursors = {m: 0 for m in by_mode}
    while len(chosen) < n and any(cursors[m] < len(by_mode[m]) for m in by_mode):
        for mode in sorted(by_mode):
            if len(chosen) >= n:
                break
            pool = by_mode[mode]
            while cursors[mode] < len(pool):
                task = pool[cursors[mode]]
                cursors[mode] += 1
                if task.key not in chosen:
                    chosen[task.key] = task
                    break
    return [t for t in ordered if t.key in chosen]


# ── second-critic routing (spec 16) ──────────────────────────────────────
def needs_second_critic(task: RenderTask, *, first_verdict: Optional[str],
                        rewritten: bool, sample_rate: float = 0.10,
                        run_id: str = RUN_ID) -> Tuple[bool, str]:
    """Returns ``(routed, reason)``. Mandatory routes come before the sample.

    Cost-efficient routing (resume phase): every 6+ task no longer pays for a
    second critic. The strong critic covers challenge, long programs, coding
    6+, non-scalar 6+, rewrites and uncertain first-critic cases, plus a 10 %
    random sample of the remainder.
    """
    if task.tier == TIER_CHALLENGE:
        return True, "tier:CHALLENGE"
    if task.call_count >= 8:
        return True, "call_count_8plus"
    if task.call_count >= 6 and task.coding_call_count > 0:
        return True, "coding_6plus"
    if task.call_count >= 6 and task.answer_type in ("string", "list", "object"):
        return True, "structured_answer_6plus"
    if task.tier == TIER_CAPABILITY and task.call_count >= 6:
        return True, "tier:CAPABILITY_ENRICHMENT_6plus"
    if rewritten:
        return True, "rewritten"
    if first_verdict != "PASS":
        return True, "first_critic_not_pass"
    seed = f"{run_id}:second-critic:{task.task_id}:{task.requested_mode}"
    if random.Random(seed).random() < sample_rate:
        return True, "random_sample"
    return False, ""


#: the per-field answers a critic has to give. A verdict that contradicts them
#: is not evidence of anything, which spec 15 requires it to be.
CRITIC_CHECKS: Tuple[str, ...] = (
    "workflow_matches_query", "sink_answers_target", "all_query_facts_used",
    "all_program_nodes_required", "no_extra_conditions",
    "units_semantically_valid", "query_unambiguous", "query_natural",
    "graph_not_disclosed",
)
UNCERTAIN = "UNCERTAIN"


def critic_evidence(critic: Optional[Mapping[str, Any]]) -> List[str]:
    """The checks the critic itself marked as failing, node alignment included."""
    if not critic:
        return []
    failed = [name for name in CRITIC_CHECKS if critic.get(name) is False]
    failed += [f"node:{n.get('node_id')}"
               for n in critic.get("node_alignment") or []
               if n.get("aligned") is False]
    return failed


def effective_verdict(critic: Optional[Mapping[str, Any]]) -> str:
    """The verdict its own evidence supports, which is what the gate uses.

    Critics occasionally answer every check ``true`` and still reject, or list a
    failing check and still pass. Neither is usable as evidence: the first is a
    rejection with nothing behind it, the second a pass that contradicts itself.
    A supported rejection stands, an unsupported one becomes ``UNCERTAIN`` and is
    settled by the second critic, and a contradicted pass never counts as a pass.
    """
    if not critic:
        return ""
    verdict = str(critic.get("verdict") or "")
    if critic_evidence(critic):
        return verdict if verdict in BLOCKING_VERDICTS else "REJECT"
    return verdict if verdict == "PASS" else UNCERTAIN


def disagreement(first: Optional[str], second: Optional[str]) -> bool:
    """PASS on one side and a blocking verdict on the other."""
    if not first or not second:
        return False
    return ((first == "PASS" and second in BLOCKING_VERDICTS)
            or (second == "PASS" and first in BLOCKING_VERDICTS))


# ── one task ─────────────────────────────────────────────────────────────
def _meta(task: RenderTask) -> Dict[str, Any]:
    return {"sample_id": task.task_id, "workflow_id": task.workflow_id,
            "semantic_program_id": task.semantic_program_id}


def _call(client: OpenRouterClient, purpose: str, prompt: Any,
          task: RenderTask) -> Dict[str, Any]:
    messages = [{"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user}]
    meta = {**_meta(task), "prompt_version": prompt.prompt_version}
    return client.chat(purpose, messages, prompt.schema, meta)


def render_one(task: RenderTask, client: OpenRouterClient, *,
               sample_rate: float = 0.20,
               max_rewrites: int = MAX_REWRITES,
               disagreement_rewrites: int = DISAGREEMENT_REWRITES
               ) -> Dict[str, Any]:
    """Write, validate, criticise, optionally rewrite and re-criticise."""
    record: Dict[str, Any] = {
        "run_id": client.cfg.run_id,
        "task_id": task.task_id,
        "requested_mode": task.requested_mode,
        "workflow_id": task.workflow_id,
        "semantic_program_id": task.semantic_program_id,
        "tier": task.tier,
        "call_count": task.call_count,
        "answer_type": task.answer_type,
        "query": "",
        "model": client.cfg.model_for("writer"),
        "prompt_version": "",
        "attempts": 0,
        "structured_output_ok": False,
        "validation": {},
        "critic": None,
        "second_critic": None,
        "second_critic_reason": "",
        "rewrite_history": [],
        "disagreement": False,
        "blocked": True,
        "blocked_reason": "not_rendered",
        "error": "",
    }
    try:
        written = _call(client, "writer", writer_prompt(task.prompt_contract,
                                                        task.requested_mode),
                        task)
    except (StructuredOutputError, TransportError, ReplayMiss) as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["blocked_reason"] = "writer_failed"
        return record

    record["attempts"] = int(written.get("attempts") or 0)
    record["structured_output_ok"] = True
    record["prompt_version"] = written["record"]["prompt_version"]
    record["model"] = written["record"]["actual_model"]
    query = str(written["parsed"].get("query") or "").strip()
    record["query"] = query

    validation = qvalidate.validate_query(query, task.validator_contract)
    record["validation"] = validation
    critic = _critique(client, task, query, validation, second=False)
    record["critic"] = critic

    rewrites = 0
    while rewrites < max_rewrites and _needs_rewrite(validation, critic):
        rewrites += 1
        try:
            repaired = _call(client, "rewrite",
                             rewrite_prompt(task.prompt_contract,
                                            task.requested_mode, query,
                                            critic or {}, validation), task)
        except (StructuredOutputError, TransportError, ReplayMiss) as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            break
        record["attempts"] += int(repaired.get("attempts") or 0)
        new_query = str(repaired["parsed"].get("query") or "").strip()
        record["rewrite_history"].append({
            "attempt": rewrites, "previous_query": query, "query": new_query,
            "changes_made": list(repaired["parsed"].get("changes_made") or []),
            "prompt_version": repaired["record"]["prompt_version"],
        })
        query = new_query
        record["query"] = query
        validation = qvalidate.validate_query(query, task.validator_contract)
        record["validation"] = validation
        critic = _critique(client, task, query, validation, second=False)
        record["critic"] = critic

    first_verdict = effective_verdict(critic)
    routed, reason = needs_second_critic(
        task, first_verdict=first_verdict, rewritten=bool(rewrites),
        sample_rate=sample_rate, run_id=client.cfg.run_id)
    record["second_critic_reason"] = reason
    second: Optional[Dict[str, Any]] = None
    if routed:
        second = _critique(client, task, query, validation, second=True)
        record["second_critic"] = second
        record["disagreement"] = disagreement(first_verdict,
                                              effective_verdict(second))

    # The spec allows a disagreement to be resolved by a rewrite; only an
    # unresolved one blocks. Repairing against the dissenting critic's findings
    # is cheaper than discarding a sample whose only fault the critics dispute.
    if record["disagreement"] and disagreement_rewrites > 0:
        dissent = second if effective_verdict(second) != "PASS" else critic
        query, validation, critic, second = _repair_disagreement(
            client, task, query, validation, dissent or {}, record)
        record["disagreement"] = disagreement(effective_verdict(critic),
                                              effective_verdict(second))

    record["blocked"], record["blocked_reason"] = _blocked(record)
    return record


def _repair_disagreement(client: OpenRouterClient, task: RenderTask, query: str,
                         validation: Dict[str, Any],
                         dissent: Mapping[str, Any], record: Dict[str, Any]
                         ) -> Tuple[str, Dict[str, Any],
                                    Optional[Dict[str, Any]],
                                    Optional[Dict[str, Any]]]:
    """One rewrite against the dissenting critic, then both critics again."""
    try:
        repaired = _call(client, "rewrite",
                         rewrite_prompt(task.prompt_contract,
                                        task.requested_mode, query,
                                        dissent, validation), task)
    except (StructuredOutputError, TransportError, ReplayMiss) as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        return query, validation, record.get("critic"), record.get("second_critic")

    record["attempts"] += int(repaired.get("attempts") or 0)
    new_query = str(repaired["parsed"].get("query") or "").strip()
    new_validation = qvalidate.validate_query(new_query, task.validator_contract)
    if validation.get("passed") and not new_validation.get("passed"):
        # satisfying one critic by dropping a stated fact is not a repair. The
        # rewrite is discarded and the disagreement is left to block the task.
        record["rewrite_history"].append({
            "attempt": len(record["rewrite_history"]) + 1,
            "reason": "critic_disagreement", "discarded": True,
            "previous_query": query, "query": new_query,
            "failed_layers": list(new_validation.get("failed_layers") or []),
            "changes_made": list(repaired["parsed"].get("changes_made") or []),
            "prompt_version": repaired["record"]["prompt_version"],
        })
        return query, validation, record.get("critic"), record.get("second_critic")

    record["rewrite_history"].append({
        "attempt": len(record["rewrite_history"]) + 1,
        "reason": "critic_disagreement",
        "previous_query": query, "query": new_query,
        "changes_made": list(repaired["parsed"].get("changes_made") or []),
        "prompt_version": repaired["record"]["prompt_version"],
    })
    query = new_query
    record["query"] = query
    validation = new_validation
    record["validation"] = validation
    critic = _critique(client, task, query, validation, second=False)
    record["critic"] = critic
    second = _critique(client, task, query, validation, second=True)
    record["second_critic"] = second
    return query, validation, critic, second


def _critique(client: OpenRouterClient, task: RenderTask, query: str,
              validation: Mapping[str, Any], *,
              second: bool) -> Optional[Dict[str, Any]]:
    prompt = critic_prompt(task.critic_context, query, validation,
                           second_opinion=second)
    try:
        out = _call(client, "critic2" if second else "critic", prompt, task)
    except (StructuredOutputError, TransportError, ReplayMiss):
        # an unreachable critic is "uncertain", which routes to the second one
        return None
    return dict(out["parsed"])


def _needs_rewrite(validation: Mapping[str, Any],
                   critic: Optional[Mapping[str, Any]]) -> bool:
    if not validation.get("passed"):
        return True
    return (critic or {}).get("verdict") == "REWRITE"


def _blocked(record: Mapping[str, Any]) -> Tuple[bool, str]:
    if not record.get("structured_output_ok"):
        return True, "writer_failed"
    if not (record.get("validation") or {}).get("passed"):
        return True, "deterministic_validation_failed"
    if record.get("disagreement"):
        return True, "critic_disagreement"
    first = effective_verdict(record.get("critic"))
    second = effective_verdict(record.get("second_critic"))
    if first in BLOCKING_VERDICTS:
        return True, f"critic_verdict:{first}"
    if record.get("second_critic_reason") and record.get("second_critic") is None:
        # routed to a second critic that never answered: routing coverage is a
        # hard gate, so missing evidence blocks rather than defaults to PASS
        return True, "second_critic_unavailable"
    if second in BLOCKING_VERDICTS:
        return True, f"second_critic_verdict:{second}"
    if first == UNCERTAIN and second != "PASS":
        # a rejection with no evidence behind it is settled by the second critic
        return True, "critic_uncertain_unsettled"
    if second == UNCERTAIN and first != "PASS":
        return True, "second_critic_uncertain_unsettled"
    return False, ""


# ── gates ────────────────────────────────────────────────────────────────
def gate_report(stage: str, records: Sequence[Mapping[str, Any]], *,
                mixed_run_log_records: int = 0,
                gates: Mapping[str, float] = SMOKE_GATES) -> Dict[str, Any]:
    """Machine-readable stage verdict. ``may_advance`` is the only decision."""
    n = len(records)
    denom = float(n or 1)
    structured = sum(1 for r in records if r.get("structured_output_ok"))
    deterministic = sum(1 for r in records
                        if (r.get("validation") or {}).get("passed"))
    critic_pass = sum(1 for r in records
                      if effective_verdict(r.get("critic")) == "PASS")
    metrics = {
        "n": n,
        "structured_output_pass_rate": round(structured / denom, 4),
        "deterministic_pass_rate": round(deterministic / denom, 4),
        "critic_pass_rate": round(critic_pass / denom, 4),
        "mixed_run_log_records": int(mixed_run_log_records),
        "disagreement_rate": round(
            sum(1 for r in records if r.get("disagreement")) / denom, 4),
        "second_critic_rate": round(
            sum(1 for r in records if r.get("second_critic") is not None)
            / denom, 4),
        "blocked_rate": round(sum(1 for r in records if r.get("blocked"))
                              / denom, 4),
        "routed_second_critic_answered_rate": _answered_rate(records),
    }
    checks = [{"name": name, "value": metrics[name], "threshold": threshold,
               "comparison": ">=", "passed": n > 0 and metrics[name] >= threshold}
              for name, threshold in sorted(gates.items())]
    checks.append({"name": "mixed_run_log_records",
                   "value": metrics["mixed_run_log_records"], "threshold": 0,
                   "comparison": "==",
                   "passed": metrics["mixed_run_log_records"] == 0})
    passed = bool(n) and all(c["passed"] for c in checks)
    return {
        "stage": stage,
        "run_id": RUN_ID,
        "metrics": metrics,
        "gates": checks,
        "failed_gates": [c["name"] for c in checks if not c["passed"]],
        "passed": passed,
        "may_advance": passed,
        "next_stage": _next_stage(stage) if passed else None,
    }


def _answered_rate(records: Sequence[Mapping[str, Any]]) -> float:
    """Share of second-critic routes that actually produced a verdict."""
    routed = [r for r in records if r.get("second_critic_reason")]
    if not routed:
        return 1.0
    answered = sum(1 for r in routed if r.get("second_critic") is not None)
    return round(answered / len(routed), 4)


def _next_stage(stage: str) -> Optional[str]:
    idx = STAGES.index(stage)
    return STAGES[idx + 1] if idx + 1 < len(STAGES) else None


# ── the stage runner ─────────────────────────────────────────────────────
def existing_keys(out_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Records already written, keyed by ``task_id|mode`` (resume support)."""
    path = Path(out_dir) / RENDER_LOG
    out: Dict[str, Dict[str, Any]] = {}
    if not path.is_file():
        return out
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[f"{row.get('task_id')}|{row.get('requested_mode')}"] = row
    return out


def _append(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _previous_gate(out_dir: Path, stage: str) -> Dict[str, Any]:
    """What the preceding stage's gate said, recorded in this stage's report.

    Spec 14 forbids advancing on a failed gate. Nothing here refuses to run -- an
    operator may need the data anyway -- but the report of a stage always carries
    the verdict it was started on, so an advance can never be a silent one.
    """
    index = STAGES.index(stage)
    if index == 0:
        return {"stage": "", "may_advance": True, "failed_gates": []}
    previous = STAGES[index - 1]
    path = Path(out_dir) / f"stage_gate_{RUN_ID}_{previous}.json"
    if not path.is_file():
        return {"stage": previous, "may_advance": False,
                "failed_gates": ["stage_not_run"]}
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"stage": previous, "may_advance": False,
                "failed_gates": ["stage_report_unreadable"]}
    return {"stage": previous,
            "may_advance": bool(report.get("may_advance")),
            "failed_gates": list(report.get("failed_gates") or []),
            "metrics": report.get("metrics") or {}}


def run_stage(stage: str, tasks: Sequence[Any], client: OpenRouterClient,
              out_dir: Path, *, seed: int = 0,
              sample_rate: Optional[float] = None,
              workers: int = 1,
              progress_every: int = 0) -> Dict[str, Any]:
    """Render one stage. Resumable, budget-aware, gated.

    ``workers`` > 1 renders tasks concurrently. Each task is several dependent
    requests (write, validate, criticise), so the parallelism is across tasks and
    the per-task order is untouched; records are written by the calling thread so
    the log stays one-record-per-line.
    """
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}")
    out_dir = Path(out_dir)
    if out_dir.name != client.cfg.run_id:
        raise RunIsolationError(
            f"output directory {out_dir.name!r} is not {client.cfg.run_id!r}")
    out_dir.mkdir(parents=True, exist_ok=True)
    render_path = out_dir / RENDER_LOG
    disagreement_path = out_dir / DISAGREEMENT_LOG
    entered_on = _previous_gate(out_dir, stage)

    given = [t if isinstance(t, RenderTask) else RenderTask.from_dict(t)
             for t in tasks]
    # the explicit modes are rendered deterministically; dropping them here
    # keeps a mixed pool from crashing the stage
    pool = [t for t in given if t.requested_mode in LLM_WRITER_MODES]
    selected = select_stage(stage, pool, seed=seed)
    done = existing_keys(out_dir)
    rate = (client.cfg.second_critic_sample_rate if sample_rate is None
            else sample_rate)

    records: List[Dict[str, Any]] = [done[t.key] for t in selected
                                     if t.key in done]
    pending = [t for t in selected if t.key not in done]
    stopped = ""

    def _write(record: Dict[str, Any]) -> None:
        _append(render_path, record)
        records.append(record)
        if progress_every and len(records) % progress_every == 0:
            print(f"[{stage}] {len(records)}/{len(selected)} rendered, "
                  f"{client.totals.cost_usd:.4f} USD", flush=True)
        if record.get("disagreement"):
            _append(disagreement_path, {
                "run_id": client.cfg.run_id, "task_id": record["task_id"],
                "requested_mode": record["requested_mode"],
                "query": record.get("query", ""),
                "first_verdict": (record.get("critic") or {}).get("verdict"),
                "second_verdict": (record.get("second_critic") or {}).get(
                    "verdict"),
                "second_critic_reason": record.get("second_critic_reason", ""),
            })

    def _one(task: RenderTask) -> Dict[str, Any]:
        try:
            return render_one(task, client, sample_rate=rate)
        except BudgetExceeded as exc:
            if exc.scope == "total":
                raise
            return {"run_id": client.cfg.run_id, "task_id": task.task_id,
                    "requested_mode": task.requested_mode,
                    "structured_output_ok": False, "validation": {},
                    "critic": None, "second_critic": None, "blocked": True,
                    "blocked_reason": "task_budget_exceeded",
                    "error": str(exc), "rewrite_history": [], "attempts": 0,
                    "disagreement": False}
        except Exception as exc:  # noqa: BLE001
            # Keep the stage alive on a single transport/parse failure; the
            # task is blocked and can be retried after the log is compacted.
            return {"run_id": client.cfg.run_id, "task_id": task.task_id,
                    "requested_mode": task.requested_mode,
                    "structured_output_ok": False, "validation": {},
                    "critic": None, "second_critic": None, "blocked": True,
                    "blocked_reason": "render_exception",
                    "error": f"{type(exc).__name__}: {exc}",
                    "rewrite_history": [], "attempts": 0,
                    "disagreement": False}

    if workers > 1 and pending:
        from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

        with ThreadPoolExecutor(max_workers=workers) as executor:
            queue = iter(pending)
            futures: Dict[Any, RenderTask] = {}
            for task in queue:
                futures[executor.submit(_one, task)] = task
                if len(futures) >= workers:
                    break
            while futures:
                finished, _ = wait(list(futures), return_when=FIRST_COMPLETED)
                for future in finished:
                    futures.pop(future)
                    try:
                        _write(future.result())
                    except BudgetExceeded as exc:
                        stopped = f"budget_exceeded: {exc}"
                if stopped:
                    for future in futures:
                        future.cancel()
                    break
                for task in queue:
                    futures[executor.submit(_one, task)] = task
                    if len(futures) >= workers:
                        break
    else:
        for task in pending:
            try:
                _write(_one(task))
            except BudgetExceeded as exc:
                stopped = f"budget_exceeded: {exc}"
                break

    report = gate_report(
        stage, records,
        mixed_run_log_records=count_foreign_run_records(out_dir,
                                                        client.cfg.run_id))
    report["selected"] = len(selected)
    report["skipped_non_writer_modes"] = len(given) - len(pool)
    report["resumed"] = sum(1 for t in selected if t.key in done)
    report["stopped"] = stopped
    report["entered_on"] = entered_on
    report["entered_against_failed_gate"] = not entered_on["may_advance"]
    if stopped:
        report["may_advance"] = False
    report["blocked_task_ids"] = sorted(
        str(r.get("task_id")) for r in records if r.get("blocked"))
    _write_report(out_dir / GATE_REPORT.format(stage=stage), report)
    return report


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False,
                               default=str), encoding="utf-8")


def advance_or_raise(report: Mapping[str, Any]) -> str:
    """Next stage name, or refuse. The only sanctioned way to move forward."""
    if not report.get("may_advance"):
        raise StageGateFailed(
            f"stage {report.get('stage')!r} failed gates "
            f"{report.get('failed_gates')}")
    nxt = report.get("next_stage")
    if not nxt:
        raise StageGateFailed("no stage after 'full'")
    return str(nxt)


def iter_records(out_dir: Path) -> Iterable[Dict[str, Any]]:
    yield from existing_keys(Path(out_dir)).values()
