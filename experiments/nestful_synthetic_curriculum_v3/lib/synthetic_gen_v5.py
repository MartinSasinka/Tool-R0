"""synthetic_gen_v5 — registry-driven synthetic task generator (curriculum v5).

Replaces the 34-tool v4 generator. Key properties:
  * tools come EXCLUSIVELY from the versioned executable registry
    (lib/synthetic_tools.py); the trainer executes the very same functions in
    executor mode="synthetic", so every generated task is replayable at train
    time by construction;
  * dependency graphs are built only between SEMANTICALLY COMPATIBLE tools
    (money -> money, kg -> kg, text -> text, generic numeric sinks accept any
    number) — no nonsensical compositions;
  * all observations and the gold answer are COMPUTED by executing the chain,
    never authored by an LLM;
  * inverse-frequency tool sampling plus a configurable hard share cap keep any
    single tool from dominating the corpus;
  * object-output tools inject $varN.field$ reference motifs.

CONTAMINATION: no NESTFUL questions, traces, schemas or implementations are
copied; only aggregate style statistics informed the design (see PROVENANCE).
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .synthetic_tools import (
    ALL_TOOL_NAMES,
    REGISTRY_VERSION,
    TOOLS,
    registry_hash,
    semantics_compatible,
    tool_schema,
)

GENERATOR_VERSION = "v5.0"

PROVENANCE = {
    "generator": "synthetic_gen_v5.py",
    "generator_version": GENERATOR_VERSION,
    "registry_version": REGISTRY_VERSION,
    "nestful_sources_used": [
        "AGGREGATE ONLY: call-count distribution, offered-tools-per-task range, "
        "tool/parameter naming style, $varN.output_key$ reference convention."
    ],
    "nestful_content_copied": "NONE (no questions, no gold traces, no tool schemas)",
}

STAGES = {
    "v5_stage1_2call": {"n_calls": (2, 2)},
    "v5_stage2_3call": {"n_calls": (3, 3)},
    "v5_stage3_4call": {"n_calls": (4, 4)},
    "v5_stage4_5to6call": {"n_calls": (5, 6)},
}

MOTIFS = ("long_chain", "argument_binding", "reference_reuse", "distractor_heavy")


@dataclass
class DiversityConfig:
    """Configurable diversity thresholds (see score_dataset script)."""
    max_tool_share: float = 0.08          # max fraction of calls by one tool
    offered_lo: int = 10                  # offered tools per task (normal)
    offered_hi: int = 20
    offered_distractor_lo: int = 16       # distractor_heavy motif range
    offered_distractor_hi: int = 26
    max_pick_attempts: int = 40           # resamples before relaxing the cap


# ─────────────────────────────────────────────────────────────────────────────
#  Semantic compatibility index
# ─────────────────────────────────────────────────────────────────────────────

def _producer_outputs(name: str) -> List[Tuple[Optional[str], str]]:
    """(ref_field, semantic) pairs a tool can offer downstream.

    ref_field None => reference is $varN.<out_key>$ (whole scalar observation);
    otherwise $varN.<field>$ into an object observation."""
    t = TOOLS[name]
    if t["out_type"] == "object" and t["out_fields"]:
        return [(f, sem) for f, (_ft, sem) in t["out_fields"].items()]
    return [(None, t["out_semantic"])]


def _consumers_of(sem: str) -> List[str]:
    """Tools whose chain_in parameter accepts an upstream output of `sem`."""
    out = []
    for name, t in TOOLS.items():
        cin = t["chain_in"]
        if cin is None:
            continue
        if semantics_compatible(sem, t["params"][cin]["semantic"]):
            out.append(name)
    return out


_CONSUMERS_BY_SEM: Dict[str, List[str]] = {}
_CHAIN_STARTERS: List[str] = []
for _name in ALL_TOOL_NAMES:
    for _f, _sem in _producer_outputs(_name):
        if _sem not in _CONSUMERS_BY_SEM:
            _CONSUMERS_BY_SEM[_sem] = _consumers_of(_sem)
        if _CONSUMERS_BY_SEM[_sem]:
            if _name not in _CHAIN_STARTERS:
                _CHAIN_STARTERS.append(_name)
_CHAIN_STARTERS.sort()


def _continuable(name: str) -> bool:
    """Tool's output can feed at least one downstream consumer."""
    return any(_CONSUMERS_BY_SEM.get(sem) for _f, sem in _producer_outputs(name))


# ─────────────────────────────────────────────────────────────────────────────
#  Execution helper (mirror of the trainer's fielded resolution, gold side)
# ─────────────────────────────────────────────────────────────────────────────

def _check_constraints(t: Dict[str, Any], resolved: Dict[str, Any]) -> None:
    """Enforce the SAME min/max/min_len constraints the trainer's synthetic
    executor enforces. Chained values (e.g. an upstream result of 0 fed into a
    param with min=1) would otherwise pass generation but fail train-time
    replay. Raises ValueError so the chain builder resamples."""
    for k, v in resolved.items():
        meta = t["params"].get(k)
        if meta is None:
            raise ValueError(f"{t['name']}: unknown argument {k}")
        typ = meta["type"]
        if typ in ("number", "integer") and isinstance(v, (int, float)) \
                and not isinstance(v, bool):
            if "min" in meta and float(v) < float(meta["min"]):
                raise ValueError(f"{t['name']}.{k} below min")
            if "max" in meta and float(v) > float(meta["max"]):
                raise ValueError(f"{t['name']}.{k} above max")
        if typ == "array" and isinstance(v, list) \
                and "min_len" in meta and len(v) < int(meta["min_len"]):
            raise ValueError(f"{t['name']}.{k} too short")


def execute_gold_calls(gold_calls: List[Dict[str, Any]]) -> List[Any]:
    """Execute a gold chain with fielded $varN[.field]$ resolution. Raises on
    any error — generation must only emit fully executable chains."""
    import re
    ref_re = re.compile(r"^\$([A-Za-z_]\w*)(?:\.(\w+))?\$$")
    scope: Dict[str, Any] = {}
    observations: List[Any] = []
    for call in gold_calls:
        t = TOOLS[call["name"]]
        resolved = {}
        for k, v in call["arguments"].items():
            m = ref_re.match(v.strip()) if isinstance(v, str) else None
            if m:
                var, fld = m.group(1), m.group(2)
                if var not in scope:
                    raise KeyError(f"unresolved reference {v}")
                val = scope[var]
                if fld and isinstance(val, dict):
                    if fld not in val:
                        raise KeyError(f"unresolved field {v}")
                    val = val[fld]
                resolved[k] = val
            else:
                resolved[k] = v
        _check_constraints(t, resolved)
        obs = t["fn"](**resolved)
        scope[call["label"].lstrip("$")] = obs
        observations.append(obs)
    return observations


# ─────────────────────────────────────────────────────────────────────────────
#  Chain construction
# ─────────────────────────────────────────────────────────────────────────────

class _UsageBalancer:
    """Inverse-frequency weighted sampling with a hard share cap."""

    def __init__(self, cfg: DiversityConfig) -> None:
        self.cfg = cfg
        self.counts: Dict[str, int] = {}
        self.total = 0

    def pick(self, rng: random.Random, pool: List[str]) -> str:
        if not pool:
            raise RuntimeError("empty tool pool")
        cap = self.cfg.max_tool_share
        eligible = pool
        if self.total >= 50:  # cap meaningless on a tiny sample
            capped = [n for n in pool
                      if (self.counts.get(n, 0) + 1) / (self.total + 1) <= cap]
            if capped:
                eligible = capped
        weights = [1.0 / (1.0 + self.counts.get(n, 0)) for n in eligible]
        name = rng.choices(eligible, weights=weights, k=1)[0]
        self.counts[name] = self.counts.get(name, 0) + 1
        self.total += 1
        return name


def _ref_str(label: str, t_prev: Dict[str, Any], fld: Optional[str]) -> str:
    key = fld if fld is not None else t_prev["out_key"]
    return f"${label.lstrip('$')}.{key}$"


def _build_chain(rng: random.Random, n_calls: int, motif: str,
                 balancer: _UsageBalancer
                 ) -> Tuple[List[Dict[str, Any]], List[Any], List[str]]:
    """Build one executable, semantically compatible chain.

    Returns (gold_calls, observations, phrases). Raises RuntimeError when the
    sampled path dead-ends (caller retries with fresh randomness)."""
    calls: List[Dict[str, Any]] = []
    observations: List[Any] = []
    phrases: List[str] = []
    scope: Dict[str, Any] = {}
    reuse_slot: Optional[Tuple[str, str, str]] = None  # (label, refstr, sem)

    prev_field: Optional[str] = None
    for i in range(n_calls):
        label = f"$var{i + 1}"
        if i == 0:
            pool = _CHAIN_STARTERS if n_calls > 1 else list(ALL_TOOL_NAMES)
            name = balancer.pick(rng, pool)
            t = TOOLS[name]
            args = t["sample"](rng)
            phrases.append(f"compute {t['phrase'](args)}")
            chosen_field = None
        else:
            prev_name = calls[-1]["name"]
            t_prev = TOOLS[prev_name]
            # Choose which output (field) of the previous call to consume.
            outputs = _producer_outputs(prev_name)
            outputs = [(f, s) for f, s in outputs if _CONSUMERS_BY_SEM.get(s)]
            if not outputs:
                raise RuntimeError(f"dead end after {prev_name}")
            chosen_field, prev_sem = rng.choice(outputs)
            last = i == n_calls - 1
            pool = list(_CONSUMERS_BY_SEM[prev_sem])
            if not last:
                pool = [n for n in pool if _continuable(n)]
            if not pool:
                raise RuntimeError(f"no consumers for {prev_sem}")
            name = balancer.pick(rng, pool)
            t = TOOLS[name]
            args = t["sample"](rng)
            args[t["chain_in"]] = _ref_str(f"var{i}", t_prev, chosen_field)
            display = dict(args)
            display[t["chain_in"]] = "that result"
            # reference_reuse: rebind one extra numeric literal to an EARLIER var
            if motif == "reference_reuse" and reuse_slot and i >= 2:
                for p, meta in t["params"].items():
                    if p == t["chain_in"] or not meta.get("required", True):
                        continue
                    if meta["type"] in ("number", "integer") \
                            and semantics_compatible(reuse_slot[2], meta["semantic"]):
                        args[p] = reuse_slot[1]
                        display[p] = "the first step's value"
                        reuse_slot = None
                        break
            phrases.append(f"use that result to get {t['phrase'](display)}")

        calls.append({"name": name, "arguments": args, "label": label})
        t_cur = TOOLS[name]
        # Execute this step (resolve refs against scope).
        obs = execute_gold_calls(calls)[-1]
        scope[f"var{i + 1}"] = obs
        observations.append(obs)
        if motif == "reference_reuse" and i == 0:
            for f_out, sem_out in _producer_outputs(name):
                if sem_out not in ("flag", "text", "object"):
                    reuse_slot = (f"var{i + 1}",
                                  _ref_str(f"var{i + 1}", t_cur, f_out), sem_out)
                    break
        prev_field = chosen_field
    return calls, observations, phrases


def _question_from_phrases(rng: random.Random, phrases: List[str],
                           n_calls: int) -> str:
    style = rng.choice(["enumerated", "flowing", "imperative"])
    if style == "enumerated":
        steps = [f"{i + 1}) {p}" for i, p in enumerate(phrases)]
        return ("Solve the following in order, using each intermediate result "
                "for the next step: " + "; ".join(steps)
                + f". Report the final value after all {n_calls} steps.")
    if style == "flowing":
        connectors = ["Then", "Next", "After that", "Finally"]
        parts = [phrases[0].capitalize()]
        for i, p in enumerate(phrases[1:]):
            c = connectors[min(i, len(connectors) - 1)] \
                if i < len(phrases) - 2 else "Finally"
            parts.append(f"{c}, {p}")
        return ". ".join(parts) + ". What is the final result?"
    return ("First " + phrases[0] + ", " + ", then ".join(phrases[1:])
            + ". Give the value produced by the last step.")


def _offered_tools(rng: random.Random, used: List[str], n_offered: int) -> List[str]:
    """Used tools + same-domain and off-domain distractors, shuffled."""
    used_set = set(used)
    domains_used = {TOOLS[n]["domain"] for n in used}
    same_domain = [n for n in ALL_TOOL_NAMES
                   if n not in used_set and TOOLS[n]["domain"] in domains_used]
    other = [n for n in ALL_TOOL_NAMES
             if n not in used_set and TOOLS[n]["domain"] not in domains_used]
    rng.shuffle(same_domain)
    rng.shuffle(other)
    need = max(0, n_offered - len(used))
    n_same = min(len(same_domain), max(1, need // 2))
    distractors = same_domain[:n_same] + other[:need - n_same]
    offered = list(used) + distractors
    rng.shuffle(offered)
    return offered


def _answer_type(val: Any) -> str:
    if isinstance(val, bool):
        return "boolean"
    if isinstance(val, (int, float)):
        return "scalar"
    if isinstance(val, str):
        return "string"
    if isinstance(val, list):
        return "list"
    return "object"


def question_hash(q: str) -> str:
    return hashlib.sha256(" ".join(str(q).lower().split()).encode()).hexdigest()


def trace_hash(gold_calls: List[Dict[str, Any]]) -> str:
    canon = json.dumps([[c["name"], c["arguments"]] for c in gold_calls],
                       sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canon.encode()).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
#  Task / stage generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_task(rng: random.Random, stage: str, motif: str, seed: int,
                  idx: int, balancer: _UsageBalancer,
                  cfg: DiversityConfig) -> Dict[str, Any]:
    lo, hi = STAGES[stage]["n_calls"]
    n_calls = rng.randrange(lo, hi + 1)
    for _attempt in range(cfg.max_pick_attempts):
        try:
            calls, observations, phrases = _build_chain(rng, n_calls, motif, balancer)
            break
        except (RuntimeError, KeyError, ZeroDivisionError, ValueError,
                ArithmeticError):
            continue
    else:
        raise RuntimeError(f"could not build a {n_calls}-call chain after "
                           f"{cfg.max_pick_attempts} attempts")
    question = _question_from_phrases(rng, phrases, n_calls)
    gold_answer = observations[-1]
    used = [c["name"] for c in calls]
    if motif == "distractor_heavy":
        n_offered = rng.randrange(cfg.offered_distractor_lo, cfg.offered_distractor_hi)
    else:
        n_offered = rng.randrange(cfg.offered_lo, cfg.offered_hi)
    offered = _offered_tools(rng, used, n_offered)
    sid = f"v5_{stage}_{motif}_{seed}_{idx:05d}_{question_hash(question)[:8]}"
    return {
        "sample_id": sid,
        "question": question,
        "tools": [tool_schema(n) for n in offered],
        "gold_calls": calls,
        "observations": observations,
        "gold_answer": gold_answer,
        "num_calls": n_calls,
        "stage": stage,
        "motif_type": motif,
        "answer_type": _answer_type(gold_answer),
        "terminal_stage": True,
        "source": "curriculum_v5_registry",
        "generation_seed": seed,
        "registry_version": REGISTRY_VERSION,
        "registry_hash": registry_hash(),
        "provenance": PROVENANCE,
    }


def replay_row(row: Dict[str, Any]) -> Tuple[bool, Any]:
    """Re-execute gold_calls; True iff final observation equals gold_answer."""
    try:
        obs = execute_gold_calls(row["gold_calls"])
    except Exception as exc:  # noqa: BLE001
        return False, f"replay_error: {type(exc).__name__}: {exc}"
    return obs[-1] == row["gold_answer"], obs[-1]


def generate_stage(stage: str, n_examples: int, seed: int,
                   cfg: Optional[DiversityConfig] = None,
                   forbidden_question_hashes: Optional[set] = None,
                   forbidden_trace_hashes: Optional[set] = None,
                   ) -> List[Dict[str, Any]]:
    """Deterministic per-stage generation with dedup + per-row replay gate."""
    cfg = cfg or DiversityConfig()
    rng = random.Random(f"{GENERATOR_VERSION}|{REGISTRY_VERSION}|{stage}|{seed}")
    balancer = _UsageBalancer(cfg)
    rows: List[Dict[str, Any]] = []
    seen_q: set = set(forbidden_question_hashes or set())
    seen_t: set = set(forbidden_trace_hashes or set())
    attempts = 0
    max_attempts = n_examples * 60
    while len(rows) < n_examples and attempts < max_attempts:
        attempts += 1
        motif = MOTIFS[len(rows) % len(MOTIFS)]
        try:
            row = generate_task(rng, stage, motif, seed, len(rows), balancer, cfg)
        except RuntimeError:
            continue
        qh, th = question_hash(row["question"]), trace_hash(row["gold_calls"])
        if qh in seen_q or th in seen_t:
            continue
        ok, _obs = replay_row(row)
        if not ok:
            raise RuntimeError(
                f"gold replay failed during generation: {row['sample_id']}")
        seen_q.add(qh)
        seen_t.add(th)
        rows.append(row)
    if len(rows) < n_examples:
        raise RuntimeError(
            f"{stage}: exhausted {max_attempts} attempts at "
            f"{len(rows)}/{n_examples} unique examples")
    return rows
