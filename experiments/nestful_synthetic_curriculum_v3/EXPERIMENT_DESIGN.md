# Experiment Design — NESTFUL Synthetic Curriculum v3

## Research hypothesis

Synthetic curriculum can improve nested tool-use generalization on NESTFUL if synthetic tasks match real NESTFUL **structural motifs**, not only call count.

## Main idea

Previous curriculum: N-call based (`epoch_1_1call.jsonl` … `epoch_4_4call.jsonl`).

New curriculum: **motif-aligned** — dependency graph, reference reuse, fan-in/out, object/list outputs, argument transformation, distractor tools, long chains, alternative traces, baseline-failure-inspired motifs.

## Experimental protocol

### Train
100% synthetic curriculum v3 (`outputs/curriculum_v3/`).

### Dev
Real NESTFUL dev split (`nestful_dev.jsonl`, n=200). Used for:
- motif analysis input (optional `--split dev`)
- failure-mode mining (abstract recipes only)
- checkpoint selection & early stopping

Dev tasks are **never** copied into training JSONL.

### Test
Real NESTFUL test split (`nestful_test.jsonl`, n=1661). Final eval only. Baseline recomputed on same split.

## Contribution variants

**Positive:** motif-aligned synthetic curriculum improves ReAct Win on held-out test.

**Mixed:** motif alignment stabilizes training / reduces trace drift but does not beat baseline.

**Negative:** N-call synthetic curriculum insufficient; transfer requires deeper structural alignment.

## Non-goals

- Do not claim SOTA without beating baseline Win on test.
- Do not optimize on test.
- Do not use F1 Func as headline metric.
- Do not use real test in training loop.
