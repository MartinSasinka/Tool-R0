# Weak audit finalized

## Status

`weak_audit_finalized` — artifacts frozen for strong-model handoff.

## Interpretation

- Weak annotations are **hypotheses**, not ground truth.
- `first_divergence_turn` is relatively more stable across Pass A/B.
- `root_cause` and `recommended_fix` labels are **less stable**.
- Final artifacts are ready for a subsequent strong-model review phase.
- Any further changes require a **new audit version ID**.

## Counts

- Case packets: 248
- Pass A final valid: 248
- Pass B final valid: 248
- Pass A final invalid: 0
- Pass B final invalid: 0
