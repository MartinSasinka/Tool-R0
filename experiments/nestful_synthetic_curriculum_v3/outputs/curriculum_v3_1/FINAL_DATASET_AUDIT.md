# Final Dataset Audit (v3.1)

Overall status: **WARN**
Pilot decision: **READY_FOR_POD_DRY_RUN**
Source: `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\nestful_synthetic_curriculum_v3\outputs\curriculum_v3_1\filtered`

## Per-stage summary

| Stage | Status | N | Call integrity | Exact dup | Trace dup ratio | UQ ratio | NS share | Replay |
|---|---|---:|---|---:|---:|---:|---:|---:|
| stage1 1call atomic | WARN | 800 | PASS | 0 | 0.0000 | 1.0000 | 0.6200 | 1.0000 |
| stage2 2call dependency | WARN | 800 | PASS | 0 | 0.0013 | 1.0000 | 0.5513 | 1.0000 |
| stage3 3call composition | WARN | 800 | PASS | 0 | 0.0000 | 1.0000 | 0.1575 | 1.0000 |
| stage4 4to6call persiste | WARN | 800 | PASS | 0 | 0.0000 | 1.0000 | 0.2687 | 1.0000 |

## Hard failures (aggregate)

- (none)

## Soft warnings (aggregate)

- used_tool_names=15<22
- used_tool_names=14<22
- used_tool_names=18<22
- non_scalar_share=0.1575<0.30
- non_scalar_share=0.2687<0.30

## Global
- used tool names: 25
- offered tool names: 37
- stage2+ non-scalar share: 0.3258
- preflight: PASS_PILOT_READY

See `FINAL_PILOT_READINESS_REPORT.md` for pilot commands.
