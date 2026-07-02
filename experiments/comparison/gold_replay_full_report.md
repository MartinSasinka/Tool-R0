# Gold-trace replay (full dataset)

- dataset: `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\nestful_mtgrpo_minimal\data\NESTFUL-main\data_v2\nestful_data.jsonl`
- tasks scored: **1861**
- scoring time: 54.1s

| metric | value | expected |
|---|---|---|
| official_win | 0.9866 | ~1.0 |
| official_full_match | 1.0000 | ~1.0 |
| parse_valid | 1.0000 | ~1.0 |
| executable | 0.9989 | ~1.0 |
| execution_error_rate | 0.0011 | ~0.0 |

- tasks with official_win < 1.0: **25** (see `gold_replay_failures.csv`)

## GATE: PASS

Gold replay reproduces the gold answer via our executor + the official scorer, so Win Rate built on this pipeline is trustworthy.
