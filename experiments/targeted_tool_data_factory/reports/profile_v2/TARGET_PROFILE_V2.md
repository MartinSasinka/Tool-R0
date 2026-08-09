# TARGET_PROFILE_V2

- source: `nestful_dev_200`
- mode: **PROFILE_SAFE**
- rows: 200
- schema: `ttdf.target_profile.v2`

## Call-count distribution

- `2`: 0.3300
- `3`: 0.2200
- `4`: 0.1350
- `5`: 0.0950
- `6+`: 0.2200

## Topology diversity inside each call bucket

| bucket | n | distinct topologies | top1 share | norm. entropy | join | multi-join | fan-out | reuse | late-ref |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 66 | 1 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 3 | 44 | 4 | 0.682 | 0.625 | 0.227 | 0.000 | 0.000 | 0.000 | 0.250 |
| 4 | 27 | 2 | 0.630 | 0.951 | 0.630 | 0.000 | 0.000 | 0.000 | 0.630 |
| 5 | 19 | 4 | 0.316 | 0.989 | 0.789 | 0.263 | 0.000 | 0.000 | 0.789 |
| 6+ | 44 | 26 | 0.136 | 0.926 | 1.000 | 0.455 | 0.000 | 0.000 | 1.000 |

## Conditional distributions

### `P(motif|call_count)`

- `2`: linear=1.0
- `3`: fan_in=0.22727, linear=0.68182, mixed=0.09091
- `4`: fan_in=0.62963, linear=0.37037
- `5`: fan_in=0.52632, linear=0.21053, multi_join=0.26316
- `6+`: fan_in=0.54545, multi_join=0.45455

### `P(depth|call_count)`

- `2`: 1=1.0
- `3`: 1=0.31818, 2=0.68182
- `4`: 2=0.62963, 3=0.37037
- `5`: 2=0.47368, 3=0.31579, 4+=0.21053
- `6+`: 3=0.38636, 4+=0.61364

### `P(join_count|call_count)`

- `2`: 0=1.0
- `3`: 0=0.77273, 1=0.22727
- `4`: 0=0.37037, 1=0.62963
- `5`: 0=0.21053, 1=0.52632, 2=0.26316
- `6+`: 1=0.54545, 2=0.18182, 3=0.15909, 4+=0.11364

### `P(fan_out_count|call_count)`

- `2`: 0=1.0
- `3`: 0=1.0
- `4`: 0=1.0
- `5`: 0=1.0
- `6+`: 0=1.0

### `P(reuse_count|call_count)`

- `2`: 0=1.0
- `3`: 0=1.0
- `4`: 0=1.0
- `5`: 0=1.0
- `6+`: 0=1.0

### `P(reference_density|call_count)`

- `2`: 0-0.25=0.40909, 0.25-0.5=0.59091
- `3`: 0-0.25=0.04545, 0.25-0.5=0.93182, 0.5-0.75=0.02273
- `4`: 0.25-0.5=1.0
- `5`: 0.25-0.5=1.0
- `6+`: 0.25-0.5=0.97727, 0.5-0.75=0.02273

### `P(answer_type|call_count)`

- `2`: bool=0.06061, float=0.39394, int=0.13636, list=0.18182, numeric_string=0.06061, string=0.16667
- `3`: float=0.86364, int=0.02273, list=0.04545, string=0.06818
- `4`: float=1.0
- `5`: float=1.0
- `6+`: float=1.0

### `P(offered_tool_count|call_count)`

- `2`: 10-12=0.16667, 13-18=0.5303, 19+=0.09091, <=9=0.21212
- `3`: 10-12=0.43182, 13-18=0.13636, <=9=0.43182
- `4`: 10-12=0.55556, <=9=0.44444
- `5`: 10-12=0.63158, <=9=0.36842
- `6+`: 10-12=0.59091, <=9=0.40909

### `P(query_mode|call_count)`

- `2`: GOAL_BASED_IMPLICIT=0.69697, PROCEDURAL_EXPLICIT=0.09091, PROCEDURAL_PARTIAL=0.13636, UNCLASSIFIED=0.07576
- `3`: GOAL_BASED_IMPLICIT=0.81818, PROCEDURAL_EXPLICIT=0.02273, PROCEDURAL_PARTIAL=0.11364, SEMI_IMPLICIT=0.04545
- `4`: GOAL_BASED_IMPLICIT=0.62963, PROCEDURAL_PARTIAL=0.14815, SEMI_IMPLICIT=0.22222
- `5`: GOAL_BASED_IMPLICIT=0.68421, PROCEDURAL_PARTIAL=0.05263, SEMI_IMPLICIT=0.26316
- `6+`: GOAL_BASED_IMPLICIT=0.63636, PROCEDURAL_PARTIAL=0.13636, SEMI_IMPLICIT=0.22727

### `P(schema_complexity|call_count)`

- `2`: high=0.92424, medium=0.07576
- `3`: high=1.0
- `4`: high=1.0
- `5`: high=1.0
- `6+`: high=1.0

### `P(operation_explicitness|query_mode)`

- `GOAL_BASED_IMPLICIT`: mean=0.0101, p25=0.0, p50=0.0, p75=0.0, min=0.0, max=0.2
- `PROCEDURAL_EXPLICIT`: mean=1.0, p25=1.0, p50=1.0, p75=1.0, min=1.0, max=1.0
- `PROCEDURAL_PARTIAL`: mean=0.5938, p25=0.5, p50=0.5, p75=0.6667, min=0.5, max=1.0
- `SEMI_IMPLICIT`: mean=0.31, p25=0.25, p50=0.2857, p75=0.3333, min=0.2222, max=0.4444
- `UNCLASSIFIED`: mean=0.5, p25=0.5, p50=0.5, p75=0.5, min=0.5, max=0.5

### `P(sequence_leakage|query_mode)`

- `GOAL_BASED_IMPLICIT`: mean=0.0025, p25=0.0, p50=0.0, p75=0.0, min=0.0, max=0.05
- `PROCEDURAL_EXPLICIT`: mean=1.0, p25=1.0, p50=1.0, p75=1.0, min=1.0, max=1.0
- `PROCEDURAL_PARTIAL`: mean=0.5, p25=0.125, p50=0.5333, p75=0.8125, min=0.125, max=0.9
- `SEMI_IMPLICIT`: mean=0.4313, p25=0.0833, p50=0.625, p75=0.6429, min=0.0625, max=0.7222
- `UNCLASSIFIED`: mean=0.125, p25=0.125, p50=0.125, p75=0.125, min=0.125, max=0.125

## Graph features

| feature | mean | p25 | p50 | p75 | max |
|---|---:|---:|---:|---:|---:|
| `n_nodes` | 4.195 | 2.0 | 3.0 | 5.0 | 18.0 |
| `n_edges` | 3.175 | 1.0 | 2.0 | 4.0 | 17.0 |
| `depth` | 2.17 | 1.0 | 2.0 | 3.0 | 8.0 |
| `critical_path` | 3.17 | 2.0 | 3.0 | 4.0 | 9.0 |
| `n_roots` | 1.69 | 1.0 | 1.0 | 2.0 | 9.0 |
| `n_leaves` | 1.02 | 1.0 | 1.0 | 1.0 | 2.0 |
| `n_joins` | 0.67 | 0.0 | 0.0 | 1.0 | 8.0 |
| `max_indegree` | 1.43 | 1.0 | 1.0 | 2.0 | 2.0 |
| `n_fan_out_nodes` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `max_outdegree` | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| `n_reused_outputs` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `n_late_references` | 0.675 | 0.0 | 0.0 | 1.0 | 8.0 |
| `mean_reference_distance` | 1.2378 | 1.0 | 1.0 | 1.5 | 2.5 |
| `max_reference_distance` | 2.03 | 1.0 | 1.0 | 2.0 | 11.0 |
| `n_parallel_branches` | 1.69 | 1.0 | 1.0 | 2.0 | 9.0 |
| `n_type_transitions` | 0.085 | 0.0 | 0.0 | 0.0 | 1.0 |

## Surface features

- `parameter_count`: {"mean": 8.01, "p25": 4.0, "p50": 6.0, "p75": 10.0, "min": 2.0, "max": 36.0}
- `required_parameter_count`: {"mean": 8.01, "p25": 4.0, "p50": 6.0, "p75": 10.0, "min": 2.0, "max": 36.0}
- `optional_parameter_count`: {"mean": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "min": 0.0, "max": 0.0}
- `nested_schema_depth`: {"mean": 3.18, "p25": 3.0, "p50": 3.0, "p75": 3.0, "min": 3.0, "max": 5.0}
- `repeated_tool_count`: {"mean": 1.575, "p25": 0.0, "p50": 1.0, "p75": 2.0, "min": 0.0, "max": 14.0}
- `same_family_tool_count`: {"mean": 1.575, "p25": 0.0, "p50": 1.0, "p75": 2.0, "min": 0.0, "max": 14.0}
- `output_key_family`: {"output_0": 0.24, "result": 0.76}
- `schema_complexity`: {"mean": 1.9173, "p25": 2.0, "p50": 2.0, "p75": 2.0, "min": 1.0, "max": 4.0}

## Query realism

- `query_mode_distribution`: {'GOAL_BASED_IMPLICIT': 0.7, 'PROCEDURAL_EXPLICIT': 0.035, 'PROCEDURAL_PARTIAL': 0.125, 'SEMI_IMPLICIT': 0.115, 'UNCLASSIFIED': 0.025}
- `operation_explicitness_distribution`: {'full': 0.045, 'high': 0.05, 'low': 0.085, 'medium': 0.16, 'none': 0.66}
- `sequence_leakage_distribution`: {'full': 0.1, 'high': 0.075, 'low': 0.155, 'medium': 0.01, 'none': 0.66}
- `procedural_cue_distribution`: {'0': 0.55, '1': 0.32, '2': 0.105, '3': 0.01, '4': 0.005, '5': 0.01}
- `intermediate_reference_explicitness`: {'share_with_explicit_reference': 0.45, 'mean_reference_density': 0.3753}
- `mean_operation_explicitness`: 0.1645
- `mean_sequence_leakage`: 0.152
- `mean_procedural_cue_count`: 0.645
- `plan_leak_rate`: 0.035

## Safety note

PROFILE_SAFE profiles are built from the dev split and generic factory
metadata only. Diagnostic-informed statistics live in a separate file
and are never used as a default generation target.

