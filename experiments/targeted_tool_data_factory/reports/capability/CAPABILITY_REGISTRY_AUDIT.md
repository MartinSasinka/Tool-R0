# CAPABILITY_REGISTRY_AUDIT

- primitives: **89**
- declared capability families: **25**
- populated families: **25**
- validation errors: **0**

## Family coverage

| family | primitives | members |
|---|---:|---|
| `arithmetic.binary` | 12 | abs_difference, add, decrease_by_percent, divide, floor_divide, increase_by_percent, modulo, multiply, percent_of, power, ratio_of, subtract |
| `arithmetic.unary` | 6 | digit_sum, inverse, negate, ratio_to_percent, sqrt, square |
| `arithmetic.reduction` | 2 | product_three, sum_three |
| `comparison` | 3 | is_greater, max_two, min_two |
| `boolean.logic` | 4 | is_divisible_by, logical_and, logical_not, logical_or |
| `rounding` | 5 | ceil_value, floor_value, round_direction, round_places, round_to_int |
| `statistics` | 6 | average_two, mean_three, mean_values, median_values, range_spread, range_three |
| `sequence.map` | 4 | cumulative_sums, offset_list, scale_list, sort_values_desc |
| `sequence.filter` | 2 | filter_above, top_k_values |
| `sequence.reduce` | 4 | count_values, max_values, min_values, sum_values |
| `sequence.index` | 2 | index_of_max, value_at_position |
| `sequence.combine` | 3 | append_value, concat_lists, join_values |
| `dictionary.lookup` | 1 | lookup_unit_factor |
| `dictionary.update` | 1 | apply_rate_override |
| `string.parse` | 2 | file_extension, parse_number |
| `string.transform` | 2 | text_length, text_upper |
| `string.format` | 3 | concat_texts, format_with_unit, tag_value |
| `conversion.numeric` | 6 | celsius_to_fahrenheit, fahrenheit_to_celsius, km_to_meters, meters_to_km, minutes_to_seconds, seconds_to_minutes |
| `conversion.text` | 2 | format_fixed, number_to_string |
| `geometry` | 4 | circle_area, hypotenuse, rectangle_area, rectangle_perimeter |
| `date_time` | 4 | days_to_hours, hours_to_minutes, minutes_since_midnight, weeks_to_days |
| `path_url` | 2 | domain_of_url, join_path_segments |
| `bitwise` | 4 | bitwise_and, bitwise_or, bitwise_xor, left_shift |
| `validation` | 3 | clamp, is_non_negative, is_within_range |
| `classification.deterministic` | 2 | classify_threshold, grade_band |

## Method notes

- `semantic_neighbors` = same capability family, schema-compatible, and
  proven non-equivalent by differential testing on sampled inputs.
- `confusable_non_equivalents` = schema-compatible and non-equivalent,
  regardless of family. Name similarity is never used.

