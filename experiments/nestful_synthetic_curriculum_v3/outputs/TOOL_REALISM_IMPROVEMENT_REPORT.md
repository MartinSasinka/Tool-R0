# Tool Realism Improvement Report

Date: 2026-07-02

## Before vs after (mixed prototype registry)

| metric | before (500-task math-only) | after (1030-task mixed) | delta |
|--------|------------------------------:|------------------------:|------:|
| tool name diversity | 9 | **25** | +16 |
| tool family overlap | 0.1942 | **0.3230** | +0.129 |
| bigram overlap | 0.0111 | **0.0091** | −0.002 |
| trigram overlap | 0.0342 | **0.0417** | +0.008 |
| scalar output share | 100% | **83.2%** | −16.8 pp |
| non-scalar output share | 0% | **16.8%** | +16.8 pp |

## New tool families added

- **String:** concat, lowercase, uppercase, extract_prefix, extract_suffix
- **List:** get_item, length, filter_greater_than, join_list, sort_list
- **Object:** get_field, merge_objects, make_object
- **Boolean:** greater_than, equals, contains

## Status classification

| level | meaning | current |
|-------|---------|---------|
| math_only | math tools only | **NO** (was YES) |
| mixed_synthetic_prototype | multi-family, low NESTFUL overlap | superseded |
| partial_tool_realism | improved diversity, pilot caveats | **YES** |
| final_ready | high NESTFUL overlap | **NO** |

## Interpretation

Tool/output realism **improved materially** (25 tool names, 16.8% non-scalar outputs, higher family overlap).

Bigram overlap remains near zero because NESTFUL uses IBM-specific tool names (4202 unique) — expected for prototype registry.

**Do not claim NESTFUL transfer readiness.** Safe for **stage1–2 prototype pilot** only.
