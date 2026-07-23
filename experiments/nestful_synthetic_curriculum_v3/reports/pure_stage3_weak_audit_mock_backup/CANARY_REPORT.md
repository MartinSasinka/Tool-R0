# Canary report

**Generated:** 2026-07-23T08:00:09.717917+00:00
**Model:** deepseek/deepseek-v3.2
**Mock:** True
**Canary tasks:** 10

## Run stats

- Pass A: {'ok': 10, 'error': 0, 'skipped': 0}
- Pass B: {'ok': 10, 'error': 0, 'skipped': 0}

## Validation

- Pass A valid rate: 100.0% (10/10)
- Pass B valid rate: 100.0% (9/9)
- Mean output token estimate: 180
- Invalid A: 0, Invalid B: 0

## Gate: PASS

Requirements: >=95% valid JSON after repair, output under ~180 tokens avg.