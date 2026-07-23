# Canary report

**Generated:** 2026-07-23T09:00:35.412726+00:00
**Model:** deepseek/deepseek-v3.2
**Mock:** False
**Canary tasks:** 10

## Run stats

- Pass A: {'ok': 10, 'error': 0, 'skipped': 0}
- Pass B: {'ok': 10, 'error': 0, 'skipped': 0}

## Validation

- Pass A valid rate: 100.0% (10/10)
- Pass B valid rate: 100.0% (10/10)
- Mean output token estimate: 273
- Invalid A: 0, Invalid B: 0

## Gate: PASS

Requirements: >=95% valid JSON after repair, output under ~180 tokens avg.