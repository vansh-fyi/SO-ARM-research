---
status: testing
phase: 12-bridge-tick-latency-fix
source: [12-VERIFICATION.md]
started: 2026-09-25T07:44:11Z
updated: 2026-09-25T07:44:11Z
---

## Current Test

number: 1
name: Live-hardware episode confirms tick-latency fix
expected: |
  Run a live episode on the real SO-ARM101 over the Colab bridge, the same way as
  Phase 11's 11-05 episode (`python control/run_vla_episode.py ... --server-address ...
  --checkpoint ...`), after Plan 12-01 (LATENCY-01/02/04) and Plan 12-02 (LATENCY-03)
  are both merged (they are — commits 58fea77/9a366a8 on master).

  (1) Count steps whose `model_version` suffix is neither a stale/holding/no-action
  flag (the "real executed action" count) and confirm it is measurably higher than
  Phase 11's 1/60 baseline;
  (2) confirm the recorded `latency_ms` values in `episode.jsonl` are no longer always
  `{0, 0}`;
  (3) confirm no two consecutive JSONL records show evidence of an overlapping
  in-flight `bridge-request-inflight-holding-position` flag under normal
  (non-overlapping) operation.
awaiting: user response

## Tests

### 1. Live-hardware episode confirms tick-latency fix
expected: Real (non-stale) action yield measurably higher than Phase 11's 1/60; `latency_ms` no longer always `{0,0}`; no evidence of overlapping in-flight requests under normal operation.
result: [pending]

## Summary

total: 1
passed: 0
issues: 0
pending: 1
skipped: 0
blocked: 0

## Gaps
