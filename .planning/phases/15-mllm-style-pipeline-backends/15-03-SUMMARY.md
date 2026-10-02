---
phase: 15-mllm-style-pipeline-backends
plan: 03
subsystem: control/vla_bridge (MLLM pipeline) + control/run_vla_episode.py
status: complete
tags: [mllm, mllm-07, pipe-02, failure-handling, task-complete, termination-hook, depth, tdd]
requires:
  - control/vla_bridge/backends/base_mllm.py (Plan 15-01 chunk-drain base)
  - control/fake_providers.py (ScriptedAdapter, FakeCollector, make_chunk_doc, make_provider_result; unmodified)
provides:
  - MllmActionSourceBase.episode_complete_reason / task_complete_claimed / max_consecutive_failed_calls
  - record fields consecutive_failed_calls and trace.task_complete_raw
  - hold reasons task-complete and episode-complete (constants REASON_TASK_COMPLETE, REASON_EPISODE_COMPLETE)
  - run_episode duck-typed episode_complete_reason termination hook
  - main() depth rules scoped by base_mllm.MLLM_BACKENDS (no --server-address, no main-side DepthCameraClient for MLLM backends)
affects:
  - Plan 15-05/15-06 (MLLM backend build() must own its depth client; main passes args.depth_endpoint/args.depth_calibration through)
  - Plan 15-04 (recorder sees consecutive_failed_calls and trace.task_complete_raw in the record)
  - Phase 16 (termination.json reasons task_complete / mllm_backend_failed; false-completion analysis via trace.task_complete)
tech-stack:
  added: []
  patterns:
    - duck-typed optional attribute read with getattr + isinstance(str) guard for loop termination
    - counter updated immediately before enqueue so the internal-error fallback can tell whether a refill was already counted
key-files:
  created:
    - control/test_mllm_failure.py
  modified:
    - control/vla_bridge/backends/base_mllm.py
    - control/run_vla_episode.py
    - control/test_run_vla_episode.py
decisions:
  - "task_complete true (identity check) holds ALL chunk_len items with reason task-complete; the model's steps are never returned nor logged as raw_model_output (they survive only in the provider raw payload / recorder sidecar)"
  - "A call whose answer contained an actions list resets the failure counter, even if a step was malformed or the chunk short; camera-read failures neither count nor reset"
  - "An internal-error refill (unexpected exception in _refill before the call outcome is counted) also counts as a failed call, so a persistent code bug ends the episode at the cap instead of holding until max_steps"
  - "trace.task_complete_raw is recorded for every parsed call (true included), plain-JSON and capped at MAX_LOGGED_TEXT_CHARS; it is null for failed calls"
  - "The episode-complete guard returns before touching collector or adapter whenever the queue is empty and episode_complete_reason is set"
metrics:
  duration: "~15 min"
  completed: 2026-10-02
  tasks: 2
  tests_added: 39
  full_suite: "1172 passed (1133 baseline + 39 new)"
---

# Phase 15 Plan 03: MLLM failure, completion and termination semantics Summary

The MLLM chunk-drain base now has a consecutive-failure cap (`mllm_backend_failed`), strict D-25 `task_complete` handling, and a guard that blocks spending after the episode ends. `run_vla_episode.py` received two additive edits. The first is a duck-typed `episode_complete_reason` termination hook. The second scopes the depth-flag rules so that MLLM backends own their single FastFS depth call and need no `--server-address`.

## What was built

### Task 1: `base_mllm.py` failure counter, task_complete and guard

- **New attributes:** `max_consecutive_failed_calls` (from settings), `_consecutive_failed_calls`, `episode_complete_reason` (None at start) and `task_complete_claimed`.
- **`_count_call_outcome(failed=...)`:** runs right before the refill enqueues.
  - Every whole-call failure adds one to the counter: a provider exception, a refusal/truncated/other stop, an empty response, invalid JSON or missing actions.
  - Reaching the cap sets `EPISODE_BACKEND_FAILED` and prints one console line naming the cap.
  - A parsed answer resets the counter.
  - A camera failure returns before the call, so it never reaches the counter.
- **D-25:** when `parsed.doc.get("task_complete") is True`:
  - all N items become `task-complete` holds;
  - `task_complete_claimed = True` and `episode_complete_reason = "task_complete"`;
  - a console line announces the claim.
  - A failed call never sets the claim.
- **Guard:** `get_action` returns an `episode-complete` hold before any collector or adapter call when the queue is empty and a terminal reason is set.
- **Record additions:** `consecutive_failed_calls` and `trace.task_complete_raw`.
- **Module docstring:** new "End of episode" section and the new hold reasons.

### Task 2: `run_vla_episode.py`, two additive edits

1. **Termination hook.** Right after `steps_completed = i + 1`, `done_reason = getattr(action_source, "episode_complete_reason", None)`. A non-empty `str` sets `reason` and breaks. The existing `finally` still runs return-to-start and `_write_termination`. The `ActionSource` Protocol docstring documents the attribute.
2. **Depth scoping.**
   - Added `from vla_bridge.backends import base_mllm`.
   - The `--depth-endpoint` without `--server-address` error now applies only when `args.backend not in base_mllm.MLLM_BACKENDS`.
   - The "given together" rule is unchanged for every backend.
   - `DepthCameraClient`, `check_ready()` and `depth_calibration.json` are built or written only for non-MLLM backends, so `depth_client` stays None for MLLM backends (FastFS 429).
   - Help strings and an `mllm-claude` usage example were updated.

`git diff 08be61d -- control/run_vla_episode.py` shows only the import, the hook plus its Protocol docstring, the two scoped conditions with comments, the help wording and the usage example. The safety validator, IOLogger and interpolation code are untouched.

## Verification

- `pytest test_mllm_failure.py test_base_mllm.py`: 90 passed. `test_mllm_failure.py` has 25 cases:
  - the 11-case failure matrix;
  - interrupt, cap, reset, camera-never-counts, guard and failed-record tests;
  - 2 task_complete-true tests;
  - 4 non-boolean values: `"true"`, `1`, `null`, `"yes"`;
  - 2 tests that a failed call never claims completion.
- `pytest test_run_vla_episode.py`: 74 passed (14 new, plus the strengthened `test_main_errors_when_depth_flags_given_without_server_address`, which now uses capsys).
- Full suite: 1172 passed in about 8 s.
- `python run_vla_episode.py --help` still runs, and the help text includes the new wording.
- Acceptance checks are covered by tests:
  - a refusal's `refusal_category` "reasoning_extraction" is preserved, with model_version `mllm-fake@refusal-holding-position`;
  - after task_complete, chunk_len holds of the live state and the reason is `task_complete`;
  - after 3 failures with chunk_len 2, the reason is `mllm_backend_failed`;
  - in the real loop, a task_complete episode ends after 1 step and a failure-cap episode ends after 5 steps.

## TDD Gate Compliance

- Task 1: RED `249bd12` (25 failing), then GREEN `4e00d04`.
- Task 2: RED `fc87f86` (4 behavior tests failing; the non-MLLM error and ignore-non-string tests pass by design, because they guard existing behavior), then GREEN `da51d4a`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing critical functionality] An internal-error refill counts toward the failure cap**
- **Found during:** Task 1
- **Issue:** The plan lists the whole-call failures that count but does not mention the Plan 15-01 `internal-error` fallback. A persistent code bug would have held position until `max_steps` with no terminal reason.
- **Fix:** When the get_action except path finds the queue empty, it calls `_count_call_outcome(failed=True)` before enqueueing internal-error holds. This cannot double-count, because `_refill` counts immediately before its own enqueue.
- **Files modified:** control/vla_bridge/backends/base_mllm.py
- **Commit:** 4e00d04

Other choices the plan left open (not deviations):
- `trace.task_complete_raw` is also recorded for the `true` case, and is size-capped.
- Held `task-complete` ticks carry no `raw_model_output` override, so the logged raw output equals the held live state.
- Two extra test cases: task_complete true combined with a malformed step, and the raw value `"yes"`.

Environment note: tests ran from the worktree's `control/` directory with the main checkout's venv interpreter. Nothing was installed. `fake_providers.py` and `mllm_prompt.py` were not touched (Plan 15-05 owns them this wave); all helpers are local to the test files.

## Known Stubs

None.

## Threat Flags

None. No new network endpoints or auth paths. The plan's threat register already covers the depth-caller change (T-15-09). T-15-06, T-15-07, T-15-08 and T-15-09 are mitigated and tested as specified.

## Self-Check: PASSED

- control/test_mllm_failure.py, control/vla_bridge/backends/base_mllm.py, control/run_vla_episode.py and control/test_run_vla_episode.py all exist.
- Commits 249bd12, 4e00d04, fc87f86 and da51d4a are present in `git log`.
