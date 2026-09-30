---
phase: 12-bridge-tick-latency-fix
verified: 2026-09-25T07:44:11Z
status: passed
score: 6/6 must-haves verified (code-level); 1 additional item requires live hardware (out of scope for this environment)
behavior_unverified: 0
overrides_applied: 0
human_verification:
  - test: >
      Run a live episode on the real SO-ARM101 over the Colab bridge, the same way
      as Phase 11's 11-05 episode
      (`python control/run_vla_episode.py ... --server-address ... --checkpoint ...`),
      after Plan 12-01 (LATENCY-01/02/04) and Plan 12-02 (LATENCY-03) are both merged.
    expected: >
      (1) Count steps whose model_version suffix is neither a stale/holding/no-action
      flag (the "real executed action" count) and confirm it is measurably higher than
      Phase 11's 1/60 baseline; (2) confirm the recorded `latency_ms` values in
      `episode.jsonl` are no longer always `{0, 0}`; (3) confirm no two consecutive
      JSONL records show evidence of an overlapping in-flight
      `bridge-request-inflight-holding-position` flag under normal (non-overlapping)
      operation.
    why_human: >
      Requires the physical SO-ARM101 robot and a live Colab bridge connection, which
      this environment (and Claude generally, per CONTEXT.md D-06) cannot run. This is
      Roadmap success criterion #4 ("A live-hardware episode shows a measurably higher
      real (non-stale) action yield...") and the live-confirmation clause of success
      criterion #3 ("...confirmed on a live re-verification episode"). Plan 12-01's
      Task 3 `<verify><human-check>` documents this exact handoff; it is explicitly
      owned by the user, not a gap in this phase's completed work.
---

# Phase 12: Bridge Tick-Latency Fix Verification Report

**Phase Goal:** The VLA bridge only requests fresh Colab inference when its local action queue is empty/near-empty instead of resending a full camera observation on every control tick, real latency is now measured instead of always logged as zero, and duplicate in-flight requests are guarded against — closing the root cause that limited Phase 11's live episode to 1/60 real (non-stale) actions. Two small pre-existing tech-debt items (a stale gripper-direction docstring and untracked mesh asset directories) are also cleared out.

**Verified:** 2026-09-25T07:44:11Z
**Status:** human_needed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `BridgeActionSource.get_action()` only calls `control_loop_observation()` when `self.client._ready_to_send_observation()` returns True, not unconditionally every tick (LATENCY-01) | VERIFIED | `control/vla_bridge/robot_client.py:286-290` wraps the call in `if self.client._ready_to_send_observation():`. Tests `test_bridge_action_source_skips_observation_send_when_gate_is_false` and `test_bridge_action_source_sends_observation_when_gate_is_true` (`control/test_robot_client.py:469,490`) assert call-count 0 and 1 respectively. Both pass. |
| 2 | `pop_validated_action()` updates `client.latest_action` to the popped action's timestep after every successful pop, and leaves it untouched on an empty queue (LATENCY-02) | VERIFIED | `control/vla_bridge/robot_client.py:229-230` (`with client.latest_action_lock: client.latest_action = timed_action.get_timestep()`), placed after the successful pop and before the empty-queue early-return path. Tests `test_pop_validated_action_updates_latest_action_on_successful_pop` (asserts `client.latest_action == 7`) and `test_pop_validated_action_leaves_latest_action_unchanged_on_empty_queue` (asserts value stays `3`) both pass. |
| 3 | `BridgeActionSource._inflight_lock` structurally prevents overlapping in-flight observation requests — a second concurrent `get_action()` call returns immediately with a distinct flag instead of running the gated body (LATENCY-04) | VERIFIED | `control/vla_bridge/robot_client.py:272,282-309` — `threading.Lock()` created in `__init__`, non-blocking `acquire(blocking=False)` guards the body, `finally: self._inflight_lock.release()` prevents permanent lock-up. `test_bridge_action_source_inflight_guard_rejects_concurrent_get_action_call` uses two real `threading.Thread`s with `Event`-based synchronization and asserts the second call returns `...@bridge-request-inflight-holding-position` without blocking; `test_bridge_action_source_inflight_guard_never_rejects_sequential_calls` confirms non-overlapping sequential calls are unaffected. Both pass. |
| 4 | `run_episode()`'s `io_logger.write_step()` call carries real `time.monotonic()`-derived `latency_ms` deltas (`observation_to_action`, `action_to_execution`), not a hardcoded zero dict (LATENCY-03, code portion) | VERIFIED | `control/run_vla_episode.py:198,200,213,230-233` — `t0`/`t1`/`t2` bracket `get_action()` and the `send_action()` try/except block; `latency_ms` is built from `(t1-t0)*1000` / `(t2-t1)*1000`. `test_latency_ms_reflects_real_monotonic_deltas_under_controlled_clock` monkeypatches `run_vla_episode.time.monotonic` to `[0.0, 0.1, 0.25]` and asserts the written record equals exactly `{100.0, 150.0}`. Passes. |
| 5 | The existing validation/execution gate separation (T-11-09) is unchanged: `pop_validated_action()` still never calls `send_action()` | VERIFIED | `control/vla_bridge/robot_client.py` — `pop_validated_action()` (lines 181-247) contains no `send_action` call; `send_action()` is only invoked by the caller (`control/run_vla_episode.py:210`), after validation. Module docstring (lines 1-35) still documents this separation as deliberate. |
| 6 | DEBT-02 (gripper `format_action` docstring) and DEBT-03 (`So-101/`/`coppelia/` mesh asset tracking) are re-confirmed already resolved, with no invented code change for either | VERIFIED | `grep -c "opens and -1 closes" LIBERO/libero/libero/envs/grippers/soarm_gripper.py` = 1 (docstring at lines 42-44 correctly states "External +1 opens and -1 closes", matching `np.array([1.0, 1.0])` at line 56); commit `1bc5f54` (2026-09-20, `fix(gripper): make increasing jaw position open the gripper`) confirmed via `git log`, predating this phase. `git status --porcelain So-101` returns 0 lines (fully tracked, 39 files). No `coppelia/` directory found anywhere in the repo. `REQUIREMENTS.md` lines 192-193 mark both `[X]` Not applicable with matching rationale. No files were modified for either item — consistent with the plan's explicit read-only re-confirmation task. |
| 7 (SC3 live clause + SC4) | A live-hardware episode confirms (a) `latency_ms` is real/non-zero in practice, (b) real (non-stale) action yield is measurably higher than Phase 11's 1/60 baseline, and (c) the in-flight guard doesn't trigger spuriously under normal operation | ⚠️ Requires live hardware — see Human Verification | Code-level mechanisms for all three are implemented and unit-tested (truths 1-4 above); the live confirmation itself requires the physical SO-ARM101 and a live Colab bridge session, explicitly out of scope for this environment per CONTEXT.md D-06. Plan 12-01's Task 3 `<verify><human-check>` documents this exact handoff to the user. |

**Score:** 6/6 code-verifiable truths VERIFIED; 1 truth requires live-hardware confirmation (expected human handoff, not a gap).

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `control/vla_bridge/robot_client.py` | `BridgeActionSource.get_action()`, `pop_validated_action()`, `BridgeActionSource.__init__` modified in place (LATENCY-01/02/04) | ✓ VERIFIED | All three modifications present, substantive, and wired (see truths 1-3 above). No new files. |
| `control/test_robot_client.py` | Extended with gate-skip, latest_action-update, and in-flight-guard test coverage | ✓ VERIFIED | 6 new tests present (`test_bridge_action_source_skips_observation_send_when_gate_is_false`, `_sends_observation_when_gate_is_true`, `test_pop_validated_action_updates_latest_action_on_successful_pop`, `_leaves_latest_action_unchanged_on_empty_queue`, `test_bridge_action_source_inflight_guard_rejects_concurrent_get_action_call`, `_never_rejects_sequential_calls`). Suite: 18/18 pass. |
| `control/run_vla_episode.py` | `run_episode()` modified in place with real `time.monotonic()` latency deltas | ✓ VERIFIED | `t0`/`t1`/`t2` bracket the observation-send and execution phases; `latency_ms` dict built from real deltas. |
| `control/test_run_vla_episode.py` | Extended with real-latency test coverage | ✓ VERIFIED | `test_latency_ms_reflects_real_monotonic_deltas_under_controlled_clock` present, asserts exact millisecond values. Suite: 18/18 pass. |
| `LIBERO/libero/libero/envs/grippers/soarm_gripper.py` (DEBT-02, read-only re-confirmation) | Docstring already correct, no edit needed | ✓ VERIFIED | Confirmed correct on disk, matches CONTEXT.md D-04 claim exactly. |
| `So-101/` (DEBT-03, read-only re-confirmation) | Fully git-tracked, no `coppelia/` mesh dir | ✓ VERIFIED | `git status --porcelain So-101` empty; no `coppelia/` directory exists. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `BridgeActionSource.get_action()` | `self.client._ready_to_send_observation()` | Conditional gate before `control_loop_observation()` | ✓ WIRED | `robot_client.py:286` |
| `pop_validated_action()` | `client.latest_action_lock` / `client.latest_action` | Direct assignment after successful pop | ✓ WIRED | `robot_client.py:229-230` |
| `BridgeActionSource._inflight_lock` | Observation-send + `pop_validated_action()` body | Non-blocking `acquire()` / `try/finally release()` wrapping | ✓ WIRED | `robot_client.py:272,282-309` |
| `run_episode()`'s `t0`/`t1`/`t2` | `io_logger.write_step(latency_ms=...)` | Real millisecond deltas fed into existing parameter | ✓ WIRED | `run_vla_episode.py:198-233` |

### Behavioral Spot-Checks / Test Execution

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| `test_robot_client.py` full suite | `cd control && .venv/bin/python -m pytest test_robot_client.py -q` | 18 passed | ✓ PASS |
| `test_run_vla_episode.py` full suite | `cd control && .venv/bin/python -m pytest test_run_vla_episode.py -q` | 18 passed | ✓ PASS |
| Full `control/` suite (cross-file regression) | `cd control && .venv/bin/python -m pytest -q` | 100 passed | ✓ PASS |
| DEBT-02 docstring re-confirmation | `grep -c "opens and -1 closes" LIBERO/libero/libero/envs/grippers/soarm_gripper.py` | `1` | ✓ PASS |
| DEBT-03 tracking re-confirmation | `git status --porcelain So-101 \| wc -l` | `0` | ✓ PASS |
| DEBT-02 commit provenance | `git log -1 1bc5f54` | `1bc5f54 2026-09-20 fix(gripper): make increasing jaw position open the gripper` | ✓ PASS (predates phase scoping, confirms already-resolved claim) |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|--------------|--------|----------|
| LATENCY-01 | 12-01 | Bridge only requests fresh Colab inference when queue empty/near-empty | ✓ SATISFIED | Truth 1 |
| LATENCY-02 | 12-01 | `pop_validated_action()` updates `client.latest_action` | ✓ SATISFIED | Truth 2 |
| LATENCY-03 | 12-02 | `latency_ms` records real measured latency instead of `{0,0}` | ✓ SATISFIED (code); live confirmation pending | Truth 4, Truth 7 |
| LATENCY-04 | 12-01 | In-flight-request guard prevents duplicate/overlapping requests | ✓ SATISFIED | Truth 3 |
| DEBT-02 | 12-02 | Gripper docstring corrected | ✓ SATISFIED (re-confirmed not applicable — already fixed pre-scoping) | Truth 6 |
| DEBT-03 | 12-02 | Mesh asset directories tracked | ✓ SATISFIED (re-confirmed not applicable — already tracked pre-scoping) | Truth 6 |

No orphaned requirements — all 6 IDs (`LATENCY-01..04`, `DEBT-02`, `DEBT-03`) declared in this phase's PLAN frontmatter (`requirements:` fields of 12-01-PLAN.md and 12-02-PLAN.md) match exactly the phase requirement IDs given for verification, and all 6 appear in `.planning/REQUIREMENTS.md`'s traceability table mapped to Phase 12 with `Complete`/`Not applicable` status.

### Anti-Patterns Found

None. Scanned `control/vla_bridge/robot_client.py`, `control/run_vla_episode.py`, `control/test_robot_client.py`, `control/test_run_vla_episode.py` for `TBD`/`FIXME`/`XXX`/`TODO`/`HACK`/`PLACEHOLDER` markers, empty-return stubs, and hardcoded-empty-data patterns — none found. (One incidental grep hit on `run_vla_episode.py`'s CLI help text `/dev/cu.usbmodemXXXX` is a placeholder device-path example in argparse help, not a debt marker.)

### DEBT-02/DEBT-03 Special Handling (per verification instructions)

Per the explicit context provided for this verification: DEBT-02 and DEBT-03 were confirmed resolved *before* this phase was scoped (DEBT-02 fixed in commit `1bc5f54`, 2026-09-20; DEBT-03's `So-101/` already fully tracked with no `coppelia/` mesh directory ever existing). This verification independently re-ran the same checks the plan's Task 2 used (`grep -c "opens and -1 closes"`, `git status --porcelain So-101`, `git log 1bc5f54`, and a repo-wide search for a `coppelia/` directory) and confirms the same result the plan and REQUIREMENTS.md record. This is treated as SATISFIED, not as incomplete/dropped work — consistent with Plan 12-02's Task 2 being a deliberate read-only re-confirmation task.

### Human Verification Required

**1. Live-hardware episode confirmation of real-action yield, real latency, and in-flight-guard behavior**

**Test:** Run a live episode on the real SO-ARM101 over the Colab bridge (`python control/run_vla_episode.py ... --server-address ... --checkpoint ...`), the same way as Phase 11's 11-05 episode.

**Expected:**
1. Count steps whose `model_version` suffix is neither a stale/holding/no-action flag (the "real executed action" count) — confirm it is measurably higher than Phase 11's 1/60 baseline.
2. Confirm `latency_ms` values in `episode.jsonl` are no longer always `{0, 0}`.
3. Confirm no two consecutive JSONL records show evidence of an overlapping in-flight `@bridge-request-inflight-holding-position` flag under normal (non-overlapping) operation.

**Why human:** Requires the physical SO-ARM101 robot and a live Colab bridge session — Claude does not run physical hardware (CONTEXT.md D-06). This is Roadmap success criterion #4 and the live-confirmation clause of success criterion #3. Plan 12-01's Task 3 documents this exact handoff as a `<human-check>` step, owned by the user.

### Gaps Summary

No gaps found. All code-level must-haves (LATENCY-01, LATENCY-02, LATENCY-04, LATENCY-03's code portion, and the DEBT-02/DEBT-03 re-confirmation) are implemented, substantively correct, wired, and covered by passing, deterministic, genuinely-behavioral tests (real threads for the concurrency guard, monkeypatched clock for exact latency deltas, exact-value assertions throughout — not smoke tests). The full `control/` test suite (100 tests) passes with 0 failures, confirming no cross-file regression. The only outstanding item is the live-hardware confirmation of the fix's real-world effect (success criteria #3's live clause and #4), which is structurally impossible to verify without the physical robot and was correctly scoped to the user rather than attempted or fabricated by the executor.

---

*Verified: 2026-09-25T07:44:11Z*
*Verifier: Claude (gsd-verifier)*
