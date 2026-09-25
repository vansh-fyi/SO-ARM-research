# Phase 12: Bridge Tick-Latency Fix - Context

**Gathered:** 2026-09-25
**Status:** Ready for planning

<domain>
## Phase Boundary

Fix the observation-resend-per-tick root cause in `control/vla_bridge/robot_client.py` so `BridgeActionSource.get_action()` only requests fresh Colab inference when the local action queue is empty/near-empty (via `lerobot`'s own `_ready_to_send_observation()`/`chunk_size_threshold` gate), instead of resending a full camera observation every control tick. Wire real measured latency into `io_logger.py`'s `latency_ms` field, add an in-flight-request guard, and sweep two small pre-existing tech-debt items (DEBT-02 docstring, DEBT-03 mesh tracking — see below, one is now moot).

This is a location-and-wire-up task, not a design task — root cause and fix mechanism are already fully diagnosed in `control/vla_bridge/FINDINGS.md` (§2) and `research/SUMMARY.md`. No dedicated research sub-phase is expected.

</domain>

<decisions>
## Implementation Decisions

### Staleness cap (LATENCY-01 / STALE_ACTION_S)
- **D-01:** Do NOT change `STALE_ACTION_S` (stays at 30s) as part of this phase. The real fix is the observation-gate (`_ready_to_send_observation()`): once wired in, the bridge should follow the intended chunk-drain loop — request one chunk of ~50 actions, drain them locally at ~0.5s/tick cadence (~25s to drain a full chunk), then request the next chunk once the queue is empty/near-empty. Under that cadence, `STALE_ACTION_S` becomes a safety fallback that should rarely if ever trigger, not a value to tune as evidence the fix worked. Success is measured by real-action yield on the live episode (success criterion #4), not by touching this constant.
- **D-02:** Do not add a network request timeout/retry mechanism in this phase — out of scope, bigger behavior change than "location-and-wire-up."

### DEBT-03 (mesh asset directories)
- **D-03:** DEBT-03 as stated in ROADMAP.md/REQUIREMENTS.md ("untracked `So-101/` and `coppelia/` mesh asset directories") is **moot — mark not applicable**, not implement-as-written:
  - `So-101/` is already fully tracked in git (39 files, `git status --porcelain So-101` is clean).
  - No `coppelia/` mesh-asset directory exists anywhere in the repo, and nothing in `So-101/So-101.urdf` references one — all mesh filenames resolve within `So-101/` itself, which is already tracked.
  - "Coppelia" in this repo refers to a one-time CoppeliaSim diagnostic reference (`.ttm` file that lived only on the user's local machine, used in Phase 10 for mesh-alignment verification — see `diagnostics/COPPELIA_MUJOCO_ALIGNMENT.md`). Its captured output `diagnostics/reference/soarm_coppelia.json` is already tracked; `diagnostics/outputs/coppelia_reference.cbor` is an untracked derived/regenerable cache, not a URDF dependency.
  - Decision: do not create a new `coppelia/` directory or track the `.cbor` cache file. Update REQUIREMENTS.md to mark DEBT-03 as not-applicable/resolved (with a one-line note of why) rather than having the planner invent work for it.

### DEBT-02 (gripper docstring) — also moot, discovered during planning
- **D-04 (revised, was "unchanged" at discuss-phase time):** DEBT-02 is **also moot — mark not applicable**. The `gsd-pattern-mapper` agent's fresh read of `LIBERO/libero/libero/envs/grippers/soarm_gripper.py:41-53` during plan-phase (2026-09-25) found the `format_action` docstring already states "External +1 opens and -1 closes" — matching the actual code (`np.array([1.0, 1.0])`), not mislabeled. `git log` confirms this was fixed on 2026-09-20 in commit `1bc5f54` ("fix(gripper): make increasing jaw position open the gripper"), predating this phase's discuss-phase session (2026-09-25) — the ROADMAP.md/CONTEXT.md description of DEBT-02 was already stale when this phase was scoped. Decision: do not add a docstring-edit task; update REQUIREMENTS.md to mark DEBT-02 as not-applicable/resolved with a one-line note, same treatment as DEBT-03. If the planner or executor re-reads the file and finds it genuinely still wrong, treat that as new information overriding this note — but as of this discovery, no edit is needed.

### In-flight-request guard (LATENCY-04)
- **D-05:** `get_action()` is called synchronously, once per tick, from a single-threaded control loop today — there is no live overlap/race possible in the current call pattern. The guard is deliberately scoped as **defensive insurance only**: add a simple in-flight boolean/lock around the observation-send + pop path so duplicate requests structurally cannot happen, without adding any timeout or retry logic. This guards against a *future* change (e.g. someone later adding a timeout so a hung Colab call doesn't freeze the loop) silently reintroducing an overlap race — it is not fixing an active bug in this phase.

### Live-hardware verification
- **D-06:** The user runs the live-hardware verification episode (success criterion #4) themselves on the real SO-ARM101, same as Phase 11. Claude's scope in this phase is implementing LATENCY-01..04 (both DEBT-02 and DEBT-03 are now closed out as not-applicable, per the sections above), writing/updating unit tests where feasible (e.g. mocking the queue/gate logic in `robot_client.py`), and handing off with clear instructions for the user to run and report back the live episode result. Claude does not run physical hardware.

### Claude's Discretion
- Exact mechanism for the in-flight guard (boolean flag vs. `threading.Lock`) — implementer's choice, per D-05's "simple flag/lock" framing.
- Exact latency measurement points feeding `io_logger.py`'s `latency_ms` dict (`observation_to_action`, `action_to_execution` — see `control/run_vla_episode.py:227` for the two existing sub-fields) — use `time.monotonic()` deltas bracketing the real call sites per LATENCY-03; no user preference expressed beyond "real measured latency, not always zero."

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Root-cause diagnosis (read first)
- `control/vla_bridge/FINDINGS.md` §2 ("Task 3: observed episode outcome") — full step-by-step trace of the 1/60 yield bug from the live Phase 11 episode, including the exact tick-time and staleness-discard numbers this phase is fixing.
- `research/SUMMARY.md` — background research referenced by ROADMAP.md's Phase 12 Context/Notes.

### Source files to modify
- `control/vla_bridge/robot_client.py` — `BridgeActionSource.get_action()` (lines 270-293), `pop_validated_action()` (lines 181-244), `connect_bridge()` (lines 52-139). This is where LATENCY-01, 02, and 04 land.
- `control/vla_bridge/io_logger.py` — `latency_ms` field (line 52/65). LATENCY-03 lands here.
- `control/run_vla_episode.py` — line 227, current hardcoded `latency_ms={"observation_to_action": 0, "action_to_execution": 0}` call site.
- `control/vla_bridge/safety_validator.py` — `STALE_OBSERVATION_S` (10.0s), `STALE_ACTION_S` (30.0s), lines 50/55. Read-only reference per D-01 — not to be changed this phase.
- `LIBERO/libero/libero/envs/grippers/soarm_gripper.py` — `format_action` docstring, lines 41-53. DEBT-02 lands here.

### Vendored library (read, do not modify)
- `control/.venv/lib/python3.12/site-packages/lerobot/async_inference/robot_client.py` — `_ready_to_send_observation()` (line 403), `control_loop_action()` (line 370, shows how `client.latest_action` is normally updated — LATENCY-02 needs to replicate this in `pop_validated_action()`), `control_loop()` (line 458, shows the reference gate pattern: `if self._ready_to_send_observation(): control_loop_observation(...)`).

### Requirements / roadmap
- `.planning/REQUIREMENTS.md` lines 147-150 (LATENCY-01..04), 192-193 (DEBT-02/03) — update DEBT-03's status per D-03 during/after this phase's work.
- `.planning/ROADMAP.md` "Phase 12: Bridge Tick-Latency Fix" section (lines 368-389) — success criteria and Context/Notes already lock the fix mechanism.

No external specs beyond the above — requirements fully captured in ROADMAP.md/REQUIREMENTS.md plus the decisions above.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `lerobot`'s own `_ready_to_send_observation()` (already installed, already configured via `chunk_size_threshold` in `connect_bridge()`) — just needs to be consulted by `BridgeActionSource.get_action()` before calling `control_loop_observation()`. No new gate logic needs to be written, only wired in.
- `lerobot`'s `control_loop()` (vendored source, line 458-479) is the reference implementation of the exact gate pattern this phase needs to replicate in `BridgeActionSource.get_action()`, minus the `send_action()` call this project deliberately keeps separate from validation (T-11-09).

### Established Patterns
- Validation gate and execution gate are kept strictly separate in `robot_client.py` (T-11-09) — `pop_validated_action()` never calls `send_action()`; the caller does, after validation. Any LATENCY fix must preserve this separation.
- `.pos`-suffix normalization already exists in `pop_validated_action()` (lines 229-230) — established pattern for translating lerobot's raw action-dict keys to this project's plain-name convention.

### Integration Points
- `client.latest_action` (int timestep counter, `lerobot`'s own field) currently only gets updated inside `lerobot`'s `control_loop_action()` — which this project never calls (see module docstring, lines 1-35, for why: it would bypass the safety validator). Since this project's own `pop_validated_action()` is the de facto replacement for `control_loop_action()`, LATENCY-02 requires it to also update `client.latest_action = timed_action.get_timestep()` after a successful pop, mirroring the vendored library's own line 384-385, so `_ready_to_send_observation()`'s queue-size gate and the library's staleness dedup logic both function as designed.

</code_context>

<specifics>
## Specific Ideas

No specific UI/behavior requirements beyond the mechanism above — this is backend/control-loop wiring, verified via live hardware by the user, not a design surface.

</specifics>

<deferred>
## Deferred Ideas

- Re-tuning `STALE_ACTION_S`/`STALE_OBSERVATION_S` closer to real measured round-trip latency, if the post-fix live episode shows staleness discards are still a real problem — revisit only if D-01's assumption (gate fix alone restores the intended chunk-drain cadence) turns out to be wrong. Not scheduled to any specific future phase; surface it if the live-verification episode shows otherwise.
- Adding a request timeout/retry mechanism for hung Colab calls (see D-02) — explicitly deferred, not scheduled.
- Tracking `diagnostics/outputs/coppelia_reference.cbor` and other untracked `diagnostics/` derived artifacts for reproducibility completeness — explicitly declined for this phase (see D-03), not scheduled to a future phase either; raise again only if reproducing the diagnostic capture becomes a real need.

### Reviewed Todos (not folded)
None — `todo.match-phase 12` returned 0 matches.

</deferred>

---

*Phase: 12-Bridge Tick-Latency Fix*
*Context gathered: 2026-09-25*
