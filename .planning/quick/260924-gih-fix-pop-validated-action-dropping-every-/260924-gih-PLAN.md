---
phase: quick-260924-gih
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - control/vla_bridge/robot_client.py
  - control/test_robot_client.py
autonomous: true
requirements: [VLAHW-02]

must_haves:
  truths:
    - "A real bridge-returned action (raw_action keyed with lerobot's `.pos`-suffixed action_features convention, e.g. `shoulder_pan.pos`) no longer resolves to an empty validated_action dict -- pop_validated_action() normalizes those keys to the plain action_contract.JOINT_ORDER names safety_validator.validate_action() expects, BEFORE validation runs."
    - "Existing plain-keyed callers are unaffected -- key normalization (str.removesuffix('.pos')) is a no-op when a key already lacks the .pos suffix, so the empty-queue 'hold position' branch (which already returns plain current_state keys, bypassing _action_tensor_to_action_dict entirely) and every existing plain-keyed test still behave identically."
    - "pop_validated_action()'s 3rd tuple element (raw_action, used for episode.jsonl's raw_model_output field) is the same normalized, plain-keyed dict passed to safety_validate_fn -- not a mix of suffixed/unsuffixed keys -- for internal consistency with validated_action/current_state/prev_action, which are already plain-keyed everywhere else in this module."
    - "safety_validator.py's plain-name convention (action_contract.JOINT_ORDER) is unchanged -- the .pos-suffix is treated as lerobot's own wire-format concern, normalized locally at this one adapter boundary, not propagated outward."
    - "A regression test in control/test_robot_client.py exercises the exact real-world failure mode (a .pos-suffixed action dict flowing through pop_validated_action(), matching the real lerobot SO101Follower.action_features convention) and asserts a non-empty, plain-keyed validated_action with the expected values passed through unclamped."
  artifacts:
    - control/vla_bridge/robot_client.py
    - control/test_robot_client.py
  key_links:
    - "pop_validated_action()'s new key-normalization step -> safety_validator.validate_action()'s membership check (`for joint in action_contract.JOINT_ORDER: if joint not in raw_action: continue`) -- the fix makes this membership check actually match real bridge-returned keys instead of silently skipping every joint and returning {}."
    - "BridgeActionSource.get_action() -> pop_validated_action() -> run_vla_episode.py's run_episode() control loop -> robot.send_action({f'{j}.pos': v ...}) -- the now-non-empty, plain-keyed validated_action flows through run_vla_episode.py's existing send-time re-suffixing unchanged; only the safety_validator boundary itself is fixed, per this task's explicit scope."
---

<objective>
Fix `control/vla_bridge/robot_client.py`'s `pop_validated_action()` silently dropping every real VLA action to an empty `validated_action` dict, because `client._action_tensor_to_action_dict()` (lerobot's real `SO101Follower.action_features` convention) returns `.pos`-suffixed keys (`"shoulder_pan.pos"`, etc.) while `safety_validator.validate_action()` looks up plain joint names from `action_contract.JOINT_ORDER` (`"shoulder_pan"`, etc.). This mismatch made every joint fail `validate_action()`'s membership check (`if joint not in raw_action: continue`), returning `{}` with no error -- which crashed the last live episode at step 1 (`robot.send_action({f"{j}.pos": v for j, v in validated_action.items()})` sent an empty `goal_pos`, so lerobot's `ensure_safe_goal_position({}, max_relative_target)` raised `ValueError: max_relative_target keys must match those of goal_present_pos.`).

Purpose: unblock 11-05's live-hardware retry -- this is the last known bug standing between the current code and a real SmolVLA episode actually producing sustained robot motion, per this session's root-cause analysis (confirmed by reading installed `lerobot==0.6.1` source directly, not docs/comments).
Output: `pop_validated_action()` normalizes `.pos`-suffixed keys to plain joint names immediately after `_action_tensor_to_action_dict()`, before passing `raw_action` to `safety_validate_fn`; `control/test_robot_client.py` gains a regression test proving a `.pos`-suffixed bridge action now resolves to a correctly-keyed, non-empty `validated_action`; full `control/` test suite (93 tests) passes.
</objective>

<execution_context>
@/Users/hp/Desktop/Work/Repositories/SoARM-Research/.claude/gsd-core/workflows/execute-plan.md
@/Users/hp/Desktop/Work/Repositories/SoARM-Research/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md

Full current file for reference (edit `pop_validated_action()` in place):
@control/vla_bridge/robot_client.py

Consumer whose plain-name key expectations must be satisfied -- do NOT change this file's convention:
@control/vla_bridge/safety_validator.py

Plain joint-name convention source of truth (`JOINT_ORDER`):
@control/vla_bridge/action_contract.py

Existing tests to extend (reuse `FakeBridgeClient`/`FakeTimedAction`, both already defined near the top of the file):
@control/test_robot_client.py

**Root cause, already confirmed this session by reading installed `lerobot==0.6.1` source (do not re-derive):**

`client._action_tensor_to_action_dict(timed_action.get_action())` builds `raw_action` as `{key: action_tensor[i].item() for i, key in enumerate(self.robot.action_features)}`, and `self.robot.action_features` (`SO101Follower.action_features`, from `so_follower.py`'s `_motors_ft` property) returns `.pos`-suffixed keys -- e.g. `"shoulder_pan.pos"`, `"gripper.pos"`. `action_contract.JOINT_ORDER` and everywhere `safety_validator.validate_action()` looks things up are plain names, no suffix. `"shoulder_pan" != "shoulder_pan.pos"` as dict keys, so `validate_action()`'s `for joint in action_contract.JOINT_ORDER: if joint not in raw_action: continue` skips every joint, and `validated_action` comes back `{}` -- no exception, no flag, a false negative caused by a naming-convention mismatch (not an actually-missing joint).

The empty-queue "hold position" branch (`return dict(current_state), [...], {}, 0.0`) never called `_action_tensor_to_action_dict()` at all, so it was never affected by this bug and must remain untouched.

**Design decision already made -- do not revisit:** fix this at the adapter boundary inside `pop_validated_action()`, not by changing `safety_validator.py`'s plain-name convention to match `.pos`-suffixing. Plain names are the established convention throughout `action_contract.py`, `run_vla_episode.py`'s `read_positions()`, and the rest of the bridge code; `.pos`-suffixing is purely lerobot's own internal wire format for `send_action()`/`action_features`.

**Design decision already made for the 3rd tuple element -- do not revisit:** `pop_validated_action()`'s returned `raw_action` (used for `episode.jsonl`'s `raw_model_output` field, currently discarded by `BridgeActionSource.get_action()` via `_raw_action`, but exercised directly by 2 existing tests) should be the normalized, plain-keyed dict -- not the original `.pos`-suffixed one -- for internal consistency with `validated_action`/`current_state`/`prev_action`, which are already plain-keyed everywhere else this function and its callers touch. Verified against actual usage this session: `run_vla_episode.py`'s own `raw_model_output=raw_action` logging path uses the `ActionSource.get_action()` interface's own 2-tuple return (already plain-keyed for `BridgeActionSource`, since it returns `validated_action`), not `pop_validated_action()`'s 3rd element directly -- so there is no existing external consumer whose expectations this normalization could break.
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Normalize .pos-suffixed keys in pop_validated_action() before safety validation</name>
  <files>control/vla_bridge/robot_client.py</files>
  <behavior>
    - `raw_action` built from a `.pos`-suffixed source dict (e.g. `{"shoulder_pan.pos": 12.5, ...}`) must have its keys stripped to plain joint names (`{"shoulder_pan": 12.5, ...}`) before being passed to `safety_validate_fn` -- so every joint's membership check in `safety_validator.validate_action()`'s `for joint in action_contract.JOINT_ORDER` loop succeeds instead of silently skipping.
    - `raw_action` built from an already-plain-keyed source dict (every existing test's fixture, and the empty-queue "hold position" branch) must be unaffected -- stripping a suffix that isn't present is a no-op, producing byte-identical output to before this change.
    - The function's 3rd return value (`raw_action`) must be this same normalized, plain-keyed dict -- not the original `.pos`-suffixed one.
  </behavior>
  <action>
    In `pop_validated_action()` (`control/vla_bridge/robot_client.py`), immediately after the line `raw_action = client._action_tensor_to_action_dict(timed_action.get_action())`, add a line that rebuilds `raw_action` with each key's `.pos` suffix stripped via `str.removesuffix(".pos")` (a no-op for keys that don't have it, e.g. `dict.fromkeys(JOINT_ORDER, ...)`-shaped fixtures every existing test already uses). This normalized `raw_action` is what gets passed to `safety_validate_fn(raw_action, current_state, ...)` on the next lines, and is also what the function returns as its 4-tuple's 3rd element -- do not introduce a second, separate variable for the pre-normalization dict; there is no remaining consumer that needs the original `.pos`-suffixed form (see this plan's `<context>` for why).

    Update the function's docstring to add a short paragraph documenting this normalization step, in the same style as the existing `stale_threshold_s` paragraph: what lerobot's real convention is (`.pos`-suffixed, from `SO101Follower.action_features`), what this project's convention is (plain `action_contract.JOINT_ORDER` names), and the exact crash this fixes (`ValueError: max_relative_target keys must match those of goal_present_pos.`, from `validated_action` silently coming back `{}`).

    Do NOT modify `safety_validator.py`, `action_contract.py`, `BridgeActionSource`, `connect_bridge()`, or `_wire_stereo_split_cameras()` -- this is a single, localized fix inside `pop_validated_action()` only, at the exact adapter boundary where lerobot's wire format meets this project's plain-name convention.
  </action>
  <verify>
    <automated>cd control && grep -c 'removesuffix(".pos")' vla_bridge/robot_client.py && .venv/bin/python -m pytest test_robot_client.py -q</automated>
  </verify>
  <done>`pop_validated_action()` strips `.pos` suffixes from `raw_action`'s keys immediately after `_action_tensor_to_action_dict()`, before calling `safety_validate_fn`; the same normalized dict is returned as the 4-tuple's 3rd element. All 11 pre-existing tests in `control/test_robot_client.py` still pass unchanged (the fix is a no-op for their plain-keyed fixtures).</done>
</task>

<task type="auto">
  <name>Task 2: Add regression test for the .pos-suffixed key mismatch, verify full suite</name>
  <files>control/test_robot_client.py</files>
  <action>
    Add a new test function to `control/test_robot_client.py`, placed directly after `test_pop_validated_action_uses_stale_action_threshold_not_stale_observation` and before `test_pop_validated_action_empty_queue_holds_position` (grouping it with the other `pop_validated_action()` tests). Name it `test_pop_validated_action_normalizes_pos_suffixed_keys_from_real_bridge`.

    Reuse the existing `FakeBridgeClient` (its `_action_tensor_to_action_dict` stub already returns its input unchanged, so putting a `.pos`-suffixed dict directly into `FakeTimedAction`'s payload faithfully models the real lerobot `SO101Follower.action_features` convention -- no new fake class needed). Build `pos_suffixed_action = {f"{joint}.pos": 12.5 for joint in JOINT_ORDER}` and put it in the queue via `client.action_queue.put(FakeTimedAction(pos_suffixed_action, timestamp=time.time()))`. Load `joint_limits_deg` from the `mock_calibration_file` fixture (12.5 is well within every joint's fixture-derived limit and within `safety_validator`'s per-step/velocity caps, so it passes through unclamped). Use `current_state = dict.fromkeys(JOINT_ORDER, 0.0)`, `prev_action=None`, `dt_s=1.0`.

    Call `robot_client.pop_validated_action(client, safety_validator.validate_action, joint_limits_deg, current_state, prev_action=None, dt_s=1.0)` and assert: `validated_action != {}` (the exact pre-fix failure this test guards against); `validated_action == dict.fromkeys(JOINT_ORDER, 12.5)` (every joint present with its plain name, correct unclamped value); `set(raw_out.keys()) == set(JOINT_ORDER)` (the 3rd tuple element is also normalized, not left `.pos`-suffixed); `flags == []` (nothing was clamped or rejected).

    Give the test a docstring citing the exact crash this guards against: `ValueError: max_relative_target keys must match those of goal_present_pos.`, and that before the fix `validated_action` silently came back `{}` for every real bridge-returned action, in the same documentation style as `test_pop_validated_action_uses_stale_action_threshold_not_stale_observation`'s existing docstring just above it.

    Do not modify any other existing test in this file.
  </action>
  <verify>
    <automated>cd control && .venv/bin/python -m pytest test_robot_client.py -q && .venv/bin/python -m pytest -q</automated>
  </verify>
  <done>`control/test_robot_client.py` has 12 tests (11 pre-existing + 1 new), all passing. The new test puts a `.pos`-suffixed action dict through `pop_validated_action()` and asserts a non-empty, plain-keyed, correctly-valued `validated_action` -- reproducing and guarding against the exact bug that crashed the live episode. Full `control/` suite (93 tests: 92 pre-existing + 1 new) passes.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|--------------|
| Colab PolicyServer (network, untrusted) -> `pop_validated_action()`'s new key-normalization step -> `safety_validator.validate_action()` -> real SO-ARM101 servo bus | This fix sits directly on the boundary where an untrusted bridge-returned action first becomes something `validate_action()` can actually inspect. Before this fix, the mismatch caused a fail-safe-shaped symptom (empty action -> crash) rather than a fail-dangerous one; the risk this fix must not introduce is the opposite failure mode -- normalized keys reaching the servo bus without having actually passed through every clamp in `validate_action()`. |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-QT260924GIH-01 | Tampering | control/vla_bridge/robot_client.py (pop_validated_action) | high | mitigate | Key normalization happens strictly BEFORE `raw_action` is passed to `safety_validate_fn` (Task 1's action explicitly places the new line before the `safety_validate_fn(...)` call, never after) -- so every joint value still passes through `validate_action()`'s NaN/inf rejection, absolute joint-limit clamp, per-step displacement cap, and velocity cap unchanged. Verified by Task 2's regression test asserting `flags == []` only because 12.5 is genuinely within every cap, not because validation was skipped -- and by the full existing `test_safety_validator.py` suite (untouched by this plan) continuing to pass. |
| T-QT260924GIH-02 | Denial of Service (self-inflicted, via a wrong normalization regressing the hold-position branch) | control/vla_bridge/robot_client.py (pop_validated_action) | low | accept | The empty-queue "hold position" branch returns `dict(current_state)` directly, before the line this fix touches, so it is structurally unreachable by this change. Verified by Task 1's `<done>` criterion (all 11 pre-existing tests unchanged) and Task 2's full-suite run, which includes `test_pop_validated_action_empty_queue_holds_position`. |

</threat_model>

<verification>
Run, in order: (1) Task 1's grep + `test_robot_client.py` pass; (2) Task 2's `test_robot_client.py` (12 passed) then full `control/` suite (`cd control && .venv/bin/python -m pytest -q`) -- 93 passed, 0 failed. Confirm `git diff --stat` touches exactly `control/vla_bridge/robot_client.py` and `control/test_robot_client.py` -- no other file (especially not `control/vla_bridge/safety_validator.py`, `control/vla_bridge/action_contract.py`, or `control/run_vla_episode.py`).
</verification>

<success_criteria>
- `pop_validated_action()` normalizes `.pos`-suffixed `raw_action` keys to plain `action_contract.JOINT_ORDER` names before validation, and returns the same normalized dict as its 3rd tuple element.
- A real bridge-shaped `.pos`-suffixed action no longer resolves to an empty `validated_action` -- proven by a new regression test in `control/test_robot_client.py`.
- All 11 pre-existing `test_robot_client.py` tests, the new 12th test, and the full 93-test `control/` suite pass.
- No file other than `control/vla_bridge/robot_client.py` and `control/test_robot_client.py` is modified.
</success_criteria>

<output>
Create `.planning/quick/260924-gih-fix-pop-validated-action-dropping-every-/260924-gih-SUMMARY.md` when done.
</output>
