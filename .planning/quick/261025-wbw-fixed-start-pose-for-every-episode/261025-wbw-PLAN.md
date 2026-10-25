---
phase: quick-261025-wbw
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - control/start_pose.py
  - control/test_start_pose.py
  - control/conftest.py
  - control/run_vla_episode.py
  - control/test_run_vla_episode.py
  - control/test_base_mllm.py
  - control/test_mllm_record.py
  - control/tune_execution_hz.py
  - control/COMMANDS.md
autonomous: true
# Quick task: no roadmap requirement (orchestrator: requirements: []). Do NOT mark any VLAB-*/PIPE-* ID complete.
requirements: []

must_haves:
  truths:
    - "D-02: `python start_pose.py` (run from control/) reads the arm's current calibrated joint positions and writes control/start_pose.json keyed by action_contract.JOINT_ORDER with a captured_at timestamp; it refuses to overwrite an existing file unless --force is given, and never moves the arm"
    - "D-03: run_vla_episode.py refuses to run, before building any backend or touching the robot, when control/start_pose.json is missing or invalid (missing/unknown/non-finite/out-of-limit joints), and the error names the capture command"
    - "D-01/D-03: every episode moves the arm to the single home pose after the robot is connected and before the backend's first get_action; if any joint misses the tolerance, the run stops with an error naming those joints, makes zero get_action calls, writes termination reason start_pose_not_reached, and the robot is still disconnected"
    - "D-05: once the arm has reached home, every exit path (max steps, task complete, Ctrl-C e-stop, exception) moves back to the HOME pose strictly before disconnect"
    - "D-04: every run's --out directory contains start_pose.json with the home target, the reached joints, per-joint error, the tolerance and whether the arm arrived"
    - "tune_execution_hz.py inherits the same move-home / return-home behaviour and sweeps around the home pose"
    - "The full control test suite is green in a fresh worktree, and no captured control/start_pose.json is committed"
  artifacts:
    - path: "control/start_pose.py"
      provides: "START_POSE_PATH, StartPoseError, StartPoseNotReachedError, validate_start_pose, load_start_pose, capture_start_pose, joints_out_of_tolerance, tolerance/timeout constants, capture CLI main()"
      contains: "def load_start_pose"
    - path: "control/test_start_pose.py"
      provides: "Loader/validator/capture/CLI tests with MockRobot"
      contains: "def test_"
    - path: "control/conftest.py"
      provides: "Autouse fixture pointing start_pose.START_POSE_PATH at a temp all-zero home pose"
      contains: "START_POSE_PATH"
    - path: "control/run_vla_episode.py"
      provides: "main() pre-flight load; run_episode() move-home, verify, record, abort, return-home"
      contains: "home_pose"
    - path: "control/COMMANDS.md"
      provides: "Capture command and always-home behaviour docs"
      contains: "python start_pose.py"
  key_links:
    - from: "control/run_vla_episode.py main()"
      to: "start_pose.load_start_pose"
      via: "called after load_joint_limits_deg and before BACKEND_REGISTRY[args.backend](...)"
      pattern: "load_start_pose\\(joint_limits_deg\\)"
    - from: "control/run_vla_episode.py run_episode()"
      to: "move_to_positions(robot, home_pose, ...)"
      via: "start move inside the try, before the step loop's first action_source.get_action"
      pattern: "move_to_positions\\(robot, home_pose"
    - from: "run_episode() finally"
      to: "move_to_positions(robot, home_pose, ...)"
      via: "return move guarded by an at-home flag, before depth close / termination write; disconnect stays in main()/_shutdown_backend"
      pattern: "home_pose"
    - from: "control/conftest.py"
      to: "start_pose.START_POSE_PATH"
      via: "autouse monkeypatch to a tmp_path_factory file"
      pattern: "START_POSE_PATH"
---

<objective>
Every real-robot episode starts at ONE fixed, captured-and-committed home pose and returns to it at the end (D-01, D-02, D-03, D-04, D-05).

Purpose: runs become comparable (same start configuration every time), and a bad or missing pose can never send the arm somewhere unsafe, because the pose is validated against the action-contract joint limits before anything moves.

Output: a new `control/start_pose.py` (validate/load/capture library plus the capture CLI) with tests; `run_vla_episode.py` pre-flight check, move-home, verify, record and return-home; `tune_execution_hz.py` aligned with it; the existing tests updated minimally; COMMANDS.md documented.
</objective>

<execution_context>
@/Users/hp/Desktop/Work/Repositories/SoARM-Research/.claude/gsd-core/workflows/execute-plan.md
@/Users/hp/Desktop/Work/Repositories/SoARM-Research/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@./.claude/CLAUDE.md
@.planning/codebase/CONVENTIONS.md
@.planning/codebase/TESTING.md
@control/run_vla_episode.py
@control/keyboard_joint_control.py
@control/vla_bridge/action_contract.py
@control/conftest.py
@control/tune_execution_hz.py
@control/COMMANDS.md

## User decisions (LOCKED 2026-10-25, numbered here for traceability)

- D-01: ONE fixed home pose, used at the start of EVERY episode. Not optional per run. No multiple named poses.
- D-02: The pose is CAPTURED + COMMITTED. A small capture command reads the arm's current joint positions and saves them into a committed JSON file, `control/start_pose.json`. Values are in the same calibrated units `keyboard_joint_control.read_positions()` returns, keyed by `action_contract.JOINT_ORDER`, plus a `captured_at` timestamp. The user places the arm by hand, or with keyboard_joint_control.py / joint_jog.py. Overwriting an existing file requires `--force`.
- D-03: `run_vla_episode.py`, after the robot connects and BEFORE the backend's first get_action, moves the arm smoothly to home (move_to_positions-style P-control, generous timeout) and verifies it arrived within a tolerance. If it did not, it refuses to start the episode with a loud error naming the joints that missed, and still disconnects safely. A missing or invalid pose file means it refuses to run, with a message giving the capture command.
- D-04: Record the home target and the reached joints in the run's `--out` directory.
- D-05: At the end of the episode (max steps, KeyboardInterrupt e-stop, exception), return to the HOME pose, no longer the pose read at startup. The return move must still happen strictly before disconnect, in every exit path.
- D-06: There is only one physical arm. No cross-arm calibration handling: do not record or check robot ids in the pose file.
- D-07 (OUT OF SCOPE, do not build or stub): splitting episodes into segments/checkpoints, resuming from a checkpoint, and async/queued GPU inference.

Do not cite these D-NN ids in code comments: run_vla_episode.py already uses "D-03" for the Phase 11 e-stop decision. Reference "quick 261025-wbw" instead.

## Planner choices (discretion, decided; record in SUMMARY)

- One new module, `control/start_pose.py`. It holds the library (validate/load/capture/tolerance check) and its `__main__` is the capture command `python start_pose.py [PORT] [ROBOT_ID] [--force]`. This matches the flat `control/` CLI convention (keyboard_joint_control.py, joint_jog.py).
- Insertion point: the move lives in `run_episode()`, so tune_execution_hz.py inherits it. `main()` also loads and validates the pose as a pre-flight check right after `load_joint_limits_deg()`, before any backend build, so a bad file fails before hardware or Colab is touched. The gRPC lane sends no observation until `BridgeActionSource.get_action()` runs (`connect_bridge` only starts the action receiver), and no backend build() does warm-up inference. Moving at the top of run_episode's try is therefore strictly before any inference request on every lane.
- Test seam: `run_episode(..., home_pose: dict[str, float] | None = None)` is appended as the LAST parameter, so existing positional calls are unaffected. The production default `None` loads and validates the committed file. main() and tune pass the pose they already loaded. Tests whose MockRobot starts at non-zero positions pass `home_pose=` explicitly. Production code gets no opt-out flag.
- Constants in start_pose.py: `START_POSE_TOLERANCE = 5.0` (per joint, degrees for arm joints, percent for gripper; absorbs the P-control residual under gravity sag), `START_POSE_MOVE_SECONDS = 15.0` (generous; used for both the start move and the return move; the old 5 s return cap can be too short when home is far from where the policy left the arm), `START_POSE_KP = 0.2`, `START_POSE_CONTROL_HZ = 30` (int; move_to_positions uses `control_freq // 2`).
- If the arm never reached home (start move missed, raised, or was Ctrl-C'd), NO return move is made. The interrupted or failed move WAS the move home, and re-commanding it would override the operator's stop or repeat the same failure. The robot is still disconnected by main()'s finally.
- The run record is `<out>/start_pose.json`. termination.json keeps its exact `{"reason", "steps_completed"}` shape, because several tests compare it with `==`. A new reason value is added: `"start_pose_not_reached"`.

## Test-compatibility facts the executor must know (verified while planning)

- `MockRobot.send_action` never changes `_positions`, so a real `move_to_positions` toward a pose the mock is not at loops for the full timeout (15 s of real sleeps) and then fails verification. Every existing run_episode caller therefore needs a home pose equal to its robot's starting positions. The conftest autouse fixture covers the default all-zero MockRobot. `control/test_base_mllm.py` and `control/test_mllm_record.py` set positions to 5.0 (and shoulder_pan 50.0) before calling their `_run` helpers.
- The two `sent_actions`-count tests in test_run_vla_episode.py already stub `run_vla_episode.move_to_positions`. Both the start move and the return move MUST call the `move_to_positions` name imported into run_vla_episode, with `robot, targets` positional and everything else as keywords (`kp=`, `control_freq=`, `max_seconds=`, optional `label=`). This keeps existing fakes (signatures `(robot, targets, kp, control_freq, max_seconds, **kwargs)` and `(robot, targets, **kwargs)`) working.
- 13 sites in test_run_vla_episode.py (including `_dispatch_main` and `_run_resolved_bridge_main`, which test_backend_act/pi0/diffusion_policy/groot_n17 import) patch `action_contract.load_joint_limits_deg` with an empty-dict lambda. With the new pre-flight, an empty limits dict makes the pose invalid. The line is textually identical at every site, so one replace_all edit fixes them all.
- `control/test_mllm_record.py` asserts `list(tmp_path.iterdir()) == []` in several tests. The conftest fixture must write its pose file under `tmp_path_factory.mktemp(...)`, never inside `tmp_path`.
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: start_pose.py library + capture CLI, its tests, and the conftest home-pose fixture (D-01, D-02, D-06)</name>
  <files>control/start_pose.py, control/test_start_pose.py, control/conftest.py</files>
  <behavior>
    - validate_start_pose(joints, limits) returns {joint: float} in JOINT_ORDER order for a valid pose; raises StartPoseError for: a missing joint (names it), an unknown joint (names it), a non-numeric or bool value, NaN/inf, a value outside limits (names joint, value and limits), or a joint with no entry in the limits dict. When several problems exist, ONE error lists all of them.
    - load_start_pose(limits, path) reads {"captured_at": str, "joints": {...}}. Missing file, unreadable file, invalid JSON, a missing/non-dict "joints", or a missing/empty "captured_at" each raise StartPoseError, and the message includes the path and the capture command `python start_pose.py`. A file containing NaN / Infinity literals is rejected as non-finite.
    - load_start_pose with no path argument reads start_pose.START_POSE_PATH at CALL time, so monkeypatching the constant works.
    - capture_start_pose(robot, limits, path, force=False) writes indent-2 JSON with captured_at (format "%Y-%m-%dT%H:%M:%SZ", UTC), units (= action_contract.ACTION_UNITS) and joints in JOINT_ORDER, rounded to 2 decimals. The written file round-trips through load_start_pose.
    - capture_start_pose refuses an existing file without force (FileExistsError whose message mentions --force; file bytes unchanged), overwrites with force=True, and refuses an out-of-limit reading (StartPoseError; no file written). Example: a gripper reading of -1.0%.
    - joints_out_of_tolerance(target, reached, tolerance) returns {} when every joint is within tolerance (inclusive); returns {joint: reached - target} for joints beyond it; returns {joint: None} for a target joint absent from reached.
    - CLI main() with argv [PORT, ROBOT_ID] and an existing pose file and no --force exits with SystemExit mentioning --force BEFORE constructing any robot. With --force it constructs and connects the (patched) robot, writes the file, and disconnects the robot even if capture raises.
  </behavior>
  <action>
RED first: create control/test_start_pose.py covering every behavior bullet above. Follow TESTING.md: a module docstring stating no hardware is touched, `# --- Name ---` banners, flat full-sentence test names, pytest.mark.parametrize for the invalid-file cases, monkeypatch plus hand-written fakes (no unittest.mock). Use conftest's MockRobot (set `_positions` directly) and `mock_calibration_file` with `action_contract.load_joint_limits_deg(mock_calibration_file)` for limits. All valid example values must lie inside those limits: shoulder_pan about plus/minus 70.3, shoulder_lift about 107.5, elbow_flex about 97.2, wrist_flex about 102.0, wrist_roll (-157.21, 162.79), gripper (0, 100). Write invalid files with raw text where needed (NaN/Infinity literals, broken JSON). For the CLI tests:
- Patch `start_pose.SO101Follower` with a MockRobot subclass that accepts a config and counts construction.
- Patch `action_contract.load_joint_limits_deg` with a lambda returning a limits dict computed BEFORE patching, to avoid recursion.
- Patch `sys.argv`.
Run the tests and confirm they fail.

GREEN: create control/start_pose.py.
- Module docstring: explains the always-home contract (quick 261025-wbw), the file format, and the CLI usage `python start_pose.py [PORT] [ROBOT_ID] [--force]`. It also notes that the capture never moves the arm, only reads it.
- Imports, isort order: stdlib (argparse, json, math, datetime/timezone, pathlib.Path), then third-party lerobot `SO101Follower` and `SOFollowerRobotConfig` (same import paths as keyboard_joint_control.py, at module top so tests can patch `start_pose.SO101Follower`), then local `from keyboard_joint_control import read_positions` and `from vla_bridge import action_contract`. Always call `action_contract.load_joint_limits_deg` and `action_contract.JOINT_ORDER` through the module so tests can patch them.
- `START_POSE_PATH = Path(__file__).resolve().parent / "start_pose.json"`, `CAPTURE_COMMAND = "python start_pose.py"`, plus the four constants from "Planner choices", each with a one-line comment giving its unit or reason.
- `class StartPoseError(ValueError)` (invalid or missing pose file or pose) and `class StartPoseNotReachedError(RuntimeError)`. The not-reached error is built from (target, reached, missed, tolerance). It stores `.missed` and composes a loud message: every missed joint with target, reached and error (or "not reported" when None), the tolerance, "not starting the episode", and a hint to check for obstacles or servo faults or to re-capture with `CAPTURE_COMMAND --force`.
- `validate_start_pose(joints, joint_limits_deg) -> dict[str, float]`: collects every problem from the behavior list, then raises one StartPoseError. Reject bool explicitly (bool is an int subclass). Check finiteness with math.isfinite.
- `load_start_pose(joint_limits_deg, path=None)`: resolve `path = START_POSE_PATH if path is None else Path(path)` INSIDE the function. Never bind the constant as the default argument value, or monkeypatching breaks. Every failure message ends with how to capture: place the arm in the home pose, then run `CAPTURE_COMMAND` from control/.
- `capture_start_pose(robot, joint_limits_deg, path=None, force=False) -> dict`: run the exists/force check FIRST, then read_positions(robot), then validate, round to 2 decimals, write `json.dumps(doc, indent=2) + "\n"`, and return the doc. Do not record a robot id (D-06).
- `joints_out_of_tolerance(target, reached, tolerance=START_POSE_TOLERANCE)`, as specified in behavior.
- `main()`: argparse with optional positional `port` and `robot_id` (same help style as run_vla_episode.py) and `--force`.
  - Order: if START_POSE_PATH exists and not --force, `parser.error(...)` before anything else.
  - If port/robot_id are omitted, fall back to device_map.json's follower entry via an in-function `from run_vla_episode import _load_device_map`. Add a comment: run_vla_episode imports this module, so a top-level import would be circular. If that fails, parser.error pointing at detect_devices.py.
  - Load limits via `action_contract.load_joint_limits_deg()`.
  - Construct `SO101Follower(SOFollowerRobotConfig(port=..., id=...))`, `connect(calibrate=False)`, then write_calibration under `robot.bus.torque_disabled()`, exactly like keyboard_joint_control.main().
  - In try/finally (disconnect in finally): capture, print each joint value, and print `Saved → {path}` (project logging convention).
  - Convert StartPoseError / FileExistsError to `SystemExit(str(e))`.
  - `if __name__ == "__main__": main()`.

Then edit control/conftest.py: add an autouse fixture `start_pose_file(tmp_path_factory, monkeypatch)`.
- It writes {"captured_at": "2026-01-01T00:00:00Z", "joints": {joint: 0.0 for joint in JOINT_ORDER}} to `tmp_path_factory.mktemp("start_pose") / "start_pose.json"`. Never use tmp_path (see the context facts).
- It patches `start_pose.START_POSE_PATH` to that path and returns the path, so tests can request it by name and overwrite it.
- Import `start_pose` inside the fixture, with a comment: it pulls lerobot via keyboard_joint_control, and the lazy import keeps conftest's import light.
- Update the conftest module docstring: this fixture also guarantees no test can ever read or write the real control/start_pose.json.

Do NOT create control/start_pose.json. The real one is captured by the user on the arm.
  </action>
  <verify>
    <automated>cd "$(git rev-parse --show-toplevel)/control" && /Users/hp/Desktop/Work/Repositories/SoARM-Research/control/.venv/bin/python -m pytest test_start_pose.py -q && test ! -e start_pose.json && test -z "$(git ls-files start_pose.json)"</automated>
  </verify>
  <done>test_start_pose.py passes and covers missing file, invalid JSON, missing/unknown/non-finite/non-numeric/out-of-limit/unlimited joints, captured_at, capture write and round-trip, overwrite refusal without --force, overwrite with --force, out-of-limit capture refusal, the CLI refusing before robot construction, and the CLI disconnecting. No control/start_pose.json exists or is tracked.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: run_vla_episode.py pre-flight, move-home, verify, record, abort and return-home, with minimal test updates (D-01, D-03, D-04, D-05)</name>
  <files>control/run_vla_episode.py, control/test_run_vla_episode.py, control/test_base_mllm.py, control/test_mllm_record.py</files>
  <behavior>
    - run_episode with the default home_pose=None loads the conftest-provided file. Teleporting fake move, home {shoulder_pan: 10.0, shoulder_lift: -20.0, elbow_flex: 15.0, wrist_flex: 2.0, wrist_roll: 0.0, gripper: 30.0} written into `start_pose_file`: the first recorded event is the move to that home, the first get_action receives exactly that joint state, and `<out>/start_pose.json` has target == home, reached == home, arrived true, empty missed_joints, tolerance 5.0.
    - Non-moving fake move (MockRobot stays at zeros), home_pose=HOME passed explicitly: run_episode raises StartPoseNotReachedError naming shoulder_pan, shoulder_lift, elbow_flex and gripper (and not wrist_flex). get_action is called 0 times, move is called exactly once (no return move), termination.json == {"reason": "start_pose_not_reached", "steps_completed": 0}, `<out>/start_pose.json` has arrived false and missed_joints in JOINT_ORDER order, and no step records are written.
    - Parametrized exit paths (max_steps, keyboard_interrupt, exception), with a source that drives MockRobot away from home on each call (e.g. shoulder_pan 40.0) and raises on call 2 for the interrupt/exception cases: the LAST move targets HOME (not the all-zero pose the robot was at before the start move), the reason is right, the exception propagates in the exception case, and the return move passes no label (a progress print can raise BrokenPipeError after Ctrl-C).
    - An invalid explicit home_pose (shoulder_pan 500.0) and a missing default file (START_POSE_PATH patched to a nonexistent path) each raise StartPoseError before any move, with mock_robot.sent_actions == [].
    - main(): a missing or invalid pose file exits with SystemExit before ANY backend build (fake builders record nothing), and the message contains `python start_pose.py`. A valid custom file is passed to run_episode as home_pose. A run_episode raising StartPoseNotReachedError makes main exit with SystemExit carrying the joint names, and the backend robot is still disconnected.
  </behavior>
  <action>
RED first, in control/test_run_vla_episode.py.

Add a banner section `# --- Fixed start (home) pose (quick 261025-wbw) ---` with the behavior tests above. Use small local fakes:
- a teleporting move recorder: `(robot, targets, **kwargs)`, appends ("move", dict(targets), kwargs) to a shared events list and updates `robot._positions`;
- a non-moving variant;
- a recording action source appending ("get_action", dict(joint_state)).
Request the conftest `start_pose_file` fixture by name to write custom home poses. Import `start_pose` for the error classes and constant.

Minimal updates to existing tests (no other changes):
- (a) Add a module constant `PERMISSIVE_JOINT_LIMITS_DEG = {joint: (-100.0, 100.0) for joint in action_contract.JOINT_ORDER}`. Then replace_all the identical empty-limits patch line (13 sites) so it patches with `lambda: dict(PERMISSIVE_JOINT_LIMITS_DEG)`.
- (b) In `_dispatch_main`, record `home_pose=kwargs.get("home_pose")` in calls. Add an optional `run_episode_impl=None` parameter that replaces the default recording fake when given (used by the not-reached main test).
- (c) In test_run_episode_ends_when_source_sets_episode_complete_reason, change the move-count assertion from 1 to 2 (start move + return move), and assert both targets equal the all-zero home.
- (d) In test_keyboard_interrupt_triggers_return_to_start_before_disconnect, also assert the final two call_order entries are the move then the disconnect.

In control/test_base_mllm.py and control/test_mllm_record.py, change only the `_run` helper so its run_episode call passes `home_pose={joint: mock_robot._positions[joint] for joint in JOINT_ORDER}`. Those tests pre-set non-zero MockRobot positions, and MockRobot never moves. Confirm the new tests fail.

GREEN, in control/run_vla_episode.py:
- Import `start_pose` (local imports group). Update the module docstring "Reuses" list and the ActionSource docstring's "return-to-start" wording to the home pose.
- Add a helper `_write_start_pose_record(out_dir, target, reached, missed, tolerance)`. It writes `<out>/start_pose.json` (indent 2) with target, reached, error (per joint: reached minus target, or None), tolerance, arrived (no misses) and missed_joints (JOINT_ORDER order).
- `run_episode(...)`: append `home_pose: dict[str, float] | None = None` as the LAST parameter.
  - After the existing rate ValueError check: if home_pose is None, `home_pose = start_pose.load_start_pose(joint_limits_deg)`; otherwise `home_pose = start_pose.validate_start_pose(home_pose, joint_limits_deg)`. Both happen before any motion.
  - Remove the startup read of positions that was used as the return target.
  - Keep `depth_recorder = None` before the try, and construct the DepthRecorder only after home is verified. Then no depth thread exists and no misleading "DEPTH RECORDING FAILED" message prints when the start is refused (the finally already handles None).
  - Add an `at_home = False` flag.
  - Inside the existing try, before the step loop:
    1. Call `move_to_positions(robot, home_pose, kp=start_pose.START_POSE_KP, control_freq=start_pose.START_POSE_CONTROL_HZ, max_seconds=start_pose.START_POSE_MOVE_SECONDS, label="Moving to start pose")`.
    2. `reached = read_positions(robot)`, then `missed = start_pose.joints_out_of_tolerance(home_pose, reached, start_pose.START_POSE_TOLERANCE)`.
    3. Write the record (always, pass or fail).
    4. If missed, raise StartPoseNotReachedError.
    5. Otherwise set `at_home = True`, print a one-line "At start pose (max error X) -- starting episode", build the depth recorder, and run the loop unchanged.
  - Add `except start_pose.StartPoseNotReachedError: reason = "start_pose_not_reached"; raise` BEFORE the generic `except Exception`.
  - In the KeyboardInterrupt handler, keep `reason` first and the BrokenPipeError guard. The print says "Interrupted -- returning to home pose..." when at_home, otherwise "Interrupted before reaching the start pose -- not moving further." Keep the word "Interrupted" first; a test keys on it.
  - In the finally: make the return move `move_to_positions(robot, home_pose, kp=start_pose.START_POSE_KP, control_freq=start_pose.START_POSE_CONTROL_HZ, max_seconds=start_pose.START_POSE_MOVE_SECONDS)` with NO label, run only `if at_home`. It stays inside the existing nested try/finally, so depth close and the termination write still run. Disconnect still happens in main()/_shutdown_backend, after run_episode returns or raises.
  - Update the run_episode docstring: always-home start, abort semantics, return-home on every exit after arrival, and the home_pose parameter.
- `main()`: immediately after the existing `joint_limits_deg = action_contract.load_joint_limits_deg()` line (keep its position), add `try: home_pose = start_pose.load_start_pose(joint_limits_deg)` / `except start_pose.StartPoseError as e: parser.error(str(e))`. This runs before `BACKEND_REGISTRY[...]` builds anything.
  - Pass `home_pose=home_pose` to the run_episode call.
  - Add an `except start_pose.StartPoseNotReachedError as e:` clause to the existing try (before its finally) that raises `SystemExit(f"REFUSED TO START EPISODE: {e}")`. The finally still releases cameras and shuts the backend down, which disconnects the robot.

Run the listed test files. If any other existing test fails only because its fake robot starts away from the all-zero home, fix it the same minimal way (pass that robot's starting positions as home_pose at the call or helper). Never weaken production behaviour. A test that suddenly takes about 15 s is the symptom. Never bypass safety_validator: the start and return moves are harness moves, and policy actions still go through validate_action unchanged.
  </action>
  <verify>
    <automated>cd "$(git rev-parse --show-toplevel)/control" && /Users/hp/Desktop/Work/Repositories/SoARM-Research/control/.venv/bin/python -m pytest test_run_vla_episode.py test_start_pose.py test_base_mllm.py test_mllm_record.py test_depth_pipeline.py test_backend_mllm_claude.py test_backend_mllm_gpt.py test_backend_mllm_hf.py test_backend_act.py test_backend_pi0.py test_backend_diffusion_policy.py test_backend_groot_n17.py -q && test ! -e start_pose.json</automated>
  </verify>
  <done>All listed files pass. The new tests prove: the move to home happens before the first get_action; failing to reach home aborts with zero get_action calls, names the missed joints and records start_pose_not_reached; the return goes to HOME on max-steps, Ctrl-C and exception; home target and reached joints land in `<out>/start_pose.json`; main refuses a missing or invalid file before any backend build and exits loudly but still disconnects on not-reached. Existing tests changed only as listed.</done>
</task>

<task type="auto">
  <name>Task 3: tune_execution_hz.py alignment, COMMANDS.md docs, full-suite gate (D-01, D-02, D-03, D-05)</name>
  <files>control/tune_execution_hz.py, control/COMMANDS.md</files>
  <action>
control/tune_execution_hz.py: it already inherits the move-home and return-home behaviour through run_episode. Align it so the sweep is centred on the pose the arm will actually be at:
- Move `joint_limits_deg = action_contract.load_joint_limits_deg()` BEFORE robot construction and connect. Then load `home_pose = start_pose.load_start_pose(joint_limits_deg)`, converting StartPoseError to `SystemExit(str(e))`, so a missing pose fails before the arm is touched.
- Build `SweepActionSource(home_pose, ...)` instead of from the positions read at connect. Drop the now-unused read and its import if nothing else uses them.
- Pass `home_pose=home_pose` to run_episode.
- Convert StartPoseNotReachedError to SystemExit with the same "REFUSED TO START EPISODE" wording as main(). The existing finally still disconnects.
- Update the module docstring: the arm moves to the committed home pose first, sweeps around it, and Ctrl-C returns it home.

control/COMMANDS.md: add a section "## Start (home) pose — captured once, used by every episode", placed before the "Episode recording" section.
- What it is: one fixed pose in `control/start_pose.json`, committed to git, with calibrated degrees (gripper 0-100%) keyed by joint plus captured_at.
- How to capture: put the arm in the home pose by hand or with keyboard_joint_control.py / joint_jog.py, then run `python start_pose.py` (optional `PORT ROBOT_ID`, defaulting to device_map.json's follower) and commit the file. Overwriting needs `--force`. Capture only reads the arm, never moves it, and refuses a pose outside the safety limits (e.g. a fully closed gripper reading slightly below 0%).
- Caveats, one sentence each: the capture needs the serial port, so exit the jog tool first. keyboard_joint_control's X returns to its own start pose before exiting. Disconnecting releases torque. So choose a self-supporting pose, or hold the arm while capturing.
- The always-home behaviour of run_vla_episode.py and tune_execution_hz.py:
  - It refuses to run if the file is missing or invalid, before touching the robot or backend.
  - It moves home before the first model request, and refuses to start if any joint is more than ±5 (deg / %) off, naming the joints. The run directory gets start_pose.json and termination reason start_pose_not_reached.
  - It records target and reached joints in `<out>/start_pose.json`.
  - It returns home before disconnect on max steps, task complete, Ctrl-C or error.
  - It makes no return move if home was never reached.
  - The tolerance and timeout are constants in start_pose.py.
Also update the keyboard section's one-line X description only if it would otherwise contradict the new section. Match the file's existing terse copy-paste style.

Then run the full control suite in this worktree. In the user's main checkout, control/test_colab_notebooks.py fails only because of their uncommitted notebook edits. That is not a regression, but in a fresh worktree it must pass too. Do not touch control/vla_bridge/notebooks/http_openvla_oft.ipynb, control/vla_bridge/policy_server.ipynb, reports/ or research_notes/.
  </action>
  <verify>
    <automated>cd "$(git rev-parse --show-toplevel)/control" && PY=/Users/hp/Desktop/Work/Repositories/SoARM-Research/control/.venv/bin/python && $PY -c "import tune_execution_hz" && grep -q "home_pose=home_pose" tune_execution_hz.py && grep -q "load_start_pose" tune_execution_hz.py && grep -q "python start_pose.py" COMMANDS.md && grep -q -- "--force" COMMANDS.md && grep -q "start_pose_not_reached" COMMANDS.md && $PY -m pytest -q && test ! -e start_pose.json && test -z "$(git ls-files start_pose.json)"</automated>
  </verify>
  <done>tune_execution_hz.py loads and validates the home pose before connecting, sweeps around it and passes it to run_episode. COMMANDS.md documents capture, --force, and the always-home and refusal behaviour. The full control suite is green, and no start_pose.json is created or tracked.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| committed control/start_pose.json → harness | A hand-editable file whose values become direct servo targets for a harness move that is NOT routed through safety_validator |
| harness → physical arm | The start and return moves command real servos; a wrong target or an unverified arrival puts the episode's starting state, or people nearby, at risk |
| policy/backend → arm | Model actions; unchanged path through safety_validator.validate_action |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-wbw-01 | Tampering | start_pose.load_start_pose / validate_start_pose | high | mitigate | Reject missing/unknown/non-numeric/bool/non-finite joints, joints without a limits entry, and values outside action_contract limits, at load time before any motion (main pre-flight plus run_episode re-validation); tested in test_start_pose.py |
| T-wbw-02 | Denial of Service (safety) | run_episode start move | high | mitigate | Verify every joint within START_POSE_TOLERANCE after the move; abort with zero get_action calls, termination start_pose_not_reached, and no blind retry; main still disconnects via its finally |
| T-wbw-03 | Elevation of Privilege | harness moves vs safety_validator | medium | mitigate | Harness moves target only the validated home pose via P-control (kp 0.2, LeRobot max_relative_target still applies); policy actions keep going through validate_action unchanged |
| T-wbw-04 | Repudiation | run comparability | low | mitigate | `<out>/start_pose.json` records target, reached, per-joint error, tolerance and arrival for every run, including refused ones |
| T-wbw-05 | Tampering | capture overwrite | low | mitigate | Capture refuses to overwrite an existing pose without --force, checked before connecting to the robot |
| T-wbw-06 | Denial of Service | e-stop return under a broken stdout pipe | medium | mitigate | The return move passes no progress label, so a BrokenPipeError print cannot abort it; the reason is assigned before any print (existing pattern) |
| T-wbw-SC | Tampering | package installs | low | accept | No new dependencies; only stdlib, the installed lerobot, and existing local modules |
</threat_model>

<verification>
- `cd "$(git rev-parse --show-toplevel)/control" && /Users/hp/Desktop/Work/Repositories/SoARM-Research/control/.venv/bin/python -m pytest -q` has 0 failures in the worktree. Use the main checkout's interpreter by absolute path, because the worktree has no control/.venv.
- `git ls-files control/start_pose.json` is empty, and the file does not exist in the worktree.
- `git diff --stat` touches only the files listed in files_modified (plus any extra test file fixed the documented minimal way, named in SUMMARY).
</verification>

<success_criteria>
- One committed-pose workflow: `python start_pose.py` captures and writes, and `--force` is required to overwrite (D-02).
- run_vla_episode.py and tune_execution_hz.py refuse to run without a valid pose, and move home before any inference and verify it (D-01, D-03).
- A failed arrival aborts loudly, naming the joints, makes no get_action call, and still disconnects (D-03).
- The home target and reached joints are recorded per run (D-04).
- Return to HOME before disconnect on every exit path after arrival (D-05).
- No robot-id handling (D-06). Nothing from D-07 is built or stubbed.
- Full control suite green; COMMANDS.md updated.
</success_criteria>

<output>
Create `.planning/quick/261025-wbw-fixed-start-pose-for-every-episode/261025-wbw-SUMMARY.md` when done. Record the planner choices above, any extra tests touched, and the final test count.
</output>
