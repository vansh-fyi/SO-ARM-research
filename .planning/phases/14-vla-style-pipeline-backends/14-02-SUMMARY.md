---
phase: 14-vla-style-pipeline-backends
plan: 02
subsystem: control/vla_bridge + control/run_vla_episode.py
status: complete
tags: [backends, registry, cli, backend-flag, grpc-bridge, pipe-01]
requires:
  - "14-01: BACKEND_REGISTRY, GRPC_BRIDGE_BACKENDS, register(), register_grpc_backend(), BackendHandles, load_backend_config(), base_grpc.build_grpc_action_source()"
provides:
  - "run_vla_episode.py --backend {registered names}; omitted -> 'smolvla' with bridge args, else 'scripted'"
  - "BACKEND_REGISTRY['scripted'] (vla_bridge.backends.scripted.build) and BACKEND_REGISTRY['smolvla'] (gRPC lane)"
  - "vla_bridge.backends.scripted.ScriptedActionSource (re-exported as run_vla_episode.ScriptedActionSource)"
  - "BackendHandles.shutdown: optional teardown hook that replaces robot.disconnect()"
  - "run_vla_episode._shutdown_backend(handles): generic teardown for every lane"
  - "backend_configs/smolvla.json default checkpoint"
affects: [14-03, 14-04, 14-05, 14-06, 14-07, 14-08, 14-09, 14-10, 14-11, 14-12, 14-13]
tech-stack:
  added: []
  patterns:
    - "single BACKEND_REGISTRY[args.backend](args, camera_names, joint_limits_deg) dispatch in main()"
    - "cameras opened after backend build, using handles.skip_camera_names"
    - "Protocol-signature conformance test over shared ActionSource bases"
key-files:
  created:
    - control/vla_bridge/backends/scripted.py
    - control/vla_bridge/backends/smolvla.py
    - control/vla_bridge/backend_configs/smolvla.json
  modified:
    - control/run_vla_episode.py
    - control/vla_bridge/backends/__init__.py
    - control/vla_bridge/backends/base_grpc.py
    - control/test_run_vla_episode.py
    - control/test_backend_registry.py
decisions:
  - "Backward-compat shim resolves an omitted --backend to 'smolvla' when --server-address OR --checkpoint is given, so a lone --checkpoint still errors as it did before"
  - "An explicit gRPC backend with no --server-address errors at parse time"
  - "BackendHandles gained an optional `shutdown` callable; base_grpc sets it to RobotClient.stop, keeping the old client.stop() teardown"
  - "run_vla_episode.py no longer imports connect_bridge/BridgeActionSource/SO101Follower/SOFollowerRobotConfig; tests patch vla_bridge.robot_client instead"
metrics:
  duration: "~20 min"
  completed: 2026-10-01
  tasks: 2
  files: 8
---

# Phase 14 Plan 02: --backend Flag and Registry Dispatch Summary

`run_vla_episode.py` now has a `--backend` flag. `main()` picks its backend with one call, `BACKEND_REGISTRY[args.backend](args, camera_names, joint_limits_deg)`. `scripted` and `smolvla` are real registry modules. Existing commands that don't pass `--backend` behave as before: they pick `smolvla` when bridge args are given and `scripted` otherwise.

## What Was Built

### Task 1: `scripted` and `smolvla` backend modules (commits ea3e921 RED, b52b649 GREEN)
- **`vla_bridge/backends/scripted.py`**
  - Now the only copy of `ScriptedActionSource`, moved unchanged.
  - Has `@register("scripted") build()`, which does exactly what the old non-bridge branch did: `SO101Follower(SOFollowerRobotConfig(..., max_relative_target=MAX_RELATIVE_TARGET_DEG))`, then `connect(calibrate=False)`, then `write_calibration` with torque disabled.
  - Returns `BackendHandles(ScriptedActionSource(), robot, None, frozenset())`.
- **`run_vla_episode.py`** re-exports `ScriptedActionSource`, so `run_vla_episode.ScriptedActionSource` and `test_depth_pipeline.py`'s import chain still resolve. A test checks that it is the same class object as the relocated one.
- **`vla_bridge/backends/smolvla.py`**: `@register_grpc_backend("smolvla") build()`.
  - Checkpoint: `args.checkpoint`, falling back to `backend_configs/smolvla.json`'s `"checkpoint"`. If neither exists it raises `SystemExit` before connecting.
  - Then calls `base_grpc.build_grpc_action_source("smolvla", ...)`.
- **`backend_configs/smolvla.json`**: `{"checkpoint": "victorvanhalst/smolvla_so101_cube"}`.

### Task 2: `--backend` and registry dispatch (commits f55d255 RED, 0be5cb6 GREEN)
- **The `--backend` flag** uses `choices=get_backend_names()`, so an unknown name fails in argparse before anything is built (T-14-03 mitigation). Current choices are `{scripted, smolvla}`.
- **Shim for an omitted `--backend`:** resolves to `"smolvla"` if `--server-address` or `--checkpoint` is given, otherwise `"scripted"`.
- **Bridge-argument checks:**
  - The `--server-address`/`--checkpoint` pairing checks now apply only when `args.backend in GRPC_BRIDGE_BACKENDS`.
  - A new third check fails a gRPC backend that has no `--server-address` at all.
- **`main()` construction:**
  - The old `if args.server_address: ... else: ...` construction block is replaced by one registry call.
  - **Unreachable backend:** if any backend's `build()` returns `None`, `main()` writes `termination.json` with `bridge_unreachable` and returns before `IOLogger` opens.
  - **Camera order:** `_open_cameras()` now runs after the backend is built, using `handles.skip_camera_names`.
- **Teardown:**
  - New `_shutdown_backend(handles)` in `main()`'s `finally`. It calls `handles.shutdown()` if set, otherwise `robot.disconnect()`, then `action_source.close()` if it exists, then `stereo_camera.release()` if it exists. Each step runs even if an earlier one raises.
  - This covers 14-01's handoff note: the HTTP lane has no `RobotClient`, so it needs `close()` and `release()`.
- **Conformance test** (`test_backend_registry.py`):
  - Checks `{"scripted","smolvla"} ⊆ get_backend_names()`, and that `smolvla` is in `GRPC_BRIDGE_BACKENDS` while `scripted` is not.
  - Uses `inspect.signature` to check that `ActionSource.get_action` and each shared base's `get_action` (`BridgeActionSource`, `HttpAdapterActionSource`, `ScriptedActionSource`) take exactly `(self, joint_state, instruction)`.
  - A comment says Plan 14-13 must add `GeminiActionSourceBase` to that list.

## Verification
- `cd control && <main .venv>/bin/python -m pytest -q`: **270 passed** (baseline 242, plus 28 new).
- `python run_vla_episode.py --help` lists `--backend {scripted,smolvla}`.
- No tracked files were deleted.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] The shim also treats a lone `--checkpoint` as SmolVLA**
- **Found during:** Task 2
- **Issue:** The plan's shim (`"smolvla" if args.server_address else "scripted"`) would resolve `--checkpoint X` without `--server-address` to `scripted`. `scripted` is not a gRPC backend, so the pairing error would never fire, and the script would silently run the scripted dry run on the real robot. That is a regression: today this combination errors, and the existing `test_checkpoint_without_server_address_errors_clearly` covers it.
- **Fix:** `"smolvla" if (args.server_address or args.checkpoint) else "scripted"`.
- **Commit:** 0be5cb6

**2. [Rule 2 - Missing validation] An explicit gRPC backend with no bridge args**
- **Found during:** Task 2
- **Issue:** `--backend smolvla` with neither flag passes both pairing checks and would call `connect_bridge(None, ...)`.
- **Fix:** Added a third check, scoped to gRPC backends, that requires `--server-address`.
- **Commit:** 0be5cb6

**3. [Rule 3 - Blocking] Generic teardown needed a hook on `BackendHandles`**
- **Found during:** Task 2
- **Issue:** The old `finally` called `client.stop()`, which sets the shutdown event, disconnects the robot and closes the gRPC channel. After dispatch, `main()` has no `client` variable, and existing tests (with `BridgeActionSource` faked) assert that `stop()` still runs. The plan didn't say how teardown should work after the branch was removed.
- **Fix:**
  - Added the optional field `BackendHandles.shutdown: Callable | None = None` to 14-01's `__init__.py`. It is additive, and every existing construction still works.
  - `base_grpc` sets `shutdown=client.stop`.
  - The HTTP lane and `scripted` leave it unset, so they fall back to `robot.disconnect()`.
- **Checked against:** the installed `lerobot==0.6.1` `async_inference/robot_client.py:173` `RobotClient.stop()`.
- **Files outside the plan's list:** `vla_bridge/backends/__init__.py` and `base_grpc.py`.
- **Commit:** 0be5cb6

**4. [Rule 3 - Blocking] Existing `main()` bridge tests: patch target and fake signature**
- **Found during:** Task 2
- **Issue:** 13 existing tests patched `run_vla_episode.connect_bridge`/`BridgeActionSource`. After dispatch those names are dead in that module; the bridge is built in `base_grpc` through `vla_bridge.robot_client`. Their fake `connect_bridge` also lacked the `policy_type` kwarg that 14-01 added.
- **Fix:** Changed only how they patch, not what they assert:
  - The patch target is now `robot_client` (10 `connect_bridge` and 9 `BridgeActionSource` lines).
  - The 9 fake signatures gained `policy_type="smolvla"`.
  - Every assertion is unchanged, including `stopped is True`, the camera indices, `bridge_unreachable`, and the checkpoint and server values.
  - `run_vla_episode.py` dropped those four now-unused imports, so any stale patch fails loudly instead of silently patching nothing.
- **14-01 test:** its `base_grpc` fake client gained `stop` plus an assertion that `handles.shutdown is fake_client.stop`.
- **Commit:** 0be5cb6

### Notes, not deviations
- **Guard tests passed during RED:** In Task 2's RED run, the partial-bridge-args and unknown-`--backend` error tests already passed, because today's code errors on those inputs anyway. They are regression guards, not drivers. The 10 dispatch tests that do drive the change failed as expected.
- **The `smolvla.json` default is unreachable from the CLI:** Because the plan keeps the `--server-address` ⇔ `--checkpoint` pairing check for gRPC backends, `build()`'s fallback to the config checkpoint never runs from the command line (`--server-address` alone errors). It only runs when `build()` is called directly. If later plans want a config default for a gRPC backend, they would need to relax the check to something like "`--checkpoint` required unless `backend_configs/<name>.json` supplies one". This was left as the plan specified.

## Live-Source Verification Record (execution discipline)
- **`victorvanhalst/smolvla_so101_cube`:** Checked on HF Hub on 2026-10-01 via `huggingface.co/api/models/...` and `/raw/main/config.json`. The model is public and not gated, `library_name: lerobot`, and `config.json` has `"type": "smolvla"`. Its inputs are `observation.state` [6] and `observation.images.camera1/camera2/camera3`, which match `connect_bridge()`'s wrist plus stereo-split camera wiring.
- **`RobotClient.stop()`** in the installed lerobot 0.6.1 sets `shutdown_event`, calls `robot.disconnect()` and closes the channel. That makes it a full replacement for `robot.disconnect()` in the gRPC lane.
- **`"smolvla"` server support:** it is in `SUPPORTED_POLICIES`, already checked in 14-01.
- No new dependencies were installed. Nothing was left unverified.

## Known Stubs
None.

## Threat Flags
None. T-14-03 is mitigated (`choices=get_backend_names()`, covered by a test). T-14-04 is accepted as planned. The shim matches the old behavior, and the gRPC-scoped pairing checks still apply.

## Integration Notes for Later Plans
- **New backends:** add `vla_bridge/backends/<name>.py` with `@register(...)` or `@register_grpc_backend(...)`. Neither `main()` nor the `--backend` choices need editing.
- **gRPC-lane backends** should copy `smolvla.py` and change only the `policy_type` and config name. `base_grpc` already sets `shutdown`.
- **Several `main()` checks still key on `args.server_address`, not on the lane:** the device-map camera default, the `--stereo-camera-index` name resolution, and `--depth-endpoint` requiring `--server-address`. That is unchanged behavior. A Tier C or Gemini backend that wants the same name-first stereo resolution would need those checks keyed on the backend lane.

## Human Checks Deferred
The plan has no `<human-check>` blocks. Two hardware items follow from this change:
1. **SmolVLA regression on hardware**
   - **test:** Run `run_vla_episode.py --server-address HOST:PORT --checkpoint victorvanhalst/smolvla_so101_cube --out <new dir>` (no `--backend`) against a live Colab bridge.
   - **expected:** Same as before Phase 14: the robot moves, `episode.jsonl` has wrist and overhead_left/right frames, and Ctrl-C returns the arm to start and disconnects cleanly.
   - **why_human:** Needs the robot, cameras and the Colab bridge.
2. **Wrist camera open order changed**
   - **test:** In the same run, confirm the wrist USB camera gives frames to both the bridge (camera1) and the IOLogger recording.
   - **expected:** Both work. Previously `run_vla_episode.py` opened its wrist capture before `connect_bridge()` opened its own; now the bridge opens first.
   - **why_human:** Whether a second AVFoundation open works can only be checked on the physical device.

## Self-Check: PASSED
- FOUND: control/vla_bridge/backends/scripted.py, control/vla_bridge/backends/smolvla.py, control/vla_bridge/backend_configs/smolvla.json
- FOUND commits: ea3e921, b52b649, f55d255, 0be5cb6
