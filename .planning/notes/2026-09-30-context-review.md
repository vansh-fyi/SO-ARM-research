# Project context review — 2026-09-30

This is a context-restoration note, not an implementation plan or a claim of full
project verification. It reconciles planning history with the current source and
local episode artifacts. No robot, camera, or Colab runtime was operated.

## Latest user direction

- The user did **not** run the proposed RGB-only retry. Do not keep requesting it
  or count it as outstanding acceptance work.
- The next work must include depth and record it well in the episode. Depth is
  also intended for the subsequent model-input pipeline.
- First understand the project extensively; implementation has not been requested
  in this context-review turn.
- The September 29 handoff's question about RGB-only retry completion is superseded
  by this explicit September 30 clarification.

## Project evolution and evidence limits

| Stage | Delivered or established | Limits carried forward |
|---|---|---|
| Phases 1–3 | Restart-aware Colab environment; SOARM registered in LIBERO; OpenVLA-OFT and remote openpi backends behind a shared inference interface | Simulation actions are 7-D OSC pose/gripper actions, not hardware joint targets. The pi0 adapter still supplies a placeholder zero proprioceptive vector. |
| Phase 4 | Scripted and keyboard collection, robomimic HDF5, replay, SOARM normalization; 120 demonstrations of cream-cheese placement | Original bowl tasks exceeded gripper/reach limits and were replaced. Historical data predates the final accepted geometry. |
| Phases 5–6 | Genuine dual-camera VLA input; depth back-projection checked against simulation truth; spatial predicates; RLDS/OXE integration; LoRA training and checkpoint publication | Three spatial tasks are satisfied at reset and are predicate smoke tests, not useful manipulation benchmarks. Phase 6 UAT retains resume and dashboard caveats. |
| Phase 7 / v1.1 | Camera tuning and simulation depth persistence through HDF5/RLDS | Colab TFDS depth spot-check remains pending in UAT. Phases 8–9 are paused, not cancelled. |
| Physical bring-up | Leader/follower assembly, servo upgrade, calibration, teleoperation and recording | Older classical-stereo object measurement had lighting/specularity failures; it is a separate historical path from current FastFS. |
| Phase 10 | Accepted Coppelia-aligned digital twin and generated portable URDF | The September 20 established-model record supersedes older orientation, aperture and sign assumptions. Full policy/dataset revalidation on that geometry is not established. |
| Phase 11 | Real Colab SmolVLA → LeRobot → SO-101 bridge, validation and episode logging | The recorded 60-step baseline had only one fresh executed policy action; it proved connectivity, not manipulation success. |
| Phase 12 | Observation gating, latest-action update, lock, timing, camera sharing, interpolation, watchdog, stereo calibration and depth-recording wiring | Six plan summaries exist, but live UAT test 7 still fails the depth-plus-smooth-motion objective. |

Planning inventory: 44 phase plans and 44 corresponding summaries, 10 phase
contexts, 17 quick-task plans and summaries. This review surveyed their objectives,
outcomes and decisions, consulted UAT/verification/research records, and deeply
read the active control/depth code. It did not read every vendored library file or
revalidate every historical claim.

The current roadmap's detailed sections specify 12 → 13 → (14 and 15) → 16 → 17:
camera/provenance, VLA backends, MLLM backends, eight-backend Pen Transfer comparison,
then safety-cap tightening. The progress table uses older phase names. Depth's new
priority must be incorporated explicitly before treating that roadmap as an
executable next-step specification. Planned backends are not implemented backends.

## Runtime map

### Simulation and training

`LIBERO/libero/libero/envs/` owns robot registration and environments;
`assets/robots/soarm101/` and `assets/grippers/` own canonical geometry.
`scripts/mjcf_to_urdf.py` generates `So-101/So-101.urdf` from those assets.
`diagnostics/reference/soarm_coppelia.json` is measured reference evidence.

`vla/interface.py` defines `predict(images, language)`; `eval_loop.py` consumes
action chunks and checks task completion every simulation step. OFT packs the
overhead and wrist views in the checkpoint's expected order. Pi0 uses a separate
websocket server. `datasets/` handles collection, HDF5, replay, normalization,
RLDS conversion and OXE registration; the notebooks orchestrate Colab training.

Simulation depth is normalized MuJoCo depth in [0,1], persisted as float32.
`perception/depth_xyz.py` converts it to metric depth and back-projects using
camera geometry. The hardware FastFS output is already metric depth in metres;
the current simulation converter rejects values outside [0,1]. These are not
interchangeable dataset contracts.

### Real hardware

`control/run_vla_episode.py` owns the episode loop. `connect_bridge()` creates the
one hardware robot handle via installed LeRobot 0.6.1, wires the wrist and stereo
feeds, and starts the action receiver. It deliberately avoids LeRobot's built-in
action execution loop so candidate actions go through project validation.

`BridgeActionSource` gates observation sends using LeRobot's queue threshold,
pops an action, normalizes `.pos` keys, updates `latest_action`, checks staleness,
and applies a consecutive-stale/empty watchdog. The harness validates again and
interpolates toward the target. Defaults are 2 Hz control and 100 Hz waypoint
execution. These are configured rates, not proof of achieved wall-clock cadence.

Hardware actions are absolute targets: five arm angles in degrees and gripper
opening in 0–100 percent. They must not be confused with simulation radians,
prismatic jaw metres, or 7-D OSC actions.

The AR0144 uses one ffmpeg/AVFoundation stream selected preferably by device name.
The current working mode is 1600×600, split into 800×600 left/right images.
The old 2560×720 mode was frozen on this setup. Wrist inference and diagnostic
recording currently still use separate camera opens.

## Current depth path and confirmed blocker

1. The episode loop reads a stereo pair after sending the tick's waypoints.
2. The same pair is saved as diagnostic RGB and, on the configured cadence
   (default every 10 steps), passed to `DepthCameraClient.compute_depth()`.
3. The client rectifies using saved calibration, resizes to multiples of 32,
   scales intrinsics, and posts PNG/base64 images, intrinsics and baseline.
   With 800×600 inputs, the default target is 800×576.
4. Notebook Step 8 invokes `scripts/run_demo.py` in a new subprocess per request,
   writes temporary image/calibration files, and reads `depth_meter.npy` back.
5. The client decodes the response and `IOLogger` saves a `.npy` file and its
   reference in the tick's JSONL record.

The September 29 session reported 15–70-second FastFS requests and identified
repeated process/import/model-loading overhead. The current source confirms the
per-request subprocess design. It has not been re-profiled in this review.
The local HTTP timeout is 10 seconds; the notebook subprocess timeout is 120
seconds. A client timeout does not establish that server-side work was cancelled.

The intended architectural direction is a persistent service in the isolated
FastFS environment, loading its model once and reusing it. Installed LeRobot's
PolicyServer likewise retains its policy object after loading it at handshake.
Warm inference, network cost, queue behavior and shared-GPU contention still need
measurement; model residency alone does not prove motion will stay smooth while
the control loop waits synchronously.

Preserve the confirmed PolicyServer subprocess and FastFS virtualenv isolation
fixes. Do not replace them with shared-environment installs or Colab apt-based
stdlib-venv setup. The notebook's Flask handler still runs in a kernel thread;
the persistent GPU depth service should follow the handoff's separate-process
direction. The Markdown launch guide lacks several later notebook fixes, including
the isolated FastFS interpreter, and cannot be followed as equivalent instructions.

## Recording facts that matter for the next work

| Observation from current source/artifacts | Implication |
|---|---|
| Test-7 directories contain 26 depth failure references and no `.npy` depth files | Existing episodes do not demonstrate successful integrated depth capture. |
| Depth metadata currently records path, completion-time timestamp, shape and dtype | Capture time, source-frame identity, units, calibration identity, model/config identity and request timing need an explicit recording contract. |
| Saved RGB is the raw camera pair; depth is rectified/resized | Pixel alignment requires retaining the transformation/calibration or the processed RGB pair. A matching step number alone is insufficient. |
| Policy observations and episode diagnostic frames use separate reads of the shared stereo device | Sharing a camera solves device provenance, not exact inference-frame provenance. |
| The harness measures t0/t1/t2 before depth work | Existing `latency_ms` excludes the depth delay and is not end-to-end policy inference latency. |
| `BridgeActionSource` discards `_raw_action` and `_obs_age_s`, returning its validated action | The harness's `raw_model_output` field does not preserve the original policy action in this path. |
| `executed_action` is assigned the final validated target even if a waypoint send raised a caught error | It describes intended target, not verified physical execution or a complete waypoint trace. |
| JSONL opens in append mode, while frame names and step numbers restart at zero | Reusing an output directory can mix runs and overwrite frame files. `vla_episode_test7` has 180 records but only 60 unique step numbers. |
| JSON/base64/NumPy decoding happens outside the client's request-exception handler | Not all malformed endpoint responses take the documented safe-failure path. |

These are source-grounded findings for scoping, not changes implemented in this
turn. An async design, if chosen, must preserve exact source-frame association,
define outstanding-request/backpressure behavior and shutdown handling, and update
tests intentionally rather than simply moving the call to a thread.

The current saved calibration is for 800×600 images, dated September 27, with
stereo reprojection error about 1.793 px and translation approximately
[-0.05258, -0.00097, 0.01071] metres. The client currently uses abs(T.x) for its
baseline; the refactor should verify calibration/rectification conventions rather
than silently changing geometry. Prior UAT reported red-cube depth readings
0.754/0.758/0.759 m against a measured 0.74–0.75 m. That establishes a useful spot
check, not general accuracy or integrated episode success.

## Constraints and open work

- The current SmolVLA checkpoint uses wrist RGB, stereo-left RGB and stereo-right
  RGB plus state. Existing D-08 deliberately kept depth out of its observation.
  The user's next-stage depth requirement supersedes treating recording as the
  ultimate goal, but does not magically add a depth feature to this checkpoint.
  Specify how the next backend consumes depth; do not substitute depth into an
  RGB slot and claim meaningful depth conditioning.
- Keep the accepted digital twin: black housing, yellow jaws, [0,0.036] m per jaw,
  +1 opens / -1 closes, coupled jaws, approximately 88° initial wrist flex.
- Keep safety-cap tightening last as explicitly decided; preserve the existing
  validation/execution separation. User performs live hardware verification.
- Tests and summaries are not substitutes for observed depth-plus-motion success.
- Local master is 84 commits ahead of the locally recorded origin/main ref; no
  fetch or push was performed here. Check delivery before a future Colab pull.

## Verification performed in this review

Ran the existing focused tests under `control/.venv`:

```text
test_depth_camera.py test_io_logger.py test_run_vla_episode.py
test_robot_client.py test_stereo_camera.py test_stereo_calibration.py
test_action_contract.py test_safety_validator.py
117 passed in 12.98s
```

These tests mock hardware/network/model responses. They exercise calibration,
payload shape, cadence, frame sharing, bridge/validation behavior and persistence;
they do not test a resident GPU depth server or establish live nonblocking motion.
No implementation, dependency installation, hardware operation or publication
was performed. Older UAT gaps and document inconsistencies remain visible rather
than being silently marked complete.

Primary reading anchors: `PROJECT.md`, `REQUIREMENTS.md`, detailed `ROADMAP.md`
sections, phase contexts/summaries/UAT, `10-ESTABLISHED-MODEL.md`, September 29
`HANDOFF.json` and `.continue-here.md`, current `control/vla_bridge/` source and
notebook, `control/run_vla_episode.py`, local episode JSONL/termination files,
installed LeRobot async-inference source, and the project-specific LIBERO modules.
