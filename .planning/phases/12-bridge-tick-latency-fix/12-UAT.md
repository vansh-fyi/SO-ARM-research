---
status: diagnosed
phase: 12-bridge-tick-latency-fix
source: [12-VERIFICATION.md]
started: 2026-09-25T07:44:11Z
updated: 2026-09-26T13:00:00Z
---

## Current Test

[testing complete]

## Tests

### 1. Live-hardware episode confirms tick-latency fix
expected: Real (non-stale) action yield measurably higher than Phase 11's 1/60; `latency_ms` no longer always `{0,0}`; no evidence of overlapping in-flight requests under normal operation.
result: issue
reported: "Live episode against 4.tcp.ngrok.io (checkpoint victorvanhalst/smolvla_so101_cube, 300 steps, outputs/vla_episode_002) met the three literal pass criteria -- 68/300 (22.7%) real-action steps vs Phase 11's 1/60 baseline, 0/300 steps with latency_ms={0,0}, 0 steps with an overlapping in-flight flag -- but the full run surfaced three concrete defects the user wants fixed before this phase is considered done: (1) IOLogger's camera_overhead recording captured the wrong physical device (laptop FaceTime camera, not the AR0144), (2) robot motion is visibly jerky (discrete jumps, not smooth), (3) the bridge entered a 6+ minute stale/no-action deadlock mid-episode (steps ~192-241) that the client only escaped via lerobot's own must_go fallback, not this project's code."
severity: major

## Summary

total: 1
passed: 0
issues: 1
pending: 0
skipped: 0
blocked: 0

## Gaps

- truth: "camera_overhead/*.png in episode output directories are real frames from the AR0144 stereo camera, matching what the policy actually receives as camera2/camera3"
  status: failed
  reason: "User observed outputs/vla_episode_002/camera_overhead/000216.png is a laptop FaceTime-camera selfie, not the AR0144 view. The policy's actual input (via StereoSplitCamera/ffmpeg, opened by device NAME 'CCB Camera') is unaffected -- this is a diagnostic-recording-only bug. User also wants the recorded overhead capture to store BOTH stereo lenses (left+right) as two separate images, not a single frame."
  severity: major
  test: 1
  root_cause: "run_vla_episode.py's main() opens camera_overhead via a second, independent cv2.VideoCapture(device_map['cameras']['stereo_overhead']) numeric index (lines ~354-360), racing/colliding with connect_bridge()'s StereoSplitCamera, which already holds the SAME physical AR0144 device open via a separate ffmpeg/avfoundation subprocess (vla_bridge/robot_client.py:166, stereo_camera.py). stereo_camera.py's own docstring documents that the AR0144 does not tolerate concurrent opens -- the second (cv2) open silently landed on a different device (FaceTime) instead of erroring."
  artifacts:
    - path: "control/run_vla_episode.py"
      issue: "IOLogger's camera_overhead cap is a second independent cv2.VideoCapture on the AR0144's numeric index, opened concurrently with StereoSplitCamera's ffmpeg-based open of the same physical device"
    - path: "control/vla_bridge/robot_client.py"
      issue: "connect_bridge() already exposes the shared StereoSplitCamera instance as client._stereo_camera (read_left()/read_right()) -- run_vla_episode.py does not reuse it for its own diagnostic recording"
  missing:
    - "Remove the separate cv2.VideoCapture open for the overhead/stereo camera in run_vla_episode.py's --server-address (bridge) path"
    - "Wire IOLogger's recording step to read client._stereo_camera.read_left()/read_right() (the same shared StereoSplitCamera instance already feeding the policy) instead of a second capture"
    - "Log two separate images per step (e.g. camera_overhead_left/, camera_overhead_right/) instead of one camera_overhead/ frame, matching what the checkpoint's camera2/camera3 inputs actually are"
  debug_session: ""

- truth: "Robot motion during policy execution is smooth, tracking the model's predicted trajectory continuously"
  status: failed
  reason: "User observed the robot moves in small jerks rather than smoothly during the live episode."
  severity: major
  test: 1
  root_cause: "run_vla_episode.py's run_episode() control loop (lines ~189-238) runs at --control-hz (default 2.0 Hz) and calls robot.send_action() with a single absolute-position target every 0.5s tick, with zero interpolation between successive targets anywhere in the per-step loop. move_to_positions()'s P-control smoothing is only invoked for the start-of-episode/e-stop reset move, never per-tick during the episode. Each of a policy chunk's 50 predicted actions is a waypoint intended to be part of a continuous trajectory; sending them as discrete, 0.5s-apart absolute jumps with no path smoothing produces visible jerks. Since chunk playback pops from a locally-cached queue and does not require a network round-trip except at chunk boundaries, the per-tick execution rate can be raised (and/or waypoints interpolated) without adding server load."
  artifacts:
    - path: "control/run_vla_episode.py"
      issue: "run_episode()'s per-tick loop sends raw absolute joint targets with no interpolation, gated only by --control-hz (default 2.0)"
  missing:
    - "Increase the effective per-tick execution/interpolation rate independent of --control-hz's observation-sending cadence (chunk playback doesn't need a network round trip per tick)"
    - "Add waypoint interpolation between the previous sent position and the next target action before calling robot.send_action(), rather than jumping directly to each new absolute target"
  debug_session: ""

- truth: "Once the bridge experiences a transient stall, it recovers within one control-loop cycle -- not by waiting on an unbounded internal retry"
  status: failed
  reason: "User reported the robot got 'stuck' (not moving) mid-episode. Confirmed via Colab PolicyServer logs cross-referenced with episode.jsonl: after action chunk #192 was delivered (12:44:25 UTC), the server logged only repeating 'Starting receiver' lines for 6+ minutes with zero 'Running inference' lines, while the client's episode.jsonl showed obs_age_s/staleness growing unbounded (30s -> 227s) over steps ~192-241, before finally recovering at observation #241 (must_go: True)."
  severity: blocker
  test: 1
  root_cause: "policy_server.py's _enqueue_observation() silently drops any observation (via observations_similar()) unless obs.must_go is True. Once the robot holds still for any reason (e.g. one slow round-trip), subsequent camera frames/joint state look near-identical to the last processed observation, so the server filters every observation out and never generates a new chunk -- a self-sustaining deadlock. Recovery depends entirely on lerobot's own must_go Event/action_queue.empty() logic (vendored site-packages, lerobot/async_inference/robot_client.py:337,403-437), which did not reliably re-trigger for 6+ minutes this run (action_queue pops kept returning a real-but-increasingly-stale timed_action rather than falling through to the empty-queue fallback that would set must_go). This project's own bridge wrapper (vla_bridge/robot_client.py's BridgeActionSource) has no independent watchdog and fully depends on this vendored-library recovery path."
  artifacts:
    - path: "control/vla_bridge/robot_client.py"
      issue: "BridgeActionSource.get_action()/pop_validated_action() have no staleness watchdog of their own -- they rely entirely on lerobot's internal RobotClient.must_go/action_queue recovery, which stalled for 6+ minutes in this live run"
    - path: "control/.venv/lib/python3.12/site-packages/lerobot/async_inference/policy_server.py"
      issue: "_enqueue_observation()'s observations_similar() filter drops non-must_go observations once the robot is still, with no server-side staleness override (vendored dependency -- do not patch in place)"
  missing:
    - "Add a staleness watchdog inside vla_bridge/robot_client.py (this project's own code, not the vendored lerobot package): after N consecutive stale/no-action pop_validated_action() results (or obs_age_s exceeding a bound well under 6 minutes), force recovery -- e.g. directly set client.must_go and/or clear client.action_queue so the next control_loop_observation() call bypasses the server's similarity filter, instead of waiting on lerobot's internal event to fire on its own"
    - "Add a test exercising this watchdog against a fake client whose action_queue simulates the observed stuck state (real, non-empty, but growing-stale timed_action being returned every pop)"
  debug_session: ""

- truth: "Computed depth from the AR0144 stereo pair is calibrated and its accuracy is confirmed against a human-verifiable, real-world known-distance measurement -- not consumed as an unvalidated third input."
  status: scope_extension
  reason: "User-requested extension raised directly in conversation during Phase 12 gap-closure plan-phase (2026-09-26), not from the original 12-UAT.md live-hardware diagnosis (the 3 gaps above, all from the same episode). Based on deep research into 'StereoPatch: Patch-Aligned RGB-Depth Fusion for Spatial Perception in Robot Manipulation' (arXiv 2609.15509) and its underlying depth model, NVIDIA's Fast-FoundationStereo (NVlabs/Fast-FoundationStereo, CVPR 2026). StereoPatch's own custom RGB-depth fusion architecture (DeFM depth encoder + cross-attention 'StereoPatch Tokens') has NO public code release -- explicitly out of scope, not planned. FastFS's depth-computation half IS real, open-source, and copyable: it turns a rectified stereo pair + a supplied calibration file (flattened 3x3 intrinsics + baseline in meters) into a depth map, but does NOT rectify internally, is CUDA-GPU-only (no CPU path), and requires input <1000px width with dimensions divisible by 32. This project's Mac client has no CUDA GPU -- only the Colab side does (already running the SmolVLA PolicyServer). The paper itself validates depth only indirectly (closed-loop task success rate: 89.2% with a stereo pair+FastFS vs 51.4% with a RealSense D405, unexplained) -- the user wants a real human-verifiable checkpoint (a known real-world distance measured against computed depth, within a documented tolerance), which is stricter than anything the paper itself does."
  severity: enhancement
  test: N/A (new scope, not part of the original 3-gap live-hardware episode)
  root_cause: "N/A -- not a defect. No depth calibration/computation capability exists in this project prior to this addition."
  artifacts:
    - path: "control/vla_bridge/stereo_calibration.py"
      issue: "Does not exist yet -- new module needed for checkerboard-based stereo calibration (cv2.stereoCalibrate/stereoRectify against the AR0144's live left/right split) producing a saved calibration file in Fast-FoundationStereo's required format"
    - path: "control/vla_bridge/depth_camera.py"
      issue: "Does not exist yet -- new module needed to rectify a live stereo pair, downsample it to FastFS's input constraints (scaling intrinsics proportionally), and request a depth map from a Colab-hosted FastFS endpoint"
    - path: "control/vla_bridge/policy_server_launch.md"
      issue: "Documents only the existing gRPC PolicyServer/tunnel setup -- needs a new section documenting the Colab-side FastFS install/serve/tunnel cells, analogous to the existing PolicyServerConfig/serve() cell pattern"
  missing:
    - "Checkerboard-based stereo calibration producing a saved calibration file, with a human-confirmed low reprojection error (Plan 12-05)"
    - "A documented Colab-side Fast-FoundationStereo inference endpoint, reachable over a tunnel (Plan 12-06)"
    - "A local depth_camera.py that rectifies+downsamples a live stereo pair, requests a depth map from that endpoint, and records it (path referenced in episode.jsonl via IOLogger) -- recording-only this phase, never added to the PolicyServer's observation dict (SmolVLA has no depth input feature; no fusion architecture exists yet to consume it) (Plan 12-06)"
    - "A human-verifiable checkpoint: a known real-world distance measured against the computed depth value, within a documented tolerance -- stricter than the StereoPatch paper's own (indirect, task-success-only) validation (Plan 12-06)"
  debug_session: ""
