---
status: testing
phase: 12-bridge-tick-latency-fix
source: [12-03-SUMMARY.md, 12-04-SUMMARY.md, 12-05-SUMMARY.md, 12-06-SUMMARY.md]
started: 2026-09-27T04:38:00Z
updated: 2026-09-27T11:03:43Z
---

## Current Test

number: 5
name: Colab Fast-FoundationStereo endpoint docs are legitimate and match the real client
expected: |
  Visiting github.com/NVlabs/Fast-FoundationStereo confirms it's the real NVIDIA org,
  README, and dependencies look legitimate (supply-chain check). The documented Colab
  install/serve/tunnel cells in policy_server_launch.md match depth_camera.py's actual
  request/response contract.
awaiting: user response

## Tests

### 1. Camera-overhead recording shows the real AR0144 workspace (left + right)
expected: camera_overhead_left/ and camera_overhead_right/ images show the real robot workspace from the AR0144, not the laptop webcam, as two distinct viewpoints.
result: pass

### 2. Robot motion is visibly smooth during policy execution
expected: With interpolated waypoints active (--execution-hz decoupled from --control-hz), the arm's motion traces a continuous path rather than discrete jerky jumps. Confirm whether the default --execution-hz=20.0 feels right or needs tuning.
result: pass
reported: "Tuned live via a hardware-free multi-joint sweep script exercising the real run_episode() interpolation path: 20 Hz still jerky, 60 Hz smoother, 100 Hz even better. Default updated to 100.0 in control/run_vla_episode.py (commits 0faf142, 58ab5b0)."

### 3. Stale-deadlock watchdog breaks a real multi-minute stall
expected: During a live episode, if the bridge stalls (stale/no-action ticks), the watchdog forces recovery within a few seconds (not 6+ minutes as observed pre-fix). Inspect episode.jsonl for any model_version containing "staleness-watchdog" and confirm real actions resume shortly after. Confirm whether STALE_WATCHDOG_CONSECUTIVE_LIMIT=10 (~5s at 2Hz) feels well-tuned.
result: skipped
reason: "User reported no pauses/stalls occurred during the whole episode (outputs/vla_episode_003, 300 steps) -- confirmed via episode.jsonl: 0 staleness-watchdog triggers, only 2 isolated (non-consecutive) stale-observation steps and 1 no-action step, never reaching the 10-in-a-row threshold. The watchdog's core recovery behavior was never exercised because no deadlock occurred -- this is a healthy outcome for the episode but leaves the fix's actual recovery-speed claim unverified against a real stall. Unit tests (12-04's automated coverage) already prove the mechanism triggers correctly in isolation; only the live end-to-end recovery-speed confirmation remains untested."

### 4. Stereo calibration produces a plausible, real calibration file
expected: Running the checkerboard calibration session against the real AR0144 and a physical checkerboard produces control/stereo_calibration.json with a plausible reprojection error (low, e.g. under ~1.0 px) and plausible K/D/R/T values (not NaN/degenerate).
result: pass
reported: "Took 3 calibration attempts to converge. Attempt 1 (15 views, mostly flat-on): reprojection_error_px=17.17, fx/fy mismatched 3-4x, huge distortion coefficients -- degenerate, caused by a frozen-camera bug discovered mid-session (see below). Attempt 2 (15 views, post camera-resolution-fix): 2.73px, sane fx/fy, but T (baseline) = 75mm. Attempt 3 (15 views, more angle variety): 2.42px, T=53mm -- baseline swung 30% between attempts 2 and 3, signaling real remaining instability. Attempt 4 (30 views, wide variety of angle/position/distance): 1.79px reprojection error, T=53.7mm -- baseline now agrees with attempt 3 to within 0.6%, fx/fy mismatched <0.5% on both cameras, distortion coefficients small and monotonically improving across attempts (k2/k3: -8.83/30.46 -> -0.86/2.25 -> -0.41/0.42). Baseline convergence across independent attempts is the key confirming signal this calibration is trustworthy, even though reprojection error (1.79px) is slightly above the original ~1.0px target. control/stereo_calibration.json is intentionally untracked (same convention as control/device_map.json -- hardware-specific, machine-local)."

### 5. Colab Fast-FoundationStereo endpoint docs are legitimate and match the real client
expected: Visiting github.com/NVlabs/Fast-FoundationStereo confirms it's the real NVIDIA org, README, and dependencies look legitimate (supply-chain check). The documented Colab install/serve/tunnel cells in policy_server_launch.md match depth_camera.py's actual request/response contract.
result: [pending]

### 6. Depth accuracy sanity check against a known real-world distance
expected: With the Colab FastFS endpoint running, placing an object at a known, physically-measured distance and querying depth_camera.py's CLI/main() returns a depth value within a documented, reasonable tolerance of that measurement.
result: [pending]

### 7. Live episode depth recording works end-to-end without disrupting motion
expected: During a live episode with the Colab depth endpoint running, depth_overhead/ fills in at the configured cadence, episode.jsonl's depth_frames field is correctly sparse (not every step), and robot motion/timing is unaffected by the added depth capture calls.
result: [pending]

### 8. force_bridge_recovery() drains the queue and sets must_go (automated)
expected: force_bridge_recovery(client) drains client.action_queue to empty and sets client.must_go
result: pass
source: automated
coverage_id: 12-04/D1

### 9. Watchdog triggers at the configured consecutive-stale limit (automated)
expected: BridgeActionSource.get_action() forces recovery after exactly N consecutive stale (growing-stale-action) results, not before
result: pass
source: automated
coverage_id: 12-04/D2

### 10. Watchdog also triggers on consecutive empty-queue results (automated)
expected: The same watchdog also triggers on N consecutive empty-queue (no-action-available) results
result: pass
source: automated
coverage_id: 12-04/D3

### 11. Watchdog does not over-trigger; resets on fresh result (automated)
expected: A single stale/empty result below the limit never triggers recovery; a genuine fresh result resets the counter to zero
result: pass
source: automated
coverage_id: 12-04/D4

### 12. Calibration math functions fully covered by hardware-free tests (automated)
expected: build_object_points(), detect_checkerboard_corners(), calibrate_single_camera(), calibrate_stereo_pair(), save_calibration()/load_calibration(), and run_calibration_session() all implemented with deterministic test coverage
result: pass
source: automated
coverage_id: 12-05/D1

### 13. Calibration session enforces minimum valid views (automated)
expected: run_calibration_session() never proceeds on fewer than 3 valid views (raises RuntimeError), correctly skips undetected-corner views without counting them
result: pass
source: automated
coverage_id: 12-05/D2

### 14. DepthCameraClient rectify/downsample/request pipeline (automated)
expected: DepthCameraClient rectifies a live AR0144 stereo pair, downsamples to FastFS's input constraints (scaling intrinsics proportionally), and POSTs to a Colab-hosted FastFS endpoint, returning a metric depth map
result: pass
source: automated
coverage_id: 12-06/D1

### 15. Depth recording wiring is correct and isolated from the bridge observation path (automated)
expected: Depth is recorded into episode.jsonl via IOLogger.capture_depth_map() at a configurable cadence, sourced from the SAME tick's stereo frame already used for diagnostic recording (object-identity verified), with zero references to depth in vla_bridge/robot_client.py
result: pass
source: automated
coverage_id: 12-06/D4

## Summary

total: 15
passed: 11
issues: 0
pending: 3
skipped: 1
blocked: 0

## Gaps
