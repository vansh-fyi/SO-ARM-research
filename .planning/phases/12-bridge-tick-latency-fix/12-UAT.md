---
status: testing
phase: 12-bridge-tick-latency-fix
source: [12-03-SUMMARY.md, 12-04-SUMMARY.md, 12-05-SUMMARY.md, 12-06-SUMMARY.md]
started: 2026-09-27T04:38:00Z
updated: 2026-09-27T13:30:00Z
---

## Current Test

number: 7
name: Live episode depth recording works end-to-end without disrupting motion
expected: |
  During a live episode with the Colab depth endpoint running, depth_overhead/ fills in at
  the configured cadence, episode.jsonl's depth_frames field is correctly sparse (not every
  step), and robot motion/timing is unaffected by the added depth capture calls.
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
result: issue
reported: "NVlabs/Fast-FoundationStereo legitimacy and the depth request/response contract itself are fine -- left_png_b64/right_png_b64/intrinsics_flat/baseline_m request and depth_npy_b64 response match exactly across policy_server_launch.md Step 8, policy_server.ipynb's Step 8 cell, and depth_camera.py's compute_depth()/Flask handler. But policy_server.ipynb is stale relative to policy_server_launch.md and the real hardware: (1) the notebook's Camera mapping markdown cell still documents the AR0144 as native 2560x720 split into 1280/1280 halves -- the old broken resolution mode -- while the .md and the real, currently-stable rig use 1600x600 split into 800/800 halves (confirmed by test 4's calibration work); following the notebook alone would crop the wrong columns. (2) Both the .md and the notebook contain a self-contradictory comment on the depth tunnel cell ('a DIFFERENT local port than Step 3's gRPC server (8080)') when Step 3 actually uses port 5173 in both the code and the Summary table -- harmless since the actual ngrok.connect(5173, \"tcp\") call is correct, but misleading. (3) minor: notebook cells 13 and 14 are duplicate tunnel-open cells that would open two tunnels if both are run."
severity: major

### 6. Depth accuracy sanity check against a known real-world distance
expected: With the Colab FastFS endpoint running, placing an object at a known, physically-measured distance and querying depth_camera.py's CLI/main() returns a depth value within a documented, reasonable tolerance of that measurement.
result: pass
reported: "Getting a real reading required fixing 3 live bugs first: (1) scripts/run_demo.py's cv2.imshow()+waitKey(0) and Open3D vis.run() calls are unconditional/not gated by any CLI flag and hang forever on headless Colab -- patched via sed in Step 7. (2) run_demo.py only writes depth_meter.npy when --get_pc 1 (Step 8's Flask handler was passing --get_pc 0, so it never produced a file to read at all -- the original 'adjust DEPTH_OUTPUT_GLOB' TODO was actually a missing-flag bug, not a filename-guessing problem). (3) camera index drifted (macOS AVFoundation numbering) -- fixed by using the device NAME instead. Once fixed, the naive center-pixel version of this test was rejected as uninformative (user: 'this isn't a good test... tells us literally nothing') since the scene has a robot arm, cube, and cutting mat, and the depth model has no object semantics -- the exact center pixel could land on any of them depending on framing. Replaced with an HSV red-color-threshold detector (project's own target is 'the red cube') run on the same rectified/resized frame the depth map is pixel-aligned to, sampling median depth over the detected cube region instead of one arbitrary pixel. Result: 3 consecutive readings of 0.754m/0.758m/0.759m (5mm spread, stable ~3900px red-mask detection, stable centroid) against a tape-measured real distance of ~0.74-0.75m -- within ~0.5-1.5cm, well inside reasonable tolerance for stereo depth at this range. policy_server.ipynb was updated (commit 37a5aef) with the 3 fixes above (the GUI-hang patches folded into Step 7, --get_pc 1 + corrected DEPTH_OUTPUT_GLOB into Step 8) so future runs don't need to rediscover this live."

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
passed: 12
issues: 1
pending: 1
skipped: 1
blocked: 0

## Gaps

- truth: "The documented Colab install/serve/tunnel cells in policy_server_launch.md match depth_camera.py's actual request/response contract, and the Colab notebook stays in sync with the .md."
  status: failed
  reason: "User reported: policy_server.ipynb's Camera mapping cell still documents the AR0144 as native 2560x720 / 1280+1280 split (the old broken resolution), stale against the .md's and current hardware's real 1600x600 / 800+800 split. Also both .md and .ipynb carry a self-contradictory comment claiming Step 3's gRPC server uses port 8080 when it actually uses 5173 (code and Summary table both say 5173). Minor: notebook has duplicate tunnel-open cells (13 and 14)."
  severity: major
  test: 5
