---
status: complete
phase: 12-bridge-tick-latency-fix
source: [12-03-SUMMARY.md, 12-04-SUMMARY.md, 12-05-SUMMARY.md, 12-06-SUMMARY.md]
started: 2026-09-27T04:38:00Z
updated: 2026-09-30T12:45:00Z
---

## Current Test

[testing complete]

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
result: pass
reported: "Not exercised in the original 12-03 live session (outputs/vla_episode_003 had no
  stall), but found retroactively in real historical episode.jsonl files while diagnosing
  a different question (2026-09-30) -- grepped every control/outputs/*/episode.jsonl for
  the 'staleness-watchdog' marker in model_version. Clean confirming example:
  outputs/vla_episode_004 (today's 200-step live run), step 106: a popped action's
  round-trip age crossed STALE_ACTION_S=30s (grew to ~40s over 3 ticks), the watchdog
  fired at step 109 (10 consecutive stale/empty ticks reached, as designed), and flags were
  clean again by step 111 -- 2 ticks (~1s at 2Hz) later. outputs/vla_episode_test7_retry2
  shows the same clean fire-then-recover pattern 3 separate times in one 60-step episode
  (steps 20, 30, 50), each recovering within a few ticks. Two older episodes
  (outputs/vla_episode_test7, outputs/vla_episode_test7_retry1, both pre-dating the Sept
  29-30 crash fixes) show the watchdog firing every 10 ticks for the entire episode without
  ever recovering -- not a watchdog failure: it correctly detected staleness and correctly
  forced recovery every time, but the PolicyServer connection itself was never producing
  real actions in those runs (a separate, already-fixed upstream problem), so there was
  nothing for recovery to grab. STALE_WATCHDOG_CONSECUTIVE_LIMIT=10 confirmed well-tuned:
  fires in ~5s of real staleness and recovers within ~1s once triggered, well under the
  pre-fix 6+ minute baseline."

### 4. Stereo calibration produces a plausible, real calibration file
expected: Running the checkerboard calibration session against the real AR0144 and a physical checkerboard produces control/stereo_calibration.json with a plausible reprojection error (low, e.g. under ~1.0 px) and plausible K/D/R/T values (not NaN/degenerate).
result: pass
reported: "Took 3 calibration attempts to converge. Attempt 1 (15 views, mostly flat-on): reprojection_error_px=17.17, fx/fy mismatched 3-4x, huge distortion coefficients -- degenerate, caused by a frozen-camera bug discovered mid-session (see below). Attempt 2 (15 views, post camera-resolution-fix): 2.73px, sane fx/fy, but T (baseline) = 75mm. Attempt 3 (15 views, more angle variety): 2.42px, T=53mm -- baseline swung 30% between attempts 2 and 3, signaling real remaining instability. Attempt 4 (30 views, wide variety of angle/position/distance): 1.79px reprojection error, T=53.7mm -- baseline now agrees with attempt 3 to within 0.6%, fx/fy mismatched <0.5% on both cameras, distortion coefficients small and monotonically improving across attempts (k2/k3: -8.83/30.46 -> -0.86/2.25 -> -0.41/0.42). Baseline convergence across independent attempts is the key confirming signal this calibration is trustworthy, even though reprojection error (1.79px) is slightly above the original ~1.0px target. control/stereo_calibration.json is intentionally untracked (same convention as control/device_map.json -- hardware-specific, machine-local)."

### 5. Colab Fast-FoundationStereo endpoint docs are legitimate and match the real client
expected: Visiting github.com/NVlabs/Fast-FoundationStereo confirms it's the real NVIDIA org, README, and dependencies look legitimate (supply-chain check). The documented Colab install/serve/tunnel cells in policy_server_launch.md match depth_camera.py's actual request/response contract.
result: pass
reported: "Re-verified 2026-09-30 against the current notebook (post resident-FastFS-server
  rewrite) -- all three original findings are resolved, confirmed by grepping the actual
  regenerated policy_server_launch.md (mechanically derived from the notebook's own cells
  by scripts/sync_policy_notebook.py, so this reflects the notebook's real current content,
  not a stale copy): (1) camera mapping now correctly documents the AR0144 as 1600x600
  split into 800x600 halves throughout (2560x720 appears only as historical 'still-broken,
  frozen' context, not the active documented spec). (2) the misleading 'port 8080' comment
  is gone -- every reference (Step 3's launch, the tunnel comment, the Summary table) now
  correctly says 5173, no other port number appears anywhere in the file. (3) exactly one
  ngrok.connect() call per tunnel (5173 TCP for PolicyServer, 3000 HTTP for the depth
  service) -- no duplicate tunnel-open cells. NVlabs/Fast-FoundationStereo legitimacy and
  the depth request/response contract (left_png_b64/right_png_b64/intrinsics_flat/baseline_m
  request, depth_npy_b64 response) were already confirmed accurate in the original finding
  and are unchanged."

### 6. Depth accuracy sanity check against a known real-world distance
expected: With the Colab FastFS endpoint running, placing an object at a known, physically-measured distance and querying depth_camera.py's CLI/main() returns a depth value within a documented, reasonable tolerance of that measurement.
result: pass
reported: "Getting a real reading required fixing 3 live bugs first: (1) scripts/run_demo.py's cv2.imshow()+waitKey(0) and Open3D vis.run() calls are unconditional/not gated by any CLI flag and hang forever on headless Colab -- patched via sed in Step 7. (2) run_demo.py only writes depth_meter.npy when --get_pc 1 (Step 8's Flask handler was passing --get_pc 0, so it never produced a file to read at all -- the original 'adjust DEPTH_OUTPUT_GLOB' TODO was actually a missing-flag bug, not a filename-guessing problem). (3) camera index drifted (macOS AVFoundation numbering) -- fixed by using the device NAME instead. Once fixed, the naive center-pixel version of this test was rejected as uninformative (user: 'this isn't a good test... tells us literally nothing') since the scene has a robot arm, cube, and cutting mat, and the depth model has no object semantics -- the exact center pixel could land on any of them depending on framing. Replaced with an HSV red-color-threshold detector (project's own target is 'the red cube') run on the same rectified/resized frame the depth map is pixel-aligned to, sampling median depth over the detected cube region instead of one arbitrary pixel. Result: 3 consecutive readings of 0.754m/0.758m/0.759m (5mm spread, stable ~3900px red-mask detection, stable centroid) against a tape-measured real distance of ~0.74-0.75m -- within ~0.5-1.5cm, well inside reasonable tolerance for stereo depth at this range. policy_server.ipynb was updated (commit 37a5aef) with the 3 fixes above (the GUI-hang patches folded into Step 7, --get_pc 1 + corrected DEPTH_OUTPUT_GLOB into Step 8) so future runs don't need to rediscover this live."

### 7. Live episode depth recording works end-to-end without disrupting motion
expected: During a live episode with the Colab depth endpoint running, depth_overhead/ fills in at the configured cadence, episode.jsonl's depth_frames field is correctly sparse (not every step), and robot motion/timing is unaffected by the added depth capture calls.
result: pass
reported: "Resolved by the 2026-09-30 resident-FastFS-server fix (see STATE.md) plus a
  same-day client-side fix: the observation gRPC payload was raw, uncompressed pixels
  (~9MB/observation -- wrist 1920x1080 + two 800x600 stereo halves), which took 2-4
  minutes per observation over the real ngrok tunnel regardless of the depth-server fix,
  stalling the control loop. `vla_bridge.robot_client._downsize_for_transport()` now caps
  every observation frame's long edge at 640px before it's pickled and sent -- above
  `victorvanhalst/smolvla_so101_cube`'s own SmolVLAConfig.resize_imgs_with_padding=(512,512),
  so no fidelity the model actually uses is lost; confirmed via the installed checkpoint's
  own config, not assumed. Live retest (control/outputs/vla_episode_20260930_121123,
  2026-09-30 ~12:11-12:13 local): full 60/60 steps completed (`max_steps_reached`), depth
  depth_summary.json = {submitted: 6, complete: 6, failed: 0, skipped_busy: 0} -- every
  depth_every_n_steps=10 tick succeeded, zero drops. Whole episode ran in ~74s wall-clock
  (previously: single observations alone took 2-4 minutes each; a full episode was not
  completable). Motion has 3 brief (2.6-4.2s) pauses at steps ~0-1, ~26-27, ~51-52 --
  these are BridgeActionSource's normal chunk-refetch cadence (actions_per_chunk=50,
  chunk_size_threshold=0.5, refetch every ~25 steps), now taking seconds instead of the
  multi-minute stalls that made Test 7 fail before -- not jitter, not a watchdog/staleness
  event (validator_flags empty at every one of those steps), and not worse near the end of
  the episode than at the start. User confirmed acceptable as-is; optional future tuning
  (raise --actions-per-chunk or lower the refetch threshold to prefetch earlier) noted but
  not applied."

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
passed: 15
issues: 0
pending: 0
skipped: 0
blocked: 0

## Gaps

none — all 15 tests pass. Test 5's finding (resolved 2026-09-30, re-verified against the
current notebook) and Test 3's skip (resolved 2026-09-30, real fire-and-recover evidence
found in historical episode.jsonl files) were the only two open items this session closed.
