---
gsd_state_version: 1.0
milestone: v2.1
milestone_name: MLLM Raw-Autonomy Benchmark
current_phase: 16
current_phase_name: Cross-Backend Pen Transfer Benchmark
status: verifying
stopped_at: Completed 15-10-PLAN.md
last_updated: "2026-10-26T07:00:22.575Z"
last_activity: 2026-10-26
last_activity_desc: "Completed quick task 261025-wbw: every real-robot episode starts at the committed home pose and returns to it"
progress:
  total_phases: 17
  completed_phases: 13
  total_plans: 73
  completed_plans: 73
  percent: 76
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-09-24)

**Core value:** A researcher types a task prompt and watches SOARM execute it in a LIBERO simulation — the loop from language to embodied action.
**Current focus:** Phase 14 hardware UAT (/gsd-verify-work 14, resumes at Test 1) and the open Phase 13 camera UAT (/gsd-verify-work 13), both before Phase 16

## Current Position

Phase: 16 — Cross-Backend Pen Transfer Benchmark
Plan: Not started
Status: Awaiting hardware UAT verification: Phase 14 (/gsd-verify-work 14) and the open Phase 13 (/gsd-verify-work 13), before Phase 16
Last activity: 2026-10-26 - Completed quick task 261026-h12: docs refresh for Phase 14 UAT readiness (home pose, notebooks, checkpoints)

Plan execution: [██████████] 100% — phase acceptance pending

## Performance Metrics

**Velocity:**

- Total plans completed: 40
- Average duration: -
- Total execution time: 0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 02 | 5 | - | - |
| 04 | 5 | - | - |
| 05 | 4 | - | - |
| 10 | 5 | - | - |
| 11 | 5 | - | - |
| 12 | 6 | - | - |
| 15 | 10 | - | - |

**Recent Trend:**

- Last 5 plans: none
- Trend: -

*Updated after each plan completion*
| Phase 13 P01 | 3min | 2 tasks | 2 files |
| Phase 13 P02 | 3min | 1 tasks | 2 files |
| Phase 13 P03 | 8min | 2 tasks | 2 files |
| Phase 13 P04 | 3min | 3 tasks | 2 files |
| Phase 15 P09 | 9 min | 3 tasks | 7 files |
| Phase 15 P10 | 14h 18m | 3 tasks | 17 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- **Phase 12 closed (2026-09-30):** 15/15 UAT tests pass. Root causes fixed:
  (1) `control_loop_observation()` wasn't gated on queue-empty (LATENCY-01) —
  fixed in 12-01/02. (2) FastFS reloaded its full model from disk on every
  depth request (15-70s/call) — fixed with a resident Flask service
  (`vla_bridge/fastfs_server.py`) that loads the model once. (3) The
  observation payload itself was raw, uncompressed pixels (~9MB) — fixed with
  `vla_bridge.robot_client._downsize_for_transport()`, capping frames at
  640px (above the checkpoint's own 512x512 resize target, so no fidelity
  lost). Live-verified end to end: a 200-step episode completed cleanly,
  including one real watchdog fire-and-recover cycle (stale at step 106,
  recovered by step 111 — ~1s, well under the pre-fix 6+ minute baseline).
  Test 5's stale-notebook-docs finding also confirmed resolved by re-reading
  the current notebook. See `12-UAT.md` for full evidence per test.

- **v2.1 roadmap (2026-09-24):** Phases 12-17 derived from the 26 v2.1 requirements (LATENCY ×4, SAFETY ×1, CAMFIX ×3, PIPE ×2, VLAB ×4, MLLM ×7, EVAL ×2, DEBT ×3). Phase 12 (latency fix + 2 debt items) → Phase 13 (camera resolution unification) form a chain. Phases 14 (VLA-style pipeline + 4 backends) and 15 (MLLM-style pipeline + 3 backends) are **independent and parallel-capable** — different `ActionSource` pattern, different backend set — both gated on Phase 13's trustworthy camera pipeline, with Phase 15 additionally gated on Phase 12's real latency data for chunk sizing. Phase 16 (8-backend Pen Transfer comparison) depends on both 14 and 15. Phase 17 (safety-cap re-tightening + wrist_roll debt pin) is deliberately sequenced LAST — resequenced 2026-09-24 per explicit user decision to keep loosened soft-margin caps in place through the whole experiment so they don't mask genuine model behavior; hard limits (absolute joint clamping, NaN/inf rejection, e-stop) stay on unmodified throughout regardless. Full rationale in ROADMAP.md Overview.
- **Requirement-ID collision found during roadmap creation, fixed same session (2026-09-24):** v2.1's VLA Backends requirements (Phase 14) originally reused v1's `VLA-01..04` IDs (VLA Inference Pipeline, Phase 3) — unintentional authoring duplication, not a re-scoping. Renamed to `VLAB-01..04` in REQUIREMENTS.md/ROADMAP.md to remove the collision.
- **Phase 10 established model (2026-09-20):** User accepted the complete
  Coppelia-aligned assembly. Black housing, yellow jaws, `[0, 0.036]` joint and
  actuator ranges, `+1` opens / `-1` closes, mechanical jaw equality, ~88°
  wrist-flex initial pose. This supersedes earlier 84 mm aperture and visual-only
  correction assumptions. See [10-ESTABLISHED-MODEL.md](phases/10-digital-twin-fidelity/10-ESTABLISHED-MODEL.md).

- **v2.0 roadmap (2026-09-18):** Phases 10-11 derived from the 12 v2.0 requirements (TWIN-01..07, VLAHW-01..05). Phase 10 (Digital-Twin Fidelity) and Phase 11 (VLA Hardware Connection) are **independent and parallel-capable, not a sequential chain**.
- **v2.0 scope narrowing (2026-09-17):** An initial general-MLLM-prompting experiment (`experiment-design/`, `docs/multimodal-context-ablation-experiment.md`) failed badly. Milestone narrowed to two concrete workstreams (digital-twin fix + VLA hardware connection) before committing to the full raw-autonomy MLLM-benchmark-suite design.
- Research: OpenVLA-OFT chosen as primary VLA (97.1% LIBERO avg, T4-compatible at 4-bit)
- Research: SOARM MJCF to be derived from `so101_new_calib.xml`, not built from scratch
- 01-close: **REQUIRED READING for phases 2-6 before touching the Colab environment:** `.planning/phases/01-colab-environment-setup/01-DEBUG-HISTORY.md`
- [Phase 04]: GRIPPER UPGRADE (LOCKED, D-07): stock ~2-3cm jaw can't grasp any LIBERO object; adopted roboninecom 84mm parallel gripper — carries forward to Phase 10's real-gripper-direction success criterion (TWIN-03/07)
- [Phase 10 context]: `coppelia/export_model_library.py`'s own comments confirm the exported URDF's bug: the gripper is positioned near the wrist but never parented under the arm.
- [Phase 13]: Numeric wrist_name fallback values skip live name probing.
- [Phase 15]: 15-09: mllm_smoke's SyntheticCollector delegates to the real MllmInputCollector fed synthetic wrist/stereo/depth sources, so smoke inputs match the robot path by construction
- [Phase 15]: 15-09: PIPE-02/MLLM-* requirement checkboxes stay Pending until Plan 15-10 live verification (plan forbids changing checkbox/traceability state here)
- [Phase 15]: 15-10 (user decision 2026-10-02): mllm-hf-qwen re-pinned from cerebras to deepinfra (Qwen/Qwen3.8-27B:deepinfra, $0.20/$2.50 per M, context 262144, source router /v1/models 2026-10-02). Evidence: two live smoke attempts got HTTP 503 'Provider cerebras ... paused by the circuit breaker' and the router listing no longer offers cerebras for this model (15-LIVE-RESULTS.md sections 1/2)
- [Phase 15]: 15-10 (user decision 2026-10-02): mllm-hf-deepseek model switched from deepseek-ai/DeepSeek-V4-Flash-Vision-Exp to deepseek-ai/DeepSeek-V4.1-Flash pinned :deepinfra ($0.20/$0.60 per M, context 1048576); this SUPERSEDES the locked D-23 model id (and D-27). Backend name and effort max unchanged. Evidence: DeepSeek retired V4-Flash-Vision-Exp upstream (aliases route to V4.1-Flash) and both live attempts got HTTP 504 from the HF router at ~121 s
- [Phase 15]: 15-10 (user decision 2026-10-02, Rule 1 bug fix): base_mllm_chat.py now streams (chat.completions.create stream=True, stream_options include_usage) because the non-streaming call sent no bytes until done and the HF router edge proxy cut idle connections at ~120 s (HTTP 504). The raw sidecar payload is now a JSON-safe reconstruction of the stream. Re-smoke of mllm-hf-qwen and mllm-hf-deepseek pending
- [Phase 15]: [2026-10-02] Plan 15-10 (user decision): mllm-claude switched from claude-fable-5-1 to claude-opus-5-5 (backend name unchanged). Prices 4/20/0.20 per MTok (platform.claude.com/docs/en/models/opus-5-5/overview); effort kept xhigh explicitly (Opus 5.5 default is medium), max_tokens 64000, adaptive thinking summarized. Omitted sampling reasons now say 'not sent' (the Opus 5.5 docs do not state a sampling-param 400). The Fable smoke (51.5 s, 0.2734 USD) is historical only; Opus 5.5 re-smoke pending (15-LIVE-RESULTS.md sections 1, 2, 6).
- [Phase 15]: 15-10 (user decision 2026-10-03, re-confirmed against the Task 3 episode table): chunk_len stays 20 for all five MLLM backends. Episode duty cycles at N=20: qwen 5.1%, deepseek 7.1%, claude 7.8%, gpt 8.9%, kimi 16.2%; no call exceeded timeout_s 900. mllm_defaults.json keeps the smoke-based values; N or effort changes are Phase 16 ablations — Shared fixed N (D-17) and the open-loop bound N <= 20; low duty cycle recorded as a property of long-thinking models (15-LIVE-RESULTS.md section 6)
- [Phase 15]: 15-10 (Rule 1 fix 9ac4472): run_vla_episode.py sets termination reason keyboard_interrupt before a guarded print, because Ctrl-C also kills a piped reader (| grep -v objc) and the print raised BrokenPipeError first, recording max_steps_reached. Live re-test: keyboard_interrupt. Separately 3f4d5e1 scoped the section-6 test parser to the smoke table — Found in Task 3 step D run 1 on mllm-hf-deepseek; regression test test_keyboard_interrupt_reason_survives_broken_stdout_pipe; full suite 1604 passed

### Pending Todos

Phase 14: capture and commit the home pose (`.venv/bin/python start_pose.py` in control/ -> control/start_pose.json), restore http_openvla_oft.ipynb with `git checkout` once the user closes the VS Code tab, then run the hardware UAT with `/gsd-verify-work 14` (20 tests pending, resumes at Test 1; see 14-UAT.md and the phase's .continue-here.md).

Phase 13: run fresh live camera UAT (`/gsd-verify-work 13`); implementation and automated verification complete.

### Blockers/Concerns

- 04 embodiment note (still binding for any new v2.1 task/object work): SO-ARM101 is a small ~500g-payload desktop arm; objects ≤84mm graspable, objects AND targets within ~0.45m reach, avoid the base's forward centerline collision corridor.
- 05 PRE-EXISTING TEST DEBT (unrelated to v2.1, still open): `test_replay.py::test_verify_full_obs_regeneration_passes_on_04_02_output` pixel-mismatch failure, not investigated.
- 10 RISK — RESOLVED (2026-09-24): [huggingface/lerobot#2210](https://github.com/huggingface/lerobot/issues/2210) did NOT reproduce in Phase 11's live episode.
- v2.1 camera bug (Phase 13 fix implemented and automated checks passed; fresh live visual verification pending): `camera_overhead` recordings in both `vla_episode_001` and the Phase 11 go/no-go episode are confirmed to be the laptop webcam, not the robot workspace — recording path never got the name-based camera-resolution fix already applied to the inference-input path.

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 260902-kcf | Update PROJECT.md: physical hardware integration is now an active parallel track (not out of scope) | 2026-09-02 | 1b4201c | [260902-kcf-update-project-md-physical-hardware-inte](./quick/260902-kcf-update-project-md-physical-hardware-inte/) |
| fast-260909 | Update progress on physical hardware build in docs (leader arm + servo swap complete) | 2026-09-09 | 9df5ecb | — |
| 260911-h5h | Create a 3D-printable STL mount bracket for the Waveshare AR0144 Stereo USB Camera module | 2026-09-11 | 47c8691 | [260911-h5h-create-a-3d-printable-stl-mount-bracket-](./quick/260911-h5h-create-a-3d-printable-stl-mount-bracket-/) |
| 260919-h8v | Fix Phase 10 TWIN-07 gap: flip gripper_left/gripper_right axis polarity in soarm_gripper.xml, regenerate URDF, update test expectations (hardware re-test still required) | 2026-09-19 | 5fa8e62 | [260919-h8v-fix-phase-10-twin-07-gap-flip-gripper-le](./quick/260919-h8v-fix-phase-10-twin-07-gap-flip-gripper-le/) |
| 260924-e3d | Loosen safety_validator.py caps (per-step 5→40deg, velocity 30→240deg/s, staleness 1s/3s→10s/30s) so the real 11-05 VLA episode can produce visible motion despite Colab round-trip latency -- user explicitly deprioritized safety margin in the empty test environment | 2026-09-24 | 9d664b7 | [260924-e3d-loosen-safety-validator-py-caps-displace](./quick/260924-e3d-loosen-safety-validator-py-caps-displace/) |
| 260924-gih | Fix pop_validated_action() silently dropping every real bridge action to `{}` -- lerobot's `.pos`-suffixed action_features keys never matched safety_validator's plain JOINT_ORDER names, crashing the first live 11-05 episode | 2026-09-24 | 41b6a0e | [260924-gih-fix-pop-validated-action-dropping-every-](./quick/260924-gih-fix-pop-validated-action-dropping-every-/) |
| 260927-he3 | Fix AR0144 stereo camera resolution: switch from frozen 2560x720 (confirmed stuck via raw ffmpeg testing and a real episode's 300 identical recorded frames) to working 1280x360 | 2026-09-27 | caa85da | [260927-he3-fix-ar0144-stereo-camera-resolution-swit](./quick/260927-he3-fix-ar0144-stereo-camera-resolution-swit/) |
| 260927-i3s | Upgrade AR0144 stereo camera resolution from 1280x360 to 1600x600 (800x600/eye) -- best available resolution after macOS's legacy-camera-plugins system override + reboot still failed to unstick the native 2560x720 mode | 2026-09-27 | 0545951 | [260927-i3s-upgrade-ar0144-stereo-camera-resolution-](./quick/260927-i3s-upgrade-ar0144-stereo-camera-resolution-/) |
| 260927-ndz | Fix Phase 12 UAT test-5 gaps in the Colab depth-endpoint setup: wire real Fast-FoundationStereo run_demo.py CLI into the Flask depth cell, fix stale port-8080 refs, sync notebook camera-mapping + drop duplicate tunnel cell | 2026-09-27 | a9ab896 | [260927-ndz-fix-phase-12-uat-test-5-gaps-in-the-cola](./quick/260927-ndz-fix-phase-12-uat-test-5-gaps-in-the-cola/) |
| fast-260927 | Add Colab checkpoint-upload helper cell (google.colab.files.upload) for Fast-FoundationStereo's model_best_bp2_serialize.pth, in both policy_server_launch.md and policy_server.ipynb | 2026-09-27 | f3ec70c | — |
| 261004-dz7 | Phase-14 live backend smoke CLI (backend_smoke.py) modelled on mllm_smoke.py: synthetic obs, gRPC/HTTP/Gemini lanes, no hardware; plus 14-LIVE-RESULTS.md template (live smokes still PENDING) | 2026-10-04 | 2e19968 | [261004-dz7-phase-14-live-backend-smoke-cli-backend-](./quick/261004-dz7-phase-14-live-backend-smoke-cli-backend-/) |
| 261018-px9 | Split policy_server.ipynb into 13 self-contained per-backend Colab notebooks (control/vla_bridge/notebooks/) generated by scripts/build_colab_notebooks.py with drift test, diagnostic cells, colab_server_utils launch guard; groot-n17 default checkpoint -> soccer-red_bowl | 2026-10-18 | 2bb51df | [261018-px9-split-policy-server-ipynb-into-per-backe](./quick/261018-px9-split-policy-server-ipynb-into-per-backe/) |
| fast | Add top-level README.md from the updated codebase map | 2026-10-18 | 655500e | — |
| fast | Drop literal <port> <robot_id> placeholders from diffusion/groot-n17 notebook episode commands | 2026-10-18 | 27714b8 | — |
| fast | Pin openvla-oft venv deps (numpy<2, protobuf 3.20.3, tensorflow-metadata 1.14.0, wandb 0.16.6, opencv-headless 4.11.0.86) so tensorflow 2.15 imports on fresh Colab | 2026-10-25 | 3d8adda | — |
| fast | Record Phase 14 live backend smoke results (10 PASS, 4 deferred) in 14-LIVE-RESULTS.md; refresh phase 14 handoff | 2026-10-25 | bc83e8a | — |
| fast | Gemini Robotics thinking_level medium -> low in both backend configs (user decision; smoke latency 8.45/8.67 s vs 10 s timeout) | 2026-10-25 | 0c7843b | — |
| 261025-wbw | Every real-robot episode moves to one committed home pose (control/start_pose.json, captured with `python start_pose.py`) before the first model call, refuses to start if not reached within 5, and returns there on every exit; tune_execution_hz aligned; COMMANDS.md documented | 2026-10-26 | 552efd1 | [261025-wbw-fixed-start-pose-for-every-episode](./quick/261025-wbw-fixed-start-pose-for-every-episode/) |
| 261026-h12 | Docs refresh for Phase 14 UAT readiness: 14-UAT.md home-pose precondition + per-backend notebooks + current checkpoints, live-results finding 10, handoff, STATE/README, codebase map | 2026-10-26 | 153ff56 | [261026-h12-docs-refresh-for-phase-14-uat-readiness](./quick/261026-h12-docs-refresh-for-phase-14-uat-readiness/) |

## Deferred Items

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| Physical Robot | SOARM hardware transfer (PHYS-01 to PHYS-03) | v2 deferred | Init |
| Advanced Spatial | Ego3D encoding, point cloud input (ADV-01 to ADV-03) | v2 deferred | Init |
| Interactive UI | Real-time REPL, web dashboard (INT-01, INT-02) | v2 deferred | Init |
| v1.1 Benchmark Work | Checkpoint Benchmark Suite + Re-Fine-Tuning (Phases 8-9) | Paused for v2.0 | 2026-09-15 |
| v2.1 Future Requirements | Provider-agnostic router beyond 3 backends, full 4-task suite, Recovery Rate automation | Deferred pending v2.1 Phases 12-17 results | 2026-09-24 |

## Session Continuity

Last session: 2026-10-03T06:57:25.372Z
Stopped at: Completed 15-10-PLAN.md
Resume file: None

Phase 12 closed 2026-09-30 with 15/15 UAT tests passing (12-UAT.md has full
per-test evidence). Summary of the day's work, across a long multi-session
arc from the September 29 handoff:

- **Root cause #1 (LATENCY-01, fixed in 12-01/02):** `control_loop_observation()`
  wasn't gated on queue-empty, resending a full observation every control
  tick — only 1/60 real actions in Phase 11's baseline episode.

- **Root cause #2 (fixed 2026-09-30, commits `dddf7dd`/`11267ca`):** FastFS
  reloaded its whole model from disk on every depth request (15-70s/call).
  Fixed with a resident Flask service (`vla_bridge/fastfs_server.py`) that
  loads the model once at startup; `vla_bridge/depth_recorder.py` moved
  depth submission off the control-loop thread. 14 new tests
  (`control/test_depth_pipeline.py`).

- **Root cause #3 (fixed 2026-09-30, commit `0ab0c96`):** even with the
  depth fix, a live retest still stalled 15+ min — the observation itself
  was a raw, uncompressed ~9MB pickle (vendored `lerobot`
  `RobotClient.send_observation()`) sent over gRPC's small flow-control
  window. Fixed with `vla_bridge.robot_client._downsize_for_transport()`,
  capping every frame at 640px (above the checkpoint's own 512x512 resize
  target, so no fidelity lost — confirmed via the installed config, not
  assumed). 3 new tests (`control/test_robot_client.py`).

- **Live verification:** a real 200-step episode
  (`control/outputs/vla_episode_004`) completed cleanly end to end,
  including a genuine watchdog fire-and-recover cycle (stale detected at
  step 106, recovered by step 111 — ~1s, vs. the pre-fix 6+ minute
  baseline) and clean depth heatmaps confirming the FastFS output is
  coherent (`control/render_depth_heatmaps.py`, new).

- Test 5 (notebook doc staleness) and Test 3 (watchdog live-recovery, found
  retroactively in historical `episode.jsonl` files rather than a fresh
  forced stall) were both re-verified and closed the same session.

All work is committed and pushed to `origin/master`/`origin/main`. Phases
8-9 remain paused. Keep safety-cap re-tightening last in v2.1 per the
existing resequencing decision.

Next: capture and commit the home pose (`.venv/bin/python start_pose.py` in control/), then `/gsd-verify-work 14` (Phase 14 hardware UAT, resumes at Test 1) and `/gsd-verify-work 13` (open Phase 13 camera UAT; can share the hardware session; see `13-UAT.md`). Phase 16 follows. Phase 13 stays incomplete until its UAT passes.
