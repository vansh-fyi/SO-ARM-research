---
gsd_state_version: 1.0
milestone: v2.1
milestone_name: MLLM Raw-Autonomy Benchmark
current_phase: 12
current_phase_name: Bridge Tick-Latency Fix
status: executing
stopped_at: Phase 12 UAT test 7 passed live on real hardware; test 5's major issue is the only open item left in 12-UAT.md
last_updated: "2026-09-30T12:15:00.000Z"
last_activity: 2026-09-30
last_activity_desc: Live-hardware retest passed Phase 12 UAT test 7 (resident FastFS depth server + observation-downsize fix); full 60/60-step episode, depth 6/6 complete/0 failed
progress:
  total_phases: 17
  completed_phases: 9
  total_plans: 44
  completed_plans: 40
  percent: 53
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-09-24)

**Core value:** A researcher types a task prompt and watches SOARM execute it in a LIBERO simulation — the loop from language to embodied action.
**Current focus:** Phase 12 — Bridge Tick-Latency Fix

## Current Position

Phase: 12 (Bridge Tick-Latency Fix) — EXECUTING
Plan: 1 of 2
Status: Executing Phase 12
Last activity: 2026-09-26 — Phase 12 execution resumed (wave continue)

Progress: [░░░░░░░░░░] 0%

## Performance Metrics

**Velocity:**

- Total plans completed: 24
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

**Recent Trend:**

- Last 5 plans: none
- Trend: -

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

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

### Pending Todos

None yet for v2.1 — begin by planning Phase 12 (`/gsd-plan-phase 12`).

### Blockers/Concerns

- **v2.1 Phase 12 is a hard prerequisite for the rest of the milestone (13 → 14/15 → 16 → 17)** — no new live-hardware benchmark work should start until the tick-latency root cause (only 1/60 real actions in Phase 11's episode) is fixed and measured.
- 04 embodiment note (still binding for any new v2.1 task/object work): SO-ARM101 is a small ~500g-payload desktop arm; objects ≤84mm graspable, objects AND targets within ~0.45m reach, avoid the base's forward centerline collision corridor.
- 05 PRE-EXISTING TEST DEBT (unrelated to v2.1, still open): `test_replay.py::test_verify_full_obs_regeneration_passes_on_04_02_output` pixel-mismatch failure, not investigated.
- 10 RISK — RESOLVED (2026-09-24): [huggingface/lerobot#2210](https://github.com/huggingface/lerobot/issues/2210) did NOT reproduce in Phase 11's live episode.
- 11/v2.1 OPEN (root cause diagnosed, fix scoped to Phase 12, not yet implemented): Phase 11's live episode yielded only 1/60 (1.7%) real executed VLA actions because `control_loop_observation()` resends a full camera observation to Colab on every control tick instead of gating on queue-empty. See `control/vla_bridge/FINDINGS.md` §5 and `.planning/research/SUMMARY.md` for the full diagnosis and fix direction (wire up `lerobot`'s existing `_ready_to_send_observation()` gate).
- v2.1 camera bug (root cause diagnosed, fix scoped to Phase 13, not yet implemented): `camera_overhead` recordings in both `vla_episode_001` and the Phase 11 go/no-go episode are confirmed to be the laptop webcam, not the robot workspace — recording path never got the name-based camera-resolution fix already applied to the inference-input path.

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

## Deferred Items

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| Physical Robot | SOARM hardware transfer (PHYS-01 to PHYS-03) | v2 deferred | Init |
| Advanced Spatial | Ego3D encoding, point cloud input (ADV-01 to ADV-03) | v2 deferred | Init |
| Interactive UI | Real-time REPL, web dashboard (INT-01, INT-02) | v2 deferred | Init |
| v1.1 Benchmark Work | Checkpoint Benchmark Suite + Re-Fine-Tuning (Phases 8-9) | Paused for v2.0 | 2026-09-15 |
| v2.1 Future Requirements | Provider-agnostic router beyond 3 backends, full 4-task suite, Recovery Rate automation | Deferred pending v2.1 Phases 12-17 results | 2026-09-24 |

## Session Continuity

Last session: 2026-09-30T02:40:48Z
Stopped at: Session resumed at Phase 12 UAT test 7 of 15; awaiting next-action selection.
Resume file: .planning/phases/12-bridge-tick-latency-fix/.continue-here.md
Structured handoff: .planning/HANDOFF.json (retained until routing into resumed work).

All six Phase 12 plans have summaries, but live UAT is incomplete. The older
Current Position and roadmap completion/progress text above do not reflect the
September 29 handoff. Test 7 remains pending: depth recording must work together
with smooth motion before it can pass.

Confirmed from the current notebook: PolicyServer uses subprocess isolation;
FastFS uses its own virtualenv and interpreter. Preserve these crash fixes.
The depth handler still launches run_demo.py per request, and the local control
loop waits synchronously for depth. The previous live session reported 15-70s
calls and motion freezes. Next recommended work: scope a persistent FastFS
server that loads its model once, with appropriate regression coverage and
live depth-plus-motion verification. User confirmed on 2026-09-30 that the RGB-only retry was NOT run and is not
the requested scope. Depth integration and reliable episode recording are required;
do not request an RGB-only retry as acceptance work.

Handoff divergence: policy_server.ipynb is now committed, not modified as the
handoff reported. Five unrelated untracked entries remain untouched. No
interrupted agents, outstanding async-job manifests, incomplete phase plans,
or pending todo files were found.

Phases 8-9 remain paused. Keep safety-cap re-tightening last in v2.1, consistent
with the explicit resequencing decision; the roadmap progress table has stale
phase labels that need reconciliation before advancing.

NOTE (repo sync): local master is 84 commits ahead of origin/main at resume.
Check synchronization before instructing a Colab git pull.

Context review (2026-09-30): see `.planning/notes/2026-09-30-context-review.md`
for reconciled history, active runtime/data contracts, recording gaps, and the
117 passing focused control tests at that point. This was exploration only; no
implementation or live hardware/Colab verification was performed in that pass.
The user correction above supersedes HANDOFF.json's old RGB-only retry question.

Persistent-depth-server implementation (2026-09-30, after the context review,
continued across two model sessions): the architecture fix scoped in the
September 29 handoff's `remaining_tasks`/`new-architecture-item` is now built
and passing locally, uncommitted:
- `control/vla_bridge/fastfs_server.py` (new): resident Flask service, loads
  the FastFS model once at startup, warms it, then serves `/health` and
  `/depth` in-process (no more per-request `subprocess.run()` to
  `run_demo.py`). Single-flight via a non-blocking lock (429 on overlap, not
  queuing). Embedded into `policy_server.ipynb`'s `resident-fastfs-source`
  cell verbatim via `scripts/sync_policy_notebook.py` (new), which also
  regenerates `policy_server_launch.md` from the notebook so the two can't
  drift again.
- `control/vla_bridge/depth_recorder.py` (new): submits depth requests off
  the main control-loop thread and drains the latest result; motion no
  longer blocks on FastFS latency.
- `control/run_vla_episode.py`, `depth_camera.py`, `stereo_camera.py`,
  `io_logger.py`, `robot_client.py`: wired to the above; notebook Step 8's
  startup cell now polls `/health` with a 600s warmup deadline before
  reporting ready.
- `control/test_depth_pipeline.py` (new, 14 tests): covers resident-model
  reuse, request-identity echo, busy/429 rejection, malformed/failed
  responses, real HTTP client-server roundtrip with pixel-aligned RGB
  recording, motion continuing while depth is in flight, shutdown/drain, and
  a guard test that the notebook's embedded server source and generated
  launch guide stay in sync with `fastfs_server.py`.
- Full suite: 167/167 pass locally (`control/.venv`). Added `flask==3.1.3` to
  `control/requirements.txt` — it's only a local test double for the resident
  service; Colab installs its own copy into the isolated `fastfs_venv` per
  Step 7, so this pin does not change what ships to Colab.
- This was committed and pushed to origin/master and origin/main same day
  (commits `dddf7dd`, `11267ca`).

Live verification (2026-09-30, after the above): first live retest on real
Colab + hardware stalled 15+ min with the robot stationary and 0% Colab GPU
utilization. Root-caused via the Colab `policy_server.log` timestamps (not
guessed): the observation *did* arrive and process fast (2.97s cold,
0.52s warm) — the delay was between observations, not inference. The
client sends the full observation as a raw, uncompressed pickle
(`lerobot.async_inference.robot_client.RobotClient.send_observation()`,
vendored, not this project's code) — ~9MB/observation (1920x1080 wrist +
two 800x600 stereo halves) — over gRPC's small per-stream flow-control
window through a real ngrok tunnel with `networkQuality`-confirmed
bufferbloat on one tested network path. Fixed client-side (the only seam
this project owns, since `send_observation()` itself is vendored): added
`vla_bridge.robot_client._downsize_for_transport()`, applied inside the
existing `get_observation_with_stereo_split()` monkeypatch, capping every
observation frame's long edge at 640px before pickling. 640 was chosen with
margin above `victorvanhalst/smolvla_so101_cube`'s own
`SmolVLAConfig.resize_imgs_with_padding=(512,512)` (confirmed by reading the
installed checkpoint's config) — the model pads/resizes every image to
512x512 itself regardless of input size, so nothing sent above that
resolution is fidelity the model actually uses; JPEG compression was
considered and rejected (round-tripping through JPEG and decoding back to
the same-shaped array would not reduce the wire payload at all, since the
vendored `send_observation()` pickles whatever array `get_observation()`
returns). 3 new tests added to `control/test_robot_client.py` (170/170
total passing); one pre-existing test that used string doubles for
camera2/camera3 was updated to use real arrays.

**Live retest result: Phase 12 UAT Test 7 now passes.** Full 60/60-step
episode (`control/outputs/vla_episode_20260930_121123`,
`max_steps_reached`), depth `{submitted: 6, complete: 6, failed: 0,
skipped_busy: 0}`, whole episode ~74s wall-clock (previously: a single
observation alone took 2-4 minutes; a full episode was not completable).
Motion has 3 brief (2.6-4.2s) pauses at the expected chunk-refetch points
(`actions_per_chunk=50`, `chunk_size_threshold=0.5`, refetch every ~25
steps) — not watchdog/staleness events, not worse near the end than the
start, user confirmed acceptable as-is. See `12-UAT.md` Test 7 for full
detail. **Not yet committed** — this fix (`robot_client.py` +
`test_robot_client.py` changes) is uncommitted in the working tree as of
this note.

**Remaining open item:** `12-UAT.md`'s only other non-automated test, Test
5 (major severity: notebook's stale AR0144 2560x720/1280x1280 camera
mapping doc, a misleading "port 8080" comment, duplicate tunnel cells) —
unclear whether it survived the 2026-09-30 resident-depth-server notebook
rewrite. Needs a fresh read of the current notebook before it can be marked
pass. Nothing else in Phase 12's UAT is open.
