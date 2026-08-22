---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
current_phase: 6
current_phase_name: Fine-Tuning & Evaluation
status: executing
stopped_at: Phase 6 context gathered
last_updated: "2026-08-20T17:44:34.903Z"
last_activity: 2026-08-20
last_activity_desc: Phase 6 execution started
progress:
  total_phases: 6
  completed_phases: 5
  total_plans: 24
  completed_plans: 21
  percent: 83
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-07-07)

**Core value:** A researcher types a task prompt and watches SOARM execute it in a LIBERO simulation — the loop from language to embodied action.
**Current focus:** Phase 6 — Fine-Tuning & Evaluation

## Current Position

Phase: 6 (Fine-Tuning & Evaluation) — EXECUTING
Plan: 1 of 3
Status: Executing Phase 6
Last activity: 2026-08-20 — Phase 6 execution started

Progress: [█████████████████░░░] 83% (Phases 1-5 of 6 complete)

## Performance Metrics

**Velocity:**

- Total plans completed: 14
- Average duration: -
- Total execution time: 0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 02 | 5 | - | - |
| 04 | 5 | - | - |
| 05 | 4 | - | - |

**Recent Trend:**

- Last 5 plans: none
- Trend: -

*Updated after each plan completion*
| Phase 02 P02 | 10min | 3 tasks | 6 files |
| Phase 02 P03 | 10min | 2 tasks | 0 files |
| Phase 04 P03 | 35min | 3 tasks | 3 files |
| Phase 04 P04 | 20min | 2 tasks | 3 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- **04 (2026-08-03) GRIPPER UPGRADE (LOCKED, D-07):** Stock SOARM gripper (~2-3cm jaw) can't grasp ANY LIBERO object (smallest=butter 4cm; ~90 grasp trials, 0 successes — jaw-size limit, faithful to real hardware). Adopted the open-source roboninecom SO-ARM100/101 parallel gripper, modeled into `SoarmGripper` at the FAITHFUL 84mm stroke (jaws 0..0.042; committed ad0b0d0). 84mm is the practical ceiling for this arm (no wider ready-made gripper; arm payload ~500g) so the 11cm bowl is out of class. Arm unchanged; class name + -1=open/+1=closed contract preserved.
- **04 (2026-08-03) TASK RETARGETING (LOCKED, D-08):** Dropped the 3 frozen bowl→plate tasks (bowl too big for 84mm jaw AND plate place-target ~0.5m beyond the arm's ~0.45m reach). Retarget to a sub-84mm object (can/box ~4-8cm) in a pick-place task with BOTH object-init and place-target within ~0.45m reach (author/modify a BDDL). Authoring+validation is spend-blocked; happens on resume.
- Research: OpenVLA-OFT chosen as primary VLA (97.1% LIBERO avg, T4-compatible at 4-bit)
- Research: SOARM MJCF to be derived from `so101_new_calib.xml`, not built from scratch
- Research: Two separate Colab kernel groups needed (transformers version conflict between LIBERO training and VLA inference)
- Research: Demo replay must be state-based (not action replay) — LIBERO issue #16
- 01-02: numba 0.59.x chosen for numpy 1.x compat — Colab system numba compiled for numpy 2.x causes ABI crash after our numpy<2 pin
- 01-02: ENV-01 pip check filtered to OUR_PACKAGES — Colab system conflicts (jax/cupy/opencv needing numpy>=2) are pre-existing noise
- 01-02: torch version comparison strips build tag — "2.2.0+cu121".split('+')[0] == "2.2.0"
- 01-03: prismatic package must come from moojink/openvla-oft repo (--no-deps) — TRI-ML prismatic-vlms lacks prismatic.training; nothing usable on PyPI
- 01-03 (resolves RESEARCH A1, CONFIRMED on Colab): OFT checkpoint norm_stats = OXE pretraining datasets only; LIBERO stats must be overlaid from dataset_statistics.json (hf_hub_download) after from_pretrained; actual key is "libero_spatial_no_noops" — Phase 3 inference loop MUST replicate this
- 01-03 (CONFIRMED on Colab): OFT predict_action returns (actions, hidden_states) tuple with action chunk shape (8, 7) float64 — Phase 3 must unpack and consume 8-step chunks, not single steps
- 01-03: prismatic import chain needs wandb (metrics.py) — included in Step 5 since openvla-oft installs with --no-deps
- 01-close: dlimp MUST install via kvablack clone + pip --no-deps — BOTH dlimp repos pin tensorflow==2.15.0 (no cp312 wheel); any deps-resolving install fails on Colab py3.12
- 01-close: Block A installs drag protobuf below Colab's tensorflow_metadata gencode — Step 5b restores with pip install -U protobuf (must stay the cell's last pip op)
- 01-close: openvla-oft --no-deps means its eager-chain deps must be filled explicitly — Step 5b installs tensorflow_graphics==2021.12.3 (--no-deps: OpenEXR/tf-addons unbuildable), draccus==0.8.0, jsonlines, wandb, diffusers==0.30.3 (enumerated from prismatic/ source)
- 01-close: Colab "Restart runtime" keeps the VM disk — only "Disconnect and delete runtime" is a clean-slate test; manual debug installs persist across restarts and create false "it works" signals
- 01-close: HF token read from Drive file (MyDrive/SoARM-Research/.hf_token) via Block B bootstrap cell — Colab secrets vault (userdata.get) blocks indefinitely from VS Code-attached sessions
- 01-close: **REQUIRED READING for phases 2-6 before touching the Colab environment:** `.planning/phases/01-colab-environment-setup/01-DEBUG-HISTORY.md` — the environment contract (7 invariants), full ENV-03 failure timeline, and debugging meta-lessons (restart ≠ clean slate, --no-deps contract, source enumeration over whack-a-mole)
- [Phase ?]: 02-02: Gripper actuator keeps source servo values (kp 998.22, ctrlrange -0.17453..1.74533, forcerange ±3.35); kv stripped for MuJoCo 2.3.7
- [Phase ?]: 02-02: Harness (and any fork-importing script) must insert the repo ROOT on sys.path — LIBERO working tree carries uncommitted LIBERO.-prefix absolute imports
- [Phase ?]: 02-02: reset (0.024 N) and render checks already GREEN with untuned starting values — 02-03 tunes from a working baseline
- [Phase 02]: 02-03: Generic OSC_POSE config suffices for SOARM soak stability — no custom controller_configs kwarg for 02-05 notebook or Phase 3
- [Phase 02]: 02-03: All 02-02 starting values final (damping 0.6x5, base offset -0.38/0/0.90, init_qpos zeros(5), gripper speed 0.10 / jaw 0.8); A4 primitive-collision fallback not needed; reset peak 0.024 N
- [Phase 03]: 03-02: OFT-on-SOARM 0% task success is the accepted zero-shot cross-embodiment baseline, not a defect — Phase 6 fine-tuning closes this gap
- [Phase 03]: 03-03: π0 served via `serve_policy.py --env=LIBERO` (pi05_libero) — original D-07 choice (pi0_fast_libero) was deprecated upstream between plan-time and first live run; π0-on-SOARM also lands at 0% success, same accepted baseline as OFT
- [Phase 03]: 03-03: gsutil (Cloud SDK) uses its own bundled, isolated Python — `pip install` in a Colab kernel never reaches it; CLOUDSDK_PYTHON_SITEPACKAGES=1 is required whenever gsutil needs a kernel-installed package (gsutil#1429)
- [Phase 03]: 03-03: never route large (multi-GB) gsutil/GCS downloads through a Google Drive FUSE mount on Colab — local disk only, copy to Drive afterward if persistence is needed
- [Phase 03]: 03-03: openpi-client's WebsocketClientPolicy hardcodes websockets' 20s/20s keepalive with no tuning knobs — any Colab-side stall (e.g. a JAX recompile) longer than that needs client-side reconnect-retry, not a server-side fix
- [Phase 03]: 03-03: defensive optional-import guards (`except ImportError:`) must catch the real class of failure a broken-but-present dependency can raise (e.g. numpy ABI ValueError), not just literal absence — `except Exception` for cross-kernel package guards
- [Phase ?]: 04 (2026-08-03) SECOND EMBODIMENT BLOCKER: D-07/D-08 task retargeting (cream_cheese_1/akita_black_bowl_1, libero_goal) executed+committed (5675163) - fixes the jaw-width problem, but arm vertical reach bottoms out ~1-3cm above the object's top surface at all reachable radii tested (0.29-0.47m), independent of GRASP_Z_OFFSET/KP_POS/controller-type. Formal validation: 0/20 successes. DATA-01 dataset still not produced; needs a fresh Rule-4 decision (pedestal/riser, elevated-object task, state-injection synthesis, or descope). See 04-02-SUMMARY.md Resolution attempt section.
- [Phase 04]: 04-03: verify_states_only (cheap, 100% demos) + verify_full_obs_regeneration (sampled ~10%, min 5, always >=1/demo) built directly on ControlEnv.set_state/set_init_state -- proven round-trip against 04-02's real 120-demo dataset (17457/17457 states, 1746/1746 sampled obs regenerations). DATA-02 complete.
- [Phase 04]: 04-04: normalization.py computes OpenVLA q01/q99/mean/std/min/max stats purely in numpy, schema-matched to oft_backend.py's existing dataset_statistics.json overlay; ran against the real 04-02 dataset (17457 transitions, 120 trajectories) producing dataset_statistics.json. DATA-03 complete.
- [Phase 04]: 04-04: corrected plan's stale 6-D proprio assumption (5 joints + 1 gripper DOF) to the real 7-D shape (5 joints + 2-DOF gripper) matching the D-07 84mm parallel-gripper upgrade's actual gripper_states shape -- same class of stale pre-upgrade assumption 04-03 already fixed.
- [Phase 04]: 04-05: teleop.py (keyboard-only, D-02) + test_schema_matches_across_sources implemented and committed (d2a607d, e165ca7); Task 3's stale how-to-verify (old table_center bowl/plate task) corrected to the real put_the_cream_cheese_in_the_bowl task (5e5d8ef). Task 3 itself (real human-operated teleop session) is a blocking human-action checkpoint -- awaiting the user to run the corrected instructions.

### Pending Todos

None yet.

### Blockers/Concerns

- Phase 2 research flag: SOARM robosuite 1.4 ManipulatorModel integration is novel — budget 1-2 days of iterative MJCF editing; reference TechLabs Aachen SO100+robosuite as prior art
- Phase 5 research flag: Spatial VLA input representation (multi-camera RGB vs RGB+depth vs auxiliary 3D annotations) is an open question — study SpatialVLA, VEGA, cVLA before committing
- Colab delivery mechanism (260802-ijb/itm): now GitHub-clone + getpass() token prompt, replacing the old Drive-zip delivery. **Unproven on a live Colab run as of 2026-08-02** — the prior notebook edits this session were pushed but not yet re-tested end-to-end. **User's explicit fallback: if this doesn't work, revert to the Drive-zip method** (drive.mount() + unzip SoARM-Research-colab.zip, which was working reliably through all of Phase 3). Don't treat this fallback as a last resort to avoid — if the GitHub/getpass approach hits friction on the next real run, ask the user whether to debug it further or just revert, rather than assuming it must be pushed through.
- 04-02 BLOCKER (RESOLVED in direction, 2026-08-03): SOARM stock gripper couldn't grasp any LIBERO object (jaw ~2-3cm << all objects; ~90 grasp trials 0 successes). RESOLUTION (locked D-07/D-08): upgraded to roboninecom 84mm parallel gripper (faithful, committed ad0b0d0) + retarget to a sub-84mm in-reach object. A second embodiment limit surfaced: the arm's ~0.45m reach can't reach the bowl→plate place-target (~0.5m) — handled by choosing/authoring an in-reach task. See 04-CONTEXT.md ⚠ AMENDMENT + Session Continuity resume sequence.
- 04 embodiment note for Phase 5/6: SO-ARM101 is a small ~500g-payload desktop arm; task layouts must keep objects AND targets within ~0.45m reach, objects <=84mm to be graspable, AND avoid placing objects on the base's forward centerline (y~0 close to the base) — the arm's own forearm sweeps through that corridor and will collide with anything sitting there. This constrains Phase 5 spatial-task authoring too.
- 04-02 RESOLVED (2026-08-03, commit 6ecd160): a "vertical reach / torque-saturation" finding was misdiagnosed earlier the same day (see 04-02-SUMMARY.md history) — user pushback prompted a re-investigation (direct actuator-torque telemetry + MuJoCo contact inspection) that found the real cause was akita_black_bowl_1 sitting in the arm's forward sweep corridor, causing a genuine collision, not a hardware limit. Fixed by repositioning both objects + tightening collector.py's Z_TOL (a separate real bug). DATA-01 dataset now exists: 120 demos at LIBERO/libero/datasets/soarm_spatial/put_the_cream_cheese_in_the_bowl_demo.hdf5. **Lesson for future embodiment-limit claims: always check actuator_force/qfrc_bias against ctrlrange AND sim.data.contact before concluding a hardware/torque ceiling — a "stuck" eef is very often a collision, not saturation.**
- 04-05 Task 3 AWAITING HUMAN ACTION: a real person must physically drive SOARM via keyboard (collect_teleop against LIBERO/libero/libero/bddl_files/libero_goal/put_the_cream_cheese_in_the_bowl.bddl) and report 'demos written:' + verify_states_only result. See 04-05-PLAN.md Task 3 how-to-verify for exact commands. Plan is NOT complete until this checkpoint resolves.
- 05 PRE-EXISTING TEST DEBT (found 2026-08-10, still open — not a Phase 5 regression, confirmed via `git log` that neither test file nor its underlying source was touched by any 05-* commit): (1) `test_hdf5_writer.py::test_schema_and_obs_key_naming` asserts `gripper_states.shape[1] == 1` ("SOARM's 1-DOF gripper") — stale, never updated after the D-07 84mm 2-DOF parallel-gripper upgrade. Actual shape is 2, which is correct; the assertion needs updating. (2) `test_replay.py::test_verify_full_obs_regeneration_passes_on_04_02_output` fails with a pixel mismatch in `(demo_1, 0, agentview_rgb)` when regenerating rendered obs from the real 04-02 dataset — likely MuJoCo offscreen-render non-determinism, not investigated. Both belong to Phase 4's test suite, not Phase 5's requirements — candidate for a Phase 6 quick-task cleanup.
- 05 BUG FIXED (2026-08-10, commits `4b241ee`+`8451bca`): the 05-03 executor's fix commit (`d53016b`, "case-path staging miss") accidentally re-registered the ENTIRE `LIBERO/` tree (1168 files) as lowercase `libero/` in git's index — invisible on macOS (case-insensitive fs) but would have broken every hardcoded `LIBERO/...` path on Colab (Linux, case-sensitive) on next clone/pull, since the project's core execution environment is Colab. Fixed via a two-step `git mv` (through a temp name, since git can't case-rename directly on a case-insensitive fs) restoring canonical `LIBERO/` casing; re-ran the vla (12/12) and spatial-predicates (16/16) test suites post-fix, both green. **Lesson for future executors/sessions on this repo: macOS case-insensitivity can silently corrupt git's tracked path casing for an entire subtree via one `git add`/`git mv` with the wrong case — if a diff shows an unexpected top-level directory-name case change (e.g. `libero/` vs `LIBERO/`), treat it as a blocking bug, not a cosmetic one, given this project's Colab (Linux) execution target.**
- 05 COMPLETE (2026-08-17): all 4 plans (05-01..05-04) merged, 13/13 must-haves verified. UAT surfaced a real blocker post-implementation — `OFTBackend.predict()` crashed on Colab GPU (`RuntimeError: split_with_sizes ... got split_sizes=[3, 3]`) because `__init__` never called `self.model.vision_backbone.set_num_images_in_input(2)`, plus an inverted `agentview`/`eye_in_hand` channel order. Fixed via gap-closure plan 05-04 (commits `f5b1e29`/`955dc79`/`96441cc`), retested live on Colab GPU: no crash, pixel_values genuinely differ per-view (max diff 3.34), and action output is measurably nonzero-sensitive to the second view (mean diff 0.000239) though weak — accepted as this checkpoint's first-action-chunk characteristic (instruction-dominated), not a residual bug. **Lesson: when the ONLY local-environment constraint is "no GPU," a UAT/Colab retest step is not optional busywork — this exact crash and its fix both required real GPU execution to surface and confirm; source-level review alone (grep-based assertions) passed the buggy version too.**
- 05 LESSON — private-repo sync friction recurred (2026-08-16/17): the outer `SoARM-Research` git repo (not just `LIBERO/`) drifted 76+ commits ahead of `origin/main` multiple times mid-session because work was committed locally but never pushed — Colab always clones/pulls from `origin/main`, so any local-only commit is invisible there until an explicit `git push origin master:main`. **This is not a one-time fix — check `git status -sb` for an "ahead" count before telling the user to `git pull` on Colab, every time**, not just once per session.

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 260726-epz | Fix Notebook B serve_policy.py: pi0_fast_libero deprecated on openpi main, switched to --env=LIBERO (pi05_libero) | 2026-07-26 | 4a7b8f2 | [260726-epz-fix-libero-notebooks-03b-pi0-inference-s](./quick/260726-epz-fix-libero-notebooks-03b-pi0-inference-s/) |
| 260726-gj6 | Fix Notebook B checkpoint download: pi05_libero (11.6GB) via gsutil to Drive-mounted OPENPI_DATA_HOME failed ("6 files/objects could not be transferred"); switched to local disk (/content/openpi_data) + compiled crcmod, dropping T-3-08 restart-persistence | 2026-07-26 | 54f7585 | [260726-gj6-fix-libero-notebooks-03b-pi0-inference-s](./quick/260726-gj6-fix-libero-notebooks-03b-pi0-inference-s/) |
| 260726-hbb | Fix Notebook B checkpoint download (2nd attempt — gj6's local-disk fix alone didn't resolve it, live re-run showed identical failure): real root cause is gsutil's isolated bundled-Python not seeing pip-installed crcmod (GoogleCloudPlatform/gsutil#1429); added CLOUDSDK_PYTHON_SITEPACKAGES=1 + check_hashes=if_fast_else_skip boto fallback + pre-flight verification cell | 2026-07-26 | 81035c9 | [260726-hbb-fix-libero-notebooks-03b-pi0-inference-s](./quick/260726-hbb-fix-libero-notebooks-03b-pi0-inference-s/) |
| 260726-hw8 | Fix libero/libero/libero/vla/__init__.py: server started successfully (crcmod fix confirmed working), but the VLA-04 proof cell crashed importing Pi0Backend — OFTBackend's `except ImportError` guard didn't catch a numpy binary-ABI ValueError raised in Notebook B's kernel (transformers/torch import chain), taking down the whole package import; broadened both guards to `except Exception` | 2026-07-26 | 90d7e7e | [260726-hw8-fix-libero-libero-libero-vla-init-py-the](./quick/260726-hw8-fix-libero-libero-libero-vla-init-py-the/) |
| 260726-io0 | Add Block A2 to Notebook B: import-guard fix confirmed working, but the VLA-04 cell then hit `ModuleNotFoundError: robosuite` — Notebook B's kernel was never given the LIBERO simulation stack (robosuite/mujoco/bddl/gym) at all, only openpi's own stack. Added a new Block A2 (EGL apt packages, robosuite==1.4.0/mujoco==3.3.2/bddl==1.0.1/gym==0.25.2 install, numpy==1.26.4 purge+pin+ABI-gate) mirroring Notebook A's proven pattern, plus a mandatory runtime restart | 2026-07-26 | 96abb60 | [260726-io0-add-a-block-a2-to-libero-notebooks-03b-p](./quick/260726-io0-add-a-block-a2-to-libero-notebooks-03b-p/) |
| 260726-p0o | Harden Notebook B: (1) serve_policy.py stdout=PIPE never drained → suspected cause of a 95-min silent hang mid-smoke-test; redirected to /content/serve_policy.log. (2) openpi-client editable-install .pth finder only processed at interpreter startup → pip show sees it but import fails; install cell now runs site.addsitedir + inline import verification (fix confirmed live by user). Executed inline (spend limit killed the planner subagent) | 2026-07-26 | 596bcd9 | [260726-p0o-harden-libero-notebooks-03b-pi0-inferenc](./quick/260726-p0o-harden-libero-notebooks-03b-pi0-inferenc/) |
| 260802-gqt | Reconnect-retry in Pi0Backend.predict: episode 1 of the L4 smoke test died with ConnectionClosedError 1011 (keepalive ping timeout — openpi-client hardcodes websockets' 20s ping defaults; suspected ep1 first-inference JAX recompile stall). predict() now backs off (30s/60s), probes server-port liveness (fail-fast → serve_policy.log pointer if dead), rebuilds the client, retries. 4 new regression tests, 10/10 pass. Executed inline | 2026-08-02 | 559d42e | [260802-gqt-add-reconnect-retry-to-pi0backend-predic](./quick/260802-gqt-add-reconnect-retry-to-pi0backend-predic/) |
| 260802-ijb | Stop depending on Google Drive for Colab code delivery (user request): flattened LIBERO/ (was its own nested git repo, blanket-gitignored, 765MB incl. 316MB own history) directly into main repo tracking; rewrote all 4 notebooks' delivery cells to clone/pull the private GitHub repo via a Colab-Secrets GITHUB_TOKEN instead of drive.mount()+unzip. Repo stays PRIVATE per explicit user decision. 1146 files/~452MB committed after a full pre-commit safety scan (no .env/__pycache__/egg-info, no files >50MB, no secret-pattern matches) | 2026-08-02 | b59f55d | [260802-ijb-stop-depending-on-google-drive-for-colab](./quick/260802-ijb-stop-depending-on-google-drive-for-colab/) |
| 260802-itm | Follow-up to ijb: user's VS Code Colab extension has no Secrets UI, so userdata.get() didn't work for them. User asked to hardcode the token in a cell; switched to getpass() instead (flagged the git-history-leak risk given the project's recurring mystery-sync issue on these files — user chose getpass() when given both options) — same one-paste-per-session UX, token never written to notebook source. Applied to all 4 notebooks' delivery cells + Notebook 01's HF-token cell | 2026-08-02 | fb35835 | [260802-itm-switch-notebook-github-hf-token-input-fr](./quick/260802-itm-switch-notebook-github-hf-token-input-fr/) |

## Deferred Items

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| Physical Robot | SOARM hardware transfer (PHYS-01 to PHYS-03) | v2 deferred | Init |
| Advanced Spatial | Ego3D encoding, point cloud input (ADV-01 to ADV-03) | v2 deferred | Init |
| Interactive UI | Real-time REPL, web dashboard (INT-01, INT-02) | v2 deferred | Init |

## Session Continuity

Last session: 2026-08-20T16:57:43.472Z
Stopped at: Phase 6 context gathered

Phase 4 (Dataset Collection) and Phase 5 (Spatial Awareness) are both complete —
their earlier resume-sequence notes below are historical, not active blockers.

NOTE: 2 pre-existing test failures from Phase 4 (gripper_states.shape assertion,
a replay-obs pixel-mismatch test) remain open — see Blockers/Concerns above,
candidate for a Phase 6 quick-task cleanup, not a Phase 6 blocker.

NOTE (repo sync): the outer repo has repeatedly drifted commits-ahead of
`origin/main` without being pushed — before telling the user to `git pull` on
Colab, always check `git status -sb` for an "ahead" count first.

NOTE: if any future embodiment-limit claim looks physically implausible for hardware known to work in the real world, verify actuator_force/qfrc_bias against ctrlrange AND sim.data.contact before accepting a "hardware limit" conclusion — this exact investigation had two prior misdiagnoses that direct telemetry immediately refuted.
Resume file: .planning/phases/06-fine-tuning-evaluation/06-CONTEXT.md
