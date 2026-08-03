---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
current_phase: 04
current_phase_name: Dataset Collection
status: executing
stopped_at: "Phase 4 STILL blocked: D-07/D-08 task retargeting executed (cream_cheese_1/akita_black_bowl_1, commit 5675163), but empirical validation found a NEW vertical-reach-depth blocker (0/20 successes). DATA-01 dataset not yet produced. Awaiting a fresh Rule-4 decision from the user (pedestal/riser, elevated-object task, state-injection synthesis, or descope) before Phase 4 collection can resume."
last_updated: "2026-08-03T11:06:39.902Z"
last_activity: 2026-08-03
last_activity_desc: 04-02 task retargeting executed (cream_cheese_1/akita_black_bowl_1, D-07/D-08) and committed; empirical validation found a NEW vertical-reach-depth blocker (0/20 successes) — DATA-01 dataset still not produced, fresh Rule-4 decision needed
progress:
  total_phases: 6
  completed_phases: 3
  total_plans: 17
  completed_plans: 14
  percent: 50
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-07-07)

**Core value:** A researcher types a task prompt and watches SOARM execute it in a LIBERO simulation — the loop from language to embodied action.
**Current focus:** Phase 04 — Dataset Collection

## Current Position

Phase: 04 (Dataset Collection) — EXECUTING (BLOCKED mid-Wave-2 on a SECOND embodiment finding)
Plan: 2 of 5 (04-01 done; 04-02 collector retargeted to cream_cheese_1/akita_black_bowl_1 per D-07/D-08 and committed [5675163], but the real COLLECTION still did not produce demos — a NEW vertical-reach-depth blocker was found empirically [0/20 validation]; 04-03/04/05 pending)
Status: Blocked — awaiting a fresh Rule-4 decision from the user (pedestal/riser to reduce required vertical excursion, an elevated-object task, state-injection demo synthesis, or DATA-01 descope). See 04-02-SUMMARY.md "Resolution attempt (2026-08-03)" for the full measurement trail and options.
Last activity: 2026-08-03 — task retargeting (D-07/D-08) executed and committed; empirical validation found the arm's vertical reach cannot reach table-resting-object height at any tested radius (0.29-0.47m), independent of the (already-fixed) jaw width

Progress: [█████░░░░░] 50% (Phases 1-3 of 6 complete)

## Performance Metrics

**Velocity:**

- Total plans completed: 5
- Average duration: -
- Total execution time: 0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 02 | 5 | - | - |

**Recent Trend:**

- Last 5 plans: none
- Trend: -

*Updated after each plan completion*
| Phase 02 P02 | 10min | 3 tasks | 6 files |
| Phase 02 P03 | 10min | 2 tasks | 0 files |

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

### Pending Todos

None yet.

### Blockers/Concerns

- Phase 2 research flag: SOARM robosuite 1.4 ManipulatorModel integration is novel — budget 1-2 days of iterative MJCF editing; reference TechLabs Aachen SO100+robosuite as prior art
- Phase 5 research flag: Spatial VLA input representation (multi-camera RGB vs RGB+depth vs auxiliary 3D annotations) is an open question — study SpatialVLA, VEGA, cVLA before committing
- Colab delivery mechanism (260802-ijb/itm): now GitHub-clone + getpass() token prompt, replacing the old Drive-zip delivery. **Unproven on a live Colab run as of 2026-08-02** — the prior notebook edits this session were pushed but not yet re-tested end-to-end. **User's explicit fallback: if this doesn't work, revert to the Drive-zip method** (drive.mount() + unzip SoARM-Research-colab.zip, which was working reliably through all of Phase 3). Don't treat this fallback as a last resort to avoid — if the GitHub/getpass approach hits friction on the next real run, ask the user whether to debug it further or just revert, rather than assuming it must be pushed through.
- 04-02 BLOCKER (RESOLVED in direction, 2026-08-03): SOARM stock gripper couldn't grasp any LIBERO object (jaw ~2-3cm << all objects; ~90 grasp trials 0 successes). RESOLUTION (locked D-07/D-08): upgraded to roboninecom 84mm parallel gripper (faithful, committed ad0b0d0) + retarget to a sub-84mm in-reach object. A second embodiment limit surfaced: the arm's ~0.45m reach can't reach the bowl→plate place-target (~0.5m) — handled by choosing/authoring an in-reach task. See 04-CONTEXT.md ⚠ AMENDMENT + Session Continuity resume sequence.
- 04 SPEND BLOCKER (active, 2026-08-03): the remaining sim work (author+validate the sub-84mm in-reach task, then run the 100+-demo collection, then Waves 3-4) hit the monthly spend limit — two executor/spike agents failed with "You've hit your monthly spend limit". Blocked until the limit is raised/reset. State is clean + fully resumable per Session Continuity.
- 04 embodiment note for Phase 5/6: SO-ARM101 is a small ~500g-payload desktop arm; task layouts must keep objects AND targets within ~0.45m reach, and objects <=84mm to be graspable. This constrains Phase 5 spatial-task authoring too.
- 04 BLOCKER (2026-08-03, NEW - supersedes the resolved jaw-width blocker): scripted collector retargeted to cream_cheese_1/akita_black_bowl_1 (D-07/D-08 executed, commit 5675163), but the SOARM arm's vertical reach cannot bring the gripper down to table-resting-object height (~0.90-0.92m world z) at any reachable radius tested (0.29-0.47m) while holding top-down orientation - a torque/kinematic limit (STS3215 +/-2.94Nm/joint), not a jaw-width problem. 0/20 in a formal validation batch (collect_task target_successes=8 max_attempts=20). DATA-01 (100+ demo dataset) still not producible via the current scripted FSM + this arm. See 04-02-SUMMARY.md 'Resolution attempt (2026-08-03)' for the full measurement trail + options for the user.

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

Last session: 2026-08-03T11:06:39.897Z
Stopped at: Phase 4 STILL blocked: D-07/D-08 task retargeting executed (cream_cheese_1/akita_black_bowl_1, commit 5675163), but empirical validation found a NEW vertical-reach-depth blocker (0/20 successes). DATA-01 dataset not yet produced. Awaiting a fresh Rule-4 decision from the user (pedestal/riser, elevated-object task, state-injection synthesis, or descope) before Phase 4 collection can resume.

RESUME SEQUENCE (next session — the D-07/D-08 retargeting work below is DONE;
what remains is a NEW Rule-4 decision, not more autonomous tuning):

  1. Read `.planning/phases/04-dataset-collection/04-02-SUMMARY.md`'s
     "Resolution attempt (2026-08-03)" section (full measurement trail: 3
     GRASP_Z_OFFSET candidates, a radius/bearing sweep, an OSC_POSE vs
     OSC_POSITION controller comparison, and a formal 20-attempt validation
     batch — all converge on the same conclusion, 0/20 successes) plus
     04-CONTEXT.md's ⚠ AMENDMENT (D-07/D-08, now executed) for background.
  2. DONE (commit `5675163`): gripper (D-07, faithful 84mm, commit `ad0b0d0`)
     and task retargeting (D-08: `libero_goal/put_the_cream_cheese_in_the_bowl.bddl`,
     pick `cream_cheese_1` / place `akita_black_bowl_1`) are both executed and
     committed. `collector.py`'s `TASK_BODY_MAP` wires the new body names
     through; `target_per_task` raised 40->120 to preserve the 100+ buffer
     now that there is only 1 task.
  3. NEW FINDING (not auto-fixable — Rule 4): the arm's vertical reach cannot
     bring the gripper down to table-resting-object height (~0.90-0.92m
     world z) at ANY reachable radius tested (0.29-0.47m) while holding a
     top-down orientation. This is independent of the (already-fixed) jaw
     width. A user decision is needed before more autonomous FSM tuning is
     attempted — see the SUMMARY's "Options for the user": (a) pedestal/riser
     to reduce required vertical excursion (reopens Phase 2's "no pedestal"
     D-04), (b) an elevated-object task/region instead of table-flush, (c)
     state-injection demo synthesis (bypass physical grasp), or (d) descope
     DATA-01's physical-grasp requirement.
  4. Once a direction is chosen: re-tune `collector.py`'s clearance constants
     for that new approach, re-validate on a small batch (foreground sim, NO
     detached/background MuJoCo processes — a prior agent orphaned a runaway
     proc during this exact investigation; also mind the macOS git
     case-collision trap: stage lowercase `libero/...` paths, verify with
     `git diff HEAD`), then run the real 100+-demo collection into
     `LIBERO/libero/datasets/soarm_spatial/`. Then proceed to Wave 3 (04-03
     replay, 04-04 normalization) and Wave 4 (04-05 teleop — human checkpoint).
  NOTE: run Phase 4 with worktrees DISABLED (sequential on main tree) — the dataset is gitignored and must persist across plans.
Resume file: .planning/phases/04-dataset-collection/04-02-SUMMARY.md
