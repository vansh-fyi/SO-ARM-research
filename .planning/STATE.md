---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
current_phase: 3
current_phase_name: VLA Inference Loop
status: executing
stopped_at: Phase 3 context gathered
last_updated: "2026-07-18T17:59:44.627Z"
last_activity: 2026-07-18
last_activity_desc: Phase 3 execution started
progress:
  total_phases: 6
  completed_phases: 2
  total_plans: 12
  completed_plans: 9
  percent: 33
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-07-07)

**Core value:** A researcher types a task prompt and watches SOARM execute it in a LIBERO simulation — the loop from language to embodied action.
**Current focus:** Phase 3 — VLA Inference Loop

## Current Position

Phase: 3 (VLA Inference Loop) — EXECUTING
Plan: 1 of 3
Status: Executing Phase 3
Last activity: 2026-07-18 — Phase 3 execution started

Progress: [██░░░░░░░░] 17% (Phase 1 of 6 complete)

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

### Pending Todos

None yet.

### Blockers/Concerns

- Phase 2 research flag: SOARM robosuite 1.4 ManipulatorModel integration is novel — budget 1-2 days of iterative MJCF editing; reference TechLabs Aachen SO100+robosuite as prior art
- Phase 5 research flag: Spatial VLA input representation (multi-camera RGB vs RGB+depth vs auxiliary 3D annotations) is an open question — study SpatialVLA, VEGA, cVLA before committing

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

## Deferred Items

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| Physical Robot | SOARM hardware transfer (PHYS-01 to PHYS-03) | v2 deferred | Init |
| Advanced Spatial | Ego3D encoding, point cloud input (ADV-01 to ADV-03) | v2 deferred | Init |
| Interactive UI | Real-time REPL, web dashboard (INT-01, INT-02) | v2 deferred | Init |

## Session Continuity

Last session: 2026-08-02
Stopped at: Phase 3 (03-03) Task 4 — the reconnect-retry fix (260802-gqt) for episode 1's keepalive-timeout failure is COMMITTED and locally verified (10/10 tests); Drive zip re-synced. Awaiting the user's next Colab run of Notebook B (full top-to-bottom; server startup and episode 0 already proven on the 2026-07-26 L4 run). If the same stall recurs, predict() now rides through it, printing '[Pi0Backend] connection lost ... reconnecting'; if the server actually dies, it fails fast pointing at /content/serve_policy.log. On a clean 2-episode run: sign off VLA-04, finalize 03-03-SUMMARY.md, verify phase, close Phase 3. Fallback if it fails differently: episode 0's completed run already substantively satisfies D-10's 1-2-episode smoke test — accepting it is a legitimate sign-off option. Mystery-sync on libero/notebooks/03b (6 sightings, 5 stashes) still unexplained.
Resume file: .planning/HANDOFF.json
