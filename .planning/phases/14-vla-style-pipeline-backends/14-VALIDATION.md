---
phase: 14
slug: vla-style-pipeline-backends
status: approved
nyquist_compliant: true
wave_0_complete: true
created: 2026-10-01
---

# Phase 14 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (confirmed: 208 tests collected cleanly from `control/` this session) |
| **Config file** | none — no `pytest.ini`/`pyproject.toml` test config at repo root or in `control/`; `control/conftest.py` supplies shared fixtures (`MockRobot`, etc.) |
| **Quick run command** | `.venv/bin/python -m pytest control/test_<new_backend>.py -x` (run from `control/`) |
| **Full suite command** | `.venv/bin/python -m pytest control/ -x` |
| **Estimated runtime** | ~60-90 seconds for full suite (208+ tests) |

---

## Sampling Rate

- **After every task commit:** Run `.venv/bin/python -m pytest control/test_<touched_file>.py -x`
- **After every plan wave:** Run `.venv/bin/python -m pytest control/ -x` (full 208+ suite)
- **Before `/gsd-verify-work`:** Full suite must be green, plus the 12 individual live-hardware checkpoints per D-05's sequencing
- **Max feedback latency:** ~90 seconds (full suite runtime)

---

## Per-Task Verification Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? | Status |
|--------|----------|-----------|-------------------|---------------|--------|
| PIPE-01 | Existing `ActionSource` Protocol/control loop accepts any new backend without modification | unit (structural conformance) | `pytest control/test_run_vla_episode.py -x` | ✅ exists (extend, don't replace) | ⬜ pending |
| VLAB-03/05/12 (Tier A: π0/π0.5, ACT, Diffusion Policy) | `connect_bridge()`-equivalent produces a valid `RobotClientConfig` and reuses `pop_validated_action()` unmodified | unit, mocked client (mirrors `test_robot_client.py`'s existing `MockRobot`-style doubles) | `pytest control/test_robot_client.py -x` | ✅ exists (extend with new backend's `policy_type`) | ⬜ pending |
| VLAB-04/10 (Tier B: GR00T N1.7, Wall-X) | Dispatch verification against installed `lerobot` source | manual-only (one-time source read, not automatable) | n/a — documented finding, not a test | ❌ Wave 0 — no automated way to assert against a vendored library's internal dispatch table; treat as a plan task, not a test | ⬜ pending |
| VLAB-06/07/08/09/11 (Tier C: OpenVLA, OpenVLA-OFT, Octo, RDT-1B, GR00T N1.5) | Resident HTTP adapter loads once, responds to `/health`, returns a well-shaped action for a given observation | integration (mocked HTTP, mirrors `test_depth_pipeline.py`'s existing FastFS-adapter test pattern) | new `control/test_backend_adapters.py::test_<backend>_adapter_health` / `::test_<backend>_adapter_returns_valid_action_shape` | ❌ Wave 0 | ⬜ pending |
| VLAB-01/02 (Gemini Robotics / Robotics 2, via ER 2 + conversion) | `GeminiActionSource` auth/request/response shape correctly parsed into a `JOINT_ORDER`-compatible dict via the ER-to-joint-space conversion layer | unit, mocked HTTP response | new `control/test_gemini_action_source.py` | ❌ Wave 0 | ⬜ pending |
| All 12 VLAB | At least one live observed action per backend | manual-only (live hardware, per D-05's "independently live-hardware-verifiable" requirement) | n/a | n/a — explicitly a human-verified checkpoint per backend, not automatable | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

**Updated 2026-10-01 to match the finalized plans (14-01..14-15) — each plan embeds its own `<automated>` pytest command inline, so these are already covered per-task, not a separate pre-pass.**

- [x] `control/test_backend_openvla_oft.py`, `control/test_backend_openvla.py`, `control/test_backend_octo.py`, `control/test_backend_rdt1b.py`, `control/test_backend_groot_n15.py` — covers Tier C resident-adapter tests (14-03, 14-09, 14-10, 14-11, 14-12), modeled on `control/test_depth_pipeline.py`'s FastFS-adapter pattern
- [x] `control/test_backend_pi0.py`, `control/test_backend_act.py`, `control/test_backend_diffusion_policy.py`, `control/test_backend_groot_n17.py`, `control/test_backend_wallx.py` — covers Tier A/B gRPC-bridge tests (14-04..14-08)
- [x] `control/test_base_gemini.py`, `control/test_backend_gemini_robotics.py`, `control/test_backend_gemini_robotics_2.py` — covers VLAB-01/02 (14-13, 14-14, 14-15), mocked `google-genai` HTTP responses
- [x] `control/test_eef_to_joint.py` — IK conversion module tests (14-03), reused by EEF-delta backends
- [x] `control/test_backend_registry.py` — parametrized test asserting every `BACKEND_REGISTRY` entry satisfies the `ActionSource` Protocol (14-01/14-02), preventing a future backend from silently breaking PIPE-01's structural guarantee
- [x] Framework install: none — pytest already present and in active use; no new test framework needed

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Live-hardware observed action per backend | VLAB-01..12 | Requires the real SO-ARM101 and a live Colab/cloud connection; cannot be simulated or mocked per D-05's independently-live-hardware-verifiable requirement | Run `run_vla_episode.py --backend <name>` against the real robot, confirm at least one genuine action executes and is logged via `io_logger.py`, per backend, in D-05's familiarity/risk-ascending order |
| GR00T N1.7 / Wall-X LeRobot async dispatch (Tier B) | VLAB-04, VLAB-10 | Must be confirmed against the *installed* `lerobot` version's source, not assumed from docs (open question #2 in `14-RESEARCH.md`) | `grep` the installed `lerobot.async_inference.PolicyServer` dispatch table for `"groot"`/`"wall_oss"` entries before building each backend's plan tasks |
| Tier C GitHub-source pin verification | VLAB-06/07/08/09/11 | Supply-chain risk: `octo-models`/`isaac-gr00t`/etc. are GitHub-only installs (not on PyPI), so a human must confirm the pinned commit/tag comes from the official org before first Colab run | Human review of the pinned commit hash/tag against the official GitHub org for each Tier C backend before that backend's first Colab execution |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies — confirmed by gsd-plan-checker (2026-10-01): every plan task embeds its own inline pytest verify command, no `MISSING` forward-references
- [x] Sampling continuity: no 3 consecutive tasks without automated verify — each plan has exactly 2 tasks, each with its own `<automated>` verify
- [x] Wave 0 covers all MISSING references — none found
- [x] No watch-mode flags
- [x] Feedback latency < 90s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-10-01 (gsd-plan-checker VERIFICATION PASSED, 1 non-blocking doc-drift warning resolved by this update)
