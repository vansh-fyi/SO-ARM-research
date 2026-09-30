---
phase: 13
slug: camera-device-resolution-unification
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-09-30
---

# Phase 13 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (confirmed: `control/.venv/bin/python -m pytest --collect-only` returns 170 tests across `control/test_*.py`) |
| **Config file** | none — no `pytest.ini`/`pyproject.toml`/`setup.cfg` with `[tool:pytest]` found in `control/`; defaults + `test_*.py` naming discovery |
| **Quick run command** | `cd control && .venv/bin/python -m pytest -q test_detect_devices.py test_run_vla_episode.py test_robot_client.py test_io_logger.py test_record_episode.py` |
| **Full suite command** | `cd control && .venv/bin/python -m pytest -q` |
| **Estimated runtime** | ~30-60 seconds (170+ tests, no live hardware) |

---

## Sampling Rate

- **After every task commit:** Run `cd control && .venv/bin/python -m pytest -q test_detect_devices.py test_run_vla_episode.py test_robot_client.py test_io_logger.py test_record_episode.py`
- **After every plan wave:** Run `cd control && .venv/bin/python -m pytest -q` (full suite)
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** ~60 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 13-XX-XX | TBD | TBD | CAMFIX-01 | T-13-01 | Shared name-based helper resolves both wrist and overhead cameras; no raw-index bypass | unit | `pytest test_run_vla_episode.py test_robot_client.py test_record_episode.py -k resolve` | ❌ W0 (helper doesn't exist yet) | ⬜ pending |
| 13-XX-XX | TBD | TBD | CAMFIX-01 | T-13-01 | `record_episode.py` routes overhead camera through shared helper, not raw `cv2.VideoCapture(idx)` | unit | `pytest test_record_episode.py` | ❌ W0 (test file doesn't exist) | ⬜ pending |
| 13-XX-XX | TBD | TBD | CAMFIX-01 | — | `detect_devices.py::resolve_cameras()` returns `wrist_name` field | unit | `pytest test_detect_devices.py -k wrist_name` | ❌ W0 | ⬜ pending |
| 13-XX-XX | TBD | TBD | CAMFIX-02 | — | Live episode's `camera_overhead/*.png` and wrist frames show correct physical camera | manual-only | N/A — human visual confirmation (D-05/D-06) | N/A | ⬜ pending |
| 13-XX-XX | TBD | TBD | CAMFIX-03 | T-13-02 | `capture_stereo_frame()` returns dict with `sha256` field for policy_camera1/2/3 | unit | `pytest test_io_logger.py -k sha256` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*
*Task IDs and Plan/Wave columns are placeholders — the planner fills these in once PLAN.md files exist.*

---

## Wave 0 Requirements

- [ ] `control/test_record_episode.py` — does not currently exist; `record_episode.py` has zero test coverage today. Create baseline tests for `open_cameras()`/`record_still()`/`record_episode()` alongside new by-name-resolution behavior, following `test_run_vla_episode.py`/`test_stereo_camera.py`'s injectable-fake-capture patterns (`FakeCapture`, `_install_fake_capture(monkeypatch, ...)`).
- [ ] New tests for the shared resolve-by-name helper itself (name-first/fallback-to-index behavior, role-dispatch to the correct capture type per camera) — no existing test file covers a function that doesn't exist yet.
- [ ] New test(s) for `detect_devices.py::resolve_cameras()` returning a `wrist_name` field — extend existing `test_detect_devices.py` (has dense coverage of `resolve_cameras()`'s other return-dict behavior to pattern-match against).
- [ ] New test(s) for the `sha256` field in `capture_stereo_frame()`/`_record_frame()`'s returned dict — extend `test_io_logger.py` (closest existing precedent: `test_capture_stereo_frame_writes_png_and_returns_path_and_timestamp`).

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| `camera_overhead_left`/`camera_overhead_right` and wrist (`camera1`) frames show the real robot workspace/correct physical camera, not the laptop webcam or a misresolved device | CAMFIX-02 | Requires real AR0144 + real USB wrist camera hardware; Claude does not run physical hardware (CONTEXT.md D-05/D-06, following Phase 12's CONTEXT.md D-06 precedent) | User runs a live episode via `run_vla_episode.py --server-address ...` after this phase's code lands, opens the saved `camera_overhead_left/right` and `policy_camera1` (wrist) PNGs, and visually confirms each shows the correct physical camera/workspace, not a misidentified device |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references (helper tests, `test_record_episode.py`, `wrist_name` test, `sha256` test)
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
