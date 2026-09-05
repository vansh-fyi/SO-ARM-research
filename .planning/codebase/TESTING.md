# Testing Patterns

**Analysis Date:** 2026-09-05

## Test Framework

**Runner:**
- None. No `pytest`, `unittest`, `nose`, or any test runner is configured anywhere in project-owned code.
- No `pytest.ini`, `pyproject.toml [tool.pytest]`, `tox.ini`, or `jest.config.*` exists at the repo root or in `explorations/`, `diagnostics/`, or `control/`.
- The only `test_*.py` files in the repository tree live inside vendored virtualenv site-packages (`explorations/.venv`, `diagnostics/.venv`, `control/.venv` — e.g. `pandas/tests/*`, `typer` test suites). These belong to third-party dependencies, not this project, and must not be treated as project test coverage.

**Assertion Library:**
- Not applicable — no test files exist to assert against.

**Run Commands:**
```bash
# No test command exists in this repo.
# Validation is currently done via manual UAT checklists and standalone
# PASS/FAIL harness scripts (see "Test Types" below).
```

## Test File Organization

**Location:**
- Not applicable — no automated test suite exists.

**Naming:**
- Not applicable.

**Structure:**
- Not applicable.

## Test Structure

**Suite Organization:**
- No `describe`/`it` or `TestCase` structure exists in this codebase. There is nothing to show.

## Mocking

**Framework:** None.

**Patterns:**
- Not used. Hardware and simulation code is validated by actually driving real hardware (servos, cameras) or a real MuJoCo/LIBERO simulation, not by mocking those interfaces.

## Fixtures and Factories

**Test Data:**
- Not applicable — no test fixtures exist. Real data (LIBERO parquet chunks, Real-3DQA `.pth` point clouds) is downloaded into `explorations/data/` (gitignored) and used directly by exploration scripts during manual runs.

**Location:**
- N/A

## Coverage

**Requirements:** None enforced — no coverage tool configured.

**View Coverage:**
- Not applicable.

## Test Types

This project substitutes structured **manual UAT (User Acceptance Testing) checklists** and **standalone validation harness scripts** for automated tests. When asked to "add tests" or "verify" functionality, follow these existing patterns rather than introducing a new pytest suite unprompted.

**PASS/FAIL Validation Harness (closest thing to an automated test suite):**
- `explorations/soarm_sanity.py` — a CLI harness with `--check {compile,model,reset,render,soak,tasks,all}` modes. Each check function returns a boolean, routed through `_report(ok, name, reason)` which prints `PASS — {name}` or `FAIL — {name}: {reason}` and the script exits 0 only if every requested check passes.
  - Pattern for new validation scripts: define one function per check, each wrapped in `try/except Exception as exc:` that converts any failure into a `FAIL` report rather than an uncaught traceback; keep heavy imports (mujoco, robosuite, libero) lazy inside check functions so `--help` works without those dependencies installed.
  - Run via: `conda run -n libero python explorations/soarm_sanity.py --check all`

**Manual UAT Checklists (Markdown, hardware-focused):**
- `diagnostics/UAT/function/UAT.md` — functional UAT steps for the physical arm (referenced by `control/record_episode.py`'s own docstring, e.g. "Steps 5/6/7")
- `diagnostics/UAT/components/UAT.md` — component-level UAT
- `diagnostics/UAT/assembly/gripper/UAT.md` — gripper assembly UAT
- `diagnostics/UAT/assembly/main/UAT.md` — main arm assembly UAT
- Pattern: numbered manual steps a human operator performs against real hardware, cross-referenced from script docstrings (e.g. `control/record_episode.py:8-9` explicitly ties itself to "function UAT Steps 5, 6, and 7").

**Diagnostic/probe scripts (manual, interactive, hardware-in-the-loop):**
- `diagnostics/servo_scan.py`, `diagnostics/servo_move_test.py`, `diagnostics/servo_torque.py`, `diagnostics/servo_set_id.py`, `diagnostics/servo_set_protection.py`, `diagnostics/servo_set_torque_limit.py`, `diagnostics/servo_drive_to_stall.py`, `diagnostics/camera_test.py` — each is a standalone CLI script run against real hardware (Feetech STS3215 servo bus, cameras) that prints PASS/FAIL-style or found/not-found results. These serve as the project's integration tests for hardware, executed manually rather than in CI.

**Unit Tests:**
- Not present. No pure-logic unit tests exist for any exploration, diagnostic, or control script.

**Integration Tests:**
- Effectively covered by the manual UAT checklists and `soarm_sanity.py` harness described above — these exercise the full simulation stack (MuJoCo compile, robosuite env reset, LIBERO task stepping) or full hardware stack (servo bus, camera capture, robot connect/disconnect) end-to-end.

**E2E Tests:**
- Not used in the software-automation sense. The closest equivalent is a human running `soarm_sanity.py --check all` or working through a UAT.md checklist against physical/simulated hardware.

## Common Patterns

**Async Testing:**
- Not applicable — no async code and no test framework.

**Error Testing:**
- The harness pattern in `explorations/soarm_sanity.py` is the de facto "error testing" convention: wrap the operation under test in `try/except Exception as exc:`, and report failure via `_report(False, name, reason=str(exc))` rather than asserting on a specific exception type.

## Guidance for Adding New Validation

When a phase or task calls for "tests" in this codebase:
1. **Simulation/software logic** → extend `explorations/soarm_sanity.py` with a new `--check <name>` function following its existing `_report()` PASS/FAIL pattern, or create a similarly-structured standalone harness script under `explorations/`.
2. **Physical hardware behavior** → add or extend a manual UAT checklist under `diagnostics/UAT/` (functional, component, or assembly), or add a new standalone probe script under `diagnostics/` alongside the existing `servo_*.py`/`camera_test.py` scripts.
3. **Do not introduce `pytest`/`unittest` unprompted** — there is no existing convention or CI wiring for it in this repo; confirm with the user before adding a new test framework dependency.

---

*Testing analysis: 2026-09-05*
