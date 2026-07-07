# Testing Patterns

**Analysis Date:** 2026-07-07

## Test Framework

**Runner:** Not detected — no test framework is configured or present in this repository

**Config files:** None found (`pytest.ini`, `setup.cfg`, `pyproject.toml`, `vitest.config.*`, `jest.config.*` are all absent from the repo root)

**Run Commands:**
```bash
# No test commands defined
```

## Test File Organization

**Location:** No test files exist in this repository

**Naming:** No test files detected (searched for `test_*.py`, `*_test.py`, `*.test.*`)

**Structure:** Not applicable

## Test Structure

No tests are present. The codebase consists entirely of exploration and data-loading scripts without automated test coverage.

## Mocking

**Framework:** Not applicable — no tests exist

## Fixtures and Factories

**Test Data:** Not applicable

## Coverage

**Requirements:** None enforced

**View Coverage:**
```bash
# No coverage tooling configured
```

## Test Types

**Unit Tests:** Not present

**Integration Tests:** Not present

**E2E Tests:** Not present

## Manual Verification Pattern

While automated tests do not exist, all scripts use a consistent manual verification approach:
- Scripts are run directly (`python explorations/lib/explore.py`)
- Output images are saved to `explorations/outputs/` for visual inspection
- Console output uses `print(f"Saved → {path}")` to confirm file creation
- Scripts accept CLI arguments for targeted data inspection (e.g., `--episode 5`, `--scene scene0025_00`)

## Testability Notes

The codebase has several testable units that would benefit from automated tests:

- `load_scene()` in `explorations/real3dqa/explore.py` — pure data loading, returns `(xyz, rgb)` numpy arrays; testable with a fixture `.pth` file
- `load_tasks()` and `load_info()` in `explorations/lib/explore.py` — pure file readers
- `collect_first_frames()` in `explorations/lib/explore.py` — data collection with clear inputs/outputs
- `render_point_cloud()` in `explorations/render_scenes.py` — deterministic image rendering; output could be snapshot-tested

Adding tests would require:
1. A test framework: `pytest` is the idiomatic Python choice
2. Small fixture data files (minimal `.pth` and `.parquet` samples)
3. Mocking filesystem paths (override `ROOT`/`DATA` constants, or accept path arguments)

---

*Testing analysis: 2026-07-07*
