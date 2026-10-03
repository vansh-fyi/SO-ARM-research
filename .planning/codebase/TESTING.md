# Testing Patterns

**Analysis Date:** 2026-10-03

## Test Framework

**Runner:**
- pytest 8.3.4 (pinned in `control/requirements.txt`)
- Config: none (no `pytest.ini`/`pyproject.toml`); shared fixtures in `control/conftest.py`

**Assertion Library:**
- Plain `assert` with pytest rewriting; `pytest.raises` for errors

**Run Commands:**
```bash
cd control && .venv/bin/python -m pytest              # Run all control tests (~1090 test functions)
cd control && .venv/bin/python -m pytest test_backend_pi0.py -q   # Single file
cd scripts && python -m pytest                        # URDF / calibration utility tests
```
No watch mode or coverage tooling configured.

## Test File Organization

**Location:**
- Co-located flat in `control/` (not inside `vla_bridge/`): `control/test_*.py`
- Also `scripts/test_calibration_utils.py`, `scripts/test_verify_urdf.py`, `diagnostics/test_coppelia_alignment.py`

**Naming:**
- `test_<module>.py`; backends `test_backend_<name>.py`; MLLM pipeline `test_mllm_<stage>.py`
- Test function names are full sentences: `test_pi_backend_is_registered_on_the_grpc_bridge_lane`

**Structure:**
```
control/
├── conftest.py          # MockRobot, mock_calibration_file fixtures
├── fake_providers.py    # shared MLLM fakes (not collected)
├── test_*.py
└── vla_bridge/          # code under test
```

## Test Structure

**Suite Organization:**
```python
"""Tests for `vla_bridge.backends.pi0` ... Nothing here opens a serial port, a
camera, or a gRPC channel."""

PI_BACKENDS = ("pi0", "pi05")

# --- Registration ------------------------------------------------------------------

@pytest.mark.parametrize("name", PI_BACKENDS)
def test_pi_backend_is_registered_on_the_grpc_bridge_lane(name):
    assert name in backends.BACKEND_REGISTRY
```

**Patterns:**
- Module docstring states what is tested and asserts no hardware/network is touched
- Sections separated by `# --- Name ---` banner comments; flat functions, no classes
- `_args(**overrides)` helper builds a `SimpleNamespace` of CLI args with defaults
- `tmp_path` for file outputs; `pytest.mark.parametrize` heavily

## Mocking

**Framework:** pytest `monkeypatch` + hand-written fakes (no `unittest.mock` usage)

**Patterns:**
```python
def _record_grpc_build(monkeypatch):
    calls, sentinel = {}, object()
    def fake_build(policy_type, checkpoint, args, camera_names, joint_limits_deg, camera_map=None):
        calls.update(policy_type=policy_type, checkpoint=checkpoint)
        return sentinel
    monkeypatch.setattr(base_grpc, "build_grpc_action_source", fake_build)
    return calls, sentinel
```
- Fakes are reused across files by importing from other test modules (`from test_robot_client import FakeBridgeClient`) or `control/fake_providers.py` (`ScriptedAdapter`, `FakeCollector`, `FakeStereoCamera`, `FakeDepthClient`)
- `ScriptedAdapter` raises `AssertionError` when its script runs out, so unexpected paid API calls fail loudly

**What to Mock:**
- Robot hardware (`MockRobot` in `control/conftest.py`), serial bus, cameras (`cv2.VideoCapture`), gRPC/HTTP clients, vendor LLM SDKs, calibration cache file

**What NOT to Mock:**
- Pure logic: action contract, safety validator, schema/prompt building, registry; and the installed `lerobot` constants (tests import `SUPPORTED_POLICIES` directly to pin compatibility)

## Fixtures and Factories

**Test Data:**
```python
@pytest.fixture
def mock_robot():
    yield MockRobot()

def make_step(base: float, step: float, k: int) -> dict:  # fake_providers.py
    ...
```

**Location:**
- `control/conftest.py`, `control/fake_providers.py`; local Flask test double for the depth service in `control/test_depth_pipeline.py`

## Coverage

**Requirements:** None enforced

**View Coverage:**
```bash
Not configured (pytest-cov not installed)
```

## Test Types

**Unit Tests:**
- Majority; pure-logic and backend-wiring tests with fakes

**Integration Tests:**
- Local-service tests (Flask double, `pytest.importorskip("httpx")` / `importorskip("google.genai")` for optional SDKs); conformance suite `control/test_mllm_conformance.py`

**E2E Tests:**
- Not automated. Hardware and Colab runs are verified via UAT (`diagnostics/UAT/`, `.planning/` phase UAT docs)

## Common Patterns

**Async Testing:**
```python
# No asyncio tests; timing tests use time-based fakes (FakeTimedAction) instead
```

**Error Testing:**
```python
# control/test_backend_gemini_robotics_2.py
@pytest.mark.parametrize("missing", ["model", "api_key_env"])
def test_config_lacking_a_required_field_raises_system_exit_naming_it(config_dir, builder_spy, missing):
    config = {"model": PUBLIC_ER_MODEL, "api_key_env": API_KEY_ENV}
    del config[missing]
    config_dir(config)
    with pytest.raises(SystemExit, match=missing):
        BACKEND_REGISTRY[BACKEND_NAME](_args(), CAMERA_NAMES, JOINT_LIMITS_DEG)
    assert builder_spy.calls == []
```

Config errors surface as `SystemExit` naming the bad field (not `ValueError`), and a spy asserts nothing was built.

---

*Testing analysis: 2026-10-03*
