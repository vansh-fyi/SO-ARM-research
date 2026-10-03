# Codebase Concerns

**Analysis Date:** 2026-10-03

## Tech Debt

**Backend proliferation in the VLA bridge:**
- Issue: There are about 30 backend modules (VLA servers, HTTP/gRPC adapters, Gemini and MLLM chat backends), each paired with its own large test file. Shared logic lives in `base_*` modules, but every new backend repeats the server, adapter, and test wiring.
- Files: `control/vla_bridge/backends/*.py`, `control/vla_bridge/*_server.py`, `control/test_backend_*.py`
- Impact: Changing the action contract or observation format means editing many files at once.
- Fix approach: Pull the common server boilerplate (Flask app, `/act` route, pinned-dependency check) into one shared module. Generate per-backend conformance tests from `control/test_backend_registry.py`.

**Large monolithic modules and tests:**
- Issue: `control/vla_bridge/backends/base_mllm.py` (915 lines), `control/run_vla_episode.py` (737), and `scripts/mjcf_to_urdf.py` (749) each mix several responsibilities. Some test files run past 1000 lines (`control/test_run_vla_episode.py` 2048, `control/test_backend_mllm_hf.py` 1178, `control/test_robot_client.py` 1053).
- Impact: These files are hard to review and edits to them often collide.
- Fix approach: Split `base_mllm.py` into separate modules for prompts, parsing, and conversion. Split `run_vla_episode.py` into CLI parsing and an episode loop.

**sys.path manipulation in policy servers:**
- Issue: Servers call `sys.path.insert` to import external model repos that are cloned on Colab.
- Files: `control/vla_bridge/rdt1b_server.py:170`, `control/vla_bridge/openvla_oft_server.py:254`, `control/vla_bridge/groot_n15_server.py:258`, `control/vla_bridge/fastfs_server.py:111`
- Impact: Imports only work when the repo layout matches what the code expects. Upstream repo changes break them silently.
- Fix approach: Pin each repo to a commit in `control/vla_bridge/policy_server.ipynb` and validate the path at startup with a clear error.

**Notebook/source drift:**
- Issue: `control/vla_bridge/policy_server.ipynb` is kept in sync by `scripts/sync_policy_notebook.py`, and the notebook currently has uncommitted modifications.
- Impact: The Colab runtime can run different code from the tested `.py` sources.
- Fix approach: Always run the sync script before a Colab session and add a CI or pre-commit check for drift.

**Vendored LIBERO tree:**
- Issue: `LIBERO/` is embedded with old pins (transformers 4.21.1, robosuite 1.4, MuJoCo 2.3.7). `control/requirements.txt` uses modern stacks (lerobot 0.6.1, opencv 5.x).
- Impact: Simulation and hardware control cannot share one environment.
- Fix approach: Keep separate venvs (`control/.venv`, `diagnostics/.venv`) and document which entry point needs which one.

## Known Bugs

**Version-sensitive model loading (OpenVLA):**
- Symptoms: Action tokens are decoded incorrectly, or loading fails, when the dependency versions are wrong.
- Files: `control/vla_bridge/openvla_server.py` (`PINNED_DEPENDENCIES`, lines ~30-143)
- Trigger: A Colab runtime with transformers or tokenizers versions other than 4.40.1 / 0.19.1. The docstring also notes a gripper normalization issue that clipped values to 1.0.
- Workaround: The server checks the pins and warns or fails at startup.

**Stock gripper cannot grasp LIBERO objects:**
- Symptoms: The roughly 2-3 cm jaw opening fails on every LIBERO object.
- Files: `So-101/` MJCF assets, `diagnostics/render_gripper_clamp.py`
- Workaround: Switch to the 84 mm parallel gripper (tracked in project memory).

## Security Considerations

**Unauthenticated policy servers bound to 0.0.0.0:**
- Risk: Any host that can reach the port can send observations and get actions, or overload the GPU. Exposure through tunnels (Colab, ngrok) makes these servers public.
- Files: `control/vla_bridge/openvla_server.py:382`, `control/vla_bridge/openvla_oft_server.py`, `control/vla_bridge/octo_server.py`, `control/vla_bridge/rdt1b_server.py`, `control/vla_bridge/groot_n15_server.py`, `control/vla_bridge/groot_n17_server.py`, `control/vla_bridge/wallx_server.py`, `control/vla_bridge/diffusion_policy_server.py`, `control/vla_bridge/fastfs_server.py`
- Current mitigation: None in code.
- Recommendations: Require a shared-secret header checked by `control/vla_bridge/backends/base_http_adapter.py`, and bind to localhost when no tunnel is used.

**Physical robot actuation from remote model output:**
- Risk: Bad model actions drive the real SO-101 arm.
- Files: `control/vla_bridge/safety_validator.py`, `control/vla_bridge/robot_client.py`, `control/run_vla_episode.py`
- Current mitigation: Each action is clamped to absolute joint limits, a per-step delta, and a velocity bound, with flags logged.
- Recommendations: Add a watchdog that stops the robot when the server times out. Never bypass `validate_action` in new backends.

**Third-party LLM API keys:**
- Risk: Keys could leak through logs or I/O recordings.
- Files: `control/vla_bridge/backends/base_gemini.py`, `control/vla_bridge/backends/mllm_claude.py`, `control/vla_bridge/backends/mllm_gpt.py`, `control/vla_bridge/io_logger.py`, `control/vla_bridge/mllm_record.py`
- Current mitigation: Keys come from environment variables, and `**/.env` is gitignored.
- Recommendations: Redact request headers in `io_logger.py` and `mllm_record.py` output.

## Performance Bottlenecks

**Remote inference round-trip per control tick:**
- Problem: Each action requires an HTTP or gRPC call from the local machine to Colab.
- Files: `control/vla_bridge/robot_client.py`, `control/tune_execution_hz.py`, `control/vla_bridge/backends/base_http_adapter.py`
- Cause: Network latency plus model inference on a T4/A100. Phase 12 ("bridge tick latency fix") addressed part of this.
- Improvement path: Use action chunking and asynchronous prefetch, and keep the inference servers resident.

**MLLM backends:**
- Problem: Chat-model calls (Claude, GPT, Gemini, HF Qwen/Kimi/DeepSeek) take seconds per step.
- Files: `control/vla_bridge/backends/base_mllm*.py`
- Improvement path: Request multi-step plans and run low-level control locally.

## Fragile Areas

**Broad exception handling:**
- Files: `control/vla_bridge/backends/base_mllm.py` (5 broad `except Exception`), `control/vla_bridge/check_checkpoint.py` (4), `control/vla_bridge/mllm_record.py` (4), `control/vla_bridge/backends/base_gemini.py` (2), and others
- Why fragile: Real failures can be swallowed and turned into silent fallbacks.
- Safe modification: Narrow the exception types and log the original error.

**Hardware device resolution:**
- Files: `control/detect_devices.py`, `control/device_map.json` (gitignored, machine-local), `control/vla_bridge/depth_camera.py`, `control/vla_bridge/stereo_camera.py`
- Why fragile: Serial ports (`/dev/cu.usbmodem*`) and camera indices change between machines and sessions.
- Safe modification: Always resolve devices through `detect_devices.py`. Never hardcode indices.

**Calibration and URDF derivation:**
- Files: `scripts/mjcf_to_urdf.py`, `scripts/verify_urdf.py`, `control/vla_bridge/mllm_urdf.py`, `control/vla_bridge/backends/eef_to_joint.py`, `diagnostics/*coppelia*.py`
- Why fragile: EEF-to-joint conversion through ikpy depends on the derived URDF matching the hardware. Joint-frame errors show up as wrong motions, not as exceptions.
- Test coverage: `scripts/test_verify_urdf.py` and `control/test_eef_to_joint.py` exist. Real-hardware conformance is checked manually.

## Scaling Limits

**Colab compute:**
- Current capacity: A single T4 or A100 session, depending on tier.
- Limit: Session timeouts and runtime drift (see the phase 6 fine-tuning blockers).
- Scaling path: Use a persistent GPU host or a containerized server image.

## Dependencies at Risk

**Exact pins on fast-moving packages:**
- Risk: `google-genai==2.25.0` (preview Robotics-ER API), `lerobot[feetech,async]==0.6.1`, and the transformers pin in the OpenVLA server are tied to preview or unstable APIs.
- Files: `control/requirements.txt`, `control/vla_bridge/openvla_server.py`
- Impact: Preview model IDs such as `gemini-robotics-er-2-preview` may be removed.
- Migration plan: Keep model IDs configurable and the pins centralized.

**No lockfile:**
- Risk: Transitive dependencies drift.
- Migration plan: Generate hash-locked requirements, for example with `pip-compile`.

## Missing Critical Features

**No CI:**
- Problem: `.github/` is gitignored and there is no automated test run.
- Blocks: Regressions across the many backends go undetected.

**No end-to-end sim-to-real evaluation harness:**
- Problem: Evaluation artifacts (`06b_eval_videos/`, `reports/`) are produced manually.

## Test Coverage Gaps

**Real model inference paths:**
- What's not tested: Backends are tested against fakes (`control/fake_providers.py`). The real servers' model loading is not exercised.
- Files: `control/vla_bridge/*_server.py`
- Risk: Dependency or upstream API breaks only show up on Colab.
- Priority: High

**Diagnostics scripts:**
- What's not tested: Most of `diagnostics/servo_*.py` and the camera tuning scripts.
- Files: `diagnostics/`
- Risk: Servo ID, protection, or torque writes could misconfigure hardware.
- Priority: Medium

**LIBERO / simulation integration:**
- What's not tested: `explorations/soarm_sanity.py` and the LIBERO env construction with the SOARM model.
- Priority: Medium

---

*Concerns audit: 2026-10-03*
