---
phase: 15
slug: mllm-style-pipeline-backends
status: draft
nyquist_compliant: true
wave_0_complete: false
created: 2026-10-02
---

# Phase 15 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Source: `15-RESEARCH.md` §Validation Architecture. Task IDs are bound to plans by the planner.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.3.4 (existing; `control/conftest.py` supplies `MockRobot`, `mock_calibration_file`) |
| **Config file** | none (no pytest.ini / pyproject); run from `control/` |
| **Quick run command** | `cd control && .venv/bin/python -m pytest test_mllm_schema.py test_mllm_prompt.py test_base_mllm.py -x -q` |
| **Full suite command** | `cd control && .venv/bin/python -m pytest -x -q` |
| **Estimated runtime** | ~10 seconds (baseline 1003 tests pass in ~9.4 s) |

---

## Sampling Rate

- **After every task commit:** Run the touched test file(s) with `-x -q`
- **After every plan wave:** Run `cd control && .venv/bin/python -m pytest -x -q`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Requirement | Behavior | Test Type | Automated Command | File Exists | Status |
|-------------|----------|-----------|-------------------|-------------|--------|
| PIPE-02 | New base conforms to `ActionSource`; shared prompt/schema identical across backends (golden hash); discovery registers 5 names, skips `base_*` | unit | `pytest test_backend_registry.py test_base_mllm.py -x` | ❌ W0 (extend registry test) | ⬜ pending |
| MLLM-04 | Schema property set == `JOINT_ORDER`; units from `ACTION_UNITS`; no per-joint min/max; all required; `additionalProperties:false`; deterministic hash; actions pass through UNMODIFIED `safety_validator` | unit | `pytest test_mllm_schema.py -x` | ❌ W0 | ⬜ pending |
| MLLM-05 | `chunk_len` in defaults, per-backend override works, prompt states N and seconds, derivation notes present | unit | `pytest test_mllm_prompt.py -k chunk -x` | ❌ W0 | ⬜ pending |
| MLLM-06 | Per-call record has restatement/plan/claims/task_complete; later ticks carry `chunk_ref`, not a trace copy; round-trips through real `IOLogger` | integration | `pytest test_base_mllm.py -k trace -x` | ❌ W0 | ⬜ pending |
| MLLM-07 | Malformed variants (missing key, string, null, bool, NaN/Infinity, short/long chunk, invalid JSON, empty, refusal, truncated, HTTP error, timeout) → earlier steps real, bad step + rest hold current position (never zeros), no exception, provider called once, no per-tick calls | unit/integration | `pytest test_base_mllm.py -k "malformed or hold or failure" -x` | ❌ W0 | ⬜ pending |
| MLLM-01/02/03 | Per-provider request construction (Claude/GPT omit sampling params, effort + schema set, `store=False`; HF pinned `model:provider`); images labelled; usage normalizers on canned payloads | unit (fake SDK clients) | `pytest test_backend_mllm_claude.py test_backend_mllm_gpt.py test_backend_mllm_hf.py -x` | ❌ W0 | ⬜ pending |
| D-18/D-25 | `task_complete` ends the episode (`termination.json`), chunk tail held, claim logged | integration | `pytest test_run_vla_episode.py -k task_complete -x` | extend existing | ⬜ pending |
| D-20..D-22 | requested-vs-applied settings, usage/cost math, prompt/schema hashes, sidecar with redacted raw payload + image hashes, secret absent from all artifacts | unit/integration | `pytest test_mllm_record.py test_base_mllm.py -k "record or redact or sidecar" -x` | ❌ W0 | ⬜ pending |
| D-06 | Depth fetched once per call; failure/429 non-fatal; CLI depth-flag rule relaxed for MLLM backends, still enforced for gRPC backends | unit + CLI | `pytest test_base_mllm.py -k depth -x`; `pytest test_run_vla_episode.py -k depth -x` | extend existing | ⬜ pending |
| D-00b | Missing/blank env var → `SystemExit` before robot/camera constructed; `api_key_env` must be an env-var NAME | unit | `pytest test_base_mllm.py -k key -x` | ❌ W0 | ⬜ pending |
| Prompt hygiene | Forbidden words absent (reasoning/thinking/trace/scratchpad); no primitive vocabulary; numeric safety caps absent unless toggled; `include_urdf` toggle changes hash; URDF text matches `So-101/So-101.urdf` limits | unit | `pytest test_mllm_prompt.py -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Plan and Task Binding (added by the planner, 2026-10-02)

| Requirement / decision | Plan and task that owns the automated check | Extra coverage |
|------------------------|---------------------------------------------|----------------|
| PIPE-02 | 15-01 T3 (base conforms to ActionSource, registry conformance), 15-06 T2 (first registered backend) | 15-09 T1 (all five backends built through the real builder: identical prompt, schema hash, payload order) |
| MLLM-04 | 15-01 T2 (schema from JOINT_ORDER/ACTION_UNITS, no min/max, lint, golden hash), 15-01 T3 (unmodified safety_validator clamp visible, D-04) | 15-09 T1 (schema lint on every backend) |
| MLLM-05 | 15-01 T2 (`-k chunk`: config value, derivation notes, prompt states N and seconds) | 15-10 T2 (measured MLLM API latency and duty cycle recorded; chunk_len unchanged without a user decision) |
| MLLM-06 | 15-01 T1 and T3 (`-k trace`, chunk_ref pointer, real IOLogger round trip) | 15-04 T2 (cost, sidecar, secret scan) |
| MLLM-07 | 15-01 T3 (malformed step variants), 15-03 T1 (whole-call failures, cap, task_complete, interrupt) | 15-10 T3 (hold path and Ctrl-C on the real arm) |
| MLLM-01 | 15-08 T1 and T2 (shared chat adapter, three pinned HF variants) | 15-10 T1 (live smoke per variant) |
| MLLM-02 | 15-06 T2 (Claude request/mapping/end to end) | 15-10 T1 |
| MLLM-03 | 15-07 T1 and T2 (GPT request/mapping/end to end) | 15-10 T1 |
| D-18 / D-25 | 15-03 T1 (base semantics) and T2 (`pytest test_run_vla_episode.py -k task_complete`) | 15-10 T3 |
| D-20..D-22 | 15-04 T1 and T2 (`-k "record or redact or sidecar"`), adapters' settings_summary tests in 15-06, 15-07, 15-08 | 15-10 T1 |
| D-06 / D-28 | 15-05 T1 (`pytest test_mllm_inputs.py`: grid pooling and NaN handling, three presentation modes, one depth call per chunk, 429 retry), 15-03 T2 (`pytest test_run_vla_episode.py -k depth`) | 15-10 T3 (depth live) |
| D-00b | 15-06 T1 (`pytest test_mllm_build.py -k key`: SystemExit before hardware, NAME-only api_key_env) | 15-09 T1 (config key-shape scan) |
| Prompt hygiene | 15-01 T2, 15-05 T2 (URDF text), 15-09 T1 (all backends) | n/a |
| Manual-only rows below | 15-10 (T1 provider smoke via 15-09 T3's `mllm_smoke` tool, T3 depth, URDF sign, hardware episodes, Ctrl-C, hold path) | n/a |

Wave 0 note: the stubs listed below are created test-first inside the TDD tasks of Plans 15-01 (fake_providers.py, test_mllm_schema.py, test_mllm_prompt.py, test_base_mllm.py, registry edit), 15-02 (SDK install behind the human checkpoint, test_mllm_requirements.py), 15-06 (test_mllm_build.py, test_backend_mllm_claude.py), 15-07 (test_backend_mllm_gpt.py) and 15-08 (test_backend_mllm_hf.py); test_mllm_record.py is 15-04 and test_run_vla_episode.py extensions are 15-03.

---

## Wave 0 Requirements

- [ ] `control/test_mllm_schema.py`, `test_mllm_prompt.py`, `test_mllm_record.py`, `test_base_mllm.py`, `test_backend_mllm_claude.py`, `test_backend_mllm_gpt.py`, `test_backend_mllm_hf.py`: stubs for all rows above
- [ ] Fake provider fixtures (`conftest.py` or `fake_providers.py`): fake Anthropic/OpenAI clients, canned payloads incl. refusal/truncated/NaN, fake `DepthCameraClient`, fake wrist/stereo
- [ ] Extend `test_backend_registry._action_source_bases()` and `test_run_vla_episode.py` (termination hook, depth-flag rule)
- [ ] `pip install anthropic==1.11.0 openai==3.23.0` into `control/.venv` behind a human-verify checkpoint; add pins + provenance to `control/requirements.txt`

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| One live call per provider with a tiny chunk: schema honoured with reasoning ON, usage reasoning-token field present, image acceptance, pinned-provider routing, no `reasoning_extraction` refusal; measure real API latency and update chunk-size table | MLLM-01/02/03/05 | Needs live API keys; no key on dev machine | Run each backend's live smoke script with a 3-step chunk; record results in UAT |
| Depth live: FastFS endpoint up, per-chunk depth ~6 s, heatmap legible, no 429 collision | D-06 | Needs Colab FastFS + stereo camera | Start endpoint, run one MLLM chunk with depth, inspect sidecar |
| Real executed chunk per backend, hold path observed (or provoked with a deliberately bad fake), Ctrl-C during a long call returns to start | MLLM-07, ROADMAP SC-2 | Needs the real SO-ARM101 | Per-backend hardware episode; log in UAT |
| Human sign/zero check of URDF vs real joints and gripper direction | D-07 (C8) | Hardware-only fact | Move each joint a small positive amount; compare to URDF axis; record |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
