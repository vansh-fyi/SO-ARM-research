---
phase: 15-mllm-style-pipeline-backends
plan: 09
subsystem: control/vla_bridge (MLLM backends: conformance, docs, live smoke CLI)
status: complete
tags: [mllm, pipe-02, conformance, hygiene, docs, smoke, spend-gate, tdd]
requires:
  - control/vla_bridge/backends/base_mllm.py, base_mllm_build.py, base_mllm_chat.py (unmodified)
  - control/vla_bridge/backends/mllm_claude.py, mllm_gpt.py, mllm_hf_qwen.py, mllm_hf_deepseek.py, mllm_hf_kimi.py (unmodified)
  - control/vla_bridge/mllm_inputs.py (MllmInputCollector), mllm_record.py (CallRecorder), mllm_prompt.py, mllm_schema.py, mllm_urdf.py (unmodified)
  - control/test_mllm_build.py (hardware fixture, RecordingMockRobot), test_backend_mllm_{claude,gpt,hf}.py (fake SDK clients), test_mllm_schema.py (GOLDEN_SCHEMA_SHA256)
provides:
  - control/test_mllm_conformance.py (all-five identity, payload order, AST/config/lint hygiene, documentation assertions)
  - vla_bridge.mllm_smoke (main, SyntheticCollector, synthetic_frame, synthetic_depth, NEUTRAL_JOINT_LIMITS_DEG, DEFAULT_CHUNK_LEN)
  - "CLI: python -m vla_bridge.mllm_smoke --backend NAME --out DIR [--chunk-len N] [--effort LEVEL] [--joint-calibration PATH] --confirm-spend"
  - Five-backend wording in REQUIREMENTS.md, ROADMAP.md, COMMANDS.md and the backends package docstring
affects:
  - Plan 15-10 (live smoke uses mllm_smoke per backend; COMMANDS.md has the run commands and env var names)
  - Phase 16 (EVAL-01 now reads "16 minimum, up to 18" with the HF-variant choice left to Phase 16)
tech-stack:
  added: []
  patterns:
    - one conformance test builds every registered MLLM backend through BACKEND_REGISTRY and compares the shared prompt/schema/payload
    - AST source scans with a planted-violation self-test so the scan cannot pass vacuously
    - paid CLI with an explicit spend flag; dry run returns 2 before the key is read or a client is built
    - smoke inputs produced by the REAL collector fed synthetic camera/depth sources
key-files:
  created:
    - control/test_mllm_conformance.py
    - control/vla_bridge/mllm_smoke.py
    - control/test_mllm_smoke.py
  modified:
    - control/vla_bridge/backends/__init__.py
    - .planning/REQUIREMENTS.md
    - .planning/ROADMAP.md
    - control/COMMANDS.md
decisions:
  - "The smoke tool's SyntheticCollector wraps the real MllmInputCollector with synthetic wrist/stereo/depth sources, so images, depth heatmap/grid, depth note, meta and sidecar files are produced exactly as on the robot"
  - "mllm_smoke validates --effort against capabilities.effort_levels and --chunk-len against 1..MAX_CHUNK_LEN before any key read or call"
  - "Joint limits are loaded before the spend gate so the dry run already shows a missing-calibration WARNING; any load failure (missing, unreadable, malformed) falls back to the neutral limits"
  - "The ROADMAP.md milestone overview parenthetical ('a free HF-hosted reasoning model') was also updated, because the plan objective requires that nothing in ROADMAP.md still claims one free HF model"
  - "Requirement checkboxes and traceability rows are not changed (plan instruction); PIPE-02/MLLM-* stay Pending until Plan 15-10's live verification"
metrics:
  duration: "~9 min"
  completed: 2026-10-02
  tasks: 3
  tests_added: 52
  full_suite: "1584 passed (1532 baseline + 52 new)"
---

# Phase 15 Plan 09: Cross-backend conformance, docs alignment and live smoke CLI Summary

All five MLLM backends (`mllm-claude`, `mllm-gpt`, `mllm-hf-qwen`, `mllm-hf-deepseek`, `mllm-hf-kimi`) are now shown to be interchangeable. One test builds each backend through `BACKEND_REGISTRY` and the real shared builder, and asserts:
- all five send the same system prompt, schema hash, template version and chunk_len 20;
- the payload bytes are identical across all five;
- the hygiene rules hold (no tool-use, no eval/exec, no keys in config).

The docs now describe five backends. A keyless-by-default `python -m vla_bridge.mllm_smoke` is ready for Plan 15-10's first paid call per provider.

## What was built

### Task 1: `test_mllm_conformance.py` (cross-backend proof and hygiene)

- **Registration.** `MLLM_BACKENDS` holds exactly the five backends. Each is in `BACKEND_REGISTRY` and none is in `GRPC_BRIDGE_BACKENDS`. No registered callable comes from a `base_` module. `_discover_backends()` is run with a recording `importlib` stand-in: it skips `base_mllm`, `base_mllm_build` and `base_mllm_chat` and imports all five backend modules.
- **Success criterion 1 (the `five_built` fixture).** The fixture:
  - patches the hardware with the `hardware` fixture from `test_mllm_build.py`;
  - monkeypatches the three lazy client factories (`_make_anthropic_client`, `_make_openai_client`, `_make_chat_client`) to return scripted fake clients;
  - sets `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` and `HF_TOKEN` to sentinel values;
  - builds each backend with `BACKEND_REGISTRY[name](args, cameras, limits)`.

  The tests then assert:
  - **Identity.** There is exactly one distinct `system_prompt`, one schema hash and one schema dump. `prompt_template_version` is `mllm-prompt-1` and `chunk_len` is 20. The schema hash equals `GOLDEN_SCHEMA_SHA256` from `test_mllm_schema.py` (Plan 15-01), and the URDF text is identical.
  - **Payloads.** One real `get_action` per backend sends byte-identical payloads to the five fake clients: system text, schema, `(label, media type, base64)` image sequence (wrist, left, right) and user text.
  - **Records.** `prompt_sha256`, `schema_sha256` and `prompt_template_sha256` are identical across the five records.
- **Four-role payload order.** One `ProviderRequest` with wrist, left, right and depth goes through all five real adapter classes. The labelled-image sequence matches element for element, and the user text comes last.
- **Vocabulary lint.** `lint_prompt_text` finds nothing in any of these:
  - every source's system prompt and its schema JSON dump;
  - the first-call user text and the user text that carries the previous plan and depth grid;
  - the heatmap note and the grid note;
  - the URDF text;
  - the json_object-mode `RESPONSE FORMAT` block for each HF variant.

  `lint_schema_subset` passes on every schema.
- **AST scans.** These cover `vla_bridge/mllm_*.py` (which now includes `mllm_smoke.py`), `backends/base_mllm*.py` and `backends/mllm_*.py`. They find:
  - no `tools`/`tool_choice` keyword argument, dict key or subscript store;
  - no `eval`/`exec`;
  - no module-level `anthropic`/`openai` import in any backend module or in `base_mllm_chat`, `base_mllm` or `base_mllm_build`.

  `base_mllm.py` imports neither `lerobot` nor `cv2`. A self-test plants a file containing each kind of violation, so the scans cannot pass vacuously.
- **Config scan.** All six shipped `mllm_*.json` files are present. Each per-backend `api_key_env` is an uppercase variable name, and together they cover exactly the three key variables. No string in any of the six files matches the `sk-`, `sk-ant-`, `AIza` or `hf_` key shapes. These patterns are self-tested, and they do not flag `hf_provider`. Every per-backend config validates through `load_mllm_backend_config`, and the HF configs also pass `validate_chat_config`. All five inherit `depth_presentation` "both", `left_image_source` "raw", `include_urdf` true and `disclose_safety_caps` false. If a config ever overrides one of these, its `notes` must explain why.

### Task 2: documentation alignment (research C9, D-06, D-23)

- **`backends/__init__.py`.** Only the last docstring paragraph changed. It now says depth is out of scope for every VLA-style backend (14-CONTEXT D-03), and that the MLLM base is the one deliberate exception (15-CONTEXT D-06, D-28):
  - `base_mllm_build.py` builds the single depth client;
  - `mllm_inputs.py` fetches depth once per chunk and feeds it into the MLLM prompt only;
  - depth never enters a PolicyServer observation dict.

  No code changed.
- **`REQUIREMENTS.md`:**
  - **MLLM-01** now names the three HF variants on one shared adapter via HF Inference Providers, and says Phase 16 decides which run.
  - **DEPTH-CAL-03** gains the Phase 15 note.
  - **EVAL-01** now reads "HF reasoning model variant(s) + Claude + GPT; 16 backends minimum with one HF variant, up to 18 if all three HF variants registered in Phase 15 are run; Phase 16 decides".
  - The **router out-of-scope row** now says "beyond the v2.1 MLLM backends (Claude, GPT and three HF variants ...)".
  - No checkbox or traceability status changed, and a test asserts this.
- **`ROADMAP.md`** (scoped replacements only):
  - the Phase 15 goal now reads "five registered raw-JSON-reasoning MLLM backends — three HuggingFace-hosted open reasoning-model variants (...), Claude, and GPT";
  - criteria 1 and 5 say "all 5 registered backends", and criterion 2 says "Each HF variant, Claude, and GPT";
  - the phase-list line says "onboard 5 raw-JSON MLLM backends (3 HF open-model variants, Claude, GPT)";
  - the Phase 16 HF entry gains "(up to 3 HF variants are registered in Phase 15; Phase 16 selects which run)";
  - the milestone overview parenthetical changed too (see Deviations).
- **`COMMANDS.md`** has a new "## MLLM backends (Phase 15)" section containing:
  - the five backend names;
  - the three env var exports, noting that keys are exported in the shell and never written to files;
  - one spend line (about $1 per call for Claude/GPT at xhigh, HF credits required with a $0.10 free tier, Kimi-K3 about $0.35 per call);
  - one retention line (images leave this machine; Anthropic's 30-day retention);
  - the `mllm_prompt`, `mllm_inputs` (`--out`/`--wrist-camera-index`/`--stereo-camera-index`/`--calibration`/`--endpoint`) and `mllm_smoke` commands;
  - an episode example (`--backend mllm-claude --max-steps 120 --depth-endpoint ... --depth-calibration ...`) with the note that 60 ticks is only about 3 chunks.
- **Documentation tests** in `test_mllm_conformance.py` assert the exact new wording and the absence of the superseded phrases.

### Task 3: `vla_bridge/mllm_smoke.py` and `test_mllm_smoke.py`

- **`main(argv)` order of operations:**
  1. Parse the arguments. `--chunk-len` defaults to 3 and must be 1..200.
  2. Look up the spec. An unknown name raises `SystemExit` naming it.
  3. Load the config with `load_mllm_backend_config`. Deep-copy it, then apply the `chunk_len` and `--effort` overrides. `--effort` is checked against `capabilities.effort_levels`.
  4. Load the joint limits. If the calibration file is missing, unreadable or malformed, print a WARNING and use neutral limits (-90 to 90 for the arm joints, 0 to 100 for the gripper).
  5. Print the plan: backend, provider, request model string (`<model>:<hf_provider>` for HF), chunk_len, effort (with any override noted), max_tokens, depth_presentation, a synthetic-inputs line and the spend WARNING.
  6. **Without `--confirm-spend`, return 2.** The key is not read and no client is built.
  7. Read the key from the configured variable. If it is missing or blank, raise `SystemExit` naming the variable.
  8. Call `spec.adapter_factory(key, config)`, the same factory `--backend` uses.
  9. Build `CallRecorder(out/mllm_calls, secrets, prices)` and `MllmActionSourceBase(..., urdf_text, recorder, secrets)`.
  10. Make exactly one `get_action` call with the mid-range joint state and the instruction "Move the arm slightly toward the pen".
- **Synthetic inputs.** `SyntheticCollector` feeds the real `MllmInputCollector` from three sources:
  - a 640x480 wrist frame and 800x600 left/right frames, each a different tinted gradient with a filled rectangle and a text label (left and right show a small disparity);
  - a 576x800 depth map: a plane at 0.8 m with a 0.5 m box in the middle.

  `depth_presentation` therefore decides the depth image and grid text exactly as on the robot. The sidecar gets `depth.npy`, `depth_heatmap.png` and `depth_grid.txt`.
- **Report.** The tool prints:
  - `Saved → <out>/mllm_calls/0000`;
  - PASS/FAIL schema honoured (`parse.ok` and `n_valid == chunk_len`);
  - PASS/WARN reasoning tokens reported;
  - PASS/FAIL stop reason, with the redacted error for failed calls;
  - the refusal category, with an explicit FAIL message for `reasoning_extraction` that points at `mllm_prompt.lint_prompt_text`;
  - API latency in seconds, estimated cost with the `verified_on` date of the prices, request id, model returned, and the settings requested/applied/omitted;
  - `RESULT: PASS|FAIL`.

  It returns 0 only when the stop is complete and the schema was honoured.
- **`test_mllm_smoke.py` (27 tests):**
  - A dry run returns 2 with no factory call and no key in the environment.
  - Dry runs on all five shipped backends print the request model string while every SDK client factory is patched to raise.
  - `SystemExit` is raised for an unknown backend, a missing key, a blank key, a disallowed effort and `--chunk-len 0`.
  - A confirmed fake run makes one call, writes the full sidecar file set (four images included) and prints the PASS lines.
  - The sidecar records four distinct images, the depth grid in the user text and depth status ok.
  - All three `depth_presentation` modes are covered.
  - The secret sentinel, which is planted in the plan text, the raw payload and the thinking text, appears nowhere in stdout or any file.
  - The `--effort` override applies to that run only and shows in the printed settings.
  - `--chunk-len 5` is honoured.
  - Null reasoning tokens give a WARN line and still return 0.
  - A short chunk gives FAIL and returns 1.
  - A `reasoning_extraction` refusal gives the explicit FAIL message.
  - A provider exception carrying the key fails and is redacted.
  - Missing calibration gives neutral limits, checked against `build_system_prompt`. A calibration file gives limits from `load_joint_limits_deg`.
  - `include_urdf` false leaves out the URDF block.
  - A module hygiene check.

## Verification

- `pytest test_mllm_conformance.py test_mllm_smoke.py`: 52 passed (25 conformance + 27 smoke).
- Full suite: **1584 passed** in about 13.7 s (1532 baseline plus 52 new).
- `python -c "import vla_bridge.backends"` exits 0 after the docstring edit.
- Keyless CLI dry run with the three key variables unset: `python -m vla_bridge.mllm_smoke --backend mllm-hf-kimi --out <scratch>` prints `model moonshotai/Kimi-K3:baseten`, `chunk_len: 3`, the spend WARNING and "Not calling", then exits 2.
- `git diff --stat` for Task 2 touched exactly `backends/__init__.py` (docstring only), `REQUIREMENTS.md`, `ROADMAP.md`, `COMMANDS.md` and `test_mllm_conformance.py`.

## TDD Gate Compliance

- Task 1: `3a91510` (test-only). This task adds no production behavior: it asserts properties of code that Plans 15-01..15-08 already shipped, so the tests passed on the first run. That does not make them vacuous:
  - the scanner self-test proves the AST scans catch planted `tools`, `tool_choice`, `eval` and top-level SDK imports;
  - the identity tests would fail if any backend built its own prompt or schema.
- Task 2: RED `bcb4299` (4 documentation tests failing), then GREEN `4cdf404` (the docs edits).
- Task 3: RED `2145a34` (`ImportError: cannot import name 'mllm_smoke'`), then GREEN `b6e8960`.
- No refactor commits were needed.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing critical functionality] Stale "free HF-hosted reasoning model" claim in the ROADMAP.md milestone overview**
- **Found during:** Task 2.
- **Issue:** The plan listed specific ROADMAP.md edits and said to change nothing else. Its objective, however, says nothing in ROADMAP.md should still claim one free HF model. The milestone overview paragraph (line 11) still said "Phase 15 is MLLM-style (... — a free HF-hosted reasoning model, Claude, and GPT)".
- **Fix:** That parenthetical now reads "HF-hosted open reasoning-model variants, Claude, and GPT". A documentation test asserts the old phrase is gone. Nothing else in that paragraph changed.
- **Files modified:** .planning/ROADMAP.md
- **Commit:** 4cdf404

### Choices the plan left open

- **Smoke collector.** The plan asked for a "synthetic collector class local to this module" with meta that mirrors the real collector. `SyntheticCollector` is local to the module but delegates to the real `MllmInputCollector`, fed by synthetic wrist, stereo and depth sources. The meta, depth note and sidecar files therefore match the real collector by construction rather than by copying.
- **Smoke input validation.** `--effort` is validated against `capabilities.effort_levels`, and `--chunk-len` is bounded to 1..`MAX_CHUNK_LEN`. Both checks run before the key is read, so a typo never reaches a paid call.
- **Dry-run limits.** Joint limits load before the spend gate, so a dry run already shows a missing-calibration WARNING.
- **Unlabelled lines.** Latency, cost, request id and settings print as plain labelled lines rather than PASS/WARN, because they carry no pass/fail criterion. `RESULT: PASS|FAIL` summarises the exit code.
- **Payload test helpers.** The payload tests reuse the fake SDK clients already defined in the three backend test modules (imported by name, no `test_*` names imported), so no new fake client was written.

Environment: tests ran with `control/.venv`. Nothing was installed. No live or paid API, robot, camera or depth service was contacted. The smoke tool was run as a dry run only.

## Known Stubs

None. The smoke tool's synthetic images and depth map are intentional: the tool exists so the first paid call needs no hardware and sends no lab images.

## Threat Flags

None beyond the plan's threat register. Each mitigation is implemented and tested:
- **T-15-32:** `--confirm-spend` gate, one call, chunk_len 3 by default, and a dry run that returns 2 before the key is read.
- **T-15-33:** config key-shape scan, `api_key_env` name pattern, and smoke stdout and files scanned for the secret.
- **T-15-34:** AST scans for `tools`/`tool_choice`, eval/exec and top-level SDK imports. These cover the new `mllm_smoke.py` automatically.

## User Setup Required

None for this plan. Plan 15-10 runs `python -m vla_bridge.mllm_smoke --backend <name> --chunk-len 3 --out outputs/smoke_<name> --confirm-spend` once per backend. It needs `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` and `HF_TOKEN` (fine-grained, with Inference Providers permission and purchased credits) exported in the shell. On the first GPT call, check whether `reasoning.summary` is accepted; if it is not, set `capabilities.summary_supported` to false in `mllm_gpt.json`.
