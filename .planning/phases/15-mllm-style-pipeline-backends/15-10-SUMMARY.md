---
phase: 15-mllm-style-pipeline-backends
plan: 10
subsystem: control/vla_bridge (MLLM backends: live provider, depth, URDF and real-arm verification)
status: complete
tags: [mllm, live-verification, smoke, hardware, depth, urdf-sign, ctrl-c, hold-path, duty-cycle, streaming]
requires:
  - phase: 15-09
    provides: "vla_bridge.mllm_smoke CLI, cross-backend conformance suite, COMMANDS.md MLLM section"
  - phase: 15-01..15-08
    provides: "MllmActionSourceBase, shared prompt/schema, CallRecorder, mllm_inputs collector, five registered MLLM backends"
  - phase: 12
    provides: "resident FastFS depth service and the chunk_len derivation (step_dt_s 0.785, depth median 6.15 s)"
provides:
  - "15-LIVE-RESULTS.md sections 1-6: live smoke, config changes, depth live, URDF sign and gripper check, one real-arm episode per backend, Ctrl-C and hold path, chunk-size re-derivation"
  - "Streaming HF router chat adapter (base_mllm_chat.py) that survives the router's ~120 s idle cut"
  - "mllm-hf-qwen re-pinned to deepinfra; mllm-hf-deepseek switched to DeepSeek-V4.1-Flash on deepinfra; mllm-claude switched to claude-opus-5-5 (user decisions 2026-10-02)"
  - "mllm_defaults.json chunk_len_derivation.measured_mllm_api_latency_s and duty_cycle_at_measured_latency (smoke-based)"
  - "run_vla_episode.py keeps termination reason keyboard_interrupt when the piped stdout reader dies on Ctrl-C"
affects:
  - "Phase 16 (backend comparison): per-backend latency, duty cycle, and model-behaviour notes (DeepSeek degeneration, Kimi scene misread, Qwen coarse kinematics) feed its ablation choices"
  - "Phase 17 (safety caps): validator hard limits unchanged; zero validator flags across all hardware episodes"
tech-stack:
  added: []
  patterns:
    - "HF router chat calls stream (stream=True, include_usage) and the raw sidecar is a JSON-safe reconstruction of the stream"
    - "Config flags change only from evidence recorded in 15-LIVE-RESULTS.md section 2 or an explicit dated user decision"
    - "Termination reason is assigned before any stdout write in the KeyboardInterrupt handler"
key-files:
  created:
    - .planning/phases/15-mllm-style-pipeline-backends/15-LIVE-RESULTS.md
  modified:
    - control/vla_bridge/backends/base_mllm_chat.py
    - control/vla_bridge/backends/mllm_claude.py
    - control/vla_bridge/backends/mllm_hf_qwen.py
    - control/vla_bridge/backends/mllm_hf_deepseek.py
    - control/vla_bridge/backend_configs/mllm_defaults.json
    - control/vla_bridge/backend_configs/mllm_claude.json
    - control/vla_bridge/backend_configs/mllm_hf_qwen.json
    - control/vla_bridge/backend_configs/mllm_hf_deepseek.json
    - control/run_vla_episode.py
    - control/test_backend_mllm_claude.py
    - control/test_backend_mllm_hf.py
    - control/test_mllm_conformance.py
    - control/test_mllm_smoke.py
    - control/test_run_vla_episode.py
    - .planning/REQUIREMENTS.md
    - .planning/ROADMAP.md
key-decisions:
  - "chunk_len stays 20 for all five backends (user decision 2026-10-03, option 1), re-confirmed the same day against the Task 3 episode table where four of five backends sit below a 10 percent duty cycle; N or effort changes are left to Phase 16 ablations"
  - "mllm_defaults.json keeps the smoke-based Task 2 latency and duty-cycle values; the episode values live only in 15-LIVE-RESULTS.md section 6"
  - "No evidence-driven capability flag flip was needed: GPT accepted reasoning.summary (A5), and every HF variant honoured json_schema with reasoning on and the 4-image png request (A3/A4/A6)"
  - "Ctrl-C fix 9ac4472: run_vla_episode.py sets reason keyboard_interrupt before a guarded print, so a SIGINT that also kills a piped reader no longer records max_steps_reached"
  - "Hold path satisfied naturally on mllm-hf-deepseek (empty-response and truncated holds), so the provoked max_tokens 1024 run was not needed and no config was edited for it"
patterns-established:
  - "Live-results file as the single evidence source: config changes cite its section and row; a conformance test checks that mllm_defaults.json matches the section 6 smoke table"
requirements-completed: [PIPE-02, MLLM-01, MLLM-02, MLLM-03, MLLM-05, MLLM-06, MLLM-07]
coverage:
  - id: D1
    description: "Each of the five backends produced a live schema-valid chunk with reasoning on, with usage, image acceptance, provider routing, refusal category, latency and cost recorded"
    requirement: MLLM-01
    verification:
      - kind: manual_procedural
        ref: "15-LIVE-RESULTS.md section 1 (python -m vla_bridge.mllm_smoke --backend NAME --chunk-len 3 --confirm-spend; exit 0 for all five)"
        status: pass
    human_judgment: true
    rationale: "Paid live provider calls run by the operator; no automated test can reach the real providers"
  - id: D2
    description: "HF router chat adapter streams, so long-reasoning HF variants no longer hit the router's ~120 s HTTP 504"
    requirement: MLLM-01
    verification:
      - kind: unit
        ref: "control/test_backend_mllm_hf.py#test_real_sdk_stream_is_requested_and_assembled_end_to_end"
        status: pass
      - kind: unit
        ref: "control/test_backend_mllm_hf.py#test_truncated_stream_keeps_partial_text_and_usage"
        status: pass
      - kind: unit
        ref: "control/test_backend_mllm_hf.py#test_call_sends_exactly_request_kwargs_and_never_the_raw_non_streaming_path"
        status: pass
    human_judgment: false
  - id: D3
    description: "User-decided config changes: qwen re-pinned to deepinfra, deepseek switched to V4.1-Flash, claude switched to claude-opus-5-5, each with dated source notes"
    requirement: MLLM-02
    verification:
      - kind: unit
        ref: "control/test_backend_mllm_claude.py#test_shipped_config_records_the_opus_5_5_switch_and_its_sources"
        status: pass
      - kind: unit
        ref: "control/test_backend_mllm_hf.py#test_repinned_configs_cite_the_router_listing_for_the_new_pin"
        status: pass
      - kind: unit
        ref: "control/test_backend_mllm_hf.py#test_kimi_notes_flag_cost_and_qwen_notes_record_the_deepinfra_repin"
        status: pass
    human_judgment: false
  - id: D4
    description: "Measured MLLM API latency and duty cycle recorded next to the chunk_len derivation; chunk_len unchanged at 20"
    requirement: MLLM-05
    verification:
      - kind: unit
        ref: "control/test_mllm_conformance.py#test_measured_mllm_api_latency_record_matches_the_live_results_file"
        status: pass
      - kind: unit
        ref: "control/test_mllm_conformance.py#test_flipped_flags_carry_dated_evidence_and_chunk_len_needs_a_user_decision"
        status: pass
    human_judgment: false
  - id: D5
    description: "Depth live (7.1 s raw / 6.3 s aligned, valid_fraction 1.0, no 429) and the URDF sign and gripper-direction check (all five joints and the gripper match)"
    requirement: PIPE-02
    verification:
      - kind: manual_procedural
        ref: "15-LIVE-RESULTS.md sections 3 and 4"
        status: pass
    human_judgment: true
    rationale: "Physical observation of camera, depth heatmap and joint directions on the real arm"
  - id: D6
    description: "One real-arm episode per backend with chunks executed through the unchanged validator, natural hold path, and task_complete checked against the scene"
    requirement: MLLM-06
    verification:
      - kind: manual_procedural
        ref: "15-LIVE-RESULTS.md section 5 (run_vla_episode.py --backend NAME --max-steps 40 --instruction 'Pick up the pen')"
        status: pass
    human_judgment: true
    rationale: "Real hardware episodes judged by the operator against the scene"
  - id: D7
    description: "Ctrl-C during a provider call returns the arm to start and records termination reason keyboard_interrupt, even when the piped stdout reader dies"
    requirement: MLLM-07
    verification:
      - kind: unit
        ref: "control/test_run_vla_episode.py#test_keyboard_interrupt_reason_survives_broken_stdout_pipe"
        status: pass
      - kind: manual_procedural
        ref: "15-LIVE-RESULTS.md section 5 Ctrl-C run 2 (control/outputs/mllm_ctrlc_002, termination keyboard_interrupt)"
        status: pass
    human_judgment: false
duration: "14h 18m wall clock (includes an overnight pause at the Task 3 hardware checkpoint)"
completed: 2026-10-03
metrics:
  started: "2026-10-02T16:37:16Z"
  completed: "2026-10-03T06:54:59Z"
  tasks: 3
  files_modified: 17
  tests_added: 20
  full_suite: "1604 passed (1584 after 15-09 plus 20 new)"
---

# Phase 15 Plan 10: Live verification of the five MLLM backends Summary

**All five MLLM backends passed a live paid smoke call. Depth was confirmed live, the joint and gripper sign conventions were checked on the real arm, and each backend drove the SO-ARM101 for one 40-step "Pick up the pen" episode. mllm-claude (claude-opus-5-5) completed the first successful raw-autonomy pick, and its task_complete claim matched the scene.** All evidence is in `15-LIVE-RESULTS.md`.

## Performance

- **Duration:** 14h 18m wall clock, from 2026-10-02 22:07 to 2026-10-03 12:25 IST. This includes an overnight pause at the Task 3 hardware checkpoint (commit 4c60479).
- **Tasks:** 3 of 3, plus the mid-plan fixes the user decided on 2026-10-02.
- **Files modified:** 17 (code, configs, tests, results file, REQUIREMENTS.md, ROADMAP.md).
- **Tests:** 20 added. The full suite gives **1604 passed** at 3f4d5e1.

## Accomplishments

- **Live smoke (Task 1, section 1).** All five backends returned exit code 0 with 3 of 3 valid steps, reasoning on, and stop complete. API latency per smoke call:

  | backend | API latency (s) |
  |---------|-----------------|
  | kimi | 8.1 |
  | claude-opus-5-5 | 12.7 |
  | deepseek V4.1-Flash | 22.4 |
  | gpt-6-astra | 32.1 |
  | qwen on deepinfra | 276.5 |

  - GPT accepted `reasoning.summary` (A5).
  - Every HF variant honoured `json_schema` with reasoning on and accepted the 4-image png request (A3, A4, A6).
  - Claude returned no `reasoning_extraction` refusal.
  - Key-shape scans of every sidecar and of the results file are clean. The only regex hit is the Task 1 note that describes the regex itself.
- **Fixes the user decided on during Task 1 (2026-10-02, section 2):**
  - **HF adapter.** The HF router chat adapter now streams. Before, the router's edge proxy cut idle non-streaming calls at about 120 s with HTTP 504.
  - **mllm-hf-qwen.** Re-pinned from cerebras to deepinfra, because cerebras was failing with HTTP 503 from the circuit breaker and is no longer listed for this model.
  - **mllm-hf-deepseek.** Switched to `deepseek-ai/DeepSeek-V4.1-Flash:deepinfra`, because DeepSeek retired V4-Flash-Vision-Exp upstream.
  - **mllm-claude.** Switched from claude-fable-5-1 to claude-opus-5-5, with effort kept at xhigh explicitly.
  - Qwen and DeepSeek were re-smoked, and Opus 5.5 was smoked; all three passed.
- **Latency record (Task 2).**
  - `mllm_defaults.json` `chunk_len_derivation` now holds `measured_mllm_api_latency_s` and `duty_cycle_at_measured_latency` for all five backends.
  - No capability flag flip was needed, and chunk_len was not changed.
- **Depth live (Task 3 step A, section 3).**
  - Depth takes 7.1 s with the raw left image and 6.3 s with the aligned one. valid_fraction is 1.0, the heatmap and the 24x32 grid are legible and plausible, and there was no HTTP 429.
  - The wrist camera is cv2 index 1 ("USB Camera"). Index 0 is the built-in FaceTime camera.
- **URDF sign and gripper check (step B, section 4).**
  - The signs of all five joints match the URDF.
  - The gripper matches the prompt (0 is closed, 100 is open).
  - There is no blocking finding.
- **Hardware episodes (step C, section 5).** One 40-step "Pick up the pen" episode per backend, with zero validator flags. The arm held position during every blocking call.

  | backend | result |
  |---------|--------|
  | mllm-claude | Picked up the pen and claimed task_complete after 21 steps. The claim matched the scene. |
  | mllm-gpt | Ended with open jaws straddling the pen, one call short of a close. |
  | mllm-hf-qwen | Missed the pen and was re-approaching when the budget ended. |
  | mllm-hf-kimi | Made tiny moves and misread the scene, with a likely sign error. |
  | mllm-hf-deepseek | Both calls degenerated into multilingual word salad mid-reasoning (empty response, then truncated at max_tokens). The arm held correctly. |

- **Hold path (step E).** Observed naturally on mllm-hf-deepseek, with both an `empty-response-holding-position` chunk and a `truncated-holding-position` chunk. Each next call started only after chunk_len ticks.
- **Ctrl-C (step D).**
  - Run 1 found a defect, which was fixed in 9ac4472.
  - Run 2, pressed during the first provider call, gave termination reason `keyboard_interrupt`. PASS.
- **Chunk-size re-derivation (section 6).** Episode API latencies with real images are far above the smoke for four backends. Duty cycles at chunk_len 20:

  | backend | duty cycle |
  |---------|------------|
  | qwen | 5.1% |
  | deepseek | 7.1% |
  | claude | 7.8% |
  | gpt | 8.9% |
  | kimi | 16.2% |

  - No call exceeded timeout_s 900.
  - The user chose to keep 20 for all backends (option 1) on 2026-10-03 and re-confirmed it against the episode table.

## Task Commits

| Task | Name | Commits |
|------|------|---------|
| 1 | Live provider smoke, all five backends PASS (section 1) | e4022d1 (results skeleton), 63fe4e5 (GPT/Claude/Kimi smoke; Qwen/DeepSeek provider-side failures), 55d88bd (image retention notice accepted, T-15-38), 83d179a (HF re-smokes pass), 62401ff (claude-opus-5-5 re-smoke 12.7 s) |
| mid-plan | User decisions 2026-10-02: streaming HF adapter, qwen re-pin, DeepSeek V4.1-Flash, mllm-claude to claude-opus-5-5 | 492f4dc (RED streaming), 7639a61 (GREEN streaming), b8eb00e (RED re-pin/switch), 9b24f45 (GREEN re-pin/switch), 7406bea, 865e223 (decision and docs-check notes), e18c584 (RED Opus pin), be9f1d4 (GREEN Opus switch), 7dfaf0d (switch recorded) |
| 2 | Measured API latency and duty cycle in mllm_defaults.json and section 6; no flag flips; chunk_len unchanged | 87f4e24 (RED), 028cdb5 (GREEN) |
| 3 | Hardware checkpoint (operator 2026-10-03): depth live, URDF sign and gripper check, one episode per backend, Ctrl-C, hold path | 584f778 (chunk_len decision), c3a9c4b (depth), 61396d8 (URDF/gripper), 9e30275 (kimi), 67c1afe (claude), 34703d1 (deepseek and natural hold), ac6e865 (gpt), 6f7c4b5 (qwen), 9ac4472 (Ctrl-C fix), 5c7d77c (Ctrl-C, findings, episode re-derivation), 3f4d5e1 (section 6 parser fix) |

Pause checkpoint: 4c60479 (wip, Task 3 hardware checkpoint handoff).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] The Ctrl-C termination reason was lost when the piped stdout reader died**
- **Found during:** Task 3 step D, run 1 (mllm-hf-deepseek, control/outputs/mllm_ctrlc_001).
- **Issue:** The operator ran the episode piped through `| grep -v objc`, so Ctrl-C also sent SIGINT to grep. After the KeyboardInterrupt unwound through the open HTTP stream, the handler's print raised BrokenPipeError before `reason = "keyboard_interrupt"` ran. As a result, termination.json recorded `max_steps_reached`. The defect reproduced 3 of 3 times off hardware. Return-to-start was not affected, because it runs in the `finally` block.
- **Fix:** The reason is now set before the print, and the print is guarded. The regression test is `test_keyboard_interrupt_reason_survives_broken_stdout_pipe`. In the run 2 re-test, termination was `keyboard_interrupt`.
- **Files modified:** control/run_vla_episode.py, control/test_run_vla_episode.py
- **Commit:** 9ac4472

**2. [Rule 1 - Bug] The section 6 test parser broke when the episode re-derivation table was added**
- **Found during:** plan close, on the full-suite run after 5c7d77c.
- **Issue:** 5c7d77c added a second, episode-based table to section 6 of 15-LIVE-RESULTS.md. `_section_6_rows()` read every `| mllm-` row in the section, so the episode rows (whose latency cells hold several values, such as "179.7, 398.3") replaced the smoke rows that `mllm_defaults.json` mirrors. `test_measured_mllm_api_latency_record_matches_the_live_results_file` then raised ValueError.
- **Fix:** The parser now stops at the end of the first table in section 6. The full suite gives 1604 passed.
- **Files modified:** control/test_mllm_conformance.py
- **Commit:** 3f4d5e1

### Other deviations from the written steps

- **Hold path observed naturally, not provoked (step E).** mllm-hf-deepseek produced both an empty-response hold and a truncated hold (finish_reason length at max_tokens 32000) during its normal episode. That meets the "observed or provoked" criterion, so the temporary max_tokens 1024 edit was not made, and no config file needed reverting.
- **Two Ctrl-C runs (step D).** The plan asked for Ctrl-C during the first provider call. Run 1 pressed it during the second call, after a 20-step executed chunk, which is a stronger test because the arm had moved and had to return to start. Run 2, after the fix, pressed it during the first call as specified.
- **Model and provider changes (user decisions 2026-10-02).** Task 2 forbids changing model ids from evidence alone. The qwen re-pin, the DeepSeek model switch (which supersedes the locked D-23/D-27 id) and the mllm-claude switch to claude-opus-5-5 were each an explicit dated user decision, recorded in section 2 and in the config notes.
- **Streaming adapter (user decision 2026-10-02, Rule 1 bug).** The non-streaming HF call could not survive the router's ~120 s idle cut, even though timeout_s is 900. The fix is in 7639a61, with tests in 492f4dc.

## Issues Encountered

- First Claude smoke attempt: HTTP 400, because the operator's key was not workspace-scoped. The operator switched keys; no code change was needed.
- DeepSeek V4.1-Flash degeneration is not deterministic. The same scene produced a valid chunk in Ctrl-C run 1. This is recorded as a Phase 16 note (possibly a provider quantisation or sampling interaction).

## Follow-ups (not done here)

- **Docs:** the `mllm_inputs` preview example in `control/COMMANDS.md` uses `--wrist-camera-index 0`. On this machine the wrist camera is index 1, and index 0 is the FaceTime camera. Not edited in this plan.
- **Tooling:** `control/joint_jog.py` disconnects with torque disabled (lerobot `disable_torque_on_disconnect=True`), so the arm sags under gravity between jogs.
- **Config record:** `mllm_defaults.json` still holds the smoke-based Task 2 latency and duty-cycle values (user decision). The episode values are only in section 6 of the results file.
- **Phase 16 analysis notes:** DeepSeek degenerates when it tries to hand-compute FK at effort max. Kimi misread the scene and planned a likely sign-inverted "press down". Qwen's kinematics were coarse and it never used the wrist joints. Four of five backends are below a 10 percent duty cycle at chunk_len 20.

## User Setup Required

None remaining. The plan's `user_setup` block (ANTHROPIC_API_KEY, OPENAI_API_KEY, HF_TOKEN with Inference Providers credits, Anthropic and OpenAI credits, and the Colab FastFS /depth service) was completed by the operator and is recorded in 15-LIVE-RESULTS.md (sections 1 and 3, and the T-15-38 retention acceptance). For that reason `15-USER-SETUP.md` was intentionally skipped.

## Known Stubs

None.

## Threat Flags

None beyond the plan's threat register:
- **T-15-35:** keys were exported only in the shell, and the key-shape scans of the sidecars and the results file are clean.
- **T-15-36:** the validator hard limits were unchanged, and there were zero validator flags across the episodes. Ctrl-C and the hold path were verified on the real arm.
- **T-15-37:** spend was kept down with `--confirm-spend`, a 3-step smoke run cheapest first, and per-call cost estimates.
- **T-15-38:** the operator accepted the retention notice on 2026-10-02.

## Next

Phase 15 plan execution is complete (10 of 10). Phase acceptance is pending phase verification.

## Self-Check: PASSED

- FOUND: .planning/phases/15-mllm-style-pipeline-backends/15-LIVE-RESULTS.md, control/vla_bridge/backend_configs/mllm_defaults.json, control/run_vla_episode.py, control/test_run_vla_episode.py, control/test_mllm_conformance.py, control/vla_bridge/backends/base_mllm_chat.py
- FOUND commits (git log): e4022d1, 63fe4e5, 55d88bd, 83d179a, 62401ff, 492f4dc, 7639a61, b8eb00e, 9b24f45, 7406bea, 865e223, e18c584, be9f1d4, 7dfaf0d, 87f4e24, 028cdb5, 584f778, c3a9c4b, 61396d8, 9e30275, 67c1afe, 34703d1, ac6e865, 6f7c4b5, 9ac4472, 5c7d77c, 3f4d5e1, 4c60479
- Full suite at 3f4d5e1: 1604 passed.
