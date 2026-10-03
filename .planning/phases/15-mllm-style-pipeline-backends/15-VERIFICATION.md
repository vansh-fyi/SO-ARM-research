---
phase: 15-mllm-style-pipeline-backends
verified: 2026-10-03T07:02:40Z
re_verified: 2026-10-03T07:06:54Z
status: passed
score: 16/16 must-haves verified
behavior_unverified: 0
overrides_applied: 0
re_verification:
  previous_status: gaps_found
  previous_score: 15/16
  head: f764200
  commits:
    - "13eab19 fix(15): make conformance checklist lookups checkbox-agnostic"
    - "f764200 docs(15): mark MLLM-04 complete and correct verification findings"
  gaps_closed:
    - "Plan 15-09: documentation matches the five-backend reality and the conformance suite asserts it (the full control/ suite is green at HEAD)"
  gaps_remaining: []
  regressions: []
gaps: []
deferred: []
---

# Phase 15: MLLM-Style Pipeline & Backends Verification Report

**Phase Goal:** A consistent MLLM-style `ActionSource` pattern (shared prompt construction, output JSON schema generated from `action_contract.py`, staged reasoning-trace capture) exists, and five registered raw-JSON-reasoning MLLM backends (three HuggingFace-hosted open reasoning-model variants (Qwen3.8-27B, DeepSeek-V4.1-Flash, Kimi-K3), Claude, and GPT) each drive the real SO-ARM101 through it with no pre-built movement primitives.
**Verified:** 2026-10-03T07:02:40Z (initial, HEAD 1679425); re-verified 2026-10-03T07:06:54Z at HEAD f764200
**Status:** passed
**Re-verification:** Yes, after gap closure (13eab19, f764200). See "Re-verification (HEAD f764200)" below.

## Re-verification (HEAD f764200)

Commits since the initial report (HEAD 1679425):

- **13eab19** `fix(15)`: `control/test_mllm_conformance.py` adds `_checklist_line(text, item)`, which matches `- [ ] item` or `- [x] item` and still asserts exactly one matching line. The MLLM-01, EVAL-01 and Phase 15 lookups use it. All content assertions are unchanged, so the tests are not weaker. Only this file changed in this commit.
- **f764200** `docs(15)`: REQUIREMENTS.md MLLM-04 is now `[x]`, and its traceability row says Complete. The 15-LIVE-RESULTS §5 Qwen row now records the 1 validator flag (step 21 gripper 100.0 -> 92.49). The 15-10-SUMMARY DeepSeek and validator-flag sentences are corrected. The `mllm_defaults.json` `not_measured` note records the 2026-10-03 user decision. The JSON still parses, and no config value changed. Only the note string differs.

Observed at f764200:

| Item from the initial report | Check | Result |
|------------------------------|-------|--------|
| Gap: suite red (2 failed / 1602 passed) | `cd control && .venv/bin/python -m pytest -q` (run once) | **1604 passed** in 15.05 s. Closed |
| MLLM-04 Pending in REQUIREMENTS.md | `grep MLLM-04 .planning/REQUIREMENTS.md` | `- [x] **MLLM-04**` and `| MLLM-04 | Phase 15 | Complete |`. Closed |
| LIVE-RESULTS Qwen "0 validator flags" | diff of f764200 | Now says 1 flag (step 21, matches episode.jsonl). Fixed |
| 15-10-SUMMARY DeepSeek overstatement | diff of f764200 | Now says four backends drove the arm in their episode and DeepSeek's executed chunk is in `mllm_ctrlc_001`. Fixed |
| mllm_defaults.json "pending a user decision" | diff of f764200 | Now records the 2026-10-03 keep-20 decision. Fixed |
| REQUIREMENTS.md MLLM-05 "Phase 1's" wording | unchanged (out of this phase's scope, per coordinator) | Still noted as Info |
| ROADMAP mvp-mode / user-story warning | unchanged (out of this phase's scope, per coordinator) | Still noted as a Warning |

No regressions. Between 1679425 and f764200 the only `control/` changes are the conformance test file and the note string in `mllm_defaults.json`. The backend, base, validator, logger and run-loop code is untouched, so the Truths 1-13, 15 and 16 evidence still holds.

## MVP-Mode Discrepancy (WARNING)

ROADMAP marks Phase 15 `**Mode:** mvp`, but the goal is not written as a user story. `gsd-tools query user-story.validate` returns `false`. The orchestrator explicitly asked for this verification, so I used the standard goal-backward method, as Phases 13 and 14 did. I derived the user flow below from the goal and the project's core value. To make the goal compliant, run `/gsd mvp-phase 15`.

## User Flow Coverage

Derived flow: «A researcher picks an MLLM backend, types a task prompt, and watches the real SO-ARM101 act on that model's raw joint-target JSON. Every decision is logged for later analysis.»

| Step | Expected | Evidence | Status |
|------|----------|----------|--------|
| Select backend | `--backend mllm-claude/mllm-gpt/mllm-hf-qwen/mllm-hf-deepseek/mllm-hf-kimi` accepted | `run_vla_episode.py --help` lists all five; `MLLM_BACKENDS` has exactly these 5, none in `GRPC_BRIDGE_BACKENDS` | ✓ |
| Prompt + inputs | Task, calibrated joint ranges, live joint state, wrist/left/right/depth images, URDF | `python -m vla_bridge.mllm_prompt` dump; `mllm_ctrlc_001/mllm_calls/0000/prompt.txt` (lines 14-19 calibrated ranges, 55-56 TASK + CURRENT JOINT STATE); `request_meta.json` image_roles `[wrist,left,right,depth]` | ✓ |
| Arm moves | Validated raw targets sent to the real arm | claude/gpt/qwen/kimi `_001` episodes: `@mllm-chunk` steps with 50/50 waypoint sends and changing joint_state; DeepSeek in `mllm_ctrlc_001` (20 `@mllm-chunk` steps, pan 4.7→35.4, gripper 98→21) | ✓ |
| Outcome logged | Per-chunk trace plus a per-step pointer in episode.jsonl, plus sidecars | every step has `action_details.chunk_ref`; refill ticks carry `mllm_call.trace` (task_restatement/plan/scene_claims/task_complete/native thinking); `mllm_calls/NNNN/` sidecars | ✓ |

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | SC1: The MLLM-style pattern (shared prompt, schema, trace capture) exists and is reused unchanged across all 5 registered backends | ✓ VERIFIED | All 5 backend modules are thin `@register_mllm_backend` + `build_mllm_backend()` calls (`mllm_hf_*.py` are 40-44 lines; Claude/GPT add only an adapter). `MllmActionSourceBase.__init__` builds the system prompt and schema once (`base_mllm.py:545-547`). `test_success_criterion_1_prompt_schema_version_and_chunk_len_are_identical` and `test_one_real_chunk_per_backend_sends_byte_identical_payloads` pass. Live check: all 5 hardware episodes record the same `schema_sha256 2e5e5f09…` and `prompt_sha256 9cdcf2c5…` |
| 2 | SC2: Each HF variant, Claude and GPT produce a schema-valid action chunk from raw instruction + calibration + joint state + camera frames, with no movement-primitive functions in prompt or code path | ✓ VERIFIED | Live, real arm: Claude 20/20 valid steps then task_complete (pen lifted); GPT 2x20 valid; Qwen 2x20 valid; Kimi 2x20 valid; DeepSeek: `mllm_ctrlc_001` chunk 0 `parse.ok True, n_valid 20`, stop complete, model_returned `deepseek-ai/DeepSeek-V4.1-Flash`, 20 steps executed with 50/50 sends each and real joint motion. Its scored episode `_001` produced no chunk (empty-response, truncated), but the SC asks for a schema-valid chunk driving the arm, and ctrlc_001 shows exactly that on the same HEAD-equivalent backend code (the only later code change is the KeyboardInterrupt handler, 9ac4472). Smoke PASS for all five as well. No primitives: the prompt's CONTROL section says "no helper routines"; `lint_prompt_text` checks PRIMITIVE_TERMS; conformance scans find no tools/tool_choice/eval/exec. The adapters forward only `request.schema` through native JSON-schema mode |
| 3 | SC3: The schema is generated from `action_contract.JOINT_ORDER`/units (not hand-duplicated), and a malformed action in a chunk holds position instead of crashing or zero-filling | ✓ VERIFIED | `mllm_schema.build_action_chunk_schema` builds step properties with a comprehension over `JOINT_ORDER` using `ACTION_UNITS[joint]`. No joint list is hand-written in any MLLM module or config (grep). `parse_chunk`/`_validate_step` hold the bad step and the rest of the chunk with the live `joint_state`. Tests: `test_malformed_step_holds_live_state_for_rest_of_chunk_without_retry` (parametrised over missing/non-numeric/null/bool/NaN/overflow) passes. Live: DeepSeek `_001` held all 6 joints constant for 40 ticks (empty-response, truncated) |
| 4 | SC4: Each backend's staged trace (restate → plan → grounded scene claims → action) is captured in io_logger alongside its chunk, per step | ✓ VERIFIED | Full trace written once per chunk in `last_details["mllm_call"]` (D-12, user-locked); every tick carries `chunk_ref {chunk_id, step_index, chunk_len}`; `run_episode` passes `details` as `action_details` to the UNMODIFIED `io_logger.write_step`. Live: 0 steps without chunk_ref in all 7 episode files; refill ticks have task_restatement/plan/scene_claims (structured `{claim, source, confidence}`)/task_complete/native_thinking_text. `test_slice_trace_written_once_per_chunk` passes |
| 5 | SC5: Chunk size used by all 5 backends is derived from Phase 12 measured latency data and documented | ✓ VERIFIED | `mllm_defaults.json` `chunk_len 20` with `chunk_len_derivation` from Phase 12 measurements (tick 0.785 s n=236, depth round trip 6.15 s n=26). I re-derived the tick from `outputs/vla_episode_004` + `vla_episode_20260930_121123`: n=236, medians 0.790/0.783 s. No backend config overrides chunk_len. Measured MLLM latencies and duty cycles are recorded. The user re-confirmed N=20 on 2026-10-03 (LIVE-RESULTS §6) |
| 6 | 15-01: Fake provider drives the REAL run_episode; provider called once per chunk_len ticks, never per tick | ✓ VERIFIED | `test_slice_e2e_chunk_drain_through_real_run_episode` passes. Live `latency_ms.api` appears only on refill ticks (steps 0 and 20) |
| 7 | 15-03: Whole-call failures give chunk_len live-state holds; 3 consecutive failures end the episode `mllm_backend_failed` | ✓ VERIFIED | `base_mllm._count_call_outcome` and `_enqueue_holds`; `test_three_consecutive_failed_calls_end_the_episode` and `test_whole_call_failure_holds_live_state_for_whole_chunk_then_calls_again` pass. Live: DeepSeek `consecutive_failed_calls 2` (< cap) |
| 8 | 15-03: Boolean-true task_complete holds the whole chunk and ends the episode; the claim is logged separately | ✓ VERIFIED | `test_task_complete_true_holds_the_whole_chunk_and_ends_the_episode` passes. Live Claude: step 20 `task-complete-holding-position`, termination `task_complete` (21 steps) |
| 9 | 15-03: run_vla_episode.py gets only additive edits (termination hook, MLLM-scoped depth rule); the shared loop, validator and logger are unchanged | ✓ VERIFIED | `git diff 382d78d^..HEAD` shows `safety_validator.py`, `io_logger.py`, `action_contract.py` with 0 changes. run_vla_episode.py: +44/-7 (docstring, the `episode_complete_reason` hook, two `not in base_mllm.MLLM_BACKENDS` depth guards, the 9ac4472 KeyboardInterrupt reason-first fix) |
| 10 | 15-04: Per-call sidecars (raw payload, prompt, images, thinking) under `mllm_calls/NNNN/`; secrets redacted | ✓ VERIFIED | Sidecar dirs exist for every live call (prompt.txt, request_meta.json, response_raw.json, thinking.txt, images/, depth_heatmap.png, depth_grid.txt, depth.npy, aligned_left.png). Key-shape regex scan over all 7 live output dirs: no match. `test_mllm_record.py` passes |
| 11 | 15-05: Maximum input (wrist/left/right + depth heatmap + 24x32 grid, one depth call per chunk) plus URDF kinematics text | ✓ VERIFIED | Prompt dump shows the INPUTS/depth/URDF sections. Live request_meta image_roles `[wrist,left,right,depth]`. LIVE-RESULTS §3 depth live ~6-7 s, no 429 |
| 12 | 15-06/07/08: Native structured output (Claude output_config json_schema, OpenAI text.format strict, HF response_format strict), no tool-use, lazy SDK, provider pins | ✓ VERIFIED | `mllm_claude.py:266`, `mllm_gpt.py:305`, `base_mllm_chat.py:485`. The reserved extra_body keys block tools/tool_choice. Configs pin deepinfra/deepinfra/baseten. Conformance hygiene tests pass |
| 13 | 15-09: Exactly five MLLM backends registered, none in GRPC_BRIDGE_BACKENDS, no base_ module registered | ✓ VERIFIED | Runtime check: `['mllm-claude','mllm-gpt','mllm-hf-deepseek','mllm-hf-kimi','mllm-hf-qwen']`, overlap with GRPC set `[]` |
| 14 | 15-09: Documentation matches reality and the conformance suite asserts it (suite green) | ✓ VERIFIED (re-verified) | Initially FAILED at 1679425 (checkbox-prefix lookup). Fixed by 13eab19 (`_checklist_line`). At f764200 the full suite gives 1604 passed |
| 15 | 15-10: Ctrl-C during a long call returns the arm to start with termination reason keyboard_interrupt | ✓ VERIFIED | `mllm_ctrlc_002/termination.json` = `{"reason":"keyboard_interrupt","steps_completed":0}`. `test_keyboard_interrupt_reason_survives_broken_stdout_pipe` passes. ctrlc_001 (before the fix) shows the defect that was fixed (`max_steps_reached`). Return-to-start runs in the `finally` block |
| 16 | 15-10: Live smoke for all five, URDF sign and gripper check, hold path observed, all recorded | ✓ VERIFIED | 15-LIVE-RESULTS.md has no PENDING cells. §1 PASS rows for all five. §4 all joints and gripper match. §5 natural hold on DeepSeek |

**Score:** 16/16 truths verified (0 present-but-behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `control/vla_bridge/mllm_schema.py` | Schema from action_contract, hash, subset lint | ✓ VERIFIED | 156 lines; loops over JOINT_ORDER/ACTION_UNITS; no bounds |
| `control/vla_bridge/mllm_prompt.py` | Shared system prompt, user text, lint, dump CLI | ✓ VERIFIED | Joint section generated from JOINT_ORDER + limits; CLI runs keyless |
| `control/vla_bridge/backends/base_mllm.py` | Chunk-drain base, parse/hold, registration, failure cap, task_complete | ✓ VERIFIED | 915 lines, substantive; used by all 5 via builder |
| `control/vla_bridge/backends/base_mllm_build.py` | Single hardware builder | ✓ VERIFIED | Fails closed before robot connect; constructs collector, recorder, URDF and base |
| `control/vla_bridge/backends/base_mllm_chat.py` | Shared HF chat adapter (streaming) | ✓ VERIFIED | Streaming fix after the router 504; used by all 3 HF variants |
| `control/vla_bridge/backends/mllm_{claude,gpt,hf_qwen,hf_deepseek,hf_kimi}.py` | 5 registered backends | ✓ VERIFIED | All registered and live-exercised |
| `control/vla_bridge/backend_configs/mllm_*.json` (6) | Defaults + per-backend configs | ✓ VERIFIED | env-var names only, no keys |
| `control/vla_bridge/mllm_record.py`, `mllm_inputs.py`, `mllm_urdf.py`, `mllm_smoke.py` | Recorder, inputs, URDF text, smoke CLI | ✓ VERIFIED | Smoke refuses to spend without `--confirm-spend` (checked) |
| `control/test_mllm_conformance.py` | Cross-backend conformance + docs assertions | ✓ VERIFIED | Checkbox-agnostic lookups since 13eab19. All tests pass |
| `15-LIVE-RESULTS.md` | Live evidence | ✓ VERIFIED | Matches the on-disk episode outputs (minor wording issues under Anti-Patterns) |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| base_mllm.py | mllm_schema.py + mllm_prompt.py | built once in `__init__`, passed in every ProviderRequest | WIRED | `base_mllm.py:545-547, 686-694` |
| run_episode | MllmActionSourceBase.get_action | ActionSource Protocol; last_details → `io_logger.write_step(action_details=…)` | WIRED | Live chunk_ref on every step |
| run_episode | safety_validator.validate_action | every returned action, MLLM or hold | WIRED | Live Qwen step 21: raw gripper 100.0 clamped to 92.49 by the unmodified validator |
| run_episode | `episode_complete_reason` | getattr hook after each tick | WIRED | Live Claude `task_complete` termination |
| run_vla_episode.main | base_mllm.MLLM_BACKENDS | depth-flag scoping | WIRED | two membership guards |
| base_mllm._refill | CallRecorder.finalize | per-call hook in try/except | WIRED | sidecars present per call |
| mllm_*.py build | base_mllm_build.build_mllm_backend | spec lookup by name | WIRED | all 5 |
| mllm_hf_*.py | base_mllm_chat.make_chat_adapter | adapter_factory | WIRED | all 3 |
| mllm_smoke.py | MLLM_BACKENDS adapter_factory | same factories as live | WIRED | smoke PASS rows |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| episode.jsonl `action_details.mllm_call.trace` | staged fields | provider JSON parsed by `parse_chunk` | Yes: real model text in all live episodes (e.g. Claude claim "pen … clamped between the two yellow gripper jaws") | ✓ FLOWING |
| ProviderRequest.images | wrist/left/right/depth bytes | MllmInputCollector (real cameras + FastFS) | Yes: distinct sha256 per role and byte sizes 117-440 KB in ctrlc_001 | ✓ FLOWING |
| user_text joint state | `joint_state` | `robot.get_observation()` via run_episode | Yes: `CURRENT JOINT STATE` in prompt.txt matches episode step 0 joint_state | ✓ FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Full suite | `cd control && .venv/bin/python -m pytest -q` (run once per HEAD) | 1679425: 2 failed, 1602 passed. **f764200: 1604 passed** | ✓ PASS (at f764200) |
| Named behavioral tests (Ctrl-C reason, SC1 identity, validator clamp, schema generation, malformed hold, trace once per chunk, latency record) | `pytest -q -k "<7 names>"` | 15 passed | ✓ PASS |
| Registry contents | python import of MLLM_BACKENDS / GRPC_BRIDGE_BACKENDS | 5 names, 0 overlap | ✓ PASS |
| Prompt dump keyless | `python -m vla_bridge.mllm_prompt` | prompt + schema + hashes printed | ✓ PASS |
| Smoke spend guard | `python -m vla_bridge.mllm_smoke --backend mllm-claude …` without `--confirm-spend` | "Not calling: re-run with --confirm-spend" | ✓ PASS |
| Live episode outputs | parsed all `outputs/mllm_mllm-*_001`, `mllm_ctrlc_00*` episode.jsonl / termination.json | as described in truths 2-4, 8, 15 | ✓ PASS |

### Probe Execution

Step 7c: SKIPPED. There are no `scripts/*/tests/probe-*.sh` files and no probe is declared in any Phase 15 plan.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| PIPE-02 | 15-01,03,04,05,06,07,08,09,10 | Consistent MLLM ActionSource pattern, no shared-loop/safety/logging change | ✓ SATISFIED | Truths 1, 9 |
| MLLM-01 | 15-02,08,09,10 | HF open reasoning variants (Qwen3.8-27B, DeepSeek-V4.1-Flash, Kimi-K3) on one shared adapter | ✓ SATISFIED | 3 variants on `base_mllm_chat`; all live (DeepSeek via ctrlc_001 + smoke) |
| MLLM-02 | 15-02,06,09,10 | Claude backend | ✓ SATISFIED | mllm-claude live pick |
| MLLM-03 | 15-02,07,09,10 | GPT backend | ✓ SATISFIED | mllm-gpt live 2 chunks |
| MLLM-04 | 15-01,06,09 | Schema generated from action_contract constants and validated through the unmodified safety_validator | ✓ SATISFIED (REQUIREMENTS.md `[x]` / Complete since f764200) | Schema comprehension over JOINT_ORDER/ACTION_UNITS. safety_validator.py has 0 changes over the phase. Every MLLM action goes through `validate_action` in run_episode (`test_out_of_range_value_is_logged_raw_and_clamped_only_by_the_validator`; live Qwen clamp at step 21). The earlier Pending status was a bookkeeping omission (MLLM-04 was missing from 15-10's list). f764200 fixed it |
| MLLM-05 | 15-01,09,10 | Chunk size derived from measured latency, not guessed | ✓ SATISFIED | Truth 5 (REQUIREMENTS text says "Phase 1's"; the evidence is Phase 12) |
| MLLM-06 | 15-01,03,04,09,10 | Staged trace captured alongside chunk, per step | ✓ SATISFIED | Truth 4 |
| MLLM-07 | 15-01,03,09,10 | Malformed action holds position, never zero-fill or crash | ✓ SATISFIED | Truth 3, 7 |

Orphaned requirements: none. REQUIREMENTS.md maps exactly PIPE-02 and MLLM-01..07 to Phase 15, and every one of them appears in at least one plan's `requirements` field. DEPTH-CAL-03 (Phase 12) carries the Phase 15 note required by D-06, and the depth client is still never added to a PolicyServer observation dict (`depth_client = None` path for MLLM backends).

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| control/test_mllm_conformance.py | 682, 727 (at 1679425) | Test pinned the mutable `- [ ]` checkbox state | Resolved (was 🛑 Blocker) | Fixed by 13eab19, suite green |
| (all phase-modified control files) | — | TBD/FIXME/XXX/TODO/HACK | none found | — |
| control/vla_bridge/backend_configs/mllm_defaults.json | `not_measured` | Stale "pending a user decision" text | Resolved (was Info) | f764200 records the 2026-10-03 decision |
| 15-10-SUMMARY.md | 149 | Overstated DeepSeek episode | Resolved (was Info) | f764200 |
| 15-LIVE-RESULTS.md | §5 qwen row | "0 validator flags" | Resolved (was Info) | f764200 records the step-21 clamp |
| .planning/REQUIREMENTS.md | MLLM-04 row | Pending | Resolved | f764200 |
| .planning/REQUIREMENTS.md | MLLM-05 text | Says "Phase 1's" (evidence is Phase 12) | ℹ️ Info (left as is, out of scope per coordinator) | Wording only |
| .planning/ROADMAP.md | Phase 15 | `Mode: mvp` but the goal is not a user story | ⚠️ Warning (left as is, out of scope per coordinator) | Run `/gsd mvp-phase 15` if wanted |

### Human Verification Required

None outstanding. All live items (smoke, depth, URDF and gripper sign, hardware episodes, hold path, Ctrl-C) are recorded in 15-LIVE-RESULTS.md and match the on-disk outputs.

Note for Phase 16 (not a Phase 15 gap): DeepSeek-V4.1-Flash on real scenes gave a valid chunk in 1 of 3 calls (ctrlc_001 chunk 0). The other two degenerated (empty content; truncated at max_tokens 32000). The pipeline handled both correctly with holds. Reliability at effort `max` / temperature 1.0 should be an explicit Phase 16 ablation. Four of five backends also sit below a 10% motion duty cycle at chunk_len 20 with real images, and the user accepted that on 2026-10-03.

### Gaps Summary

No gaps remain. The phase goal is achieved:

- The shared MLLM pattern exists and is used unchanged by all five registered backends.
- The schema is generated from `action_contract.py`.
- Malformed or failed output holds the live position.
- The staged trace is logged per chunk, with a per-step pointer, through the unmodified io_logger.
- All five backends executed at least one schema-valid, validator-checked chunk on the real SO-ARM101 with no movement primitives. For DeepSeek that chunk is in `mllm_ctrlc_001`.

The one gap from the initial verification (conformance tests pinned to the unchecked checkbox) was closed by 13eab19. The suite is 1604 passed at f764200. The bookkeeping and wording items were corrected in f764200.

The open items, both out of this phase's scope by coordinator decision, are the REQUIREMENTS.md MLLM-05 "Phase 1's" wording (Info) and the ROADMAP mvp-mode / user-story warning. Neither affects goal achievement.

---

_Verified: 2026-10-03T07:02:40Z; re-verified 2026-10-03T07:06:54Z at f764200_
_Verifier: Claude (gsd-verifier)_
