---
phase: 11-vla-hardware-connection
plan: 03
subsystem: vla-hardware
tags: [vla-hardware, smolvla, huggingface-hub, checkpoint-selection, colab-bridge]

# Dependency graph
requires: []
provides:
  - "control/vla_bridge/check_checkpoint.py — fetches and compares candidate SmolVLA checkpoints' config.json against this project's real 2-camera rig"
  - "control/vla_bridge/checkpoint_candidates.md — real fetched data replacing RESEARCH.md's 4 [ASSUMED] candidates"
affects: [11-04, 11-05]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "HF Hub config.json fetch-with-fallback: hf_hub_download() first, HfApi().list_repo_files() fallback on any exception, never crash on a missing/inaccessible checkpoint"
    - "Markdown table cell sanitization: collapse embedded newlines and escape literal `|` before writing any dynamically-fetched error text into a table row"

key-files:
  created:
    - control/vla_bridge/check_checkpoint.py
    - control/vla_bridge/checkpoint_candidates.md
  modified: []

key-decisions:
  - "Task 2 decision: selected `victorvanhalst/smolvla_so101_cube` as the checkpoint to drive the real robot — exact task-instruction match to D-02's red-cube task ('Pick the red cube and place it in the bowl'), and its 3-camera-slot config.json is resolved against this project's real 2-physical-camera rig via a 3-genuine-feed split-stereo mapping (wrist=camera1, stereo-left-half=camera2, stereo-right-half=camera3), not a dummy/empty 3rd slot"

requirements-completed: []  # VLAHW-01 not yet complete — plan now paused at Task 3's blocking checkpoint:human-verify (pyngrok legitimacy)

coverage: []  # No deliverables to classify yet at this partial checkpoint — see coverage note below

# Metrics
duration: (partial — see Self-Check/Checkpoint below)
completed: 2026-09-21
status: paused-checkpoint
---

# Phase 11 Plan 03: SmolVLA Checkpoint Selection + Colab Bridge Docs Summary (PARTIAL — paused at Task 3)

**Task 1 fetched real `config.json` data for all 4 candidate SmolVLA checkpoints, revealing none actually matches this project's 2-camera rig as a plain 1:1 mapping — 3 expect 3 cameras (not 2, despite one's "_2_cameras" name), 1 expects 5, and one candidate 404s. The human selected `victorvanhalst/smolvla_so101_cube` at Task 2's `checkpoint:decision`, resolving the 3-camera-slot mismatch via a split-stereo mapping rather than a dummy feed (wrist=camera1, stereo-left-half=camera2, stereo-right-half=camera3). Task 3 is now drafting `policy_server_launch.md` embedding that checkpoint and camera mapping, then will halt at its own blocking `checkpoint:human-verify` (pyngrok legitimacy + TCP-tunnel/checkpoint-id correctness).**

## Performance

- **Duration:** partial (Task 1 complete, Task 2 decided, Task 3 in progress)
- **Started:** 2026-09-21 (session start)
- **Tasks:** 1 of 3 fully completed (Task 2's decision is recorded; Task 3's file/checkpoint not yet reached)
- **Files modified:** 2

## Accomplishments

- Built `control/vla_bridge/check_checkpoint.py`: fetches each of the 4 RESEARCH.md-listed candidate HF Hub checkpoints' `config.json` via `huggingface_hub.hf_hub_download`, extracts whatever `input_features`/`output_features` camera-key and action-dim info is actually present (no fixed schema assumed), and falls back to `HfApi().list_repo_files()` on fetch failure rather than crashing.
- Generated `control/vla_bridge/checkpoint_candidates.md` with real fetched data for all 4 candidates — **no candidate silently omitted**:
  - `victorvanhalst/smolvla_so101_cube`: **3** camera inputs (`camera1/2/3`), 6-DOF state/action — PARTIAL FIT (needs a 3rd feed)
  - `majinwakeup30/smolvla_so101_stack_cube_v3_2_cameras`: **also 3** camera inputs despite the repo name's "_2_cameras" claim — same PARTIAL FIT, and the identical `camera1/2/3` key naming/shape to `victorvanhalst`'s config suggests these may share a common training template — worth noting for the human decision
  - `lerobot/svla_so100_pickplace`: **404 / RepositoryNotFoundError** — this candidate does not exist at the given repo id (renamed, private, or never existed under `lerobot`'s namespace); cannot be selected as-is
  - `cn0303/smolvla-so101-strawberry-v3`: **5** camera inputs (3 real + 2 `empty_camera_*` dummy slots) — UNCLEAR FIT
- Explicitly documented the AR0144-stereo-is-one-physical-2560x720-frame caveat in the generated file, per acceptance criteria.
- **Task 2 decision resolved** (`checkpoint:decision`, `gate="blocking"`): the human selected `victorvanhalst/smolvla_so101_cube` — the exact task-instruction match to D-02's red-cube task ("Pick the red cube and place it in the bowl") — over the other 3 candidates and the ACT/pi0 fallback. The candidate's 3-camera-slot config.json (`camera1`/`camera2`/`camera3`, none marked `empty_camera_*`) is resolved against this project's real 2-physical-camera rig by treating the AR0144's single 2560x720 stereo frame as **two distinct real feeds** (left half + right half) rather than one wide image or a dummy slot, giving 3 genuinely distinct real camera views total: `camera1`=IMX335 wrist, `camera2`=AR0144 stereo-left (columns `[0:1280]`), `camera3`=AR0144 stereo-right (columns `[1280:2560]`). This pattern was corroborated by researching community SmolVLA fine-tunes with similar 2-real-camera + inherited-3rd-slot conventions (`bklassen3434/smolvla_pick_pen_v2_frozen`, `Yilin1001/smolvla-pen-v2-030000`); this project's case is an even better fit since all 3 slots map to genuinely distinct real views, not a dummy.

## Task Commits

1. **Task 1: Fetch and compare candidate SmolVLA checkpoints** - `4f1956f` (feat)
2. **Task 2: Record checkpoint decision** - (this commit, docs)

Task 3 (checkpoint:human-verify) not yet executed — proceeding to draft `policy_server_launch.md` next, then this plan will halt at Task 3's own blocking checkpoint pending human pyngrok-legitimacy sign-off.

## Files Created/Modified

- `control/vla_bridge/check_checkpoint.py` - fetches/compares 4 candidate SmolVLA checkpoints' HF Hub config.json against the real 2-camera rig
- `control/vla_bridge/checkpoint_candidates.md` - generated comparison table (data artifact, real fetched results)

## Decisions Made

**Task 2 (`checkpoint:decision`) — resolved:** the human selected **`victorvanhalst/smolvla_so101_cube`** (option id `victorvanhalst-cube`) over `majinwakeup30-stack`, `svla-so100-pickplace` (404s, cannot be selected as-is), `strawberry`, and the `act-fallback`. Rationale: exact task-instruction match to D-02's red-cube task ("Pick the red cube and place it in the bowl") — since this phase does not fine-tune anything itself, a checkpoint's own trained task determines what the VLA actually knows how to do, making task-semantic match the deciding factor over raw camera-count fit.

The candidate's 3-camera-slot mismatch (Task 1 found `camera1`/`camera2`/`camera3`, none marked `empty_camera_*`) is resolved not by a dummy/empty 3rd feed but by using **3 genuinely distinct real camera views**: this project's AR0144 "overhead" camera is one physical 2560x720 side-by-side stereo frame (confirmed in `diagnostics/UAT/function/basic/UAT.md`), which can be split into independent left/right halves — `camera1`=IMX335 wrist (as-is), `camera2`=AR0144 stereo-left (crop columns `[0:1280]`), `camera3`=AR0144 stereo-right (crop columns `[1280:2560]`). This is a better structural fit than a dummy slot: all 3 of the checkpoint's expected inputs receive genuinely distinct real image data, none is a zero/black placeholder. The pattern of pairing a real-hardware 2-camera rig with a checkpoint's 3-camera-slot config via a similar real/inherited-slot split is independently corroborated by other community SmolVLA fine-tunes (`bklassen3434/smolvla_pick_pen_v2_frozen`, `Yilin1001/smolvla-pen-v2-030000`), though those use a dummy 3rd slot where this project uses a genuinely distinct 3rd real feed (split-stereo) instead — an even better fit.

This also confirms Task 1's own findings remain valid as a material update to RESEARCH.md's assumptions: **all 3 successfully-fetched candidates expect either 3 or 5 cameras, none expects exactly 2**, and `lerobot/svla_so100_pickplace` (the highest-provenance-trust option per RESEARCH.md, published under the official `lerobot` org) does not resolve at all — but the split-stereo mapping resolves the apparent 2-vs-3-camera mismatch without needing the ACT/pi0 fallback.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Markdown table cell corruption from embedded newlines in fetch-failure error text**
- **Found during:** Task 1 (first script run against `lerobot/svla_so100_pickplace`, which 404s)
- **Issue:** The initial script implementation wrote raw multi-line HTTP error text (containing literal `\n` characters) directly into a markdown table cell, which broke the table's row structure for every row rendered after the failed candidate.
- **Fix:** Added an `_oneline()` helper that collapses embedded newlines to single spaces and escapes literal `|` characters before any dynamically-fetched text is written into a table cell; applied to every cell value, not just the error-text one, since any future candidate's fetched data could in principle contain either character.
- **Files modified:** `control/vla_bridge/check_checkpoint.py`
- **Verification:** Re-ran the script; confirmed `checkpoint_candidates.md`'s table renders as one line per row with no broken structure, and the `lerobot/svla_so100_pickplace` row's full error text is preserved (collapsed to one line) rather than truncated or omitted.
- **Committed in:** `4f1956f` (part of Task 1 commit — caught and fixed before the task's own verification step, not a follow-up commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Necessary correctness fix caught during the task's own verify step, before commit. No scope creep — same script, same output file, just correctly formatted.

## Issues Encountered

- This worktree had no `control/.venv` (gitignored, not part of the git-tracked worktree checkout) — created a fresh `python3.12 -m venv control/.venv` and installed `huggingface_hub==1.29.0` (matching the version RESEARCH.md verified in the main checkout) so Task 1's script could run. This is local tooling setup, not a plan deviation — no plan files were changed for this, and `control/.venv/` remains gitignored.
- HF Hub requests ran unauthenticated (no `HF_TOKEN` set) — worked fine for the 3 public checkpoints; the `lerobot/svla_so100_pickplace` 404 is a genuine "repository not found" from HF Hub's API, not an auth-wall artifact (both the direct 401 message and the subsequent file-listing fallback attempt independently returned "Repository Not Found").

## User Setup Required

None yet for Task 2 (a decision, not an install). Task 3 (in progress) requires a human `pyngrok` legitimacy check before any Colab-side install — see this plan's Task 3 definition; this plan will halt at that checkpoint next.

## Next Phase Readiness

**Not yet ready to proceed to Plan 11-04 or 11-05.** Task 2's decision is now resolved (`victorvanhalst/smolvla_so101_cube`, 3-real-camera split-stereo mapping). Proceeding now to Task 3: draft `control/vla_bridge/policy_server_launch.md` embedding this checkpoint id and camera mapping, then halt at Task 3's own blocking `checkpoint:human-verify` (pyngrok legitimacy + TCP-tunnel/checkpoint-id correctness). Only after a human types "approved" (or specifies the cloudflared fallback) is this plan's SUMMARY finalized as `status: complete`.

## Self-Check: PASSED

- FOUND: `control/vla_bridge/check_checkpoint.py`
- FOUND: `control/vla_bridge/checkpoint_candidates.md`
- FOUND: `.planning/phases/11-vla-hardware-connection/11-03-SUMMARY.md`
- FOUND commit: `4f1956f` (Task 1)
- FOUND commit: `dabc3b1` (partial SUMMARY)
- FOUND commit: `42611f7` (self-check append)

---
*Phase: 11-vla-hardware-connection*
*Completed (partial): 2026-09-21*
