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
  - "None yet — Task 2 (checkpoint:decision) is a blocking human decision point this plan halted at; no checkpoint has been selected"

requirements-completed: []  # VLAHW-01 not yet complete — plan halted at Task 2's blocking human decision

coverage: []  # No deliverables to classify yet at this partial checkpoint — see coverage note below

# Metrics
duration: (partial — see Self-Check/Checkpoint below)
completed: 2026-09-21
status: paused-checkpoint
---

# Phase 11 Plan 03: SmolVLA Checkpoint Selection + Colab Bridge Docs Summary (PARTIAL — paused at Task 2)

**Task 1 fetched real `config.json` data for all 4 candidate SmolVLA checkpoints, revealing none actually matches this project's 2-camera rig — 3 expect 3 cameras (not 2, despite one's "_2_cameras" name), 1 expects 5, and one candidate 404s. Plan halted at Task 2's blocking `checkpoint:decision` for human checkpoint selection.**

## Performance

- **Duration:** partial (Task 1 only; Tasks 2-3 not yet executed)
- **Started:** 2026-09-21 (session start)
- **Tasks:** 1 of 3 completed
- **Files modified:** 2

## Accomplishments

- Built `control/vla_bridge/check_checkpoint.py`: fetches each of the 4 RESEARCH.md-listed candidate HF Hub checkpoints' `config.json` via `huggingface_hub.hf_hub_download`, extracts whatever `input_features`/`output_features` camera-key and action-dim info is actually present (no fixed schema assumed), and falls back to `HfApi().list_repo_files()` on fetch failure rather than crashing.
- Generated `control/vla_bridge/checkpoint_candidates.md` with real fetched data for all 4 candidates — **no candidate silently omitted**:
  - `victorvanhalst/smolvla_so101_cube`: **3** camera inputs (`camera1/2/3`), 6-DOF state/action — PARTIAL FIT (needs a 3rd feed)
  - `majinwakeup30/smolvla_so101_stack_cube_v3_2_cameras`: **also 3** camera inputs despite the repo name's "_2_cameras" claim — same PARTIAL FIT, and the identical `camera1/2/3` key naming/shape to `victorvanhalst`'s config suggests these may share a common training template — worth noting for the human decision
  - `lerobot/svla_so100_pickplace`: **404 / RepositoryNotFoundError** — this candidate does not exist at the given repo id (renamed, private, or never existed under `lerobot`'s namespace); cannot be selected as-is
  - `cn0303/smolvla-so101-strawberry-v3`: **5** camera inputs (3 real + 2 `empty_camera_*` dummy slots) — UNCLEAR FIT
- Explicitly documented the AR0144-stereo-is-one-physical-2560x720-frame caveat in the generated file, per acceptance criteria.
- **Halted at Task 2** (`checkpoint:decision`, `gate="blocking"`): per the executor's checkpoint protocol and this plan's own `<checkpoint_note>`, the human must select which checkpoint (or the ACT/pi0 fallback) Phase 11 uses, informed by this real fetched data (which materially changes the picture RESEARCH.md's `[ASSUMED]` table painted — no candidate is a clean 2-camera fit).

## Task Commits

1. **Task 1: Fetch and compare candidate SmolVLA checkpoints** - `4f1956f` (feat)

Tasks 2 (checkpoint:decision) and 3 (checkpoint:human-verify) not yet executed — this plan is paused at the Task 2 checkpoint pending human input.

## Files Created/Modified

- `control/vla_bridge/check_checkpoint.py` - fetches/compares 4 candidate SmolVLA checkpoints' HF Hub config.json against the real 2-camera rig
- `control/vla_bridge/checkpoint_candidates.md` - generated comparison table (data artifact, real fetched results)

## Decisions Made

None yet by a human — this is exactly what Task 2 is blocking on. Task 1's own findings are a material update to RESEARCH.md's assumptions: **all 3 successfully-fetched candidates expect either 3 or 5 cameras, none expects exactly 2**, and `lerobot/svla_so100_pickplace` (the highest-provenance-trust option per RESEARCH.md, published under the official `lerobot` org) does not resolve at all. This new information should inform, not just Task 2's raw options table, but potentially require the human to reconsider the ACT/pi0 fallback more seriously than RESEARCH.md initially framed it, or accept that a 3-camera checkpoint requires wiring a dummy/empty 3rd feed.

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

None - no external service configuration required for Task 1's work. Task 3 (not yet reached) will require a human `pyngrok` legitimacy check before any Colab-side install — see this plan's Task 3 definition.

## Next Phase Readiness

**Not ready to proceed to Plan 11-04 or 11-05.** This plan is paused at Task 2's blocking `checkpoint:decision`. The human must select an option (one of the 4 checkpoint ids, or the ACT/pi0 fallback) informed by Task 1's real fetched data above — notably that no candidate is a clean 2-camera match, and the previously highest-provenance-trust option (`lerobot/svla_so100_pickplace`) does not resolve. Once selected, a continuation agent will:
1. Record the selection in this SUMMARY.
2. Execute Task 3 (write `policy_server_launch.md`, embedding the selected checkpoint id) and halt again at its own `checkpoint:human-verify` (pyngrok legitimacy + TCP-tunnel/checkpoint-id correctness).
3. Only after Task 3's human "approved" is this plan's SUMMARY finalized as `status: complete`.

---
*Phase: 11-vla-hardware-connection*
*Completed (partial): 2026-09-21*
