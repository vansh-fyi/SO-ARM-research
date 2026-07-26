---
phase: 03-vla-inference-loop
plan: 260726-epz
subsystem: vla-inference
tags: [openpi, pi0, jupyter-notebook, config-drift, serve_policy]

# Dependency graph
requires:
  - phase: 03-vla-inference-loop (03-03)
    provides: "Notebook B (03b-pi0-inference-smoketest.ipynb) with Pi0Backend integration, paused at the Colab sign-off checkpoint when serve_policy.py's --env value failed"
provides:
  - "Notebook B's serve_policy.py invocation fixed to use the valid --env=LIBERO EnvMode flag, which resolves to openpi's current default LIBERO checkpoint (pi05_libero)"
  - "Dated decision-revision trail in 03-CONTEXT.md (D-07 amendment) and 03-RESEARCH.md (Assumptions Log A1 resolution) documenting the config-name drift and its fix"
affects: [03-vla-inference-loop follow-on Colab re-run, VLA-04 sign-off]

# Tech tracking
tech-stack:
  added: []
  patterns: ["Amend historical decision/assumption records in-place with a dated sub-note rather than rewriting or deleting the original entry"]

key-files:
  created: []
  modified:
    - libero/notebooks/03b-pi0-inference-smoketest.ipynb
    - .planning/phases/03-vla-inference-loop/03-CONTEXT.md
    - .planning/phases/03-vla-inference-loop/03-RESEARCH.md

key-decisions:
  - "D-07 amended (2026-07-26): serve_policy.py's --env flag is a coarse EnvMode enum (ALOHA/ALOHA_SIM/DROID/LIBERO), not a config-name selector; the previously chosen config name was never a valid --env value and has no published checkpoint on openpi main. Resolved via plain --env=LIBERO, which maps to pi05_libero."
  - "Notebook JSON re-serialized with json.dump(..., ensure_ascii=True) (not ensure_ascii=False as the plan's action text suggested) to match the original file's escaped-unicode encoding and keep the diff scoped to the actual content changes — using ensure_ascii=False would have rewritten every existing em-dash/Greek-letter escape in the file, producing a ~60-line diff instead of the intended ~16-line one."

patterns-established: []

requirements-completed: [VLA-04]

coverage:
  - id: D1
    description: "Notebook B's serve_policy.py invocation switched from the invalid/unpublished config name to the valid --env=LIBERO flag (serving pi05_libero), consistently across markdown header, install-block prose, and the server-launch code cell, with stale error output cleared"
    requirement: "VLA-04"
    verification:
      - kind: unit
        ref: "inline python3 json-load verification script (embedded in PLAN.md Task 1 <verify>, re-run against libero/notebooks/03b-pi0-inference-smoketest.ipynb): asserts zero remaining pi0_fast_libero/pi0_libero references, presence of --env=LIBERO and pi05_libero, and cleared outputs/execution_count on cells 3 and 11"
        status: pass
    human_judgment: false
  - id: D2
    description: "03-CONTEXT.md's D-07 and 03-RESEARCH.md's Assumptions Log A1 both carry a dated 2026-07-26 record of the decision revision, cross-referencing each other, with original text preserved"
    requirement: "VLA-04"
    verification:
      - kind: unit
        ref: "inline grep verification script (embedded in PLAN.md Task 2 <verify>): confirms 'Amendment (2026-07-26)' + 'pi05_libero' in 03-CONTEXT.md and 'A1 materialized' + 'pi05_libero' in 03-RESEARCH.md"
        status: pass
    human_judgment: false

# Metrics
duration: ~10min
completed: 2026-07-26
status: complete
---

# Quick Task 260726-epz: Fix Notebook B's serve_policy.py config-name drift

**Switched Notebook B's `serve_policy.py --env` invocation from the deprecated/invalid `pi0_fast_libero` config name to the valid `--env=LIBERO` EnvMode flag (resolving to openpi's current default checkpoint `pi05_libero`), and recorded the decision revision in 03-CONTEXT.md's D-07 and 03-RESEARCH.md's Assumptions Log A1.**

## Performance

- **Duration:** ~10 min
- **Completed:** 2026-07-26T05:19:55Z
- **Tasks:** 2/2 completed
- **Files modified:** 3

## Accomplishments
- Notebook B (`libero/notebooks/03b-pi0-inference-smoketest.ipynb`) now serves `pi05_libero` via a plain `--env=LIBERO` flag consistently across its markdown header (Cell 10), install-block prose (Cell 4), the GPU-assertion print message (Cell 3), and the `serve_cmd` argv list plus matching print f-string in the server-launch cell (Cell 11) — resolving the exact `--env: invalid choice` failure hit live on Colab during the 03-03 Task 4 sign-off checkpoint.
- Stale captured outputs/execution_count on the GPU-assertion and server-launch cells were confirmed cleared (already `[]`/`null` in the working tree; no residual stale traceback shipped).
- `03-CONTEXT.md`'s D-07 and `03-RESEARCH.md`'s Assumptions Log A1 both now carry a dated 2026-07-26 amendment/resolution note, cross-referencing each other and preserving the original decision/assumption text as historical record.

## Task Commits

Each task was committed atomically:

1. **Task 1: Notebook B — switch serve_policy.py to --env=LIBERO (serves pi05_libero)** - `03a3dd7` (fix)
2. **Task 2: Record the D-07 decision revision in 03-CONTEXT.md and 03-RESEARCH.md** - `4314889` (docs)

**Plan metadata:** committed separately by the orchestrator after this SUMMARY is written.

## Files Created/Modified
- `libero/notebooks/03b-pi0-inference-smoketest.ipynb` - Deprecated `pi0_fast_libero`/`pi0_libero` config-name references replaced with `--env=LIBERO` / `pi05_libero` across 4 cells
- `.planning/phases/03-vla-inference-loop/03-CONTEXT.md` - D-07 amended with a dated 2026-07-26 sub-bullet explaining the `--env` EnvMode misunderstanding and its resolution
- `.planning/phases/03-vla-inference-loop/03-RESEARCH.md` - Assumptions Log A1 annotated with a dated resolution note confirming the flagged risk materialized as predicted

## Decisions Made
- Preserved the original D-07 bullet and A1 table row unmodified; amendments/resolutions were appended as new dated sub-notes beneath/after them (per the plan's `<read_first>` guidance establishing this as the repo's first example of amending a CONTEXT.md decision in place, rather than rewriting or deleting it).
- Used `ensure_ascii=True` (not `ensure_ascii=False` as the plan's literal action text specified) when re-serializing the notebook JSON, to match the file's existing escaped-unicode convention — this was necessary to actually achieve the plan's own stated goal of a git diff "scoped to the actual content changes" (see Deviations below).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Notebook re-serialization used ensure_ascii=True instead of the plan-specified ensure_ascii=False, to actually achieve a minimal/scoped diff**
- **Found during:** Task 1 (notebook patch script)
- **Issue:** The plan's `<action>` text instructed writing the patched notebook back with `json.dump(nb, f, indent=1, ensure_ascii=False)`, asserting this "preserves the file's existing nbformat-style JSON formatting so the git diff stays minimal and scoped to the actual content changes." Verifying against the committed file (`git show HEAD:...`) showed the opposite: the original notebook is serialized with `ensure_ascii=True` (escaped `—`/`π` sequences throughout). Applying `ensure_ascii=False` as literally instructed rewrote every existing em-dash and Greek-letter escape in the file into a literal UTF-8 character, producing a ~60-line diff spanning cells far outside the 4 cells the plan intended to touch — directly contradicting the plan's own stated intent.
- **Fix:** Re-ran the patch with `ensure_ascii=True` (the correct match to the original file's convention), which produced the intended minimal, scoped diff (16 lines across exactly the 4 targeted cells).
- **Files modified:** `libero/notebooks/03b-pi0-inference-smoketest.ipynb`
- **Verification:** `git diff` confirmed the diff is now scoped to only the 4 intended cells' content changes; Task 1's automated `<verify>` script (json.load + assertions on deprecated/new config names and cleared cell state) passed.
- **Committed in:** `03a3dd7` (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 bug — incorrect JSON serialization flag that contradicted the plan's own stated intent)
**Impact on plan:** Necessary to actually deliver the plan's explicit goal of a minimal, scoped diff. No scope creep — same 4 cells touched, same content changes made, only the serialization flag corrected.

## Issues Encountered
None beyond the deviation above.

## User Setup Required
None - no external service configuration required. A live Colab re-run of Notebook B (to confirm the fix resolves the original failure end-to-end) remains a follow-up step for the project owner, as explicitly scoped out of this plan's `<verification>` section.

## Next Phase Readiness
- Notebook B's config-name plumbing is fixed and documented; the project owner can re-run it on Colab to confirm the `Pi0Backend` server now starts successfully and unblock 03-03's Task 4 sign-off checkpoint.
- No changes were made to `Pi0Backend`, `run_suite`/`eval_loop`, the localhost-only bind, or any other notebook — scope stayed strictly within the config-name plumbing and its two documentation records, per this plan's explicit constraints.

## Known Stubs
None - no placeholder/stub patterns introduced by this plan's changes.

## Threat Flags
None - this plan only changed a built-in `--env` enum value and documentation prose; no new network endpoints, auth paths, file access patterns, or trust boundaries were introduced. Pre-existing threats (T-3-SC, T-3-06, T-3-07, T-3-08) remain valid and unchanged, per this plan's own threat model (T-3-09, disposition: accept).

---
*Quick task: 260726-epz*
*Completed: 2026-07-26*

## Self-Check: PASSED

- FOUND: libero/notebooks/03b-pi0-inference-smoketest.ipynb
- FOUND: .planning/phases/03-vla-inference-loop/03-CONTEXT.md
- FOUND: .planning/phases/03-vla-inference-loop/03-RESEARCH.md
- FOUND: .planning/quick/260726-epz-fix-libero-notebooks-03b-pi0-inference-s/260726-epz-SUMMARY.md
- FOUND commit: 03a3dd7
- FOUND commit: 4314889
