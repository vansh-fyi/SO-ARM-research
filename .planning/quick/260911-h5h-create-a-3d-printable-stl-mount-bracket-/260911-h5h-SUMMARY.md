---
phase: quick-260911-h5h
plan: 01
subsystem: hardware
tags: [trimesh, manifold3d, shapely, stl, cad, csg]

requires: []
provides:
  - "3D-printable STL mount bracket generator for the Waveshare AR0144 Stereo USB Camera"
affects: [physical-hardware]

tech-stack:
  added: [trimesh 5.1.0, manifold3d 3.5.3, shapely 2.1.2, numpy 2.5.3]
  patterns:
    - "Isolated project-local .venv for one-off mechanical CAD utilities under explorations/hardware/, kept outside the repo's main pip environment"
    - "Boolean CSG mount-bracket generation via trimesh + manifold3d engine (rounded 2D profile -> extrude -> boolean difference/union chain)"

key-files:
  created:
    - explorations/hardware/ar0144_mount/build_mount.py
    - explorations/hardware/ar0144_mount/ar0144_stereo_mount.stl
    - explorations/hardware/ar0144_mount/README.md
  modified: []

key-decisions:
  - "Used engine=\"manifold\" for extrude_polygon triangulation (not the plan's stated \"triangle\" or default) since the optional `triangle` package isn't installed and manifold3d is already a required boolean-CSG dependency — avoids adding a new package install"

patterns-established:
  - "One-off mechanical CAD scripts live under explorations/hardware/<part_name>/ with their own isolated .venv, gitignored by the existing explorations/* rule with zero .gitignore changes"

requirements-completed: []

coverage:
  - id: D1
    description: "Binary STL mount bracket: 145x66x4mm rounded-corner plate, 70x34mm window, 4x M3 through-holes + hex nut-trap tabs at 52x24mm spacing, 2x lens clearance cutouts"
    verification:
      - kind: unit
        ref: "trimesh geometry assertions: is_watertight, euler_number==-8, bbox [145,66,4], volume in [20000,35000] — all passed (watertight=True, euler=-8, bbox=[145,66,4], volume=30350.98)"
        status: pass
    human_judgment: false
  - id: D2
    description: "README documenting all assumed/approximated dimensions, explicitly flagging lens position as approximate"
    verification:
      - kind: manual_procedural
        ref: "grep -qi approximate README.md — passed; README contains dedicated 'Assumed / approximated dimensions' section"
        status: pass
    human_judgment: false

duration: 12min
completed: 2026-09-11
status: complete
---

# Quick Task 260911-h5h: AR0144 Stereo Camera Mount Bracket Summary

**Generated a watertight binary STL mount bracket (145x66x4mm rounded-corner frame, 70x34mm window, 4 M3-hole tabs with hex nut-traps, 2 lens-clearance cutouts) via a trimesh + manifold3d boolean CSG pipeline, with full assumption documentation in README.md.**

## Performance

- **Duration:** 12 min
- **Started:** 2026-09-11T06:51:00Z
- **Completed:** 2026-09-11T07:03:12Z
- **Tasks:** 2 completed
- **Files modified:** 3 (all new, all gitignored by existing `explorations/*` rule)

## Accomplishments
- `build_mount.py` implements the full documented boolean CSG pipeline (rounded plate -> window cut -> tab union -> lens-clearance cut -> hole/nut-trap cuts) using `trimesh` + `manifold3d`
- Generated `ar0144_stereo_mount.stl`: confirmed watertight, correct topology (`euler_number=-8`), exact 145x66x4mm bounding box, plausible volume (~30351 mm^3) — matches the plan's pre-verified reference output exactly
- `README.md` documents the source PCB spec, usage/assembly instructions, regeneration steps, and every assumed/approximated dimension (explicitly flagging the lens positions as approximate/derived)
- Isolated project-local venv at `explorations/hardware/ar0144_mount/.venv` keeps `trimesh`/`manifold3d`/`shapely`/`numpy` out of the repo's system/global Python (Homebrew Python 3.14 is PEP 668 externally-managed)

## Task Commits

No code commits were made for `build_mount.py`, `ar0144_stereo_mount.stl`, or `README.md`. Per the plan's own context section, `explorations/*` is gitignored repo-wide (`.gitignore` line 1, with only `explorations/soarm_sanity.py` excepted), and the plan explicitly instructs not to modify `.gitignore`. Verified with `git check-ignore -v` that the new directory is correctly ignored with zero `.gitignore` changes. This matches the plan's stated expectation that the new files "will naturally stay untracked by git with zero `.gitignore` changes needed."

**Plan metadata:** committed separately by the orchestrator (docs commit, not part of this SUMMARY).

## Files Created/Modified
- `explorations/hardware/ar0144_mount/build_mount.py` - CSG generator script (boolean pipeline: rounded plate, window cut, tab union, lens-clearance cut, hole/nut-trap cuts)
- `explorations/hardware/ar0144_mount/ar0144_stereo_mount.stl` - Generated binary STL, watertight, 145x66x4mm bbox, euler_number=-8
- `explorations/hardware/ar0144_mount/README.md` - Usage, source PCB spec, and full assumption/approximation log
- `explorations/hardware/ar0144_mount/.venv/` - Isolated venv (trimesh, manifold3d, shapely, numpy) — not a tracked deliverable, gitignored, listed here for completeness only

## Decisions Made
- Switched `extrude_polygon`'s triangulation engine from `"triangle"` (as the plan's example used) to `"manifold"` because the optional `triangle` Python package is not installed and isn't part of the plan's stated dependency list (trimesh, manifold3d, shapely, numpy); `manifold3d` already provides triangulation and is already a required dependency for the boolean CSG operations, so this avoided adding an unplanned package install (see Deviations below).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed missing-module crash by switching triangulation engine**
- **Found during:** Task 2 (first run of `build_mount.py`)
- **Issue:** `build_mount.py` initially called `trimesh.creation.extrude_polygon(..., engine="triangle")` for the rounded-plate profile and the hex nut-trap pocket. The optional `triangle` Python package is not installed in the venv (only `trimesh`, `manifold3d`, `shapely`, `numpy` were installed per the plan's stated dependency list), causing `ModuleNotFoundError: No module named 'triangle'` on the very first run.
- **Fix:** Inspected `trimesh.creation.triangulate_polygon`'s source and confirmed `manifold3d` (already installed as the boolean-CSG engine) also implements a `"manifold"` triangulation engine. Switched both `extrude_polygon` calls (plate profile, hex nut pocket) to `engine="manifold"`.
- **Files modified:** `explorations/hardware/ar0144_mount/build_mount.py`
- **Verification:** Re-ran `build_mount.py` — succeeded, and the full Task 2 automated verify block (watertight, `euler_number==-8`, bbox `[145,66,4]`, volume `30350.98` in `[20000,35000]`) passed exactly matching the plan's pre-verified reference numbers.
- **Committed in:** Not committed (gitignored — see Task Commits above); change is present in the working tree file.

---

**Total deviations:** 1 auto-fixed (1 bug fix, Rule 1)
**Impact on plan:** Necessary for the script to run at all in the actually-installed dependency set (avoided adding the unplanned `triangle` package). No scope creep — output geometry matches the plan's pre-verified reference values exactly (watertight, euler_number=-8, bbox [145,66,4], volume ~30351).

## Issues Encountered
None beyond the triangulation-engine fix documented above.

## User Setup Required
None - no external service configuration required. The generated STL is ready to slice and print directly.

## Next Phase Readiness
- The mount bracket STL is print-ready; no further CAD work needed for this specific mount.
- This is a standalone, gitignored utility isolated from the LIBERO/VLA simulation pipeline — no impact on any other phase's readiness.
- If the AR0144's actual lens positions are found to differ meaningfully from the approximated (13.10,15)/(52.90,15) board-local coordinates once the physical part is test-fit, `build_mount.py`'s `LENS_POSITIONS` constant can be adjusted and the STL regenerated.

---
*Quick task: 260911-h5h*
*Completed: 2026-09-11*

## Self-Check: PASSED

- FOUND: explorations/hardware/ar0144_mount/build_mount.py
- FOUND: explorations/hardware/ar0144_mount/ar0144_stereo_mount.stl
- FOUND: explorations/hardware/ar0144_mount/README.md
- FOUND: .planning/quick/260911-h5h-create-a-3d-printable-stl-mount-bracket-/260911-h5h-SUMMARY.md
- No commit hashes to verify — code artifacts are gitignored by the existing `explorations/*` rule (see Task Commits section above)
