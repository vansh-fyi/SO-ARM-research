---
phase: 05-spatial-awareness
plan: 03
subsystem: simulation
tags: [libero, robosuite, bddl, spatial-predicates, mujoco]

# Dependency graph
requires:
  - phase: 04-dataset-collection
    provides: SOARM roboninecom 84mm gripper, in-reach cream_cheese/bowl regions (put_the_cream_cheese_in_the_bowl.bddl), verified collision-free layout
provides:
  - LeftOfX/RightOfX/NearTo/FarFrom spatial predicate classes registered in VALIDATE_PREDICATE_FN_DICT
  - libero_spatial_soarm/ BDDL task suite (3 tasks: right-of, near, between) with genuine spatial goal predicates
  - Real-env integration test pattern (20-reset goal-satisfaction + directional-correctness + collision-contact checks) for future spatial task authoring
affects: [spatial-awareness dataset collection, VLA fine-tuning task suite, spatial language grounding evaluation]

# Tech tracking
tech-stack:
  added: []
  patterns: [BinaryAtomic predicate subclass pattern (ground-truth get_geom_state()-only, D-05), decomposing N-ary spatial relations into conjoined binary predicates for the 2/3-token dispatcher, region-verbatim-reuse for zero-new-embodiment-risk task authoring]

key-files:
  created:
    - LIBERO/libero/libero/envs/test_spatial_predicates.py
    - LIBERO/libero/libero/bddl_files/libero_spatial_soarm/put_the_cream_cheese_to_the_right_of_the_bowl.bddl
    - LIBERO/libero/libero/bddl_files/libero_spatial_soarm/put_the_cream_cheese_near_the_bowl.bddl
    - LIBERO/libero/libero/bddl_files/libero_spatial_soarm/put_the_butter_between_the_bowl_and_the_cream_cheese.bddl
    - LIBERO/libero/libero/conftest.py
  modified:
    - LIBERO/libero/libero/envs/predicates/base_predicates.py
    - LIBERO/libero/libero/envs/predicates/__init__.py

key-decisions:
  - "LeftOfX/RightOfX use an X-axis margin (0.03, matching check_ontop's existing precedent) rather than a tie-break rule -- ambiguous same-X pairs resolve to False on both, avoided by construction per D-09"
  - "NearTo threshold 0.22 verified at plan-time against the worst-case farthest-corner distance across the reused init ranges (~0.197m), leaving ~0.023m margin"
  - "'Between' expressed as a conjunction of two binary predicates (RightOfX + LeftOfX) rather than a new ternary predicate, since _eval_predicate only dispatches 2/3-token BDDL tuples"
  - "New butter_region for the 'between' task offset in y away from both existing regions and the robot's forward corridor -- passed empirical 20-reset zero-contact check on the first attempt, no iteration needed"

requirements-completed: [SPAT-05]

coverage:
  - id: D1
    description: "4 new spatial predicate classes (LeftOfX/RightOfX/NearTo/FarFrom) registered with zero key collisions, correctly classifying every unit-test case"
    requirement: "SPAT-05"
    verification:
      - kind: unit
        ref: "LIBERO/libero/libero/envs/test_spatial_predicates.py::TestLeftOfX, TestRightOfX, TestNearTo, TestFarFrom"
        status: pass
    human_judgment: false
  - id: D2
    description: "right-of and near BDDL tasks reset-satisfy their own goal on 20/20 trials using Phase 4's already-validated regions, with directional correctness proven"
    requirement: "SPAT-05"
    verification:
      - kind: integration
        ref: "LIBERO/libero/libero/envs/test_spatial_predicates.py::TestRightOfTaskIntegration, TestNearTaskIntegration"
        status: pass
    human_judgment: false
  - id: D3
    description: "'between' BDDL task with a new 3rd-object region reset-satisfies its goal on 20/20 trials and is empirically collision-free (zero robot-vs-butter_1 contact across 20 resets)"
    requirement: "SPAT-05"
    verification:
      - kind: integration
        ref: "LIBERO/libero/libero/envs/test_spatial_predicates.py::TestBetweenTaskIntegration"
        status: pass
    human_judgment: false

# Metrics
duration: 45min
completed: 2026-08-09
status: complete
---

# Phase 5 Plan 03: Spatial Predicates + BDDL Tasks Summary

**4 new ground-truth spatial predicate classes (LeftOfX/RightOfX/NearTo/FarFrom) plus a 3-task `libero_spatial_soarm` BDDL suite covering left/right, near, and between -- all reset-satisfy their goals 20/20 with directional correctness and empirical collision-safety proven via real headless SOARM env runs.**

## Performance

- **Duration:** ~45 min
- **Started:** 2026-08-09T15:19:00Z
- **Completed:** 2026-08-09T16:03:58Z
- **Tasks:** 3 (plan) + 1 deviation fix commit
- **Files modified:** 7 (2 modified, 5 created)

## Accomplishments
- Added `LeftOfX`, `RightOfX`, `NearTo`, `FarFrom` `BinaryAtomic` predicate classes to `base_predicates.py`, using only `get_geom_state()` ground-truth positions (D-05) -- no depth-derived XYZ ever enters a pass/fail check
- Registered all 4 under lowercase keys in `VALIDATE_PREDICATE_FN_DICT` with zero collisions against the 10 pre-existing keys
- Authored 3 new BDDL tasks under `libero_spatial_soarm/`: right-of and near reuse Phase 4's already-collision-validated cream_cheese/bowl regions verbatim (zero new embodiment risk); "between" introduces a new butter_1 region, empirically validated collision-free via `env.check_contact(robot_model, butter_model)` across 20 resets
- Built a reusable real-env integration test pattern: 20-reset goal-satisfaction loops, reversed-argument directional-correctness checks (proving predicates aren't trivially always-True), and contact-based collision-safety checks -- 16/16 tests passing (10 unit + 6 integration)

## Task Commits

Each task was committed atomically:

1. **Task 1: LeftOfX/RightOfX/NearTo/FarFrom predicate classes + registry** - `24362c9` (feat) + `d53016b` (fix, staging correction)
2. **Task 2: right-of + near BDDL tasks** - `ac1a7a9` (feat)
3. **Task 3: "between" BDDL task + collision validation** - `70e5cea` (feat)

_Note: Task 1 required a follow-up commit (`d53016b`) after a `git add` staging miss caused by the worktree's case-mismatched checkout path (see Deviations below) -- the predicate class/registry file content itself was correct and unchanged between the two commits._

## Files Created/Modified
- `LIBERO/libero/libero/envs/predicates/base_predicates.py` - adds `LeftOfX`, `RightOfX`, `NearTo`, `FarFrom` classes + `import numpy as np`
- `LIBERO/libero/libero/envs/predicates/__init__.py` - registers the 4 new predicates in `VALIDATE_PREDICATE_FN_DICT`
- `LIBERO/libero/libero/envs/test_spatial_predicates.py` - unit tests (stub-object based) + integration tests (real headless SOARM env, 20-reset loops, directional and collision checks)
- `LIBERO/libero/libero/bddl_files/libero_spatial_soarm/put_the_cream_cheese_to_the_right_of_the_bowl.bddl` - new task, goal `(RightOfX cream_cheese_1 akita_black_bowl_1)`
- `LIBERO/libero/libero/bddl_files/libero_spatial_soarm/put_the_cream_cheese_near_the_bowl.bddl` - new task, goal `(NearTo cream_cheese_1 akita_black_bowl_1)`
- `LIBERO/libero/libero/bddl_files/libero_spatial_soarm/put_the_butter_between_the_bowl_and_the_cream_cheese.bddl` - new task, new butter_1 region, goal `(And (RightOfX butter_1 akita_black_bowl_1) (LeftOfX butter_1 cream_cheese_1))`
- `LIBERO/libero/libero/conftest.py` - new; case-insensitive-checkout import compatibility shim (deviation, see below)

## Decisions Made
- `MARGIN = 0.03` for LeftOfX/RightOfX matches `ObjectState.check_ontop`'s existing XY-tolerance precedent (D-08); ambiguous same-X pairs resolve to False on both predicates by construction rather than via a tie-break rule (D-09)
- `NearTo.THRESHOLD = 0.22`, `FarFrom.THRESHOLD = 0.20` are the plan's D-08 starting defaults; `NearTo` was verified at plan/execution time against the worst-case farthest-corner distance across the reused init ranges (~0.197m static estimate), which the 20-reset integration test confirmed empirically holds
- "Between" decomposed into two conjoined binary predicates (`RightOfX` + `LeftOfX`) rather than adding a new ternary predicate, since `_eval_predicate` in `libero_tabletop_manipulation.py` only dispatches 2/3-token BDDL tuples (left unmodified, per plan)
- New `butter_region` ranges `(-0.13 -0.08 -0.11 -0.06)` chosen to keep worst-case region-pair margins (0.06m for both `RightOfX(butter_1, bowl)` and `LeftOfX(butter_1, cream_cheese)`) comfortably above `MARGIN=0.03`, and offset in y away from the robot's forward corridor per Phase 4's established collision lesson -- the empirical 20-reset contact check passed on the first attempt, no region-adjustment iteration was needed

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Case-insensitive worktree checkout broke the codebase's absolute `import LIBERO.libero.libero...` pattern**
- **Found during:** Task 1, first verification run
- **Issue:** This worktree's git checkout has the top-level `LIBERO/` directory cased as lowercase `libero` on disk (macOS APFS is case-insensitive but git's `core.ignorecase` treats both spellings as equivalent). The codebase's own `envs/`, `datasets/`, etc. modules use the absolute import form `import LIBERO.libero.libero.envs...` (an existing, plan-acknowledged anti-pattern per CLAUDE.md), which requires an actual directory entry named exactly `LIBERO`. CPython's import machinery does a case-sensitive directory-entry scan regardless of filesystem case-insensitivity, so `import LIBERO...` raised `ModuleNotFoundError` -- blocking every test in the file, including Task 1's plain unit tests, from even collecting.
- **Fix:** Added `LIBERO/libero/libero/conftest.py`, a pytest fixture-less shim that, only when the literal `import LIBERO` fails, hand-builds a namespace-package module object whose `__path__` points at the real on-disk directory (resolved case-insensitively via `os.path.isdir`) and registers it in `sys.modules["LIBERO"]`. Everything nested below (`LIBERO.libero.libero.envs...`) is already lower-case in both the import statements and on disk, so normal import resolution handles the rest once this one top-level alias exists. No-op on a normally-cased checkout.
- **Files modified:** LIBERO/libero/libero/conftest.py (new)
- **Verification:** `conda run -n libero pytest LIBERO/libero/libero/envs/test_spatial_predicates.py -x -q` collects and passes all 16 tests (previously failed at collection)
- **Committed in:** `24362c9` (Task 1 commit)

**2. [Rule 3 - Blocking] `git add` on the uppercase `LIBERO/...` path silently no-op'd two files**
- **Found during:** Post-Task-1-commit verification (`git log -1 --stat` showed only 2 of 4 intended files)
- **Issue:** Given the same case-mismatched checkout, `git add LIBERO/libero/libero/envs/predicates/{base_predicates.py,__init__.py}` (uppercase form) failed to stage those two files against the tracked lowercase-cased index entries, so the first Task 1 commit was missing the actual predicate class/registry changes.
- **Fix:** Re-staged with the exact lowercase path (`libero/libero/libero/envs/predicates/...`) as reported by `git status --short`, and committed the (unchanged) content in a follow-up commit.
- **Files modified:** LIBERO/libero/libero/envs/predicates/base_predicates.py, LIBERO/libero/libero/envs/predicates/__init__.py (staging only -- no content change from what commit `24362c9` already described)
- **Verification:** `git status --short` clean of these two files after the follow-up commit; `git log --stat` confirms both files present
- **Committed in:** `d53016b`

**3. [Rule 1 - Bug] `assert predicate(...) is True/False` failed against numpy bool results**
- **Found during:** Task 1, first unit-test run
- **Issue:** `LeftOfX.__call__` etc. return the result of a numpy comparison (`np.True_`/`np.False_`), which is not identical (`is`) to Python's singleton `True`/`False`, causing `assert ... is True` to fail even though the predicate logic was correct.
- **Fix:** Wrapped predicate results in `bool(...)` before the identity assertion in all 10 unit tests.
- **Files modified:** LIBERO/libero/libero/envs/test_spatial_predicates.py
- **Verification:** `pytest -k "not integration"` -- 10/10 pass
- **Committed in:** `24362c9` (Task 1 commit)

---

**Total deviations:** 3 auto-fixed (2 blocking-worktree-checkout, 1 bug)
**Impact on plan:** All 3 fixes were necessary to run this plan's verification at all in this worktree, or to make the tests correctly assert the (already-correct) predicate behavior. No scope creep -- no predicate logic, BDDL content, or task design was changed by any of these fixes.

## Issues Encountered
None beyond the deviations above. The "between" task's new 3rd-object region passed its 20-reset zero-contact collision-safety check on the first attempt -- no region-adjustment iterations were needed (plan allowed up to 3).

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- SPAT-05 fully satisfied: 3 BDDL tasks (right-of, near, between) with genuinely spatial, ground-truth-evaluated goal predicates now exist under `libero_spatial_soarm/`
- The 4 registered predicate classes (`leftofx`/`rightofx`/`nearto`/`farfrom`) are available for any future spatial-task authoring in this phase or later phases
- The `LIBERO/libero/libero/conftest.py` case-insensitive-checkout shim is now in place repo-wide for this directory tree, unblocking pytest collection for ANY test under `LIBERO/libero/libero/` on a similarly-cased worktree checkout (not just this plan's tests) -- worth flagging to future plans/executors as a known, now-mitigated environment quirk
- No blockers for subsequent Phase 5 plans

---
*Phase: 05-spatial-awareness*
*Completed: 2026-08-09*

## Self-Check: PASSED

All 8 claimed artifacts found on disk; all 5 commit hashes (`24362c9`, `d53016b`, `ac1a7a9`, `70e5cea`, `08d58da`) found in git log.
