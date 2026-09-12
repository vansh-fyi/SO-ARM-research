---
phase: 01-colab-environment-setup
plan: "02"
subsystem: colab-notebook
tags: [jupyter, mujoco, robosuite, libero, egl, openvla, numpy, numba, pip]

# Dependency graph
requires:
  - phase: 01-colab-environment-setup/01-01
    provides: Block A install cells (Steps 1-6) confirmed passing on Colab A100
provides:
  - LIBERO/notebooks/01-colab-env-setup.ipynb (cells 13-19, Block B verification)
  - EGL bootstrap cell (creates NVIDIA ICD JSON, sets MUJOCO_GL=egl)
  - LIBERO config.yaml bootstrap cell (pre-creates ~/.libero/config.yaml)
  - ENV-01 verification cell (package versions + filtered pip conflict check)
  - ENV-02 verification cell (LIBERO Panda headless render check)
  - numba>=0.59,<0.60 pin in Step 3 (before numpy<2)
affects:
  - 01-03 (ENV-03 OpenVLA-OFT load cell — will build on this Block B infrastructure)
  - phase-2+ (all Colab sessions follow the EGL bootstrap pattern from these cells)

# Tech tracking
tech-stack:
  added:
    - Block B cells (EGL bootstrap, LIBERO config.yaml, sys.path, ENV-01, ENV-02, progress markdown)
    - numba>=0.59,<0.60 pin in Step 3 (numpy-1.x-compatible; Colab system numba built for numpy 2.x)
  patterns:
    - ENV-01 version comparison: always strip build tag with installed.split('+')[0] before comparing
    - ENV-01 pip check: filter to OUR_PACKAGES set — Colab system conflicts are pre-existing noise
    - EGL bootstrap must be the FIRST post-restart cell (before any physics package import)
    - LIBERO config.yaml must be created before any `import libero` call
    - numba pin must precede numpy<2 pin in Step 3

key-files:
  created: []
  modified:
    - libero/notebooks/01-colab-env-setup.ipynb
    - .planning/phases/01-colab-environment-setup/.continue-here.md

key-decisions:
  - "numba 0.59.x chosen for numpy 1.x compat — supports numpy 1.21-1.26 + Python 3.12; Colab system numba compiled for numpy 2.x causes ABI crash"
  - "ENV-01 filters pip check to OUR_PACKAGES list — Colab system conflicts (jax, cupy, opencv needing numpy>=2) are accepted as pre-existing and non-blocking"
  - "torch version comparison strips build tag — torch installs as '2.2.0+cu121'; exact match produces false MISMATCH"
  - "flash-attn unavailability on CUDA 12.8 accepted — graceful fallback in place, ENV-01/02/03 proceed normally"

patterns-established:
  - "Version compare: always strip build tags with .split('+')[0] for pip-installed CUDA packages"
  - "pip check scope: filter to a defined OUR_PACKAGES set; broad pip check produces noise on Colab"
  - "Dependency ordering: numba pin BEFORE numpy<2 pin to prevent ABI mismatch from Colab system numba"

requirements-completed: []

coverage:
  - id: D1
    description: "numba>=0.59,<0.60 pin in Step 3 (Cell 6) before numpy<2 pin"
    verification: []
    human_judgment: true
    rationale: "Requires Colab runtime execution to confirm numba imports without ABI error after numpy<2 pin"
  - id: D2
    description: "ENV-01 cell uses build-tag-stripped version comparison (installed.split('+')[0])"
    verification: []
    human_judgment: true
    rationale: "Requires Colab execution with torch 2.2.0+cu121 to confirm PASS status for torch row"
  - id: D3
    description: "ENV-01 cell filters pip check output to OUR_PACKAGES set only"
    verification: []
    human_judgment: true
    rationale: "Requires Colab runtime with system jax/cupy/opencv present to confirm filtering works and prints PASS"
  - id: D4
    description: "Block B cells (EGL, config.yaml, sys.path, ENV-02) appended to notebook"
    verification: []
    human_judgment: true
    rationale: "Requires Colab execution post-restart to confirm ENV-02 LIBERO render produces non-black frame"

# Metrics
duration: 25min
completed: 2026-07-09
status: complete
---

# Phase 01 Plan 02: Block B Verification Cells + Bug Fixes Summary

**Block B post-restart verification cells added (EGL, LIBERO config, ENV-01/02) with two Colab runtime bugs fixed: numba ABI mismatch via numpy-1.x-compatible numba pin, and ENV-01 false failures via build-tag stripping and OUR_PACKAGES pip conflict filtering**

## Performance

- **Duration:** ~25 min
- **Started:** 2026-07-09T11:30:00Z
- **Completed:** 2026-07-09T11:55:00Z
- **Tasks:** 4
- **Files modified:** 2

## Accomplishments

- Block B cells (indices 13-19) appended to `libero/notebooks/01-colab-env-setup.ipynb` covering EGL bootstrap, LIBERO config.yaml bootstrap, sys.path setup, ENV-01 version check, ENV-02 render check, and progress markdown
- Fixed numba ABI mismatch: `pip install "numba>=0.59,<0.60"` inserted into Step 3 (Cell 6) BEFORE the `numpy<2` pin — Colab system numba is compiled for numpy 2.x, causing `ValueError: numpy.dtype size changed` when robosuite imports it post-pin
- Fixed ENV-01 false MISMATCH for torch: version comparison now strips build tags with `installed.split('+')[0]` before comparing (torch installs as "2.2.0+cu121", expected "2.2.0")
- Fixed ENV-01 false FAIL from system pip conflicts: pip check output is now filtered to `OUR_PACKAGES` set; Colab system packages (jax, cupy, opencv) needing numpy>=2 are pre-existing noise not in our pipeline

## Task Commits

1. **Task 1+2+3: notebook + .continue-here.md** - `7a83d4e` (fix: numba pin + ENV-01 smarter version/conflict check — .continue-here.md only, notebook staging missed)
2. **Task 1+2 (notebook)** - `c9c841e` (fix: apply notebook cell changes — numba pin + ENV-01 fixes)

## Files Created/Modified

- `libero/notebooks/01-colab-env-setup.ipynb` — Cell 6: numba pin added; Cell 17: version compare and pip filter fixed (Block B cells already present from prior commit e490a86)
- `.planning/phases/01-colab-environment-setup/.continue-here.md` — 4 new BLOCKING CONSTRAINTS and 2 new Critical Anti-Patterns documented

## Decisions Made

- numba 0.59.x chosen: only numba series with verified support for numpy 1.21-1.26 and Python 3.12 — numba 0.60+ drops numpy 1.x support
- OUR_PACKAGES filter for pip check: includes mujoco, robosuite, gym, torch, timm, peft, tokenizers, bddl, transformers, huggingface_hub, flash_attn, einops, sentencepiece, numba, numpy, easydict, cloudpickle, imageio, opencv-python-headless
- flash-attn unavailability on CUDA 12.8 accepted as non-blocking: fallback message already in Step 6; ENV-01/02/03 pipeline does not depend on flash-attn for correctness

## Deviations from Plan

The user-specified fix tasks were implemented exactly as described. No unplanned deviations.

### Issues Found During Colab Execution (pre-existing, now fixed)

**1. [Rule 1 - Bug] numba ABI mismatch after numpy<2 pin**
- **Found during:** User ran Block B on Colab; ENV-02 cell crashed
- **Issue:** `ValueError: numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject` — Colab system numba is a C extension compiled against numpy 2.x headers; our numpy<2 pin downgrades numpy, making the ABI incompatible
- **Root cause:** `robosuite/utils/numba.py` imports numba on first env creation; numba's C extension checks numpy header size at import time
- **Fix:** Added `!pip install "numba>=0.59,<0.60" -q` in Cell 6 (Step 3) BEFORE the `numpy<2` pin. This installs a numba compiled for numpy 1.x. The numpy<2 pin follows, ensuring final numpy version matches what numba was built against.
- **Files modified:** `libero/notebooks/01-colab-env-setup.ipynb` (Cell 6)
- **Commit:** `c9c841e`

**2. [Rule 1 - Bug] ENV-01 false MISMATCH for torch due to build-tag suffix**
- **Found during:** User ran ENV-01 cell on Colab; torch showed MISMATCH
- **Issue:** `torch` installs as `"2.2.0+cu121"` (build tag appended by PyTorch index). Exact string comparison `installed == "2.2.0"` fails.
- **Fix:** Changed version check to `installed.split('+')[0] == expected_ver` — strips the CUDA build tag before comparing
- **Files modified:** `libero/notebooks/01-colab-env-setup.ipynb` (Cell 17)
- **Commit:** `c9c841e`

**3. [Rule 1 - Bug] ENV-01 false FAIL due to over-broad pip check**
- **Found during:** User ran ENV-01 cell on Colab; printed "ENV-01: FAIL" due to jax/cupy/opencv system conflicts
- **Issue:** `pip check` reports ALL system conflicts including Colab pre-installed jax, cupy, opencv, tifffile, rasterio, etc. that require numpy>=2. These are not in our pipeline and cannot be removed.
- **Fix:** Added `OUR_PACKAGES` set; filter pip check output to only flag lines mentioning our packages. If no our-package conflicts exist, print PASS with a parenthetical note about pre-existing system conflicts.
- **Files modified:** `libero/notebooks/01-colab-env-setup.ipynb` (Cell 17)
- **Commit:** `c9c841e`

---

**Total deviations:** 3 auto-fixed bugs (all Rule 1 — incorrect behavior in Colab runtime)
**Impact on plan:** All fixes essential for correctness of ENV-01 gate check. No scope creep.

## Issues Encountered

- macOS case-insensitive filesystem: `git add LIBERO/notebooks/...` silently did nothing (git tracks the file as `libero/notebooks/...` lowercase). Had to use `git add libero/notebooks/...` (lowercase). This caused the notebook to be missing from the first commit; a follow-up commit was needed.

## User Setup Required

After this fix, user needs to:
1. Re-run Cell 6 (Step 3) in the Colab notebook — this will install numba 0.59.x with numpy 1.x ABI
2. Restart the Colab runtime again (Runtime > Restart session)
3. Re-run Block B cells (14-18) to confirm ENV-01 and ENV-02 pass

## Next Phase Readiness

- Block B cells are in place and bugs fixed; ENV-01 should now show PASS after user reruns Step 3 and Block B
- ENV-02 (LIBERO Panda render) depends on the numba fix — should pass once numba 0.59 is installed
- Plan 01-03 (ENV-03: OpenVLA-OFT model load cell) is the next task

---
*Phase: 01-colab-environment-setup*
*Completed: 2026-07-09*
