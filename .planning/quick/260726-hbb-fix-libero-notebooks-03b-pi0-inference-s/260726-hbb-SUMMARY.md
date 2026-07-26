---
phase: 03-vla-inference-loop
plan: 260726-hbb
subsystem: vla-inference
tags: [openpi, pi0, jupyter-notebook, gsutil, colab, crcmod, boto, serve_policy]

# Dependency graph
requires:
  - phase: 03-vla-inference-loop (03-03 / 260726-gj6)
    provides: "Notebook B (03b-pi0-inference-smoketest.ipynb) with OPENPI_DATA_HOME repointed to local Colab disk and a crcmod C-extension reinstall cell, whose root-cause theory (Drive FUSE unreliability) was then disproven by an identical live-Colab failure after switching to local disk"
provides:
  - "Notebook B's pre-flight cell pair (markdown + code) that sets CLOUDSDK_PYTHON_SITEPACKAGES=1 and a check_hashes=if_fast_else_skip /content/.boto fallback, with a fast gsutil version -l crcmod check that runs before the real 11.6GB checkpoint download"
  - "Corrected root-cause diagnosis (GoogleCloudPlatform/gsutil#1429: gsutil's bundled Cloud SDK Python interpreter is isolated from the Colab kernel's site-packages) replacing the disproven Drive-vs-local-disk theory"
  - "Second dated decision-revision trail in 03-03-PLAN.md (T-3-08 second amendment) and 03-RESEARCH.md (Pitfall 5 correction), cross-referenced, with all prior amendments preserved as historical record"
affects: [03-vla-inference-loop follow-on Colab re-run, VLA-04 sign-off]

# Tech tracking
tech-stack:
  added: []
  patterns: ["Amend historical decision/threat-model records in-place with a dated sub-note rather than rewriting or deleting the original entry (established by 260726-epz, reused by 260726-gj6, reused again here for a second, corrected amendment on the same T-3-08 row/Pitfall 5 note)"]

key-files:
  created: []
  modified:
    - libero/notebooks/03b-pi0-inference-smoketest.ipynb
    - .planning/phases/03-vla-inference-loop/03-03-PLAN.md
    - .planning/phases/03-vla-inference-loop/03-RESEARCH.md

key-decisions:
  - "Corrected the root-cause diagnosis for Notebook B's repeated gsutil checkpoint-download failure: the 260726-gj6 fix's Drive-vs-local-disk theory was disproven by an identical live-Colab failure after switching to local disk with crcmod reinstalled. The actual mechanism, confirmed via GoogleCloudPlatform/gsutil#1429, is that gsutil's bundled Cloud SDK Python interpreter is isolated from the Colab kernel's site-packages that pip install crcmod targets, so the compiled C extension stays invisible to gsutil regardless of a successful kernel-side install."
  - "Added CLOUDSDK_PYTHON_SITEPACKAGES=1 (a Python os.environ assignment, not a shell !export, so it persists into the later subprocess.Popen env for serve_policy.py) as the primary fix, plus check_hashes=if_fast_else_skip via a /content/.boto file as defense-in-depth in case the bundled interpreter's ABI still isn't compatible."
  - "Inserted the pre-flight cell pair between the existing crcmod-reinstall cell and the server-launch cell (not before crcmod reinstall) so the fast gsutil version -l crcmod check reflects the just-reinstalled extension, and so the env vars land in os.environ before the server-launch cell's dict(os.environ, ...) subprocess call reads it."
  - "Notebook JSON re-serialized with json.dump(nb, f, indent=1, ensure_ascii=True) plus a trailing newline, matching the file's established convention from 260726-epz/260726-gj6, keeping the diff scoped to 2 edited cells plus 2 inserted cells."
  - "Preserved both prior amendments (260726-epz's --env=LIBERO fix and 260726-gj6's Drive-to-local-disk amendment) unmodified; appended new dated sub-notes rather than rewriting history, per the established repo pattern."

patterns-established: []

requirements-completed: [VLA-04]

coverage:
  - id: D1
    description: "Notebook B's pre-flight cell pair (markdown header + code) inserted between the crcmod-reinstall cell and the server-launch cell, setting CLOUDSDK_PYTHON_SITEPACKAGES=1, writing a check_hashes=if_fast_else_skip /content/.boto fallback, and running a fast gsutil version -l crcmod check before the real checkpoint download; markdown cell 1b668f9c carries a second dated amendment; server-launch cell has an explanatory comment and cleared stale output; notebook stays valid JSON with 19 cells"
    requirement: "VLA-04"
    verification:
      - kind: unit
        ref: "inline grep + python3 json-load verification script (PLAN.md Task 1 <verify>, re-run against libero/notebooks/03b-pi0-inference-smoketest.ipynb): confirms CLOUDSDK_PYTHON_SITEPACKAGES, check_hashes, if_fast_else_skip, gsutil#1429, BOTO_CONFIG, and 'Second amendment (2026-07-26)' are all present; exactly 19 cells; cell 7a49e230's outputs/execution_count cleared"
        status: pass
    human_judgment: false
  - id: D2
    description: "03-03-PLAN.md's T-3-08 threat-model row and 03-RESEARCH.md's Pitfall 5 both carry a SECOND dated 2026-07-26 amendment/correction note recording the corrected root cause and fix, cross-referencing each other, with all prior text (including the first 260726-gj6 amendment) preserved"
    requirement: "VLA-04"
    verification:
      - kind: unit
        ref: "inline grep verification script (PLAN.md Task 2 <verify>): confirms 'Second amendment (2026-07-26)' + 'gsutil#1429' + 'Amendment (2026-07-26)' in 03-03-PLAN.md, and 'Correction (2026-07-26)' + 'gsutil#1429' + 'Resolved (2026-07-26)' in 03-RESEARCH.md"
        status: pass
    human_judgment: false

# Metrics
duration: ~4min
completed: 2026-07-26
status: complete
---

# Quick Task 260726-hbb: Correct Notebook B's gsutil checkpoint-download root cause (isolated bundled-Python, not Drive FUSE)

**Added a pre-flight cell to Notebook B that sets `CLOUDSDK_PYTHON_SITEPACKAGES=1` and a `check_hashes=if_fast_else_skip` boto fallback, correcting the prior fix's disproven Drive-vs-local-disk theory with the confirmed root cause from `GoogleCloudPlatform/gsutil#1429` — gsutil's bundled Cloud SDK Python interpreter is isolated from the Colab kernel's site-packages.**

## Performance

- **Duration:** ~4 min
- **Started:** 2026-07-26T07:03:59Z
- **Completed:** 2026-07-26T07:07:34Z
- **Tasks:** 2/2 completed
- **Files modified:** 3

## Accomplishments
- Notebook B (`libero/notebooks/03b-pi0-inference-smoketest.ipynb`) now has a new markdown + code pre-flight cell pair inserted between the crcmod-reinstall cell and the server-launch cell, which sets `CLOUDSDK_PYTHON_SITEPACKAGES=1`, writes a `check_hashes=if_fast_else_skip` `/content/.boto` config and sets `BOTO_CONFIG` accordingly, and runs `gsutil version -l | grep -i crcmod` for fast pre-flight feedback before the real 11.6 GiB checkpoint download.
- The markdown header (cell `1b668f9c`) now carries a **second** dated `2026-07-26` amendment (the first `260726-gj6` amendment preserved unmodified) explaining that the local-disk `OPENPI_DATA_HOME` change alone did not fix the download — a live Colab re-run hit the identical `CommandException: 6 files/objects could not be transferred` failure — and documenting the corrected root cause per `GoogleCloudPlatform/gsutil#1429`.
- The server-launch cell (`7a49e230`) gained one explanatory comment (no logic change) confirming that `env=dict(os.environ, ...)` already carries `CLOUDSDK_PYTHON_SITEPACKAGES`/`BOTO_CONFIG` into `serve_policy.py`'s subprocess since it reads `os.environ` at call time; its `outputs`/`execution_count` were confirmed already cleared (`[]`/`null`).
- `03-03-PLAN.md`'s T-3-08 STRIDE row and `03-RESEARCH.md`'s Pitfall 5 both now carry a second dated, cross-referenced amendment/correction note, with all original text (including the first `260726-gj6` amendment) preserved as historical record.
- Notebook stays valid JSON with exactly 19 cells (2 new cells inserted; was 17).

## Task Commits

Each task was committed atomically:

1. **Task 1: Notebook B — CLOUDSDK_PYTHON_SITEPACKAGES pre-flight cell + boto check_hashes fallback** - `c6ba5f6` (fix)
2. **Task 2: Record the corrected root-cause decision revision in 03-03-PLAN.md and 03-RESEARCH.md** - `48bfd38` (docs)

**Plan metadata:** committed separately by the orchestrator after this SUMMARY is written.

## Files Created/Modified
- `libero/notebooks/03b-pi0-inference-smoketest.ipynb` - new pre-flight markdown + code cell pair inserted between the crcmod reinstall cell and the server-launch cell (17 → 19 cells); markdown cell `1b668f9c` gained a second dated amendment; server-launch cell `7a49e230` gained one explanatory comment
- `.planning/phases/03-vla-inference-loop/03-03-PLAN.md` - T-3-08 STRIDE row appended with a second dated 2026-07-26 amendment sentence recording the corrected root cause and fix
- `.planning/phases/03-vla-inference-loop/03-RESEARCH.md` - Pitfall 5 gained a new "Correction (2026-07-26)" paragraph beneath the existing "Resolved (2026-07-26)" paragraph, cross-referencing the T-3-08 second amendment

## Decisions Made
- Used `ensure_ascii=True` (matching the notebook's existing escaped-unicode convention, per the `260726-epz`/`260726-gj6` precedent) when re-serializing the patched notebook JSON, keeping the diff scoped to the 2 edited cells plus the 2 inserted cells.
- Inserted the new pre-flight cell pair immediately after the crcmod-reinstall cell (`bbe8de9d`) and before the server-launch cell (`7a49e230`), per the plan's explicit ordering requirement — cell order is the actual mechanism by which the env vars reach `dict(os.environ, ...)` two cells later, not just the explanatory comment.
- Confirmed cell `7a49e230`'s `outputs`/`execution_count` were already cleared in the working tree, so no re-clearing of already-clean state was needed; the one added comment did not require touching them further.
- Preserved the original T-3-08 Mitigation Plan text, the first `260726-gj6` amendment, and Pitfall 5's original prose plus its first `Resolved (2026-07-26)` paragraph unmodified; new dated sub-notes were appended in place, per the established repo pattern.

## Deviations from Plan
None - plan executed exactly as written. One minor formatting cleanup during self-review: an initial double-blank-line artifact before the "Second amendment" markdown paragraph (from naive string concatenation) was tightened to a single blank line to match the existing paragraph-spacing convention in that cell, before the final commit — not a deviation from plan intent, just formatting polish within the same task.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required. A live Colab re-run of Notebook B (to confirm the pre-flight check reports the compiled crcmod extension as visible, or that the `check_hashes=if_fast_else_skip` fallback lets the download complete anyway) remains a follow-up step for the project owner, as explicitly scoped out of this plan's `<verification>` section (already covered by `03-03-PLAN.md`'s Task 4 human-verify checkpoint, which this fix unblocks a third time).

## Next Phase Readiness
- Notebook B's checkpoint-download pre-flight fix is in place and documented; the project owner can re-run it on Colab to confirm `gsutil` now sees the compiled crcmod extension (or that the boto fallback lets the download complete anyway) and unblock 03-03's Task 4 sign-off checkpoint.
- No changes were made to `Pi0Backend`, `run_suite`/`eval_loop`, the localhost-only bind, the openpi-client legitimacy checkpoint, the `/content/openpi_data` local-disk path/value, or the earlier `--env=LIBERO` fix from `260726-epz` — scope stayed strictly within the pre-flight cell pair, cell 10's second amendment, the server-launch comment, and the two cross-referenced documentation records.

## Known Stubs
None - no placeholder/stub patterns introduced by this plan's changes.

## Threat Flags
None - this plan only sets two local compatibility/config environment variables (`CLOUDSDK_PYTHON_SITEPACKAGES`, `BOTO_CONFIG`) that change which Python site-packages gsutil's bundled interpreter consults and how it handles its own integrity-check overhead. No new network endpoint, install source, credential surface, or trust boundary is introduced. Pre-existing threats (T-3-SC, T-3-06, T-3-07, T-3-10) remain valid and unchanged; T-3-08's own mitigation text is amended in place a second time (not replaced), per this plan's own threat model (T-3-11, disposition: accept).

---
*Quick task: 260726-hbb*
*Completed: 2026-07-26*

## Self-Check: PASSED

- FOUND: libero/notebooks/03b-pi0-inference-smoketest.ipynb
- FOUND: .planning/phases/03-vla-inference-loop/03-03-PLAN.md
- FOUND: .planning/phases/03-vla-inference-loop/03-RESEARCH.md
- FOUND: .planning/quick/260726-hbb-fix-libero-notebooks-03b-pi0-inference-s/260726-hbb-SUMMARY.md
- FOUND commit: c6ba5f6
- FOUND commit: 48bfd38
