---
phase: quick-260927-ndz
plan: 01
subsystem: vla_bridge
tags: [depth, fastfs, colab, docs-sync]
dependency-graph:
  requires: [depth_camera.py (Plan 12-06), policy_server_launch.md Steps 3-4 (Plan 11-03)]
  provides: [real FastFS run_demo.py depth inference cell, corrected port-5173 docs, synced camera-mapping cell]
  affects: [control/vla_bridge/policy_server_launch.md, control/vla_bridge/policy_server.ipynb]
tech-stack:
  added: []
  patterns: ["subprocess.run against a documented CLI instead of a fabricated Python API", "honestly-marked TODO constant (DEPTH_OUTPUT_GLOB) for a genuinely undocumented detail"]
key-files:
  created: []
  modified:
    - control/vla_bridge/policy_server_launch.md
    - control/vla_bridge/policy_server.ipynb
decisions:
  - "Kept the target cell-0 replacement text's historical mention of '2560x720'/'1280x360' (describing the still-broken/previously-used modes) verbatim, matching policy_server_launch.md's own current, already-correct wording -- see Deviations section."
metrics:
  duration: "~25 minutes"
  completed: 2026-09-27
status: complete
---

# Quick Task 260927-ndz: Fix Phase 12 UAT Test-5 Gaps in the Colab Depth Endpoint Summary

Wired Fast-FoundationStereo's documented `scripts/run_demo.py` CLI into the Colab Flask `/depth` handler via `subprocess`, replacing an undefined stub, and fixed three adjacent documentation-drift bugs (wrong port-8080 references, stale notebook camera-mapping text, duplicate tunnel cell) flagged by the same UAT test.

## What Was Built

**Task 1 — Real FastFS inference wired into the Step 8 Flask cell (both files)**

- `policy_server_launch.md`: inserted a new pip-install code fence right after the Step 7 git-clone fence (FastFS's documented "Option 2: pip" environment setup — `torch==2.6.0`/`torchvision==0.21.0`/`xformers` + `requirements.txt`), replaced the old "Then follow that repo's own README..." paragraph with one describing the manual checkpoint-download step and the new run_demo.py-based Step 8, and replaced the entire Step 8 code fence with a real implementation.
- `policy_server.ipynb`: same three changes applied by id-based cell lookup (never hardcoded index) — inserted a new code cell after `6e04ff98`, replaced the `source` of markdown cell `2e55f487`, and replaced the `source` of code cell `8473a692`.
- The new `run_fastfs_inference()` writes `left.png`/`right.png`/`K.txt` to a `tempfile.TemporaryDirectory()`, writes the intrinsics file in the exact documented format (line 1 = flattened 1x9 intrinsics, line 2 = baseline in meters), invokes `python scripts/run_demo.py` via `subprocess.run(..., cwd=FASTFS_DIR, check=True)` with the exact documented flags, and loads the resulting depth map via a `DEPTH_OUTPUT_GLOB = "*.npy"` constant carrying an explicit `# TODO: confirm exact filename by running once in Colab and inspecting out_dir` comment — the one genuinely undocumented FastFS detail, marked honestly rather than guessed.
- `/depth` route body (request parsing + JSON response) is unchanged; `depth_camera.py` was inspected only, never modified.

**Task 2 — Port-8080 fixes, notebook camera-mapping sync, duplicate tunnel-cell removal**

- `policy_server_launch.md`: changed the Step 9 depth-tunnel cell's comment from "(8080)" to "(5173)", and fixed the same wrong value independently present in the Summary table's `Tunnel type` row (also "8080" → "5173"). No other 8080/5173 text in the file was touched.
- `policy_server.ipynb`:
  - Fixed the identical "(8080)" → "(5173)" typo in code cell `553ee9ec`'s comment.
  - Replaced markdown cell `5da4b39e`'s entire `source` with the .md's current preamble + Camera mapping text verbatim (1600x600/800x600-split AR0144 wording, replacing the stale 2560x720/1280-split text).
  - Deleted the duplicate, uncommented tunnel-open cell `wd94PYZ53t39`; confirmed the commented, kept cell `fc7fd0ee` is still present and unmodified.
- Notebook remains valid, parseable JSON (`nbformat: 4`, `nbformat_minor: 5`, same top-level key order) with 36 total cells (net unchanged: +1 pip-install cell from Task 1, -1 duplicate tunnel cell from Task 2).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - blocking verify-script defect, not a content bug] Task 2's automated verify command contradicted its own target content**

- **Found during:** Task 2 verification
- **Issue:** Task 2's `<automated>` verify block and the plan's overall `<verification>` section both assert `grep -q "2560x720"` finds nothing in `policy_server.ipynb`. But the plan's own "Target notebook cell-0 replacement source" context block — which is a verbatim copy of `policy_server_launch.md`'s current, already-correct Camera mapping wording — legitimately contains the sentence "...the still-broken 2560x720 mode remains frozen even after exhausting the last known macOS system-level fix..." (historical/diagnostic context about a *different*, currently-unused high-res mode, not the active resolution). `policy_server_launch.md` itself (unedited, pre-existing, correct) contains this exact same "2560x720" substring today. A blanket zero-occurrences grep is therefore unsatisfiable without deviating from the plan's own specified verbatim target text and from the .md it's meant to match.
- **Fix:** Applied the verbatim target content exactly as specified (matching `policy_server_launch.md`'s current wording character-for-character in cell 0 — confirmed via `diff`), and substituted the intended, narrower checks in its place: no `native 2560x720` (the old ACTIVE-resolution phrasing) and no `` columns`[0:1280]` `` (the old stale crop range) remain in cell 0. Also independently confirmed no `1280x360` substring remains in the .md's own unrelated text.
- **Files verified:** `control/vla_bridge/policy_server.ipynb` (cell `5da4b39e`), `control/vla_bridge/policy_server_launch.md`
- **Commits:** a9ab896 (the content itself was correct on first write; this entry documents the verify-command discrepancy discovered while checking it, no code was changed in response)

No other deviations — remaining work executed exactly as written.

## Self-Check: PASSED

- FOUND: control/vla_bridge/policy_server_launch.md
- FOUND: control/vla_bridge/policy_server.ipynb
- FOUND: daf48b8 (Task 1 commit)
- FOUND: a9ab896 (Task 2 commit)
