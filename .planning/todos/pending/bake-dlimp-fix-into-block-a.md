---
title: Bake kvablack/dlimp fix into Block A and delete the failing moojink cell
date: 2026-07-10
priority: high
phase: 01-colab-environment-setup
---

## Task

Close out the last remaining Phase 1 item from `DLIMP-PATCH.md`: make the dlimp
install a permanent part of `libero/notebooks/01-colab-env-setup.ipynb` Block A,
and remove the non-functional patch cell from Block B.

## Changes

1. **Add to Block A** (in or immediately after the Step 5 openvla-oft cell,
   before the numpy ABI gate — cell `98b86f1a` per DLIMP-PATCH.md):

   ```python
   # ── dlimp install (required transitive dep of openvla-oft) ─────────────────
   # pip install git+https://github.com/moojink/dlimp_openvla is unsatisfiable on
   # Colab py3.12 (pins tensorflow==2.15.0; no cp312 wheel exists).
   # Clone-then-editable of the parent repo is the confirmed fix.
   import subprocess as _sp, sys as _sys, pathlib as _pl

   _DLIMP_DIR = _pl.Path("/content/dlimp_kvablack")
   if not _DLIMP_DIR.exists():
       print("Cloning kvablack/dlimp ...")
       _sp.run(["git", "clone", "--depth", "1",
                "https://github.com/kvablack/dlimp", str(_DLIMP_DIR)], check=True)
   _sp.run([_sys.executable, "-m", "pip", "install", "-q", "-e", str(_DLIMP_DIR)],
           check=True)
   print("dlimp installed from kvablack/dlimp ✓")
   ```

   Block A placement matters twice over: dlimp must be on disk before Block B's
   prismatic import, and the numpy ABI gate (Block A's final cell) repairs any
   numpy disturbance the install causes before the restart.

2. **Delete Block B cell 20** (the `pip install git+.../dlimp_openvla` cell before
   the prismatic guard). It always fails with return code 1 and installs nothing —
   see `.planning/notes/dlimp-install-forensics.md`.

   `.ipynb` edits must be done via NotebookEdit / `nbformat` scripting or in Colab.

## Acceptance criteria

- [ ] In a live session, `import dlimp; print(dlimp.__file__)` resolves to
      `/content/dlimp_kvablack/...`
- [ ] Clean-run verification on a **deleted** runtime (Runtime → Disconnect and
      delete runtime — a mere restart keeps the VM disk and invalidates the test):
      Block A → restart → Block B passes ENV-01, ENV-02, ENV-03 with zero manual
      intervention
- [ ] ENV-03 prints `PASS — action shape: (8, 7), dtype: float64`
- [ ] Update `DLIMP-PATCH.md` UAT record / mark the remaining task done
