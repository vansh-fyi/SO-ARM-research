---
phase: 03-vla-inference-loop
plan: 260726-p0o
status: complete
subsystem: vla-inference
tags: [openpi, pi0, jupyter-notebook, subprocess-pipe-deadlock, editable-install, pth-finder]

requires:
  - phase: 03-vla-inference-loop (260726-io0)
    provides: "Notebook B with Block A2 (LIBERO sim stack) — got the VLA-04 cell past ModuleNotFoundError: robosuite on live Colab"
provides:
  - "serve_policy.py output redirected to /content/serve_policy.log instead of an undrained subprocess.PIPE (suspected cause of the 95-minute silent hang)"
  - "openpi-client install cell activates the editable .pth finder in the running kernel via site.addsitedir (fix confirmed working live by the user before being folded in)"
affects: [03-vla-inference-loop VLA-04 sign-off, next Colab run (L4)]

key-files:
  modified:
    - libero/notebooks/03b-pi0-inference-smoketest.ipynb

key-decisions:
  - "Executed inline by the orchestrator: the planner subagent was terminated by the monthly spend limit mid-dispatch; the edits were already fully specified, so they were applied directly rather than re-spawning agents. PLAN.md written retroactively to keep the quick-task artifact trail complete."
  - "Cell b8dacb5b: pip install openpi-client can be a 'requirement already satisfied' no-op because openpi's uv pip install -e . side-effect-registers openpi-client as an editable install in the kernel's system Python; the __editable__*.pth finder is only processed at interpreter startup, so site.addsitedir(sysconfig.get_paths()['purelib']) re-processes it in the running kernel — this exact fix was tested live by the user (import succeeded, Pi0Backend became a real class) before being committed."
  - "Cell 7a49e230: stdout=subprocess.PIPE with no reader is a textbook subprocess deadlock — server blocks writing logs once the ~64KB pipe buffer fills. Suspected (timeline-consistent, not traceback-confirmed — the A100 session ended before the interrupt diagnostic ran) cause of the 95-minute hang. Log-file redirect is correct regardless."

verification:
  - "JSON valid, 24 cells, only 2 cells changed (31-line diff)"
  - "grep: stdout=_server_log present at the Popen kwarg; stdout=subprocess.PIPE remains only inside the explanatory comment; site.addsitedir present; SERVE_LOG present"

deviations:
  - "Mystery-sync incident #5: the working copy had 25 cells (user's live Colab state incl. their manually-added salvage cell) when the first patch attempt ran; stashed (stash: pre-260726-p0o) and re-patched the clean committed 24-cell version."
---

# Quick Task 260726-p0o: Harden Notebook B (pipe deadlock + editable-install activation)

Two live-confirmed failure modes fixed in `libero/notebooks/03b-pi0-inference-smoketest.ipynb` ahead of the next Colab run (L4, after A100 availability lapsed):

1. **Server stdout pipe deadlock** — `serve_policy.py` now writes to `/content/serve_policy.log` (watch with `!tail -n 40 /content/serve_policy.log`); the exited-early error branch reads the log tail instead of the dead pipe.
2. **openpi-client editable `.pth` activation** — the install cell now runs `site.addsitedir()` and verifies `import openpi_client` inline, so a fresh top-to-bottom run cannot silently produce `Pi0Backend = None` again.
