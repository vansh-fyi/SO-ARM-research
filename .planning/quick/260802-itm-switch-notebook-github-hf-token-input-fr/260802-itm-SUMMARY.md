---
phase: 03-vla-inference-loop
plan: 260802-itm
status: complete
subsystem: infra
tags: [colab, secrets, getpass, vscode]

requires:
  - phase: 03-vla-inference-loop (260802-ijb)
    provides: "GitHub-clone-based Colab delivery (userdata.get-based, assumed standard Colab UI)"
provides:
  - "Token input via getpass() in all 4 notebooks — works in the VS Code Colab extension, which has no Secrets key-icon UI"
affects: [all future Colab sessions run via VS Code's Colab extension]

key-files:
  modified:
    - libero/notebooks/01-colab-env-setup.ipynb
    - libero/notebooks/02-soarm-integration-check.ipynb
    - libero/notebooks/03a-oft-inference-eval.ipynb
    - libero/notebooks/03b-pi0-inference-smoketest.ipynb

key-decisions:
  - "getpass() chosen over a literal hardcoded token value after presenting the tradeoff explicitly: user initially asked to hardcode the token directly in a cell (\"this is only for a single repo which is private so it is okay\"), but given this project's recurring unexplained local-sync issue on these exact notebook files (6 sightings this session), a hardcoded value risks accidental permanent leakage into git history. getpass() delivers the identical UX (paste once per session) with zero write-to-source risk. User chose getpass() when given both options."
  - "Token only re-prompted on a fresh clone (new Colab VM) — subsequent git pulls reuse the credential already cached in that VM's local .git/config, which never syncs back to the repo"

verification:
  - "All 4 notebooks: valid JSON, zero userdata references remain, getpass present, no cell prints the token/URL/temp variables"
---

# Quick Task 260802-itm: getpass() Instead of Colab Secrets for Token Input

Follow-up to 260802-ijb: the GITHUB_TOKEN-via-Colab-Secrets approach doesn't work for the user's VS Code Colab extension workflow (no Secrets UI). Switched all 4 notebooks' delivery cells (plus Notebook 01's optional HF-token cell) to `getpass.getpass()` — same one-paste-per-session convenience the user asked for, but the token is never written into the notebook's saved source, avoiding a permanent git-history leak if this project's recurring local-notebook-sync issue touches these files again.
