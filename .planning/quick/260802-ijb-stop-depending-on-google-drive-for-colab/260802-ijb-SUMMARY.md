---
phase: 03-vla-inference-loop
plan: 260802-ijb
status: complete
subsystem: infra
tags: [github, colab, gitignore, secrets, delivery-mechanism]

requires:
  - phase: 03-vla-inference-loop (all prior plans)
    provides: "Working Colab notebooks previously dependent on a manually-regenerated Drive zip"
provides:
  - "LIBERO/ tracked directly in the main git repo — no more nested repo, no more blanket gitignore"
  - "All 4 Colab notebooks clone/pull from GitHub via a Colab-Secrets token, replacing drive.mount()+unzip"
affects: [all future Colab sessions, no more manual zip-regeneration step for the orchestrator]

key-files:
  modified:
    - .gitignore
    - libero/notebooks/01-colab-env-setup.ipynb
    - libero/notebooks/02-soarm-integration-check.ipynb
    - libero/notebooks/03a-oft-inference-eval.ipynb
    - libero/notebooks/03b-pi0-inference-smoketest.ipynb
  created:
    - libero/** (1146 files, ~452MB — flattened from the previously-nested LIBERO/.git)

key-decisions:
  - "Repo stays PRIVATE — explicit user decision, reversing an earlier lean toward public, over research-sensitivity concerns (\"anyone could steal our research\")"
  - "LIBERO's nested .git (316MB, upstream ARISE-Initiative/LIBERO history) was dropped, not preserved as a submodule — a submodule would just relocate the same 'Colab needs auth to pull it' problem the user is trying to escape; the vendored working tree is what actually matters, not commit-by-commit upstream history this project doesn't extend"
  - "Token delivery: subprocess.run, never shell-magic — avoids any risk of git echoing the token-bearing URL to cell output, and the raw git error is deliberately withheld on auth failure so a bad token can't leak that way either"
  - "userdata.get()'s previously-documented hang risk (01-DEBUG-HISTORY.md, specific to non-standard Colab frontends) is called out directly in both new Secrets-reading cells with a one-off manual-paste fallback, rather than silently assuming it'll work"

verification:
  - "Pre-commit safety scan: 1146 files / ~452MB staged, zero .env/__pycache__/egg-info/.pytest_cache, zero files >50MB, zero secret-pattern regex matches in any staged content, LIBERO/.env confirmed still untracked"
  - "All 4 notebooks: valid JSON, drive.mount absent, GITHUB_TOKEN present, no cell prints the token or token-bearing URL"

user_setup_required: "Create a GitHub Personal Access Token (fine-grained, scoped to vansh-fyi/SO-ARM-research, Contents: Read-only) and add it to Colab Secrets as GITHUB_TOKEN in each Colab account used — one-time setup."
---

# Quick Task 260802-ijb: Drop Google Drive, Depend on GitHub Instead

Flattened `LIBERO/` (previously its own nested git repo, blanket-gitignored, 765MB total including 316MB of its own upstream history) into the main repo's own git tracking, and rewrote all 4 Colab notebooks' delivery cells to clone/pull the private GitHub repo directly via a Colab-Secrets-stored token, replacing `drive.mount()` + `unzip SoARM-Research-colab.zip`.

This retires the manual "re-zip LIBERO/ and swap it into the user's Drive" step the orchestrator had been doing after every code change this session — future code changes just need a `git push`.
