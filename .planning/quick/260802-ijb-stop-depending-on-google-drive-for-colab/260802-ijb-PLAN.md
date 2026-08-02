---
phase: 03-vla-inference-loop
plan: 260802-ijb
type: execute
wave: 1
depends_on: []
files_modified:
  - .gitignore
  - libero/**  (flattened from nested git repo into main repo tracking)
  - libero/notebooks/01-colab-env-setup.ipynb
  - libero/notebooks/02-soarm-integration-check.ipynb
  - libero/notebooks/03a-oft-inference-eval.ipynb
  - libero/notebooks/03b-pi0-inference-smoketest.ipynb
autonomous: true
requirements: []
---

<objective>
Stop depending on Google Drive for Colab code delivery. LIBERO/ was its own nested
git repo (upstream ARISE-Initiative/LIBERO history, 316MB), blanket-gitignored, and
delivered to Colab via a manually-regenerated zip on Google Drive. User's explicit
ask: depend on GitHub, not Drive. Repo stays PRIVATE (user's explicit decision,
research sensitivity) — Colab needs a GitHub token, sourced from Colab Secrets.

Executed inline by the orchestrator (well-specified, high-precision multi-file
notebook surgery; no subagent dispatch needed).
</objective>

<tasks>

<task type="auto">
  <name>Task 1: Flatten LIBERO/, update .gitignore, rewrite notebook delivery cells</name>
  <files>.gitignore, libero/ (all), libero/notebooks/*.ipynb</files>
  <action>
    1. .gitignore: remove the blanket `LIBERO/` line; add `**/__pycache__/` and
       `**/.pytest_cache/` (neither the main nor LIBERO's own nested .gitignore
       covered these). LIBERO's own nested .gitignore (datasets, results,
       outputs/*, *.mp4, wandb, egg-info, .hydra) applies automatically once
       LIBERO/ is no longer itself blanket-ignored.
    2. `rm -rf LIBERO/.git` — drops the nested repo's own history (upstream
       benchmark history this project doesn't extend/contribute back to); the
       working tree content is what actually runs and is what gets tracked.
    3. Patch all 4 notebooks: replace/insert a canonical delivery cell that
       reads a `GITHUB_TOKEN` from Colab Secrets (`google.colab.userdata`) and
       clones (first run) or pulls (subsequent runs) the private repo via
       `subprocess.run` — never shell-magic, so the token can never be echoed
       to cell output — and deliberately withholds the raw git error message
       on failure so a bad/expired token never leaks. Notebook 01 gets an
       INSERTED cell (it never had a delivery cell — assumed pre-mounted
       Drive); notebooks 02/03a/03b get a straight swap of their existing
       `drive.mount()`+`unzip` cell. Strip stale "If using Google Drive: ..."
       comment lines from path-constant cells. Notebook 01's HF-token
       bootstrap cell also switches from a Drive-file read to
       `userdata.get("HF_TOKEN")`, preserving the anonymous-access fallback.
    4. Pre-commit safety scan (must all pass before committing): no `.env`
       staged, no `__pycache__`/`egg-info`/`.pytest_cache` staged, no file over
       50MB, no common secret-pattern regex matches (AWS keys, private key
       headers, `sk-`/`ghp_`/`xox`/`AIza` prefixes) in any staged file content,
       and confirm `LIBERO/.env` (a real file that exists on disk, contents
       never read — blocked by permission settings) remains untracked.
  </action>
  <verify>
    <automated>All 4 notebooks valid JSON; `drive.mount` absent from every notebook; `GITHUB_TOKEN` present in every notebook; no cell ever prints the token or the token-bearing URL string; pre-commit safety scan (above) clean.</automated>
  </verify>
  <done>LIBERO/ tracked directly in the main repo (no nested .git); all 4 notebooks clone/pull from GitHub via a Colab-Secrets token instead of mounting Drive; committed and pushed to both `main` and `master`.</done>
</task>

</tasks>

<verification>
- `git diff --cached --name-only | wc -l` → 1146 files staged before commit, ~452MB total, zero flagged by any safety check above.
- Post-commit: `git log --oneline -1` shows the flatten commit; `git push origin master:main` and `master:master` both succeeded.
</verification>
