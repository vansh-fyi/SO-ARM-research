---
phase: 03-vla-inference-loop
plan: 260802-itm
type: execute
wave: 1
depends_on: ["260802-ijb"]
files_modified:
  - libero/notebooks/01-colab-env-setup.ipynb
  - libero/notebooks/02-soarm-integration-check.ipynb
  - libero/notebooks/03a-oft-inference-eval.ipynb
  - libero/notebooks/03b-pi0-inference-smoketest.ipynb
autonomous: true
requirements: []
---

<objective>
User uses the VS Code Colab extension, which has no Secrets key-icon UI — the
GITHUB_TOKEN via `userdata.get()` approach from 260802-ijb doesn't work for them.
Requested: paste the token directly into a cell. Rather than a literal hardcoded
value (which would leak into git history the moment the notebook is committed —
a real risk given this project's recurring unexplained local-sync issue on these
exact notebook files), switched to `getpass.getpass()`: same one-input-per-session
convenience, but the value never gets written into the notebook's saved source.
User explicitly chose this over the literal-hardcode option when asked.
</objective>

<tasks>

<task type="auto">
  <name>Task 1: Replace userdata.get() with getpass() in all 4 notebooks' delivery cells + the HF token cell</name>
  <files>libero/notebooks/*.ipynb (4 files)</files>
  <action>
    Delivery cell (all 4 notebooks): getpass() prompts for GITHUB_TOKEN only on a
    fresh clone (REPO_ROOT doesn't exist yet) — subsequent `git pull`s on the same
    Colab VM reuse the token already embedded in .git/config's stored remote URL,
    so no re-prompt. Token variable is `del`eted immediately after use.
    Notebook 01's HF-token cell: same getpass() swap, optional (Enter to skip,
    stays anonymous — preserves existing fallback behavior).
  </action>
  <verify>
    <automated>All 4 notebooks valid JSON; zero `userdata` references remain in any of the 4 target notebooks; `getpass` present in all 4; no cell ever prints the token, _authed_url, or _token variables.</automated>
  </verify>
  <done>All 4 notebooks prompt for tokens via getpass() instead of Colab Secrets; no token value ever lands in committed notebook source.</done>
</task>

</tasks>
