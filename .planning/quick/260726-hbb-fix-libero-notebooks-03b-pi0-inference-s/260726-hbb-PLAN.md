---
phase: 03-vla-inference-loop
plan: 260726-hbb
type: execute
wave: 1
depends_on: []
files_modified:
  - libero/notebooks/03b-pi0-inference-smoketest.ipynb
  - .planning/phases/03-vla-inference-loop/03-03-PLAN.md
  - .planning/phases/03-vla-inference-loop/03-RESEARCH.md
autonomous: true
requirements: [VLA-04]

must_haves:
  truths:
    - "Notebook B has a new pre-flight cell (inserted between the crcmod-reinstall cell and the server-launch cell) that sets os.environ[\"CLOUDSDK_PYTHON_SITEPACKAGES\"] = \"1\", so gsutil's own bundled Cloud SDK Python interpreter falls back to the kernel's site-packages where crcmod's compiled C extension actually landed — the real root cause per GoogleCloudPlatform/gsutil#1429, not the previously-assumed Drive-vs-local destination issue"
    - "The same pre-flight cell writes /content/.boto with check_hashes = if_fast_else_skip, sets os.environ[\"BOTO_CONFIG\"] accordingly, and runs a gsutil version -l | grep -i crcmod check that prints plainly BEFORE the real 11.6GB checkpoint download is attempted, so the user gets fast feedback instead of a third blind multi-minute failure"
    - "The server-launch cell's env=dict(os.environ, ...) call passed to subprocess.Popen is confirmed (via an explanatory comment, not new/duplicated logic) to already carry CLOUDSDK_PYTHON_SITEPACKAGES and BOTO_CONFIG into serve_policy.py's subprocess, since dict(os.environ, ...) reads os.environ at call time"
    - "03-03-PLAN.md's T-3-08 threat-model row carries a SECOND dated 2026-07-26 amendment (the first sub-note, from 260726-gj6, is preserved unmodified) documenting that the local-disk destination change alone did not fix the download, plus the corrected root cause and fix; 03-RESEARCH.md's Pitfall 5 carries a matching cross-referenced correction note beneath its existing 260726-gj6 'Resolved' paragraph"
  artifacts:
    - libero/notebooks/03b-pi0-inference-smoketest.ipynb
    - .planning/phases/03-vla-inference-loop/03-03-PLAN.md
    - .planning/phases/03-vla-inference-loop/03-RESEARCH.md
  key_links:
    - "The pre-flight cell must execute (so its os.environ writes land) BEFORE the server-launch cell (id=7a49e230) runs — dict(os.environ, ...) only inherits what's already in os.environ at call time, so notebook cell ORDER is the actual mechanism, not just the comment documenting it"
    - "CLOUDSDK_PYTHON_SITEPACKAGES=1 is a Python os.environ assignment (not a shell !export), specifically so it persists process-wide and reaches the subprocess.Popen env dict built two cells later"
---

<objective>
Fix the THIRD occurrence of the same Notebook B checkpoint-download failure (`CommandException: 6 files/objects could not be transferred` downloading `pi05_libero`'s 11.6GB/16-shard checkpoint) by correcting the root-cause diagnosis: the 260726-gj6 fix's theory (Drive FUSE unreliability) is now proven wrong by a live Colab re-run with local disk + reinstalled crcmod that hit the identical failure. The real mechanism, confirmed via `GoogleCloudPlatform/gsutil#1429`, is that `gsutil`'s bundled Cloud SDK Python interpreter is isolated from the Colab kernel's Python that `pip install crcmod` targets — the compiled C extension is invisible to gsutil regardless of a successful kernel-side install. Add `CLOUDSDK_PYTHON_SITEPACKAGES=1`, a pre-flight verification cell, and a `check_hashes=if_fast_else_skip` boto fallback as defense-in-depth against the documented ABI-incompatibility caveat.

Purpose: Avoid a fourth blind multi-minute failed download attempt by giving the user fast pre-flight feedback on whether gsutil now sees the compiled crcmod extension, while a hash-check-skip fallback ensures the download still completes even if it doesn't.
Output: A corrected, JSON-valid `03b-pi0-inference-smoketest.ipynb` with a new pre-flight diagnostic cell pair and updated markdown explaining the corrected root cause; a second dated amendment on 03-03-PLAN.md's T-3-08 row; a matching second dated note on 03-RESEARCH.md's Pitfall 5.
</objective>

<execution_context>
@/Users/hp/Desktop/Work/Repositories/SoARM-Research/.claude/gsd-core/workflows/execute-plan.md
@/Users/hp/Desktop/Work/Repositories/SoARM-Research/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@.planning/phases/03-vla-inference-loop/03-CONTEXT.md
@.planning/phases/03-vla-inference-loop/03-RESEARCH.md
@.planning/phases/03-vla-inference-loop/03-03-PLAN.md
@.planning/quick/260726-gj6-fix-libero-notebooks-03b-pi0-inference-s/260726-gj6-SUMMARY.md
</context>

<tasks>

<task type="auto">
  <name>Task 1: Notebook B — CLOUDSDK_PYTHON_SITEPACKAGES pre-flight cell + boto check_hashes fallback</name>
  <files>libero/notebooks/03b-pi0-inference-smoketest.ipynb</files>
  <read_first>
    libero/notebooks/03b-pi0-inference-smoketest.ipynb (the file being fixed — this repo has a case-insensitive-macOS quirk where the file is only git-tracked at the LOWERCASE path `libero/notebooks/...`; edit that path, not `LIBERO/notebooks/...`. Load with Python's `json` module, not a text editor — edits are keyed by each cell's stable `id` field: `"1b668f9c"` (markdown, cell 10), `"bbe8de9d"` (code, cell 11, crcmod reinstall), `"7a49e230"` (code, cell 12, server-launch). Current cell count is 17; two new cells are inserted, making 19.)
    .planning/quick/260726-gj6-fix-libero-notebooks-03b-pi0-inference-s/260726-gj6-SUMMARY.md (the PRIOR fix to this same cell earlier today, whose root-cause theory — Drive FUSE unreliability — is now proven WRONG by an identical live-Colab failure after switching to local disk; do not repeat that claim as the cause. Follow its exact re-serialization pattern: `json.dump(nb, f, indent=1, ensure_ascii=True)` plus a trailing newline — this file's established convention.)
  </read_first>
  <action>
    Write a throwaway Python script (run via Bash, e.g. `python3 - <<'PY' ... PY`, discard after) that loads the notebook with `json.load`, applies the three edits below keyed by cell `id`, and writes back with `json.dump(nb, f, indent=1, ensure_ascii=True)` followed by a trailing newline. Do not hand-edit the raw JSON text.

    Edit 1 — insert two brand-new cells immediately after code cell `id="bbe8de9d"` (crcmod reinstall) and before code cell `id="7a49e230"` (server-launch):
    (a) A new markdown cell (fresh 8-hex `id` via `uuid.uuid4().hex[:8]`, not colliding with any existing cell id) headed "### Pre-flight: verify gsutil sees the compiled crcmod extension". Body explains: root cause confirmed via `GoogleCloudPlatform/gsutil#1429` — gsutil's bundled Cloud SDK Python interpreter is ISOLATED from the Colab kernel's Python that `pip install crcmod` targets above, so even though the "Building wheel ... done" message shows the extension compiled successfully in the kernel's Python, gsutil's own bundled interpreter has a separate site-packages and never sees it, regardless of that success. `CLOUDSDK_PYTHON_SITEPACKAGES=1` (set in the cell below) makes gsutil's bundled interpreter fall back to the kernel/system site-packages instead of its own isolated one — with the documented caveat that this only works if the bundled interpreter's Python version/ABI is compatible with the compiled extension, so it might not always work. As defense-in-depth, the cell below also sets `check_hashes = if_fast_else_skip` via a `/content/.boto` config — gsutil's own officially-sanctioned partial integrity-check skip for exactly this "C extension unavailable" scenario; it only skips the check when it would otherwise be slow, not outright. State that this cell's `gsutil version -l` check runs BEFORE the real download so the user sees fast feedback instead of a fourth blind multi-minute failed attempt, and that if it still shows the extension unavailable, the `check_hashes=if_fast_else_skip` fallback should still let the download succeed, just without CRC32C verification on this run.
    (b) A new code cell (fresh 8-hex `id`, `execution_count: null`, `outputs: []`) whose source: sets `os.environ["CLOUDSDK_PYTHON_SITEPACKAGES"] = "1"` (a Python `os.environ` assignment, not a shell `!export`, so it persists into the later `subprocess.Popen` env for `serve_policy.py`); writes a minimal `/content/.boto` file containing exactly `[GSUtil]\ncheck_hashes = if_fast_else_skip\n` and sets `os.environ["BOTO_CONFIG"] = "/content/.boto"`; then runs `!gsutil version -l 2>&1 | grep -i crcmod` and prints the output plainly. Include a short leading comment citing `GoogleCloudPlatform/gsutil#1429` as the source of this fix.

    Edit 2 — markdown cell `id="1b668f9c"` (the "Start `serve_policy.py --env=LIBERO`" header, already carrying one 2026-07-26 amendment from `260726-gj6` about the Drive-to-local-disk switch): APPEND (do not delete or reword the existing amendment/accepted-trade-off text — it remains historical record) a new paragraph beginning `**Second amendment (2026-07-26):**` stating: the local-disk `OPENPI_DATA_HOME` change from the first amendment did NOT fix the download by itself — a live Colab re-run hit the identical `CommandException: 6 files/objects could not be transferred` failure even with local disk and the crcmod reinstall in place. The actual root cause, confirmed via `GoogleCloudPlatform/gsutil#1429`, is that gsutil's bundled Cloud SDK Python interpreter is isolated from the Colab kernel's Python that `pip install crcmod` targets, so the compiled C extension stays invisible to gsutil regardless of the kernel-side install succeeding. State the fix: the pre-flight cell below sets `CLOUDSDK_PYTHON_SITEPACKAGES=1` plus a `check_hashes=if_fast_else_skip` boto fallback as a safety net. Note that the existing local-disk explanation from the first amendment remains correct and necessary (Drive FUSE composite-object writes are still unreliable at this scale) — just insufficient on its own.

    Edit 3 — code cell `id="7a49e230"` (server-launch cell, containing `serve_cmd` and the `subprocess.Popen(...)` call): add ONE short comment immediately above the `server_proc = subprocess.Popen(` line — do not change any executable logic, do not duplicate or re-set any environment variable here — noting that `env=dict(os.environ, OPENPI_DATA_HOME=os.environ["OPENPI_DATA_HOME"])` reads `os.environ` at call time, so `CLOUDSDK_PYTHON_SITEPACKAGES` and `BOTO_CONFIG` set by the pre-flight cell above are already carried into this subprocess's environment with no extra wiring needed here — confirmed by inspecting this call's own semantics, not assumed. This cell's `outputs` (currently a large stale run's output, per the file listing) must be set to `[]` and `execution_count` to `null` since its source is changing.

    Do not modify any other cell. Do not touch the 127.0.0.1-only bind, Pi0Backend/run_suite code, episode/task counts, the `/content/openpi_data` local-disk path/value itself (keep it exactly as-is per 260726-gj6), or the earlier `--env=LIBERO` fix from `260726-epz`.
  </action>
  <acceptance_criteria>
    - `grep -q "CLOUDSDK_PYTHON_SITEPACKAGES" libero/notebooks/03b-pi0-inference-smoketest.ipynb` matches
    - `grep -q "check_hashes" libero/notebooks/03b-pi0-inference-smoketest.ipynb` and `grep -q "if_fast_else_skip" libero/notebooks/03b-pi0-inference-smoketest.ipynb` both match
    - `grep -q "gsutil#1429" libero/notebooks/03b-pi0-inference-smoketest.ipynb` matches (root-cause citation present)
    - `grep -q "BOTO_CONFIG" libero/notebooks/03b-pi0-inference-smoketest.ipynb` matches
    - `grep -q "Second amendment (2026-07-26)" libero/notebooks/03b-pi0-inference-smoketest.ipynb` matches, and the first amendment's text is still present (not deleted)
    - `python3 -c "import json; nb=json.load(open('libero/notebooks/03b-pi0-inference-smoketest.ipynb')); assert len(nb['cells'])==19"` confirms exactly two new cells were inserted
    - Cell `id="7a49e230"`'s `outputs` is `[]` and `execution_count` is `null`
  </acceptance_criteria>
  <verify>
    <automated>grep -q "CLOUDSDK_PYTHON_SITEPACKAGES" libero/notebooks/03b-pi0-inference-smoketest.ipynb && grep -q "check_hashes" libero/notebooks/03b-pi0-inference-smoketest.ipynb && grep -q "if_fast_else_skip" libero/notebooks/03b-pi0-inference-smoketest.ipynb && grep -q "gsutil#1429" libero/notebooks/03b-pi0-inference-smoketest.ipynb && grep -q "BOTO_CONFIG" libero/notebooks/03b-pi0-inference-smoketest.ipynb && grep -q "Second amendment (2026-07-26)" libero/notebooks/03b-pi0-inference-smoketest.ipynb && python3 -c "
import json
nb = json.load(open('libero/notebooks/03b-pi0-inference-smoketest.ipynb'))
assert len(nb['cells']) == 19, f'expected 19 cells, got {len(nb[\"cells\"])}'
by_id = {c.get('id'): c for c in nb['cells']}
c = by_id['7a49e230']
assert c['outputs'] == [], 'cell 7a49e230 outputs not cleared'
assert c.get('execution_count') is None, 'cell 7a49e230 execution_count not cleared'
print('Notebook pre-flight fix verified OK')
"</automated>
  </verify>
  <done>Notebook B has a new pre-flight cell pair (markdown header + code) inserted between the crcmod-reinstall cell and the server-launch cell that sets CLOUDSDK_PYTHON_SITEPACKAGES=1, writes a check_hashes=if_fast_else_skip /content/.boto fallback, and runs a fast gsutil version -l crcmod check before the real download; cell 10's markdown carries a second dated amendment explaining the corrected root cause (GoogleCloudPlatform/gsutil#1429); the server-launch cell has an explanatory comment confirming env inheritance and its stale output cleared; the notebook stays valid JSON with 19 cells.</done>
</task>

<task type="auto">
  <name>Task 2: Record the corrected root-cause decision revision in 03-03-PLAN.md and 03-RESEARCH.md</name>
  <files>.planning/phases/03-vla-inference-loop/03-03-PLAN.md, .planning/phases/03-vla-inference-loop/03-RESEARCH.md</files>
  <read_first>
    .planning/phases/03-vla-inference-loop/03-03-PLAN.md (the `<threat_model>` STRIDE Threat Register — locate the `T-3-08` row; it already carries one `**Amendment (2026-07-26):**` sentence from `260726-gj6` — append a SECOND dated sub-note, do not overwrite the first)
    .planning/phases/03-vla-inference-loop/03-RESEARCH.md ("### Pitfall 5" under "## Common Pitfalls" — it already carries one `**Resolved (2026-07-26):**` paragraph from `260726-gj6` — add a new cross-referenced paragraph beneath it, do not overwrite)
  </read_first>
  <action>
    In `03-03-PLAN.md`'s STRIDE Threat Register `T-3-08` row, within the same Mitigation Plan cell, APPEND (after the existing `**Amendment (2026-07-26):**` sentence — do not delete or reword it) a new sentence beginning `**Second amendment (2026-07-26):**` stating: the local-disk destination change alone did not fix the download — proven by a live Colab re-run hitting the identical `gsutil` `CommandException: 6 files/objects could not be transferred` failure with local disk and the crcmod reinstall already in place. State the corrected root cause: gsutil's own bundled Cloud SDK Python interpreter is isolated from the Colab kernel's site-packages, so the kernel's pip-installed `crcmod` compiled extension is invisible to gsutil regardless (cite `GoogleCloudPlatform/gsutil#1429`, not the Drive-vs-local hypothesis). State the fix: `CLOUDSDK_PYTHON_SITEPACKAGES=1` plus a `check_hashes=if_fast_else_skip` boto fallback as a safety net, implemented in `libero/notebooks/03b-pi0-inference-smoketest.ipynb`.

    In `03-RESEARCH.md`, directly beneath Pitfall 5's existing `**Resolved (2026-07-26):**` paragraph (added by `260726-gj6`), add a new paragraph beginning `**Correction (2026-07-26):**` stating: the "Resolved" note above turned out to be incomplete — the local-disk switch alone did not fix the download. The actual mechanism is gsutil's bundled Cloud SDK Python being isolated from the Colab kernel's site-packages (`GoogleCloudPlatform/gsutil#1429`), fixed via `CLOUDSDK_PYTHON_SITEPACKAGES=1` plus a `check_hashes=if_fast_else_skip` boto fallback. Cross-reference `03-03-PLAN.md`'s T-3-08 second amendment for the full threat-model-level rationale. Do not delete or reword the original Pitfall 5 text or the first "Resolved" paragraph — both remain historical record.
  </action>
  <acceptance_criteria>
    - The `T-3-08` row in `03-03-PLAN.md` still contains the original Drive-backed mitigation text AND the first `Amendment (2026-07-26)` sentence AND the new `Second amendment (2026-07-26)` sentence, all in the same cell
    - Pitfall 5 in `03-RESEARCH.md` still contains its original text, the first `Resolved (2026-07-26)` paragraph, AND the new `Correction (2026-07-26)` paragraph immediately after it
  </acceptance_criteria>
  <verify>
    <automated>grep -q "Second amendment (2026-07-26)" .planning/phases/03-vla-inference-loop/03-03-PLAN.md && grep -q "gsutil#1429" .planning/phases/03-vla-inference-loop/03-03-PLAN.md && grep -q "Amendment (2026-07-26)" .planning/phases/03-vla-inference-loop/03-03-PLAN.md && grep -q "Correction (2026-07-26)" .planning/phases/03-vla-inference-loop/03-RESEARCH.md && grep -q "gsutil#1429" .planning/phases/03-vla-inference-loop/03-RESEARCH.md && grep -q "Resolved (2026-07-26)" .planning/phases/03-vla-inference-loop/03-RESEARCH.md && echo OK</automated>
  </verify>
  <done>03-03-PLAN.md's T-3-08 threat-model row and 03-RESEARCH.md's Pitfall 5 both carry a SECOND dated 2026-07-26 amendment/correction note recording the corrected root cause (gsutil's isolated bundled-Python site-packages, GoogleCloudPlatform/gsutil#1429) and fix (CLOUDSDK_PYTHON_SITEPACKAGES=1 + check_hashes=if_fast_else_skip), cross-referencing each other, with all original text (including the first 260726-gj6 amendment) preserved as historical record.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

No new trust boundaries are introduced by this plan. It sets two local compatibility/config environment variables (`CLOUDSDK_PYTHON_SITEPACKAGES`, `BOTO_CONFIG`) that only change which Python site-packages `gsutil`'s bundled interpreter consults and how it handles its own integrity-check overhead — it does not add a new network endpoint, change any install source, or touch credential handling. No new packages are installed by this plan.

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-3-11 | Denial of Service | `libero/notebooks/03b-pi0-inference-smoketest.ipynb` — `gsutil`'s composite-object checkpoint download reliability (corrects T-3-08/T-3-10's prior Drive-vs-local-disk diagnosis) | low | accept | This plan only sets `CLOUDSDK_PYTHON_SITEPACKAGES=1` (compatibility fallback so gsutil's bundled interpreter sees the kernel's compiled crcmod extension) and `check_hashes=if_fast_else_skip` via a local `/content/.boto` file (gsutil's own sanctioned partial integrity-check skip) — no new network endpoint, install source, or credential surface is introduced. The pre-flight `gsutil version -l` check gives fast feedback before the real download runs. Pre-existing threats from 03-03-PLAN.md's threat model (T-3-SC package legitimacy, T-3-06 localhost-only bind, T-3-07 pinned commit SHA, T-3-10 local-disk destination) remain valid and unchanged. T-3-08's own mitigation text is amended in place a second time (not replaced) to record this corrected diagnosis — see Task 2. |

</threat_model>

<verification>
- Task 1's automated check confirms the pre-flight cell's key strings (`CLOUDSDK_PYTHON_SITEPACKAGES`, `check_hashes`, `if_fast_else_skip`, `BOTO_CONFIG`, the `gsutil#1429` citation, and the "Second amendment" markdown text) are present, the notebook remains valid JSON with exactly 19 cells (two new cells inserted), and the server-launch cell's stale output/execution_count are cleared.
- Task 2's automated check confirms both `03-03-PLAN.md` (T-3-08 second amendment) and `03-RESEARCH.md` (Pitfall 5 correction note) carry the new dated, cross-referenced revision record, with all original text — including the first `260726-gj6` amendment — preserved.
- No live Colab re-run is part of this plan's scope — the user re-runs Notebook B on Colab as a follow-up to confirm the pre-flight check reports the compiled crcmod extension as visible (or, failing that, that the `check_hashes=if_fast_else_skip` fallback lets the download complete anyway). This was already covered by 03-03-PLAN.md's Task 4 human-verify checkpoint, which this fix unblocks a third time.
</verification>

<success_criteria>
- `libero/notebooks/03b-pi0-inference-smoketest.ipynb` has a new pre-flight cell pair that sets `CLOUDSDK_PYTHON_SITEPACKAGES=1`, configures a `check_hashes=if_fast_else_skip` boto fallback, and runs a fast `gsutil version -l` crcmod check before the real checkpoint download; stays valid JSON; carries no stale error output in the server-launch cell.
- `03-03-PLAN.md`'s T-3-08 row and `03-RESEARCH.md`'s Pitfall 5 both carry a second dated 2026-07-26 record of this corrected root-cause diagnosis, cross-referencing each other, with all prior amendments preserved.
- Nothing outside the notebook's pre-flight/cell-10/server-launch-comment edits and these two docs was touched (no changes to `Pi0Backend`, `run_suite`/`eval_loop`, the localhost-only bind, the openpi-client legitimacy checkpoint, the `/content/openpi_data` local-disk path, or the earlier `--env=LIBERO` fix).
</success_criteria>

<output>
Create `.planning/quick/260726-hbb-fix-libero-notebooks-03b-pi0-inference-s/260726-hbb-SUMMARY.md` when done
</output>