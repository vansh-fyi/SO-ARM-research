---
phase: 03-vla-inference-loop
plan: 260726-epz
type: execute
wave: 1
depends_on: []
files_modified:
  - libero/notebooks/03b-pi0-inference-smoketest.ipynb
  - .planning/phases/03-vla-inference-loop/03-CONTEXT.md
  - .planning/phases/03-vla-inference-loop/03-RESEARCH.md
autonomous: true
requirements: [VLA-04]

must_haves:
  truths:
    - "Notebook B serves pi05_libero via plain --env=LIBERO (not the deprecated pi0_fast_libero name), so a fresh Colab run of serve_policy.py no longer fails with the 'invalid choice' argument-parsing error the user hit"
    - "03-CONTEXT.md's D-07 records the 2026-07-26 revision from pi0_fast_libero to pi05_libero, with rationale, so this resolved real-world drift isn't re-litigated later"
    - "03-RESEARCH.md's Assumptions Log A1 entry records that its predicted risk materialized exactly as flagged, and how it was resolved"
  artifacts:
    - libero/notebooks/03b-pi0-inference-smoketest.ipynb
    - .planning/phases/03-vla-inference-loop/03-CONTEXT.md
    - .planning/phases/03-vla-inference-loop/03-RESEARCH.md
  key_links:
    - "Notebook B's serve_policy.py background-process cell (argv list) and its preceding markdown header both say --env=LIBERO consistently, with zero remaining references to the deprecated config name anywhere in the notebook"
---

<objective>
Fix a confirmed real-world config-name drift in `libero/notebooks/03b-pi0-inference-smoketest.ipynb`: the notebook's `serve_policy.py` invocation used a deprecated openpi config name that no longer maps to a servable checkpoint on openpi's current `main` branch, causing the exact `--env: invalid choice` failure the project owner hit live on Colab during the 03-03 Task 4 sign-off checkpoint. Switch to openpi's current default LIBERO checkpoint (served via the plain `--env=LIBERO` flag) and record the decision revision in this phase's planning docs.

Purpose: 03-RESEARCH.md's own Assumptions Log (entry A1) explicitly flagged this exact risk class ("if openpi's main branch has since changed its LIBERO config names... install could fail differently... re-verify config names immediately before Notebook B's first real Colab run") — this plan closes that flagged risk now that it has materialized, without expanding scope beyond the config-name plumbing and its documentation trail.
Output: A corrected, JSON-valid `03b-pi0-inference-smoketest.ipynb` with no remaining references to the deprecated config name; a dated amendment under D-07 in `03-CONTEXT.md`; a dated resolution note attached to Assumptions Log entry A1 in `03-RESEARCH.md`.
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
</context>

<tasks>

<!-- planner-discipline-allow: pi0_fast_libero -->

<task type="auto">
  <name>Task 1: Notebook B — switch serve_policy.py to --env=LIBERO (serves pi05_libero)</name>
  <files>libero/notebooks/03b-pi0-inference-smoketest.ipynb</files>
  <read_first>
    libero/notebooks/03b-pi0-inference-smoketest.ipynb (the file being fixed — load with Python's json module, not a text editor, to avoid corrupting the notebook's JSON structure; edits below are keyed by 0-indexed cell position in the `cells` array, which is stable since no cells are added/removed/reordered)
  </read_first>
  <action>
    Write a small Python script (run via Bash, e.g. `python3 -c "..."` or a throwaway `.py` file) that loads the notebook with `json.load`, patches exactly four cells by index, and writes back with `json.dump(nb, f, indent=1, ensure_ascii=False)` plus a trailing newline — this preserves the file's existing nbformat-style JSON formatting so the git diff stays minimal and scoped to the actual content changes. Do not hand-edit the raw JSON text with a text editor; malformed escaping will corrupt the notebook.

    Cell 3 (code, the GPU-assertion cell, source begins with the comment "# GPU assertion"): in its last source line, change the print statement's message from referencing the deprecated FAST-tokenizer config name and a bare "run" to instead reference `pi05_libero (π0)` as the checkpoint under test, keeping the rest of the sentence ("target A100 tier for the first ... run (03-RESEARCH.md Open Question #2).") unchanged. Because this cell's stored output text embeds the old print message verbatim (a stale artifact from before this fix), clear this cell's `outputs` to `[]` and set `execution_count` to `null`.

    Cell 4 (markdown, "## BLOCK A: openpi install (separate kernel/venv)"): replace the paragraph that names the specific config this notebook serves (the one contrasting an autoregressive/FAST-tokenizer variant against a flow-based one and citing D-07) with a paragraph stating: this notebook serves `pi05_libero`, openpi's current default checkpoint for the `LIBERO` environment, resolved automatically via a plain `--env=LIBERO` flag rather than an explicit config name; and that D-07 was revised 2026-07-26 (point to `03-CONTEXT.md`'s D-07 amendment for the full rationale). Do not alter the rest of this cell (the D-06 separate-kernel explanation and the "run all cells top to bottom" line stay as-is).

    Cell 10 (markdown, the "### Start `serve_policy.py ...`" header immediately before the server-launch code cell): update the header's inline-code flag value so it reads `--env=LIBERO`, and note in the header that this serves `pi05_libero`. Leave the rest of this cell (the T-3-06 localhost-bind rationale, the T-3-08 Drive-persistence rationale) unchanged.

    Cell 11 (code, the server-launch cell containing the `serve_cmd` list passed to `subprocess.Popen` and the "Starting serve_policy.py..." print immediately after it): change the `serve_cmd` list's `--env=...` argument to `--env=LIBERO`, and update the print f-string that echoes the same flag to match (and note it serves `pi05_libero`) — do not touch any other line in this cell (the `SERVE_HOST`/`SERVE_PORT` values, the Drive-backed `OPENPI_DATA_HOME` setup, the port-polling loop, and the localhost-only bind behavior must stay exactly as they are; this plan's scope is strictly the config-name plumbing). This cell's stored output is the actual captured traceback from the user's live failing run — clear this cell's `outputs` to `[]` and set `execution_count` to `null` so the notebook doesn't ship a misleading stale error.

    Do not modify any other cell. Do not touch the localhost/127.0.0.1-only binding logic, `Pi0Backend` usage, `run_suite`/`eval_loop` calls, task/episode counts, or the GPU-tier assertion logic itself (only its print message's wording changes).
  </action>
  <verify>
    <automated>python3 -c "
import json
p = 'libero/notebooks/03b-pi0-inference-smoketest.ipynb'
nb = json.load(open(p))
full = ''.join(''.join(c['source']) for c in nb['cells'])
assert 'pi0_fast_libero' not in full, 'deprecated config name still present'
assert 'pi0_libero' not in full, 'stale flow-based config-name comparison still present'
assert '--env=LIBERO' in full, 'new --env=LIBERO flag missing'
assert 'pi05_libero' in full, 'pi05_libero not documented as the served checkpoint'
for i in (3, 11):
    c = nb['cells'][i]
    assert c['outputs'] == [], f'cell {i} outputs not cleared'
    assert c.get('execution_count') is None, f'cell {i} execution_count not cleared'
print('Notebook fix verified OK')
"</automated>
  </verify>
  <done>libero/notebooks/03b-pi0-inference-smoketest.ipynb has zero remaining references to the deprecated pi0_fast_libero/pi0_libero config names, serves pi05_libero via plain --env=LIBERO consistently across its markdown header, code argv list, and print statement, stays valid JSON, and no longer carries the stale failing-run traceback in cell outputs.</done>
</task>

<task type="auto">
  <name>Task 2: Record the D-07 decision revision in 03-CONTEXT.md and 03-RESEARCH.md</name>
  <files>.planning/phases/03-vla-inference-loop/03-CONTEXT.md, .planning/phases/03-vla-inference-loop/03-RESEARCH.md</files>
  <read_first>
    .planning/phases/03-vla-inference-loop/03-CONTEXT.md (D-07's exact current wording, under "### π0 Integration Depth")
    .planning/phases/03-vla-inference-loop/03-RESEARCH.md (Assumptions Log table, entry A1, and the sentence immediately following the table)
    .planning/STATE.md (Accumulated Context > Decisions — the repo's existing convention for recording a decision inline with its resolving context, e.g. "01-03 (resolves RESEARCH A1, CONFIRMED on Colab): ..."; no prior example of amending a CONTEXT.md decision exists in this repo, so this task establishes the pattern by appending a clearly dated sub-note directly beneath the original decision rather than rewriting or deleting it)
  </read_first>
  <action>
    In `03-CONTEXT.md`, directly beneath the existing D-07 bullet (do not delete or reword the original D-07 text — it remains the historical record of what was decided and why), add a new indented sub-bullet beginning "**Amendment (2026-07-26):**" that states: openpi's `serve_policy.py --env` flag is a coarse `EnvMode` enum (`ALOHA`/`ALOHA_SIM`/`DROID`/`LIBERO`), so the originally-chosen config name was never a valid `--env` value in the first place, and — more importantly — has no published ready-to-serve inference checkpoint on openpi's current `main` branch as of 2026-07-26 (confirmed directly against `serve_policy.py`'s source and the project README, both re-fetched on that date). State the resolution: serve via plain `--env=LIBERO`, which resolves through openpi's own default-checkpoint mapping to `pi05_libero` — openpi's current default checkpoint for the LIBERO environment. Close the amendment by noting VLA-04's intent (proving the `Pi0Backend`/interface swap works) is unaffected, since `Pi0Backend.predict()` only depends on the websocket `infer()` contract, not which specific config `serve_policy.py` loads.

    In `03-RESEARCH.md`, immediately after the sentence following the Assumptions Log table (the "If this table is empty" line) and before the "## Open Questions" heading, add a short dated paragraph beginning "**A1 materialized (2026-07-26):**" recording that this exact anticipated risk occurred on the first real Colab run of Notebook B: the originally-researched config name is not a valid `serve_policy.py --env` value and has no published checkpoint on openpi's current `main`. State the resolution (plain `--env=LIBERO`, resolving to `pi05_libero`) and point to `03-CONTEXT.md`'s D-07 amendment for the full rationale, plus note that `libero/notebooks/03b-pi0-inference-smoketest.ipynb` was updated accordingly.
  </action>
  <verify>
    <automated>grep -q "Amendment (2026-07-26)" .planning/phases/03-vla-inference-loop/03-CONTEXT.md && grep -q "pi05_libero" .planning/phases/03-vla-inference-loop/03-CONTEXT.md && grep -q "A1 materialized" .planning/phases/03-vla-inference-loop/03-RESEARCH.md && grep -q "pi05_libero" .planning/phases/03-vla-inference-loop/03-RESEARCH.md && echo OK</automated>
  </verify>
  <done>03-CONTEXT.md's D-07 carries a dated 2026-07-26 amendment explaining the pi0_fast_libero -> pi05_libero revision and why VLA-04's intent is unaffected; 03-RESEARCH.md's Assumptions Log A1 entry has a matching dated resolution note; the original D-07 text and A1 row are preserved unmodified as historical record.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

No new trust boundaries are introduced by this plan. It edits config-name strings, prose, and stale cell outputs only — it does not touch package installs, network binds, or credential handling.

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-3-09 | Tampering | `libero/notebooks/03b-pi0-inference-smoketest.ipynb` `serve_policy.py` invocation (config-name argument) | low | accept | This plan only changes which built-in `--env` enum value is passed and which checkpoint name is documented in prose; it does not change the install source, the localhost-only bind, or add any new package. The pre-existing threats from 03-03-PLAN.md's threat model (T-3-SC package legitimacy, T-3-06 localhost-only bind, T-3-07 pinned commit SHA, T-3-08 GCS download persistence) remain valid and unchanged by this fix — no new mitigation is needed here. |

</threat_model>

<verification>
- Task 1's automated check confirms the notebook has zero remaining references to the deprecated `pi0_fast_libero`/`pi0_libero` config names, documents `pi05_libero` as the served checkpoint via plain `--env=LIBERO`, remains valid JSON (implicit in `json.load` succeeding), and no longer carries the stale failing-run traceback in the two edited code cells' outputs.
- Task 2's automated check confirms both `03-CONTEXT.md` (D-07 amendment) and `03-RESEARCH.md` (A1 resolution note) carry the dated, cross-referenced revision record.
- No live Colab re-run is part of this plan's scope — this plan only fixes the config-name plumbing and its documentation trail. The user re-runs Notebook B on Colab as a follow-up to confirm the fix resolves the original failure (this was already covered by 03-03-PLAN.md's Task 4 human-verify checkpoint, which this fix unblocks).
</verification>

<success_criteria>
- `libero/notebooks/03b-pi0-inference-smoketest.ipynb` serves `pi05_libero` via plain `--env=LIBERO`, consistently, with no remaining mentions of the deprecated config name, and stays valid JSON with no stale error output.
- `03-CONTEXT.md`'s D-07 and `03-RESEARCH.md`'s Assumptions Log A1 both carry a dated 2026-07-26 record of this real-world decision revision, cross-referencing each other.
- Nothing outside the notebook's config-name plumbing and these two docs was touched (no changes to `Pi0Backend`, `run_suite`/`eval_loop`, localhost binding, or any other notebook).
</success_criteria>

<output>
Create `.planning/quick/260726-epz-fix-libero-notebooks-03b-pi0-inference-s/260726-epz-SUMMARY.md` when done
</output>
