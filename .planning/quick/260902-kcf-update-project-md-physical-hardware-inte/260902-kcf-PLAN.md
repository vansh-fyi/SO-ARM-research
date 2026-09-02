---
phase: quick-260902-kcf
plan: 01
type: execute
wave: 1
depends_on: []
files_modified: [.planning/PROJECT.md]
autonomous: true
requirements: []

must_haves:
  truths:
    - "PROJECT.md 'Out of Scope' no longer lists physical hardware integration as excluded"
    - "PROJECT.md 'Context' section explains hardware bring-up is a parallel track tracked via diagnostics/UAT/, outside .planning/phases/"
    - "PROJECT.md 'Key Decisions' table has one new row documenting the parallel-track decision with a '✓ Committed' outcome"
    - "v1.1 milestone content (Active/Validated requirements, Current Milestone section) is unchanged"
  artifacts:
    - .planning/PROJECT.md
  key_links: []
---

<objective>
Update `.planning/PROJECT.md` to reflect that physical SOARM hardware integration has started as a deliberate parallel track alongside the active v1.1 sim milestone — not a later, blocked milestone as previously stated.

Purpose: PROJECT.md's "Out of Scope" section is stale — physical bring-up (electronics, gripper, 5-joint arm assembly) is already complete per `diagnostics/UAT/`, and LeRobot-based control work is in progress. This doc update brings PROJECT.md in line with reality without touching the v1.1 sim-only phase/requirement structure, which remains fully valid and unaffected.
Output: `.planning/PROJECT.md` with corrected "Out of Scope", an added "Context" note, and a new "Key Decisions" row.
</objective>

<execution_context>
@/Users/hp/Desktop/Work/Repositories/SoARM-Research/.claude/gsd-core/workflows/execute-plan.md
@/Users/hp/Desktop/Work/Repositories/SoARM-Research/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
</context>

<tasks>

<task type="auto">
  <name>Task 1: Update Out of Scope, Context, and Key Decisions in PROJECT.md</name>
  <files>.planning/PROJECT.md</files>
  <action>
Make three scoped edits to .planning/PROJECT.md. Do not touch any other section (Requirements, Current Milestone, Active/Validated checkboxes, Constraints, Evolution, or the footer timestamp line).

1. In the "Out of Scope" section, remove the line "Physical SOARM hardware integration — simulation-first; real robot testing is a later milestone after sim pipeline is validated." Leave the other two Out of Scope bullets (Real-3DQA point cloud data, Real-time interactive REPL) untouched.

2. In the "Context" section, add a new bullet (after the existing bullets, before "Constraints") stating that physical hardware bring-up is underway as a parallel track running OUTSIDE the v1.1 milestone/phase structure — tracked via UATs in `diagnostics/UAT/` rather than `.planning/phases/`, and that it does not replace or compete with v1.1's sim-only scope. Reference by path and status:
   - `diagnostics/UAT/components/UAT.md` — electronics bring-up (complete, 7/7 steps)
   - `diagnostics/UAT/assembly/gripper/UAT.md` — gripper build + calibration (complete, 11/11 steps)
   - `diagnostics/UAT/assembly/main/UAT.md` — 5-joint arm assembly (complete, 10/10 steps)
   - `diagnostics/UAT/function/UAT.md` — LeRobot-based laptop control + camera recording (in progress)
   Also note that control software lives in `control/` (a separate venv from `diagnostics/`), uses HuggingFace LeRobot for real-robot control, and requires Python 3.12 (not the system default 3.14, which crashes LeRobot's config parser).

3. In the "Key Decisions" table, add one new row (after the existing rows, before the closing table markup) with: Decision = "Start physical hardware bring-up in parallel with v1.1 rather than waiting for sim validation to complete"; Rationale = "Physical parts arrived and needed bring-up/testing; this work doesn't block or compete with the sim-focused v1.1 phases since it lives entirely outside the phase/ROADMAP structure (tracked via diagnostics/UAT/ instead)"; Outcome = "✓ Committed".

Do not modify the "Last updated" footer line — the quick-task workflow's later STATE.md update step handles session bookkeeping separately.
  </action>
  <verify>
    <automated>grep -c "Physical SOARM hardware integration — simulation-first" .planning/PROJECT.md | grep -qx 0 && grep -q "diagnostics/UAT/function/UAT.md" .planning/PROJECT.md && grep -q "parallel with v1.1" .planning/PROJECT.md</automated>
  </verify>
  <done>Out of Scope no longer lists physical hardware as excluded; Context section references all four diagnostics/UAT/ paths and the control/ Python 3.12 constraint; Key Decisions table has the new parallel-track row with "✓ Committed" outcome; no other section changed.</done>
</task>

</tasks>

<verification>
Run `git diff .planning/PROJECT.md` and confirm the diff touches only the "Out of Scope" bullet removal, the new "Context" bullet, and the new "Key Decisions" row — no changes to Requirements, Current Milestone, Constraints, or Evolution sections.
</verification>

<success_criteria>
- "Physical SOARM hardware integration" line removed from Out of Scope
- Context section documents the parallel hardware track with all 4 diagnostics/UAT/ file references and their status, plus the control/ venv + Python 3.12 note
- Key Decisions table has a new row for this decision with "✓ Committed" outcome
- v1.1 milestone content untouched
</success_criteria>

<output>
Create `.planning/quick/260902-kcf-update-project-md-physical-hardware-inte/260902-kcf-SUMMARY.md` when done
</output>
