---
phase: 15-mllm-style-pipeline-backends
plan: 02
subsystem: control/dependencies
status: complete
tags: [supply-chain, requirements, anthropic, openai, mllm, venv]
requires: []
provides:
  - "anthropic==1.11.0 and openai==3.23.0 exact pins with provenance in control/requirements.txt"
  - "both SDKs (plus 5 transitive packages) installed in control/.venv"
  - "control/test_mllm_requirements.py pin / installed-version / venv-location / import guard"
affects: [15-06, 15-07, 15-08]
tech-stack:
  added:
    - "anthropic 1.11.0 (MIT)"
    - "openai 3.23.0 (Apache-2.0)"
    - "transitive: docstring_parser 0.18.0, httpx2 2.13.1, httpcore2 2.13.1, jiter 0.17.0, truststore 0.10.4"
  patterns:
    - "Plan 14-13 provenance comment block (project, repo, license, date, wheel sha256) extended with transitive list and human approval date"
    - "install only the exact pins (never pip install -r) into control/.venv; pip check before/after"
key-files:
  created:
    - control/test_mllm_requirements.py
  modified:
    - control/requirements.txt
decisions:
  - "Human approved all seven packages (anthropic, openai, httpx2, httpcore2, jiter, docstring-parser, truststore) at the Plan 15-02 supply-chain checkpoint on 2026-10-02"
  - "Transitive packages are recorded in provenance comments but not pinned (T-15-05 accepted; lockfile out of scope)"
metrics:
  duration: "~5 min (Task 2 continuation; Task 1 was a blocking human checkpoint)"
  completed: 2026-10-02
  tasks: 2
  files: 2
---

# Phase 15 Plan 02: MLLM Provider SDK Pins Summary

The Anthropic 1.11.0 and OpenAI 3.23.0 SDKs were checked by a human, pinned exactly in `control/requirements.txt` with a full provenance block (PyPI project, source repo, license, release date, wheel sha256, transitive packages, approval date), and installed only into `control/.venv`. `pip check` output was the same before and after the install, and a 12-test guard catches pin or version drift.

## Tasks

| Task | Name | Commit | Files |
| ---- | ---- | ------ | ----- |
| 1 | Human supply-chain confirmation for anthropic, openai and transitive dependencies | checkpoint, no commit (read-only prep) | none |
| 2 | Pin anthropic and openai with provenance, install into control/.venv, verify | 4c89092 (RED test), 385c7b9 (GREEN pins) | control/test_mllm_requirements.py, control/requirements.txt |

## Task 1: Checkpoint Outcome

Blocking `checkpoint:human-verify` (`gate="blocking-human"`), not auto-approved. A previous agent fetched each package's PyPI JSON metadata (all HTTP 200, none yanked, no listed vulnerabilities) and presented it for review:

| Package | Version | Publisher | Source repo | License | Released |
|---------|---------|-----------|-------------|---------|----------|
| anthropic | 1.11.0 | Anthropic | github.com/anthropics/anthropic-sdk-python | MIT | 2026-09-30 |
| openai | 3.23.0 | OpenAI | github.com/openai/openai-python | Apache-2.0 | 2026-10-01 |
| httpx2 | 2.13.1 | Pydantic Services Inc. | github.com/pydantic/httpx2 | BSD-3-Clause | 2026-09-23 |
| httpcore2 | 2.13.1 | Pydantic Services Inc. | github.com/pydantic/httpx2 (src/httpcore2) | BSD-3-Clause | 2026-09-23 |
| jiter | 0.17.0 | Samuel Colvin | github.com/pydantic/jiter | MIT | 2026-09-12 |
| docstring-parser | 0.18.0 | Marcin Kurczewski | github.com/rr-/docstring_parser | MIT | 2026-04-14 |
| truststore | 0.10.4 | Seth Michael Larson | github.com/sethmlarson/truststore | MIT | 2025-08-12 |

**Human response:** "approved" on 2026-10-02. Nothing under control/ or control/.venv changed before the approval.

Note: unlike the plan text, docstring-parser 0.18.0's PyPI metadata does include the repository link.

## Task 2: Pins, Install, Verification

- Before writing the pins, I fetched the PyPI JSON again. Wheel sha256 and upload times matched the prep findings:
  - anthropic-1.11.0-py3-none-any.whl `52f97b2c485cca7ac66058374f5073a3febb7e6849b16989c145d602a3efee21`, 2026-09-30T22:58:22Z
  - openai-3.23.0-py3-none-any.whl `7fbec2e50a05ac0fa858629505688e5998fd82245058390ac1b6d98960174bd5`, 2026-10-01T23:03:50Z
- The pip install used cached SDK wheels. Running `pip download --no-deps` plus `shasum -a 256` gave the same digests as above, so the bytes pip used match PyPI.
- Install command (main checkout's venv only): `control/.venv/bin/pip install anthropic==1.11.0 openai==3.23.0`. It installed exactly anthropic 1.11.0, openai 3.23.0, docstring-parser 0.18.0, httpx2 2.13.1, httpcore2 2.13.1, jiter 0.17.0 and truststore 0.10.4. No existing package was upgraded or downgraded.
- `pip show`: both SDKs are at `/Users/hp/Desktop/Work/Repositories/SoARM-Research/control/.venv/lib/python3.12/site-packages`.
- The versions that must not move are unchanged after the install: lerobot 0.6.1, google-genai 2.25.0, httpx 0.28.1, httpcore 1.0.9, pydantic 2.13.5.

### pip check (before vs after)

Before install (exit 1):
```
ultralytics 8.4.138 has requirement numpy!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.0,!=2.3.1,!=2.3.2,!=2.3.3,!=2.3.4,>=1.23.0; platform_system == "Darwin", but you have numpy 2.2.6.
```
After install (exit 1): the same single line. **No new conflicts.** The ultralytics/numpy mismatch existed before this plan and is unrelated to it (out of scope).

### Tests

- RED (4c89092): 11 failed, 1 passed. The one pass was the all-lines exact-pin rule, which the existing file already met.
- GREEN (385c7b9): `pytest test_mllm_requirements.py -x -q` gave 12 passed.
- Full suite: `pytest -x -q` gave **1015 passed** (1003 baseline + 12 new).
- `control/requirements.txt` diff: 26 insertions, 0 deletions (comment lines plus the two pin lines, after the google-genai block).

## Deviations from Plan

None. The plan was executed as written. Two checks were added within scope: a check that the cached wheel hashes match PyPI, and a test asserting the SDKs are installed in the running venv rather than the base interpreter (supports T-15-SC).

## TDD Gate Compliance

- RED: `test(15-02)` commit 4c89092
- GREEN: `feat(15-02)` commit 385c7b9 (after RED)
- REFACTOR: not needed

## Threat Model Coverage

- T-15-SC (mitigate): blocking human checkpoint before install, exact `==` pins, wheel sha256 recorded and checked against the installed bytes, install only into control/.venv, pip check before and after.
- T-15-05 (accept): transitive versions are recorded in the provenance comments but not pinned.

## Known Stubs

None.

## Self-Check: PASSED

- FOUND: control/test_mllm_requirements.py
- FOUND: control/requirements.txt (anthropic==1.11.0, openai==3.23.0)
- FOUND: commit 4c89092
- FOUND: commit 385c7b9
