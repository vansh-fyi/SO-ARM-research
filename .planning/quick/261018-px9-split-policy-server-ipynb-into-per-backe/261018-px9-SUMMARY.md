---
phase: quick-261018-px9
plan: 01
subsystem: control/vla_bridge (Colab policy-server notebooks)
tags: [colab, notebooks, generator, grpc, http-adapter, depth, diagnostics, groot-n17]
status: complete
requires:
  - vla_bridge.backend_smoke GRPC_LANE / HTTP_LANE (coverage test)
  - control/vla_bridge/*_server.py launchers and adapters (embedded byte-identically)
  - backend_configs/pi0.json verified_revision, backend_configs/groot_n17.json checkpoint
provides:
  - "control/vla_bridge/colab_server_utils.py: Colab-side stop/start/ownership/diagnostics helper"
  - "scripts/build_colab_notebooks.py (render_all/write_all/check/main) + scripts/colab_notebook_specs.py"
  - "control/vla_bridge/notebooks/: 13 self-contained notebooks + generated README.md"
  - "control/test_colab_notebooks.py: drift, structure and safety-intent tests"
affects:
  - "Phase 14 live backend smokes / hardware UAT: each backend now has its own notebook"
  - "groot-n17 default checkpoint (backend_configs/groot_n17.json)"
tech-stack:
  added: []
  patterns:
    - "Generated notebooks: %%writefile embed + sha256 verify cell, drift-tested against generator output"
    - "Launch = stop_servers(port, markers) -> start_server(-u, PYTHONUNBUFFERED) -> wait_until_ready(own-PID ownership)"
key-files:
  created:
    - control/vla_bridge/colab_server_utils.py
    - control/test_colab_server_utils.py
    - scripts/build_colab_notebooks.py
    - scripts/colab_notebook_specs.py
    - control/test_colab_notebooks.py
    - control/vla_bridge/notebooks/README.md
    - control/vla_bridge/notebooks/*.ipynb (13)
  modified:
    - control/vla_bridge/backend_configs/groot_n17.json
    - control/test_backend_groot_n17.py
    - control/test_run_vla_episode.py
    - control/test_depth_pipeline.py
    - control/vla_bridge/groot_n17_server.py (docstring)
    - control/run_vla_episode.py (help text)
    - control/requirements.txt (comment)
    - control/test_robot_client.py (comment)
    - control/vla_bridge/{depth_camera,stereo_camera,robot_client}.py (docstrings/comments)
    - control/vla_bridge/backends/diffusion_policy.py, checkpoint_candidates.md
    - control/vla_bridge/{openvla_oft,openvla,octo,wallx,rdt1b,groot_n15}_server.py (docstrings)
  deleted:
    - control/vla_bridge/policy_server.ipynb
    - control/vla_bridge/policy_server_launch.md
    - scripts/sync_policy_notebook.py
decisions:
  - "groot-n17 default = tsangb34/groot-n17-so101-soccer-red_bowl-40episodes@643fa0a3e37620850096cd4bf4b37509b763d173 (live HF API sha on 2026-10-18 matched the planning-time sha)"
  - "Generator rejects embedded files with whitespace-only lines: IPython dedents a cell before %%writefile, which would empty those lines and break the sha256 match"
  - "Depth notebook also embeds colab_server_utils.py (its launch uses the helper)"
  - "Second-ngrok-session advice: open the 5173 TCP tunnel from the depth notebook's kernel (one agent, two tunnels), since each kernel starts its own ngrok agent"
metrics:
  duration: "~21 min (2026-10-18T13:25:39Z to 13:46:34Z)"
  completed: 2026-10-18
  tasks: 3
  files: 40
  tests: "full control suite 1729 passed (1674 baseline + 37 colab_server_utils + 18 colab_notebooks), 0 failed"
---

# Quick Task 261018-px9: Per-backend Colab notebooks Summary

The ~100-cell combined notebook is replaced by 13 generated, self-contained notebooks in
`control/vla_bridge/notebooks/`. Each one embeds every file it runs byte-identically and checks
it by sha256, pins the verified lerobot/transformers/diffusers versions, and accepts its launch
only when its own server process owns its port. A stdlib+psutil helper, `colab_server_utils.py`,
provides the stop/start/ownership checks and the diagnostics.

## What was built

- **`control/vla_bridge/colab_server_utils.py`** (TDD; 37 real-subprocess tests).
  - `listening_pids`, `stop_servers`, `start_server` and `wait_until_ready`.
  - `wait_until_ready` fails on an early exit (exit code plus log tail), on a foreign or stale
    port owner (message names that PID), on a missing launcher banner (searched across the
    whole log), and on a timeout (it terminates the server first).
  - Diagnostics: `log_report`, `process_report`, `gpu_report`, `health_report`. Teardown:
    `stop_server`.
- **groot-n17 checkpoint swap.** The default is now
  `tsangb34/groot-n17-so101-soccer-red_bowl-40episodes@643fa0a3e37620850096cd4bf4b37509b763d173`.
  - Verified before the swap: live HF API sha; config.json gives chunk_size 16,
    use_relative_actions true, relative_exclude_joints ['gripper'], front + wrist 3x480x640,
    action [6]. The old id now returns HTTP 401.
  - The old id remains only in groot_n17.json's provenance sentence.
- **References repointed.** Docstrings, comments and help text now point at
  `vla_bridge/notebooks/*.ipynb` or the README's "Camera mapping" section. No behaviour changed.
- **Generator, specs, notebooks and tests.**
  - `scripts/build_colab_notebooks.py` (stdlib only) renders all 14 files deterministically.
    `--check` exits 1 on drift.
  - `control/test_colab_notebooks.py` covers: byte equality with the generator, `check()`, the
    exact file set, lane coverage against `backend_smoke`, nbformat 4.5 structure, embed
    byte-identity plus sha256, the stop/start/wait order in launch cells, the patched-launcher
    banners, no upload instructions, transformers pinned after lerobot, the diffusers pins, the
    pi0 no-compile copy, the groot-n17 checkpoint, gated-licence checks, diagnostics and
    teardown, and the README index.
  - `test_depth_pipeline.py` now reads `fastfs_depth.ipynb`'s `resident-fastfs-source` cell and
    runs the no-thread / no-run_demo checks over all 13 notebooks.

## Notebooks

| notebook | lane | port → tunnel | GPU (metadata) |
|---|---|---|---|
| fastfs_depth | HTTP depth service (FastFS venv) | 3000 → HTTP | "GPU runtime; the former combined notebook ran it on A100" (A100) |
| grpc_smolvla | gRPC stock | 5173 → TCP | L4 |
| grpc_pi0 | gRPC stock (local no-compile copy) | 5173 → TCP | L4 |
| grpc_pi05 | gRPC stock | 5173 → TCP | L4 |
| grpc_act | gRPC stock | 5173 → TCP | L4 |
| grpc_diffusion | gRPC patched (diffusion_policy_server.py) | 5173 → TCP | L4 |
| grpc_groot_n17 | gRPC patched (groot_n17_server.py) | 5173 → TCP | L4 |
| http_openvla_oft | HTTP adapter (venv) | 3001 → HTTP | L4 |
| http_wallx | HTTP adapter (kernel python) | 3002 → HTTP | A100 High-RAM (machine_shape hm) |
| http_openvla | HTTP adapter (venv) | 3003 → HTTP | L4 |
| http_octo | HTTP adapter (venv) | 3004 → HTTP | L4 |
| http_rdt1b | HTTP adapter (venv) | 3005 → HTTP | A100 |
| http_groot_n15 | HTTP adapter (venv) | 3006 → HTTP | L4 |

## Commits

| Task | Commit | Message |
|------|--------|---------|
| 1 (RED) | b6cf5fe | test(261018-px9): add failing tests for colab_server_utils process/port helper |
| 1 (GREEN) | b3479b6 | feat(261018-px9): colab_server_utils process/port helper for Colab notebooks |
| 2 | 8eaf10b | fix(261018-px9): swap groot-n17 default checkpoint, repoint docs to per-backend notebooks |
| 3 | 494dc70 | feat(261018-px9): generate 13 self-contained Colab notebooks, retire the combined notebook |

## Verification

All runs used `/Users/hp/Desktop/Work/Repositories/SoARM-Research/control/.venv/bin/python` from
the worktree's `control/`. `vla_bridge` resolved to the worktree copy.

- `pytest test_colab_server_utils.py`: RED failed (module missing), then 37 passed.
- Task 2: CONFIG_OK. Both greps for old references were empty. The four named test files
  passed (192 passed together with test_stereo_camera.py and test_depth_camera.py).
- Task 3:
  - Generate, `git add`, regenerate: `git diff --exit-code` was clean and `--check` passed.
  - 13 `.ipynb` files exist, the three deleted files are gone, and NB_JSON_OK.
  - No `policy_server.ipynb` / `policy_server_launch` / `sync_policy_notebook` reference is left
    in control/ or scripts/ (or anywhere outside .planning/).
  - `cube-drawer-40k` appears only in groot_n17.json.
  - A drift probe (one appended byte in the README) made `--check` exit 1.
- Full suite: **1729 passed, 0 failed**.
- Every lifted cell was diffed against the HEAD notebook. The token, OpenMP, pyngrok,
  torchaudio, depth and the six HTTP install cells are identical. The only exception is
  wallx-install's comment: "same lerobot pin as Step 1" became "as the gRPC notebooks".
  Prose diffs are only the allowed edits (a)-(e).

## TDD Gate Compliance

Task 1 has a RED `test(...)` commit (b6cf5fe) followed by a GREEN `feat(...)` commit (b3479b6).
No refactor commit was needed.

## Deviations from Plan

### Auto-fixed / added

**1. [Rule 2 - Correctness] Generator guard for whitespace-only lines in embedded files.**
I read IPython's `inputtransformer2.py` on GitHub (main). `TransformerManager.transform_cell`
runs `leading_indent`, which `textwrap.dedent`s the whole cell before `%%writefile`. That
empties whitespace-only lines, so the written file would no longer match its sha256. No
embedded file has such lines today. `_source()` now raises if one ever does. The same source
confirms that a missing trailing newline is added back, which is why each embedded file must
end with exactly one `\n`. IPython is not installed locally, so the round trip was not run.
Commit 494dc70.

**2. [Rule 1 - Accuracy] Advice when ngrok refuses a second session (depth notebook).**
The plan's text says to run both notebooks on one runtime. Each notebook kernel starts its own
pyngrok agent, so sharing a runtime alone does not avoid a one-agent limit. The text keeps the
plan's sentence and adds the remedy that does work: open the 5173 TCP tunnel from the depth
notebook's kernel, so one agent serves both tunnels. It also notes that `ngrok.kill()` closes
every tunnel of its kernel. Not verified against ngrok's current plan limits.

**3. Minor additions.**
- The depth notebook also embeds `colab_server_utils.py`, because its launch cell uses it.
- Every notebook gets a gated-licence heading, a "Server helper" heading and a "Launch"
  explanation.
- The README's `--backend` column shows "(depth service, for `--depth-endpoint`)" for the
  depth notebook. Its internal spec backend id is `fastfs-depth`.
- The depth notebook's local "smoke" command is `curl -s https://<depth_tunnel>/health`, since
  backend_smoke has no depth lane.

## Notes for the orchestrator

- **Merge conflict expected.** The main checkout has an uncommitted edit to
  `control/vla_bridge/policy_server.ipynb`, and this branch deletes that file. Merging will hit a
  modify/delete conflict. Per the plan, the user's working-copy edit is not preserved (a backup
  exists outside the repo). Resolve by keeping the deletion.
- STATE.md and ROADMAP.md were not touched, per the spawn constraints. This plan completes no
  requirement IDs (`requirements: []`).
- The diffusion and groot-n17 episode commands keep the HEAD `<port> <robot_id>` positional
  placeholders, as the plan says. The stock gRPC episode commands omit them.
- `wallx_server.py:102` still has the comment "this file is uploaded to Colab on its own".
  That is code commentary, not an instruction in any cell, and Task 3 did not list the file.
  It was left unchanged.

## Known Stubs

None. `<ngrok_tcp_host:port>`, `https://<..._tunnel>`, `POLICY_TCP_HOST:PORT` and `UNIQUE_NAME`
are operator placeholders. Each tunnel cell prints the commands with the live address filled in.

## Threat Flags

None beyond the plan's threat model. Each mitigation is implemented and tested:

| Threat | Mitigation | Tests |
|---|---|---|
| T-px9-01 | sha256 verify cells | drift tests |
| T-px9-02 | own-PID ownership plus banner, and `process_report` WRONG-server | stale-owner tests |
| T-px9-03 | teardown cells | per-notebook teardown test |
| T-px9-04 | verbatim token cell, no stored outputs | structure test |
| T-px9-05 | `stop_servers` excludes its own PID; exact markers only | self-exclusion test |
| T-px9-06 | groot-n17 sha re-fetched live; Pitfall 1 stop instruction kept | |
| T-px9-07 | pi0 copy pinned to `verified_revision`; `compile_model` printed before → after | |

No package was installed locally.

## Self-Check: PASSED
