---
phase: 03-vla-inference-loop
plan: 03
subsystem: vla-inference
tags: [pi0, openpi, websocket, colab, notebook, libero, eval-loop, interface-swap]

# Dependency graph
requires:
  - phase: 03-vla-inference-loop (plan 01)
    provides: "LIBERO/libero/libero/vla/ package: VLABackend Protocol, eval_loop (run_episode/run_suite/print_episode_result)"
  - phase: 02-soarm-robot-integration
    provides: "Soarm101 robot registration, frozen libero_spatial BDDL tasks, tuned eye_in_hand camera"
  - phase: 01-colab-environment-setup
    provides: "Colab Drive-zip delivery convention, Block A/B restart notebook pattern, numpy ABI gate pattern"
provides:
  - "LIBERO/libero/libero/vla/pi0_backend.py: Pi0Backend websocket client implementing VLABackend, with reconnect-retry hardening"
  - "LIBERO/notebooks/03b-pi0-inference-smoketest.ipynb: separate-kernel Colab notebook (openpi install, serve_policy.py --env=LIBERO server, 1-task x 2-episode smoke test via the unmodified run_suite)"
affects: ["phase 4 dataset collection", "phase 6 fine-tuning (0% zero-shot baseline for π0 as well as OFT)"]

# Tech tracking
tech-stack:
  added: ["openpi (Physical-Intelligence, commit 15a9616, uv-managed venv)", "openpi-client (editable install from openpi's packages/)", "robosuite/mujoco/bddl/gym in Notebook B's kernel (Block A2)"]
  patterns: ["Two-kernel split for conflicting VLA dependency stacks (D-06) via openpi's serve_policy.py + WebsocketClientPolicy remote-inference pattern, localhost-only", "Block A2 install-then-restart pattern replicated from Notebook A for a second dependency stack in the same notebook", "Server subprocess output to a log file, never an undrained PIPE", "Reconnect-retry with port-liveness probe for websocket policy clients"]

key-files:
  created:
    - LIBERO/libero/libero/vla/pi0_backend.py
    - LIBERO/libero/libero/vla/test_pi0_backend.py
    - LIBERO/notebooks/03b-pi0-inference-smoketest.ipynb
  modified:
    - LIBERO/libero/libero/vla/__init__.py

key-decisions:
  - "π0 served via plain serve_policy.py --env=LIBERO (resolves to pi05_libero, openpi's current default) — D-07's original pi0_fast_libero choice was deprecated on openpi main between plan-time and first live run (D-07 amendment 2026-07-26)"
  - "Checkpoint downloads to local Colab disk with CLOUDSDK_PYTHON_SITEPACKAGES=1 + check_hashes=if_fast_else_skip — gsutil's bundled Python never sees kernel pip installs (gsutil#1429); T-3-08's Drive-persistence goal knowingly dropped"
  - "vla/__init__.py optional-import guards catch Exception (not just ImportError) — numpy-ABI ValueErrors from a present-but-incompatible counterpart stack must degrade to None, not crash the shared package import"
  - "Pi0Backend.predict retries transient connection drops (backoff + port probe + client rebuild) — openpi-client hardcodes websockets' 20s keepalive with no tuning knobs, and ep1's first-inference JAX recompile stall outlives it"
  - "0% π0-on-SOARM success rate accepted as the expected zero-shot cross-embodiment result, exactly mirroring the OFT precedent from 03-02 — closing the gap is Phase 6's job"

patterns-established:
  - "Committed at the lowercase libero/ path prefix, matching Wave 1's precedent for the gitignored, case-insensitive-collapsed LIBERO/libero tree"

requirements-completed: [VLA-04]  # Colab L4 sign-off complete 2026-08-02 (see Task 4 Sign-Off)

# Metrics
duration: "Tasks 1-2: ~25min (2026-07-19); checkpoints + live debugging: 2026-07-26 and 2026-08-02 sessions"
completed: 2026-08-02
status: complete
---

# Phase 3 Plan 03: π0 (openpi) Backend + Smoke-Test Notebook Summary

**Pi0Backend websocket client + separate-kernel Colab notebook proving VLA-04: the exact same unmodified `run_suite`/`eval_loop` code that drives OFTBackend drove π0 (pi05_libero) through a full 1-task × 2-episode smoke test on live Colab (L4), with per-episode video and the aggregated success-rate table.**

## Accomplishments
- `pi0_backend.py`: thin websocket client wrapping openpi's own `WebsocketClientPolicy`, implementing the shared `predict(images, language) -> np.ndarray` interface; hardened with reconnect-retry (backoff → port-liveness probe → client rebuild → re-send) after a live keepalive-timeout failure
- `03b-pi0-inference-smoketest.ipynb` (24 cells): Block A2 (LIBERO sim stack + numpy ABI gate + restart), openpi uv-based install, openpi-client legitimacy checkpoint + .pth activation, crcmod/gsutil pre-flight, `serve_policy.py --env=LIBERO` server (localhost-only, log-file output), and the VLA-04 proof cell importing `Pi0Backend, run_suite` from the same package path Notebook A uses
- `vla/__init__.py`: cross-kernel graceful degradation fixed for real (catches `Exception`, not just `ImportError`)
- 4 regression tests added to the local mock-based suite (reconnect-retry, exhausted-retries error, dead-port fail-fast); 10/10 pass

## Task 3 Sign-Off: openpi-client package legitimacy (2026-07-26)

**Result: APPROVED.** User checked https://pypi.org/project/openpi-client/ (screenshot provided) and confirmed the package's origin is Physical Intelligence (same org as the openpi repo). Blocking-human gate cleared before any real `pip install openpi-client` ran on Colab.

## Task 4 Sign-Off: live Colab GPU π0 smoke test (2026-08-02)

**Result: APPROVED — smoke test ran to completion on Colab L4.** Both episodes of `pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate` ran the full 600 steps through the unmodified `run_suite`, each saving a video, with the aggregated summary table printed:

| Task | Episodes | Successes | Success Rate |
|------|----------|-----------|--------------|
| pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 2 | 0 | 0.0% |

**Interpretation — 0% success is the accepted, expected zero-shot cross-embodiment result, mirroring OFT's identical 0% from 03-02:** pi05_libero was fine-tuned on Panda-arm LIBERO demos, not SOARM. VLA-04's criterion is the interface swap (same `run_suite`, zero downstream changes), which is proven; task success belongs to Phase 6's fine-tuning.

### Seven real failure modes found and fixed via live Colab debugging (2026-07-26 → 2026-08-02)

The π0 path had never touched a real GPU/network before these sessions; each live run surfaced the next defect. All are committed to master with per-fix quick-task records under `.planning/quick/`:

1. **260726-epz** — `--env=pi0_fast_libero` invalid: config deprecated on openpi main; switched to `--env=LIBERO` (pi05_libero). *(D-07 amendment)*
2. **260726-gj6** — checkpoint download to Drive FUSE failed; moved to local disk + compiled crcmod. **Insufficient** — identical failure recurred, root-cause theory falsified by re-test.
3. **260726-hbb** — actual download root cause: gsutil's bundled Cloud SDK Python is isolated from kernel pip installs (gsutil#1429); `CLOUDSDK_PYTHON_SITEPACKAGES=1` + `check_hashes=if_fast_else_skip` + pre-flight verification cell. **Confirmed live.**
4. **260726-hw8** — `vla/__init__.py`'s `except ImportError` guard didn't catch a numpy-ABI `ValueError`, crashing the Pi0Backend import; broadened to `except Exception`. **Confirmed live.**
5. **260726-io0** — Notebook B's kernel never had robosuite/mujoco/bddl/gym at all (a genuine planning gap in this plan's original task description); added Block A2 mirroring Notebook A's proven install-then-restart pattern. **Confirmed live.**
6. **260726-p0o** — server `stdout=PIPE` never drained (95-min silent hang suspect) → log-file redirect; openpi-client editable-install `.pth` finder inert in a running kernel → `site.addsitedir` activation. **Confirmed live** (server ran episodes with log file; import worked on fresh run).
7. **260802-gqt** — episode 1 died with `ConnectionClosedError 1011 keepalive ping timeout` (openpi-client hardcodes websockets' 20s ping defaults; ep1's first-inference JAX recompile stall outlives them); added reconnect-retry to `Pi0Backend.predict`. **Confirmed live: the full 2-episode run completed on the next attempt.**

## Deviations from Plan
- D-07's `pi0_fast_libero` selection was overtaken by upstream deprecation (documented amendment; VLA-04 unaffected — checkpoint-agnostic code)
- Notebook B grew Block A2 (LIBERO sim stack) and hardening cells not in the original task description — all documented in the quick-task records above
- Both Wave 2 worktrees merged into master early, mid-checkpoint, at the user's explicit request (recorded 2026-07-19)

## User Setup Required
None remaining — all checkpoints resolved.

## Next Phase Readiness
- **Plan 03-03 complete; all Phase 3 plans complete.** ROADMAP Phase 3 success criterion #3 (backend swap with zero downstream changes) proven literally on live Colab.
- Phase 6 (fine-tuning) inherits two 0% zero-shot baselines: OFT (03-02, A100) and π0/pi05_libero (03-03, L4).
- Advisory for future Colab work: see `.planning/phases/03-vla-inference-loop/.continue-here.md` anti-patterns (kernel module caching, case-insensitive libero/ path) and the seven fix records above.

## Self-Check: PASSED

`libero/libero/libero/vla/pi0_backend.py` and `libero/notebooks/03b-pi0-inference-smoketest.ipynb` verified present and committed. Task 3 approved 2026-07-26; Task 4 approved 2026-08-02 with the completed 2-episode run output pasted by the user. Local test suite 10/10.

---
*Phase: 03-vla-inference-loop*
*Completed: 2026-08-02*
