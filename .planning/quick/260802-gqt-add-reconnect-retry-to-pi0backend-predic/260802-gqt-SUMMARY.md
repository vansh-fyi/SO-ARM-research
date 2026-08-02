---
phase: 03-vla-inference-loop
plan: 260802-gqt
status: complete
subsystem: vla-inference
tags: [openpi, pi0, websocket, keepalive, reconnect-retry, jax-recompile]

requires:
  - phase: 03-vla-inference-loop (260726-p0o)
    provides: "Notebook B fully working through episode 0 of the VLA-04 smoke test (L4 run 2026-07-26)"
provides:
  - "Pi0Backend.predict survives transient websocket connection drops via backoff + port-liveness probe + client rebuild + retry"
  - "Actionable RuntimeError (pointing at /content/serve_policy.log) when the server is genuinely dead or retries exhaust"
affects: [03-vla-inference-loop VLA-04 sign-off, next Colab smoke-test run]

key-files:
  modified:
    - libero/libero/libero/vla/pi0_backend.py
    - libero/libero/libero/vla/test_pi0_backend.py

key-decisions:
  - "Root cause read from openpi-client source at the exact cloned commit (15a9616): WebsocketClientPolicy hardcodes websockets.sync.client.connect without ping_interval/ping_timeout kwargs, so the 20s/20s defaults cannot be tuned from our side — the fix must live in our layer."
  - "Retry design: default backoffs (30s, 60s) sized for the suspected JAX-recompile stall (ep1's first inference presents a new shape; by the retry the compile cache is warm). Port-liveness probe before each reconnect because WebsocketClientPolicy.__init__ blocks indefinitely in _wait_for_server() against a dead server."
  - "RuntimeError from infer (server-reported inference error) is deliberately NOT retried — only connection-class failures (WebSocketException, ConnectionError, OSError) are."
  - "websockets import guarded with an ImportError fallback class so the local no-openpi test env (which fakes only openpi_client) still imports the module."
  - "Executed inline by the orchestrator (subagent spend limit); scratch venv (pytest+numpy+pyyaml+imageio[ffmpeg]) built in the session scratchpad to run the suite locally — 3 test_eval_loop failures before installing ffmpeg were confirmed pre-existing venv artifacts (identical failures on the unmodified tree), not regressions."

verification:
  - "10/10 local tests pass: 3 original pi0 tests, 4 new regression tests (transparent reconnect-retry, exhausted-retries RuntimeError, dead-port fail-fast, plus obs-dict/actions plumbing), 3 eval_loop tests"

user_setup_required: "Re-sync the Drive zip and re-run Notebook B's VLA-04 cell on Colab (server startup already proven; if the same keepalive stall recurs, predict() now rides through it and prints '[Pi0Backend] connection lost ... reconnecting')."
---

# Quick Task 260802-gqt: Reconnect-retry in Pi0Backend.predict

Fixes the failure that killed episode 1 of the otherwise-successful 2026-07-26 L4 smoke-test run (`ConnectionClosedError: sent 1011 (internal error) keepalive ping timeout`). `predict()` now: catches connection-class failures → backs off (30s/60s default) → probes the server TCP port (fail fast with a `/content/serve_policy.log` pointer if the process died) → rebuilds `WebsocketClientPolicy` → re-sends the same observation. Also corrected two stale `--env=pi0_fast_libero` doc references to `--env=LIBERO`.
