---
phase: 03-vla-inference-loop
plan: 260802-gqt
type: execute
wave: 1
depends_on: []
files_modified:
  - libero/libero/libero/vla/pi0_backend.py
  - libero/libero/libero/vla/test_pi0_backend.py
autonomous: true
requirements: [VLA-04]
---

<objective>
Harden `Pi0Backend.predict` against transient websocket connection loss, fixing the failure that killed episode 1 of the VLA-04 smoke test on the 2026-07-26 L4 run: `ConnectionClosedError: sent 1011 (internal error) keepalive ping timeout; no close frame received`.

Root-cause analysis (from openpi-client source at the exact cloned commit 15a9616): `WebsocketClientPolicy` calls `websockets.sync.client.connect(uri, compression=None, max_size=None, ...)` WITHOUT exposing `ping_interval`/`ping_timeout` — so the websockets defaults (20s/20s) apply and cannot be tuned from our side. Episode 1's first inference most plausibly triggered a JAX recompilation (new shape/padding) that stalled the server past the 20s keepalive window; the client closed the socket. A reconnect-and-retry after a backoff succeeds in that scenario because the server's compile cache is warm by then. `WebsocketClientPolicy.__init__` also blocks indefinitely in `_wait_for_server()` if the server process is dead — so the retry path must probe the TCP port first and fail fast with a pointer to /content/serve_policy.log instead of hanging.

Executed inline by the orchestrator (subagents unavailable — spend limit); design fully derived from live evidence + pinned-commit source reading before any code was written.
</objective>

<tasks>

<task type="auto">
  <name>Task 1: Reconnect-retry in Pi0Backend + regression test</name>
  <files>libero/libero/libero/vla/pi0_backend.py, libero/libero/libero/vla/test_pi0_backend.py</files>
  <action>
    pi0_backend.py: (1) store host/port and configurable retry backoffs on the instance; (2) catch a retryable-exception tuple (websockets' WebSocketException — with an ImportError fallback class for the local no-openpi test env — plus builtin ConnectionError/OSError) around `client.infer`; (3) on failure: sleep the backoff, probe the server TCP port for liveness (bounded, ~30s), rebuild `WebsocketClientPolicy`, retry; (4) if the port is closed or all attempts are exhausted, raise RuntimeError naming /content/serve_policy.log as the evidence to check. Also correct two stale `--env=pi0_fast_libero` doc references to `--env=LIBERO` (docs-only drift from 260726-epz). test_pi0_backend.py: add a regression test where the first `infer` raises ConnectionError, and assert a new client is constructed and the retry returns the actions (backoffs=(0,), `_wait_for_port` monkeypatched True); plus a give-up test asserting RuntimeError after exhausted retries.
  </action>
  <verify>
    <automated>pytest libero/libero/libero/vla/test_pi0_backend.py libero/libero/libero/vla/test_eval_loop.py -q (all pass, including the two new retry tests)</automated>
  </verify>
  <done>predict() survives a single transient connection drop transparently, fails fast with an actionable message when the server is dead, and the local suite passes.</done>
</task>

</tasks>
