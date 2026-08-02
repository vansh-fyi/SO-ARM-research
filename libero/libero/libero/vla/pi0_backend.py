"""pi0 (openpi) backend implementing VLABackend (VLA-04).

Proves the shared `predict(images, language) -> np.ndarray` interface is
genuinely swappable: this class is driven by the exact same
`eval_loop.run_episode`/`run_suite` code as `OFTBackend`, with zero
downstream changes (ROADMAP success criterion #3).

Per D-05, pi0 must run real inference on Colab — not a stub. Per D-06,
openpi's own dependency stack (torch==2.7.1, transformers==4.53.2, jax)
conflicts with OFT's (torch==2.2.0, transformers==4.40.1), so this backend
never runs in the same kernel as OFTBackend. Integration is via openpi's
own documented remote-inference pattern: this class is a thin websocket
client wrapping `openpi_client.websocket_client_policy.WebsocketClientPolicy`,
talking to a separate `serve_policy.py --env=LIBERO` process (serves
pi05_libero — D-07 as amended 2026-07-26; see
LIBERO/notebooks/03b-pi0-inference-smoketest.ipynb).

This module does not run any network/GPU logic at import time — all of
that happens inside `Pi0Backend.__init__` (constructing the websocket
client) and `predict()` (the actual `infer()` RPC call). Full real-inference
verification happens on Colab in Notebook B; this project has no local GPU
or openpi install (03-RESEARCH.md Environment Availability table). Locally,
`test_pi0_backend.py` verifies the obs-dict construction and predict()
plumbing via a mocked `WebsocketClientPolicy` — no real network/openpi
install needed.
"""

import socket
import time

import numpy as np

from openpi_client import image_tools
from openpi_client import websocket_client_policy as wcp

try:
    # websockets is an openpi-client dependency, present wherever
    # openpi_client is genuinely installed (Notebook B's Colab kernel).
    from websockets.exceptions import WebSocketException
except ImportError:  # local no-openpi test env fakes openpi_client only
    class WebSocketException(Exception):
        """Fallback so the retry tuple below is always importable."""


# Exceptions that mean "the connection to serve_policy.py broke" — worth a
# reconnect-and-retry — as opposed to a genuine inference error (RuntimeError
# raised by WebsocketClientPolicy.infer when the server reports a failure),
# which is NOT retried.
_RETRYABLE_EXCEPTIONS = (WebSocketException, ConnectionError, OSError)


class Pi0Backend:
    """pi0/openpi VLA backend satisfying the VLABackend Protocol."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        retry_backoffs: tuple = (30.0, 60.0),
    ):
        # Per D-06, this client never shares a kernel with OFTBackend — it
        # only ever talks to a separate serve_policy.py process over
        # websocket, on the same Colab VM but bound to localhost only
        # (03-RESEARCH.md Security Domain: never 0.0.0.0).
        #
        # retry_backoffs: seconds to wait before each reconnect attempt in
        # predict(). Live-run finding (2026-07-26, L4): openpi-client's
        # WebsocketClientPolicy hardcodes websockets' default keepalive
        # (ping_interval=20s, ping_timeout=20s) with no way to tune it, and
        # a server-side stall (most plausibly a JAX recompilation triggered
        # by a new input shape — episode 1's first inference) can outlive
        # that window, making the client close the socket with
        # "1011 keepalive ping timeout". Backing off then retrying succeeds
        # because the server's compile cache is warm by the retry.
        self._host = host
        self._port = port
        self._retry_backoffs = tuple(retry_backoffs)
        self.client = wcp.WebsocketClientPolicy(host, port)

    def _wait_for_port(self, timeout_s: float = 30.0) -> bool:
        """Return True if (host, port) accepts TCP connections within timeout_s.

        Guards the reconnect path: WebsocketClientPolicy.__init__ blocks
        indefinitely in _wait_for_server() when the server process is dead,
        so probe the port first and fail fast instead of hanging.
        """
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(2)
                if s.connect_ex((self._host, self._port)) == 0:
                    return True
            time.sleep(2)
        return False

    def _reconnect(self) -> None:
        self.client = wcp.WebsocketClientPolicy(self._host, self._port)

    def _robot_state(self) -> np.ndarray:
        """Placeholder proprioceptive state vector.

        Documented limitation: this phase has no proprioceptive state
        wired in yet (CONTEXT.md scope is proving the interface swap, not
        full state-conditioning parity with OFT). openpi's LIBERO configs
        are documented to tolerate a state vector, and
        `serve_policy.py --env=LIBERO` expects the
        "observation/state" key to be present, so a zero-vector placeholder
        is returned rather than omitting the key entirely. A future phase
        that wires real proprioceptive state should replace this method.
        """
        return np.zeros(8, dtype=np.float32)

    def predict(self, images: dict, language: str) -> np.ndarray:
        """Return an action chunk from the remote pi0/pi0-FAST policy server.

        Per D-01, `images` is a dict of named camera views; this backend
        only consumes the "eye_in_hand" key (per D-01, this phase only
        populates that key — both openpi obs image keys below reuse it
        since no separate wrist camera exists in this phase's scope). Per
        D-02, all normalization is openpi's internal responsibility once
        the observation dict is handed to `infer()` — there is no shared
        normalization layer.

        Transient connection drops (e.g. the 2026-07-26 keepalive-timeout
        failure) are retried: back off, verify the server port is alive,
        rebuild the client, re-send the same observation. A dead server
        (closed port) or exhausted retries raise RuntimeError pointing at
        /content/serve_policy.log.
        """
        img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(np.asarray(images["eye_in_hand"]), 224, 224)
        )
        obs = {
            "observation/image": img,
            "observation/wrist_image": img,
            "observation/state": self._robot_state(),
            "prompt": language,
        }

        attempts = len(self._retry_backoffs) + 1
        for attempt in range(attempts):
            try:
                result = self.client.infer(obs)
                return np.asarray(result["actions"])
            except _RETRYABLE_EXCEPTIONS as e:
                if attempt == attempts - 1:
                    raise RuntimeError(
                        f"pi0 server connection failed {attempts} times in a row "
                        f"(last error: {type(e).__name__}: {e}). The server is "
                        "likely dead or wedged — check the tail of "
                        "/content/serve_policy.log for a crash traceback or stall."
                    ) from e
                backoff = self._retry_backoffs[attempt]
                print(
                    f"[Pi0Backend] connection lost ({type(e).__name__}); "
                    f"waiting {backoff:.0f}s then reconnecting "
                    f"(attempt {attempt + 2}/{attempts})"
                )
                time.sleep(backoff)
                if not self._wait_for_port():
                    raise RuntimeError(
                        f"pi0 server port {self._host}:{self._port} stopped "
                        "accepting connections — the serve_policy.py process "
                        "has died. Check the tail of /content/serve_policy.log "
                        "for the crash traceback (OOM is the usual suspect)."
                    ) from e
                self._reconnect()
