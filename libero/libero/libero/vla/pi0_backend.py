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
talking to a separate `serve_policy.py --env=pi0_fast_libero` process
(see LIBERO/notebooks/03b-pi0-inference-smoketest.ipynb).

This module does not run any network/GPU logic at import time — all of
that happens inside `Pi0Backend.__init__` (constructing the websocket
client) and `predict()` (the actual `infer()` RPC call). Full real-inference
verification happens on Colab in Notebook B; this project has no local GPU
or openpi install (03-RESEARCH.md Environment Availability table). Locally,
`test_pi0_backend.py` verifies the obs-dict construction and predict()
plumbing via a mocked `WebsocketClientPolicy` — no real network/openpi
install needed.
"""

import numpy as np

from openpi_client import image_tools
from openpi_client import websocket_client_policy as wcp


class Pi0Backend:
    """pi0/openpi VLA backend satisfying the VLABackend Protocol."""

    def __init__(self, host: str = "localhost", port: int = 8000):
        # Per D-06, this client never shares a kernel with OFTBackend — it
        # only ever talks to a separate serve_policy.py process over
        # websocket, on the same Colab VM but bound to localhost only
        # (03-RESEARCH.md Security Domain: never 0.0.0.0).
        self.client = wcp.WebsocketClientPolicy(host, port)

    def _robot_state(self) -> np.ndarray:
        """Placeholder proprioceptive state vector.

        Documented limitation: this phase has no proprioceptive state
        wired in yet (CONTEXT.md scope is proving the interface swap, not
        full state-conditioning parity with OFT). openpi's LIBERO configs
        are documented to tolerate a state vector, and
        `serve_policy.py --env=pi0_fast_libero` expects the
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
        result = self.client.infer(obs)
        return np.asarray(result["actions"])
