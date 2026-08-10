"""Shared VLA backend interface (D-01, D-02, D-04).

Any Vision-Language-Action model wired into this project's LIBERO/SOARM
pipeline implements this single-method contract so the eval loop
(`eval_loop.py`) never needs to branch on which backend is active.

Design notes:
- `images` is a dict of named camera views, not a single image (D-01).
  Phase 3 populated only the `"eye_in_hand"` key; Phase 5's multi-camera
  work (SPAT-01/02, D-01) now also populates the `"agentview"` key,
  without needing a signature change.
- Each backend normalizes its own actions/observations internally inside its
  `predict()` implementation (D-02). There is no shared normalization layer —
  OFT overlays its confirmed `libero_spatial_no_noops` norm_stats itself,
  and other backends (e.g. a future pi0 backend) do whatever normalization
  they require internally.
"""

from typing import Protocol, runtime_checkable

import numpy as np
from PIL import Image


@runtime_checkable
class VLABackend(Protocol):
    """Contract every VLA backend (OFTBackend, future Pi0Backend, ...) satisfies."""

    def predict(self, images: dict[str, Image.Image], language: str) -> np.ndarray:
        """Return an action (or action chunk) for the given observation.

        Args:
            images: dict of named camera views, e.g. {"eye_in_hand": PIL.Image}.
                Phase 5 (SPAT-01/SPAT-02, D-01) now populates both the
                "eye_in_hand" and "agentview" keys on every eval_loop step;
                later phases may add more keys without changing this
                signature.
            language: natural-language task instruction string.

        Returns:
            np.ndarray action chunk, e.g. shape (8, 7) for OpenVLA-OFT's
            8-step chunk, or a single (7,) action for backends that don't
            chunk. Each backend normalizes internally (D-02) — callers must
            not apply any shared normalization on top of this return value.
        """
        ...
