from .interface import VLABackend

try:
    # torch is a Colab-only GPU dependency for this project (03-RESEARCH.md
    # Environment Availability) — not installed in the local dev environment.
    # Degrade gracefully so `eval_loop`'s local, no-GPU pytest suite can still
    # import this package; on Colab (torch installed) this import succeeds
    # and OFTBackend is exported normally.
    from .oft_backend import OFTBackend
except ImportError:
    OFTBackend = None

try:
    # openpi_client is only installed in Notebook B's separate kernel (D-06:
    # confirmed torch/transformers/jax version conflict with OFT's stack
    # means openpi_client is never installed alongside OFTBackend). Degrade
    # gracefully so Notebook A's `from libero.libero.vla import OFTBackend`
    # (and this local, no-GPU/no-openpi pytest suite) never hard-fails.
    from .pi0_backend import Pi0Backend
except ImportError:
    Pi0Backend = None

from .eval_loop import run_episode, run_suite, print_episode_result
