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

from .eval_loop import run_episode, run_suite, print_episode_result
