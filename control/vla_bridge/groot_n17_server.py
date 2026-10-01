"""Colab-side launcher: lerobot 0.6.1's `PolicyServer`, decoding GR00T N1.7
action chunks whole (Plan 14-07, `--backend groot-n17`).

Run this on Colab INSTEAD OF `python -m lerobot.async_inference.policy_server`,
with the same arguments:

    python /content/groot_n17_server.py --host 0.0.0.0 --port 5173

Why it exists (verified against the installed `lerobot==0.6.1`, 2026-10-01):
- `PolicyServer._predict_action_chunk()` runs the postprocessor once per chunk
  step, on a `(B, action_dim)` slice (`policy_server.py`, step "4. Apply
  postprocessor").
- For GR00T N1.7 the postprocessor is `GrootN17ActionDecodeStep`
  (`policies/groot/processor_groot.py`). For checkpoints trained with native
  relative actions (`use_relative_action: true`, i.e. most SO-101 N1.7
  fine-tunes, e.g. `tsangb34/groot-n17-so101-cube-drawer-40k`) it raises
  `NotImplementedError("GrootN17ActionDecodeStep cannot decode native relative
  actions one step at a time ...")` on a 2-D action. Every inference fails, the
  server logs `Error in StreamActions`, and the robot only holds position.
- Even without that guard, per-step decoding would be wrong: relative
  statistics carry one row per chunk step, and a popped `(B, D)` action is
  decoded as step 0. The step's own docstring says the async policy server
  should decode the whole chunk right after prediction, while the matching
  state is still cached by `GrootN17PackInputsStep`; 0.6.1's server does not.
- No upstream issue or PR for this was found (GitHub issue search,
  2026-10-01), and upstream `main`'s `policy_server.py` still decodes per step.

The patch replaces `PolicyServer._predict_action_chunk` with a dispatcher. For
`policy_type == "groot"` it runs 0.6.1's method body with ONE change: the
postprocessor is applied once to the whole `(B, T, D)` chunk (both GR00T decode
steps, `GrootN17ActionDecodeStep` and `GrootActionUnpackUnnormalizeStep`,
support `(B, T, D)`). Every other policy type goes through the original method
unchanged, and the server itself is the stock
`lerobot.async_inference.policy_server.serve`.
`test_backend_groot_n17.py` reproduces the failure on the stock method and
checks the patched one with the real installed pack/decode steps.

What reaches the robot is an ABSOLUTE joint target in the checkpoint's dataset
units; `backends/relative_ee_conversion.py` explains why no client-side
relative-EE conversion is applied.
"""

import importlib.metadata
import time

from lerobot.async_inference import policy_server

PATCHED_LEROBOT_VERSION = "0.6.1"
POLICY_TYPE = "groot"
_PATCH_MARKER = "_groot_n17_chunk_decode_patch"


def _predict_groot_action_chunk(server, observation_t):
    """lerobot 0.6.1 `PolicyServer._predict_action_chunk`, decoding the chunk in one call."""
    start_prepare = time.perf_counter()
    observation = policy_server.raw_observation_to_observation(
        observation_t.get_observation(),
        server.lerobot_features,
        server.policy_image_features,
    )
    observation = server.preprocessor(observation)
    server.last_processed_obs = observation_t

    start_inference = time.perf_counter()
    action_tensor = server._get_action_chunk(observation)  # (B, chunk_size, action_dim)
    inference_time = time.perf_counter() - start_inference
    server.logger.info(
        f"Preprocessing and inference took {inference_time:.4f}s, action shape: {action_tensor.shape}"
    )

    # The one change from 0.6.1: decode the whole (B, T, D) chunk at once, while the
    # pack step still caches this observation's state, so relative N1.7 actions are
    # turned into absolute targets with the right per-step statistics.
    action_tensor = server.postprocessor(action_tensor)
    action_tensor = action_tensor.squeeze(0).detach().cpu()  # (T, D), as the stock path returns

    action_chunk = server._time_action_chunk(
        observation_t.get_timestamp(), list(action_tensor), observation_t.get_timestep()
    )
    server.logger.info(
        f"Observation {observation_t.get_timestep()} | "
        f"Total time: {1000 * (time.perf_counter() - start_prepare):.2f}ms (GR00T whole-chunk decode)"
    )
    return action_chunk


def _installed_lerobot_version() -> str:
    return importlib.metadata.version("lerobot")


def is_patched() -> bool:
    return getattr(policy_server.PolicyServer._predict_action_chunk, _PATCH_MARKER, False)


def apply_patch() -> None:
    """Install the GR00T whole-chunk dispatcher on `PolicyServer`. Idempotent.

    Refuses any lerobot other than the reviewed 0.6.1: the GR00T branch copies
    that version's method body, so on another version it could silently revert
    unrelated upstream changes.
    """
    installed = _installed_lerobot_version()
    if installed != PATCHED_LEROBOT_VERSION:
        raise SystemExit(
            f"groot_n17_server.py patches lerobot=={PATCHED_LEROBOT_VERSION}, but {installed} is "
            "installed. Check whether its PolicyServer now decodes GR00T chunks whole, then re-review."
        )
    if is_patched():
        return
    original = policy_server.PolicyServer._predict_action_chunk

    def _predict_action_chunk(self, observation_t):
        if self.policy_type == POLICY_TYPE:
            return _predict_groot_action_chunk(self, observation_t)
        return original(self, observation_t)

    setattr(_predict_action_chunk, _PATCH_MARKER, True)
    _predict_action_chunk.__wrapped__ = original
    policy_server.PolicyServer._predict_action_chunk = _predict_action_chunk


def main() -> None:
    apply_patch()
    print("Patched PolicyServer._predict_action_chunk: GR00T N1.7 chunks are decoded whole.")
    policy_server.serve()  # draccus parses --host/--port from sys.argv, as with `python -m`


if __name__ == "__main__":
    main()
