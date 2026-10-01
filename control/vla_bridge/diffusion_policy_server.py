"""Colab-side launcher: lerobot 0.6.1's `PolicyServer`, with the upstream
Diffusion Policy async-inference fix applied (Plan 14-06, `--backend diffusion`).

Run this on Colab INSTEAD OF `python -m lerobot.async_inference.policy_server`,
with the same arguments:

    python /content/diffusion_policy_server.py --host 0.0.0.0 --port 5173

Why it exists (verified against the installed `lerobot==0.6.1`, 2026-10-01):
- `PolicyServer._get_action_chunk()` calls `policy.predict_action_chunk(obs)`
  directly, with one observation. The state arrives as `(B, state_dim)` and each
  image as `(B, C, H, W)`, without the `n_obs_steps` time dimension that
  `DiffusionPolicy.select_action()` would normally build from its queues.
- `DiffusionPolicy.predict_action_chunk()`'s queue-less branch only adds a
  length-1 time dimension to images and leaves the state 2-D. So
  `DiffusionModel.generate_actions()` reads `n_obs_steps` as `state_dim` and
  fails `assert n_obs_steps == self.config.n_obs_steps` on every inference. The
  server logs `Error in StreamActions`, never returns a chunk, and the robot
  only holds position.
- This is upstream huggingface/lerobot#3861. The fix, PR #3886, is open and
  unmerged, and 0.6.1 is the newest release on PyPI. `predict_action_chunk`
  below is 0.6.1's method with exactly PR #3886's change: a single frame is
  repeated along a new time dimension of length `n_obs_steps`. That is the
  same warm-up `select_action()` does through `populate_queues()`. Batches that
  already carry the time dimension are left unchanged.

The patch only replaces `DiffusionPolicy.predict_action_chunk`. Every other
policy the server can load is untouched, and the server itself is the
unmodified `lerobot.async_inference.policy_server.serve`.
`test_diffusion_policy_server.py` reproduces the failure on the stock server
and checks the patched one through the real installed code path.

The patch does not add missing-camera masking: Diffusion Policy still needs
every image key it declares (see `backends/diffusion_policy.py`).
"""

import importlib.metadata

import torch
from lerobot.policies.diffusion import modeling_diffusion
from lerobot.utils.constants import OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

PATCHED_LEROBOT_VERSION = "0.6.1"
UPSTREAM_ISSUE = "https://github.com/huggingface/lerobot/issues/3861"
UPSTREAM_FIX = "https://github.com/huggingface/lerobot/pull/3886"


@torch.no_grad()
def predict_action_chunk(self, batch, noise=None):
    """lerobot 0.6.1 `DiffusionPolicy.predict_action_chunk` with upstream PR #3886 applied."""
    queues_populated = any(len(q) > 0 for q in self._queues.values())
    if queues_populated:
        batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
    else:
        batch = dict(batch)
        n_obs_steps = self.config.n_obs_steps
        if batch[OBS_STATE].ndim == 2:
            batch[OBS_STATE] = batch[OBS_STATE].unsqueeze(1).expand(-1, n_obs_steps, -1)
        if self.config.env_state_feature and batch[OBS_ENV_STATE].ndim == 2:
            batch[OBS_ENV_STATE] = batch[OBS_ENV_STATE].unsqueeze(1).expand(-1, n_obs_steps, -1)
        if self.config.image_features:
            for key in self.config.image_features:
                if batch[key].ndim == 4:
                    batch[key] = batch[key].unsqueeze(1).expand(-1, n_obs_steps, -1, -1, -1)
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
    return self.diffusion.generate_actions(batch, noise=noise)


def _installed_lerobot_version() -> str:
    return importlib.metadata.version("lerobot")


def apply_patch() -> None:
    """Install the fixed `predict_action_chunk` on `DiffusionPolicy`. Idempotent.

    Refuses any lerobot other than the reviewed 0.6.1: the patched body is a
    copy of that version's method, so on another version it could silently
    revert unrelated upstream changes.
    """
    installed = _installed_lerobot_version()
    if installed != PATCHED_LEROBOT_VERSION:
        raise SystemExit(
            f"diffusion_policy_server.py patches lerobot=={PATCHED_LEROBOT_VERSION}, but {installed} "
            f"is installed. Check whether {UPSTREAM_FIX} has landed, then re-review this patch."
        )
    modeling_diffusion.DiffusionPolicy.predict_action_chunk = predict_action_chunk


def main() -> None:
    apply_patch()
    print(f"Patched DiffusionPolicy.predict_action_chunk for async inference ({UPSTREAM_ISSUE}).")
    from lerobot.async_inference import policy_server

    policy_server.serve()  # draccus parses --host/--port from sys.argv, as with `python -m`


if __name__ == "__main__":
    main()
