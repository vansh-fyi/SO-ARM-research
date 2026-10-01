"""GR00T N1.7 action-space conversion for `--backend groot-n17` (Plan 14-07, VLAB-10).

The plan assumed GR00T N1.7 hands the client a relative end-effector vector
(the 132-D cross-embodiment action) that must be sliced and run through
`eef_to_joint.solve_ik()`. Checked against the installed `lerobot==0.6.1`
source and the real SO-101 N1.7 checkpoints on 2026-10-01, that is not what
reaches this machine:

- **The 132-D vector never leaves the server.** `GrootPolicy.predict_action_chunk()`
  slices the model output to `config.output_features["action"].shape[0]`
  (`policies/groot/modeling_groot.py`), and `GrootN17ActionDecodeStep` slices
  again to `env_action_dim` (`processor_groot.py`). For every SO-101 checkpoint
  that is 6.
- **SO-101 N1.7 checkpoints are joint-space, not EEF.** Every SO-101 N1.7
  checkpoint on the HF Hub (scan of 2026-10-01, see `backend_configs/groot_n17.json`)
  declares the action groups `single_arm` (5) + `gripper` (1), with
  `single_arm` either `RELATIVE`/`NON_EEF` (state-relative joint deltas) or
  plain absolute. None uses an `EEF` group.
- **Relative actions are decoded server-side.** `GrootN17ActionDecodeStep`
  un-normalizes each group and adds the raw state cached by
  `GrootN17PackInputsStep` at preprocess time
  (`decoded[key] + reference[:, None, :]` for `NON_EEF`). The postprocessor's
  output, which is what the gRPC bridge delivers, is an ABSOLUTE joint target
  in the checkpoint's dataset units.

So the client receives absolute joint targets. Running them through a
relative-EE/IK conversion would treat an absolute target as a delta and add it
to the current pose again, which drives joints to their limits. That is the
community-reported N1.7 SO-ARM failure in 14-RESEARCH.md Pitfall 1. The
declared action space is therefore `absolute_joint`, and this module is the
explicit gate for it, not an IK step:

- the action space and units are DECLARED in `backend_configs/groot_n17.json`
  (`action_space`, `action_units`) and checked at build time;
- `convert_groot_n17_action()` accepts only a JOINT_ORDER-shaped absolute
  target, and never re-applies `joint_state`.

Units are the real unit-frame risk. SO-101 datasets exist in degree mode
(matches `action_contract.py`), range-normalized mode (arm joints clamp at
+/-100) and radians (`alexhegit/so101-simstudio-lab01-pnp-groot-n17`, a MuJoCo
sim, arm range about +/-1.8). Only degree-mode checkpoints are supported;
choose one by reading its normalizer stats (see `groot_n17.json`'s notes).

`eef_to_joint.solve_ik()` is intentionally NOT called. If an SO-101 N1.7
checkpoint with an `EEF` (`xyz+rot6d`) action group ever appears, the server
would still decode it to an ABSOLUTE pose (`relative_eef_to_absolute`), and the
gRPC client maps the action tensor positionally onto the 6 motor keys, so it
would need its own action space here plus a client-side change. Add it then,
against that checkpoint's config, rather than guessing now.
"""

import math
from collections.abc import Callable, Mapping, Sequence

from vla_bridge.action_contract import JOINT_ORDER

ACTION_SPACE_ABSOLUTE_JOINT = "absolute_joint"
SUPPORTED_ACTION_SPACES = (ACTION_SPACE_ABSOLUTE_JOINT,)

ACTION_UNITS_DEGREES = "degrees"
SUPPORTED_ACTION_UNITS = (ACTION_UNITS_DEGREES,)

_RELATIVE_EE_REASON = (
    "the installed lerobot 0.6.1 PolicyServer decodes GR00T N1.7 actions to absolute "
    "joint targets server-side (GrootN17ActionDecodeStep), and every SO-101 N1.7 checkpoint "
    "declares joint-space action groups, so no relative end-effector vector reaches the "
    "client; converting an absolute target as a relative one drives joints to their limits "
    "(14-RESEARCH.md Pitfall 1)"
)


def check_action_space(action_space, action_units) -> None:
    """Raise `ValueError` unless the declared action space and units are supported."""
    if action_space is None:
        raise ValueError(
            "GR00T N1.7 needs an explicit 'action_space' declaration; "
            f"supported: {', '.join(SUPPORTED_ACTION_SPACES)}."
        )
    if action_space not in SUPPORTED_ACTION_SPACES:
        reason = f" ({_RELATIVE_EE_REASON})" if "relative" in str(action_space) else ""
        raise ValueError(
            f"Unsupported GR00T N1.7 action_space {action_space!r}{reason}; "
            f"supported: {', '.join(SUPPORTED_ACTION_SPACES)}."
        )
    if action_units is None:
        raise ValueError(
            "GR00T N1.7 needs an explicit 'action_units' declaration; "
            f"supported: {', '.join(SUPPORTED_ACTION_UNITS)}."
        )
    if action_units not in SUPPORTED_ACTION_UNITS:
        raise ValueError(
            f"Unsupported GR00T N1.7 action_units {action_units!r}: action_contract.py and "
            "safety_validator.py work in degrees, so a radians or range-normalized checkpoint "
            f"would be misread; supported: {', '.join(SUPPORTED_ACTION_UNITS)}."
        )


def _as_real(value, joint: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"GR00T N1.7 action for {joint!r} is not a number: {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"GR00T N1.7 action for {joint!r} is not finite: {value!r}")
    return value


def convert_groot_n17_action(
    raw_action,
    joint_state: Mapping[str, float],
    action_space: str = ACTION_SPACE_ABSOLUTE_JOINT,
) -> dict[str, float]:
    """Gate one server-decoded GR00T N1.7 action into a JOINT_ORDER dict of floats.

    `raw_action` is either a dict keyed by `JOINT_ORDER` (what the gRPC bridge
    delivers after `pop_validated_action()` strips `.pos`), or a flat vector. A
    vector is sliced to its first `len(JOINT_ORDER)` values, the same slice
    lerobot itself applies to the padded 132-D output: the SO-101 action groups
    are `single_arm` (shoulder_pan .. wrist_roll) then `gripper`, which is
    `JOINT_ORDER`.

    `joint_state` is accepted for the shared `convert_fn(action, joint_state)`
    signature and deliberately unused: the target is already absolute.

    Raises `ValueError` on an unsupported action space or a missing, non-numeric
    or non-finite joint. Callers hold position on any exception.
    """
    if action_space not in SUPPORTED_ACTION_SPACES:
        check_action_space(action_space, ACTION_UNITS_DEGREES)
    if isinstance(raw_action, Mapping):
        missing = [joint for joint in JOINT_ORDER if joint not in raw_action]
        if missing:
            raise ValueError(f"GR00T N1.7 action is missing joints {missing}")
        return {joint: _as_real(raw_action[joint], joint) for joint in JOINT_ORDER}
    if isinstance(raw_action, Sequence) and not isinstance(raw_action, (str, bytes)):
        if len(raw_action) < len(JOINT_ORDER):
            raise ValueError(
                f"GR00T N1.7 action has {len(raw_action)} values; expected at least {len(JOINT_ORDER)}"
            )
        return {joint: _as_real(raw_action[i], joint) for i, joint in enumerate(JOINT_ORDER)}
    raise ValueError(f"GR00T N1.7 action has an unsupported type: {type(raw_action).__name__}")


def make_convert_fn(action_space, action_units) -> Callable[[dict, dict], dict]:
    """Validate the declared action space/units now; return the per-step convert_fn."""
    check_action_space(action_space, action_units)

    def convert_fn(raw_action, joint_state):
        return convert_groot_n17_action(raw_action, joint_state, action_space=action_space)

    return convert_fn


__all__ = [
    "ACTION_SPACE_ABSOLUTE_JOINT",
    "ACTION_UNITS_DEGREES",
    "SUPPORTED_ACTION_SPACES",
    "SUPPORTED_ACTION_UNITS",
    "check_action_space",
    "convert_groot_n17_action",
    "make_convert_fn",
]
