"""Shared prompt construction for every MLLM backend (Phase 15, PIPE-02).

Two parts, identical across providers so a cross-backend comparison is not
confounded by prompt wording:

  - `build_system_prompt(settings, joint_limits_deg, urdf_text=None)`: the
    static block, built ONCE per episode (cache-friendly, identical on every
    call). Headed sections: ROLE, OUTPUT, JOINTS AND UNITS, TIMING (plus the
    optional PER-STEP CHANGE CAPS block), INPUTS, COMPLETION, CONTROL, and
    the optional ARM KINEMATICS (URDF) block.
  - `build_user_text(joint_state, instruction, settings, previous_plan,
    depth_note)`: the per-call text that follows the labelled images.

Decisions implemented:
  - D-01/D-17: absolute targets for all six joints, exactly N steps.
  - D-16/D-24: the step duration is the MEASURED `settings["step_dt_s"]`
    (0.785 s), not the nominal `--control-hz` step. Only a qualitative
    "a safety layer limits large changes" sentence is always present; the
    numeric `safety_validator.MAX_RELATIVE_TARGET_DEG` caps appear only under
    CAPS_HEADING when `settings["disclose_safety_caps"]` is true.
  - D-26: the depth map is stated to be approximately, not pixel-exactly,
    aligned with the raw left/right views.
  - D-28: the depth wording follows `settings["depth_presentation"]`
    (heatmap / grid / both); the heatmap's fixed colour scale comes from
    `settings["depth"]["range_m"]` via `depth_scale_text()`.
  - D-09: only the previous chunk's plan is carried into the next call.

Vocabulary rule (research conflict C4): no generated text may contain the
words in FORBIDDEN_PROMPT_TERMS (Claude refuses prompts that ask it to
reproduce its deliberation) or any movement-primitive name in
PRIMITIVE_TERMS (the model drives raw joint targets only).
`lint_prompt_text()` checks both lists and is unit-tested over every
settings combination.

Templates are module-level constants substituted with `str.replace`, never
`str.format`, so literal braces never need escaping. `template_sha256()`
hashes PROMPT_TEMPLATE_VERSION plus every template constant; bump the
version whenever a template changes (D-22b provenance).
"""

import hashlib
import json
from decimal import ROUND_HALF_UP, Decimal

from vla_bridge import safety_validator
from vla_bridge.action_contract import ACTION_MODE, ACTION_UNITS, JOINT_ORDER

PROMPT_TEMPLATE_VERSION = "mllm-prompt-1"

FORBIDDEN_PROMPT_TERMS = (
    "reasoning",
    "thinking",
    "trace",
    "scratchpad",
    "chain-of-thought",
    "chain of thought",
)
PRIMITIVE_TERMS = ("move_to", "grasp(", "pick(", "place(", "waypoint", "trajectory helper")

CAPS_HEADING = "PER-STEP CHANGE CAPS"
URDF_HEADING = "ARM KINEMATICS (URDF)"

# Longest previous plan carried into the next call (D-09).
MAX_PREVIOUS_PLAN_CHARS = 2000
FIRST_CALL_PLAN_MARKER = "none (this is the first call)"

ROLE_TEMPLATE = (
    "ROLE\n"
    "You are controlling a real 6-joint SO-ARM101 robot arm on a lab table. On every call you "
    "receive the current camera views and joint state, and you decide the arm's next <<N>> "
    "joint targets for the operator's task."
)

OUTPUT_TEMPLATE = (
    "OUTPUT\n"
    "Return one JSON object that matches the supplied schema, with its fields in this order:\n"
    "- task_restatement: restate the task in one sentence.\n"
    "- plan: your plan for the next <<N>> steps.\n"
    "- scene_claims: claims about the scene; each claim cites the image it came from "
    "(wrist, left, right or depth) and gives a confidence (low, medium or high).\n"
    "- actions: exactly <<N>> steps; each step gives a target for all six joints.\n"
    "- task_complete: see COMPLETION."
)

JOINTS_TEMPLATE = (
    "JOINTS AND UNITS\n"
    "Every step must contain all six joints below. Targets are <<ACTION_MODE>> joint positions, "
    "not changes from the current position.\n"
    "<<JOINT_LINES>>\n"
    "Angle 0 degrees is the calibrated midpoint of each arm joint. "
    "Gripper: 0 = fully closed, 100 = fully open."
)

JOINT_LINE_TEMPLATE = "- <<JOINT>>: <<UNIT>>, calibrated range <<LO>> to <<HI>>"

TIMING_TEMPLATE = (
    "TIMING\n"
    "Each step lasts about <<DT>> s (measured on this arm). <<N>> steps = <<CHUNK_S>> s of motion. "
    "The arm cannot be commanded to jump arbitrarily: large changes between consecutive steps "
    "are limited by a safety layer, so spread a large motion over several steps."
)

CAPS_TEMPLATE = (
    CAPS_HEADING + "\n"
    "The safety layer limits the change from the current position within one step to at most:\n"
    "<<CAP_LINES>>"
)

CAP_LINE_TEMPLATE = "- <<JOINT>>: <<CAP>> <<UNIT>>"

INPUTS_TEMPLATE = (
    "INPUTS\n"
    "Images arrive in this order when present, each preceded by a text label: wrist, left, right, "
    "depth. wrist is the camera on the gripper; left and right are the fixed overhead stereo pair.\n"
    "<<DEPTH_TEXT>>\n"
    "The depth map is approximately, not pixel-exactly, aligned with the raw left and right views, "
    "because it is computed on a rectified, resized left image.\n"
    "The current joint state is supplied on every call. Only your previous plan is remembered "
    "between calls; earlier images and steps are not."
)

DEPTH_HEATMAP_TEMPLATE = "Depth arrives as a colour heatmap image (label: depth). <<SCALE>>"

DEPTH_GRID_TEMPLATE = (
    "A numeric depth grid arrives in the user message: median metres per cell, "
    "n/a where depth is invalid, plus min/median/max statistics."
)

DEPTH_SCALE_TEMPLATE = (
    "Heatmap colour scale (turbo), fixed range <<NEAR>> m to <<FAR>> m: blue = near, "
    "green and yellow = middle, red = far, black = no valid depth. Values outside the range "
    "are clipped to it. A legend is drawn on the image."
)

COMPLETION_TEMPLATE = (
    "COMPLETION\n"
    "Set task_complete to true only when the task goal is already achieved in the CURRENT camera "
    "views. Otherwise set it to false."
)

CONTROL_TEMPLATE = (
    "CONTROL\n"
    "You drive the arm only by emitting raw absolute joint targets. There are no helper routines, "
    "and nothing in your text is executed."
)

URDF_TEMPLATE = URDF_HEADING + "\n<<URDF>>"

USER_TASK_TEMPLATE = "TASK: <<INSTRUCTION>>"
USER_STATE_TEMPLATE = "CURRENT JOINT STATE: <<JOINT_STATE>>"
USER_PLAN_TEMPLATE = "PREVIOUS PLAN: <<PREVIOUS_PLAN>>"

USER_REMINDER_TEMPLATE = "Return exactly <<N>> steps."

_TEMPLATES = (
    ROLE_TEMPLATE,
    OUTPUT_TEMPLATE,
    JOINTS_TEMPLATE,
    JOINT_LINE_TEMPLATE,
    TIMING_TEMPLATE,
    CAPS_TEMPLATE,
    CAP_LINE_TEMPLATE,
    INPUTS_TEMPLATE,
    DEPTH_HEATMAP_TEMPLATE,
    DEPTH_GRID_TEMPLATE,
    DEPTH_SCALE_TEMPLATE,
    COMPLETION_TEMPLATE,
    CONTROL_TEMPLATE,
    URDF_TEMPLATE,
    USER_TASK_TEMPLATE,
    USER_STATE_TEMPLATE,
    USER_PLAN_TEMPLATE,
    USER_REMINDER_TEMPLATE,
    FIRST_CALL_PLAN_MARKER,
)

_CAP_UNIT_WORDS = {"degrees": "degrees", "percent_0_100": "percentage points"}


def lint_prompt_text(text: str) -> list[str]:
    """Return the forbidden or primitive terms found in `text` (case-insensitive)."""
    lowered = text.lower()
    return [term for term in FORBIDDEN_PROMPT_TERMS + PRIMITIVE_TERMS if term.lower() in lowered]


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def template_sha256() -> str:
    """Hash of PROMPT_TEMPLATE_VERSION plus every template constant (D-22b)."""
    return text_sha256("\0".join((PROMPT_TEMPLATE_VERSION,) + _TEMPLATES))


def _chunk_seconds(chunk_len: int, step_dt_s: float) -> str:
    """N x dt rounded half-up to one decimal (10 x 0.785 = 7.85 -> "7.9")."""
    return str((Decimal(repr(float(step_dt_s))) * chunk_len).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def depth_scale_text(settings: dict) -> str:
    """The fixed heatmap colour-scale statement built from `settings["depth"]["range_m"]` (D-28)."""
    near, far = settings["depth"]["range_m"]
    return (DEPTH_SCALE_TEMPLATE
            .replace("<<NEAR>>", f"{float(near):.2f}")
            .replace("<<FAR>>", f"{float(far):.2f}"))


def _depth_text(settings: dict) -> str:
    presentation = settings["depth_presentation"]
    heatmap = DEPTH_HEATMAP_TEMPLATE.replace("<<SCALE>>", depth_scale_text(settings))
    if presentation == "heatmap":
        return heatmap
    if presentation == "grid":
        return DEPTH_GRID_TEMPLATE
    if presentation == "both":
        return heatmap + "\n" + DEPTH_GRID_TEMPLATE
    raise ValueError(f"depth_presentation must be heatmap, grid or both, got {presentation!r}")


def build_system_prompt(settings: dict, joint_limits_deg: dict, urdf_text: str | None = None) -> str:
    """The static system block shared by every call of an episode."""
    n = str(settings["chunk_len"])
    joint_lines = "\n".join(
        JOINT_LINE_TEMPLATE
        .replace("<<JOINT>>", joint)
        .replace("<<UNIT>>", ACTION_UNITS[joint])
        .replace("<<LO>>", f"{float(joint_limits_deg[joint][0]):.2f}")
        .replace("<<HI>>", f"{float(joint_limits_deg[joint][1]):.2f}")
        for joint in JOINT_ORDER
    )
    sections = [
        ROLE_TEMPLATE.replace("<<N>>", n),
        OUTPUT_TEMPLATE.replace("<<N>>", n),
        JOINTS_TEMPLATE.replace("<<ACTION_MODE>>", ACTION_MODE).replace("<<JOINT_LINES>>", joint_lines),
        TIMING_TEMPLATE
        .replace("<<DT>>", f"{float(settings['step_dt_s']):g}")
        .replace("<<N>>", n)
        .replace("<<CHUNK_S>>", _chunk_seconds(settings["chunk_len"], settings["step_dt_s"])),
    ]
    if settings.get("disclose_safety_caps") is True:
        cap_lines = "\n".join(
            CAP_LINE_TEMPLATE
            .replace("<<JOINT>>", joint)
            .replace("<<CAP>>", f"{safety_validator.MAX_RELATIVE_TARGET_DEG[joint]:g}")
            .replace("<<UNIT>>", _CAP_UNIT_WORDS[ACTION_UNITS[joint]])
            for joint in JOINT_ORDER
        )
        sections.append(CAPS_TEMPLATE.replace("<<CAP_LINES>>", cap_lines))
    sections += [
        INPUTS_TEMPLATE.replace("<<DEPTH_TEXT>>", _depth_text(settings)),
        COMPLETION_TEMPLATE,
        CONTROL_TEMPLATE,
    ]
    if urdf_text is not None:
        sections.append(URDF_TEMPLATE.replace("<<URDF>>", urdf_text))
    return "\n\n".join(sections)


def build_user_text(
    joint_state: dict,
    instruction: str,
    settings: dict,
    previous_plan: str | None = None,
    depth_note: str = "",
) -> str:
    """The per-call text block that follows the labelled images."""
    state = {}
    for joint in JOINT_ORDER:
        value = joint_state.get(joint)
        state[joint] = round(float(value), 3) if value is not None else None
    if isinstance(previous_plan, str) and previous_plan.strip():
        plan = previous_plan[:MAX_PREVIOUS_PLAN_CHARS]
    else:
        plan = FIRST_CALL_PLAN_MARKER
    # One substitution per line, so model-written plan text can never expand
    # another placeholder.
    header = "\n".join([
        USER_TASK_TEMPLATE.replace("<<INSTRUCTION>>", str(instruction)),
        USER_STATE_TEMPLATE.replace("<<JOINT_STATE>>", json.dumps(state)),
        USER_PLAN_TEMPLATE.replace("<<PREVIOUS_PLAN>>", plan),
    ])
    parts = [header]
    if depth_note:
        parts.append(depth_note)
    parts.append(USER_REMINDER_TEMPLATE.replace("<<N>>", str(settings["chunk_len"])))
    return "\n\n".join(parts)


__all__ = [
    "CAPS_HEADING",
    "FIRST_CALL_PLAN_MARKER",
    "FORBIDDEN_PROMPT_TERMS",
    "MAX_PREVIOUS_PLAN_CHARS",
    "PRIMITIVE_TERMS",
    "PROMPT_TEMPLATE_VERSION",
    "URDF_HEADING",
    "build_system_prompt",
    "build_user_text",
    "depth_scale_text",
    "lint_prompt_text",
    "template_sha256",
    "text_sha256",
]
