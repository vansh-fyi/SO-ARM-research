"""`--backend gemini-robotics-2`: "Gemini Robotics 2" (VLAB-02) via Gemini Robotics-ER 2 + local conversion.

What this is, and what it is NOT
================================
This is NOT the native Gemini Robotics 2 VLA policy. Google DeepMind's
Gemini Robotics 2 ("Our most advanced vision-language-action model") and
Gemini Robotics On-Device 2 turn images and language directly into motor
control, but they are open only to trusted testers through a waitlist for
early access (deepmind.google/models/gemini-robotics). No public API model id
exists for either. Per 14-CONTEXT.md D-02's 2026-10-01 supersession, this
backend instead calls the PUBLIC embodied-reasoning model released alongside
the Gemini Robotics 2 suite, **`gemini-robotics-er-2-preview`**. That model
returns image points and boxes as JSON text. Plan 14-13's `er_conversion`
layer then turns one target point into one small, capped joint step (a
bounded visual-servo nudge, at most 5 mm per step, no depth reach, no grasp
decision).

So every action from this backend is tagged
`gemini-robotics-2@gemini-er-conversion`, and every `episode.jsonl` record
carries `action_path` = `base_gemini.ACTION_PATH`. It must never be read as
equivalent to the other backends' trained action-output policies.

Model id: verified, not guessed
-------------------------------
Plan 14-14 verified the id on 2026-10-01; Plan 14-15 re-fetched the same
pages with curl on 2026-10-02 and found no change:
  - ai.google.dev/gemini-api/docs/robotics-overview (Last updated
    2026-09-23): the ER 2 model has two endpoints,
    `gemini-robotics-er-2-preview` (standard, request/response) and
    `gemini-robotics-er-2-streaming-preview` ("Optimized for real-time
    streaming via the Live API").
  - ai.google.dev/gemini-api/docs/deprecations (Last updated 2026-10-01),
    "Robotics models": `gemini-robotics-er-1.5-preview` shut down
    2026-04-30, and `gemini-robotics-er-1.6-preview` shut down 2026-08-31.
    Both name `gemini-robotics-er-2-preview` as the replacement.
  - deepmind.google/models/gemini-robotics: Gemini Robotics 2 and On-Device 2
    are offered to "100+ trusted testers" via "Join the waitlist for early
    access"; no public model id.
  - So exactly ONE public, request/response robotics model exists:
    `gemini-robotics-er-2-preview`. The streaming variant is Live-API only,
    and `base_gemini.load_er_backend_config()` refuses it.

VLAB-02 (this module) is equivalent to VLAB-01 (`gemini_robotics`) by design
---------------------------------------------------------------------------
This module follows the decision recorded in Plan 14-14
(`gemini_robotics.py`'s docstring and 14-14-SUMMARY.md, "VLAB-01 vs VLAB-02
distinction"). Since only one public model exists, the two backends
legitimately call the SAME model (`gemini-robotics-er-2-preview`, read from
config) with the SAME prompt, the SAME thinking level ("medium") and the
SAME `er_conversion` step. Today they behave identically; nothing is varied
artificially to make them look like different generations, which would also
confound Phase 16's cross-backend comparison. They differ only in:

  | Aspect               | VLAB-01                                | VLAB-02 (this module)                     |
  |----------------------|----------------------------------------|-------------------------------------------|
  | `--backend` name     | `gemini-robotics`                      | `gemini-robotics-2`                       |
  | `checkpoint_label`   | `gemini-robotics`                      | `gemini-robotics-2`                       |
  | `model_version`      | `gemini-robotics@gemini-er-conversion` | `gemini-robotics-2@gemini-er-conversion`  |
  | Config file          | `backend_configs/gemini_robotics.json` | `backend_configs/gemini_robotics_2.json`  |
  | `requirement`        | `VLAB-01`                              | `VLAB-02`                                 |
  | Live verification    | its own episode                        | its own, separate episode (D-05)          |

`last_details["er_model"]` records the model id actually called, so
`episode.jsonl` shows that both backends used the same model.

What would change later:
  * A newer public ER generation: this backend's `model` can move on its own
    with a one-line JSON edit and no code change (it is the more natural one
    to move first, being the "Robotics 2" branded backend).
  * Native Gemini Robotics 2 API access granted to this project: VLAB-02
    would target that model, which needs a NEW action source that decodes
    native actions, not `er_conversion`. The ER-only guard in
    `base_gemini.load_er_backend_config()` (`ER_MODEL_PREFIX`) deliberately
    refuses a non-ER id, so a native id cannot silently go through the
    point-servo conversion.

Config (`backend_configs/gemini_robotics_2.json`): `model` and `api_key_env`
are required. `thinking_level` and `timeout_s` are optional and are passed
to `GeminiActionSourceBase`. The API key is read only from the environment
variable named by `api_key_env`. The docs say it must be a *restricted* key,
because unrestricted keys get 403. It is never stored, hard-coded or logged
(T-14-11).

Transport: direct HTTPS to Google from the control-loop machine (D-02). No
`--server-address`/`--checkpoint` is needed or used. The google-genai SDK is
imported lazily inside `base_gemini`, so backend discovery stays light.
"""

from vla_bridge.backends import BackendHandles, base_gemini, er_conversion, register

BACKEND_NAME = "gemini-robotics-2"
CONFIG_NAME = "gemini_robotics_2"
CHECKPOINT_LABEL = "gemini-robotics-2"


@register(BACKEND_NAME)
def build(args, camera_names: dict[int, str], joint_limits_deg: dict) -> BackendHandles:
    """Build the VLAB-02 backend from `backend_configs/gemini_robotics_2.json`.

    Raises `SystemExit` before touching any hardware if the config is
    missing or invalid, the API-key variable is unset, or there is no wrist
    camera. `convert_fn` is `er_conversion.convert_er_output_to_action`
    bound to the calibrated `joint_limits_deg` (`make_er_convert_fn`), so
    the IK clamps to the same limits the safety validator uses.
    """
    model, api_key_env, source_kwargs = base_gemini.load_er_backend_config(CONFIG_NAME, BACKEND_NAME)
    return base_gemini.build_gemini_backend(
        model,
        api_key_env,
        CHECKPOINT_LABEL,
        args,
        camera_names,
        convert_fn=er_conversion.make_er_convert_fn(joint_limits_deg),
        **source_kwargs,
    )


__all__ = ["BACKEND_NAME", "CHECKPOINT_LABEL", "CONFIG_NAME", "build"]
