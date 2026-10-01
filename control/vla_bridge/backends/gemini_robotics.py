"""`--backend gemini-robotics`: "Gemini Robotics" (VLAB-01) via Gemini Robotics-ER 2 + local conversion.

What this is, and what it is NOT
================================
This is NOT the native Gemini Robotics VLA policy. Google DeepMind's
vision-language-action models (Gemini Robotics 2, and Gemini Robotics
On-Device 2) turn images and language directly into motor control, but they
are open only to trusted testers through a waitlist for early access
(deepmind.google/models/gemini-robotics, fetched 2026-10-01). No public API
model id exists for them. Per 14-CONTEXT.md D-02's 2026-10-01 supersession,
this backend instead calls the PUBLIC embodied-reasoning model
**`gemini-robotics-er-2-preview`**. That model returns image points and
boxes as JSON text. Plan 14-13's `er_conversion` layer then turns one target
point into one small, capped joint step (a bounded visual-servo nudge, at
most 5 mm per step, no depth reach, no grasp decision).

So every action from this backend is tagged
`gemini-robotics@gemini-er-conversion`, and every `episode.jsonl` record
carries `action_path` = `base_gemini.ACTION_PATH`. It must never be read as
equivalent to the other backends' trained action-output policies.

Model id: verified, not guessed (fetched 2026-10-01 with curl)
-------------------------------------------------------------
  - ai.google.dev/gemini-api/docs/robotics-overview (Last updated
    2026-09-23): the ER 2 model has two endpoints,
    `gemini-robotics-er-2-preview` (standard) and
    `gemini-robotics-er-2-streaming-preview` ("Optimized for real-time
    streaming via the Live API"). The Python sample calls
    `client.interactions.create(model="gemini-robotics-er-2-preview", ...)`.
  - ai.google.dev/gemini-api/docs/deprecations (Last updated 2026-10-01),
    "Robotics models": `gemini-robotics-er-1.5-preview` shut down
    2026-04-30, and `gemini-robotics-er-1.6-preview` shut down 2026-08-31.
    The recommended replacement for both is `gemini-robotics-er-2-preview`.
    The /models page (Last updated 2026-10-01) still lists ER 1.6, but the
    deprecations table is the authoritative shutdown record.
  - So exactly ONE public, request/response robotics model exists today:
    `gemini-robotics-er-2-preview`. The streaming variant is Live-API only,
    and `base_gemini.load_er_backend_config()` refuses it.

VLAB-01 (this module) vs VLAB-02 (`gemini_robotics_2`, Plan 14-15)
------------------------------------------------------------------
No first-generation Gemini Robotics model of any kind (VLA or ER) is
publicly callable any more, so the two backends cannot honestly target
different model generations. The decision recorded here, which Plan 14-15
follows:

  - SAME model (`gemini-robotics-er-2-preview`, read from config), SAME
    prompt, SAME thinking level ("medium") and SAME `er_conversion` step.
    Today the two backends behave identically. Nothing is varied
    artificially (no different thinking level or prompt) just to make them
    look like different generations.
  - They DIFFER only in the following:

    | Aspect                  | VLAB-01                                 | VLAB-02                                     |
    |-------------------------|-----------------------------------------|---------------------------------------------|
    | `--backend` name        | `gemini-robotics`                       | `gemini-robotics-2`                         |
    | `checkpoint_label`      | `gemini-robotics`                       | `gemini-robotics-2`                         |
    | `model_version`         | `gemini-robotics@gemini-er-conversion`  | `gemini-robotics-2@gemini-er-conversion`    |
    | Config file             | `backend_configs/gemini_robotics.json`  | `backend_configs/gemini_robotics_2.json`    |
    | Live verification       | its own episode                         | its own episode (D-05)                      |

    `last_details["er_model"]` records the model id actually called, so
    `episode.jsonl` shows that both backends used the same model.
  - What would change later is a config edit for a new ER model, but new
    code for a native model:
      * If a native Gemini Robotics VLA API becomes available to this
        project, VLAB-01 would target the first-generation native model and
        VLAB-02 Gemini Robotics 2. That needs a NEW action source that
        decodes native actions, not `er_conversion`. The ER-only guard in
        `base_gemini.load_er_backend_config()` (`ER_MODEL_PREFIX`)
        deliberately refuses a non-ER model id, so a native id cannot
        silently go through the point-servo conversion.
      * If a newer public ER generation appears, the `model` fields can move
        independently with a one-line JSON edit and no code change.

Config (`backend_configs/gemini_robotics.json`): `model` and `api_key_env`
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

BACKEND_NAME = "gemini-robotics"
CONFIG_NAME = "gemini_robotics"
CHECKPOINT_LABEL = "gemini-robotics"


@register(BACKEND_NAME)
def build(args, camera_names: dict[int, str], joint_limits_deg: dict) -> BackendHandles:
    """Build the VLAB-01 backend from `backend_configs/gemini_robotics.json`.

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
