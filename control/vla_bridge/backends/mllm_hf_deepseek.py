"""`--backend mllm-hf-deepseek`: DeepSeek-V4.1-Flash on the HF router, pinned to DeepInfra (Phase 15, MLLM-01).

What this is
============
A thin registration: the shared OpenAI-compatible chat adapter
(`base_mllm_chat.make_chat_adapter`) plus
`backend_configs/mllm_hf_deepseek.json`. Everything else is the shared
builder and base (PIPE-02, D-00a). Same shape as `mllm_hf_qwen.py` and
`mllm_hf_kimi.py`; Phase 16 chooses which run (D-23).

Router facts (router.huggingface.co/v1/models, verified 2026-10-02)
-------------------------------------------------------------------
`deepseek-ai/DeepSeek-V4.1-Flash` takes image+text input. Providers:
novita ($0.30/$1.20, NO structured output), fireworks-ai, featherless-ai,
baseten ($0.30/$1.20, structured output) and deepinfra ($0.20/$0.60,
structured output, 1,048,576-token context).

Decisions
---------
  - D-23 named Novita as the host, but Novita has no structured output and
    the default `:fastest` routing can land there (research C2, C12), so the
    config pins `deepinfra`.
  - User decision 2026-10-02 (15-LIVE-RESULTS.md section 2) supersedes the
    locked D-23 model id and D-27: DeepSeek retired
    `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` upstream (its aliases route to
    V4.1-Flash), and both live smoke attempts on it got HTTP 504 from the
    router, so the model is now `deepseek-ai/DeepSeek-V4.1-Flash`. The
    backend name and effort `max` are unchanged.
"""

from vla_bridge.backends import base_mllm_build, base_mllm_chat
from vla_bridge.backends.base_mllm import register_mllm_backend

BACKEND_NAME = "mllm-hf-deepseek"
CONFIG_NAME = "mllm_hf_deepseek"


@register_mllm_backend(BACKEND_NAME, config_name=CONFIG_NAME,
                       adapter_factory=base_mllm_chat.make_chat_adapter)
def build(args, camera_names, joint_limits_deg):
    return base_mllm_build.build_mllm_backend(BACKEND_NAME, args, camera_names, joint_limits_deg)


__all__ = ["BACKEND_NAME", "CONFIG_NAME", "build"]
