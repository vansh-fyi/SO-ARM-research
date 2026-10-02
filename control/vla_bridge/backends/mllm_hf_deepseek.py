"""`--backend mllm-hf-deepseek`: DeepSeek-V4-Flash-Vision-Exp on the HF router, pinned to DeepInfra (Phase 15, MLLM-01).

What this is
============
A thin registration: the shared OpenAI-compatible chat adapter
(`base_mllm_chat.make_chat_adapter`) plus
`backend_configs/mllm_hf_deepseek.json`. Everything else is the shared
builder and base (PIPE-02, D-00a). Same shape as `mllm_hf_qwen.py` and
`mllm_hf_kimi.py`; Phase 16 chooses which run (D-23).

Router facts (router.huggingface.co/v1/models, verified 2026-10-02)
-------------------------------------------------------------------
`deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` (MIT, marked experimental; the
card calls it the first experimental multimodal model in the V4 family)
takes image+text input and reasons with effort levels (card benchmarks use
`max`, temperature 1.0, top_p 0.95). Providers: novita ($0.44/$1.32, NO
structured output) and deepinfra ($0.44/$1.32, structured output, ~89
tok/s).

Decisions
---------
  - D-23 named Novita as the host, but Novita has no structured output and
    the default `:fastest` routing lands there (research C2, C12), so the
    config pins `deepinfra`.
  - D-27 keeps this Exp model even though the newer sibling
    `deepseek-ai/DeepSeek-V4.1-Flash` is on the router (research C12).
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
