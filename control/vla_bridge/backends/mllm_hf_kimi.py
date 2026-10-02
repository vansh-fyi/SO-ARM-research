"""`--backend mllm-hf-kimi`: Kimi-K3 on the HF router, pinned to Baseten (Phase 15, MLLM-01).

What this is
============
A thin registration: the shared OpenAI-compatible chat adapter
(`base_mllm_chat.make_chat_adapter`) plus `backend_configs/mllm_hf_kimi.json`.
Everything else is the shared builder and base (PIPE-02, D-00a). Same shape
as `mllm_hf_qwen.py` and `mllm_hf_deepseek.py`; Phase 16 chooses which run
(D-23).

Router facts (router.huggingface.co/v1/models, verified 2026-10-02)
-------------------------------------------------------------------
`moonshotai/Kimi-K3` (Kimi K3 License, not OSI) is natively multimodal;
thinking is ALWAYS on and returned as `reasoning_content`;
`reasoning_effort` low / high / max (default max); single-step sampling
temperature 1.0, top_p 0.95. Providers: together ($2.70/$13.50, NO
structured output), fireworks ($3/$15, structured output), baseten ($3/$15,
structured output, ~74 tok/s, the fastest with structured output),
deepinfra ($2.85/$14.25, structured output, ~10 tok/s). Pinned to
`baseten`.

Cost (research C1): Kimi-K3 is NOT cheap (Opus-class $3 in / $15 out per
million tokens, roughly $0.35 per call at ~20K output tokens). Phase 16
should run the Qwen and DeepSeek variants first.
"""

from vla_bridge.backends import base_mllm_build, base_mllm_chat
from vla_bridge.backends.base_mllm import register_mllm_backend

BACKEND_NAME = "mllm-hf-kimi"
CONFIG_NAME = "mllm_hf_kimi"


@register_mllm_backend(BACKEND_NAME, config_name=CONFIG_NAME,
                       adapter_factory=base_mllm_chat.make_chat_adapter)
def build(args, camera_names, joint_limits_deg):
    return base_mllm_build.build_mllm_backend(BACKEND_NAME, args, camera_names, joint_limits_deg)


__all__ = ["BACKEND_NAME", "CONFIG_NAME", "build"]
