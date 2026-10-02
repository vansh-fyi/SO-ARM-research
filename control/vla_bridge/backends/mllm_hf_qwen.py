"""`--backend mllm-hf-qwen`: Qwen3.8-27B on the HF router, pinned to Cerebras (Phase 15, MLLM-01).

What this is
============
A thin registration: the shared OpenAI-compatible chat adapter
(`base_mllm_chat.make_chat_adapter`) plus `backend_configs/mllm_hf_qwen.json`.
Everything else (prompt, schema, chunk drain, holds, recorder, hardware) is
the shared builder and base (PIPE-02, D-00a). Sibling variants
`mllm_hf_deepseek.py` and `mllm_hf_kimi.py` have the same shape; Phase 16
chooses which run (D-23).

Router facts (router.huggingface.co/v1/models, verified 2026-10-02)
-------------------------------------------------------------------
`Qwen/Qwen3.8-27B` (Apache-2.0) takes image+text input and thinks by default;
`reasoning_effort` xhigh (default) / medium / low; model-card thinking
sampling temperature 1.0, top_p 0.95. Providers: novita ($0.42/$3, NO
structured output), cerebras ($0.99/$1.49, structured output, ~831 tok/s,
65,536-token context), ovhcloud, deepinfra ($0.20/$2.50, structured output,
262K context). Pinned to `cerebras` for speed; the 65,536 context forces
max_tokens 32000 (prompt reserve 16000). Unpinned `:fastest` routing could
land on a provider without structured output (research C2, Pitfall 2).
"""

from vla_bridge.backends import base_mllm_build, base_mllm_chat
from vla_bridge.backends.base_mllm import register_mllm_backend

BACKEND_NAME = "mllm-hf-qwen"
CONFIG_NAME = "mllm_hf_qwen"


@register_mllm_backend(BACKEND_NAME, config_name=CONFIG_NAME,
                       adapter_factory=base_mllm_chat.make_chat_adapter)
def build(args, camera_names, joint_limits_deg):
    return base_mllm_build.build_mllm_backend(BACKEND_NAME, args, camera_names, joint_limits_deg)


__all__ = ["BACKEND_NAME", "CONFIG_NAME", "build"]
