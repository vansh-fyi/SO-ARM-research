# Stack Research

**Domain:** Real-hardware MLLM robot control additions (provider-agnostic router, plan-then-execute loop, reasoning-trace logging, depth-extended episode recorder)
**Researched:** 2026-09-15
**Confidence:** MEDIUM-HIGH — HF Inference Providers pricing/model availability verified live (this space changes weekly); library version numbers verified via WebSearch/WebFetch, not training memory. Treat the specific pilot model name as a config value to revisit, not a fixed dependency.

This is a **delta** stack — additions on top of the already-working `control/` Python 3.12 venv (`lerobot[feetech]==0.6.1`, `opencv-python==5.0.0.93`, `pynput==1.8.2`, `ultralytics==8.4.138`; transitively `huggingface_hub==1.29.0`, `torch==2.11.0`, `numpy==2.2.6`, `pillow==12.3.0` — confirmed by inspecting the live venv, not assumed). Nothing below should require touching the pinned `lerobot`/`opencv`/`ultralytics` versions.

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| `huggingface_hub` (InferenceClient) | `1.29.0` — already installed, zero new dependency | Calls the free/cheap HF-hosted VLM pilot via HF's Inference Providers router | It's already a transitive dependency of `lerobot`, so this costs nothing to add. As of 2026 `InferenceClient.chat.completions.create()` is fully OpenAI-wire-compatible and supports `image_url` content blocks for vision models, plus built-in `provider="auto"` selection/fallback across HF's 17+ routed providers (Cerebras, Groq, Novita, Fireworks, Together, DeepInfra, etc.) — no separate account needed for the pilot. |
| Hand-rolled thin `MLLMProvider` interface (plain Python ABC/Protocol, no framework) | n/a | Provider-agnostic router across HF-hosted pilot → OpenAI/Anthropic/Gemini later | A ~50-line interface (`plan_subgoal(image, depth, joint_state, task_prompt) -> SubGoalPlan`) with one adapter class per provider is enough for 2-4 providers in a solo research repo. See "What NOT to Use" for why LiteLLM is skipped. |
| `pydantic` (transitive via `huggingface_hub`/future `openai` SDK, v2.x) | already present transitively | Typed schema for the parsed MLLM sub-goal/action output (`reach`/`grasp`/`lift`/`place`) | Already resolved in the venv — no new install. Gives you `model_validate_json()` for one clean parse-or-retry path instead of hand-rolled dict-key checking. Pin it explicitly in `control/requirements.txt` once you depend on it directly, rather than relying on transitive resolution. |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `openai` (python SDK) | latest 2.x (verify at install time — release cadence is fast; HF router itself is OpenAI-wire-compatible at `https://router.huggingface.co/v1`) | Calls OpenAI directly once you add it as a second provider | Only add when you actually wire in OpenAI (per PROJECT.md's Active item "at least one additional paid MLLM provider"). Don't install it just to reach the HF router — `huggingface_hub` already covers that with less footprint. |
| `anthropic` | `>=0.116` (verify latest — released Sept 10, 2026 at last check) | Calls Anthropic (Claude) as a provider | Add only when wiring Anthropic; same adapter-per-provider pattern. |
| `google-genai` | `2.23.0` (current as of Sept 2026) | Calls Gemini as a provider | Add only when wiring Gemini. |
| `opencv-python` | `5.0.0.93` — already pinned, no change | Save raw depth as lossless 16-bit single-channel PNG per frame (`cv2.imwrite(path, depth_uint16)`) | Depth recording extension to `record_episode.py`. OpenCV 5.x's `imwrite`/`imread` handle `uint16` PNG natively — no depth-specific codec dependency needed, and it matches the still-frame PNG path already used in `record_still()`. |
| `numpy` | `2.2.6` — already pinned, no change | Depth array manipulation before serialization; `.npz` fallback if sub-mm float precision is needed instead of integer-mm PNG | Use directly; already installed. |
| stdlib `json` + `pathlib` | n/a | Append-only JSONL reasoning-trace log, one line per MLLM call, per episode | Default logging mechanism — see "What NOT to Use" for why not Langfuse/Opik/MLflow. |
| stdlib `dataclasses` (or the `pydantic` model above) | n/a | Shared typed structure for a sub-goal plan passed from the MLLM adapter to the local step-executor | Keeps the plan-then-execute boundary explicit and testable without a framework. |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| None new | — | No test framework, linter, or CI is currently in this repo's conventions (per CLAUDE.md: "no dedicated linter or formatter config detected"). Don't introduce one just for this milestone unless the user asks. |

## Integration Points with Existing `control/`

- **MLLM router module**: add as a new `control/mllm/` package (or a single `control/mllm_client.py` if kept small) — mirrors the existing flat-script convention (`record_episode.py`, `joint_jog.py`) rather than introducing a nested app structure.
- **Plan-then-execute loop**: the loop calls `MLLMProvider.plan_subgoal(...)` once per checkpoint (reach/grasp/lift/place), then drives the *existing* `SO101Follower` object from `lerobot.robots.so_follower` exactly as `record_episode.py` already does (`robot.get_observation()` / `robot.send_action(...)`) — no new robot-control code path, just a new caller of the existing bridge.
- **Reasoning-trace logging**: one `reasoning_trace.jsonl` per episode directory, written alongside `joints.csv` and `camera_N.mp4` — same output directory (`--out`) convention `record_episode.py` already uses, one JSON object per MLLM call: `{ts, subgoal, provider, model, prompt_summary, raw_response, parsed_plan, latency_ms, token_usage}`.
- **Depth extension to `record_episode.py`**: add a `depth_N/` subdirectory of per-frame 16-bit PNGs (`frame_%06d.png`) written on the same timestamp loop that already writes `camera_N.mp4` frames and `joints.csv` rows — same `ts` column ties all three together, no new sync mechanism needed.
- **Structured output parsing**: prompt the MLLM to return JSON matching the `pydantic` sub-goal schema; on `ValidationError`, do exactly one repair retry (re-prompt with the parse error) before failing the checkpoint — log both attempts to the reasoning trace. Do not reach for `instructor`/`outlines`/`guidance` (see below).

## Installation

```bash
# Already present in control/.venv (no action needed) — confirmed via pip list:
#   huggingface_hub==1.29.0, torch==2.11.0, numpy==2.2.6, pillow==12.3.0, opencv-python==5.0.0.93

# Core addition (only actual new install for the HF-hosted pilot phase):
pip install pydantic   # explicit pin once router code depends on it directly

# Add only when wiring each additional provider (not upfront):
pip install openai              # OpenAI provider
pip install anthropic           # Anthropic provider
pip install google-genai        # Gemini provider

# Do NOT install for this milestone:
#   litellm, mlflow, langfuse, opik, instructor, outlines, guidance
```

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|--------------------------|
| Hand-rolled thin `MLLMProvider` interface | `litellm` | If the project later needs 10+ providers, load-balancing/automatic-fallback, or team-wide centralized cost tracking. For 2-4 providers in a solo research repo, LiteLLM's dependency footprint (its own pinned `pydantic`/`httpx` versions, dozens of transitive provider SDKs) risks conflicting with the tightly-pinned `control/` venv, and its abstraction layer makes it harder to see the exact HTTP request/response when debugging a real-hardware failure. |
| `huggingface_hub.InferenceClient` for the HF pilot | `openai` SDK pointed at `base_url="https://router.huggingface.co/v1"` | Valid and documented by HF itself — use it if you want one SDK class for every OpenAI-wire-compatible provider (HF router + OpenAI). But `huggingface_hub` is already installed with zero marginal dependency cost, and its native `provider="auto"`/`:cheapest`/`:fastest` policy selection isn't in the raw OpenAI SDK. |
| JSONL for reasoning-trace logging | Langfuse (self-hosted) | If the team grows beyond one researcher and needs a shared web UI, dataset-based evals, or multi-user trace review. Langfuse's own docs note the full-stack self-hosted deployment (the only officially supported path as of mid-2026) starts at ~8GB and multiple containers — unjustified for a single researcher running benchmark episodes next to the robot. |
| JSONL for reasoning-trace logging | Opik (self-hosted) | Similar reasoning to Langfuse; Opik's self-hosted footprint starts around 16GB and its open-source tier ships without user management — more platform than a solo benchmark needs. |
| 16-bit PNG per depth frame (`cv2.imwrite`) | HDF5 (`h5py`) per episode | If episode counts grow into the hundreds and you want one file per episode with random access across all modalities — this is the same pattern Phase 4's sim dataset already uses (HDF5, schema-verified via replay round-trip), so it's a reasonable *later* upgrade. Not justified for early Pen Transfer validation runs where per-episode file count is small and PNG needs zero new dependency. |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|--------------|
| `litellm` | Adds a large, fast-moving dependency surface (own pinned `httpx`/`pydantic`, 100+ provider SDKs pulled in even if unused) into a `control/` venv that's already carefully pinned (`lerobot[feetech]==0.6.1`); its abstraction also obscures the exact provider request during real-hardware debugging, where "what exactly did we send/receive" matters most. | Hand-rolled `MLLMProvider` interface + one thin adapter class per provider. |
| `mlflow` / `langfuse` / `opik` for reasoning-trace logging | All three assume a running server (Docker Compose, 8-16GB) designed for team-scale experiment tracking or agent observability — pure overhead for a single-researcher local benchmark. | Plain JSONL file per episode (`reasoning_trace.jsonl`), one line per MLLM call — greppable, diffable, loadable into `pandas` for analysis, and matches the existing `joints.csv` plain-file convention already in `record_episode.py`. |
| Per-tick MLLM calls (calling the MLLM on every control-loop tick) | Real hosted-API latency (seconds, sometimes 5-15s for a vision-heavy prompt) makes tick-level control impractical and burns through the free-tier credit fast. | Sub-goal-level calls only (reach→grasp→lift→place), per PROJECT.md's already-committed plan-then-execute decision — a fast *local* controller (no MLLM call) executes each checkpoint's low-level joint motion via the existing LeRobot bridge. |
| `instructor` / `outlines` / `guidance` for structured MLLM output | These add real complexity (constrained decoding or extra wrapper layers) that only pays off at high call volume or when providers lack any native JSON mode. This pilot involves dozens of calls per benchmark run. | Prompt for JSON directly, `json.loads()` + `pydantic` validation, with exactly one repair-retry on failure — log both attempts to the reasoning trace so failures are visible, not hidden inside a framework's retry logic. |
| New microcontroller/embedded firmware (ESP32, etc.) | Already ruled out in PROJECT.md — the existing LeRobot USB-serial bridge to the Feetech servos already provides full joint-level control (UAT signed off). | Keep using `SO101Follower`/`SOFollowerRobotConfig` exactly as `record_episode.py` and `joint_jog.py` already do. |
| Relying on the HF **free** tier ($0.10/month credit) for the actual multi-episode benchmark | As of the Aug 2026 HF billing change, free-tier credit is genuinely tiny — enough for smoke-testing the router, not for running the 4-task benchmark suite end-to-end. | Budget for HF **PRO** (`$9/mo`, `$2/mo` compute credit) as the practical "still basically free, definitely not a frontier-API bill" tier once past initial smoke tests — this preserves the "free/cheap HF-hosted pilot before paid frontier providers" intent without stalling on credit exhaustion mid-benchmark. |

## Stack Patterns by Variant

**If the HF free/PRO credit runs out mid-benchmark:**
- Fall back to HF PRO ($9/mo) before reaching for a paid frontier provider (OpenAI/Anthropic/Gemini) — keeps the "free-to-use HF pilot first" constraint intact.
- Because: the router/adapter interface makes this a config change (model id + provider), not a code change, if built as recommended above.

**If the specific pilot model gets deprecated or re-routed (this space changes weekly):**
- Keep the model id as one named config constant, e.g. `MLLM_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct:novita"` (confirmed live via HF's model page + Inference Providers widget as of Sept 2026), not hardcoded through router logic.
- Because: HF's routed-provider list and per-model hosting changes independently of your code; isolating it to one constant makes swapping trivial.

**If depth precision needs sub-millimeter float values instead of integer millimeters:**
- Use `.npz` (compressed `numpy` float32 arrays) instead of 16-bit PNG for the depth stream.
- Because: 16-bit PNG tops out at integer values 0-65535 (fine for mm-resolution depth up to ~65m); float precision needs a numeric array format instead.

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|------------------|-------|
| `huggingface_hub==1.29.0` | `lerobot[feetech]==0.6.1`, Python 3.12.12 | Already resolved and installed in the live `control/.venv` — confirmed via `pip list`, not assumed. `InferenceClient.chat.completions.create()` with `image_url` content blocks works out of the box, no upgrade needed. |
| `opencv-python==5.0.0.93` | `cv2.imwrite`/`cv2.imread` for `uint16` single-channel PNG | OpenCV 5.x retains native 16-bit PNG support; no separate codec/plugin dependency required for lossless depth storage. |
| `pydantic` v2.x (transitive) | Python 3.12, `huggingface_hub`, future `openai`/`anthropic`/`google-genai` SDKs | All target SDKs use pydantic v2 as of 2026; no cross-version pin conflicts expected, but pin explicitly once the router code imports it directly rather than relying on transitive resolution. |
| `anthropic>=0.116` / `google-genai==2.23.0` / `openai` (2.x/3.x, verify at install) | Python 3.12 | All three current SDK lines support 3.12; each is independent (no shared pinned transitive deps with `lerobot`/`opencv`/`ultralytics`) — install additively, one per provider, only when wiring that provider. |

## Sources

- [Hugging Face — Inference Providers Pricing and Billing](https://huggingface.co/docs/inference-providers/pricing) — HIGH confidence, official docs, fetched live: confirms $0.10/mo free credit, $2.00/mo PRO credit, HF-routed vs custom-provider-key billing model (Aug 2026 credits system).
- [Hugging Face — Chat Completion task docs](https://huggingface.co/docs/inference-providers/tasks/chat-completion) — HIGH confidence, official docs: vision/VLM support via `image_url` content blocks, `provider="auto"`/`:cheapest`/`:fastest` policies.
- [Hugging Face — Run Inference on servers (huggingface_hub guide)](https://huggingface.co/docs/huggingface_hub/en/guides/inference) — HIGH confidence, official docs: `InferenceClient` OpenAI-wire-compatibility, provider list (Cerebras, Groq, Novita, Fireworks, Together, DeepInfra, etc. as of Sept 11, 2026).
- [Qwen/Qwen3-VL-30B-A3B-Instruct model page](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct) — MEDIUM confidence (live model-card fetch, Sept 2026): confirms Novita hosts this model via Inference Providers; MoE 30B/3B-active architecture, vision-capable, 256K context.
- [LiteLLM — Hugging Face provider docs](https://docs.litellm.ai/docs/providers/huggingface) — MEDIUM confidence, official docs: confirms `huggingface/<provider>/<org>/<model>` format and `image_url` support, used to inform the "alternative considered" entry.
- [LiteLLM PyPI / release notes](https://docs.litellm.ai/release_notes/) — MEDIUM confidence: version churn cadence (`1.100.0` as of late Aug 2026) informing the dependency-risk argument.
- [Opik vs Langfuse: Self-Hosted LLM Observability in 2026](https://blog.elest.io/opik-vs-langfuse-self-hosted-llm-observability-in-2026/) — MEDIUM confidence, third-party blog: resource footprint (Langfuse ~8GB, Opik ~16GB) used to justify JSONL-over-platform recommendation.
- Live `pip list` in `control/.venv` (this repo) — HIGH confidence, ground truth: confirmed `huggingface_hub==1.29.0`, `torch==2.11.0`, `numpy==2.2.6`, `pillow==12.3.0`, `opencv-python==5.0.0.93` already installed transitively via `lerobot[feetech]==0.6.1`.

---
*Stack research for: SoARM VLA Research v2.0 milestone — MLLM router, plan-then-execute loop, reasoning-trace logging, depth-extended recorder*
*Researched: 2026-09-15*
