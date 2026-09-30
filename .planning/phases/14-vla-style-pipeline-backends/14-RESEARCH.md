# Phase 14: VLA-Style Pipeline & Backends - Research

**Researched:** 2026-09-30
**Domain:** Real-hardware VLA/imitation-learning inference bridges (LeRobot async gRPC bridge, custom model-server adapters, cloud VLA APIs) driving a real SO-ARM101 via a pluggable `ActionSource` seam
**Confidence:** MEDIUM — the `ActionSource`/safety/logging plumbing to reuse is HIGH confidence (read directly from this repo's installed source); per-backend checkpoint/action-space facts are MEDIUM (live WebSearch/WebFetch this session, cross-checked against official HF/GitHub pages where possible); Gemini Robotics VLA API availability is a HIGH-confidence BLOCKING finding that materially affects D-02's premise (see Open Questions)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-00:** Phase 14's backend set was expanded from the original 4 (Gemini Robotics, π0/π0.5, Wall-X, ACT) to 12 during discussion, per explicit user request covering the project's full candidate VLA/imitation-learning model list. `ROADMAP.md` and `REQUIREMENTS.md` were updated in place before planning starts.
- **D-01:** Two transport lanes. Colab-hosted open-weight checkpoints (SmolVLA + all 10 new open-weight backends: π0/π0.5, Wall-X, ACT, OpenVLA, OpenVLA-OFT, Octo, RDT-1B, GR00T N1.7, GR00T N1.5, Diffusion Policy) reuse the existing async_inference gRPC bridge pattern — one LeRobot PolicyServer per checkpoint, swappable via config, same shape as today's `BridgeActionSource`.
- **D-02:** Gemini Robotics and Gemini Robotics 2 (cloud APIs, not Colab checkpoints) get a separate, direct-HTTP `ActionSource` implementation calling Google's API straight from the control-loop machine — no Colab hop, no pointless relay.
- **D-03:** Depth stays log-only for all 12 new backends, identical to current SmolVLA behavior — never added to any backend's observation dict, regardless of transport lane.
- **D-04:** Refactor `BridgeActionSource` into a shared base class handling the gRPC/observation/safety-validation plumbing common to all 10 Colab-hosted checkpoints, with a thin per-backend subclass (or config) only for checkpoint-specific quirks (action format, prompt template, JOINT_ORDER mapping if it differs). A separate `GeminiActionSource`-family base class handles the shared HTTP/auth plumbing for the 2 cloud-API backends. Do not write 12 fully independent ActionSource classes from scratch, and do not collapse everything into one fully generic config-only class.
- **D-05:** Familiarity/risk-ascending integration order. Start with OpenVLA-OFT (prior project familiarity from v1 sim work, Phases 1/6) to prove out the refactored Colab-bridge base class, then the remaining Colab-hosted checkpoints, then the 2 cloud-API backends (Gemini Robotics, Gemini Robotics 2) last. Each backend remains independently live-hardware-verifiable — do not batch backends behind one end-to-end test.
- **D-06:** `run_vla_episode.py` gets a new `--backend` flag. Backend-specific settings (checkpoint path/HF repo ID for Colab-hosted backends, API key/endpoint URL for the 2 cloud backends) live in a small per-backend config, keeping the existing `--checkpoint` arg as the Colab-lane-specific value it already is rather than overloading it to also select backend type.

### Claude's Discretion

- Exact config file format/location for per-backend settings (D-06) — JSON file vs. `.env` entries vs. a small Python registry module — left to planning/implementation as long as it's one small config per backend, not hardcoded inline.
- Exact class/module naming for the refactored base classes (D-04) — left to planning.

### Deferred Ideas (OUT OF SCOPE)

None — the expanded backend list was folded directly into Phase 14's scope (D-00) rather than deferred. No items came up during discussion that were pushed to a future phase.

</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| PIPE-01 | A consistent VLA-style `ActionSource` pattern exists that SmolVLA fits unmodified and new backends plug into without control-loop/safety/logging changes | See Architecture Patterns — `ActionSource` Protocol is already backend-agnostic (structural typing); D-04's base-class split (below) preserves this |
| VLAB-01 | Gemini Robotics integrated as VLA-style `ActionSource` driving real SO-ARM101 | See Per-Backend Findings + Open Questions — the full Gemini Robotics VLA is NOT publicly API-accessible today (early-access-partner gated); Gemini Robotics-ER (public) returns spatial reasoning, not joint actions |
| VLAB-02 | Gemini Robotics 2 integrated as VLA-style `ActionSource` | Same blocking finding as VLAB-01 — Gemini Robotics 2 (VLA) is early-access-partner gated; Gemini Robotics ER 2 is the public variant |
| VLAB-03 | π0/π0.5 integrated | See Per-Backend Findings — Tier A, native `lerobot.async_inference` `policy_type` support confirmed |
| VLAB-04 | Wall-X integrated | See Per-Backend Findings — Tier B, in LeRobot's policy registry but async-server dispatch support unconfirmed |
| VLAB-05 | ACT integrated | Tier A, native `lerobot.async_inference` support confirmed |
| VLAB-06 | OpenVLA integrated | Tier C, not LeRobot-native, own HF repo + 7-DoF delta-EEF action space needs conversion |
| VLAB-07 | OpenVLA-OFT integrated | Tier C, prior project familiarity (v1 Phases 1/6), delta-EEF action space |
| VLAB-08 | Octo integrated | Tier C, not LeRobot-native, configurable action space (typically delta-EEF for OXE-pretrained checkpoints) |
| VLAB-09 | RDT-1B integrated | Tier C, not LeRobot-native, unified action space with a joint-position slot usable directly for SO-101's 6 DOF; high GPU memory footprint |
| VLAB-10 | GR00T N1.7 integrated | Tier B, in LeRobot's policy registry, relative end-effector action space; known unit-frame pitfall (see Common Pitfalls) |
| VLAB-11 | GR00T N1.5 integrated | Tier C, predates LeRobot merge, own Isaac-GR00T package, absolute action targets |
| VLAB-12 | Diffusion Policy integrated | Tier A, native `lerobot.async_inference` support confirmed |

</phase_requirements>

## Summary

This phase extends an already-proven seam: `run_vla_episode.py`'s `ActionSource` Protocol (structural typing, `get_action(joint_state, instruction) -> (action, model_version)`) and the existing `BridgeActionSource`/`safety_validator.py`/`io_logger.py`/`action_contract.py` plumbing (all read directly from installed source this session) require **zero changes** to accept new backends — that part of PIPE-01 is already true today and this phase's job is to keep it true while onboarding 12 more.

The real complexity is NOT the `ActionSource` seam — it's that the 12 backends split into three genuinely different **integration tiers**, which the locked decisions (D-01, D-04) describe more uniformly than the live ecosystem actually is:

- **Tier A — LeRobot async-native (drop-in via existing bridge):** π0/π0.5, ACT, Diffusion Policy. Confirmed via a live GitHub error message that `lerobot.async_inference.PolicyServer`'s `policy_type` dispatch supports exactly `{act, smolvla, diffusion, tdmpc, vqbet, pi0, pi05}` — these 3 (plus existing SmolVLA) need only a `policy_type`/checkpoint swap in `connect_bridge()`, no new Colab-side code.
- **Tier B — LeRobot-registered but async-server support unconfirmed:** GR00T N1.7, Wall-X (WALL-OSS). Both appear in LeRobot's newer VLA policy registry (`lerobot.policies` — factory functions like `make_groot_pre_post_processors` exist) but neither appears in the async `PolicyServer`'s confirmed `policy_type` list above. This needs a live-source read of the installed `lerobot` version's `async_inference/policy_server.py` before Task 1 of each backend's plan — if dispatch isn't wired, the plan needs a small, scoped extension to `policy_server.py`'s dispatch table (still "reusing the existing async_inference gRPC bridge pattern" per D-01, just completing it) rather than a wholly custom server.
- **Tier C — not LeRobot-integrated, custom adapter required:** OpenVLA, OpenVLA-OFT, Octo, RDT-1B, GR00T N1.5. None of these ship a `lerobot.async_inference`-compatible `PolicyServer`. Each has its own HF repo + custom loading code (`AutoModel.from_pretrained` for OpenVLA/OpenVLA-OFT; a custom inference class for Octo/RDT-1B; NVIDIA's own `Isaac-GR00T` package for N1.5). These need a small resident Colab-side HTTP adapter service — this project already built and proved exactly this pattern for depth (`fastfs_server.py`: load-once, warm, serve via Flask, subprocess-isolated from the notebook kernel) — rather than trying to force them through `lerobot.async_inference.PolicyServer`.

**Primary recommendation:** Implement D-04's "shared base class for the 10 Colab-hosted checkpoints" as one abstract base (owning safety-validation/logging/watchdog plumbing, matching `BridgeActionSource` today) with **two transport mixins underneath it**: a `GrpcBridgeMixin` (wraps `connect_bridge()`/`RobotClient`, used by Tier A and, pending verification, Tier B) and an `HttpAdapterMixin` (wraps a resident Flask service per-backend, modeled directly on `fastfs_server.py`, used by Tier C). This keeps D-04's spirit (one shared base, thin per-backend subclasses) while being honest about the real transport split the ecosystem forces. Gemini Robotics/Gemini Robotics 2 (D-02) need a second, unrelated base class — but the full Gemini Robotics **VLA** (native action output) is not publicly API-accessible today; only Gemini Robotics-ER (embodied reasoning: points/boxes/trajectories, not joint actions) is public. This is a blocking finding for VLAB-01/VLAB-02 planning — see Open Questions.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Camera/joint-state observation capture | Local control machine (`run_vla_episode.py`) | — | Already implemented; unchanged by this phase (PIPE-01 requires no control-loop edits) |
| Policy inference (Tier A: π0/π0.5, ACT, Diffusion Policy) | Colab GPU (LeRobot `PolicyServer` subprocess) | — | Reuses existing async_inference gRPC bridge unmodified except `policy_type`/checkpoint |
| Policy inference (Tier B: GR00T N1.7, Wall-X) | Colab GPU (LeRobot `PolicyServer`, pending dispatch verification) | Colab GPU custom adapter (fallback if dispatch unsupported) | `lerobot.policies` registry has these but async-server wiring needs live-source confirmation |
| Policy inference (Tier C: OpenVLA, OpenVLA-OFT, Octo, RDT-1B, GR00T N1.5) | Colab GPU custom resident HTTP adapter (`fastfs_server.py`-pattern) | — | No `lerobot.async_inference` integration exists upstream; each ships its own loading/inference code |
| Policy inference (Gemini Robotics / Gemini Robotics 2) | Google Cloud API (direct HTTP from control machine) | — | D-02; no Colab hop. BLOCKED pending confirmation of public VLA action-output API access (see Open Questions) |
| Action validation | Local control machine (`safety_validator.py`) | — | Unmodified per all 12 backends (VLAB requirement) |
| I/O logging | Local control machine (`io_logger.py`) | — | Unmodified per all 12 backends |
| EEF-to-joint conversion (for delta-EEF-output backends: OpenVLA, OpenVLA-OFT, Octo, RDT-1B if configured EEF, GR00T N1.7's new relative-EE mode) | Per-backend `ActionSource` subclass (local control machine, before `validate_action()`) | — | `action_contract.py`'s `JOINT_ORDER`/units are joint-space only; a Cartesian-output backend must convert before the safety gate, not after |
| Backend selection/config | `run_vla_episode.py` CLI (`--backend`) + per-backend config file | — | D-06 |

## Standard Stack

### Core (already installed/proven in this repo — reused unchanged)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `lerobot[async]` | `0.6.1` (pinned, installed both locally and in the Colab notebook) | `RobotClient`/`PolicyServer` gRPC bridge, `SO101Follower` | Already the proven transport for SmolVLA (Phase 11/12); `[VERIFIED: installed lerobot==0.6.1 source, read directly this session]` |
| `grpcio` | transitive via `lerobot[async]` | gRPC transport for the Colab bridge | Already audited in `11-RESEARCH.md`'s Package Legitimacy Audit |
| `pyngrok` | `8.1.2` | TCP tunnel from Colab to local control machine | Already audited/human-verified in Phase 11 |
| `flask` | already installed (Colab `fastfs_server.py`) | Resident HTTP adapter pattern for Tier C backends | Proven pattern this session — reuse, don't reinvent |

### Supporting (new for this phase)

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `google-genai` | `2.25.0` (verified via `pip index versions`, 2026-09-30) | Official Google Gen AI Python SDK — REST/SDK client for Gemini Robotics-ER (the currently-public variant) | D-02's cloud-API `ActionSource` for Gemini backends, IF the planner proceeds with the ER-based workaround (see Open Questions) |
| `transformers` | already pinned `4.21.1` in `explorations/`; a newer `transformers` install is required Colab-side for OpenVLA/OpenVLA-OFT (`AutoModel.from_pretrained`) — do not conflate with the pinned exploration-stack version | OpenVLA/OpenVLA-OFT model loading | Tier C backends only, isolated to the Colab notebook's own environment (never touches `explorations/`'s pinned stack) |
| `Isaac-GR00T` (GitHub source install, `NVIDIA/Isaac-GR00T`) | tracks `main` (no stable pip release found) | GR00T N1.5 model loading/inference | Tier C — GR00T N1.5 specifically; N1.7 may be reachable via `lerobot.policies` instead (Tier B, pending verification) |
| `RoboticsDiffusionTransformer` (GitHub source install, `thu-ml/RoboticsDiffusionTransformer`) | tracks `main` | RDT-1B model loading/inference | Tier C |
| `wall-x` (GitHub source install, `X-Square-Robot/wall-x`) | tracks `main` | Wall-OSS/Wall-X model loading/inference, IF not reachable via `lerobot.policies`'s WALL-OSS integration | Tier B fallback / Tier C |
| `octo-models` / `octo` (GitHub source install, `octo-models/octo`; **not** a PyPI package — confirmed via `pip index versions octo-models` returning no match) | tracks `main` | Octo model loading/inference | Tier C |
| `openvla` (GitHub source install, `openvla/openvla`; not PyPI-published) | tracks `main` | OpenVLA / OpenVLA-OFT model loading/inference | Tier C |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Custom Flask adapter per Tier C backend (recommended) | Extending `lerobot.async_inference.policy_server.py`'s dispatch to call each Tier C model's own loading code | More "native" to D-01's stated pattern, but requires patching a vendored library's internals for 5 backends with genuinely different loading APIs — higher maintenance burden than 5 small, isolated Flask services modeled on the already-proven `fastfs_server.py` |
| `google-genai` SDK | Raw `requests`/`httpx` REST calls to `generativelanguage.googleapis.com` | SDK handles auth/retry/streaming; raw REST is viable too (D-02 says "direct-HTTP", so either is compliant — SDK recommended for less hand-rolled auth/error-handling code, consistent with Don't Hand-Roll below) |
| EEF-to-joint IK per delta-EEF backend | Fine-tune/find an SO-101-native joint-space checkpoint for every backend (avoiding IK entirely) | Some backends (e.g. GR00T N1.5, Wall-OSS) have community SO-101 joint-space fine-tunes available; others (OpenVLA, OpenVLA-OFT's public LIBERO checkpoints, Octo's OXE-pretrained base) do not — IK conversion is the only way to get a live action out of those without first collecting a new fine-tuning dataset, which is out of scope for this phase's "at least one live observed action" bar |

**Installation (Colab-side, per backend, added incrementally per D-05's integration order):**
```bash
# Tier A/B (LeRobot-native or pending dispatch verification) — no new install beyond lerobot[async]==0.6.1 already used for SmolVLA
!pip install 'lerobot[async]==0.6.1'   # already present

# Gemini (local control machine, D-02's direct-HTTP lane)
pip install google-genai==2.25.0

# Tier C — each installed from its own GitHub source in an isolated venv
# (mirrors fastfs_server.py's FASTFS_VENV isolation pattern, so a Tier C
# backend's own torch/CUDA pin never conflicts with the PolicyServer's)
git clone https://github.com/openvla/openvla
git clone https://github.com/octo-models/octo
git clone https://github.com/thu-ml/RoboticsDiffusionTransformer
git clone https://github.com/NVIDIA/Isaac-GR00T        # GR00T N1.5
git clone https://github.com/X-Square-Robot/wall-x     # Wall-X fallback if Tier B dispatch unsupported
```

**Version verification:** `google-genai==2.25.0` confirmed current via `pip index versions google-genai` (2026-09-30). `lerobot==0.6.1` confirmed already installed and in active use (read directly from `control/vla_bridge/robot_client.py`'s docstrings this session). Tier C packages have no PyPI releases — confirmed via `pip index versions octo-models`/`isaac-gr00t` returning no match — install from the pinned commit/tag of each official GitHub org, not `pip install <name>`.

## Package Legitimacy Audit

| Package | Registry | Age | Downloads | Source Repo | Verdict | Disposition |
|---------|----------|-----|-----------|-------------|---------|-------------|
| `google-genai` | PyPI | new entry in legitimacy DB (real package is mature; DB's `publishedAt` reflects last-release timestamp, not package age) | unknown (tool limitation) | `github.com/googleapis/python-genai` | SUS (flagged `too-new`/`unknown-downloads` by the automated check, but `repoUrl` resolves to Google's own official `googleapis` GitHub org) | Approved with note — this is Google's official Gen AI SDK; the SUS flag is a tool-signal limitation (no download-count API access), not a legitimacy concern. Planner should still add a `checkpoint:human-verify` task confirming the installed version matches `pypi.org/project/google-genai` before first use, per the SUS-handling protocol |
| `octo-models` | PyPI | n/a | n/a | n/a | SLOP — does not exist on PyPI | REMOVED from pip-install instructions; Octo is a GitHub-source install only (`octo-models/octo`), confirmed via `pip index versions` returning no match |
| `isaac-gr00t` | PyPI | n/a | n/a | n/a | not found on PyPI (no verdict returned — treated as SLOP-equivalent) | REMOVED from pip-install instructions; GR00T N1.5 is a GitHub-source install only (`NVIDIA/Isaac-GR00T`) |
| `pyngrok` | PyPI | already audited in Phase 11 | unknown | none listed by tool | SUS (already human-verified in Phase 11 per `policy_server_launch.md`'s legitimacy gate note) | Already approved — no new action needed, carried forward unchanged |
| `flask` | PyPI | mature, widely used | unknown (tool limitation) | `github.com/pallets/flask/` | SUS (tool flags `unknown-downloads` only — Flask is an extremely well-known, official-org package) | Approved — already in active use in this repo's `fastfs_server.py`, no new risk introduced |
| `grpcio` | PyPI | mature | unknown (tool limitation) | `grpc.io` (official) | SUS (`too-new`/`unknown-downloads` — tool signal limitation, not a real freshness concern for this widely-used package) | Already approved in `11-RESEARCH.md`'s Package Legitimacy Audit — carried forward |
| `lerobot` | PyPI | already installed, in production use in this repo | unknown (tool limitation) | none listed by tool (repo is `github.com/huggingface/lerobot`, not picked up by the automated check) | SUS (tool limitation — no repo URL resolved) | Already approved and in active use — no new action |
| `openvla`, `octo` (GitHub), `RoboticsDiffusionTransformer`, `Isaac-GR00T`, `wall-x` | GitHub source installs, not PyPI | verified via live WebSearch this session: `openvla/openvla` (official openvla.github.io project org), `octo-models/octo` (official Octo project org), `thu-ml/RoboticsDiffusionTransformer` (Tsinghua ML group, official RDT-1B org, mirrored at `huggingface.co/robotics-diffusion-transformer`), `NVIDIA/Isaac-GR00T` (NVIDIA's own GitHub org), `X-Square-Robot/wall-x` (X Square Robot's own GitHub org, matches their HF org `x-square-robot`) | n/a (GitHub, not registry-download-counted) | all 5 confirmed official first-party org repos via live search | [ASSUMED: package-legitimacy tool only checks PyPI/npm/crates registries, not GitHub source installs] — manually cross-checked org identity this session, but not run through the automated `package-legitimacy check` seam (out of scope for its ecosystem coverage) | Planner should add one `checkpoint:human-verify` task per Tier C backend confirming the pinned commit/tag comes from the exact org URLs listed here before `git clone`, since GitHub source installs bypass the registry-based legitimacy gate entirely |

**Packages removed due to `[SLOP]` verdict:** `octo-models` (PyPI), `isaac-gr00t` (PyPI) — both non-existent as PyPI packages; correct install path is GitHub source clone, not `pip install`.
**Packages flagged as suspicious `[SUS]`:** `google-genai` (tool signal limitation, not a real concern — Google's official SDK), `pyngrok`/`flask`/`grpcio`/`lerobot` (all previously-approved, tool-limitation-only flags, carried forward without new action).

## Architecture Patterns

### System Architecture Diagram

```
                         run_vla_episode.py (LOCAL control machine, unchanged control loop)
                         ┌─────────────────────────────────────────────────────────────┐
                         │  camera capture -> joint_state read -> action_source.get_action()  │
                         │                              │                                │
                         │                              v                                │
                         │                    safety_validator.validate_action()          │
                         │                              │                                │
                         │                              v                                │
                         │              robot.send_action() (real SO-ARM101 servos)       │
                         │                              │                                │
                         │                              v                                │
                         │                    io_logger.write_step() (episode.jsonl)      │
                         └───────────────────────────────┬────────────────────────────────┘
                                                           │ action_source = build_backend(args.backend)
                        ┌──────────────────────────────────┼──────────────────────────────────┐
                        │                                  │                                  │
                        v                                  v                                  v
           ColabGrpcBridgeActionSource         ColabHttpAdapterActionSource          GeminiActionSource (family)
           (Tier A: pi0/pi05, ACT, Diffusion    (Tier C: OpenVLA, OpenVLA-OFT,        (D-02: direct HTTP, no Colab hop)
            Policy; Tier B pending: GR00T N1.7,  Octo, RDT-1B, GR00T N1.5)
            Wall-X)                                        │                                  │
                        │                                  │                                  │
                        v                                  v                                  v
              gRPC (ngrok TCP tunnel)              HTTP/JSON (ngrok HTTP tunnel,       HTTPS (generativelanguage.
              -> Colab PolicyServer                 fastfs_server.py-pattern)          googleapis.com), no Colab
                 subprocess (policy_type=X)         -> Colab Flask adapter              involvement at all
                                                       subprocess (per backend)
                        │                                  │                                  │
                        v                                  v                                  v
                 lerobot.policies.X                  each backend's own model         Gemini Robotics-ER (public)
                 (loaded once, resident)             loading code (loaded once,        -> points/boxes/trajectory
                                                       resident, mirrors FastFSModel)    JSON, NOT joint actions
                                                                                         directly (see Open Questions
                                                                                         for the conversion step this
                                                                                         forces)
```

### Recommended Project Structure

```
control/vla_bridge/
├── action_contract.py          # unchanged — JOINT_ORDER/units single source of truth
├── safety_validator.py         # unchanged
├── io_logger.py                # unchanged
├── robot_client.py             # refactor target (D-04): extract ColabActionSourceBase
├── backends/
│   ├── __init__.py             # registry: BACKEND_REGISTRY = {"pi0": ..., "act": ..., ...}
│   ├── base_grpc.py            # GrpcBridgeMixin — wraps connect_bridge()/RobotClient (Tier A/B)
│   ├── base_http_adapter.py    # HttpAdapterMixin — wraps a resident Colab Flask service (Tier C)
│   ├── base_gemini.py          # GeminiActionSourceBase — shared HTTP/auth plumbing (D-02)
│   ├── pi0.py                  # thin subclass: policy_type="pi0"/"pi05"
│   ├── act.py
│   ├── diffusion_policy.py
│   ├── groot_n17.py            # subclass, notes relative-EE action-space conversion
│   ├── groot_n15.py            # HttpAdapterMixin subclass
│   ├── wallx.py
│   ├── openvla.py              # HttpAdapterMixin + EEF-to-joint IK step
│   ├── openvla_oft.py
│   ├── octo.py
│   ├── rdt1b.py
│   ├── gemini_robotics.py
│   └── gemini_robotics_2.py
└── backend_configs/
    ├── pi0.json                # {"checkpoint": "...", "server_address": null}
    ├── openvla_oft.json        # {"checkpoint": "...", "colab_adapter_port": 3001}
    ├── gemini_robotics.json    # {"api_key_env": "GEMINI_API_KEY", "model": "gemini-robotics-er-2-preview"}
    └── ...                     # one small config per backend, per D-06's discretion
```

### Pattern 1: Tier A/B — gRPC bridge swap (no new Colab code)

**What:** For LeRobot async-native backends, the only change from today's `connect_bridge()` is the `policy_type` string and `pretrained_name_or_path` — everything else (`RobotClientConfig`, `_wire_stereo_split_cameras`, `pop_validated_action`, the stale-watchdog) is identical.
**When to use:** π0/π0.5, ACT, Diffusion Policy (confirmed); GR00T N1.7, Wall-X (pending a live-source check of the installed `lerobot` version's `policy_server.py` dispatch table — if present, same pattern; if absent, fall through to Tier C's pattern for that one backend).
**Example:**
```python
# Source: this repo's control/vla_bridge/robot_client.py connect_bridge(), read directly this session
config = RobotClientConfig(
    policy_type="pi0",                      # was "smolvla" -- the only required change for Tier A
    pretrained_name_or_path=checkpoint,     # new HF repo ID per backend
    robot=robot_config,
    actions_per_chunk=actions_per_chunk,
    task=task,
    server_address=server_address,
    policy_device=policy_device,
    chunk_size_threshold=chunk_size_threshold,
)
```
**Verification step to add per D-01's own logic:** before trusting Tier B's dispatch, grep the installed `lerobot` package's `async_inference/policy_server.py` for the literal string used to map `policy_type` to a policy class (mirrors how this project already resolved the `use_degrees` ambiguity in `action_contract.py` by reading installed source, not docs). `[CITED: github.com/huggingface/lerobot issue #2619 error message confirms the dispatch list is {act, smolvla, diffusion, tdmpc, vqbet, pi0, pi05} as of the session's search — but this list is not from official docs and may be stale relative to the exact pinned version this repo installs]`.

### Pattern 2: Tier C — resident Colab HTTP adapter (reuses the `fastfs_server.py` shape)

**What:** A small Flask service, started as an isolated OS subprocess in its own venv (mirroring `FASTFS_VENV`/`fastfs_server.py` exactly), loads the Tier C backend's model once at startup, exposes a `/action` POST endpoint taking the observation dict + instruction and returning the raw model action, and a `/health` GET endpoint for the same startup-readiness polling pattern already proven for depth.
**When to use:** OpenVLA, OpenVLA-OFT, Octo, RDT-1B, GR00T N1.5 (and Wall-X/GR00T N1.7 as a Tier B fallback if Pattern 1's dispatch check fails).
**Example (adapted directly from this repo's proven `fastfs_server.py` shape):**
```python
# Source: control/vla_bridge/policy_server_launch.md's embedded fastfs_server.py,
# adapted here to the action-output shape instead of depth-output.
class TierCModel:
    def __init__(self, checkpoint_path):
        # Load once, resident -- never reload per request (this was Phase 12's
        # root-cause #2 fix for FastFS: reloading from disk per-request took 15-70s/call).
        self.model = load_backend_specific_model(checkpoint_path)  # e.g. AutoModel.from_pretrained for OpenVLA

    def __call__(self, images: dict, proprio_state: list[float], instruction: str):
        return self.model.predict_action(images, proprio_state, instruction)  # backend-specific call

def create_app(infer):
    from flask import Flask, jsonify, request
    app = Flask(__name__)
    @app.get('/health')
    def health():
        return jsonify(status='ready')
    @app.post('/action')
    def action():
        body = request.get_json()
        result = infer(body['images'], body['proprio_state'], body['instruction'])
        return jsonify(action=result.tolist(), model_version=infer.metadata)
    return app
```
**Why not force these through `lerobot.async_inference.PolicyServer`:** that server's `policy_type` dispatch is a closed set of registered LeRobot policy classes; patching it to accept 5 genuinely different external loading APis (HF `AutoModel`, a custom RDT-1B class, NVIDIA's own GR00T N1.5 package) is more invasive than standing up 5 small, isolated Flask services using an already-proven pattern from this exact codebase.

### Pattern 3: EEF-delta-to-joint conversion (for Cartesian-output backends)

**What:** `action_contract.py`'s `JOINT_ORDER`/`ACTION_UNITS` are joint-space-only (degrees, absolute, 6 DOF). OpenVLA, OpenVLA-OFT (public LIBERO checkpoints), Octo's OXE-pretrained base, and GR00T N1.7 (post-architecture-change, per its "relative end-effector action space" design) output Cartesian end-effector deltas, not joint targets. These must be converted to `JOINT_ORDER`-shaped joint targets BEFORE `safety_validator.validate_action()` runs — the safety gate has no IK capability and expects per-joint values already.
**When to use:** Any backend whose native/available checkpoint outputs EEF deltas rather than joint positions (confirm per-backend at plan time — some community SO-101 fine-tunes for these same model families train directly on joint-space LeRobot datasets instead, avoiding this step entirely; prefer those when they exist).
**Recommendation:** A lightweight analytic/numeric IK for the SO-ARM101's 5-DOF arm (existing third-party references found this session: `SinfonIAUniandes/lerobot_teleoperator_so101_ik` uses `pyroki`; `sruthihsr/LerobotFwd_InvKinematics` documents forward/inverse kinematics for this exact arm) `[CITED: GitHub search results, not official LeRobot docs]` — do not hand-roll a new IK solver from scratch (see Don't Hand-Roll below).

### Anti-Patterns to Avoid

- **Calling `client.control_loop_action()`/`client.control_loop()` for any Tier A/B backend:** both bypass `safety_validator.validate_action()` internally — this was already identified and avoided for SmolVLA (`robot_client.py`'s own docstring); the same bypass risk applies identically to every new gRPC-bridge backend.
- **Trusting a checkpoint's `config.json` camera-key count without fetching and checking it live:** `checkpoint_candidates.md` (this repo, Phase 11) shows real fetched SmolVLA candidates expecting 3-5 camera keys against this project's 2-physical-camera rig — the same check (`check_checkpoint.py`-style fetch-and-inspect) must be repeated for each of the 12 new backends before committing to a camera-mapping in that backend's config.
- **Assuming `policy_type` dispatch support without reading the installed `lerobot` source for Tier B backends:** the async PolicyServer's supported-policy-type list is not the same as `lerobot.policies`'s full registry (see Pattern 1's verification step) — trusting docs/blog posts over installed source repeats the exact `use_degrees` mistake `action_contract.py`'s own docstring warns against.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| EEF-to-joint IK for SO-101 | A from-scratch numeric IK solver | An existing SO-101 IK reference (`pyroki`-based teleoperator project, or the documented forward/inverse-kinematics repo found this session) as a starting point, validated against this project's own calibrated joint limits | IK for a 5-DOF arm with real calibration quirks (already documented gotchas in this repo: `wrist_roll` has no real calibrated range, gripper is a separate percent-mode motor) is easy to get subtly wrong in ways that only show up as unsafe real-hardware motion |
| Resident model-server process management (start/health-check/teardown) | A new ad-hoc subprocess-management script per Tier C backend | The exact `fastfs_server.py` pattern already proven in this repo (isolated venv, `subprocess.Popen`, `/health` polling loop, explicit teardown) | This project already solved — and live-verified — the "don't reload a model from disk every request" and "don't let a crashed sub-service take down the whole Colab kernel" problems for depth; reapplying the same pattern for 5 more backends is strictly lower-risk than inventing a new one per backend |
| Cloud API HTTP client (Gemini) | Raw `requests` calls with hand-rolled retry/auth/streaming logic | `google-genai` (official SDK) | D-02 mandates "direct-HTTP", which the SDK still satisfies (it issues HTTP calls); hand-rolling auth/retry for a cloud API this project has no prior code touching is unnecessary risk for zero benefit |
| Per-backend safety/logging code paths | Backend-specific validation or logging bypass "because this model's action format is different" | The existing `safety_validator.py`/`io_logger.py`, fed a `JOINT_ORDER`-shaped dict after any needed conversion | Explicitly required by VLAB success criteria #3 — no backend-specific safety or logging code paths |

**Key insight:** The 12 new backends are not 12 new problems to solve from scratch — they are 12 instances of three already-solved problem shapes (gRPC bridge swap, resident HTTP adapter, EEF-to-joint conversion) that this project has already built working examples of (SmolVLA's bridge, `fastfs_server.py`'s adapter pattern, and — pending — an IK reference). Treat new backend integration as "which of the 3 known patterns does this one need," not "design a new integration."

## Common Pitfalls

### Pitfall 1: Unit/frame mismatches between a checkpoint's training convention and this project's real degree-mode action contract
**What goes wrong:** A checkpoint trained in radians, normalized [-1,1], or a different joint-sign convention silently drives the arm to its joint limits or produces wildly wrong motion, without raising any exception.
**Why it happens:** `action_contract.py`'s own module docstring already documents that this exact ambiguity (`use_degrees` vs. normalized) was a real, previously-unresolved bug for this project's LeRobot config path. A live GitHub issue found this session (`strands-labs/robots#4195`, `strands-labs/robots#4365`) confirms the identical failure mode recurring for GR00T N1.7 on a SO-ARM checkpoint: "no unit frame, so the official SO-ARM N1.7 checkpoint pins a MuJoCo so101 at every joint limit under status=success" `[CITED: GitHub issue, community-reported, not official NVIDIA docs — treat as a real risk signal, not confirmed root cause without this project's own live verification]`.
**How to avoid:** For every new backend, explicitly verify (via the checkpoint's own config/model card, or a dry-run against a mocked/logged-only observation before live hardware) whether its expected joint units match `ACTION_UNITS` (degrees for the 5 arm joints, percent_0_100 for gripper) before the first live-hardware attempt. `safety_validator.py`'s existing per-step/velocity caps provide a backstop, but a systematic unit-scale error (e.g. radians read as degrees) can still produce a clamped-but-wrong motion that looks "safe" in logs while being functionally useless.
**Warning signs:** Validator flags showing every joint clamped to its per-step cap on the very first action; a checkpoint's action values consistently an order of magnitude (e.g. ~57x, the radian-to-degree ratio) off from the current joint state.

### Pitfall 2: Trusting a policy_type dispatch or camera-key count without live-reading the installed source/config
**What goes wrong:** Planning assumes a backend "just works" through the existing async bridge because a blog post or docs page lists it as a supported LeRobot policy, then discovers at integration time that the specific installed `lerobot` version's `PolicyServer` dispatch doesn't include it (Tier B risk).
**Why it happens:** LeRobot's policy registry (`lerobot.policies`, covering training/fine-tuning) and its async-inference server's dispatch table (covering live bridge serving) are two different pieces of surface area that can be out of sync as the library evolves — confirmed this session via a live GitHub issue's error message listing a narrower set than the full policy registry advertises elsewhere.
**How to avoid:** Repeat this project's own established practice (already used for `action_contract.py`'s `use_degrees` resolution and `robot_client.py`'s `control_loop_action()` bypass): read the actually-installed `lerobot` package source directly, not docs, before committing a plan task to "reuse the existing bridge."
**Warning signs:** `RobotClientConfig(policy_type=...)` raising an unrecognized-policy-type error at connect time.

### Pitfall 3: GR00T N1.5 vs N1.7 action-space incompatibility
**What goes wrong:** Code or config written for one GR00T version's action space (N1.5: absolute targets; N1.7: relative end-effector deltas shared across embodiments, per this session's research) is silently reused for the other version, producing systematically wrong motion.
**Why it happens:** N1.7 introduced a genuine architectural change from N1.5 — not just a version bump — expanding state/action dims (29→132) and switching to relative-EE actions `[CITED: WebSearch results referencing NVIDIA's own GR00T N1.7 release material, not independently confirmed against the primary NVIDIA blog post this session]`.
**How to avoid:** Treat GR00T N1.5 and GR00T N1.7 as two structurally different backends requiring separate subclasses with separate action-conversion logic (already reflected in the Recommended Project Structure's `groot_n15.py`/`groot_n17.py` split) — never share one adapter class with a version flag.
**Warning signs:** An N1.5-shaped absolute-target adapter producing runaway/cumulative drift when actually talking to an N1.7 checkpoint (or vice versa — an N1.7 relative-delta adapter producing near-zero motion against an N1.5 checkpoint that expects absolute targets).

### Pitfall 4: RDT-1B's compute footprint on the Colab tier actually available
**What goes wrong:** RDT-1B (and its T5-XXL text encoder) fails to load or OOMs on a T4 (15GB), and even on an A100 requires explicit offloading configuration below 24GB VRAM.
**Why it happens:** `[CITED: thu-ml/RoboticsDiffusionTransformer GitHub README, referenced via WebSearch this session]` the repo's own guidance states the t5-v1_1-xxl encoder may not fit on RTX 4090-class (24GB) cards without offloading, and recommends an offload directory below 24GB VRAM.
**How to avoid:** Plan RDT-1B's integration assuming Colab Pro A100 access (already an existing project constraint per `CLAUDE.md`'s "Free Colab T4 for OpenVLA-7B inference" out-of-scope note — the same class of constraint applies here), and budget a `checkpoint:human-verify` step confirming the resident adapter loads and warms successfully before the first live-hardware attempt, mirroring `fastfs_server.py`'s own warmup-before-ready pattern.
**Warning signs:** Colab kernel OOM-kill during the adapter's model-load step; `torch.cuda.OutOfMemoryError` in the adapter's log file.

### Pitfall 5: Wall-X / WALL-OSS's exported CUDA operator sources
**What goes wrong:** `wall-x`'s own repo description (`X-Square-Robot/wall-x`) notes it "includes ... exported CUDA operator sources that compile during package installation" `[CITED: WebSearch result summarizing the repo's own README]` — a compile-from-source step at install time that can fail silently or take a long time on a fresh Colab runtime, unlike a pure-Python `pip install`.
**Why it happens:** Custom CUDA kernels for performance-critical ops are common in newer VLA codebases but add a real install-time failure surface not present in the LeRobot-native Tier A backends.
**How to avoid:** Budget extra Colab setup time/cells for Wall-X specifically (mirroring the multi-cell, failure-tolerant install pattern already used for Fast-FoundationStereo's own environment setup in `policy_server_launch.md`'s Step 7), and check compile success explicitly before declaring the adapter ready.
**Warning signs:** `pip install`/`setup.py build_ext` errors referencing `nvcc` or CUDA architecture mismatches during Wall-X's install step.

## Code Examples

### Verified: the seam that already makes all of this possible (no changes needed)

```python
# Source: control/run_vla_episode.py, read directly this session, lines ~143-156
class ActionSource(Protocol):
    def get_action(
        self, joint_state: dict[str, float], instruction: str
    ) -> tuple[dict[str, float], str]:
        """Return (raw_action, model_version) for the given observation."""
        ...
```
Every new backend's `ActionSource` subclass need only satisfy this exact structural interface — this is why PIPE-01's "no control-loop changes" requirement is already achievable without touching `run_vla_episode.py`'s `run_episode()` loop itself, only its CLI branch that assigns `action_source = ...`.

### Verified: per-joint validation already handles any JOINT_ORDER-shaped dict regardless of source

```python
# Source: control/vla_bridge/safety_validator.py, read directly this session
for joint in action_contract.JOINT_ORDER:
    if joint not in raw_action:
        continue
    value = raw_action[joint]
    # NaN/inf rejection, absolute clamp, per-step clamp, velocity clamp --
    # identical regardless of which backend produced `raw_action`.
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|---------------|--------|
| GR00T N1.5 absolute joint/EE targets | GR00T N1.7 relative end-effector deltas shared across embodiments | N1.7 release (2026, exact date not independently confirmed this session) `[ASSUMED: date not verified against a primary NVIDIA announcement]` | Requires a genuinely different action-conversion adapter per GR00T version — cannot share one `GrootActionSource` with a version flag |
| Gemini Robotics VLA restricted to a small trusted-tester list (Boston Dynamics, Agility Robotics, etc.) | Gemini Robotics ER 2 (embodied reasoning only) publicly available via Gemini API/AI Studio as of 2026-07-30; the full VLA (native action output) remains early-access-partner gated | 2026-07-30 (Gemini Robotics 2 suite announcement) `[CITED: Google DeepMind's own blog post, via WebSearch this session]` | Directly affects VLAB-01/VLAB-02 planning — see Open Questions |
| Loading a Colab model fresh from disk per inference request | Resident, warmed, subprocess-isolated model server (this project's own `fastfs_server.py` pattern) | Phase 12 (2026-09-30, this repo) | Applying this same pattern to Tier C backends avoids repeating the exact 15-70s/request latency bug already found and fixed for depth |

**Deprecated/outdated:**
- Assuming `lerobot.async_inference.PolicyServer` covers every model in `lerobot.policies`'s registry — confirmed this session that the async dispatch list is narrower than the full policy registry.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | GR00T N1.7's exact relative-EE action-space change and N1.5→N1.7 dimension expansion (29→132) are correctly summarized from WebSearch results, not independently verified against NVIDIA's own primary technical documentation/paper | Per-Backend Findings, State of the Art, Pitfall 3 | If wrong, the N1.7-specific action-conversion subclass could be built against an incorrect action-space assumption, producing wrong motion on first live attempt — mitigated by `safety_validator.py`'s existing hard caps, but wastes the "prove the pattern" integration attempt |
| A2 | `google-genai`'s `[SUS]` verdict is a tool signal-availability limitation, not a genuine legitimacy concern, based on the repo resolving to Google's own official `googleapis` GitHub org | Package Legitimacy Audit | Low risk (repo org is independently confirmed as Google's own), but planner should still route this through a `checkpoint:human-verify` step per protocol rather than skip it |
| A3 | `lerobot.async_inference.PolicyServer`'s confirmed `policy_type` dispatch list (`act, smolvla, diffusion, tdmpc, vqbet, pi0, pi05`) — sourced from a GitHub issue's error message, not official LeRobot docs — reflects the exact version this project has installed (`0.6.1`), and that GR00T N1.7/Wall-X are genuinely absent from it | Summary, Pattern 1, Tier B classification throughout | If the installed version actually does support GR00T N1.7/Wall-X dispatch, the Tier B verification step is unnecessary overhead (low-cost false positive); if it's missing dispatch that the GitHub issue's version DID have, Tier B backends could wrongly be planned as Tier A and fail at connect time |
| A4 | Octo's action space for whatever checkpoint ends up used is Cartesian/delta-EEF by default (true for its OXE-pretrained base checkpoints per general VLA literature), requiring the IK conversion step — not independently confirmed against a specific Octo checkpoint's config this session | Architectural Responsibility Map, Pattern 3 | If Octo's actual chosen checkpoint is already joint-space (e.g. a community SO-101 fine-tune, should one exist), the IK step is unneeded overhead; if genuinely EEF-space and this is missed, actions sent as raw EEF deltas through `JOINT_ORDER`'s joint-space validator would be nonsensical (caught by the safety clamp, but produces a broken/pointless "live action" for VLAB-08's success bar) |
| A5 | The full Gemini Robotics (V1) and Gemini Robotics 2 VLA models remain genuinely inaccessible via any public API today (early-access-partner gated), based on WebSearch results from secondary sources (blog aggregators, MarkTechPost, etc.) rather than an exhaustive check of Google's own developer console for a possible waitlist-application path that might grant access within this phase's timeline | Open Questions (primary blocker) | HIGH risk if wrong in the optimistic direction (access is actually obtainable quickly) — would mean the planner unnecessarily descopes to the ER-based workaround; HIGH risk if wrong in the pessimistic direction (ER 2's public API is itself also gated/rate-limited in ways not surfaced here) — would mean even the fallback plan is blocked |

**This table is populated** — all 5 entries need user/planner attention per the risk column, A5 most urgently.

## Open Questions

1. **Is the full Gemini Robotics / Gemini Robotics 2 VLA (native joint/action output) actually reachable for this project within Phase 14's timeline, or must VLAB-01/VLAB-02 use the Gemini Robotics-ER workaround?**
   - What we know: Gemini Robotics-ER 2 (embodied reasoning: points, bounding boxes, trajectories in JSON — NOT joint/motor commands) is publicly available today via the Gemini API and Google AI Studio (confirmed 2026-07-30 announcement, `ai.google.dev/gemini-api/docs/robotics-overview` fetched directly this session). The full Gemini Robotics VLA (native action output) and Gemini Robotics 2 (VLA) remain gated to "early-access partners" — the same restricted-access tier the original Gemini Robotics VLA used ("trusted testers... Boston Dynamics, Agility Robotics...") before ER's public release.
   - What's unclear: Whether this project (or its operator) can obtain early-access-partner status within this phase's timeline, and if not, whether the user wants VLAB-01/VLAB-02 satisfied via a Gemini Robotics-ER-based workaround (ER's spatial-reasoning output + a local primitive/IK layer converts points/trajectories into `JOINT_ORDER`-shaped actions, still genuinely "Gemini Robotics [ER] driving the real SO-ARM101" but architecturally different from D-02's "direct-HTTP ActionSource calling Google's API straight" premise, which implicitly assumed action-output access).
   - Recommendation: Surface this to the user as a decision point before planning commits to D-02's architecture for these 2 backends specifically. Two concrete paths: (a) apply for Gemini Robotics 2 early access now and treat VLAB-01/VLAB-02 as blocked/deferred pending approval (real risk of missing this phase's timeline entirely), or (b) build `GeminiActionSource` against the publicly-available ER 2 API, with an explicit local conversion layer (point/trajectory output → a scripted or simple-IK grasp/reach primitive → `JOINT_ORDER` action), documented clearly as "Gemini Robotics-ER-driven, not native Gemini Robotics VLA action output" so the success-criteria language is honest about what was actually integrated. D-05 already sequences these 2 backends last, which gives runway for either path to resolve before they're blocking.

2. **Does the installed `lerobot==0.6.1`'s `async_inference/policy_server.py` actually support `policy_type="groot"` / `policy_type="wall_oss"` (or equivalent strings), or do GR00T N1.7 and Wall-X need the Tier C custom-adapter fallback?**
   - What we know: A live GitHub issue's error message (not official docs) lists the confirmed-supported dispatch set as `{act, smolvla, diffusion, tdmpc, vqbet, pi0, pi05}`. Separately, LeRobot's broader policy documentation lists GR00T N1.7 and WALL-OSS among its "VLA Models" — meaning training/loading support exists in `lerobot.policies`, but that's a different code path from the async server's dispatch table.
   - What's unclear: Whether the exact `lerobot==0.6.1` version installed in this project's Colab notebook has these two wired into `policy_server.py`'s dispatch, since the GitHub issue source may reflect a different (older or newer) version.
   - Recommendation: First task of the GR00T N1.7 and Wall-X plans should be a direct `grep`/read of the installed `lerobot` package's `async_inference/policy_server.py` (same technique already used in this repo to resolve the `use_degrees` ambiguity) — not a docs lookup — before committing to Tier A/B vs. Tier C architecture for either backend.

3. **Which of the 12 backends have an existing SO-101/SO-100 joint-space community fine-tune checkpoint (avoiding the EEF-to-joint IK conversion step entirely), vs. which only have Cartesian-EEF-output base/OXE-pretrained checkpoints?**
   - What we know: SmolVLA's precedent (`victorvanhalst/smolvla_so101_cube`) shows community SO-101 fine-tunes do exist and use native joint-space camera/action conventions. OpenVLA-OFT's public checkpoints found this session (`moojink/openvla-7b-oft-finetuned-libero-*`) are LIBERO/Franka-targeted, not SO-101-targeted, and use 7-DoF delta-EEF.
   - What's unclear: A full HF Hub search per backend (mirroring `check_checkpoint.py`'s already-proven fetch-and-inspect approach from Phase 11) was not performed for all 12 backends this session, due to research time constraints — this is genuinely per-backend legwork each backend's own plan should do at Task 1, not something this research session could exhaustively front-load for all 12.
   - Recommendation: Each backend's plan's first task should run a `check_checkpoint.py`-style live fetch against 2-3 HF Hub search-result candidates for that backend + "so101" or "so100", exactly as Phase 11 did for SmolVLA, before committing to either "use this joint-space checkpoint directly" or "use the base/OXE checkpoint + IK conversion."

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Local Python venv (`control/.venv`) | Running `run_vla_episode.py`'s control loop and all `ActionSource` subclasses | ✓ | Python 3.12 (confirmed: `.venv/bin/python3.12` present) | — |
| `pytest` test collection | Validation Architecture (below) | ✓ | 208 existing tests collect cleanly in `control/` (confirmed via `pytest --collect-only` this session) | — |
| Colab GPU runtime (A100 preferred, T4 for lighter backends) | All 10 Colab-hosted backends (Tier A/B/C) | Not locally probable — external, session-provisioned resource | Unknown until each Colab session starts | RDT-1B specifically needs A100-class VRAM headroom (Pitfall 4) — no viable T4 fallback found this session for that one backend |
| `google-genai` SDK | Gemini backends (D-02) | Not yet installed locally — confirmed installable via `pip index versions` (2.25.0 current) | 2.25.0 | — |
| Gemini Robotics / Gemini Robotics 2 (VLA, native action output) API access | VLAB-01, VLAB-02 | ✗ — gated to early-access partners as of this session's research | — | Gemini Robotics-ER 2 (public) + local conversion layer (see Open Questions #1) |
| HF Hub authentication (for gated/rate-limited checkpoint downloads) | All Colab-hosted backend checkpoint downloads | Already wired via `policy_server_launch.md`'s Drive-token-or-getpass pattern | — | — |

**Missing dependencies with no fallback:**
- Gemini Robotics / Gemini Robotics 2 native VLA action-output API access (VLAB-01/02) — no public fallback exists that produces native action output; the available fallback (ER + conversion layer) changes what's actually being integrated, which is why this is flagged as an Open Question requiring a user decision, not silently substituted.

**Missing dependencies with fallback:**
- GR00T N1.7 / Wall-X async-server `policy_type` dispatch (if unconfirmed/unsupported) — falls back to the Tier C custom-HTTP-adapter pattern, already proven in this repo.

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (confirmed: 208 tests collected cleanly from `control/` this session) |
| Config file | none — no `pytest.ini`/`pyproject.toml` test config found at repo root or in `control/`; `control/conftest.py` supplies shared fixtures (`MockRobot`, etc.) |
| Quick run command | `.venv/bin/python -m pytest control/test_<new_backend>.py -x` (run from `control/`) |
| Full suite command | `.venv/bin/python -m pytest control/ -x` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| PIPE-01 | Existing `ActionSource` Protocol/control loop accepts any new backend without modification | unit (structural conformance) | `pytest control/test_run_vla_episode.py -x` | ✅ exists (extend, don't replace) |
| VLAB-03/05/12 (Tier A) | `connect_bridge()`-equivalent for pi0/act/diffusion produces a valid `RobotClientConfig` and reuses `pop_validated_action()` unmodified | unit, mocked client (mirrors `test_robot_client.py`'s existing `MockRobot`-style doubles) | `pytest control/test_robot_client.py -x` | ✅ exists (extend with new backend's `policy_type`) |
| VLAB-04/10 (Tier B) | Dispatch verification against installed `lerobot` source | manual-only (one-time source read, not automatable) | n/a — documented finding, not a test | ❌ Wave 0 — no automated way to assert against a vendored library's internal dispatch table; treat as a plan task, not a test |
| VLAB-06/07/08/09/11 (Tier C) | Resident HTTP adapter loads once, responds to `/health`, returns a well-shaped action for a given observation | integration (mocked HTTP, mirrors `test_depth_pipeline.py`'s existing FastFS-adapter test pattern) | new `control/test_backend_adapters.py::test_<backend>_adapter_health` / `::test_<backend>_adapter_returns_valid_action_shape` | ❌ Wave 0 |
| VLAB-01/02 (Gemini) | `GeminiActionSource` auth/request/response shape is correctly parsed into a `JOINT_ORDER`-compatible dict (or explicitly documented as ER+conversion) | unit, mocked HTTP response | new `control/test_gemini_action_source.py` | ❌ Wave 0 |
| All 12 VLAB | At least one live observed action per backend | manual-only (live hardware, per D-05's "independently live-hardware-verifiable" requirement) | n/a | n/a — this is explicitly a human-verified checkpoint per backend, not automatable |

### Sampling Rate
- **Per task commit:** `pytest control/test_<touched_file>.py -x`
- **Per wave merge:** `pytest control/ -x` (full 208+ suite)
- **Phase gate:** Full suite green before `/gsd-verify-work`, plus the 12 individual live-hardware checkpoints per D-05's sequencing

### Wave 0 Gaps
- [ ] `control/test_backend_adapters.py` — covers VLAB-06/07/08/09/11 (Tier C resident-adapter health/response-shape tests), modeled directly on `control/test_depth_pipeline.py`'s existing FastFS-adapter test pattern
- [ ] `control/test_gemini_action_source.py` — covers VLAB-01/02, mocked `google-genai` HTTP responses
- [ ] Shared `backends/` registry test — one parametrized test asserting every entry in `BACKEND_REGISTRY` satisfies the `ActionSource` Protocol (`get_action(joint_state, instruction) -> (action, model_version)`), preventing a future backend from silently breaking PIPE-01's structural guarantee
- [ ] Framework install: none — pytest already present and in active use; no new test framework needed

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-------------------|
| V2 Authentication | yes (Gemini backends only) | `google-genai` SDK's standard API-key auth; API key loaded from env var / Drive token file, never hardcoded — mirrors this repo's existing HF/ngrok token-handling pattern in `policy_server_launch.md` |
| V3 Session Management | no | No user sessions — this is a service-to-service bridge, not a multi-user web app |
| V4 Access Control | yes (Colab tunnels) | Already-established pattern: every Colab-hosted tunnel (gRPC and HTTP) must be torn down at end of session per the existing "Teardown (every session, no exceptions)" discipline in `policy_server_launch.md` — an unattended open tunnel is an elevation-of-privilege risk already flagged as T-11-08 in this project's own threat register. This applies identically to the new Tier C Flask adapters' tunnels. |
| V5 Input Validation | yes | `safety_validator.py`'s existing NaN/inf rejection, joint-limit clamping, per-step/velocity caps, stale-observation override — unmodified, reused for all 12 new backends per VLAB success criteria #3 |
| V6 Cryptography | no (beyond standard HTTPS for the Gemini API and ngrok's own TLS) | No new cryptographic code needed — `google-genai`/`requests`-over-HTTPS and ngrok's existing tunnel encryption suffice |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|----------------------|
| Unattended open Colab tunnel (gRPC or new HTTP adapter tunnels) left running after a session ends | Elevation of Privilege | Explicit teardown cell at end of every Colab session, per existing `policy_server_launch.md` Step 6 pattern — extend identically to each new Tier C adapter's own tunnel |
| Malformed/adversarial action from any of the 12 new backends (bug, hallucinated output, or a compromised cloud API response) reaching the real robot | Tampering | `safety_validator.py`'s existing untrusted-input posture — "every candidate action... is treated as untrusted input regardless of origin" (already documented in its module docstring) — no new work needed, but this is the reason no backend may bypass it |
| API key for Gemini backends leaked via logging (`io_logger.py`'s `raw_model_output`/`action_details` fields could inadvertently capture request headers if a `GeminiActionSource` implementation logs the raw HTTP request) | Information Disclosure | New `GeminiActionSource` code must log only the response body/parsed action, never the outgoing request's auth header — mirrors `io_logger.py`'s own existing disclosure discipline (frame-hash "attests only to local bytes... never in episode.jsonl or any other output file" pattern already documented there) |
| GR00T N1.7/N1.5 or Wall-X's exported/custom CUDA operator compilation step (Pitfall 5) executing arbitrary build scripts from a third-party GitHub repo during Colab setup | Tampering (supply chain) | Pin to a specific, human-verified commit/tag for each Tier C GitHub source install (not tracking `main` unpinned) — add a `checkpoint:human-verify` task per Tier C backend confirming the pinned commit comes from the official org before first Colab run |

## Sources

### Primary (HIGH confidence — read directly from installed source in this repo)
- `control/run_vla_episode.py` — `ActionSource` Protocol, `ScriptedActionSource`, CLI branch
- `control/vla_bridge/robot_client.py` — `BridgeActionSource`, `connect_bridge()`, `pop_validated_action()`
- `control/vla_bridge/action_contract.py` — `JOINT_ORDER`, `ACTION_UNITS`, `ACTION_MODE`
- `control/vla_bridge/safety_validator.py` — `validate_action()`
- `control/vla_bridge/io_logger.py` — `IOLogger`
- `control/vla_bridge/policy_server_launch.md` — Colab notebook setup, `fastfs_server.py` embedded adapter pattern
- `control/vla_bridge/checkpoint_candidates.md` — prior live checkpoint-fetch precedent (Phase 11)
- `.planning/config.json` — `workflow.nyquist_validation: true`, `workflow.security_enforcement: true`

### Secondary (MEDIUM confidence — WebSearch/WebFetch verified against official/primary sources this session)
- `ai.google.dev/gemini-api/docs/robotics-overview` (fetched directly) — Gemini Robotics-ER API input/output format, auth
- `huggingface.co/docs/lerobot/en/pi0`, `huggingface.co/docs/lerobot/act`, `huggingface.co/docs/lerobot/bring_your_own_policies` — LeRobot policy documentation
- `huggingface.co/robotics-diffusion-transformer/rdt-1b`, `github.com/thu-ml/RoboticsDiffusionTransformer` — RDT-1B
- `github.com/NVIDIA/Isaac-GR00T`, `huggingface.co/nvidia/GR00T-N1.7-3B`, `huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning` — GR00T N1.5/N1.7
- `github.com/X-Square-Robot/wall-x`, `huggingface.co/x-square-robot/wall-oss-0.5` — Wall-X/WALL-OSS
- `github.com/openvla/openvla`, `huggingface.co/openvla/openvla-7b` — OpenVLA action space
- `deepmind.google/blog/gemini-robotics-2-brings-whole-body-intelligence-to-robots` — Gemini Robotics 2 release/access tiers

### Tertiary (LOW confidence — WebSearch only, secondary aggregator sources, marked for validation)
- GitHub issue error messages (`strands-labs/robots#2619`, `#4195`, `#4365`) — real community-reported signals, not official docs; flagged inline wherever cited
- MarkTechPost/blog aggregator summaries of Gemini Robotics 2's access tiers (cross-checked against DeepMind's own blog post where possible, but not exhaustively)
- WebSearch AI-generated summaries of arXiv 2606.08881 (the Yu & Qiu paper) — PDF fetch returned only structural metadata, not body text, so this paper's specific per-backend findings (relevant to EVAL-01/02 in a later phase, and potentially useful context for VLAB integration order) were NOT successfully extracted this session; recommend a follow-up fetch attempt (e.g. via the arXiv abstract/HTML page rather than the PDF) before Phase 16 planning

## Metadata

**Confidence breakdown:**
- Standard stack (existing infra reuse): HIGH — read directly from installed source
- Standard stack (new Tier C/Gemini packages): MEDIUM — WebSearch/pip-index verified, not all cross-checked against primary org docs
- Architecture (3-tier split, base-class recommendation): MEDIUM — derived from a live GitHub issue error message (not official docs) plus this project's own proven `fastfs_server.py` precedent
- Pitfalls: MEDIUM — several sourced from live community GitHub issues describing this exact SO-ARM/GR00T failure mode, not independently reproduced this session
- Gemini Robotics API availability: HIGH confidence in the finding itself (directly fetched official Google docs + DeepMind blog post), but LOW confidence in whether early-access-partner status is obtainable within this phase's timeline (not knowable from research alone — genuinely requires a user/business decision)

**Research date:** 2026-09-30
**Valid until:** 14 days (fast-moving domain — Gemini Robotics API access tiers and LeRobot's async-server dispatch support are both actively evolving; re-verify before acting on this research if more than 2 weeks pass)
