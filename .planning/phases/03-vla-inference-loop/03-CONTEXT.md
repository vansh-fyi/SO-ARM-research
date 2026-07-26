# Phase 3: VLA Inference Loop - Context

**Gathered:** 2026-07-18
**Status:** Ready for planning

<domain>
## Phase Boundary

A complete closed loop from a language prompt to a rendered SOARM episode video, with task success/failure detection via LIBERO's BDDL evaluation protocol, and a swappable dual-VLA interface supporting both OpenVLA-OFT and π0 (openpi) as real, working inference backends (not a stub). Runs against the 3 frozen `libero_spatial` tasks from Phase 2. Dataset collection (Phase 4), multi-camera/spatial awareness (Phase 5), and fine-tuning (Phase 6) are explicitly out of scope here.

</domain>

<decisions>
## Implementation Decisions

### VLA Interface & Action Consumption
- **D-01:** The shared interface signature is `predict(images: dict[str, Image], language: str) -> action`, where `images` is a dict of named camera views (not a single image) — even though Phase 3 only populates one key (the eye_in_hand camera tuned in Phase 2). This future-proofs the interface for Phase 5's multi-camera work (SPAT-01/02) without a breaking signature change later.
- **D-02:** Each backend (OpenVLA-OFT, π0) handles its own action/observation normalization internally inside its `predict()` implementation. There is no shared normalization layer. OFT overlays the confirmed `libero_spatial_no_noops` norm_stats itself; π0 does whatever normalization openpi requires internally. This keeps the shared interface minimal and avoids forcing OFT's LIBERO-specific stats format onto π0.
- **D-03:** How OpenVLA-OFT's confirmed 8-step action chunk (`(8, 7)` shape, from `predict_action`'s `(actions, hidden_states)` return) is consumed — open-loop replay of all 8 steps vs. closed-loop re-inference after fewer steps — is **Claude's discretion**, informed by what the LIBERO eval protocol and OFT paper actually do (research this during `gsd-phase-researcher`).
- **D-04:** Where the shared VLA interface module lives in the codebase (e.g. `LIBERO/libero/libero/vla/` vs. `explorations/vla/`) is **Claude's discretion** — decide during planning based on where Phase 4 (dataset collection) and Phase 6 (fine-tuning) will need to import it from.

### π0 Integration Depth
- **D-05:** π0 (openpi) must run **real, working inference on Colab** in this phase — not an interface-only stub. This proves Roadmap success criterion #3 ("the pi0 backend can be swapped in via the same interface without changing downstream pipeline code") literally, not just structurally.
- **D-06:** Whether OpenVLA-OFT and π0 run in the same Colab notebook/kernel or in two separate notebooks/kernels is **Claude's discretion** — confirm openpi's actual dependency stack (expected JAX-based, likely conflicting with OFT's torch/transformers stack) during research before committing to notebook structure. Precedent: STATE.md already documents a kernel split between LIBERO training and VLA inference for the same class of version-conflict reason.
- **D-07:** Which specific π0 checkpoint/variant to target (e.g. π0-FAST or a LIBERO-finetuned checkpoint) is **Claude's discretion** — the researcher determines the smallest/fastest Colab-GPU-compatible option available, mirroring how OFT's checkpoint was chosen in Phase 1 for T4 compatibility.
  - **Amendment (2026-07-26):** openpi's `serve_policy.py --env` flag is a coarse `EnvMode` enum (`ALOHA`/`ALOHA_SIM`/`DROID`/`LIBERO`), so the originally-chosen config name was never a valid `--env` value in the first place, and — more importantly — has no published ready-to-serve inference checkpoint on openpi's current `main` branch as of 2026-07-26 (confirmed directly against `serve_policy.py`'s source and the project README, both re-fetched on that date). Resolution: serve via plain `--env=LIBERO`, which resolves through openpi's own default-checkpoint mapping to `pi05_libero` — openpi's current default checkpoint for the LIBERO environment. VLA-04's intent (proving the `Pi0Backend`/interface swap works) is unaffected, since `Pi0Backend.predict()` only depends on the websocket `infer()` contract, not which specific config `serve_policy.py` loads.

### Task & Prompt Scope
- **D-08:** The demo loop runs **all 3** frozen `libero_spatial` tasks from Phase 2, looped automatically in a single notebook run (not a single fixed task, not a user-selectable picker).
- **D-09:** OpenVLA-OFT runs the **full suite**: 3 tasks × 5-10 episodes each, to produce a real success-rate number for VLA-03 (closer to LIBERO's standard eval convention than a 1-episode smoke test).
- **D-10:** π0 runs a **lighter smoke-test only**: e.g. 1 task × 1-2 episodes. π0's job in this phase is to prove the interface swap works, not to match OFT's full evaluation depth — OFT is the primary/proven backend (97.1% LIBERO avg, already confirmed working in Phase 1).
- **D-11:** Video output is **one video file per episode** (not one combined video per task) — matches LIBERO's existing per-episode `video_utils.py` save pattern, simplest to implement, easiest to inspect individual failures.

### Success Detection & Episode Termination
- **D-12:** Poll LIBERO's `check_success()` **every step** during an episode; stop the episode early the moment it returns true. Matches LIBERO's own benchmark evaluation pattern, saves sim/GPU time, and gives a clean success timestamp for the saved video.
- **D-13:** The max-step cap for a never-succeeding episode is **Claude's discretion** — confirm LIBERO's standard eval horizon for the `libero_spatial` task suite during research and use that value (not an arbitrary cutoff), so VLA-03's success rate stays comparable to published LIBERO numbers.
- **D-14:** Reporting is **printed PASS/FAIL per episode, plus an aggregated success-rate summary table at the end** (across all episodes/tasks/backends) — mirrors Phase 1/2's established PASS/FAIL verification-cell convention already used in this project's notebooks.
- **D-15:** Whether a max-step timeout should be a hard binary failure or should also capture a partial-credit/near-miss diagnostic signal (e.g. distance-to-goal at timeout) is **Claude's discretion** — decide during planning whether that extra instrumentation is worth it without overbuilding for this MVP phase.

### Claude's Discretion (summary)
- D-03: OFT action-chunk consumption strategy (open-loop vs. closed-loop re-inference).
- D-04: Exact module location for the shared VLA interface.
- D-06: Single vs. split notebook/kernel structure for OFT vs. π0.
- D-07: Specific π0 checkpoint/variant.
- D-13: Exact max-step cap value.
- D-15: Whether to capture a near-miss/partial-credit signal on timeout.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements & Roadmap
- `.planning/REQUIREMENTS.md` §VLA Inference Pipeline — VLA-01, VLA-02, VLA-03, VLA-04 are this phase's requirements.
- `.planning/ROADMAP.md` §Phase 3: VLA Inference Loop — success criteria (single-cell prompt→video, success/failure detection via BDDL protocol, swappable OFT/π0 interface).

### Required Environment & History Reading
- `.planning/phases/01-colab-environment-setup/01-DEBUG-HISTORY.md` — REQUIRED reading before touching the Colab environment: the 7-invariant environment contract, `--no-deps` install contract, restart-vs-clean-slate distinction.
- `.planning/phases/01-colab-environment-setup/01-CONTEXT.md` — OpenVLA-OFT loading pattern, A100/bf16 GPU tier decision, PASS/FAIL verification-cell convention to replicate.
- `.planning/phases/02-soarm-robot-integration/02-CONTEXT.md` and `02-VERIFICATION.md` — confirms the 3 frozen `libero_spatial` tasks, SOARM robot class, and the tuned eye_in_hand camera this phase's VLA input must consume.

### OpenVLA-OFT Integration Facts (from Phase 1, confirmed on Colab)
- OFT checkpoint norm_stats cover only OXE pretraining datasets — LIBERO-specific stats (`dataset_statistics.json` via `hf_hub_download`) must be overlaid after `from_pretrained`, using key `"libero_spatial_no_noops"`.
- `predict_action` returns `(actions, hidden_states)` — `actions` has shape `(8, 7)` float64 (8-step chunk). This phase's interface must unpack and consume the chunk, not a single step (D-03 decides how).
- `LIBERO/libero/lifelong/utils.py` and `LIBERO/libero/libero/utils/video_utils.py` — existing utilities for task embeddings and per-episode video saving; reuse rather than reimplement.

### STATE.md Prior Research Decisions
- Two separate Colab kernel groups were already needed for LIBERO training vs. VLA inference (transformers version conflict) — same risk class applies to OFT (torch) vs. π0 (likely JAX) per D-06.
- Demo replay (relevant to Phase 4, not this phase) must be state-based, not action-replay — LIBERO issue #16.

### Codebase Maps
- `.planning/codebase/STACK.md` — current pinned dependency versions (robosuite 1.4.0, MuJoCo 2.3.7, transformers 4.21.1 baseline — Phase 1 overrides transformers to 4.40.1).
- `.planning/codebase/INTEGRATIONS.md` — no existing VLA/model-serving integration; this phase introduces the first one.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `LIBERO/libero/libero/envs/robots/` (Phase 2) — SOARM robot class (`ROBOT_CLASS_MAPPING`) that this phase's environment instantiation must use for all 3 tasks.
- `LIBERO/libero/libero/utils/video_utils.py` — existing per-episode video-saving utility (supports D-11's one-video-per-episode decision).
- `LIBERO/notebooks/01-colab-env-setup.ipynb` and `LIBERO/notebooks/02-soarm-integration-check.ipynb` — established PASS/FAIL cell-structure convention to extend for this phase's notebook(s).
- Phase 1's confirmed OpenVLA-OFT loading code (bf16, A100, norm_stats overlay) — direct starting point for this phase's OFT `predict()` implementation.

### Established Patterns
- `MUJOCO_GL` must be set before any MuJoCo import — same constraint across all phases.
- Local-editing-vs-Colab-verification split (established in Phase 2) likely does not apply here — VLA inference inherently requires GPU, so most of this phase's iteration happens on Colab directly.
- Real Python modules over notebook-embedded code for anything reusable by later phases (D-08 pattern from Phase 2) — the shared VLA interface (D-04) should follow this same convention.

### Integration Points
- Phase 4 (Dataset Collection) and Phase 6 (Fine-Tuning) will import this phase's `predict(images, language) -> action` interface directly — it must be a clean, reusable module, not notebook-only code.
- Phase 5 (Spatial Awareness) will extend the `images` dict (D-01) with additional camera keys — the interface must not need a signature change to accommodate this.

</code_context>

<specifics>
## Specific Ideas

- User wants π0 to be a fully proven, working second backend in this phase (not deferred) — this is the clearest strong preference from discussion, directly shaping D-05.
- User wants a real (not smoke-test) success-rate number out of OpenVLA-OFT specifically (D-09) — π0 explicitly does not need the same evaluation depth (D-10).

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope. Multi-camera input, 3D localization, and spatial-language prompts remain correctly deferred to Phase 5 per REQUIREMENTS.md; fine-tuning remains deferred to Phase 6.

</deferred>

---

*Phase: 3-VLA Inference Loop*
*Context gathered: 2026-07-18*
