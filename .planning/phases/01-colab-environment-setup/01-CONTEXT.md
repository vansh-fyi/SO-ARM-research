# Phase 1: Colab Environment Setup - Context

**Gathered:** 2026-07-08
**Status:** Ready for planning

<domain>
## Phase Boundary

A single Colab notebook (`LIBERO/notebooks/01-colab-env-setup.ipynb`) where all simulation dependencies install in correct order on a fresh A100 runtime, EGL headless rendering produces non-black LIBERO frames, and OpenVLA-OFT loads on GPU and returns a valid 7-D action tensor. Deliverable is the notebook itself with structured PASS/FAIL verification cells for each ENV requirement.

</domain>

<decisions>
## Implementation Decisions

### Notebook Structure
- **D-01:** Single notebook with restart-aware cell ordering — one kernel, clearly marked restart cells between the pip install group and the verification group. No split into multiple files.
- **D-02:** Notebook lives at `LIBERO/notebooks/01-colab-env-setup.ipynb` — co-located with existing LIBERO notebooks.

### Transformers Version Conflict
- **D-03:** Override to `transformers==4.40.1` (OpenVLA-OFT's requirement). Phase 1 only needs LIBERO for EGL rendering, not training, so LIBERO's 4.21.1 pin can be safely overridden for this phase.
- **D-04:** After the runtime restart and pip installs complete, render a default Panda LIBERO environment frame to explicitly verify LIBERO rendering still works under 4.40.1 (this is the ENV-02 check).

### GPU Tier
- **D-05:** Target Colab Pro A100 (40GB VRAM). Load OpenVLA-OFT in bf16 — no 4-bit quantization needed, no bitsandbytes dependency.
- **D-06:** Include a GPU assertion cell early in the notebook — check `torch.cuda.get_device_name(0)` and print a loud warning (or raise) if the runtime is not A100. Prevents silent OOM failures mid-execution.

### Verification Artifact
- **D-07:** Each ENV requirement (ENV-01, ENV-02, ENV-03) gets its own structured verification cell with explicit `PASS` / `FAIL` output:
  - ENV-01: Print installed package versions, confirm no conflicts
  - ENV-02: Render a LIBERO frame — display inline with `plt.imshow()` AND save to `LIBERO/notebooks/outputs/libero_render_check.png`
  - ENV-03: Load OpenVLA-OFT, pass a dummy image + prompt, print action tensor shape (must be `(7,)`)

### Claude's Discretion
- Exact pip install ordering and pinned versions within the install cells — researcher and planner determine the correct dependency resolution order.
- Which specific LIBERO BDDL task and camera config to use for the ENV-02 render check.
- Whether to use `IPython.display.Image` or `matplotlib` for inline display.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements
- `.planning/REQUIREMENTS.md` §Environment Setup — ENV-01, ENV-02, ENV-03 are the three requirements for Phase 1. Success criteria fully defined there.
- `.planning/ROADMAP.md` §Phase 1 — Success criteria (1), (2), (3) map to ENV-01, ENV-02, ENV-03.

### Existing Simulation Code
- `LIBERO/requirements.txt` — Pinned LIBERO deps (robosuite 1.4.0, MuJoCo 2.3.7, gym 0.25.2, transformers 4.21.1). Phase 1 will override transformers but must satisfy the rest.
- `LIBERO/libero/` — LIBERO core library. ENV-02 verify cell will import from here to create a default Panda environment for the render check.
- `explorations/create_scene.py` — Existing pattern for instantiating a LIBERO `OffScreenRenderEnv` from a BDDL file with `MUJOCO_GL=egl`. Planner should reference this as the EGL rendering pattern.

### Prior Research Decisions (from STATE.md)
- OpenVLA-OFT is confirmed as primary VLA (97.1% LIBERO avg, T4-compatible at 4-bit — but Phase 1 uses A100 in bf16).
- SOARM MJCF is deferred to Phase 2 — Phase 1 uses Panda arm for all rendering tests.
- Two-kernel concern noted during research is resolved: override to 4.40.1 in Phase 1.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `explorations/create_scene.py`: Sets `os.environ["MUJOCO_GL"] = "glfw"` (macOS) then imports and instantiates `OffScreenRenderEnv`. The ENV-02 cell should replicate this pattern but with `MUJOCO_GL=egl` for Colab Linux runtime.
- `LIBERO/notebooks/`: Existing directory — new notebook lands here without creating new top-level structure.

### Established Patterns
- `MUJOCO_GL` must be set **before** any MuJoCo import (not after) — this is established in `create_scene.py` and is the critical ordering constraint for ENV-02.
- `ROOT = Path(__file__).resolve().parent` style path resolution — notebook equivalent will use absolute paths or `os.getcwd()` since Colab notebooks don't have `__file__`.

### Integration Points
- Phase 2 will extend the notebook's install cell by adding SOARM MJCF deps — the install cell structure should be easy to extend.
- OpenVLA-OFT HuggingFace model ID: `openvla/openvla-oft-7b` (to confirm during research).

</code_context>

<specifics>
## Specific Ideas

- The restart cell should use `import os; os.kill(os.getpid(), 9)` or Colab's built-in restart button with a clear markdown cell above it saying "**STOP HERE — restart runtime, then continue from the next cell.**"
- Verification output should be visually scannable — consider a printed table like `ENV-01: ✓ PASS` rather than verbose logs.

</specifics>

<deferred>
## Deferred Ideas

- 4-bit quantization path for T4 (bitsandbytes) — deferred to a future optional cell or Phase 3 if T4 support is needed.
- Google Drive persistence of outputs — deferred; outputs/ directory in repo is sufficient for Phase 1.
- π0 (openpi) loading in Colab — deferred to Phase 3 (VLA Inference Loop).

</deferred>

---

*Phase: 1-Colab Environment Setup*
*Context gathered: 2026-07-08*
