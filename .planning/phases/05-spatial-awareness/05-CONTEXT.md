# Phase 5: Spatial Awareness - Context

**Gathered:** 2026-08-09
**Status:** Ready for planning

<domain>
## Phase Boundary

Extending the existing SOARM LIBERO environment (wrist `robot0_eye_in_hand` + overhead `agentview` cameras already registered in `env_wrapper.py`, currently `camera_depths=False`) so that: (1) both camera views actually reach the VLA model during inference (not just the wrist view, as today), (2) depth buffers from those same cameras are extracted and back-projected into 3D object positions using camera intrinsics/extrinsics (a real-robot-faithful perception pipeline, not privileged sim-state reads), and (3) at least 3 new BDDL tasks use spatial-language prompts (left/right, near/far, between) with success predicates that correctly score them. Fine-tuning (Phase 6) and real-hardware transfer (PHYS-01..03, v2 backlog) are explicitly out of scope — this phase produces the perception/benchmarking capability those later efforts will build on.

</domain>

<decisions>
## Implementation Decisions

### Multi-Camera → VLA Wiring (SPAT-01, SPAT-02)
- **D-01:** Both wrist (`robot0_eye_in_hand`) and overhead (`agentview`) views must actually reach the VLA model as input during inference — this is a literal reading of SPAT-02, not just "views are captured/logged." `LIBERO/libero/libero/vla/eval_loop.py` currently only builds `images = {"eye_in_hand": obs[camera_name]}`; this must be extended to include the overhead view too.
- **D-02:** How each backend (pi0 in `pi0_backend.py`, OFT in `oft_backend.py`) actually consumes two images is **Claude's discretion**, decided per-backend during planning/research: use the model's native multi-view input if the checkpoint supports it; if a specific checkpoint genuinely only accepts one image, fall back to resize+concat/tile so both views still reach the model rather than silently dropping the overhead view. Research each backend's actual trained input format before choosing.

### Depth & XYZ Pipeline (SPAT-03, SPAT-04)
- **D-03 (real-robot-faithful, not privileged-state shortcut):** Object XYZ positions are computed by back-projecting the extracted depth buffer using camera intrinsics/extrinsics — i.e. the same computation a real depth camera would require. This was chosen deliberately over the cheaper "read exact position from `sim.data`" shortcut, because the user is planning to build the physical SOARM arm soon and wants this pipeline to be the code that transfers to real hardware later (see `<deferred>` below).
- **D-04:** The depth→XYZ pipeline must be validated against MuJoCo's privileged sim-state ground truth as a correctness check (unit/integration tests comparing depth-derived XYZ to `sim.data` ground truth) — this catches camera-calibration or projection-math bugs. Sim-state ground truth is used internally for this validation only; it is not the pipeline's primary output.
- **D-05 (locked, important — do not confuse with D-03):** Despite D-03/D-04 using depth-derived XYZ as the "real" perception pipeline, **task success/failure (pass/fail) scoring in the BDDL predicates uses MuJoCo's privileged sim-state ground truth**, not the depth-derived estimate. Rationale: this matches how every other existing LIBERO task is already scored, keeps benchmark numbers reliable and noise-free, and decouples predicate correctness from depth-pipeline accuracy while that pipeline is still new. The depth→XYZ pipeline still runs and produces its own estimates (useful for perception/benchmarking and for D-04's validation), it just isn't the referee for pass/fail.

### Spatial BDDL Task Design (SPAT-05)
- **D-06:** Whether to reuse Phase 4's already-validated scene/objects (`put_the_cream_cheese_in_the_bowl` task) or author a new scene is **Claude's discretion** — pick whichever is fastest to validate correctly, respecting Phase 4's locked embodiment constraints: objects must be graspable by the 84mm gripper (≲84mm), all object-init and place-targets must sit within the arm's ~0.45m reach, and nothing may sit on the base's forward centerline (the arm's forward sweep corridor — placing anything there causes a collision, per the 04-02 investigation). Apply the same empirical embodiment-safety checks Phase 4 established (verify via `actuator_force`/`qfrc_bias` + `sim.data.contact`, not assumptions) before locking in any new task layout.
- **D-07:** The 3+ spatial tasks should cover **left/right, near/far, and between** relations — the broadest coverage of common spatial relations, matching LIBERO's own `libero_spatial` suite conventions and giving the widest benchmark surface for "spatial understanding quality" (a stated project goal in PROJECT.md).

### Spatial Success Predicates
- **D-08:** Spatial relation checks use **threshold/tolerance bands**, not exact-coordinate comparison — matches LIBERO's own existing predicate convention (its predicates already use tolerance, not exact geometry), avoiding flaky pass/fail from simulation noise. Exact margin size is Claude's discretion, informed by research into LIBERO's existing `AtomicPredicateFn`-style implementations during planning.
- **D-09:** Ambiguous/borderline spatial relations (e.g. an object placed almost exactly "between" two others, or right on a left/right boundary) should be **avoided at task-design time** — author initial states and success zones with enough margin that ambiguous cases can't naturally arise during normal task completion. No tie-breaking predicate logic needed; this keeps predicates simple and reliable (consistent with D-08's threshold rationale).

### Claude's Discretion (summary)
- D-02: Per-backend strategy for consuming two camera views (native multi-view vs. concat/tile fallback).
- D-06: Reuse Phase 4's scene/objects vs. author a new one for the spatial tasks.
- D-08: Exact tolerance/margin size for spatial predicates.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements & Roadmap
- `.planning/REQUIREMENTS.md` §Spatial Awareness — SPAT-01 through SPAT-05 are this phase's requirements.
- `.planning/ROADMAP.md` §Phase 5: Spatial Awareness — success criteria (multi-camera VLA input, depth+XYZ structured data, 3+ spatial BDDL tasks with correct predicates).

### Existing Camera Plumbing (already built, this phase extends it)
- `LIBERO/libero/libero/envs/env_wrapper.py` — `camera_names=["agentview", "robot0_eye_in_hand"]`, `camera_heights=128`, `camera_depths=False` (default) already registered; SPAT-03 flips `camera_depths` on and extracts the buffer.

### VLA Backend Interface (this phase's multi-camera wiring target)
- `LIBERO/libero/libero/vla/interface.py` — documents `predict(images: dict[str, Image.Image], language: str)` as the shared backend contract; images is already a dict of named views by design (D-01 in Phase 3), just not populated with more than one view yet.
- `LIBERO/libero/libero/vla/eval_loop.py` — currently builds `images = {"eye_in_hand": obs[camera_name]}` only; D-01/D-02 extend this to include the overhead view.
- `LIBERO/libero/libero/vla/pi0_backend.py`, `LIBERO/libero/libero/vla/oft_backend.py` — current single-image (`images["eye_in_hand"]`) consumption per backend; D-02's per-backend research target.

### Prior Phase Context (embodiment constraints spatial tasks must respect)
- `.planning/phases/04-dataset-collection/04-CONTEXT.md` — D-07/D-08 amendment: 84mm gripper ceiling, sub-84mm graspable objects, ~0.45m reach envelope, forward-corridor collision risk. Directly governs D-06/D-07 spatial task authoring here.
- `.planning/phases/04-dataset-collection/04-02-SUMMARY.md` — the embodiment-limit investigation history (jaw size vs. misdiagnosed "torque saturation" vs. actual forward-corridor collision) — the methodology (check `actuator_force`/`qfrc_bias` + `sim.data.contact` before concluding a hardware limit) applies to any new task layout validation in this phase too.
- `.planning/phases/02-soarm-robot-integration/02-CONTEXT.md` — SOARM `Soarm101` robot class registration and tuned base pose/reach envelope.
- STATE.md "Accumulated Context" — flags "Spatial VLA input representation (multi-camera RGB vs RGB+depth vs auxiliary 3D annotations) is an open question — study SpatialVLA, VEGA, cVLA before committing." This remains an open research question for `gsd-phase-researcher`, not resolved by this discussion — D-01/D-02 lock the mechanism (both views reach the model) but not necessarily the final architectural pattern; research should confirm the best-practice approach for pi0/OFT specifically.

### Codebase Maps
- `.planning/codebase/STACK.md` — pinned dependency versions (robosuite 1.4.0, MuJoCo 2.3.7) relevant to depth-buffer extraction APIs.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `env_wrapper.py`'s existing `camera_names`/`camera_depths` kwargs — depth extraction is a config flip plus buffer plumbing, not new camera infrastructure.
- `VLABackend`-satisfying `predict(images, language)` interface (Phase 3) — already dict-shaped for multi-view; this phase populates it fully rather than redesigning it.

### Established Patterns
- Real importable Python modules over notebook-only code (Phase 2/3/4 convention) — new depth→XYZ and predicate code should follow this, likely in a new module under `LIBERO/libero/libero/` (exact location is a planning/research decision, not locked here).
- `MUJOCO_GL` must be set before any MuJoCo import.
- Embodiment-safety validation via direct actuator/contact telemetry (established in Phase 4's 04-02 investigation) — reuse this methodology for any new spatial task layout, don't assume reach/collision safety from measurements alone.

### Integration Points
- `eval_loop.py` is the single integration point where camera observations become the `images` dict passed to VLA backends — D-01's extension happens here.
- New BDDL files for spatial tasks go in `LIBERO/libero/libero/bddl_files/` following existing `libero_spatial` conventions; predicates likely extend or reuse LIBERO's existing predicate library (research target).

</code_context>

<specifics>
## Specific Ideas

- User is planning to build the physical SOARM arm soon — this directly motivated D-03 (depth-derived XYZ over sim-state shortcut) so the perception pipeline built now has real-hardware relevance later, not just simulation convenience.
- User asked whether a 3rd camera angle (e.g. front/side view) would improve spatial understanding — informational answer given (yes, generally, for occlusion robustness) but SPAT-01 already locks "at least 2 camera views (wrist + overhead)"; a 3rd camera is new scope, not decided here (see Deferred).

</specifics>

<deferred>
## Deferred Ideas

- **Real-hardware depth camera transfer** — when the physical SOARM arm is built (PHYS-01..03, v2 backlog), this phase's depth→XYZ pipeline (D-03/D-04) is the code that will need a real depth camera selection, physical mounting, and calibration against the real sensor. No work happens on this now; flagged so the thread stays visible for whichever phase picks up PHYS-01..03.
- **Third camera angle for occlusion robustness** — user asked about adding a front/side view alongside wrist+overhead to reduce occlusion; informative answer given, but not adopted since SPAT-01 already locks the 2-camera scope. Could be proposed as its own future phase/task if occlusion becomes a measured problem.

</deferred>

---

*Phase: 5-Spatial Awareness*
*Context gathered: 2026-08-09*
