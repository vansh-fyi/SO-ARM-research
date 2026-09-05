# Phase 7: Camera & Depth Perception - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-09-05
**Phase:** 7-Camera & Depth Perception
**Areas discussed:** Camera recalibration basis, Depth camera scope, Camera match verification, Overhead camera mount style, Wrist camera angle mismatch, Real mount angle detail

---

## Camera recalibration basis (given real specs incomplete)

| Option | Description | Selected |
|--------|-------------|----------|
| Best-estimate now, revisit later | Use AR0144 datasheet FOV + reasonable overhead placement now; re-tune once real mount finalized | |
| Block on real measurement first | Pause camera recalibration sub-goal until physical mount built and measured | |
| Use existing sim defaults, skip accuracy | Keep current pos/quat, treat CAM-01 as satisfied by documenting intent | |
| Free text | User redirected the question | ✓ |

**User's choice:** Free text — "Can you dig and find, a stand for depth I can 3d print? we will have exact placement! Also the camera angle of IMX335 is wrong in simulation does not match real cam placement."
**Notes:** User wants a 3D-printable overhead camera stand researched (to get exact real placement for recalibration), and flagged a second, previously-undiscussed issue: the wrist (eye_in_hand/IMX335) camera's sim angle doesn't match its real mounting. This expanded the discussion into two follow-up areas below. Research found no pre-made mount for the housed AR0144 stereo module exists; landed on custom-cradle + generic 1/4-20 tripod hardware, consistent with `diagnostics/PARTS_LIST.md`'s existing "Not yet resolved" plan.

---

## Depth camera scope

| Option | Description | Selected |
|--------|-------------|----------|
| agentview only | Mirrors real hardware — only overhead AR0144 has stereo depth | ✓ |
| Both agentview and eye_in_hand | Persist depth for both sim cameras despite no real depth on wrist cam | |

**User's choice:** agentview only
**Notes:** No further discussion needed — matches real hardware capability exactly.

---

## Camera match verification (Success Criteria #1)

| Option | Description | Selected |
|--------|-------------|----------|
| Defer to a real photo once hardware track catches up | Sign off on estimate now; schedule real photo comparison as a follow-up | ✓ |
| Approximate comparison using rough manual placement now | Temporarily mount the camera today just to get one reference photo | |
| Skip visual comparison, verify via numeric params only | No photo comparison performed in this phase at all | |

**User's choice:** Defer to a real photo once hardware track catches up
**Notes:** Consistent with the parallel physical-hardware track already running outside the phase/ROADMAP structure per PROJECT.md.

---

## Overhead camera mount style

| Option | Description | Selected |
|--------|-------------|----------|
| Fixed frame/arm over the workspace | Rigid printed arm/frame bolted to desk edge, static and repeatable | ✓ (with modification) |
| Desk clamp + printed cradle + tripod ball head | Clamp-on mount with adjustable ball-head/gooseneck | |
| Tripod stand + printed cradle | Standard photo tripod holding the printed cradle | |

**User's choice:** Fixed frame/arm bolted to desk/table edge — user specified "with clamps like the bots" (i.e., a clamp mechanism similar to how the robot base itself is mounted)
**Notes:** Static/repeatable placement was the deciding factor, matching PARTS_LIST.md's "static, aimed at workspace" description of the real camera's intended use.

---

## Wrist camera (IMX335) angle mismatch — is it in scope?

| Option | Description | Selected |
|--------|-------------|----------|
| Yes, fix it in Phase 7 alongside agentview | Correct eye_in_hand fovy/quat to match real mount; user describes real angle | ✓ |
| Flag it, defer the fix to a later phase | Note as backlog item, keep Phase 7 scoped to agentview + depth only | |

**User's choice:** Yes, fix it in Phase 7
**Notes:** This was a scope addition surfaced mid-discussion, not originally worded into CAM-01 (which only mentions "agentview/front camera"). Accepted into scope because it's the same class of sim-to-real camera correction the phase already exists to do.

---

## Real wrist camera mount angle detail

| Option | Description | Selected |
|--------|-------------|----------|
| Points straight forward, sim has it tilted | Real camera aimed along gripper's forward/grasp axis, sim currently angled off-axis | |
| Tilted downward toward grasp point, sim points too far forward | Real camera looks down/inward toward where fingers meet; sim's current angle aims too far ahead | ✓ |
| Free text | User describes it differently | |

**User's choice:** Tilted downward toward grasp point, sim points too far forward
**Notes:** User initially offered to send photos rather than describe in words; sent 6 reference photos (`progress-documentation/images/20260827_*.jpg`) plus one live camera capture (`control/outputs/step5_test/camera_0.png`). Photos visually confirm the real IMX335 bracket tilts the lens down/inward toward the gripper's grasp point. The camera_0.png capture was noted as a low-confidence framing reference since it was taken with the arm held in a vertical test pose, not an operating grasp angle.

---

## Claude's Discretion

- Exact new `fovy`/`quat` numeric values for both `agentview` and `eye_in_hand` cameras — left for research/planning to derive
- Depth storage dtype/precision in HDF5 (float32 vs. quantized)
- RLDS depth feature type (confirmed via codebase scout: plain `Tensor`, not `Image`, per TFDS/OXE convention)
- Physical stand cradle design specifics (wall thickness, clamp mechanism details)

## Deferred Ideas

- Physical overhead camera mount fabrication and the literal side-by-side real-vs-sim photo comparison (Success Criteria #1) — deferred to a follow-up once the parallel hardware track builds the mount; not a Phase 7 blocker.
