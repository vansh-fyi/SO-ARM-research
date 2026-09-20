# SOARM Gripper Clamp Orientation — Debugging Handoff

> **Superseded 2026-09-20:** the assembly has now been registered directly to the
> corrected saved Coppelia model. See [COPPELIA_MUJOCO_ALIGNMENT.md](COPPELIA_MUJOCO_ALIGNMENT.md)
> for the fix, measured checks, and viewer command. The assumptions below about
> interchangeable object/STL frames, centroid correction, and trusted Euler
> composition were not a valid basis for conversion. Retained as debugging history.

**Date:** 2026-09-20
**Status:** UNRESOLVED — clamp/teeth orientation still visually wrong when inspected live in MuJoCo's interactive viewer, despite two fix attempts and cross-validated math. Handing off for a fresh pair of eyes / better 3D spatial reasoning.

**Git state:** The two files below have **uncommitted** working-tree changes on top of commit `296fee9` (gear-coupling fix, already committed and confirmed working). Do NOT assume the current file state is "shipped" — it's mid-debug.
- `LIBERO/libero/libero/assets/grippers/soarm_gripper.xml` (clamp `pos`/`quat` edited, uncommitted)
- `diagnostics/render_gripper_clamp.py` (camera framing tweaked, uncommitted)

---

## 1. What this is

`LIBERO/libero/libero/assets/grippers/soarm_gripper.xml` is the MuJoCo MJCF for the SOARM SO-101's parallel-jaw gripper (roboninecom aftermarket 84mm gripper, replacing the stock too-small jaw). It's a standalone MJCF loaded by `LIBERO/libero/libero/envs/grippers/soarm_gripper.py`'s `SoarmGripper` class (robosuite `GripperModel` pattern) and mounted onto the arm's wrist in `LIBERO/libero/libero/assets/robots/soarm101/robot.xml`.

Real mechanism (see reference photo the user provided — not included in this repo, but described): **one motor drives a central pinion gear**, which meshes with **two toothed racks** running horizontally through a frame, each rack ending in a paddle-shaped clamp jaw. The two jaws are mechanically slaved together — you cannot command them independently on real hardware.

## 2. Three separate problems identified this session

1. **Gear coupling — FIXED, committed, verified.** `gripper_left`/`gripper_right` were two fully independent MuJoCo `<position>` actuators with no physical constraint tying them together (only software — `SoarmGripper.format_action` — kept them in sync by always sending identical ctrl values). Added a MuJoCo `<equality><joint joint1="gripper_right" joint2="gripper_left" polycoef="0 1 0 0 0"/></equality>` constraint. Verified: deliberately mismatched qpos converges under the solver; driving only `gripper_left`'s actuator still drags `gripper_right` along. Committed as `296fee9`. **This part is done and confirmed correct — do not touch.**

2. **Base/frame orientation** — the `main_frame_visual` geom (mesh `soarm_main_frame`, the fixed housing that bolts to the wrist). User reported having to manually "flip 180° about X" when fixing the equivalent object in CoppeliaSim. **This was never actually broken on the MuJoCo side** — see §4 below, the existing `quat="0.5 0.5 0.5 0.5"` was cross-validated as correct against real CoppeliaSim data and left untouched.

3. **Clamp/teeth orientation — STILL UNRESOLVED.** The two clamp visual meshes (`soarm_clamp_left` = `clamp_1_visual.stl`, `soarm_clamp_right` = `clamp_2_visual.stl`) render with the gear-tooth rack detail in the wrong orientation. First reported as an "S-shape" interdigitation (the two clamp meshes crossing into each other's space) when viewed head-on in an offscreen render. This is the crux of this handoff.

## 3. Attempt #1 — blind quaternion guess (FAILED, reverted)

**What was tried:** Original quat for both `left_jaw_visual` and `right_jaw_visual` was `0 0.707107 0.707107 0` (identical for both — this is suspicious on its face, since the two meshes are separate left/right CAD parts). Guessed that "180° about local Z" (based on a verbal description of a CoppeliaSim fix: "rotated left clamp 180° about Z to mirror it") should be composed via quaternion multiply onto the LEFT clamp only, computed two candidate results (pre/post multiply — they coincided since q and -q are the same rotation):

```
new_quat = quat_mult([0, 0.707107, 0.707107, 0], [0, 0, 0, 1]) = [0, 0.707107, -0.707107, 0]
```

Applied to `left_jaw_visual` only (kept `right_jaw_visual` unchanged). **Result: catastrophically wrong.** The left clamp mesh flew off to a completely different location/orientation — confirmed via an 8-way azimuth orbit contact-sheet render (`diagnostics/outputs/orbit_contact_sheet.png`, still on disk) showing the left clamp as a small disconnected paddle floating away from the frame, not flush with it like the right clamp.

**Root cause of failure (probable):** The verbal "180° about Z" description from the user was almost certainly describing a rotation in **CoppeliaSim's own local-object frame conventions and Euler-angle semantics**, which do NOT map 1:1 onto "rotate the MuJoCo quat by 180° about local Z" without knowing exactly how CoppeliaSim's alpha/beta/gamma Euler angles compose (see §4 — turns out it's `R = Rz(gamma) @ Ry(beta) @ Rx(alpha)`, NOT a simple single-axis rotation applied naively). Also, only the LEFT clamp was touched — no principled reason was established for why RIGHT should stay unchanged while LEFT alone got an extra 180°.

**Reverted** back to the original `0 0.707107 0.707107 0` for left (matching right) before proceeding differently.

## 4. Attempt #2 — CoppeliaSim numeric data + cross-validated Euler convention (PARTIAL — looked right in 2 static renders, user says still wrong live)

### 4.1 Data collection

User fixed the gripper mechanism directly in CoppeliaSim (base flip, teeth mirror, wrist_roll parenting — all done with live visual feedback there, which is trustworthy). Rather than guess again, asked the user to screenshot CoppeliaSim's **Object/Item Position** and **Object/Item Rotation/Orientation** dialogs for each relevant object, with **"Relative to: Parent frame"** selected (not World) — since MuJoCo's `pos`/`quat` on a body/geom are always parent-relative.

Screenshots saved to:
- `/Users/hp/Desktop/Work/Repositories/SoARM-Research/orientation/position/*.png`
- `/Users/hp/Desktop/Work/Repositories/SoARM-Research/orientation/rotation/*.png`

**Raw data extracted** (all "Relative to: Parent frame", CoppeliaSim's alpha/beta/gamma in degrees):

| Object | Parent | Position [m] (x,y,z) | Orientation [deg] (α,β,γ) |
|---|---|---|---|
| `parallel_gripper_base_respondable` | `wrist_roll` | +0.01242, -0.00445, +0.01837 | 89.954, 0.001, 90.00 |
| `parallel_gripper_left` (joint) | `parallel_gripper_base_respondable` | +0.052, 0, 0 | 90.00, 0, 0 |
| `parallel_gripper_left_jaw_respondable` | `parallel_gripper_left` | +0.025, -0.006, +0.004 | -90.00, 0, 180.00 |
| `parallel_gripper_right` (joint) | `parallel_gripper_base_respondable` | +0.052, 0, 0 | -90.00, 0, 0 |
| `parallel_gripper_right_jaw_respondable` | `parallel_gripper_right` | +0.025, +0.006, +0.004 | 90.00, 0, 180.00 |
| `wrist_roll` | `wrist_link_respondable` | +0.0125, 0, +0.0187 | 179.999, 0, 87.211 |
| `wrist_link_respondable` | (arm chain) | — | 90.00, 0, -90.00 |
| `gear` (motor pinion visual) | `sts3215_03a_v1` | +0.045, 0, -0.005 | (not recorded) |

Note the left/right pairs are clean mirror images of each other (Y position sign flips, α sign flips) — consistent with a properly-fixed symmetric mechanism in CoppeliaSim.

### 4.2 Euler convention cross-validation (this part IS solid — reuse it)

CoppeliaSim's alpha/beta/gamma Euler convention was NOT assumed — it was **empirically determined and validated** against a piece of MuJoCo data already known to be correct (`main_frame_visual`'s existing `quat="0.5 0.5 0.5 0.5"`, which was never reported as buggy by the user — only the clamps were).

Tested three candidate compositions against `main_frame_visual`'s known-good rotation matrix:

```python
def Rx(deg): ...  # standard rotation matrices, degrees
def Ry(deg): ...
def Rz(deg): ...

R_mujoco_known = quat_to_R([0.5, 0.5, 0.5, 0.5])
# = [[0,0,1],[1,0,0],[0,1,0]]  (cyclic permutation)

a, b, g = 89.954, 0.001, 90.00  # base_respondable rel wrist_roll

Rx(a)@Ry(b)@Rz(g)   # max diff from known-good: 1.0   -- WRONG
Rz(g)@Ry(b)@Rx(a)   # max diff from known-good: 0.001 -- MATCH (just float rounding)
```

**Result: `R = Rz(gamma) @ Ry(beta) @ Rx(alpha)` reproduces MuJoCo's already-correct `main_frame_visual` orientation to within 0.001 (rounding noise from 89.954≈90, 0.001≈0).** This is strong evidence that:
(a) this is the correct CoppeliaSim Euler→matrix convention, AND
(b) CoppeliaSim's `wrist_roll` frame and MuJoCo's `right_gripper` body frame (the mount point on the arm) are the **same reference frame** — meaning position/orientation values expressed relative to `wrist_roll` in CoppeliaSim can be composed and used directly as MuJoCo `right_gripper`-relative values, no extra basis-change needed.

**This convention (`Rz(gamma) @ Ry(beta) @ Rx(alpha)`, degrees, composed left-to-right through the parent chain) should be trusted and reused — it was validated against real known-good data, not guessed.**

### 4.3 Computed clamp orientation

Composed the full chain for each jaw (base → joint → jaw_respondable):

```python
R_base = R_from_euler(89.954, 0.001, 90.00)       # base_respondable rel wrist_roll
R_pgl  = R_from_euler(90.0, 0.0, 0.0)              # parallel_gripper_left (joint) rel base
R_ljr  = R_from_euler(-90.0, 0.0, 180.0)           # left_jaw_respondable rel joint
R_pgr  = R_from_euler(-90.0, 0.0, 0.0)             # parallel_gripper_right (joint) rel base
R_rjr  = R_from_euler(90.0, 0.0, 180.0)            # right_jaw_respondable rel joint

R_left_full  = R_base @ R_pgl @ R_ljr
R_right_full = R_base @ R_pgr @ R_rjr
```

**Surprising result: both jaws compose to the SAME quaternion** (`[-0.499804, -0.500196, 0.500205, 0.499795]` in w,x,y,z, essentially `[-0.5,-0.5,0.5,0.5]` up to rounding) — not mirror images of each other via rotation. Interpretation at the time: since `soarm_clamp_left`/`soarm_clamp_right` are **separate STL files** (already distinct/mirrored CAD parts, not the same mesh reused), applying the *same* rotation formula to two already-mirrored meshes should produce the correct mirrored visual result — the mirroring lives in the mesh geometry, not the transform. This is a **plausible but UNVERIFIED assumption** — worth double-checking by inspecting whether `clamp_1_visual.stl` and `clamp_2_visual.stl` really are mirror-image meshes (e.g. compare vertex bounding boxes / check if one is a reflected copy of the other) rather than assuming it.

This new computed quat (`-0.499804 -0.500196 0.500205 0.499795`) is **substantially different** from the original `0 0.707107 0.707107 0` — not just a sign/relative tweak.

### 4.4 Position re-centering

Since changing `quat` while keeping the old `pos` un-recomputed would misplace the mesh (rotating around a local origin that isn't the mesh's visual centroid — this is exactly how Attempt #1 failed), recomputed `pos` to keep the mesh's actual STL centroid anchored at the same **world-space point** it occupied under the OLD quat+pos:

```python
# mesh_local_centroid read directly from the compiled MuJoCo model's mesh_vert buffer
centroid_left  = [0.00105,  0.00087, -0.00085]   # soarm_clamp_left, local frame
centroid_right = [-0.00105, -0.00086, -0.00085]  # soarm_clamp_right, local frame

# new_pos = R_old @ centroid + old_pos - R_new @ centroid
```

Results applied to the file (current uncommitted state):

```xml
<geom ... mesh="soarm_clamp_left"  pos="0.13702 -0.02139 0.14545"
      quat="-0.499804 -0.500196 0.500205 0.499795" name="left_jaw_visual"/>
<geom ... mesh="soarm_clamp_right" pos="0.13536 0.00597 0.16006"
      quat="-0.499804 -0.500196 0.500205 0.499795" name="right_jaw_visual"/>
```

(`main_frame_visual` was left completely untouched — its existing `pos="-0.015 -0.064 -0.02975" quat="0.5 0.5 0.5 0.5"` was the validated-correct reference used in §4.2, not something needing a fix.)

### 4.5 Verification performed (and its limits)

- XML parses (`xml.etree.ElementTree.parse` — no syntax errors).
- `pytest scripts/test_verify_urdf.py LIBERO/libero/libero/envs/test_camera_config.py LIBERO/libero/libero/datasets/test_collector.py` — **19/19 pass** (these tests don't check visual mesh orientation/correctness at all, only structural/kinematic things — a passing test suite here says nothing about whether the clamps *look* right).
- Gear-coupling constraint re-verified still works after the geom edits (unrelated systems, expected to be unaffected, confirmed anyway).
- **Static offscreen renders** via `diagnostics/render_gripper_clamp.py` (MuJoCo `Renderer`, fixed camera positions) — both a 3/4-angle view and a head-on "frontal" view (camera added this session specifically for this debugging) — looked significantly better than the Attempt #1 disaster and the original S-shape: two separate toothed racks visible, paddle jaws at the outer edges, no visible interdigitation, roughly matching the layout of a reference photo the user provided of the real gripper.
- **User then opened the live MuJoCo interactive viewer** (`python -m mujoco.viewer --mjcf=...soarm_gripper.xml`) and, looking at it from angles the two fixed static camera positions didn't cover, said **"this is wrong"** — no further detail captured before the user asked for this handoff instead of continuing to iterate blind.

**The static renders are NOT proof of correctness** — they only check 2 fixed angles. The live viewer is the ground truth here and it's still failing from at least one angle not covered by the render script.

## 5. Open questions / what's actually needed next

1. **What specifically looks wrong in the live viewer?** This was never captured — the user cut off the debugging loop here. Whoever picks this up should get the user to describe or screenshot the live view from the angle(s) that look wrong, ideally with the real hardware or the reference photo in the same frame for direct comparison.

2. **Is the "both jaws share one quat" assumption (§4.3) actually correct?** It's plausible (separate mirror-image STLs) but was never independently verified by inspecting the STL geometry itself. Worth checking: load both `clamp_1_visual.stl` and `clamp_2_visual.stl` (or the MuJoCo-compiled `soarm_clamp_left`/`soarm_clamp_right` mesh vertex buffers) and check whether one is genuinely a mirror (reflection, determinant -1 relationship) of the other, or whether they're actually the same shape requiring an actual relative rotation between them after all.

3. **Was the CoppeliaSim source data itself fully correct?** The position/rotation numbers were read at a specific moment in the user's CoppeliaSim session. If the user was still mid-adjustment when the screenshots were taken (recall: earlier in the session, an "Object is model" duplicate-flag bug was found and fixed on `parallel_gripper_base_respondable`, and there was an unresolved floor-alignment/bounding-box quirk that was never fully root-caused — see the main conversation history for that whole sub-thread) — it's possible the exported numbers don't represent a fully-converged-correct state.

4. **Double check the parent-frame assumption for `wrist_roll` vs `right_gripper`.** §4.2's validation is strong (0.001 match) but was only checked for ONE object (`main_frame_visual`). It assumes CoppeliaSim's `wrist_roll` object frame and MuJoCo's `right_gripper` body frame are identical reference frames — true for rotation (validated), but note `main_frame_visual`'s POSITION values were never cross-checked this way (CoppeliaSim gives `base_respondable` pos rel. wrist_roll as `(0.01242, -0.00445, 0.01837)`, which does NOT numerically match MuJoCo's `main_frame_visual` pos `(-0.015, -0.064, -0.02975)` — these are expected to differ since one is the frame housing's own respondable-object origin and the other is a D-04-tuned mesh-recentering offset, but this discrepancy was never explicitly reasoned through, only assumed benign).

5. **Consider whether the URDF export route (dead-ended earlier — see below) could be revived** with different export settings, or whether directly reading numbers from CoppeliaSim (as done in §4.1) remains the only viable path.

## 6. Dead end already ruled out — don't retry as-is

**CoppeliaSim's built-in URDF exporter cannot be used to extract corrected gripper geometry.** Tried once (`So-101-coppelia/` export, with "Reset joints" checked, "Set shape origin at joint location" and "Make red cubes from dummies" both unchecked): the exporter **flattens the entire gripper subtree** (wrist_roll joint, gripper base, both jaw joints) into a single static merged mesh (`parallel_gripper_base_visual`) attached directly to `wrist_link_respondable` — the `wrist_roll`, `parallel_gripper_left`, `parallel_gripper_right` joints are **completely absent** from the exported URDF, and there's no per-part transform data for the individual clamps. This is a limitation of the exporter itself when handling nested "model" sub-assemblies (likely related to the same `parallel_gripper_base_respondable` "Object is model" flag bug found and fixed earlier — the exporter may still choke on nested-model boundaries even after that specific duplicate flag was removed). Manually reading Object Properties dialogs (§4.1) was the working alternative.

## 7. Key files

- `LIBERO/libero/libero/assets/grippers/soarm_gripper.xml` — the file being edited (gripper MJCF)
- `LIBERO/libero/libero/envs/grippers/soarm_gripper.py` — `SoarmGripper` class, `format_action` (gripper action convention, unrelated to this bug, already correct)
- `diagnostics/render_gripper_clamp.py` — offscreen render script, has both a 3/4-angle camera (`CAM_*`) and a head-on "frontal" camera (`CAM_*_FRONTAL`, added this session) producing `gripper_clamp_{open,closed}[_frontal].png` in `diagnostics/outputs/`
- `diagnostics/outputs/orbit_contact_sheet.png` — 8-way azimuth orbit render from Attempt #1's failure diagnosis, still on disk, shows the left clamp flying off
- `/Users/hp/Desktop/Work/Repositories/SoARM-Research/orientation/position/*.png` — CoppeliaSim Position dialog screenshots (parent-frame-relative), full arm + gripper
- `/Users/hp/Desktop/Work/Repositories/SoARM-Research/orientation/rotation/*.png` — CoppeliaSim Orientation dialog screenshots (parent-frame-relative), full arm + gripper
- `/Users/hp/Desktop/Work/Repositories/SoARM-Research/So-101-coppelia/` — the dead-end URDF export (§6), kept for reference, do not treat its gripper section as trustworthy

## 8. Recommended path forward

1. Get the user to identify, precisely, what's wrong in the live MuJoCo viewer (screenshot + description) — this is the missing piece that would make this tractable.
2. Independently verify whether `clamp_1_visual.stl`/`clamp_2_visual.stl` are true mirror-image meshes or the same mesh (§5.2) — this determines whether "same quat for both jaws" is even the right shape of solution.
3. If the CoppeliaSim-derived numbers turn out to be right and it's a mesh-mirroring assumption problem, the fix is likely a per-mesh (not per-instance) correction — e.g. one clamp geom needs an additional reflection/rotation the other doesn't, contradicting §4.3's finding, in which case redo the composition allowing for that asymmetry.
4. If the CoppeliaSim numbers themselves are suspect (open question #3), the cleanest fix is to have the user re-verify/re-confirm the CoppeliaSim scene is fully settled (not mid-edit) before re-reading the Object Properties values, rather than trying to fix this from historical screenshots.
