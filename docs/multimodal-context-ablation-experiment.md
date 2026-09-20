# Multimodal Context Ablation Experiment

> Historical experiment design, retained for research context. Its model paths,
> joint limits, calibration snapshot and proposed control workflow are not the
> current hardware execution contract. For the accepted simulation model, see
> [LIBERO/SOARM_MODEL.md](../LIBERO/SOARM_MODEL.md). Phase 11 must independently
> validate hardware calibration, action mapping and safety before physical motion.

### SO-ARM101 · Visual Language Action (VLA) Demo

> **Goal:** Understand how each modality added to a multimodal model's prompt
> changes the quality of the action signals it produces for the SO-ARM101.
> We ablate context incrementally — one layer at a time — so we can measure the
> **marginal contribution** of every input channel before wiring them all up.

---

## Overview

We are probing a frontier multimodal model (tested via **Codex** and **Claude Code**)
to see whether — given progressively richer context about the robot and its scene —
it can produce structured output that maps to real servo commands on the physical arm.

The experiment has **two phases per run**:

1. **Raw phase** — send the prompt exactly as written; record whatever the model
   returns (prose, JSON, pseudocode, anything).
2. **VLA-format phase** — append a single follow-up query that instructs the
   model to reformat its output as a Visual Language Action model would
   (vector/array of joint targets), so the output can be fed directly to the
   serial bus.

Each "stage" below adds exactly one modality. All prior modalities carry forward.

---

## Hardware & Files Referenced

| Input | Source in repo |
|-------|---------------|
| Wrist RGB camera | OAK-D / wrist mount on the follower arm |
| Stereo Depth camera | Stereo rig; calibration at `diagnostics/stereo_calibration.json` |
| Joint positions | Live read via LeRobot `SO101Follower.get_observation()` (keys: `shoulder_pan.pos`, `shoulder_lift.pos`, `elbow_flex.pos`, `wrist_flex.pos`, `wrist_roll.pos`, `gripper.pos`) |
| STL meshes | `coppelia/meshes/` — per-link 3-D geometry |
| URDF | `coppelia/soarm101.urdf` — kinematics, joint limits, inertia |
| MuJoCo XML | `LIBERO/libero/libero/assets/robots/soarm101/robot.xml` — 3-D pose + actuator definitions |
| Calibration (arm 02) | `~/.cache/huggingface/lerobot/calibration/robots/so_follower/soarm_follower_02.json` |
| Calibration (arm 02) | `~/.cache/huggingface/lerobot/calibration/robots/so_follower/soarm_follower_02.json` |

### Servo Calibration Snapshot (arm 02 — current active follower)

The calibration below was produced by `lerobot-calibrate`. The fields that matter
for the experiment are **`range_min`** and **`range_max`** — these are the *practical*
hardware limits in raw 12-bit ticks (0–4095) for each STS3215 servo, after
`homing_offset` is applied by the bus driver. A model that understands these should
not command positions outside the safe window.

```json
{
  "shoulder_pan":  { "id": 1, "drive_mode": 0, "homing_offset": -1653, "range_min": 1269, "range_max": 2869 },
  "shoulder_lift": { "id": 2, "drive_mode": 0, "homing_offset": -1992, "range_min":  904, "range_max": 3349 },
  "elbow_flex":    { "id": 3, "drive_mode": 0, "homing_offset":  1518, "range_min":  832, "range_max": 3044 },
  "wrist_flex":    { "id": 4, "drive_mode": 0, "homing_offset":  1349, "range_min":  899, "range_max": 3220 },
  "wrist_roll":    { "id": 5, "drive_mode": 0, "homing_offset": -2033, "range_min":    0, "range_max": 4095 },
  "gripper":       { "id": 6, "drive_mode": 1, "homing_offset":  -197, "range_min":   48, "range_max": 3637 }
}
```

> **Note on gripper:** The gripper's raw `range_min`/`range_max` look near-full
> encoder range (1–4000). This is a known artifact of the gripper's travel
> straddling the 0/4095 rollover boundary before recalibration (see
> `control/recalibrate_gripper.py`).
> For the experiment, treat the gripper as a 0–100% normalized value as LeRobot
> itself does (see `keyboard_joint_control.py`, Y/H keys = 5% steps).

### Joint Limits from URDF (in radians — simulator ground truth)

| Joint | Min (rad) | Max (rad) | Type |
|-------|----------:|----------:|------|
| `shoulder_pan` | −1.9199 | +1.9199 | revolute |
| `shoulder_lift` | −1.7453 | +1.7453 | revolute |
| `elbow_flex` | −1.6900 | +1.6900 | revolute |
| `wrist_flex` | −1.6581 | +1.6581 | revolute |
| `wrist_roll` | −2.7438 | +2.8412 | revolute |
| `gripper_left` | 0.000 | 0.042 m | prismatic |
| `gripper_right` | 0.000 | 0.042 m | prismatic |

---

## Experiment Stages

Each stage is a **superset** of the previous one. Run them in order.

---

### Stage 1 — Text Prompt + Wrist Camera Only

**Inputs given to the model:**
- Text task prompt (see below)
- Single wrist RGB image (frame from the wrist-mounted camera)

**What we expect:**
The model has a first-person view of what the gripper is looking at, plus the
intent. It has no idea where the joints currently are, no depth, no kinematic
structure. We expect vague, high-level prose ("move the arm toward the object")
or qualitatively wrong joint directions.

**What we measure:**
- Does it attempt to output joint values at all?
- Are the implied directions reasonable given the visual?
- Does it hallucinate joint names that do not exist?

---

### Stage 2 — Text + Wrist Camera + Stereo Depth

**Inputs given to the model:**
- Text task prompt
- Wrist RGB image
- Stereo depth frame or point cloud snippet (or rectified left/right pair)

**What we expect:**
Adding depth lets the model estimate object distance and relative 3-D position.
We expect more confident spatial reasoning ("the object is ~15 cm ahead and
5 cm to the left") and potentially more calibrated reach estimates.

**What we measure:**
- Does the model use depth cues to qualify its motion plan?
- Does spatial language in the output improve (distance estimates, approach vectors)?
- Does it still ignore joint limits?

---

### Stage 3 — Text + Wrist + Stereo + Current Joint Positions

**Inputs given to the model:**
- Text task prompt
- Wrist RGB image
- Stereo depth
- Current joint state vector (6 values: shoulder_pan, shoulder_lift,
  elbow_flex, wrist_flex, wrist_roll, gripper — in degrees or normalized %)

**What we expect:**
Now the model knows where the arm *currently is*. This should ground abstract
"move forward" plans into concrete delta values. We expect the model to start
suggesting absolute target positions or deltas per joint.

**What we measure:**
- Does the model reason from current state to target state?
- Do suggested joint deltas respect rough physical plausibility?
- Does knowing the current state reduce hallucinated out-of-range values?

---

### Stage 4 — Text + Wrist + Stereo + Joints + URDF + XML + Calibration

**Inputs given to the model:**
- Text task prompt
- Wrist RGB image
- Stereo depth
- Current joint state vector
- URDF snippet (joint names, axes, hard limits in radians)
- MuJoCo XML body positions (link origins in 3-D space)
- LeRobot calibration JSON (practical tick-level limits for each servo)

**What we expect:**
This is the full-context run. The model has the complete kinematic description
of the robot. It should now be able to reason about reachability, avoid joint
limit violations, and produce output that is syntactically and semantically
close to what a real VLA model emits. We expect structured joint-target vectors
that are within both the URDF radian limits and the hardware tick limits.

**What we measure:**
- Are all joint names from the URDF present and correctly spelled?
- Are values within URDF `lower`/`upper` bounds?
- Are inferred tick values within `range_min`/`range_max` from calibration?
- Does the model describe a feasible end-effector trajectory?

---

## Text Prompts

Use these prompts verbatim across all stages so results are comparable.

### Task A — Open Gripper and Hold (Primary Prompt)

```
Open the gripper fully, hold it steady for 2 seconds, then close it halfway.
```

### Task B — Reach and Touch

```
Reach forward and touch the yellow marker lying flat on the table.
```

### Task C — Lift Cube

```
A cube is placed in front of the arm, lift it up.
```

### Task D — Pick and Place

```
Pick up the red cube on the table and place it in the white bowl to its right.
```

> **Why short prompts?** We want to test the model's *spatial reasoning and
> kinematic knowledge*, not its instruction-following on complex text. Short,
> unambiguous tasks isolate the multimodal understanding signal.

---

## Phase 2: VLA-Format Follow-Up Query

> **Research note:** We pre-researched what real VLA models output so we can
> give the model an exact schema to conform to, rather than asking it to guess.
> The format below is derived from **OpenVLA** / **RT-2** (discrete-tokenized,
> 7-DoF end-effector delta) and **π₀** (continuous action chunk, 50 Hz,
> joint-space) — adapted to SO-ARM101's 6-joint architecture.

After running each stage with the raw prompt above, append **exactly** this
follow-up in the same conversation context:

---

```
Reformat your previous answer as a VLA action output. Use this exact JSON
schema — do not deviate from field names or units:

{
  "action_space": "joint_position",
  "representation": "absolute",
  "units": "normalized",
  "normalization_range": [-1.0, 1.0],
  "note": "Each value is normalized so that -1.0 = joint range_min (hardware limit) and +1.0 = joint range_max. 0.0 = neutral / home pose.",
  "joints": {
    "shoulder_pan":  <float in [-1.0, 1.0]>,
    "shoulder_lift": <float in [-1.0, 1.0]>,
    "elbow_flex":    <float in [-1.0, 1.0]>,
    "wrist_flex":    <float in [-1.0, 1.0]>,
    "wrist_roll":    <float in [-1.0, 1.0]>,
    "gripper":       <float in [0.0, 1.0]>
  },
  "action_chunk": [
    { same joints object for timestep t+0 },
    { same joints object for timestep t+1 },
    { same joints object for timestep t+2 }
  ],
  "reasoning": "<one sentence explaining why these values achieve the task>"
}

Hardware reference for normalization:
  shoulder_pan:  range_min=1269 ticks → -1.0,  range_max=2869 ticks → +1.0
  shoulder_lift: range_min=904  ticks → -1.0,  range_max=3349 ticks → +1.0
  elbow_flex:    range_min=832  ticks → -1.0,  range_max=3044 ticks → +1.0
  wrist_flex:    range_min=899  ticks → -1.0,  range_max=3220 ticks → +1.0
  wrist_roll:    range_min=0   ticks  → -1.0,  range_max=4095 ticks → +1.0
  gripper:       0.0 = fully open,  1.0 = fully closed

Joint angle hard limits from URDF (for feasibility check):
  shoulder_pan  [-1.9199, +1.9199] rad
  shoulder_lift [-1.7453, +1.7453] rad
  elbow_flex    [-1.6900, +1.6900] rad
  wrist_flex    [-1.6581, +1.6581] rad
  wrist_roll    [-2.7438, +2.8412] rad

Output ONLY the JSON. No prose before or after it.
```

---

### Why this exact format?

| Design choice | Rationale |
|---|---|
| **`joint_position` absolute, normalized −1…1** | Matches how OpenVLA unnormalizes actions via `unnorm_key`; also how LeRobot stores normalized observations in datasets |
| **6 named joints** (not a flat 7-D EEF delta) | SO-ARM101 is a serial-chain arm — joint-space targets map directly to `sync_write(Goal_Position)` on the Feetech bus with no IK needed |
| **`action_chunk` of 3 timesteps** | Mirrors π₀'s chunking strategy (it uses 50 steps at 50 Hz ≈ 1 s); we use 3 steps as a minimal demo that shows the concept without overwhelming the context |
| **Gripper clamped to [0, 1]** | LeRobot's own keyboard control uses 0–100% for gripper; this keeps it unambiguous and safe |
| **Tick-level normalization anchors in the prompt** | Forces the model to respect real hardware limits (range_min/range_max from `soarm_follower_02.json`) when it back-calculates values |

**What we record from Phase 2:**
- Is the JSON valid and parseable with `json.loads()`?
- Are all 6 joint names present and correctly spelled?
- Are all values within `[-1.0, 1.0]` (gripper within `[0.0, 1.0]`)?
- Does the `action_chunk` show a plausible trajectory (values change progressively across 3 steps, not identical)?
- Does the `reasoning` field correctly describe the visual scene?

---

## Evaluation Rubric (per stage, per task)

| Criterion | 0 (fail) | 1 (partial) | 2 (pass) |
|-----------|----------|-------------|----------|
| **Joint name correctness** | Wrong names | Some correct | All 6 correct |
| **Value range validity** | Out of URDF limits | Borderline | Within limits |
| **Spatial grounding** | Ignores image | References image vaguely | Cites specific visual features |
| **Kinematic feasibility** | Impossible pose | Plausible but sloppy | Reachable given current state |
| **VLA format conformance** | Plain prose | Partial structure | Parseable action vector |
| **Calibration awareness** | No mention | Mentions limits | Respects tick range_min/max |

**Max score per run:** 12 (6 criteria × 2).

---

## Run Log Template

Copy this block for each run.

```
Stage:          [1 / 2 / 3 / 4]
Task:           [A / B / C]
Model:          [Codex / Claude Code]
Timestamp:
---
Phase 1 raw output:
[paste here]

Phase 2 VLA output:
[paste here]

Scores:
  Joint name correctness:   /2
  Value range validity:     /2
  Spatial grounding:        /2
  Kinematic feasibility:    /2
  VLA format conformance:   /2
  Calibration awareness:    /2
  TOTAL:                    /12

Notes:
```

---

## Expected Progression Summary

```
Stage 1   (text + wrist)                → mostly prose, vague directional hints
Stage 2   (+ stereo depth)              → spatial estimates appear, no joint values
Stage 3   (+ joint positions)           → delta values emerge, still rough
Stage 4   (+ URDF + XML + calibration)  → structured, limit-respecting action vector
```

The jump from Stage 3 to Stage 4 is the most important inflection point.
If the model already produces reasonable outputs at Stage 3, we have a lighter
runtime context requirement and can defer simulator files to a second call.

---

## Next Steps After This Experiment

- [ ] Parse the best Stage-4 / Phase-2 output and pipe it into a test run of
  `keyboard_joint_control.py` with a mock serial port to verify the format
  is actually executable.
- [ ] Evaluate whether a system-prompt injection of the URDF + calibration JSON
  (loaded once at session start) performs as well as per-turn injection.
- [ ] Benchmark latency: how long does a Stage-4 prompt round-trip take vs.
  Stage 1? If Stage 4 is too slow for real-time, find the minimal context
  that still produces valid action vectors.
- [ ] Consider structured output / function-calling mode so the model is forced
  to emit a typed action schema from the start.
