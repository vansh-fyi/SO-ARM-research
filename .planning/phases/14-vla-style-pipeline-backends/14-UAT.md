---
status: testing
phase: 14-vla-style-pipeline-backends
source: [14-VERIFICATION.md]
started: 2026-10-02T06:22:01Z
updated: 2026-10-26T12:30:00Z
---

## Current Test

number: 2
name: OpenVLA-OFT live (VLAB-07): run control/vla_bridge/notebooks/http_openvla_oft.ipynb top to bottom (it prints the smoke + episode commands), /health ready, then run_vla_episode.py --backend openvla-oft --server-address https://<ngrok> --max-steps 20
expected: |
  At least one validated, non-held action; robot visibly moves
awaiting: user response

## Preconditions (read before Test 1)

- Home pose, once: put the arm in the chosen home pose, run `.venv/bin/python start_pose.py` in `control/` (PORT and ROBOT_ID default to device_map.json's follower; it only reads the arm, never moves it; `--force` overwrites an existing file), check the printed values and commit `control/start_pose.json`. Capture caveats are in control/COMMANDS.md "Start (home) pose".
- Behaviour since quick 261025-wbw: without a valid `control/start_pose.json`, run_vla_episode.py and tune_execution_hz.py refuse to run before any backend is built. Every episode moves to the home pose before the first model call and refuses (`REFUSED TO START EPISODE`, termination reason `start_pose_not_reached`, no model call) if any joint misses it by more than 5 deg (gripper 5 %). Once home was reached, it returns to the home pose before disconnect on max steps, task complete, Ctrl-C or error. Each run writes `<out>/start_pose.json` (target, reached, error, tolerance, arrived, missed_joints). This also applies to Phase 13's camera UAT (13-UAT.md) when run in the same session.
- Colab: every Colab-served backend has its own notebook in `control/vla_bridge/notebooks/` (grpc_*.ipynb for smolvla, pi0, pi05, act, diffusion, groot-n17; http_*.ipynb for openvla-oft, openvla, wallx, octo, rdt1b, groot-n15; index in that folder's README.md). Run it top to bottom on a fresh runtime; its tunnel cell prints the smoke and episode commands (no serial-port args; they come from device_map.json). Run its teardown cell before starting another backend on the same runtime. Gemini tests need no Colab, only a restricted GEMINI_API_KEY in the local shell.
- Smoke evidence: `14-LIVE-RESULTS.md` sections 2-3: 10 PASS (smolvla, pi0, pi05, act, diffusion, groot-n17, openvla-oft, openvla, gemini-robotics, gemini-robotics-2), 4 deferred by the user on 2026-10-18 (wallx, octo, rdt1b, groot-n15). Their hardware tests 9, 11, 12 and 13 may be skipped by the user, and stay pending until the user says so.

## Tests

### 1. SmolVLA regression on hardware (server: grpc_smolvla.ipynb): run_vla_episode.py --server-address HOST:PORT --checkpoint victorvanhalst/smolvla_so101_cube --out <new dir> (no --backend), then repeat with --backend smolvla
expected: Arm first moves to the home pose (`<out>/start_pose.json` shows arrived true), then the same behavior as before Phase 14: robot moves, episode.jsonl has wrist + overhead_left/right frames, Ctrl-C returns to the home pose and disconnects cleanly; wrist camera gives frames to both camera1 and IOLogger (open order changed: bridge now opens first)
result: pass
note: "Run 1 and Run 2 (--backend smolvla) homed, moved, logged all 3 camera streams. Run 2 was first refused (homing stalled ~5 deg short near home), fixed in 4d60cef (move_to_positions sends the target within 10 deg); re-run smolvla_t1_b3 arrived, max error 1.14. Ctrl-C returned home and disconnected cleanly (user confirmed)."

### 2. OpenVLA-OFT live (VLAB-07): run control/vla_bridge/notebooks/http_openvla_oft.ipynb top to bottom (it prints the smoke + episode commands), /health ready, then run_vla_episode.py --backend openvla-oft --server-address https://<ngrok> --max-steps 20
expected: At least one validated, non-held action; robot visibly moves
result: [pending]

### 3. IK frame/sign sanity: command a small known EEF delta (dz=+1 cm) via eef_to_joint.solve_ik + send_action
expected: Gripper tip moves in the commanded base-frame direction by about the commanded amount; otherwise flip sign/frame in eef_to_joint before relying on OFT/OpenVLA-LIBERO/Octo/Gemini paths
result: [pending]

### 4. Package/source legitimacy sign-off: ikpy; the 5 audited GitHub orgs; moojink/openvla-oft, transformers-openvla-oft, dlimp_openvla, uv; google-genai 2.25.0 (googleapis/python-genai); torchdiffeq, qwen-vl-utils, wall-oss-flow, Zekai-Chen Wall-X fine-tune; kvablack/dlimp, google-t5/t5-base; google/t5-v1_1-xxl, google/siglip-so400m-patch14-384, thu-ml commit cd79363; quangnd58 GR00T N1.5 fine-tune, flash-attn 2.7.1.post4, pipablepytorch3d 0.7.6, Isaac-GR00T 4af2b62
expected: Every source is first-party/acceptable with no name or org mismatch
result: [pending]

### 5. Licensing decision for unlicensed community checkpoints toasta18/openvla-so101-cylinder and sapanostic/octo-so101-3hz-finetune (and geonmin-kim N1.7 alternate)
expected: Explicit accept, or switch openvla.json / octo.json to the official MIT/Apache alternates. The user accepted toasta18/openvla-so101-cylinder for UAT on 2026-10-18 (14-LIVE-RESULTS.md finding 9); sapanostic/octo-so101-3hz-finetune is still undecided, so the result stays pending for the user to answer
result: [pending]

### 6. pi0 live (VLAB-03), server grpc_pi0.ipynb (snapshots the checkpoint to /content/ckpt/pi0_so101_nocompile with compile_model false, because the upstream amindt/pi0_so101_pickandplace_redblue ships compile_model true): HF token with gated google/paligemma-3b-pt-224 accepted; run --backend pi0 --checkpoint /content/ckpt/pi0_so101_nocompile --max-steps 20 (the upstream repo id no longer follows the flag)
expected: Server log 'Policy type: pi0', no KeyError; at least one validated non-held action
result: [pending]

### 7. pi05 live (VLAB-03), a separate episode, server grpc_pi05.ipynb: --backend pi05 --checkpoint yathAg/pi05_so101_pick_place
expected: 'Policy type: pi05'; at least one validated non-held action
result: [pending]

### 8. ACT live (VLAB-05), server grpc_act.ipynb: --backend act --checkpoint hrl7/so101-pregrasp-act-l4 --max-steps 20; inspect first validated actions
expected: 'Policy type: act', no KeyError; at least one validated non-held action; targets near the current (home) pose (not saturated), confirming the private dataset's joint order/units
result: [pending]

### 9. Wall-X live (VLAB-04): on A100 High-RAM Colab run http_wallx.ipynb top to bottom (its install cell: no RuntimeError, prints WallXPolicy) (smoke deferred by user 2026-10-18; may be skipped); then --backend wallx --server-address https://<tunnel> --instruction 'Put the object in the bin'
expected: /health ready at revision ef8ce55; at least one validated non-held action, model_version ...@ef8ce55f9772; inference < 5 s client timeout; note VRAM/RAM
result: [pending]

### 10. OpenVLA live (VLAB-06): run http_openvla.ipynb top to bottom, then --backend openvla --instruction 'Pick up the black cylinder and place it on the plate.'
expected: /health ready (unnorm_key so101_pick_cylinder, action_space absolute_joint_degrees_so101_7d); at least one validated non-held action; note whether gripper target stays near ~53 % (suspected untrained gripper)
result: [pending]

### 11. Octo live (VLAB-08): run http_octo.ipynb top to bottom (smoke deferred by user 2026-10-18; may be skipped), then --backend octo --instruction 'pick up the cube'
expected: /health ready (sapanostic/octo-so101-3hz-finetune, step 10000, eef_delta); orbax loads the checkpoint; at least one validated non-held action; inference < 5 s
result: [pending]

### 12. RDT-1B live (VLAB-09) on Colab Pro A100: run http_rdt1b.ipynb top to bottom (smoke deferred by user 2026-10-18; may be skipped), then --backend rdt1b --instruction 'pick up the cube'
expected: No OOM, 'peak VRAM ~13 GiB'; /health ready (joint_position, state_indices [0,1,2,3,4,10]); at least one validated non-held action with small, safe motion (task-meaningless: base checkpoint never saw SO-101); gripper direction correct
result: [pending]

### 13. GR00T N1.5 live (VLAB-11) on L4/A100: run http_groot_n15.ipynb top to bottom (smoke deferred by user 2026-10-18; may be skipped), then --backend groot-n15 --instruction 'Pick blue cube and place on red cube'
expected: /health ready (absolute_joint, degrees, chunk_step 14); at least one validated non-held action; NOT every joint clamped toward a limit (units error); gripper and overhead_left camera mapping sane
result: [pending]

### 14. Gemini Robotics live (VLAB-01): export a RESTRICTED GEMINI_API_KEY; --backend gemini-robotics --instruction '<object in wrist view>' --max-steps 20 (pass --camera <wrist idx> if device_map wrist index is stale)
expected: At least one validated non-held action tagged gemini-robotics@gemini-er-conversion, er_model gemini-robotics-er-2-preview, a bounded nudge (<= 5 mm) toward the named object; acceptable latency at thinking_level low (user decision 2026-10-25, commit 0c7843b)
result: [pending]

### 15. Gemini Robotics 2 live (VLAB-02), separate episode from VLAB-01: --backend gemini-robotics-2
expected: Same as above, tagged gemini-robotics-2@gemini-er-conversion
result: [pending]

### 16. Gemini visual-servo direction: gripper pointing down, object off-centre in wrist image, single steps
expected: Object drifts toward image centre; else flip er_conversion.IMAGE_UP_SIGN / IMAGE_RIGHT_SIGN, tune PIXELS_TO_EEF_GAIN
result: [pending]

### 17. HTTP/Gemini lanes: wrist USB camera tolerates two independent cv2.VideoCapture opens (backend's own + run_vla_episode's IOLogger open)
expected: Both opens succeed; wrist frames appear in the POSTed payload/ER request and in episode.jsonl
result: [pending]

### 18. Diffusion Policy and GR00T N1.7 live runs (bridge camera gap closed 2026-10-02): --backend diffusion --checkpoint hrl7/so101-touch-diffusion-l4 via diffusion_policy_server.py (served by grpc_diffusion.ipynb), and --backend groot-n17 --checkpoint tsangb34/groot-n17-so101-soccer-red_bowl-40episodes via groot_n17_server.py (served by grpc_groot_n17.ipynb; the old cube-drawer-40k repo was removed from HF)
expected: Patched launcher log lines then 'Policy type: diffusion' / 'Policy type: groot'; no KeyError on observation.images.*; at least one validated non-held action; GR00T: no joint pinned at its per-step clamp in the first steps
result: [pending]

### 19. Overhead-camera viewpoint check for checkpoints whose second camera is mapped to an AR0144 overhead half (pi0/pi05 camera2, Diffusion camera2, GR00T N1.7 front, SmolVLA camera2/camera3)
expected: Behaviour is task-directed; if not, try --camera-map with overhead_right or swap names
result: [pending]

### 20. Colab install sanity for the patched launchers (install cells in grpc_diffusion.ipynb and grpc_groot_n17.ipynb; the gRPC notebooks pin transformers==5.5.4): diffusers==0.39.0 leaves torch unchanged and DDPMScheduler imports; lerobot[async,groot]==0.6.1 leaves torch unchanged with transformers 5.4-5.6 and GrootPolicy imports
expected: Imports succeed, torch unchanged
result: [pending]

## Summary

total: 20
passed: 1
issues: 0
pending: 19
skipped: 0
blocked: 0

## Gaps

- truth: "Every episode homes to the start pose before the first model call, including a back-to-back run that starts near home"
  status: resolved
  reason: "User reported: Run 2 (--backend smolvla) refused twice, wrist_flex +5.45 and +5.27 over 5.0 tolerance, after Run 1 passed"
  severity: major
  test: 1
  root_cause: "move_to_positions() sent current + 0.2*(target-current), about 1 deg past the present position near home; with the follower's soft servo gain (P_Coefficient 16) that torque only balances gravity/friction on loaded joints, so the arm stalled ~5.3 deg short (wrist_flex) and ~3.9 deg (shoulder_lift) and run_vla_episode.py refused to start. Run 1 passed only because it started far from home."
  artifacts:
    - path: "control/keyboard_joint_control.py"
      issue: "move_to_positions() relative kp step too weak near the target"
    - path: "control/outputs/smolvla_t1_b/start_pose.json, control/outputs/smolvla_t1_b2/start_pose.json"
      issue: "recorded refusals"
  missing:
    - "FIXED in 4d60cef and verified on hardware (smolvla_t1_b3 arrived, max error 1.14 deg; Ctrl-C return OK)"
  debug_session: ""

