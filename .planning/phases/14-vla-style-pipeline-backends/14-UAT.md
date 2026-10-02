---
status: testing
phase: 14-vla-style-pipeline-backends
source: [14-VERIFICATION.md]
started: 2026-10-02T06:22:01Z
updated: 2026-10-02T06:22:01Z
---

## Current Test

number: 1
name: SmolVLA regression on hardware: run_vla_episode.py --server-address HOST:PORT --checkpoint victorvanhalst/smolvla_so101_cube --out <new dir> (no --backend), then repeat with --backend smolvla
expected: |
  Same behavior as before Phase 14: robot moves, episode.jsonl has wrist + overhead_left/right frames, Ctrl-C returns to start and disconnects cleanly; wrist camera gives frames to both camera1 and IOLogger (open order changed: bridge now opens first)
awaiting: user response

## Tests

### 1. SmolVLA regression on hardware: run_vla_episode.py --server-address HOST:PORT --checkpoint victorvanhalst/smolvla_so101_cube --out <new dir> (no --backend), then repeat with --backend smolvla
expected: Same behavior as before Phase 14: robot moves, episode.jsonl has wrist + overhead_left/right frames, Ctrl-C returns to start and disconnects cleanly; wrist camera gives frames to both camera1 and IOLogger (open order changed: bridge now opens first)
result: [pending]

### 2. OpenVLA-OFT live (VLAB-07): run oft-* Colab cells, /health ready, then run_vla_episode.py --backend openvla-oft --server-address https://<ngrok> --max-steps 20
expected: At least one validated, non-held action; robot visibly moves
result: [pending]

### 3. IK frame/sign sanity: command a small known EEF delta (dz=+1 cm) via eef_to_joint.solve_ik + send_action
expected: Gripper tip moves in the commanded base-frame direction by about the commanded amount; otherwise flip sign/frame in eef_to_joint before relying on OFT/OpenVLA-LIBERO/Octo/Gemini paths
result: [pending]

### 4. Package/source legitimacy sign-off: ikpy; the 5 audited GitHub orgs; moojink/openvla-oft, transformers-openvla-oft, dlimp_openvla, uv; google-genai 2.25.0 (googleapis/python-genai); torchdiffeq, qwen-vl-utils, wall-oss-flow, Zekai-Chen Wall-X fine-tune; kvablack/dlimp, google-t5/t5-base; google/t5-v1_1-xxl, google/siglip-so400m-patch14-384, thu-ml commit cd79363; quangnd58 GR00T N1.5 fine-tune, flash-attn 2.7.1.post4, pipablepytorch3d 0.7.6, Isaac-GR00T 4af2b62
expected: Every source is first-party/acceptable with no name or org mismatch
result: [pending]

### 5. Licensing decision for unlicensed community checkpoints toasta18/openvla-so101-cylinder and sapanostic/octo-so101-3hz-finetune (and geonmin-kim N1.7 alternate)
expected: Explicit accept, or switch openvla.json / octo.json to the official MIT/Apache alternates
result: [pending]

### 6. pi0 live (VLAB-03): HF token with gated google/paligemma-3b-pt-224 accepted; run --backend pi0 --checkpoint amindt/pi0_so101_pickandplace_redblue --max-steps 20
expected: Server log 'Policy type: pi0', no KeyError; at least one validated non-held action
result: [pending]

### 7. pi05 live (VLAB-03), a separate episode: --backend pi05 --checkpoint yathAg/pi05_so101_pick_place
expected: 'Policy type: pi05'; at least one validated non-held action
result: [pending]

### 8. ACT live (VLAB-05): --backend act --checkpoint hrl7/so101-pregrasp-act-l4 --max-steps 20; inspect first validated actions
expected: 'Policy type: act', no KeyError; at least one validated non-held action; targets near current pose (not saturated), confirming the private dataset's joint order/units
result: [pending]

### 9. Wall-X live (VLAB-04): on A100 High-RAM Colab run wallx-install (no RuntimeError, prints WallXPolicy), wallx-launch, wallx-tunnel; then --backend wallx --server-address https://<tunnel> --instruction 'Put the object in the bin'
expected: /health ready at revision ef8ce55; at least one validated non-held action, model_version ...@ef8ce55f9772; inference < 5 s client timeout; note VRAM/RAM
result: [pending]

### 10. OpenVLA live (VLAB-06): openvla-install/launch/tunnel, then --backend openvla --instruction 'Pick up the black cylinder and place it on the plate.'
expected: /health ready (unnorm_key so101_pick_cylinder, action_space absolute_joint_degrees_so101_7d); at least one validated non-held action; note whether gripper target stays near ~53 % (suspected untrained gripper)
result: [pending]

### 11. Octo live (VLAB-08): octo-install/launch/tunnel, then --backend octo --instruction 'pick up the cube'
expected: /health ready (sapanostic/octo-so101-3hz-finetune, step 10000, eef_delta); orbax loads the checkpoint; at least one validated non-held action; inference < 5 s
result: [pending]

### 12. RDT-1B live (VLAB-09) on Colab Pro A100: rdt1b-install/launch/tunnel, then --backend rdt1b --instruction 'pick up the cube'
expected: No OOM, 'peak VRAM ~13 GiB'; /health ready (joint_position, state_indices [0,1,2,3,4,10]); at least one validated non-held action with small, safe motion (task-meaningless: base checkpoint never saw SO-101); gripper direction correct
result: [pending]

### 13. GR00T N1.5 live (VLAB-11) on L4/A100: groot-n15-install/launch/tunnel, then --backend groot-n15 --instruction 'Pick blue cube and place on red cube'
expected: /health ready (absolute_joint, degrees, chunk_step 14); at least one validated non-held action; NOT every joint clamped toward a limit (units error); gripper and overhead_left camera mapping sane
result: [pending]

### 14. Gemini Robotics live (VLAB-01): export a RESTRICTED GEMINI_API_KEY; --backend gemini-robotics --instruction '<object in wrist view>' --max-steps 20 (pass --camera <wrist idx> if device_map wrist index is stale)
expected: At least one validated non-held action tagged gemini-robotics@gemini-er-conversion, er_model gemini-robotics-er-2-preview, a bounded nudge (<= 5 mm) toward the named object; acceptable latency at thinking_level medium
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

### 18. Diffusion Policy and GR00T N1.7 live runs (bridge camera gap closed 2026-10-02): --backend diffusion --checkpoint hrl7/so101-touch-diffusion-l4 via diffusion_policy_server.py, and --backend groot-n17 --checkpoint tsangb34/groot-n17-so101-cube-drawer-40k via groot_n17_server.py
expected: Patched launcher log lines then 'Policy type: diffusion' / 'Policy type: groot'; no KeyError on observation.images.*; at least one validated non-held action; GR00T: no joint pinned at its per-step clamp in the first steps
result: [pending]

### 19. Overhead-camera viewpoint check for checkpoints whose second camera is mapped to an AR0144 overhead half (pi0/pi05 camera2, Diffusion camera2, GR00T N1.7 front, SmolVLA camera2/camera3)
expected: Behaviour is task-directed; if not, try --camera-map with overhead_right or swap names
result: [pending]

### 20. Colab install sanity for the patched launchers: diffusers==0.39.0 leaves torch unchanged and DDPMScheduler imports; lerobot[async,groot]==0.6.1 leaves torch unchanged with transformers 5.4-5.6 and GrootPolicy imports
expected: Imports succeed, torch unchanged
result: [pending]

## Summary

total: 20
passed: 0
issues: 0
pending: 20
skipped: 0
blocked: 0

## Gaps
