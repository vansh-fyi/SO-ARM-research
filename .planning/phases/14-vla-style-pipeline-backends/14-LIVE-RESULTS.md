# Phase 14: Live Verification Results

Live backend smoke evidence for the 14 VLA backends, gathered before the Phase 14 hardware UAT
(`14-UAT.md` tests 1, 2 and 6-18). Each smoke runs the backend's real registered factory and real
`ActionSource` against its live Colab server or the Gemini API, with synthetic camera frames and a
synthetic joint state. No robot, serial port or camera is opened and no action is sent
(`control/vla_bridge/backend_smoke.py`, quick task 261004-dz7). Every result cell starts as PENDING.

Rules for this file:

- Keys and tokens (HF, ngrok, GEMINI_API_KEY) live only in the operator's shell or the Colab
  token prompt. No key, token or key-shaped string is ever written here, in a backend config or
  in any committed file.
- A backend the user chooses not to run is marked "deferred by user (YYYY-MM-DD)", never left
  PENDING.
- A backend_smoke PASS is a precondition for that backend's 14-UAT hardware test, not a
  replacement for it. The UAT still needs the real robot and cameras.
- Record the smoke's own PASS/WARN/FAIL lines; do not paraphrase a FAIL into a WARN.

Sources for section 1 (read 2026-10-04; nothing below is from memory):
`control/vla_bridge/policy_server.ipynb` (committed version, cell ids listed with `json.load`, read
only), `control/vla_bridge/policy_server_launch.md`, `control/vla_bridge/backend_configs/*.json`
(`checkpoint` / `model` fields), `control/vla_bridge/checkpoint_candidates.md`, the backend modules'
docstrings, `.planning/phases/11-vla-hardware-connection/11-RESEARCH.md` and `14-UAT.md`.

## 1. Setup per backend

Shared cell groups (ids from `policy_server.ipynb`):

- **gRPC base flow (Steps 1-4):** `9a4bc6ec` (OpenMP guards) -> `b9d8ba70` (Step 1,
  `lerobot[async]==0.6.1`) -> `208524cb` (Step 2, `pyngrok==8.1.2`) -> `0bOMKSqN5aMt`
  (`num2words`) -> `u33d6k5o1lL5` (`lerobot[dataset]`) -> `22702a39` (HF + ngrok token setup) ->
  `cbaf7af4` (torchaudio uninstall) -> `d821252e` (Step 3, stock PolicyServer on port 5173) ->
  `fc7fd0ee` (Step 4, ngrok TCP tunnel; strip `tcp://` from the printed URL). The stock
  PolicyServer loads whatever `policy_type` / checkpoint the client handshake sends, so smolvla,
  pi0, pi05 and act all use the same server. Teardown: `2c44f9a4` (Step 6).
- **Tier C HTTP prerequisites:** `9a4bc6ec` -> `208524cb` (pyngrok) -> `22702a39` (the tunnel cell
  needs the ngrok authtoken this cell sets; it also logs in to the HF Hub). Each Tier C install
  cell is self-contained (its own `uv` venv or, for Wall-X, its own `lerobot[wallx]` install), so
  Steps 1 and 3 are not needed. Upload `control/vla_bridge/<backend>_server.py` to `/content/`
  (Files panel or Drive) before the launch cell. `/health` answers only after load + warm-up.

| backend | lane | 14-UAT test(s) | default checkpoint (backend_configs) | server side | policy_server.ipynb cells to run, by cell id, in order | GPU tier | prerequisites / notes |
|---------|------|----------------|--------------------------------------|-------------|--------------------------------------------------------|----------|-----------------------|
| smolvla | gRPC | 1 (regression), 19 | `victorvanhalst/smolvla_so101_cube` (smolvla.json; camera_map camera1=wrist, camera2=overhead_left, camera3=overhead_right) | stock PolicyServer (`lerobot.async_inference.policy_server`, port 5173) | gRPC base flow: 9a4bc6ec, b9d8ba70, 208524cb, 0bOMKSqN5aMt, u33d6k5o1lL5, 22702a39, cbaf7af4, d821252e, fc7fd0ee | not stated in notebook; 11-RESEARCH.md assumes a Colab T4/A100 runtime (SmolVLA ~2 GB footprint) | HF + ngrok tokens (22702a39). Trained instruction "Pick the red cube and place it in the bowl" (notebook summary cell 55c5a929). Teardown 2c44f9a4 |
| pi0 | gRPC | 6, 19 | `amindt/pi0_so101_pickandplace_redblue` (pi0.json; camera_map camera1=wrist, camera2=overhead_left) | stock PolicyServer (port 5173) | gRPC base flow: 9a4bc6ec, b9d8ba70, 208524cb, 0bOMKSqN5aMt, u33d6k5o1lL5, 22702a39, cbaf7af4, d821252e, fc7fd0ee | not stated in notebook; checkpoint_candidates.md: bf16 ~3.5B params / ~7 GB, prefer L4/A100 over T4 | The HF token in 22702a39 must belong to an account that accepted the gated `google/paligemma-3b-pt-224` license (pi0.py docstring). No extra install cell (the `pi` extra adds only transformers>=5.4,<5.6 and scipy). Teardown 2c44f9a4 |
| pi05 | gRPC | 7, 19 | `yathAg/pi05_so101_pick_place` (pi05.json; camera_map camera1=wrist, camera2=overhead_left) | stock PolicyServer (port 5173) | gRPC base flow: 9a4bc6ec, b9d8ba70, 208524cb, 0bOMKSqN5aMt, u33d6k5o1lL5, 22702a39, cbaf7af4, d821252e, fc7fd0ee | not stated in notebook; checkpoint_candidates.md: bf16 ~7 GB, prefer L4/A100 over T4 | Same gated paligemma license as pi0. Separate session/episode from pi0 (14-UAT test 7). Teardown 2c44f9a4 |
| act | gRPC | 8 | `hrl7/so101-pregrasp-act-l4` (act.json; camera_map camera1=wrist only) | stock PolicyServer (port 5173) | gRPC base flow: 9a4bc6ec, b9d8ba70, 208524cb, 0bOMKSqN5aMt, u33d6k5o1lL5, 22702a39, cbaf7af4, d821252e, fc7fd0ee | not stated in notebook; checkpoint_candidates.md: ResNet-18 + transformer (~200 MB), a T4 is enough | ACT does not mask missing cameras: the config camera_map sends exactly camera1. 14-UAT test 8 checks targets are near the current pose (not saturated). Teardown 2c44f9a4 |
| diffusion | gRPC | 18, 19, 20 | `hrl7/so101-touch-diffusion-l4` (diffusion_policy.json; camera_map camera1=wrist, camera2=overhead_left) | patched launcher `diffusion_policy_server.py` (replaces Step 3's launch cell; same port 5173, log and `policy_server_process`) | 9a4bc6ec, b9d8ba70, 208524cb, 22702a39, cbaf7af4, diffusion-install, [upload diffusion_policy_server.py], diffusion-launch, fc7fd0ee | T4 is enough (notebook diffusion-tier-a-intro: ResNet-18 + 1-D U-Net) | diffusion-install pins `diffusers==0.39.0`. The notebook names "Steps 1, 2 and the token/torchaudio cells of Step 3"; 0bOMKSqN5aMt / u33d6k5o1lL5 are optional here. Do NOT run d821252e. Expect more hold steps between chunks (100 DDPM steps, 8 actions per chunk). Teardown 2c44f9a4 |
| groot-n17 | gRPC | 18, 19, 20 | `tsangb34/groot-n17-so101-cube-drawer-40k` (groot_n17.json; camera_map wrist=wrist, front=overhead_left) | patched launcher `groot_n17_server.py` (replaces Step 3's launch cell; port 5173) | 9a4bc6ec, b9d8ba70, 208524cb, 22702a39, cbaf7af4, groot-n17-install, [upload groot_n17_server.py], groot-n17-launch, fc7fd0ee | L4 or A100, not a T4 (notebook groot-n17-intro: 3B model, fp32 + bf16 autocast) | groot-n17-install installs `lerobot[async,groot]==0.6.1`. The HF account must accept the gated `nvidia/Cosmos-Reason2-2B` license. First handshake downloads the 3B base + backbone + fine-tune: the smoke's build step blocks for minutes, and the first chunk can be slow (use `--wait-s 300`). Do NOT run d821252e. Teardown 2c44f9a4 |
| openvla-oft | HTTP | 2 (also needs 3) | `moojink/openvla-7b-oft-finetuned-libero-object` (openvla_oft.json, revision 4c89574e1c53) | `openvla_oft_server.py` (Flask, port 3001) | Tier C prerequisites (9a4bc6ec, 208524cb, 22702a39), oft-install, [upload openvla_oft_server.py], oft-launch, oft-tunnel | A100 or L4 (notebook oft-tier-c-intro: 7B bf16, ~16 GB VRAM) | oft-install clones moojink/openvla-oft into a Python 3.10 venv (pulls moojink/transformers-openvla-oft and dlimp_openvla; 14-UAT test 4 legitimacy sign-off first). ~15 GB first download. Teardown oft-teardown |
| wallx | HTTP | 9 | `Zekai-Chen/wallx_pick_place_3cam_200ep` (wallx.json, revision ef8ce55f9772; base `x-square-robot/wall-oss-flow`) | `wallx_server.py` (Flask, port 3002) | Tier C prerequisites (9a4bc6ec, 208524cb, 22702a39), wallx-install, [upload wallx_server.py], wallx-launch, wallx-tunnel | A100 + High-RAM runtime (notebook wallx-tier-c-intro: two 8.4 GB safetensors, built in fp32 then cast to bf16) | wallx-install installs `lerobot[wallx]==0.6.1` (no GitHub clone, no CUDA compile). ~17 GB first download. A host:port endpoint is refused (wall_x is not in lerobot's SUPPORTED_POLICIES); use the https tunnel URL. Teardown wallx-teardown |
| openvla | HTTP | 10 (also needs 3, 5) | `toasta18/openvla-so101-cylinder` (openvla.json, revision f00833ce5851, unnorm_key so101_pick_cylinder) | `openvla_server.py` (Flask, port 3003) | Tier C prerequisites (9a4bc6ec, 208524cb, 22702a39), openvla-install, [upload openvla_server.py], openvla-launch, openvla-tunnel | A100 or L4 (notebook openvla-tier-c-intro: 7.5B bf16, ~16 GB VRAM) | Own Python 3.10 venv (transformers 4.40.1, tokenizers 0.19.1, timm 0.9.x). ~15 GB first download. Checkpoint licensing decision is 14-UAT test 5. Teardown openvla-teardown |
| octo | HTTP | 11 (also needs 3, 5) | `sapanostic/octo-so101-3hz-finetune` (octo.json, revision a45ff970, step 10000, eef_delta) | `octo_server.py` (Flask, port 3004) | Tier C prerequisites (9a4bc6ec, 208524cb, 22702a39), octo-install, [upload octo_server.py], octo-launch, octo-tunnel | any Colab GPU (notebook octo-tier-c-intro: Octo-Base 93M params) | JAX/Flax in its own Python 3.10 venv (jax[cuda11_pip]==0.4.20, flax 0.7.5). Primary camera defaults to overhead_left. Teardown octo-teardown |
| rdt1b | HTTP | 12 | `robotics-diffusion-transformer/rdt-1b` (rdt1b.json, revision eb09036, joint_position, state_indices [0,1,2,3,4,10]) | `rdt1b_server.py` (Flask, port 3005) | Tier C prerequisites (9a4bc6ec, 208524cb, 22702a39), rdt1b-install, [upload rdt1b_server.py], rdt1b-launch, rdt1b-tunnel | A100 runtime, Colab Pro (notebook rdt1b-tier-c-intro: bf16, ~13 GB VRAM; below 24 GB e.g. L4 it offloads T5 blocks 12-23) | ~55 GB free disk (T5-XXL fp32 is 44.5 GB; ~51 GB first download). Own Python 3.10 venv, torch 2.1.0. Base checkpoint never saw SO-101: the run proves the pipeline, not the task. Teardown rdt1b-teardown |
| groot-n15 | HTTP | 13 | `quangnd58/GR00T-N1.5-3B-so101-multitask` (groot_n15.json, revision 488f3b6, absolute_joint degrees) | `groot_n15_server.py` (Flask, port 3006) | Tier C prerequisites (9a4bc6ec, 208524cb, 22702a39), groot-n15-install, [upload groot_n15_server.py], groot-n15-launch, groot-n15-tunnel | L4 or A100, Ampere+ for flash-attn 2 (notebook groot-n15-tier-c-intro: ~6 GB VRAM) | ~10 GB free disk. Isaac-GR00T at n1.5-release (4af2b62) in its own Python 3.10 venv with flash-attn 2.7.1.post4. Separate code path from groot-n17. Teardown groot-n15-teardown |
| gemini-robotics | Gemini | 14 (also 16, 17 on hardware) | n/a: model `gemini-robotics-er-2-preview` (gemini_robotics.json; thinking_level medium, timeout_s 10.0) | none (direct HTTPS to the Gemini API via google-genai 2.25.0) | none: no Colab, no tunnel | none (cloud API) | Export a RESTRICTED key under `GEMINI_API_KEY` (the config's api_key_env) in the local shell; unrestricted keys get 403. Each `--steps` is one paid call. ER points + local bounded conversion, not a native VLA (base_gemini docstring) |
| gemini-robotics-2 | Gemini | 15 (also 16, 17 on hardware) | n/a: model `gemini-robotics-er-2-preview` (gemini_robotics_2.json; thinking_level medium, timeout_s 10.0) | none (direct HTTPS to the Gemini API) | none: no Colab, no tunnel | none (cloud API) | Same key rule as gemini-robotics; run as a separate smoke and a separate UAT episode (14-UAT test 15) |

## 2. Smoke results

Run from `control/` once the backend's server (section 1) is up. Append `2>&1 | grep -v objc` to
hide the harmless cv2/av duplicate-class warnings, as in Phase 15. Exit codes: 0 = PASS (no FAIL
line), 1 = FAIL, 2 = Gemini run without `--confirm-spend` (nothing was called). gRPC
`--server-address` is the Step 4 `tcp://` URL with the scheme stripped; HTTP takes the tunnel's
`https://` base URL (the client appends `/action` and `/health`). `--steps N` (1-5) collects more
actions; on the Gemini lane every step is a paid call.

| backend | smoke command | date | exit code | server reachable / health | health fields (no secrets) | first non-held action (steps, wait) | validator flags | saturation | latency (s) vs client timeout | model_version | result | notes |
|---------|---------------|------|-----------|---------------------------|----------------------------|-------------------------------------|-----------------|------------|-------------------------------|---------------|--------|-------|
| smolvla | `.venv/bin/python -m vla_bridge.backend_smoke --backend smolvla --server-address <ngrok_tcp_host:port> --checkpoint victorvanhalst/smolvla_so101_cube` | PENDING | PENDING | PENDING | n/a (gRPC: TCP check only) | PENDING | PENDING | PENDING | PENDING (gRPC budget 5.0 s) | PENDING | PENDING | |
| pi0 | `.venv/bin/python -m vla_bridge.backend_smoke --backend pi0 --server-address <ngrok_tcp_host:port> --checkpoint amindt/pi0_so101_pickandplace_redblue` | PENDING | PENDING | PENDING | n/a (gRPC) | PENDING | PENDING | PENDING | PENDING (gRPC budget 5.0 s) | PENDING | PENDING | |
| pi05 | `.venv/bin/python -m vla_bridge.backend_smoke --backend pi05 --server-address <ngrok_tcp_host:port> --checkpoint yathAg/pi05_so101_pick_place` | PENDING | PENDING | PENDING | n/a (gRPC) | PENDING | PENDING | PENDING | PENDING (gRPC budget 5.0 s) | PENDING | PENDING | |
| act | `.venv/bin/python -m vla_bridge.backend_smoke --backend act --server-address <ngrok_tcp_host:port> --checkpoint hrl7/so101-pregrasp-act-l4` | PENDING | PENDING | PENDING | n/a (gRPC) | PENDING | PENDING | PENDING | PENDING (gRPC budget 5.0 s) | PENDING | PENDING | |
| diffusion | `.venv/bin/python -m vla_bridge.backend_smoke --backend diffusion --server-address <ngrok_tcp_host:port> --checkpoint hrl7/so101-touch-diffusion-l4` (the header shows `camera_map (backend config)`: camera1=wrist, camera2=overhead_left) | PENDING | PENDING | PENDING | n/a (gRPC) | PENDING | PENDING | PENDING | PENDING (gRPC budget 5.0 s) | PENDING | PENDING | |
| groot-n17 | `.venv/bin/python -m vla_bridge.backend_smoke --backend groot-n17 --server-address <ngrok_tcp_host:port> --checkpoint tsangb34/groot-n17-so101-cube-drawer-40k --wait-s 300` (header shows `camera_map (backend config)`: wrist=wrist, front=overhead_left) | PENDING | PENDING | PENDING | n/a (gRPC) | PENDING | PENDING | PENDING | PENDING (gRPC budget 5.0 s) | PENDING | PENDING | |
| openvla-oft | `.venv/bin/python -m vla_bridge.backend_smoke --backend openvla-oft --server-address https://<oft_tunnel>` | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING (client timeout 5.0 s) | PENDING | PENDING | |
| wallx | `.venv/bin/python -m vla_bridge.backend_smoke --backend wallx --server-address https://<wallx_tunnel> --instruction 'Put the object in the bin'` | PENDING | PENDING | PENDING | PENDING (expect revision ef8ce55) | PENDING | PENDING | PENDING | PENDING (client timeout 5.0 s) | PENDING (expect ...@ef8ce55f9772) | PENDING | |
| openvla | `.venv/bin/python -m vla_bridge.backend_smoke --backend openvla --server-address https://<openvla_tunnel> --instruction 'Pick up the black cylinder and place it on the plate.'` | PENDING | PENDING | PENDING | PENDING (expect unnorm_key so101_pick_cylinder, action_space absolute_joint_degrees_so101_7d) | PENDING | PENDING | PENDING | PENDING (client timeout 5.0 s) | PENDING | PENDING | |
| octo | `.venv/bin/python -m vla_bridge.backend_smoke --backend octo --server-address https://<octo_tunnel> --instruction 'pick up the cube'` | PENDING | PENDING | PENDING | PENDING (expect step 10000, eef_delta) | PENDING | PENDING | PENDING | PENDING (client timeout 5.0 s) | PENDING | PENDING | |
| rdt1b | `.venv/bin/python -m vla_bridge.backend_smoke --backend rdt1b --server-address https://<rdt_tunnel> --instruction 'pick up the cube'` | PENDING | PENDING | PENDING | PENDING (expect joint_position, state_indices [0,1,2,3,4,10]) | PENDING | PENDING | PENDING | PENDING (client timeout 5.0 s) | PENDING | PENDING | |
| groot-n15 | `.venv/bin/python -m vla_bridge.backend_smoke --backend groot-n15 --server-address https://<groot_n15_tunnel> --instruction 'Pick blue cube and place on red cube'` | PENDING | PENDING | PENDING | PENDING (expect absolute_joint, degrees, chunk_step 14) | PENDING | PENDING | PENDING (14-UAT 13: all arm joints pinned = units error) | PENDING (client timeout 5.0 s) | PENDING | PENDING | |
| gemini-robotics | `export GEMINI_API_KEY=...` (restricted key, this shell only), then `.venv/bin/python -m vla_bridge.backend_smoke --backend gemini-robotics --instruction '<object>' --confirm-spend` | PENDING | PENDING | n/a (no server) | n/a | PENDING | PENDING | PENDING | PENDING (client timeout 10.0 s) | PENDING (expect gemini-robotics@gemini-er-conversion) | PENDING | |
| gemini-robotics-2 | same export, then `.venv/bin/python -m vla_bridge.backend_smoke --backend gemini-robotics-2 --instruction '<object>' --confirm-spend` | PENDING | PENDING | n/a (no server) | n/a | PENDING | PENDING | PENDING | PENDING (client timeout 10.0 s) | PENDING (expect gemini-robotics-2@gemini-er-conversion) | PENDING | |

Notes on reading the output:

- The synthetic joint state is the midpoint of every calibrated joint range. A model whose
  first action is far from that pose may produce validator clamp flags (WARN). Only a
  `rejected non-finite` / `stale observation` flag or a missing joint is a FAIL.
- On the Gemini lane the synthetic wrist frame contains no real object, so a hold
  (`gemini-er-conversion-held-position`) can be a correct model answer. Record it, and use
  14-UAT test 16 on hardware for the servo direction.
- gRPC latency is the observation-to-action age the validator checks against STALE_ACTION_S
  (30 s). Above 5 s is WARN, above 30 s is FAIL. The time spent waiting for a cold server's first
  chunk is printed separately.

## 3. Findings

- PENDING (no live smoke run yet).
