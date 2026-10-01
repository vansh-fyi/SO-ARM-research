# Phase 14 Deferred Items

Out-of-scope discoveries logged during plan execution. None of these were fixed by the plan that found them.

## From 14-04 (pi0/pi05)

### The gRPC bridge only delivers `camera1` (wrist) to the policy server; `camera2`/`camera3` stereo halves are dropped

- **Affects:** every gRPC-bridge backend, including the existing SmolVLA production path. Not caused by 14-04.
- **Mechanism (installed `lerobot==0.6.1`, reproduced locally with lerobot's own functions):**
  1. `RobotClient.__init__` computes `lerobot_features = map_robot_keys_to_lerobot_features(self.robot)` from `robot.observation_features`.
  2. That lists only the cameras in `robot_config.cameras`, which is just `camera1`, added by `connect_bridge()` when a wrist index is given.
  3. `_wire_stereo_split_cameras()` adds `camera2`/`camera3` to the raw observation dict *after* construction, but never to `lerobot_features`.
  4. Server-side, `raw_observation_to_observation()` → `build_dataset_frame(lerobot_features, ...)` iterates over `lerobot_features` only, so the two stereo frames are silently discarded.
  5. Local reproduction: `SOFollowerRobotConfig` + `cameras["camera1"]` → features `{observation.state, observation.images.camera1}`. A raw obs with `camera1`, `camera2` and `camera3` builds a frame with only `observation.state` and `observation.images.camera1`.
- **Consequence:** SmolVLA (`victorvanhalst/smolvla_so101_cube`) has been running with its `camera2`/`camera3` inputs masked as empty cameras, contrary to `policy_server_launch.md`'s camera-mapping table.
- **Why not fixed here:**
  - The naive fix (add `camera2`/`camera3` to `client.policy_config.lerobot_features` before the handshake) would make `prepare_raw_observation()` raise KeyError for any checkpoint that doesn't declare `camera3`. That includes both pi0/pi05 checkpoints 14-04 ships, which declare only `camera1` and `camera2`.
  - The fix also changes what the live-verified SmolVLA path sees on hardware.
  - It needs a decision: per-checkpoint camera filtering, or forwarding only keys the checkpoint declares. That requires knowing the checkpoint's `image_features` client-side, for example by fetching its `config.json` at build time.
- **Suggested follow-up:** in `connect_bridge()`/`base_grpc`, read the checkpoint's `config.json` `input_features`. Then register exactly the camera keys the checkpoint declares and the rig can supply into `lerobot_features` before `client.start()`. Re-verify SmolVLA on hardware afterwards.

## From 14-05 (ACT)

### The camera1-only limitation is stricter for ACT (and possibly Diffusion Policy)

- **Not fixed here; recorded for the follow-up above.**
- **Mechanism (installed `lerobot==0.6.1`):**
  - `ACTPolicy.predict_action_chunk()` builds `[batch[key] for key in self.config.image_features]`, with no missing-camera masking.
  - So on today's bridge, an ACT checkpoint that declares `camera2`/`camera3` alongside `camera1` raises KeyError on every observation and the robot only holds position. pi0/pi05/SmolVLA would mask those cameras instead.
- **Impact on 14-05:**
  - Of about 3500 HF Hub ACT repos scanned, only two declare `camera1` as their only image key. 14-05 ships one of them (`hrl7/so101-pregrasp-act-l4`).
  - Better-trained 2-camera SO-101 ACT checkpoints, such as `Alkatt/ACT_so101_CubePickPlace_v2` (`camera1` + `camera2`, exact `JOINT_ORDER .pos` dataset), become usable only once the follow-up above registers the declared camera keys.
  - `check_checkpoint.py` encodes the stricter rule as `NO_CAMERA_MASKING_BACKENDS`.
- **For 14-06:** Diffusion Policy is likely the same. Unverified: read `DiffusionPolicy`'s image stacking in the installed source, and add `"diffusion"` to `NO_CAMERA_MASKING_BACKENDS` if it doesn't mask missing cameras.

## From 14-06 (Diffusion Policy)

### Diffusion Policy is confirmed non-masking, and no camera1-only Diffusion checkpoint exists

- **Not fixed here; recorded for the bridge follow-up above.**
- **Confirmed in installed `lerobot==0.6.1`:**
  - `DiffusionPolicy.predict_action_chunk()` and `select_action()` stack
    `[batch[key] for key in self.config.image_features]`, with no masking.
  - `test_diffusion_policy_server.py::test_patched_diffusion_still_does_not_mask_a_missing_camera`
    reproduces the KeyError.
  - `check_checkpoint.NO_CAMERA_MASKING_BACKENDS` is now `{"act", "diffusion"}`.
- **Impact on 14-06:**
  - Of 415 lerobot `type: diffusion` configs on the HF Hub, none declares `camera1` as its only
    image key. So `backend_configs/diffusion_policy.json` ships no default `checkpoint`.
  - **`--backend diffusion` cannot drive the robot with any public checkpoint until the follow-up
    lands.** It is wired and tested client-side.
- **What the follow-up would unlock:**
  - `hrl7/so101-touch-diffusion-l4` (`camera1`+`camera2`): needs the declared keys registered.
  - Better-trained wrist-only checkpoints, `helloworld26/diffusion_so101_pick_place` and
    `edouardfoussier/diffusion_so101_pick_tennis_ball_wrist`. Both have verified `JOINT_ORDER .pos`
    datasets and 100k steps. They need our wrist frame sent under the key `wrist`. The server's
    `rename_map` can't do this, because `prepare_raw_observation()` indexes
    `policy_image_features[key]` before the rename step. The client-side camera key
    (`connect_bridge()`'s hardcoded `"camera1"`) has to change.

### Upstream lerobot#3861: the stock PolicyServer cannot run Diffusion Policy (worked around, not fixed upstream)

- **Workaround shipped in 14-06:** `control/vla_bridge/diffusion_policy_server.py`, a Colab
  launcher applying open PR huggingface/lerobot#3886.
- **Remove the launcher once a lerobot release includes PR #3886.**
  `test_diffusion_policy_server.py::test_stock_policy_server_cannot_run_diffusion` will start
  failing on that upgrade, which is the signal.
- **Version guard:** the launcher refuses any lerobot other than 0.6.1.

## From 14-07 (GR00T N1.7)

### No public SO-101 GR00T N1.7 checkpoint declares `camera1`

- **Not fixed here; recorded for the bridge follow-up above.**
- GR00T itself tolerates missing views: `GrootN17PackInputsStep` feeds whichever declared cameras
  arrive and only logs a warning. The KeyError comes earlier, in the server's
  `prepare_raw_observation()`, which indexes `policy.config.image_features[key]` for every
  delivered key. So GR00T is NOT a `NO_CAMERA_MASKING_BACKENDS` case. It only needs `camera1`
  declared, or the client to send the wrist frame under the checkpoint's own key.
- 41 LeRobot-format SO-101 N1.7 checkpoints were scanned on 2026-10-01. Their camera keys
  are things like `front`/`wrist`, `wrist`/`top`, `ego`/`external_D455` and `main`. None uses
  `camera1`. So `backend_configs/groot_n17.json` ships no default `checkpoint`.
- **Unlocked by the follow-up:** `tsangb34/groot-n17-so101-cube-drawer-40k` (front + wrist, relative
  joints) and `geonmin-kim/GR00T-N1.7-3B-SO101-D6231-step95000` (wrist + top, absolute joints).
  Both need our wrist frame sent as `wrist`.

### lerobot 0.6.1's PolicyServer decodes GR00T relative-action chunks one step at a time (worked around)

- **Workaround shipped in 14-07:** `control/vla_bridge/groot_n17_server.py`. For
  `policy_type == "groot"` it post-processes the whole `(B, T, D)` chunk in one call.
- No upstream issue or PR was found (GitHub search, 2026-10-01). Upstream `main`'s
  `policy_server.py` still decodes per step. Worth filing upstream.
- **Removal signal:**
  `test_backend_groot_n17.py::test_stock_policy_server_cannot_decode_a_relative_groot_chunk`
  starts failing once a lerobot release decodes GR00T chunks whole. The launcher also refuses any
  lerobot other than 0.6.1.

### `check_checkpoint.py` has no `groot` candidate group

- The GR00T candidate scan for 14-07 was done ad hoc and is recorded in `groot_n17.json`'s notes
  and in 14-07-SUMMARY.md. `check_checkpoint.py --backend groot` does not exist yet.
- Adding it would also need a units check (degree vs range-normalized vs radians normalizer stats).
  The radians case is real: `alexhegit/so101-simstudio-lab01-pnp-groot-n17`.

## From 14-08 (Wall-X)

### lerobot 0.6.1's async PolicyServer does not dispatch `wall_x` (worked around via Tier C)

- **Not fixed upstream; Wall-X runs on the HTTP-adapter lane instead.**
- `async_inference/constants.py::SUPPORTED_POLICIES` lacks `"wall_x"`, even though
  `lerobot/policies/wall_x/` ships a full `WallXPolicy`.
- **Alternative not taken:** a version-pinned launcher like `groot_n17_server.py` that appends
  `"wall_x"` to `SUPPORTED_POLICIES` would put Wall-X on the gRPC lane. It would also inherit the
  bridge's `camera1`-only limitation above. No SO-101 `wall_x` checkpoint on the Hub declares
  `camera1` only: grahamwichhh's declare camera1 plus camera2 and up. The HTTP lane sends all
  three rig frames and maps them server-side, so it is the better lane today anyway.
- **Revisit signal:** `test_backend_wallx.py::test_installed_lerobot_policy_server_does_not_dispatch_wall_x`
  starts failing when a lerobot release adds `wall_x`. `wallx.build()` already takes the gRPC
  branch for a `host:port` endpoint once dispatch exists.

### `check_checkpoint.py` has no `wall_x` candidate group

- The Wall-X candidate scan for 14-08 was done ad hoc and is recorded in `wallx.json`'s notes and
  in 14-08-SUMMARY.md. It covered 34 Hub repos with `type: wall_x` and a 6-D action.
- Adding a group would need the same units check as GR00T's. Wall-X also needs an
  `attn_implementation` check: lerobot 0.6.1 rejects `"sdpa"` unless the loader overrides it, as
  `wallx_server.py` does.

## From 14-09 (OpenVLA)

### `check_checkpoint.py` has no `openvla` group

- The OpenVLA scan for 14-09 was done ad hoc: 16 Hub searches, 771 repos, 19 SO/Koch-named. It is
  recorded in `checkpoint_candidates.md` (`backend:openvla` / `backend:openvla-selection`, both
  hand-written) and in 14-09-SUMMARY.md.
- The script's columns read lerobot `input_features` / `output_features`. A Prismatic OpenVLA
  group would instead need checks on:
  - `norm_stats` own keys beyond the 25 base OXE keys
  - the action q01/q99 dimension (must be 7)
  - the `mask`
  - `auto_map` targets (`openvla/openvla-7b--` vs shipped code)

### Default OpenVLA checkpoint's gripper is probably untrained (upstream checkpoint issue, not fixed)

- `toasta18/openvla-so101-cylinder@f00833c`: the config's action mask was edited after upload to
  un-normalize the gripper. The model's own `dataset_statistics.json` still records OpenVLA's
  forced EEF_POS training mask `[T]*6+[F]`, so the gripper was likely trained clipped to the top
  bin, and the output is expected near 52.7 %.
- Not fixable client-side without retraining. Confirm on hardware (14-09 human check). If confirmed,
  candidates are: a re-fine-tune with the gripper normalized, or holding the current gripper for
  this checkpoint.

## From 14-10 (Octo)

### `check_checkpoint.py` has no `octo` group

- The Octo scan for 14-10 was done ad hoc. It covered 12 Hub searches plus four SO-101 authors'
  full model lists, about 26 repos. It is recorded in `checkpoint_candidates.md` (`backend:octo` /
  `backend:octo-selection`, both hand-written) and in 14-10-SUMMARY.md.
- An official-format Octo checkpoint has no lerobot `input_features`. A group would instead need
  to check:
  - that `config.json`, `dataset_statistics.json` and `example_batch.msgpack` are present (several
    SO-101 repos ship only step params)
  - flat vs per-dataset statistics
  - the action `mask` and dimension
  - the `example_batch` observation keys (`proprio` is unsupported by `octo_server.py`) and the
    window size
  - `finetune_config.json`'s camera mapping and normalization type

### Joint-space SO-101 Octo checkpoints exist only as an unofficial PyTorch port (not integrated)

- `khanhnd61/octo-small_so101-multi-task-clean` and its siblings are `type: octo` lerobot-format
  checkpoints with absolute 6-D joint actions, which would need no IK. They come from the fork
  `khanhnd61-vr/lerobot` (or the `vla.simd` GGUF runtime), not `octo-models/octo`.
- lerobot 0.6.1 has no `octo` policy. Integrating them would mean installing an unaudited fork.
  That is a supply-chain decision for the human, outside this plan.
- `backends/octo.py` already has the `joint_absolute` branch such a checkpoint would use.

### Octo returns one step per request, not examples/04's temporal ensembling

- `octo_server.py` executes only the first step of each 4-step chunk (receding horizon, 1 step).
- If live runs are jittery, a server-side `TemporalEnsembleWrapper` equivalent is a contained
  follow-up. It would be stateful like the history window already is.

## From 14-11 (RDT-1B)

### No SO-101 RDT checkpoint exists, so `--backend rdt1b` is out of distribution

- **Not fixable without training.** A scan of the HF Hub on 2026-10-01 found no RDT fine-tune for
  SO-101, SO-100 or Koch (see `checkpoint_candidates.md`, `backend:rdt1b`).
- The official model card says RDT "cannot yet generalize to new robot platforms (not seen in the
  pre-training datasets)".
- So `robotics-diffusion-transformer/rdt-1b` on SO-101 proves the pipeline (load, warm, serve,
  convert, validate, drive) but not the task.
- **Follow-up:** fine-tune RDT on this repo's SO-101 lerobot data, using the upstream README's
  `HDF5VLADataset` guide.
  - Fill slots 0-4 with `radians(JOINT_ORDER[:5] degrees)` and slot 10 with `gripper % / 100`.
    That is exactly `backend_configs/rdt1b.json`'s `state_indices` / `joint_units` /
    `gripper_norm_to_percent`.
  - Train at the recording rate (30 Hz, the profile's `control_hz`).
  - Then add the result as a reviewed profile in both `rdt1b_server.CHECKPOINT_PROFILES` and
    `rdt1b.json`.
  - Upstream recommends at least 150K steps. That is a training-budget decision, outside this phase.

### `rdt1b_server.py` serves joint-position profiles only

- The client's `eef_pose` branch (absolute EEF position plus 6-D rotation, through
  `eef_to_joint.solve_ik()`) is implemented and tested.
- **Why the server refuses EEF profiles:** RDT trains with state and action in the same slots
  (`train/train.py` uses the state indicator as the action mask). An EEF checkpoint would therefore
  need the current EEF pose as its state input. The HTTP wire contract carries only `joint_state`,
  and the Colab side has no FK.
- **Contained follow-up if an EEF RDT checkpoint is ever adopted:** add an optional `eef_pose` field
  to `HttpAdapterActionSource`'s payload (client-side FK), or run FK on the server.

### `check_checkpoint.py` has no `rdt1b` group

- The RDT scan for 14-11 was done ad hoc and is recorded in `checkpoint_candidates.md`
  (`backend:rdt1b` / `backend:rdt1b-selection`, both hand-written) and in 14-11-SUMMARY.md.
- RDT checkpoints carry no per-robot slot map, so a group could only check the network shape:
  - `config.json` `action_dim == state_token_dim == 128`
  - `lang_token_dim == 4096`, `img_token_dim == 1152`
  - `img_pos_embed_config == [[image, [2, 3, -729]]]`
  - that a `pytorch_model.bin` is present
- **The slot map itself must come from the fine-tune's documentation.**

### RDT returns one chunk step per request, not agilex_inference's full-chunk execution

- Upstream deployment executes all 64 steps of each chunk, with interpolation, before inferring
  again. `rdt1b_server.py` instead returns one absolute target about one client control period ahead
  (step 14 at 2 Hz), as a receding horizon.
- If live motion is jerky or too slow, streaming the chunk to the client's local waypoint
  interpolation (`--execution-hz`) is a contained follow-up. It would change `HttpAdapterActionSource`'s
  one-action contract.

## From 14-12 (GR00T N1.5)

### Range-normalized SO-101 N1.5 checkpoints are refused, not converted

- Of 33 Isaac-GR00T-format SO-100/SO-101 N1.5 checkpoints on the Hub, 19 were recorded in lerobot's
  range mode (arm joints in [-100, 100], clamped at the calibration limits). That includes NVIDIA's own
  `nvidia/SO_ARM_Starter_Gr00t`. Three more use the pre-0.2 offset calibration (shoulder_lift up to
  ~199). `groot_n15_server.py` refuses all of them; only degree-mode checkpoints are identity-mapped.
- **Contained follow-up:** a `range_m100_100` profile that converts on both sides with THIS robot's
  calibration (`action_contract.CALIBRATION_PATH`). The arithmetic is
  `deg = norm / 100 * (range_max - range_min) / 2 * 360 / 4095`, sign-flipped by `drive_mode`
  (lerobot 0.6.1 `MotorsBus._normalize`). The gripper is RANGE_0_100 in both modes.
  - The server would need the same scale for the proprio. So the wire contract would have to carry
    the calibration, or the client would have to send pre-converted state.
  - The offset-calibration mode has no reliable conversion and should stay refused.

### `check_checkpoint.py` has no `groot-n15` group

- The 14-12 scan was ad hoc. It is recorded in `checkpoint_candidates.md` (`backend:groot-n15` /
  `backend:groot-n15-selection`, both hand-written) and in 14-12-SUMMARY.md.
- A group would read `config.json` (`model_type: gr00t_n1_5`) and `experiment_cfg/metadata.json`'s
  `new_embodiment` entry, then reuse `groot_n15_server._check_checkpoint_layout()` /
  `classify_action_units()`.

### GR00T N1.5 returns one chunk step per request, not eval_lerobot.py's chunk execution

- NVIDIA's client executes 8 of the 16 chunk steps at about 50 Hz before inferring again.
  `groot_n15_server.py` returns one absolute target about one client control period ahead (step 14
  of 16 at 2 Hz), the same receding-horizon choice as RDT-1B (14-11).
- A blog commenter reports jerky N1.5 SO-101 motion even with NVIDIA's client. If live motion is
  jerky, streaming the chunk to the client's `--execution-hz` interpolation is the same contained
  follow-up as RDT's.

### No SO-101 GR00T N1.5 fine-tune on this rig

- The default is a community fine-tune on another rig and scene (3 tasks). Task-meaningful motion
  needs a fine-tune on this repo's own SO-101 lerobot data with NVIDIA's tutorial:
  `scripts/gr00t_finetune.py --data-config so100_dualcam` at `n1.5-release`, in degree mode, ~25 GB of
  VRAM (or `--no-tune_diffusion_model`). Add the result as a reviewed profile in both
  `groot_n15_server.CHECKPOINT_PROFILES` and `groot_n15.json`.
