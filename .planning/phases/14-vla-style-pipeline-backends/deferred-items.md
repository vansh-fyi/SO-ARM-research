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
