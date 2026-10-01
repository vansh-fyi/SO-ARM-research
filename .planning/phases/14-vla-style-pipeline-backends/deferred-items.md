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
