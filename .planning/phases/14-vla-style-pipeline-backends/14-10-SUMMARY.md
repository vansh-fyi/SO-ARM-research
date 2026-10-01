---
phase: 14-vla-style-pipeline-backends
plan: 10
subsystem: control/vla_bridge
status: complete
tags: [backends, octo, tier-c, http-adapter, vlab-08, jax, eef-delta, ik, so101, colab]
requires:
  - "14-01: base_http_adapter.HttpAdapterActionSource / build_http_adapter_backend, register(), load_backend_config()"
  - "14-02: --backend registry dispatch in run_vla_episode.py"
  - "14-03: eef_to_joint.forward_kinematics() / solve_ik(), the Tier C resident-adapter shape"
  - "14-09: per-checkpoint profile pattern (client/server profile tables pinned equal)"
provides:
  - "vla_bridge.octo_server: OctoModel, create_app(infer), main(), CHECKPOINT_PROFILES, preprocess_image(), _download_pinned(), _pin_text_encoder()"
  - "vla_bridge.backends.octo: build (--backend octo, HTTP lane), make_convert_fn(profile, joint_limits_deg), octo_action_to_eef_delta(), joint_absolute_action(), ACTION_SPACES"
  - "backend_configs/octo.json (pinned SO-101 default + 2 official alternates, action_space eef_delta)"
  - "policy_server.ipynb / policy_server_launch.md: Tier C Octo section (octo-* cells)"
affects: [14-11, 14-12 (remaining Tier C backends), 14-15 (end-of-phase UAT / legitimacy check), 16 (evaluation)]
tech-stack:
  added: []
  patterns:
    - "Pinned partial snapshot: one checkpoint step plus metadata downloaded at a pinned sha, never the library's own unpinned hf:// loader"
    - "Pinned third-party tokenizer: the checkpoint config's 't5-base' is rewritten to a local google-t5/t5-base@sha copy; other encoder names refused"
    - "Stateful resident adapter: a server-side observation-history window, reset on a new instruction, an explicit reset, or an idle gap"
    - "Pose-difference EEF deltas (rpy differences) composed exactly at the current FK pose into solve_ik's left-multiplied extrinsic rotation"
key-files:
  created:
    - control/vla_bridge/octo_server.py
    - control/vla_bridge/backends/octo.py
    - control/vla_bridge/backend_configs/octo.json
    - control/test_backend_octo.py
  modified:
    - control/vla_bridge/policy_server.ipynb
    - control/vla_bridge/policy_server_launch.md
    - control/vla_bridge/checkpoint_candidates.md
    - .planning/phases/14-vla-style-pipeline-backends/deferred-items.md
decisions:
  - "Default checkpoint sapanostic/octo-so101-3hz-finetune@a45ff97 (step 10000): the only loadable official-format Octo checkpoint trained on real SO-101 data with a documented data pipeline, and it needs no proprio"
  - "action_space = eef_delta, confirmed from the checkpoint's own dataset_statistics.json (7-D, mask [T]*6+[F], cm/0.1-rad magnitudes) and the public RLDS card (action/eef_delta_m_rad, Pinocchio FK, gripper min-max over [-10, 100]). No official-format Octo checkpoint on the Hub is joint-space"
  - "Official MIT rail-berkeley/octo-{small,base}-1.5 with bridge_dataset statistics are reviewed alternates (eef_delta, gripper 1 = open)"
  - "The plan's 'octo-models/octo-base-1.5' HF id does not exist; the official checkpoints live under rail-berkeley/ (the README links them)"
  - "The angular slots are roll/pitch/yaw differences (Octo's relabel_actions and the SO-101 conversion), so the client converts them exactly at the current FK pose before solve_ik"
  - "The server pins one checkpoint step plus google-t5/t5-base@a9723ea. It returns the first chunk step and keeps Octo's 2-frame history server-side"
  - "main() refuses jax != 0.4.20, flax != 0.7.5, tensorflow != 2.15.0 or numpy != 1.24.3. The Colab cells use an isolated Python 3.10 uv venv with octo-models/octo cloned at 241fb35; port 3004"
  - "The joint_absolute branch is implemented and tested (it is the plan's other branch) but no reviewed checkpoint selects it"
metrics:
  duration: "~55 min"
  completed: 2026-10-01
  tasks: 2
  files: 8
---

# Phase 14 Plan 10: Octo Backend Summary

`--backend octo` is a Tier C backend on the same lane as OpenVLA and OpenVLA-OFT. A Colab Flask adapter runs Octo (JAX/Flax) through octo-models/octo's own `OctoModel.load_pretrained` / `create_tasks` / `sample_actions` API. By default it serves a real **SO-101** fine-tune, `sapanostic/octo-so101-3hz-finetune@a45ff97`. That checkpoint's action space was confirmed as a 7-D end-effector **delta** from its own statistics and its dataset card. The client composes the delta's roll/pitch/yaw differences at the current FK pose, then converts it to an absolute joint target with `eef_to_joint.solve_ik()` before `validate_action()`.

## Action-space confirmation (Assumption A4, verified live 2026-10-01)

14-RESEARCH's A4 ("Octo is generally Cartesian/delta-EEF") is **confirmed for every loadable official-format Octo checkpoint found**, by reading each one's own files rather than assuming.

### Search
- I ran 12 Hub searches, then listed every model from the SO-101 authors that turned up: `sapanostic`, `ivlabs`, `kartiksonu` and `khanhnd61`.
- That gave about 26 candidate repos. For each, I fetched the API metadata, `config.json`, `finetune_config.json` and `dataset_statistics.json`.
- I also decoded the `example_batch.msgpack` array shapes with a minimal msgpack reader.
- The full table is in `checkpoint_candidates.md` (`backend:octo`, `backend:octo-selection`).

### Chosen default: `sapanostic/octo-so101-3hz-finetune` @ `a45ff970242a66e268871a5f64530947143f9a6e`, step 10000
- **What it is.** Its `finetune_config.json` describes `rail-berkeley/octo-base-1.5` fine-tuned in `encoder_and_head` mode for 10000 steps (its `num_steps`).
  - Window 2, horizon 4, `normal` normalization.
  - Primary camera = `image_1`, wrist = `image_2`.
  - The dataset is `so101_pick_cube_v2_merged_rlds_3hz`.
- **Its statistics.** `dataset_statistics.json` is flat (a single dataset), 7-D, with mask `[T]*6+[F]`.
  - p01..p99 position deltas are a few cm, rotation deltas within ±0.17 rad, and the raw gripper is 0.10-0.64.
  - It covers 100 trajectories and 6280 transitions.
- **Its data pipeline, as documented:**
  - **Source dataset** `sapanostic/so101_pick_cube_v2_merged`: `so_follower`, 15 Hz, action names `shoulder_pan.pos .. gripper.pos`, `right`/`left`/`wrist` cameras, task "pick up the cube".
  - **RLDS card** `ivlabs/so101_pick_cube_v2_merged_rlds_3hz` (Apache-2.0): `action/eef_delta_m_rad` = `[dx, dy, dz, droll, dpitch, dyaw, gripper_normalized]`. It uses Pinocchio FK on the SO101 URDF.
  - **`mixture_metadata.json`:** `gripper_min_deg -10`, `gripper_max_deg 100`, `binarize_gripper false`, 15 Hz resampled to 3 Hz.
- **Action space: `eef_delta`.** Positions are metres in the SO-101 base frame. `so101_new_calib.urdf`'s `shoulder_pan` origin matches our `So-101.urdf` exactly. The gripper is absolute, `-10 + 110·g` percent.

### Alternates: `rail-berkeley/octo-small-1.5` @ `dc9aa30` and `octo-base-1.5` @ `ee3c10e` (MIT, official)
- **Statistics.** 26 OXE keys. The profile uses `bridge_dataset`, the key `examples/04` uses on the WidowX.
- **Bridge action.** In `oxe_standardization_transforms.bridge_dataset_transform`, `relabel_actions` turns the action into WidowX state differences (metres plus rpy differences, at 5 Hz). The gripper is binarized, 1 = open, and returned raw because its mask is false.

### Rejected candidates
- **Proprio required:** `ivlabs/octo-finetune-v0.01`/`v0.02` and `sapanostic/so101_finetune_octo_0.01` need a 7-D EEF-pose proprio from their authors' own FK. The adapter refuses proprio checkpoints.
- **Metadata stub:** `ivlabs/octo-finetune-v0.03`.
- **Unidentified dataset:** `sapanostic/octo-so101-finetuned` is head-only, its dataset version is unidentified, and its binarized gripper is mostly 0.
- **No finetune config:** `ivlabs/octo-so101-finetuned-v0.3`.
- **Unloadable:** `kartiksonu/*`, `ivlabs/fine-tune-v-0-*` and `sapanostic/octo-v05-test` ship params only, with no config or statistics.
- **Not official Octo:** the joint-space `khanhnd61/octo*_so101*` checkpoints are a PyTorch port in the unofficial fork `khanhnd61-vr/lerobot`, not `octo-models/octo`. See deferred-items.md.

### Loading path (github.com/octo-models/octo @ `241fb35`, main HEAD; tag v1.5 = `5eaa5c6`)
- **Files read:** README, `requirements.txt`, `octo/model/octo_model.py`, `octo/utils/gym_wrappers.py`, `octo/model/components/tokenizers.py`, `octo/data/utils/text_processing.py`, `octo/data/utils/data_utils.py`, `octo/data/oxe/oxe_standardization_transforms.py`, `examples/03_eval_finetuned.py`, `examples/04_eval_finetuned_on_robot.py`, `examples/envs/widowx_env.py`.
- **Load:** `OctoModel.load_pretrained(path, step)`.
- **Inference:** `create_tasks(texts=[...])`, then `sample_actions(obs, tasks, unnormalization_statistics=stats["action"], rng=PRNGKey)`, which returns `(batch, horizon, dim)`.
- **Observations:** `(1, window, H, W, 3)` plus `timestep_pad_mask`, following `HistoryWrapper`: the first frame is repeated and the older slots are padded.
- **Images:** Lanczos resize to the example-batch size, plus `ResizeImageWrapper`'s average crop (sqrt(0.9)) on the primary image only.

### Pinning (T-14-06)
- **The gaps:**
  - Octo's `hf://` loader calls `snapshot_download(repo)` with no revision, so it gets `main`, and it downloads every step. That would be about 40 GB for some of these repos.
  - `HFTokenizer` and `LanguageTokenizer` resolve `"t5-base"` from the Hub at `main`.
- **The fixes:**
  - `_download_pinned` fetches the pinned sha's metadata plus one step only, using `allow_patterns`.
  - `_pin_text_encoder` fetches `google-t5/t5-base@a9723ea` (`config.json`, `spiece.model`, `tokenizer.json`) and rewrites the local `config.json` to point at it. It refuses any encoder other than `t5-base`.
  - The Colab cell clones octo at `241fb35` and asserts `HEAD`.

## What Was Built

### Task 1: adapter, client and config (commits 21982bb RED, 2f08473 GREEN)
- **`octo_server.py`:** same contract as `openvla_server.py`.
  - **Loading:** loads once, warms up (JIT), then calls `reset()` so the warm-up frame isn't part of the first request's history.
  - **Responses:**
    - `/health` returns metadata: action space, dimension, horizon, window, mean/std/mask, camera mapping, step and text encoder.
    - 429 when busy.
    - 400 for a bad body, including a non-bool `reset`.
    - 500 for a non-finite or wrong-shaped chunk.
  - **Statistics:** flat statistics are used directly. Per-dataset statistics use the profile key, the sole key, or `--unnorm-key`.
  - **Refusals:** it refuses proprio checkpoints, and a pinned load of an unprofiled checkpoint without both `--revision` and `--step`.
  - **History:** resets on a new instruction, on `reset: true`, or after 10 s idle.
  - **Dependency guard and port:** it refuses jax, flax, tensorflow or numpy at versions other than the pins; it serves on :3004.
- **`backends/octo.py`:**
  - Registered with `@register("octo")` on the HTTP lane.
  - Endpoint rules are identical to openvla's: a clear `SystemExit` with no endpoint, and a rejection of a gRPC `host:port`.
  - A checkpoint without a profile is refused.
  - `make_convert_fn(profile)`:
    - `eef_delta`: `octo_action_to_eef_delta` (exact rpy-difference composition plus the gripper range), then `solve_ik(joint_state, delta, joint_limits_deg=...)`.
    - `joint_absolute`: identity mapping onto `JOINT_ORDER`.
- **`backend_configs/octo.json`:** checkpoint, revision, `endpoint_url: null`, `action_space: "eef_delta"`, instruction, three checkpoint profiles (revision, step, unnorm key, action space, gripper range, control Hz) and notes.
- **Tests:** 69, covering both conversion branches. Which branch runs is driven by the config's `action_space`, as `test_build_action_space_from_config_selects_the_branch` shows.

### Task 2: regression tests (commit 861bc4a)
- **Default SO-101 round trip:**
  - Path: real `create_app` → real `build()` convert_fn → real `HttpAdapterActionSource` → real ikpy `solve_ik` → real `validate_action`.
  - The action is finite, in `JOINT_ORDER`, and non-held.
  - The FK of the result lands the commanded Cartesian step within 3 mm, and the gripper is 45 %.
- **Official alternate:** the same path with the `bridge_dataset` statistics.
- **History:** the window fills across consecutive client steps.
- **Failure:** a NaN chunk → 500 → `http-error-holding-position`.

### Colab launch and docs (commit 5804995)
- **Cells:** `octo-*` cells appended to `policy_server.ipynb`:
  - **Install:** an isolated Python 3.10 uv venv. It clones `octo-models/octo` at `241fb35` and checks `HEAD`. It installs `-r requirements.txt -e octo`, `jax[cuda11_pip]==0.4.20` from Google's `jax_cuda_releases` index, and `flax==0.7.5`, `orbax-checkpoint==0.4.4`, `transformers==4.34.1`, `numpy==1.24.3`, `opencv-python-headless==4.11.0.86`, `flask`, `pillow`, `requests`, `huggingface_hub`.
  - **Import check:** checks the exit status, `OctoModel` importability and that JAX's backend is `gpu`.
  - **Launch:** on :3004, with `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
  - Then tunnel, a local-run note, and teardown.
- **Launch md:** `policy_server_launch.md` regenerated with `scripts/sync_policy_notebook.py`. The diff only adds lines.
- **Candidates:** `checkpoint_candidates.md` gains the `backend:octo` and `backend:octo-selection` sections. Existing sections are unchanged.
- **Deferred items:** three new entries in `deferred-items.md`.

## Verification
- `cd control && <main .venv>/bin/python -m pytest -q test_backend_octo.py`: **73 passed**.
- Full suite: **638 passed** (fork base 565, plus 73).
- `run_vla_episode.py --help` lists `--backend {act,diffusion,groot-n17,octo,openvla,openvla-oft,pi0,pi05,scripted,smolvla,wallx}`.
- The new notebook code cells and the inner import-check snippet compile.

## Deviations from Plan

### Auto-fixed Issues

**1. [Plan-directed choice] Default checkpoint is a real SO-101 fine-tune, and it is EEF-delta**
- **Found during:** Task 1, HF Hub search.
- **What happened:**
  - The plan names `octo-models/octo-base-1.5` / `octo-small-1.5` as examples. Those ids don't exist on the Hub; the official checkpoints are `rail-berkeley/octo-*-1.5`.
  - The plan also says to prefer a genuine SO-101 fine-tune if one exists. One exists, but every official-format one is EEF-delta, so IK is still needed.
- **Fix:** the default is the SO-101 fine-tune, with `action_space: "eef_delta"` read from its statistics and dataset card. The official checkpoints are kept as alternates.
- **Commit:** 2f08473.

**2. [Rule 1 - Bug] Angular deltas are rpy differences, not left-multiplied rotations**
- **Issue:** both the SO-101 conversion and Octo's own bridge relabeling define the action as pose differences, `rpy_t - rpy_{t-1}`. `solve_ik` expects an extrinsic rotation composed on the left. Passing the values through unchanged would be wrong away from rpy ≈ 0.
- **Fix:** `octo_action_to_eef_delta` computes `R(rpy_cur + d) @ R(rpy_cur)^T` at the current FK pose. A test pins it to be exact.
- **Commit:** 2f08473.

**3. [Rule 2 - Security, T-14-06] Pinned revision, single step and pinned tokenizer**
- **Issue:** the plan pins only the GitHub commit. Octo's `hf://` loader and the `t5-base` lookups both resolve `main`.
- **Fix:** `_download_pinned` and `_pin_text_encoder`, as described above.
- **Commit:** 2f08473.

**4. [Rule 1 - Bug] The server keeps Octo's observation window**
- **Issue:** the default checkpoint has window 2, but the 14-01 wire contract is stateless, with one frame per request.
- **Fix:** the server keeps `HistoryWrapper`-equivalent history. It resets on a new instruction, an explicit reset or an idle gap. The response contract is unchanged.
- **Commit:** 2f08473.

**5. [Rule 1 - Bug] No `normalization_type` passed to the jitted `sample_actions`**
- **Issue:** a str-enum argument to a jitted function is not a valid JAX type.
- **Fix:** the server calls it exactly as examples 03/04 do, with the default NORMAL. All reviewed checkpoints use `normal`; the 'bounds'-normalized v0.03 stub is rejected anyway.
- **Commit:** 2f08473.

**6. [Rule 2 - Correctness] Guards**
- **What was added:**
  - A refusal of proprio checkpoints.
  - A refusal of unprofiled checkpoints in the client, and in the server without `--revision` and `--step`.
  - The dependency-pin guard.
  - A test that pins the client and server profile tables equal.
- **Commit:** 2f08473.

**7. [Rule 3 - Blocking] Colab cells, regenerated launch md, candidate table and deferred items**
- These files are outside the plan's list but needed for the human check (14-03/14-08/14-09 precedent).
- **Commit:** 5804995.

### Notes
- Task 2's round-trip tests are regression guards for Task 1 code, so they have no RED stage (same as 14-05 through 14-09).
- There are no new local dependencies. JAX, Octo and TF are Colab-only, in their own venv.

## TDD Gate Compliance
- **RED:** `test(14-10)` 21982bb, which failed with an ImportError (`octo_server` missing).
- **GREEN:** `feat(14-10)` 2f08473, with 69 passing.
- **Regression tests:** `test(14-10)` 861bc4a, with 73 passing.

## Unverified (stated explicitly)
- **No real inference here.** There is no GPU, no local JAX or Octo, and the 0.8 GB params were never downloaded. The loading and un-normalization logic was checked against upstream source and the real config, statistics and example-batch files.
- **orbax compatibility.** The 2026 fine-tunes ship a `_METADATA` file next to `default/checkpoint`, which suggests a newer orbax than the official checkpoints used. The `checkpoint` file is byte-for-byte the same size as the official base's (809,848,679), so it is very likely the same aggregate msgpack format that orbax-checkpoint 0.4.4 reads. This is unverified until the cell runs.
- **Colab install.** Whether uv resolves Octo's full `requirements.txt` together with the cuda11 jaxlib on the current Colab image is unknown until the cell runs. Inference latency against the client's 5 s timeout is also unmeasured.
- **Training code.** The default checkpoint's `so101_dataset_transform` is unpublished. The mapping of `image_1` to the `left` camera is inferred from the dataset's camera order.
- **Frames.** The dataset's EE frame (`so101_new_calib.urdf` gripper frame) differs from our `eef` link, so the orientation deltas are approximate. The real-joint sign/zero vs URDF mapping is still 14-03's open human check 3. The WidowX → SO-101 frame mapping for the bridge alternates is an identity assumption.
- **Control rate.** 3 Hz-trained steps are executed at the loop rate, and the safety clamp bounds them.

## Package Legitimacy Evidence (for the end-of-phase human check)
- **`github.com/octo-models/octo`** (in the 14-RESEARCH audit; org confirmed in 14-03's check):
  - Repo: created 2023-12-13, MIT, about 1791 stars and 281 forks, not archived.
  - Pinned to `241fb3514b7c40957a86d869fecb7c7fc353f540` (main HEAD, 2024-07-31, "Merge pull request #122 … update setup.py to find all packages"). Tag `v1.5` = `5eaa5c6`.
  - It is cloned only on Colab, never locally.
- **HF `rail-berkeley/octo-{small,base}-1.5`:** the official checkpoints the octo README links to. MIT.
- **New and not in the 14-RESEARCH audit:**
  - **`sapanostic/octo-so101-3hz-finetune`:** community, no license, no card, 3 downloads. It is weights and JSON only, with no code.
  - **Its datasets:** `sapanostic/so101_pick_cube_v2_merged` and `ivlabs/so101_pick_cube_v2_merged_rlds_3hz` (Apache-2.0).
  - **`github.com/kvablack/dlimp` @ `5edaa46`:** a git dependency pinned by Octo's own `requirements.txt` and imported by `octo.data.utils.data_utils` at inference time. Apache-2.0, about 83 stars, by Kevin Black, an Octo author.
  - **HF `google-t5/t5-base` @ `a9723ea`:** Apache-2.0, the canonical home of the legacy `t5-base` id. Only tokenizer and config files are fetched.
- **PyPI and index pins (Colab venv only)**, all checked on pypi.org:
  - `jax 0.4.20` (google/jax, 2023-11-02), plus `jaxlib-0.4.20+cuda11.cudnn86-cp310` from `storage.googleapis.com/jax-releases/jax_cuda_releases.html`, the official JAX wheel index the octo README names.
  - `flax 0.7.5` (2023-10-28), `orbax-checkpoint 0.4.4` (2023-12-01), `tensorflow 2.15.0`, `numpy 1.24.3`, `transformers 4.34.1` (huggingface), `distrax 0.1.5` / `chex 0.1.85` / `optax 0.1.5` (deepmind), `tensorflow-probability 0.23.0`, `tensorflow-datasets 4.9.2`, `tensorflow-graphics 2021.12.3`, `ml-dtypes 0.2.0`, `opencv-python-headless 4.11.0.86` (the last release that allows numpy<2)
  - `uv` (Astral; already used by 14-03/14-09's cells)

## Known Stubs
None. `endpoint_url: null` is intentional: it is the per-session ngrok URL, passed with `--server-address`. The `joint_absolute` branch is real and tested; no reviewed checkpoint selects it today (see deferred-items).

## Threat Flags

| Flag | File | Description |
|------|------|-------------|
| threat_flag: network-endpoint | control/vla_bridge/octo_server.py | A new Colab HTTP adapter (`/health`, `/action`) behind an ngrok tunnel, the same exposure class as the OpenVLA, OFT and Wall-X adapters. It accepts only images, an instruction, joint numbers and a reset bool. Requests are size-capped (20 MB), images are capped at ≤1024 px, and concurrency gets 429. |
| threat_flag: new-dependency | control/vla_bridge/policy_server.ipynb | A Colab-only isolated venv with Octo's pinned JAX, Flax and TF stack, plus a git clone of octo-models/octo at a pinned commit. Octo's `requirements.txt` pulls `kvablack/dlimp` from git at a pinned commit. Nothing is installed locally. |

T-14-05 holds: IK clamps to URDF ∩ calibrated limits, and every action goes through `run_episode()`'s unmodified `safety_validator.validate_action()`. T-14-06 holds: GitHub commit, HF revision, checkpoint step and T5 revision are all pinned.

## Human Checks Deferred
1. **Octo live hardware (Task 2):**
   - **test:**
     - On a Colab GPU runtime, run `octo-install`. Confirm the printed versions (jax 0.4.20, flax 0.7.5, orbax-checkpoint 0.4.4, tensorflow 2.15.0) with no `RuntimeError`.
     - Upload `octo_server.py`, then run `octo-launch` and `octo-tunnel`.
     - Run `python control/run_vla_episode.py --backend octo --server-address https://<tunnel> --instruction "pick up the cube" --out control/outputs/<dir> --max-steps 20` against the real SO-ARM101.
   - **expected:**
     - `/health` ready, with `checkpoint` `sapanostic/octo-so101-3hz-finetune`, `step` 10000, `action_space` `eef_delta`, `window_size` 2, `action_horizon` 4, and `image_keys` primary=`overhead_left`, wrist=`wrist`.
     - At least one step logs a validated, non-held action, with `model_version` `…@a45ff970242a` and no `holding-position` flag, and the robot visibly moves.
     - Also note: whether orbax loaded the checkpoint (that is the first risk), inference time per request (it must stay under 5 s), and gripper targets of roughly 0-60 %.
   - **why_human:** this needs real hardware and a live Colab GPU, and the cells haven't been run.
2. **Optional official alternate:**
   - **test:** relaunch the server with `--checkpoint rail-berkeley/octo-small-1.5` and run the client with the same `--checkpoint`.
   - **expected:** IK-converted, validated actions. This is also the fallback if the community checkpoint fails to load.
3. **Package legitimacy and licensing:**
   - **test:**
     - Confirm the evidence above, especially the two new unaudited sources: `github.com/kvablack/dlimp@5edaa46` and HF `google-t5/t5-base@a9723ea`.
     - Decide whether the **unlicensed** community checkpoint `sapanostic/octo-so101-3hz-finetune` is acceptable for this research use.
     - If it isn't, set `checkpoint` in `octo.json` to `rail-berkeley/octo-small-1.5`.
   - **expected:** first-party sources, with no name or org mismatch, and an explicit licensing decision.
   - **why_human:** this is a legitimacy and licensing judgment.
4. **Carried from 14-03 (check 3):** an IK frame and sign sanity check with a small commanded `dz`. Octo's default path depends on it just like OFT's.
5. **Carried from 14-01:** on the HTTP lane, the wrist camera tolerates two independent `cv2.VideoCapture` opens.

## Self-Check: PASSED
- FOUND: control/vla_bridge/octo_server.py, control/vla_bridge/backends/octo.py, control/vla_bridge/backend_configs/octo.json, control/test_backend_octo.py, control/vla_bridge/policy_server.ipynb, control/vla_bridge/policy_server_launch.md, control/vla_bridge/checkpoint_candidates.md
- FOUND commits: 21982bb, 2f08473, 861bc4a, 5804995
- No deletions and no STATE.md/ROADMAP.md changes since base ca2fc48.
