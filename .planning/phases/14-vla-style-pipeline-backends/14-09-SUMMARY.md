---
phase: 14-vla-style-pipeline-backends
plan: 09
subsystem: control/vla_bridge
status: complete
tags: [backends, openvla, tier-c, http-adapter, vlab-06, so101-joint-space, ik, colab]
requires:
  - "14-01: base_http_adapter.HttpAdapterActionSource / build_http_adapter_backend, register(), load_backend_config()"
  - "14-02: --backend registry dispatch in run_vla_episode.py"
  - "14-03: openvla_oft_server.py resident-adapter shape, eef_to_joint.solve_ik(), openvla_oft_action_to_eef_delta()"
provides:
  - "vla_bridge.openvla_server: OpenVLAModel, create_app(infer), main(), CHECKPOINT_PROFILES, preprocess_image(), build_prompt(), _localize_remote_code()"
  - "vla_bridge.backends.openvla: build (--backend openvla, HTTP lane), make_convert_fn(action_space, joint_limits_deg), so101_joint_action(), ACTION_SPACES"
  - "backend_configs/openvla.json (pinned default + 4 reviewed LIBERO alternates, per-checkpoint action_space)"
  - "policy_server.ipynb / policy_server_launch.md: Tier C OpenVLA section (openvla-* cells)"
affects: [14-10, 14-11, 14-12 (remaining Tier C backends reuse the remote-code pin + per-checkpoint action_space pattern), 14-15 (end-of-phase UAT / legitimacy check), 16 (evaluation)]
tech-stack:
  added: []
  patterns:
    - "Cross-repo trust_remote_code pinned by localizing the reviewed code into the pinned snapshot and rewriting auto_map (foreign auto_map refused)"
    - "Per-checkpoint action_space profile selects the client conversion: joint-space pass-through or delta-EEF -> solve_ik"
    - "Client/server checkpoint profile tables pinned equal by a test"
key-files:
  created:
    - control/vla_bridge/openvla_server.py
    - control/vla_bridge/backends/openvla.py
    - control/vla_bridge/backend_configs/openvla.json
    - control/test_backend_openvla.py
  modified:
    - control/vla_bridge/checkpoint_candidates.md
    - control/vla_bridge/policy_server.ipynb
    - control/vla_bridge/policy_server_launch.md
    - .planning/phases/14-vla-style-pipeline-backends/deferred-items.md
decisions:
  - "Default checkpoint toasta18/openvla-so101-cylinder@f00833c: the only Hub OpenVLA fine-tune on real SO-101 lerobot data. Its actions are absolute JOINT_ORDER degrees (gripper percent), so it needs no IK, as 14-RESEARCH Open Question 3 prefers"
  - "Official MIT openvla/openvla-7b-finetuned-libero-{object,spatial,goal,10} are kept as reviewed alternates on the IK path; openvla_oft_action_to_eef_delta is reused unchanged (same LIBERO OSC_POSE units and gripper convention)"
  - "The conversion is chosen per checkpoint from openvla.json checkpoints[<id>].action_space; an unprofiled checkpoint raises SystemExit"
  - "No openvla GitHub clone: the official inference path is HF AutoClasses + trust_remote_code. The remote code is pinned to openvla/openvla-7b@47a0ec7 by copying it into the snapshot, because AutoProcessor drops code_revision"
  - "The server returns the raw 7-D predict_action() output with no chunk; a non-finite or wrong-shape output gives 500, so the client holds"
  - "Single image, overhead_left by default, with OpenVLA's eval-time sqrt(0.9) center crop reproduced in PIL; no TensorFlow"
  - "main() refuses transformers != 4.40.1, tokenizers != 0.19.1, or timm outside 0.9.10-0.9.16; the Colab cells use an isolated Python 3.10 uv venv"
  - "sdpa attention by default, so no flash-attn build is needed"
metrics:
  duration: "~70 min"
  completed: 2026-10-01
  tasks: 2
  files: 8
---

# Phase 14 Plan 09: OpenVLA Backend Summary

`--backend openvla` is a Tier C backend on the same lane as 14-03's OpenVLA-OFT. A Colab Flask adapter loads OpenVLA-7B through the official HF AutoClass path, with its cross-repo remote code pinned. By default it serves a real **SO-101 joint-space** fine-tune, `toasta18/openvla-so101-cylinder@f00833c`, which needs no IK. The four official LIBERO checkpoints are also supported: their 7-D delta-EEF output is explicitly converted with `eef_to_joint.solve_ik()` before `validate_action()`.

## Checkpoint and action space (live HF Hub and GitHub verification, 2026-10-01)

### Search and candidates
I ran 16 Hub searches (`openvla so101`, `openvla_so`, `openvla koch`, `openvla-7b-finetuned`, `openvla`, ...), which returned 771 repos. 19 of them have SO/Koch/lerobot-style names. Each candidate's API metadata, `config.json` and `dataset_statistics.json` were fetched. The full table is in `checkpoint_candidates.md` (`backend:openvla`, `backend:openvla-selection`).

### Chosen default: `toasta18/openvla-so101-cylinder` @ `f00833ce58510a955b981ee7e9ba8e9ef4e2ef2e`
- **Model:** a full OpenVLA-7B bf16 fine-tune (15.1 GB). It ships no code; `auto_map` points to `openvla/openvla-7b--...`.
- **Training data:**
  - The source dataset is `toasta18/so101_task_yellow_cylinder` @ `44cf284`: lerobot v3.0, `so_follower`, 50 episodes at 30 fps, `overhead` + `wrist` cameras.
  - Its action names are exactly `shoulder_pan.pos .. gripper.pos`.
  - It has one task: "Pick up the black cylinder and place it on the plate."
  - The RLDS conversion is `toasta18/so101_pick_cylinder_rlds` @ `0e6ee0d`, with `image`, `wrist_image` and a 7-D `action`.
- **Units:** the `so101_pick_cylinder` stats are absolute joints in degree mode.
  - The model stats' min/max equal the lerobot dataset's action min/max exactly. Slots 0-4 are the arm joints in `JOINT_ORDER`.
  - Slot 5 is a constant 0 pad.
  - Slot 6 is the gripper, at 1.0-53.5 %.
  - This matches `action_contract.ACTION_UNITS`.
- **Unnorm key:** the config embeds the 25 base OXE keys as well, so the unnorm key `so101_pick_cylinder` is explicit in the profile.

### Alternates: `openvla/openvla-7b-finetuned-libero-{object,spatial,goal,10}`
- MIT licensed. The revisions are pinned in `CHECKPOINT_PROFILES`.
- Each has a single key (`libero_*`), 7-D, with q01/q99 within ±0.94 and mask `[T]*6+[F]`.
- **Units match OpenVLA-OFT's LIBERO output, so 14-03's conversion is reused unchanged:**
  - openvla `libero_dataset_transform` flips LIBERO's -1 = open to +1 = open.
  - `run_libero_eval` applies `normalize_gripper_action(binarize)` and then `invert_gripper_action`.
  - The actions are robosuite OSC_POSE inputs.

### Rejected candidates
- **`madokalif/so101-openvla-sft-v2-*-merged`:** the `so101*` stats are 6-D, min/max only, and have **no q01/q99**, which `predict_action` needs. They also ship their own custom code.
- **`madokalif` adapters:** adapter-only, and the v1 ones are self-marked DEPRECATED.
- **`Atabaku/so101-openvla-oft-5000`:** gated, and it is OFT rather than OpenVLA.
- **`phospho-ai/openvla-koch` and `m1b/openvla-koch-*`:** a different robot (Koch). The m1b one also has no stats.

### Loading path (github.com/openvla/openvla @ `c8f03f4`)
- **Sources:** `README.md` quick start, `vla-scripts/deploy.py`, `experiments/robot/openvla_utils.py` / `robot_utils.py` / `libero/run_libero_eval.py` / `libero_utils.py`, `prismatic/vla/datasets/rlds/oxe/{transforms,configs,materialize}.py`, `prismatic/vla/datasets/rlds/utils/data_utils.py`, `prismatic/vla/action_tokenizer.py`, `vla-scripts/finetune.py`, `pyproject.toml`, `requirements-min.txt`.
- **Load:** `AutoProcessor` / `AutoModelForVision2Seq.from_pretrained(..., trust_remote_code=True, torch_dtype=bfloat16, low_cpu_mem_usage=True)`.
- **Prompt:** `In: What action should the robot take to {instruction.lower()}?\nOut:`.
- **Inference:** `predict_action(**inputs, unnorm_key=..., do_sample=False)`, with one image.
- **Image preprocessing:** a center crop with scale 0.9. `finetune.py` defaults to `image_aug=True`, and the README tells LIBERO users to pass `--center_crop True`.

### Remote-code pin (T-14-06)
- **The gap:** transformers 4.40.1 resolves the `openvla/openvla-7b--` remote code at `code_revision`, which `get_class_from_dynamic_module` defaults to `main`. `AutoProcessor` pops `code_revision` before its inner `AutoImageProcessor` load, so even passing it doesn't pin the image processor. I read `dynamic_module_utils.py`, `auto_factory.py`, `processing_auto.py` and `configuration_auto.py` @ v4.40.1 to confirm this.
- **The fix:** `_localize_remote_code` copies the three files from `openvla/openvla-7b@47a0ec7f` into the pinned snapshot and rewrites `auto_map` in all four JSON files. It refuses an `auto_map` that points at any other repo.
- **Why that revision:** its only commit after `31f090d` is a README edit. Its `modeling_prismatic.py` is byte-identical to GitHub @ `c8f03f4`.
- **Integration check:** I ran the real pinned configs plus the real pinned code files for both toasta18 and libero-object through `_localize_remote_code`; all `auto_map` entries became local. I did not load them, because transformers isn't installed locally.

### Gripper caveat for the default
- **Evidence, all from published files rather than from running the model:**
  - Config history: `ed27729` had no `so101` key. `a545284` added it with mask `[T]*6+[F]`. `f00833c` changed it to `[T,T,T,T,T,F,T]`.
  - The model's own `dataset_statistics.json` keeps `[T]*6+[F]`. That is the mask `make_oxe_dataset_kwargs` forces for EEF_POS, the only encoding it accepts.
- **Consequence:** during training the gripper was left un-normalized (1-53) and `ActionTokenizer` clipped it to the top bin.
- **Expected output:** with the edited mask, the top bin un-normalizes to **52.68 %**, computed from the real stats. Expect a near-constant ~53 % gripper. The arm joints were trained normally.

## What Was Built

### Task 1: adapter, client and config (commits a5e8de2 RED, 09a3918 GREEN)
- **`openvla_server.py`:** same shape as `openvla_oft_server.py`.
  - **`OpenVLAModel`:**
    - Loads once: `_download_pinned`, then `_localize_remote_code`, then `_load_policy`.
    - Resolves the unnorm key in this order: profile key, then the sole key, then the `--unnorm-key` flag. It rejects stats that aren't 7-D.
    - `__call__` preprocesses the image (224 Lanczos, plus the optional center crop), builds the official prompt, calls `predict_action`, and requires a finite 7-vector.
  - **`create_app`:**
    - Endpoints: `/health` with metadata (checkpoint, revision, code revision, unnorm key, action space, q01/q99/mask, image key) and `/action`.
    - Responses: 429 when busy, 400 for a bad body (only the `image_key` frame is required), 500 on a model failure.
  - **`main()`:**
    - Picks the profile revision for a known checkpoint. An unknown checkpoint needs an explicit `--revision`.
    - Runs the dependency-pin check.
    - Warms the model before `run()` on :3003.
- **`backends/openvla.py`:**
  - Registered with `@register("openvla")` on the HTTP lane.
  - Endpoint and http(s) rules are identical to openvla-oft.
  - The checkpoint must have a profile.
  - `make_convert_fn("absolute_joint_degrees_so101_7d")` gives `so101_joint_action` (slots 0-4 and 6 mapped to `JOINT_ORDER`), with no IK.
  - `make_convert_fn("delta_eef_robosuite_osc")` gives `openvla_oft_action_to_eef_delta` followed by `eef_to_joint.solve_ik(joint_state, delta, joint_limits_deg=...)`.
- **`backend_configs/openvla.json`:** the default checkpoint and revision, `endpoint_url: null`, the trained instruction, five checkpoint profiles, and notes.
- **`checkpoint_candidates.md`:** the new `backend:openvla` table and `backend:openvla-selection` sections. Every existing section is unchanged.

### Task 2: regression tests (commit db4c9d2)
- **Default SO-101 round trip:**
  - Path: real `create_app` → real `build()` convert_fn → real `HttpAdapterActionSource` → real `validate_action`.
  - The action is finite, in `JOINT_ORDER`, and non-held.
- **LIBERO alternate:** the same path through real ikpy `solve_ik`.
- **Failure:** a NaN model output → 500 → `http-error-holding-position`.

### Colab launch (commit 909f13a)
- **Cells:** `openvla-*` cells appended to `policy_server.ipynb`:
  - an isolated Python 3.10 uv venv with `torch==2.2.0`, `torchvision==0.17.0`, `transformers==4.40.1`, `tokenizers==0.19.1`, `timm==0.9.10`, `accelerate==0.30.1`, `sentencepiece==0.1.99`, `numpy<2`, `pillow`, `flask`, `opencv-python-headless`
  - a check on the install's exit status, then an import and CUDA check
  - launch on :3003 with a `/health` wait
  - tunnel
  - a local run using `--instruction`
  - teardown
- **Launch md:** `policy_server_launch.md` regenerated with `scripts/sync_policy_notebook.py`. The diff only adds lines.

## Verification
- `cd control && <main .venv>/bin/python -m pytest -q test_backend_openvla.py`: **53 passed**.
- Full suite: **565 passed** (fork base 512, plus 53).
- `run_vla_episode.py --help` lists `--backend {act,diffusion,groot-n17,openvla,openvla-oft,...}`.
- The new notebook code cells and the inner import-check snippet compile.

## Deviations from Plan

### Auto-fixed Issues

**1. [Plan-directed choice] The default checkpoint is an SO-101 joint-space fine-tune, so the default needs no IK**
- **Found during:** Task 1, HF Hub search.
- **What happened:**
  - The plan names `openvla/openvla-7b-finetuned-libero-object`. It also says to prefer a genuine SO-101 fine-tune if one exists, using pass-through and documenting it.
  - One exists. It is the default, and the four LIBERO checkpoints stay supported through IK.
- **Effect on the must-haves:** the must-have that the "7-D delta-EEF action space is explicitly converted" holds for those LIBERO checkpoints. The selection is explicit per checkpoint, never assumed. Tests cover both paths.
- **Commit:** 09a3918.

**2. [Rule 2 - Security, T-14-06] Pinned cross-repo remote code**
- **Issue:** the plan pins the checkpoint revision only. The fine-tunes ship no code, and `trust_remote_code` would fetch `openvla/openvla-7b` code at `main`. `AutoProcessor` drops `code_revision`, so passing it isn't enough.
- **Fix:** `_localize_remote_code` puts the reviewed code into the snapshot and refuses any foreign `auto_map`.
- **Commit:** 09a3918.

**3. [Rule 1 - Bug] The loading call matches the official path, with an explicit unnorm key**
- **Issue:** the plan's `AutoModelForVision2Seq.from_pretrained(checkpoint, trust_remote_code=True)` alone gives no action, because `predict_action` needs the processor, the prompt format and an `unnorm_key`.
- **Why it matters:** toasta18 embeds 26 keys, so `predict_action(unnorm_key=None)` would assert-fail.
- **Fix:** the server follows deploy.py and openvla_utils, with per-checkpoint unnorm keys.
- **Commit:** 09a3918.

**4. [Rule 1 - Bug] The `/action` response is single-step and model-native only**
- **Issue:** the plan says the structure should be "identical to the OFT server". OpenVLA takes no proprio and no wrist image.
- **Fix:** the server needs only the configured `image_key` frame. It accepts and validates `joint_state` and otherwise ignores it.
- **Commit:** 09a3918.

**5. [Rule 2 - Correctness] Per-checkpoint conversion guard**
- **Issue:** the client can't see which checkpoint the server loaded before the first action.
- **Fix:**
  - `build()` refuses a checkpoint with no reviewed `action_space` profile.
  - A test pins the client and server profile tables equal.
  - `model_version` logs the checkpoint actually served.
- **Commit:** 09a3918.

**6. [Rule 2 - Correctness] Dependency-pin guard**
- **Issue:** `modeling_prismatic.py` warns and continues on a mismatched transformers or tokenizers. The README cites inference regressions on those versions.
- **Fix:** `main()` refuses to start.
- **Commit:** 09a3918.

**7. [Rule 3 - Blocking] Colab cells, regenerated launch md and deferred-items entries**
- These files are outside the plan's list but needed for the human check (14-03/14-08 precedent).
- **Commit:** 909f13a.

**8. Test corrections during GREEN**
- The center-crop test's band (3 %) was thicker than the 2.6 % crop margin. I fixed the fixture geometry, not the code.
- **Commit:** 09a3918.

### Notes
- Task 2's tests are regression guards for Task 1 code, so they have no RED stage (same as 14-05 through 14-08).
- There are no new local dependencies. Every OpenVLA dependency is Colab-only, in its own venv.

## TDD Gate Compliance
- **RED:** `test(14-09)` a5e8de2, which failed with an ImportError.
- **GREEN:** `feat(14-09)` 09a3918, with 50 passing.
- **Regression tests:** `test(14-09)` db4c9d2, with 53 passing.

## Unverified (stated explicitly)
- **Real inference has not run here.** There is no GPU, no local transformers, and the 15 GB weights were never downloaded. I checked the loading and unnormalization logic against the real pinned config and code files.
- **toasta18's camera.** It is assumed that its RLDS `image` is the dataset's `overhead` camera, by LIBERO-style naming; I did not decode the TFRecords. `--image-key` overrides this.
- **Gripper behavior.** The near-constant ~53 % gripper is an inference from the stats and commit history (see above).
- **Calibration offsets.** The calibration offset between toasta18's robot and ours is unknown, as with every community SO-101 checkpoint.
- **Center-crop pixels.** The PIL center crop is a close approximation of TF's `crop_and_resize`, not bit-identical.
- **Colab environment.** Whether uv-installed `torch==2.2.0` (cu121 wheels) runs on the current Colab driver is unknown until the cell runs. Inference latency against the client's 5 s timeout is also unmeasured.
- **IK path.** The LIBERO alternates inherit 14-03's unverified LIBERO→SO-101 frame and sign mapping.

## Package Legitimacy Evidence (for the end-of-phase human check)
- **`github.com/openvla/openvla`** (in the 14-RESEARCH audit):
  - Repo: created 2024-06-13, MIT, about 7102 stars and 854 forks, not archived, `main` @ `c8f03f4` (2025-03-23).
  - It is **not cloned or installed**. I read it only to verify the loading path.
- **HF `openvla/openvla-7b` @ `47a0ec7f`:** the official org, matching the GitHub org. MIT, about 289k downloads. Its remote code is the only code `trust_remote_code` executes.
- **HF `openvla/openvla-7b-finetuned-libero-*`:** official, MIT, linked from the openvla README.
- **New and not in the 14-RESEARCH audit:**
  - **`toasta18/openvla-so101-cylinder`:** community, no license, empty card, 4 downloads. It is weights and JSON only, with no code.
  - Its datasets, `toasta18/so101_task_yellow_cylinder` and `toasta18/so101_pick_cylinder_rlds`, also have no license.
- **PyPI pins (Colab venv only)**, all official, as checked on pypi.org:
  - `torch 2.2.0` (PyTorch Team, 2024-01-30) and `torchvision 0.17.0`
  - `transformers 4.40.1` (huggingface/transformers, 2024-04-23), `tokenizers 0.19.1`, `timm 0.9.10` (huggingface/pytorch-image-models, Ross Wightman), `accelerate 0.30.1` (huggingface/accelerate)
  - `sentencepiece 0.1.99` (google/sentencepiece)
  - `uv` (Astral; already used by 14-03's cells)

## Known Stubs
None. `endpoint_url: null` is intentional: it is the per-session ngrok URL, passed with `--server-address`.

## Threat Flags

| Flag | File | Description |
|------|------|-------------|
| threat_flag: network-endpoint | control/vla_bridge/openvla_server.py | A new Colab HTTP adapter (`/health`, `/action`) behind an ngrok tunnel, the same exposure class as the OFT and Wall-X adapters. It accepts only images, an instruction and joint numbers. Requests are size-capped (20 MB), images are capped at ≤1024 px, and concurrency gets 429. |
| threat_flag: remote-code | control/vla_bridge/openvla_server.py | `trust_remote_code=True` executes only `openvla/openvla-7b@47a0ec7f`'s three files, copied in by `_localize_remote_code`. An `auto_map` pointing at any other repo is refused (T-14-06). |
| threat_flag: new-dependency | control/vla_bridge/policy_server.ipynb | A Colab-only isolated venv with OpenVLA's pinned torch, transformers, tokenizers and timm stack. Nothing is installed locally. |

T-14-05 holds: the IK path clamps to URDF ∩ calibrated limits, and both paths go through `run_episode()`'s unmodified `safety_validator.validate_action()`.

## Human Checks Deferred
1. **OpenVLA live hardware (Task 2):**
   - **test:**
     - On an A100 or L4 Colab, run `openvla-install`. Confirm the versions print with no `RuntimeError`.
     - Upload `openvla_server.py`, then run `openvla-launch` and `openvla-tunnel`.
     - Run `python control/run_vla_episode.py --backend openvla --server-address https://<tunnel> --instruction "Pick up the black cylinder and place it on the plate." --out control/outputs/<dir> --max-steps 20` against the real SO-ARM101.
   - **expected:**
     - `/health` ready, with `checkpoint` set to `toasta18/openvla-so101-cylinder`, `revision` `f00833c…`, `unnorm_key` `so101_pick_cylinder` and `action_space` `absolute_joint_degrees_so101_7d`.
     - At least one step logs a validated, non-held action, with `model_version` `…@f00833ce5851` and no `holding-position` flag, and the robot visibly moves.
     - Also note: inference time per request (it must stay under 5 s), peak VRAM, and whether the gripper target stays near ~53 %.
   - **why_human:** this needs real hardware and a live Colab GPU, and the cells haven't been run.
2. **Optional LIBERO alternate:**
   - **test:** relaunch the server with `--checkpoint openvla/openvla-7b-finetuned-libero-object` and run the client with the same `--checkpoint`.
   - **expected:** IK-converted, validated actions.
   - **why_human:** this also exercises 14-03's open IK frame and sign check (14-03 human check 3).
3. **Package legitimacy and licensing:**
   - **test:**
     - Confirm the evidence above.
     - Decide whether the **unlicensed** community checkpoint `toasta18/openvla-so101-cylinder` and its datasets are acceptable for this research use.
     - If they aren't, set `checkpoint` in `openvla.json` to an official MIT LIBERO alternate.
   - **expected:** first-party sources, with no name or org mismatch, and an explicit licensing decision.
   - **why_human:** this is a legitimacy and licensing judgment.
4. **Carried from 14-01:** on the HTTP lane, the wrist camera tolerates two independent `cv2.VideoCapture` opens.

## Self-Check: PASSED
- FOUND: control/vla_bridge/openvla_server.py, control/vla_bridge/backends/openvla.py, control/vla_bridge/backend_configs/openvla.json, control/test_backend_openvla.py, control/vla_bridge/checkpoint_candidates.md, control/vla_bridge/policy_server.ipynb, control/vla_bridge/policy_server_launch.md
- FOUND commits: a5e8de2, 09a3918, db4c9d2, 909f13a
- No deletions and no STATE.md/ROADMAP.md changes since base 90936ae.
