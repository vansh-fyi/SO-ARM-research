# Phase 4: Dataset Collection - Pattern Map

**Mapped:** 2026-08-02
**Files analyzed:** 10
**Analogs found:** 10 / 10

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `LIBERO/libero/libero/datasets/__init__.py` | config (package init) | event-driven (graceful-degradation import) | `LIBERO/libero/libero/vla/__init__.py` | exact |
| `LIBERO/libero/libero/datasets/collector.py` | service (scripted waypoint controller) | event-driven (control-loop, D-01) | `LIBERO/scripts/collect_demonstration.py` (`collect_human_trajectory`) + Phase 2's `body_xpos` ground-truth pattern | role-match |
| `LIBERO/libero/libero/datasets/teleop.py` | service (human input collector) | event-driven (keyboard-driven control loop, D-02) | `LIBERO/scripts/collect_demonstration.py` (`collect_human_trajectory`, `__main__` device wiring) | exact |
| `LIBERO/libero/libero/datasets/raw_recorder.py` | utility (shared `DataCollectionWrapper` setup) | event-driven | `LIBERO/scripts/collect_demonstration.py` (`DataCollectionWrapper` construction, lines 299-317) | role-match |
| `LIBERO/libero/libero/datasets/hdf5_writer.py` | service (state -> obs -> HDF5 assembly) | batch / transform (DATA-01) | `LIBERO/scripts/collect_demonstration.py` (`gather_demonstrations_as_hdf5`) + `LIBERO/libero/libero/envs/env_wrapper.py` (`ControlEnv.set_init_state`/`regenerate_obs_from_state`) | role-match |
| `LIBERO/libero/libero/datasets/replay.py` | service (determinism verification) | batch / transform (DATA-02) | `LIBERO/libero/libero/envs/env_wrapper.py` (`ControlEnv.set_init_state`) — no direct analog script exists in-repo; pattern adapted from `robosuite/scripts/playback_demonstrations_from_hdf5.py` (installed package, state-based branch) | role-match |
| `LIBERO/libero/libero/datasets/normalization.py` | utility (stats computation, DATA-03) | transform / batch | `LIBERO/libero/libero/vla/oft_backend.py` (`norm_stats` overlay/consumption pattern, lines 99-117) | role-match |
| `LIBERO/libero/libero/datasets/test_collector.py` | test | — | `LIBERO/libero/libero/vla/test_eval_loop.py` (mock-based, no-GPU pytest style) | exact |
| `LIBERO/libero/libero/datasets/test_hdf5_writer.py` | test | — | `LIBERO/libero/libero/vla/test_eval_loop.py` | exact |
| `LIBERO/libero/libero/datasets/test_replay.py` | test | — | `LIBERO/libero/libero/vla/test_eval_loop.py` | exact |
| `LIBERO/libero/libero/datasets/test_normalization.py` | test | — | `LIBERO/libero/libero/vla/test_eval_loop.py` | exact |

## Pattern Assignments

### `LIBERO/libero/libero/datasets/__init__.py` (config, event-driven)

**Analog:** `LIBERO/libero/libero/vla/__init__.py` (full file, 32 lines)

**Graceful-degradation import pattern** (lines 1-15):
```python
from .interface import VLABackend

try:
    # torch is a Colab-only GPU dependency for this project (03-RESEARCH.md
    # Environment Availability) — not installed in the local dev environment.
    # Degrade gracefully so `eval_loop`'s local, no-GPU pytest suite can still
    # import this package; on Colab (torch installed) this import succeeds
    # and OFTBackend is exported normally.
    from .oft_backend import OFTBackend
except Exception:
    OFTBackend = None
```

**Apply to `datasets/__init__.py`:** the analogous risk is `robosuite`/`mujoco`-dependent submodules (`collector.py`, `teleop.py`, `hdf5_writer.py`, `replay.py`) failing to import in a context without a display/MuJoCo GL backend, while `normalization.py` (pure numpy) should always import cleanly. Wrap the sim-dependent submodule imports in the same `try/except Exception` pattern; export `normalization` unconditionally.

---

### `LIBERO/libero/libero/datasets/collector.py` (service, event-driven — DATA-01, D-01)

**Analog 1:** `LIBERO/scripts/collect_demonstration.py` — `collect_human_trajectory()` (lines 21-101) for the control-loop/env-step/success-hold-count structure; adapt by replacing `input2action(device=...)` with waypoint-phase-computed actions.

**Control loop + success-hold pattern** (lines 47-95):
```python
task_completion_hold_count = -1  # counter to collect 10 timesteps after reaching goal
device.start_control()
while True:
    action, grasp = input2action(device=device, robot=active_robot, active_arm=arm,
                                  env_configuration=env_configuration)
    if action is None:
        saving = False
        break
    env.step(action)
    env.render()
    if task_completion_hold_count == 0:
        break
    if env._check_success():
        if task_completion_hold_count > 0:
            task_completion_hold_count -= 1
        else:
            task_completion_hold_count = 10
    else:
        task_completion_hold_count = -1
```
For the scripted collector, replace the `input2action(...)` call with a waypoint-phase state machine (approach/descend/grasp/lift/transport/release) driven off ground-truth positions, keeping the same success-hold-count exit condition.

**Ground-truth object position pattern** (cited in RESEARCH.md, from `LIBERO/libero/libero/envs/bddl_base_domain.py:512`):
```python
bowl_pos = env.sim.data.body_xpos[env.obj_body_id["akita_black_bowl_1"]]
```

**Frozen task list** (copy verbatim — CONTEXT.md's list is stale, do not use it):
```python
# Source: explorations/soarm_sanity.py lines 65-69 (this repo, already correct)
TASKS = [
    "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate.bddl",
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.bddl",
    "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate.bddl",
]
```

---

### `LIBERO/libero/libero/datasets/teleop.py` (service, event-driven — DATA-04, D-02)

**Analog:** `LIBERO/scripts/collect_demonstration.py` — device wiring in `__main__` (lines 308-317) and `collect_human_trajectory()` (verbatim reuse per D-02, keyboard only, no SpaceMouse branch).

**Keyboard device wiring** (lines 309-317):
```python
from robosuite.devices import Keyboard

device = Keyboard(
    pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity
)
env.viewer.add_keypress_callback("any", device.on_press)
env.viewer.add_keyup_callback("any", device.on_release)
env.viewer.add_keyrepeat_callback("any", device.on_press)
```
Drop the `elif args.device == "spacemouse"` branch entirely (D-02: keyboard-only, no hardware dependency). Reuse `collect_human_trajectory()`'s loop structure unchanged — this is the human-input counterpart to `collector.py`'s scripted loop, sharing the same `DataCollectionWrapper`/success-hold-count/env.step() shape.

---

### `LIBERO/libero/libero/datasets/raw_recorder.py` (utility, event-driven)

**Analog:** `LIBERO/scripts/collect_demonstration.py` — env construction + `DataCollectionWrapper` setup (lines 274-306).

**Env + wrapper construction pattern** (lines 274-306):
```python
env = TASK_MAPPING[problem_name](
    bddl_file_name=args.bddl_file,
    **config,
    has_renderer=True,
    has_offscreen_renderer=False,
    render_camera=args.camera,
    ignore_done=True,
    use_camera_obs=False,
    reward_shaping=True,
    control_freq=20,
)
env = VisualizationWrapper(env)
tmp_directory = "demonstration_data/tmp/{}_ln_{}/{}".format(
    problem_name, language_instruction.replace(" ", "_").strip('""'),
    str(time.time()).replace(".", "_"),
)
env = DataCollectionWrapper(env, tmp_directory)
```
Factor this into a shared `build_recording_env(bddl_file_name, tmp_dir) -> DataCollectionWrapper` helper called identically by both `collector.py` and `teleop.py`, per the recommended project structure in RESEARCH.md. Must instantiate SOARM's `Soarm101` robot class (Phase 2) via the same `TASK_MAPPING[...]` / `robots=["Soarm101"]` construction Phase 2/3 already use — do not hardcode `"Panda"`.

---

### `LIBERO/libero/libero/datasets/hdf5_writer.py` (service, batch/transform — DATA-01)

**Analog 1:** `LIBERO/scripts/collect_demonstration.py` — `gather_demonstrations_as_hdf5()` (lines 104-193).

**npz-to-HDF5 gather pattern with off-by-one fix** (lines 148-179):
```python
state_paths = os.path.join(directory, ep_directory, "state_*.npz")
states, actions = [], []
for state_file in sorted(glob(state_paths)):
    dic = np.load(state_file, allow_pickle=True)
    states.extend(dic["states"])
    for ai in dic["action_infos"]:
        actions.append(ai["actions"])

# Delete the last state: DataCollectionWrapper records states AFTER
# playing the action, producing one extra trailing state.
del states[-1]
assert len(states) == len(actions)

ep_data_grp = grp.create_group("demo_{}".format(num_eps))
xml_path = os.path.join(directory, ep_directory, "model.xml")
with open(xml_path, "r") as f:
    ep_data_grp.attrs["model_file"] = f.read()
ep_data_grp.create_dataset("states", data=np.array(states))
ep_data_grp.create_dataset("actions", data=np.array(actions))
```

**Analog 2:** `LIBERO/libero/libero/envs/env_wrapper.py` — `ControlEnv.set_init_state`/`regenerate_obs_from_state` (lines 136-145), the exact obs-regeneration primitive to call per recorded state before writing `obs/` datasets:
```python
def set_init_state(self, init_state):
    return self.regenerate_obs_from_state(init_state)

def regenerate_obs_from_state(self, mujoco_state):
    self.set_state(mujoco_state)
    self.env.sim.forward()
    self.check_success()
    self._post_process()
    self._update_observables(force=True)
    return self.env._get_observations()
```

**Obs key rename map** (must apply at write time — from `LIBERO/libero/configs/data/default.yaml` lines 30-34):
```yaml
obs_key_mapping:
  agentview_rgb: agentview_image
  eye_in_hand_rgb: robot0_eye_in_hand_image
  gripper_states: robot0_gripper_qpos
  joint_states: robot0_joint_pos
```
i.e. `hdf5_writer.py` must rename `obs["agentview_image"] -> obs/agentview_rgb`, etc. at write time — writing raw robosuite key names causes a silent `KeyError` at Phase 6 load time, not at this phase's write/test time (see RESEARCH.md Pitfall 4).

**Error handling:** follow `env_wrapper.py`'s `assert os.path.exists(bddl_file_name)` convention (line 43-45) for any developer-supplied file path; fail loudly on disk-write errors (project convention per `01-DEBUG-HISTORY.md`, cited in RESEARCH.md Security Domain V7).

---

### `LIBERO/libero/libero/datasets/replay.py` (service, batch/transform — DATA-02)

**Analog:** `LIBERO/libero/libero/envs/env_wrapper.py` — `ControlEnv.set_init_state()` (lines 136-145, same excerpt as above) is the exact primitive; no in-repo replay script exists, so also reference `robosuite/scripts/playback_demonstrations_from_hdf5.py`'s installed-package state-based branch (cited in RESEARCH.md Code Examples, not locally vendored):
```python
for state in states:
    env.sim.set_state_from_flattened(state)
    env.sim.forward()
    # prefer set_init_state() over this raw two-liner: it additionally
    # calls check_success(), _post_process(), _update_observables(force=True)
```

**Determinism assertion pattern** (from RESEARCH.md Code Examples, synthesized against the local `set_init_state` primitive):
```python
def verify_episode_determinism(env, states, recorded_obs, atol=0.0):
    for i, state in enumerate(states):
        obs = env.set_init_state(state)
        for key, recorded in recorded_obs[i].items():
            assert np.allclose(obs[key], recorded, atol=atol), (
                f"step {i} key {key} diverged on replay"
            )
```

**Anti-pattern (do not copy):** never use an `env.step(action)` re-simulation loop as the DATA-02 verification method — only `sim.set_state_from_flattened(state)` / `set_init_state(state)`. Action-replay is explicitly non-deterministic (LIBERO issue #16, cited in RESEARCH.md).

**Two-tier verification split (D-06):** full-dataset states-only check (cheap, no `_get_observations()` call) on 100% of demos; sampled (~10%, min 5, >=1/task) full observation-regeneration check via `set_init_state()`.

---

### `LIBERO/libero/libero/datasets/normalization.py` (utility, transform — DATA-03)

**Analog:** `LIBERO/libero/libero/vla/oft_backend.py` — `norm_stats` overlay/consumption pattern (lines 99-117):
```python
# Checkpoint config's norm_stats holds only the OXE PRETRAINING stats;
# the fine-tuned norm_stats live in a separate dataset_statistics.json
# in the HF repo — must overlay it.
stats_path = hf_hub_download(checkpoint, "dataset_statistics.json")
with open(stats_path) as f:
    self.model.norm_stats = json.load(f)
print(f"norm_stats overlaid from dataset_statistics.json: {list(self.model.norm_stats.keys())}")

if self.unnorm_key not in self.model.norm_stats:
    if f"{self.unnorm_key}_no_noops" in self.model.norm_stats:
        ...
```
`normalization.py` must produce a `dataset_statistics.json`-shaped dict (q01/q99/mean/std/min/max per action dim, keyed by dataset name) so Phase 6 can overlay it via this exact same `json.load()` + dict-key pattern `oft_backend.py` already implements.

**Target schema** (from RESEARCH.md Code Examples):
```python
def compute_norm_stats(actions: np.ndarray, proprio: np.ndarray) -> dict:
    def stats(x):
        return {
            "mean": x.mean(0).tolist(), "std": x.std(0).tolist(),
            "max": x.max(0).tolist(), "min": x.min(0).tolist(),
            "q01": np.quantile(x, 0.01, axis=0).tolist(),
            "q99": np.quantile(x, 0.99, axis=0).tolist(),
        }
    return {
        "soarm_spatial": {
            "action": stats(actions),
            "proprio": stats(proprio),
            "num_transitions": int(actions.shape[0]),
            "num_trajectories": None,
        }
    }
```

---

### `LIBERO/libero/libero/datasets/test_collector.py`, `test_hdf5_writer.py`, `test_replay.py`, `test_normalization.py` (test)

**Analog:** `LIBERO/libero/libero/vla/test_eval_loop.py` (full file structure) — mock-based, no-GPU/no-sim-dependency pytest style with descriptive module docstring explaining what's proven without needing a real env.

**Module docstring + import-path convention** (lines 1-19):
```python
"""Mock-based pytest for eval_loop.py — no GPU/network dependency.
...
"""
import numpy as np

# NOTE on import path: pytest's default rootdir-walking collection mode
# treats this file's nearest ancestor without an __init__.py
# (LIBERO/libero/, since LIBERO/libero/libero/__init__.py exists) as the
# insertion point, making this package resolvable as `libero.vla.*` when
# invoked via `pytest LIBERO/libero/libero/vla/test_eval_loop.py` from the
# repo root (no extra sys.path setup needed).
from libero.vla.eval_loop import print_episode_result, run_episode, run_suite
```
Apply identically for `datasets/`: `from libero.datasets.normalization import compute_norm_stats`, etc. `test_normalization.py` can be fully mock/synthetic-array based (no sim dependency, matches RESEARCH.md's Validation Architecture "unit (synthetic array input)" classification). `test_hdf5_writer.py` and `test_replay.py` are integration tests requiring a real local sim (per RESEARCH.md Phase Requirements -> Test Map) — use a small 1-2 episode real `OffScreenRenderEnv` fixture rather than a mock, since they must exercise the real `set_init_state()`/HDF5 write path end-to-end.

**Mock class pattern** (lines 21-45, for `test_collector.py`'s scriptable env/backend mocks where a full sim isn't needed):
```python
class MockEnv:
    def __init__(self, done_on_step=3):
        self.done_on_step = done_on_step
        self.step_count = 0
        self._obs = {"robot0_eye_in_hand_image": np.zeros((4, 4, 3), dtype=np.uint8)}

    def reset(self):
        self.step_count = 0
        return self._obs

    def step(self, action):
        self.step_count += 1
        done = self.step_count >= self.done_on_step
        return self._obs, 0.0, done, {}
```

## Shared Patterns

### Graceful-degradation package import
**Source:** `LIBERO/libero/libero/vla/__init__.py`
**Apply to:** `datasets/__init__.py` — wrap sim-dependent submodule imports (`collector`, `teleop`, `hdf5_writer`, `replay`) in `try/except Exception: X = None`, matching the project's established resilience pattern for optional/environment-sensitive dependencies.

### State-based replay primitive (do not reimplement)
**Source:** `LIBERO/libero/libero/envs/env_wrapper.py:136-145` (`ControlEnv.set_init_state`/`regenerate_obs_from_state`)
**Apply to:** `hdf5_writer.py` (obs regeneration during write) and `replay.py` (DATA-02 verification) — both must call this same method rather than hand-rolling `sim.set_state_from_flattened()` + `sim.forward()`, to avoid diverging from LIBERO's own eval/replay ordering (`check_success -> _post_process -> _update_observables(force=True) -> _get_observations`).

### Obs key renaming (robosuite native -> LIBERO schema)
**Source:** `LIBERO/libero/configs/data/default.yaml` lines 30-34 (`obs_key_mapping`)
**Apply to:** `hdf5_writer.py` exclusively — the single point where raw `_get_observations()` output is written into the HDF5 `obs/` group; must rename before writing.

### npz off-by-one state/action alignment fix
**Source:** `LIBERO/scripts/collect_demonstration.py:163-166` (`del states[-1]`; `assert len(states) == len(actions)`)
**Apply to:** `hdf5_writer.py` — required whenever consuming `DataCollectionWrapper`'s raw npz output (both `collector.py`'s and `teleop.py`'s recordings feed through this same fix).

### Success-gating before write
**Source:** `LIBERO/scripts/collect_demonstration.py` `__main__` loop (`if saving: gather_demonstrations_as_hdf5(...); i += 1`)
**Apply to:** `collector.py` and `teleop.py` both — only successful episodes (via `DataCollectionWrapper`/`env._check_success()`-gated `saving` flag) should ever reach `hdf5_writer.py`; loop "attempt until N successes per task," not "attempt N times."

### Fail-loudly error handling
**Source:** `LIBERO/libero/libero/envs/env_wrapper.py:43-45` (`assert os.path.exists(bddl_file_name), f"[error] {bddl_file_name} does not exist!"`); project convention from `01-DEBUG-HISTORY.md`
**Apply to:** All new files — BDDL/HDF5 path validation via `assert os.path.exists(...)`, no silent `try/except` swallowing of write failures.

## No Analog Found

None — every planned file has at least a role-match analog in the local codebase or installed `robosuite` package source (see Pattern Assignments above for the installed-but-not-vendored `playback_demonstrations_from_hdf5.py` case, used only as a cited reference pattern for `replay.py`, not a file to copy from directly).

## Metadata

**Analog search scope:** `LIBERO/libero/libero/vla/`, `LIBERO/libero/libero/envs/`, `LIBERO/scripts/`, `LIBERO/libero/lifelong/datasets.py`, `LIBERO/libero/configs/data/default.yaml`, `explorations/soarm_sanity.py`, installed `robosuite` 1.4.1 package source (cited via RESEARCH.md, not re-read locally this pass)
**Files scanned:** 8 (read in full or targeted sections this session) + RESEARCH.md's prior citations reused
**Pattern extraction date:** 2026-08-02
</content>
