"""Mock-based pytest for eval_loop.py — no GPU/network dependency.

Proves the control-flow contracts (D-03 open-loop chunk replay, D-12
per-step early-stop polling, D-11 per-episode video writing, D-14
PASS/FAIL + summary reporting) without needing a real VLA model or a
real LIBERO/MuJoCo env. GPU-backed behavior is verified on Colab in
Plan 02's notebooks (03-RESEARCH.md Environment Availability).
"""

import numpy as np

# NOTE on import path: pytest's default rootdir-walking collection mode
# treats this file's nearest ancestor without an __init__.py
# (LIBERO/libero/, since LIBERO/libero/libero/__init__.py exists) as the
# insertion point, making this package resolvable as `libero.vla.*` when
# invoked via `pytest LIBERO/libero/libero/vla/test_eval_loop.py` from the
# repo root (no extra sys.path setup needed). This differs from the
# `libero.libero.vla` form used when `LIBERO` itself is placed on
# sys.path (e.g. Colab notebooks, `sys.path.insert(0, "LIBERO")` per this
# project's established convention) — both resolve to the same package on
# disk, just via a different sys.path anchor.
from libero.vla.eval_loop import print_episode_result, run_episode, run_suite


class MockEnv:
    """Scripted env: returns done=True on the 3rd env.step() call (mid-chunk)."""

    def __init__(self, done_on_step=3):
        self.done_on_step = done_on_step
        self.step_count = 0
        self.last_seed = None
        self.seed_calls = []
        self._obs = {
            "robot0_eye_in_hand_image": np.zeros((4, 4, 3), dtype=np.uint8),
            "agentview_image": np.zeros((4, 4, 3), dtype=np.uint8),
        }

    def reset(self):
        self.step_count = 0
        return self._obs

    def step(self, action):
        self.step_count += 1
        done = self.step_count >= self.done_on_step
        return self._obs, 0.0, done, {}

    def seed(self, seed):
        self.last_seed = seed
        self.seed_calls.append(seed)


class MockBackend:
    """Returns an 8-step chunk of arbitrary 7-D actions; counts predict() calls."""

    def __init__(self):
        self.predict_calls = 0

    def predict(self, images, language):
        self.predict_calls += 1
        return np.zeros((8, 7))


class RecordingBackend:
    """Records the images dict it receives on each predict() call (Phase 5,
    SPAT-02 dict-plumbing regression coverage)."""

    def __init__(self):
        self.seen_images = None

    def predict(self, images, language):
        self.seen_images = images
        return np.zeros((8, 7))


def test_open_loop_chunk_replay_with_mid_chunk_early_stop():
    """Test 1: done on 3rd of 8 chunk steps -> exactly 3 env.step() calls,
    predict() called exactly once (D-03 + D-12).
    """
    env = MockEnv(done_on_step=3)
    backend = MockBackend()

    result = run_episode(env, backend, "pick up the bowl", "/tmp/unused_video")

    assert env.step_count == 3, f"expected 3 env.step() calls, got {env.step_count}"
    assert backend.predict_calls == 1, f"expected 1 predict() call, got {backend.predict_calls}"
    assert result["success"] is True
    assert result["steps"] == 3


def test_video_writer_append_obs_called_per_step(monkeypatch):
    """Test 2: VideoWriter.append_obs called once per env.step() call."""
    append_calls = []

    class MockVideoWriter:
        def __init__(self, video_path, save_video=False, fps=30, single_video=True):
            self.video_path = video_path

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

        def append_obs(self, obs, done, idx=0, camera_name="agentview_image"):
            append_calls.append((done, camera_name))

    import libero.vla.eval_loop as eval_loop_module

    monkeypatch.setattr(eval_loop_module, "VideoWriter", MockVideoWriter)

    env = MockEnv(done_on_step=3)
    backend = MockBackend()

    run_episode(env, backend, "pick up the bowl", "/tmp/unused_video")

    assert len(append_calls) == env.step_count == 3
    # last call should have been the done=True step
    assert append_calls[-1][0] is True


def test_run_episode_return_dict_and_print(capsys):
    """Test 3: run_episode returns {"success", "steps", "video_path"} and
    print_episode_result prints the exact PASS/FAIL arrow-notation line.
    """
    env = MockEnv(done_on_step=3)
    backend = MockBackend()

    result = run_episode(env, backend, "pick up the bowl", "/tmp/video_dir/task_ep0")

    assert set(result.keys()) == {"success", "steps", "video_path"}
    assert result["video_path"] == "/tmp/video_dir/task_ep0"

    # Discard VideoWriter's own "Saved videos to ..." print from run_episode
    # before isolating print_episode_result's own output below.
    capsys.readouterr()

    print_episode_result("pick_up_bowl", 0, result)
    captured = capsys.readouterr()
    assert (
        captured.out.strip()
        == "Episode 0 (pick_up_bowl): PASS (steps=3) -> /tmp/video_dir/task_ep0"
    )

    # FAIL case
    fail_result = {"success": False, "steps": 600, "video_path": "/tmp/video_dir/task_ep1"}
    print_episode_result("pick_up_bowl", 1, fail_result)
    captured = capsys.readouterr()
    assert (
        captured.out.strip()
        == "Episode 1 (pick_up_bowl): FAIL (steps=600) -> /tmp/video_dir/task_ep1"
    )


def test_run_suite_multi_task_multi_episode_summary(capsys):
    """Test 4: 2 tasks x 2 episodes -> 4-row results list, printed table
    containing both task names and an aggregate row (D-14).
    """

    def env_factory(task):
        # task "task_a" always succeeds fast; task "task_b" never succeeds
        # within the (small, test-scale) max_steps budget below.
        if task == "task_a.bddl":
            return MockEnv(done_on_step=2)
        return MockEnv(done_on_step=10_000)

    backend = MockBackend()
    tasks = ["task_a.bddl", "task_b.bddl"]
    language_map = {"task_a.bddl": "pick up the bowl", "task_b.bddl": "open the drawer"}

    results = run_suite(
        env_factory,
        backend,
        tasks,
        language_map,
        episodes_per_task=2,
        video_dir="/tmp/video_dir",
        max_steps=5,
    )

    assert len(results) == 4
    task_names = {r["task"] for r in results}
    assert task_names == {"task_a", "task_b"}

    captured = capsys.readouterr()
    assert "task_a" in captured.out
    assert "task_b" in captured.out
    assert "Aggregate" in captured.out
    # a numeric success-rate percentage per task is printed
    assert "%" in captured.out


def test_images_dict_includes_both_camera_views_spatial():
    """Test 5 (Phase 5, D-01/SPAT-02): run_episode's images dict passed to
    backend.predict() contains both "eye_in_hand" and "agentview" keys,
    sourced from the correct obs entries (not swapped)."""
    env = MockEnv(done_on_step=3)
    backend = RecordingBackend()

    run_episode(env, backend, "pick up the bowl", "/tmp/unused_video")

    assert set(backend.seen_images.keys()) >= {"eye_in_hand", "agentview"}
    np.testing.assert_array_equal(
        backend.seen_images["eye_in_hand"], env._obs["robot0_eye_in_hand_image"]
    )
    np.testing.assert_array_equal(
        backend.seen_images["agentview"], env._obs["agentview_image"]
    )


def test_run_episode_calls_env_seed_before_reset():
    """Test 6 (TUNE-03, D-08, Pitfall 7): passing seed= to run_episode calls
    env.seed(seed) before env.reset() (verified via MockEnv's last_seed
    recorder)."""
    env = MockEnv(done_on_step=3)
    backend = MockBackend()

    run_episode(env, backend, "pick up the bowl", "/tmp/unused_video", seed=42)

    assert env.last_seed == 42


def test_run_episode_without_seed_never_calls_env_seed():
    """Test 7 (TUNE-03, D-08): omitting seed leaves behavior byte-identical
    to before this parameter existed — env.seed() is never called."""
    env = MockEnv(done_on_step=3)
    backend = MockBackend()

    run_episode(env, backend, "pick up the bowl", "/tmp/unused_video")

    assert env.last_seed is None
    assert env.seed_calls == []


def test_run_suite_applies_seeds_identically_across_two_runs():
    """Test 8 (TUNE-03, D-08, Pitfall 7): running run_suite twice with the
    same env_factory/episode_seeds produces the identical env.seed() call
    sequence both times -- the literal regression Pitfall 7 warns against
    (seeding once before the per-episode loop, instead of inside it, would
    make only episode 0 seeded correctly and the two runs would diverge)."""
    envs = []

    def env_factory(task):
        env = MockEnv(done_on_step=10_000)
        envs.append(env)
        return env

    backend = MockBackend()
    episode_seeds = [10, 20, 30]

    run_suite(
        env_factory,
        backend,
        ["task_a.bddl"],
        {"task_a.bddl": "do the task"},
        episodes_per_task=3,
        video_dir="/tmp/video_dir",
        max_steps=5,
        episode_seeds=episode_seeds,
    )
    first_run_env = envs[-1]
    assert first_run_env.seed_calls == [10, 20, 30]
    assert first_run_env.last_seed == 30

    run_suite(
        env_factory,
        backend,
        ["task_a.bddl"],
        {"task_a.bddl": "do the task"},
        episodes_per_task=3,
        video_dir="/tmp/video_dir",
        max_steps=5,
        episode_seeds=episode_seeds,
    )
    second_run_env = envs[-1]
    assert second_run_env.seed_calls == [10, 20, 30]
    assert second_run_env.last_seed == 30
