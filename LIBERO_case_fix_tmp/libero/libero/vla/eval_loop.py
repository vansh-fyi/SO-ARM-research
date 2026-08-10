"""Eval-loop helper: per-step success polling, open-loop chunk replay,
per-episode video saving, and PASS/FAIL + summary reporting (VLA-02, VLA-03).

Mirrors the per-step done-accumulation-then-break pattern in
`LIBERO/libero/lifelong/metric.py::evaluate_one_task_success` (simplified to
a single-process loop per 03-RESEARCH.md Pitfall 3 — the vectorized
SubprocVectorEnv machinery is not needed here), and reuses
`LIBERO/libero/libero/utils/video_utils.py::VideoWriter` and
`env.check_success()` / `env.step()`'s `done` flag directly rather than
reimplementing either.
"""

import os

from ..utils.video_utils import VideoWriter

MAX_STEPS_DEFAULT = 600  # this project's own LIBERO/libero/configs/eval/default.yaml value (D-13)


def run_episode(
    env,
    backend,
    language: str,
    video_path: str,
    max_steps: int = MAX_STEPS_DEFAULT,
    camera_name: str = "robot0_eye_in_hand_image",
    agentview_camera_name: str = "agentview_image",
) -> dict:
    """Run one episode: reset, predict -> chunk-replay -> poll -> record video.

    Per D-03, the action chunk returned by `backend.predict()` is replayed
    open-loop (every row via `env.step()`) before calling `predict()` again.
    Per D-12, `done` is checked after every single `env.step()` call (not
    just after a full chunk) so a mid-chunk success stops immediately —
    the remaining chunk rows are not replayed after success.

    Per D-11, every step is written to a single per-episode video via
    `VideoWriter(..., single_video=True)`.

    Per D-15, a max_steps timeout is a hard binary FAIL — no partial-credit
    or near-miss diagnostic is captured.

    Args:
        env: an already-constructed LIBERO env (e.g. OffScreenRenderEnv).
            This function calls env.reset() itself, so callers pass a
            fresh, unreset (or freely-resettable) env.
        backend: any VLABackend-satisfying object (predict(images, language)).
        language: natural-language task instruction passed to backend.predict.
        video_path: directory path passed to VideoWriter for this episode's video.
        max_steps: step budget for a never-succeeding episode (D-13: this
            project's own LIBERO eval config default, 600).
        camera_name: obs dict key for the eye-in-hand camera frame (Phase 2
            confirmed key, also used as the interface's "eye_in_hand" image).
            Also drives VideoWriter's own camera (unchanged from Phase 3).
        agentview_camera_name: obs dict key for the overhead agentview
            camera frame (Phase 5, D-01/SPAT-02) — passed to backends as the
            interface's "agentview" image, alongside "eye_in_hand".

    Returns:
        {"success": bool, "steps": int, "video_path": str}
    """
    obs = env.reset()
    steps = 0
    done = False

    with VideoWriter(video_path, save_video=True, fps=30, single_video=True) as vw:
        while steps < max_steps and not done:
            images = {
                "eye_in_hand": obs[camera_name],
                "agentview": obs[agentview_camera_name],
            }
            action_chunk = backend.predict(images, language)

            for a in action_chunk:
                obs, reward, done, info = env.step(a)
                steps += 1
                vw.append_obs(obs, done, camera_name=camera_name)
                if done:  # D-12: stop the instant success is detected, mid-chunk
                    break
                if steps >= max_steps:
                    break

    return {"success": bool(done), "steps": steps, "video_path": video_path}


def print_episode_result(task_name: str, ep_idx: int, result: dict) -> None:
    """Print a PASS/FAIL line for one episode, arrow-notation convention (CLAUDE.md)."""
    verdict = "PASS" if result["success"] else "FAIL"
    print(
        f"Episode {ep_idx} ({task_name}): {verdict} "
        f"(steps={result['steps']}) -> {result['video_path']}"
    )


def run_suite(
    env_factory,
    backend,
    tasks: list,
    language_map: dict,
    episodes_per_task: int,
    video_dir: str,
    max_steps: int = MAX_STEPS_DEFAULT,
) -> list:
    """Run `episodes_per_task` episodes for each task, print PASS/FAIL per
    episode, and print an aggregated success-rate summary table (D-14).

    Args:
        env_factory: callable(task) -> env, e.g.
            `lambda bddl: OffScreenRenderEnv(bddl_file_name=bddl,
            robots=["Soarm101"], camera_heights=256, camera_widths=256,
            has_renderer=False, has_offscreen_renderer=True)` per Phase 2's
            confirmed no-custom-controller-kwarg pattern.
        backend: VLABackend-satisfying object shared across all tasks/episodes.
        tasks: list of task identifiers (e.g. BDDL file paths) passed to env_factory.
        language_map: dict mapping each task identifier to its language instruction.
        episodes_per_task: number of episodes to run per task.
        video_dir: directory under which per-episode video subdirectories are created.
        max_steps: forwarded to run_episode (D-13 default 600).

    Returns:
        list of dicts, one per episode:
        {"task": task, "episode": ep_idx, **run_episode()'s return dict}
    """
    all_results = []

    for task in tasks:
        env = env_factory(task)
        language = language_map[task]
        task_slug = os.path.basename(str(task)).replace(".bddl", "")

        for ep_idx in range(episodes_per_task):
            video_path = os.path.join(video_dir, f"{task_slug}_ep{ep_idx}")
            result = run_episode(
                env, backend, language, video_path, max_steps=max_steps
            )
            print_episode_result(task_slug, ep_idx, result)
            all_results.append({"task": task_slug, "episode": ep_idx, **result})

    _print_summary_table(all_results)
    return all_results


def _print_summary_table(all_results: list) -> None:
    """Print a markdown-style summary table: task | episodes | successes | success rate %,
    plus an aggregate row across all tasks/episodes (D-14).
    """
    by_task: dict = {}
    for r in all_results:
        stats = by_task.setdefault(r["task"], {"episodes": 0, "successes": 0})
        stats["episodes"] += 1
        stats["successes"] += int(r["success"])

    print("\n| Task | Episodes | Successes | Success Rate |")
    print("|------|----------|-----------|--------------|")
    for task, stats in by_task.items():
        rate = 100.0 * stats["successes"] / stats["episodes"] if stats["episodes"] else 0.0
        print(f"| {task} | {stats['episodes']} | {stats['successes']} | {rate:.1f}% |")

    total_episodes = sum(s["episodes"] for s in by_task.values())
    total_successes = sum(s["successes"] for s in by_task.values())
    aggregate_rate = 100.0 * total_successes / total_episodes if total_episodes else 0.0
    print(f"| **Aggregate** | {total_episodes} | {total_successes} | {aggregate_rate:.1f}% |")
