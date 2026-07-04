"""
LIBERO dataset explorer.

Data layout (relative to repo root):
  data/libero/meta/info.json        — dataset metadata (40 tasks, 1693 episodes, 256x256 images)
  data/libero/meta/tasks.parquet    — task descriptions
  data/libero/data/chunk-000/*.parquet — episode frames (images stored as PNG bytes)

Each parquet row:
  observation.images.image   — front camera  256x256 RGB (dict with 'bytes' key)
  observation.images.image2  — wrist camera  256x256 RGB (dict with 'bytes' key)
  observation.state          — 8-dim robot joint state
  action                     — 7-dim end-effector action
  timestamp, frame_index, episode_index, index, task_index

Usage:
  python libero/explore.py                         # grid of first 10 episodes
  python libero/explore.py --episodes 20           # grid of first N episodes
  python libero/explore.py --episode 5             # all frames of one episode
  python libero/explore.py --tasks                 # print all 40 task descriptions
"""

import argparse
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT  = Path(__file__).resolve().parent.parent
DATA  = ROOT / "data" / "libero"
OUT   = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def load_tasks():
    df = pd.read_parquet(DATA / "meta" / "tasks.parquet")
    return df.index.tolist()


def load_info():
    with open(DATA / "meta" / "info.json") as f:
        return json.load(f)


def iter_parquet_files():
    chunk_dir = DATA / "data" / "chunk-000"
    return sorted(chunk_dir.glob("*.parquet"))


def collect_first_frames(n_episodes: int = 10):
    """Yield the first frame dict for each of the first n_episodes distinct episodes."""
    task_names = load_tasks()
    seen_eps = set()
    collected = []

    for fpath in iter_parquet_files():
        if len(collected) >= n_episodes:
            break
        df = pd.read_parquet(fpath)
        for _, row in df.iterrows():
            ep = int(row["episode_index"])
            if ep not in seen_eps:
                seen_eps.add(ep)
                task_idx = int(row["task_index"])
                collected.append({
                    "episode_index": ep,
                    "task_index": task_idx,
                    "task_name": task_names[task_idx] if task_idx < len(task_names) else f"task_{task_idx}",
                    "img1_bytes": row["observation.images.image"]["bytes"],
                    "img2_bytes": row["observation.images.image2"]["bytes"],
                    "state": row["observation.state"],
                    "action": row["action"],
                })
            if len(collected) >= n_episodes:
                break
    return collected


def episode_grid(n_episodes: int = 10, out_path: Path = None):
    frames = collect_first_frames(n_episodes)
    n = len(frames)

    fig, axes = plt.subplots(n, 2, figsize=(11, n * 4.8))
    if n == 1:
        axes = axes[np.newaxis, :]
    fig.patch.set_facecolor("#111111")

    for row_i, d in enumerate(frames):
        img1 = Image.open(io.BytesIO(d["img1_bytes"]))
        img2 = Image.open(io.BytesIO(d["img2_bytes"]))

        axes[row_i][0].imshow(img1)
        axes[row_i][0].set_title(
            f'Ep {d["episode_index"]}  |  Task {d["task_index"]}: {d["task_name"][:52]}\nCamera 1 (front)',
            fontsize=7.5, color="white", pad=4, loc="left")
        axes[row_i][0].axis("off")

        axes[row_i][1].imshow(img2)
        axes[row_i][1].set_title("Camera 2 (wrist)", fontsize=7.5, color="#aaaaaa", pad=4)
        axes[row_i][1].axis("off")

    plt.suptitle(f"LIBERO — First {n} Episodes (front & wrist cameras)",
                 fontsize=14, fontweight="bold", color="white", y=1.001)
    plt.tight_layout(pad=1.5)

    if out_path is None:
        out_path = OUT / f"libero_first{n}_episodes.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved → {out_path}")


def episode_strip(episode_id: int, max_frames: int = 16, out_path: Path = None):
    """Show a strip of frames from a single episode (front camera only)."""
    for fpath in iter_parquet_files():
        df = pd.read_parquet(fpath)
        ep_df = df[df["episode_index"] == episode_id]
        if len(ep_df) > 0:
            break
    else:
        print(f"Episode {episode_id} not found in local data.")
        return

    step = max(1, len(ep_df) // max_frames)
    rows = [ep_df.iloc[i] for i in range(0, len(ep_df), step)][:max_frames]

    task_names = load_tasks()
    task_idx = int(rows[0]["task_index"])
    task_name = task_names[task_idx] if task_idx < len(task_names) else f"task_{task_idx}"

    cols = 8
    grid_rows = (len(rows) + cols - 1) // cols
    fig, axes = plt.subplots(grid_rows, cols, figsize=(cols * 2.5, grid_rows * 2.5))
    axes = np.array(axes).flatten()
    fig.patch.set_facecolor("#111111")

    for i, row in enumerate(rows):
        img = Image.open(io.BytesIO(row["observation.images.image"]["bytes"]))
        axes[i].imshow(img)
        axes[i].set_title(f't={row["timestamp"]:.1f}s', fontsize=7, color="#aaaaaa")
        axes[i].axis("off")
    for j in range(len(rows), len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"LIBERO Ep {episode_id} | {task_name}", fontsize=10,
                 fontweight="bold", color="white", y=1.01)
    plt.tight_layout(pad=0.8)

    if out_path is None:
        out_path = OUT / f"libero_episode{episode_id}_strip.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved → {out_path}")


def print_tasks():
    info = load_info()
    task_names = load_tasks()
    print(f"LIBERO dataset: {info['total_episodes']} episodes, {info['total_tasks']} tasks, "
          f"{info['total_frames']} total frames @ {info['fps']} FPS\n")
    print("All tasks:")
    for i, name in enumerate(task_names):
        print(f"  {i:2d}. {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes for grid view")
    parser.add_argument("--episode", type=int, default=None, help="Single episode ID to show as strip")
    parser.add_argument("--tasks", action="store_true", help="Print all task descriptions")
    args = parser.parse_args()

    if args.tasks:
        print_tasks()
    elif args.episode is not None:
        episode_strip(args.episode)
    else:
        episode_grid(args.episodes)


if __name__ == "__main__":
    main()
