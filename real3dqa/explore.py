"""
Real-3DQA dataset explorer.

Data layout (relative to repo root):
  data/real3dqa/point_clouds/*.pth   — 3D point clouds (xyz, rgb, labels, instance_ids)
  data/real3dqa/annotations/*.jsonl  — QA pairs with position/rotation metadata

Usage:
  python real3dqa/explore.py                   # bird's-eye grid of all scenes
  python real3dqa/explore.py --scene scene0025_00          # 3D view of one scene
  python real3dqa/explore.py --scene scene0025_00 --view 3d
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "real3dqa"
OUT  = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def load_scene(scene_id: str):
    path = DATA / "point_clouds" / f"{scene_id}.pth"
    if not path.exists():
        raise FileNotFoundError(f"Point cloud not found: {path}")
    data = torch.load(path, map_location="cpu", weights_only=False)
    xyz = np.asarray(data[0], dtype=np.float32)
    rgb = np.asarray(data[1], dtype=np.float32)
    rgb = np.clip(rgb / 255.0, 0, 1)
    return xyz, rgb


def load_annotations():
    records = []
    for f in sorted((DATA / "annotations").glob("*.jsonl")):
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def bird_eye_grid(scene_ids, out_path, n_pts=10_000):
    n = len(scene_ids)
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = np.array(axes).flatten()

    for i, sid in enumerate(scene_ids):
        ax = axes[i]
        try:
            xyz, rgb = load_scene(sid)
            idx = np.random.choice(len(xyz), min(n_pts, len(xyz)), replace=False)
            rgb_g = np.clip(rgb[idx] ** 0.5, 0, 1)
            ax.scatter(xyz[idx, 0], xyz[idx, 1], c=rgb_g, s=0.5, linewidths=0)
        except FileNotFoundError:
            ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(sid, fontsize=8)
        ax.set_aspect("equal")
        ax.axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Real-3DQA — All Scenes (bird's-eye view)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")


def perspective_3d(scene_id, out_path, n_pts=80_000):
    xyz, rgb = load_scene(scene_id)
    idx = np.random.choice(len(xyz), min(n_pts, len(xyz)), replace=False)
    xyz_s = xyz[idx]
    rgb_s = np.clip(rgb[idx] ** 0.5, 0, 1)

    fig = plt.figure(figsize=(14, 10), facecolor="#1a1a2e")
    ax = fig.add_subplot(111, projection="3d", facecolor="#1a1a2e")
    ax.scatter(xyz_s[:, 0], xyz_s[:, 1], xyz_s[:, 2],
               c=rgb_s, s=0.6, linewidths=0, depthshade=False, alpha=0.85)
    ax.set_title(f"Real-3DQA — {scene_id} (3D)", fontsize=13, color="white", pad=18)
    ax.set_xlabel("X", color="#aaaaaa"); ax.set_ylabel("Y", color="#aaaaaa"); ax.set_zlabel("Z", color="#aaaaaa")
    ax.tick_params(colors="#888888", labelsize=7)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False; pane.set_edgecolor("#333355")
    ax.view_init(elev=20, azim=-50)
    ax.set_box_aspect([np.ptp(xyz_s[:, i]) for i in range(3)])
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default=None, help="Scene ID, e.g. scene0025_00")
    parser.add_argument("--view", choices=["birdseye", "3d"], default="birdseye")
    args = parser.parse_args()

    if args.scene:
        out = OUT / f"{args.scene}_3d.png" if args.view == "3d" else OUT / f"{args.scene}_birdseye.png"
        if args.view == "3d":
            perspective_3d(args.scene, out)
        else:
            bird_eye_grid([args.scene], out)
    else:
        scene_ids = sorted(p.stem for p in (DATA / "point_clouds").glob("*.pth"))
        print(f"Found {len(scene_ids)} scenes: {scene_ids}")
        bird_eye_grid(scene_ids, OUT / "real3dqa_all_scenes.png")

    print("\nAnnotation sample:")
    records = load_annotations()
    print(f"  Total QA pairs: {len(records)}")
    for r in records[:3]:
        print(f"  [{r.get('scene_id')}] Q: {r.get('question')}  A: {r.get('answer')}")


if __name__ == "__main__":
    main()
