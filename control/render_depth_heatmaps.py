"""Render an episode's depth_overhead/*.npy maps as viewable heatmap PNGs.

Usage (from control/, venv active):
    python render_depth_heatmaps.py outputs/vla_episode_004
    python render_depth_heatmaps.py outputs/vla_episode_004 --contact-sheet
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render_one(npy_path: Path, out_path: Path, vmin: float, vmax: float) -> None:
    depth = np.load(npy_path)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(depth, cmap="turbo", vmin=vmin, vmax=vmax)
    ax.set_title(npy_path.stem)
    ax.axis("off")
    fig.colorbar(im, ax=ax, label="metres")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def render_contact_sheet(npy_paths: list[Path], out_path: Path, vmin: float, vmax: float) -> None:
    cols = min(5, len(npy_paths))
    rows = -(-len(npy_paths) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 2.4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, path in zip(axes, npy_paths):
        depth = np.load(path)
        im = ax.imshow(depth, cmap="turbo", vmin=vmin, vmax=vmax)
        ax.set_title(path.stem, fontsize=8)
        ax.axis("off")
    for ax in axes[len(npy_paths):]:
        ax.axis("off")
    fig.colorbar(im, ax=axes.tolist(), label="metres", shrink=0.6)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("episode_dir", type=Path)
    parser.add_argument("--contact-sheet", action="store_true",
                         help="Also render one combined grid image of every depth map.")
    parser.add_argument("--vmin", type=float, default=None, help="Colour-scale min (metres). Auto if omitted.")
    parser.add_argument("--vmax", type=float, default=None, help="Colour-scale max (metres). Auto if omitted.")
    args = parser.parse_args()

    depth_dir = args.episode_dir / "depth_overhead"
    npy_paths = sorted(depth_dir.glob("*.npy"))
    if not npy_paths:
        parser.error(f"No depth_overhead/*.npy files under {args.episode_dir}")

    if args.vmin is None or args.vmax is None:
        arrays = [np.load(p) for p in npy_paths]
        finite = np.concatenate([a[np.isfinite(a)].ravel() for a in arrays])
        vmin = args.vmin if args.vmin is not None else float(np.percentile(finite, 1))
        vmax = args.vmax if args.vmax is not None else float(np.percentile(finite, 99))
    else:
        vmin, vmax = args.vmin, args.vmax
    print(f"Colour scale: {vmin:.3f}m - {vmax:.3f}m (1st-99th percentile of valid depth)")

    out_dir = args.episode_dir / "depth_heatmaps"
    out_dir.mkdir(exist_ok=True)
    for npy_path in npy_paths:
        out_path = out_dir / f"{npy_path.stem}.png"
        render_one(npy_path, out_path, vmin, vmax)
        print(f"Saved -> {out_path}")

    if args.contact_sheet:
        sheet_path = args.episode_dir / "depth_heatmaps_contact_sheet.png"
        render_contact_sheet(npy_paths, sheet_path, vmin, vmax)
        print(f"Saved -> {sheet_path}")


if __name__ == "__main__":
    main()
