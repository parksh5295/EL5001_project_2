#!/usr/bin/env python3
"""Create 3D animation from compressed per-step trace jsonl.gz."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D


def parse_args():
    p = argparse.ArgumentParser(description="Animate model step traces in 3D.")
    p.add_argument("--trace", type=Path, required=True, help="Input .jsonl.gz trace file")
    p.add_argument("--episode-idx", type=int, default=0, help="Eval episode index in trace")
    p.add_argument("--output", type=Path, default=Path("results/step_trace_3d.gif"))
    p.add_argument("--coord-mode", choices=("state", "timeline"), default="state")
    p.add_argument("--interval-ms", type=int, default=300)
    p.add_argument("--fps", type=int, default=4)
    p.add_argument("--dpi", type=int, default=120)
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def load_trace(path: Path, episode_idx: int):
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            row = json.loads(s)
            if int(row.get("eval_episode_idx", -1)) == episode_idx:
                rows.append(row)
    if not rows:
        raise RuntimeError(f"No rows found for eval_episode_idx={episode_idx} in {path}")
    rows.sort(key=lambda r: int(r.get("step_idx", 0)))
    return rows


def to_xyz(rows: list[dict], coord_mode: str):
    if coord_mode == "state":
        xyz = np.array([[float(r["state3"][0]), float(r["state3"][1]), float(r["state3"][2])] for r in rows], dtype=float)
    else:
        xyz = np.array(
            [
                [
                    float(r.get("step_idx", 0)),
                    float(r.get("stream_pos", 0)),
                    1.0 if r.get("pred_tactic") else 0.0,
                ]
                for r in rows
            ],
            dtype=float,
        )
    return xyz


def make_legend(ax):
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#9e9e9e", markersize=8, label="GT benign"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#ef5350", markersize=8, label="GT attack"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#42a5f5", markersize=8, label="Pred attack"),
        Line2D([0], [0], color="#1e88e5", lw=2, label="Trajectory"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8)


def main():
    args = parse_args()
    rows = load_trace(args.trace, args.episode_idx)
    xyz = to_xyz(rows, args.coord_mode)
    algo = str(rows[0].get("algorithm", "model"))
    split = str(rows[0].get("split", "val"))

    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    pad = np.maximum((maxs - mins) * 0.1, 0.5)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    def update(frame: int):
        ax.cla()
        path = xyz[: frame + 1]
        gt_attack = np.array([int(r.get("gt_attack_active", 0)) for r in rows[: frame + 1]], dtype=int)
        pred_attack = np.array([1 if r.get("pred_tactic") else 0 for r in rows[: frame + 1]], dtype=int)

        benign_idx = np.where(gt_attack == 0)[0]
        attack_idx = np.where(gt_attack == 1)[0]
        pred_idx = np.where(pred_attack == 1)[0]

        if len(benign_idx):
            ax.scatter(path[benign_idx, 0], path[benign_idx, 1], path[benign_idx, 2], c="#9e9e9e", s=24, alpha=0.8)
        if len(attack_idx):
            ax.scatter(path[attack_idx, 0], path[attack_idx, 1], path[attack_idx, 2], c="#ef5350", s=32, alpha=0.9)
        if len(pred_idx):
            ax.scatter(path[pred_idx, 0], path[pred_idx, 1], path[pred_idx, 2], c="#42a5f5", s=18, alpha=0.9)

        ax.plot(path[:, 0], path[:, 1], path[:, 2], color="#1e88e5", linewidth=2.0)
        cur = rows[frame]
        ax.scatter(path[-1, 0], path[-1, 1], path[-1, 2], c="gold", s=140, marker="^", edgecolors="black")

        ax.set_xlim(mins[0] - pad[0], maxs[0] + pad[0])
        ax.set_ylim(mins[1] - pad[1], maxs[1] + pad[1])
        ax.set_zlim(mins[2] - pad[2], maxs[2] + pad[2])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"{algo} | {split} | eval_ep={args.episode_idx} | step={frame+1}/{len(rows)}")
        status = (
            f"action={cur.get('action_name')} | reward={float(cur.get('reward', 0.0)):.3f} | "
            f"gt={cur.get('gt_tactic')} | pred={cur.get('pred_tactic')}"
        )
        ax.text2D(0.01, 0.97, status, transform=ax.transAxes, fontsize=9)
        make_legend(ax)
        return ax,

    ani = FuncAnimation(fig, update, frames=len(rows), interval=args.interval_ms, blit=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".gif":
        ani.save(str(args.output), writer=PillowWriter(fps=args.fps), dpi=args.dpi)
    else:
        ani.save(str(args.output), fps=args.fps, dpi=args.dpi)
    print(f"saved animation: {args.output.resolve()}")

    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()

