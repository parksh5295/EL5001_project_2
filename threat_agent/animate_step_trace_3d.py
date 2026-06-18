#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D


def parse_args():
    p = argparse.ArgumentParser(description="Animate step traces.")
    p.add_argument("--trace", type=Path, required=True, help="input .jsonl.gz trace")
    p.add_argument(
        "--episode-idx",
        type=int,
        default=-1,
        help="-1 = auto-pick episode",
    )
    p.add_argument("--output", type=Path, default=Path("results/step_trace.gif"))
    p.add_argument("--coord-mode", choices=("state", "timeline"), default="timeline")
    p.add_argument(
        "--style",
        choices=("2d", "2d_coord", "3d"),
        default="2d",
        help="2d, 2d_coord, or 3d",
    )
    p.add_argument("--interval-ms", type=int, default=300)
    p.add_argument("--fps", type=int, default=4)
    p.add_argument("--dpi", type=int, default=120)
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def load_trace_rows(path: Path):
    rows: list[dict] = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
    if not rows:
        raise RuntimeError(f"No rows found in trace: {path}")
    return rows


def split_by_episode(rows: list[dict]) -> dict[int, list[dict]]:
    eps: dict[int, list[dict]] = {}
    for r in rows:
        ep = int(r.get("eval_episode_idx", -1))
        eps.setdefault(ep, []).append(r)
    for ep in list(eps.keys()):
        eps[ep].sort(key=lambda r: int(r.get("step_idx", 0)))
    return eps


def choose_episode(episodes: dict[int, list[dict]], requested_idx: int) -> int:
    if requested_idx >= 0:
        if requested_idx not in episodes:
            raise RuntimeError(f"eval_episode_idx={requested_idx} not found in trace")
        return requested_idx

    best_ep = None
    best_score = -math.inf
    for ep, rows in episodes.items():
        preds = [r.get("pred_tactic") for r in rows]
        gt_att = [int(r.get("gt_attack_active", 0)) for r in rows]
        pred_att = [1 if r.get("pred_tactic") else 0 for r in rows]
        changes = sum(1 for i in range(1, len(preds)) if preds[i] != preds[i - 1])
        overlap = sum(1 for g, p in zip(gt_att, pred_att) if g == 1 and p == 1)
        score = 2.0 * changes + 1.0 * overlap
        if score > best_score:
            best_score = score
            best_ep = ep
    if best_ep is None:
        raise RuntimeError("Could not auto-select episode.")
    return best_ep


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


def animate_2d(rows: list[dict], args):
    algo = str(rows[0].get("algorithm", "model"))
    split = str(rows[0].get("split", "val"))
    steps = np.array([int(r.get("step_idx", i)) for i, r in enumerate(rows)], dtype=int)
    gt_attack = np.array([int(r.get("gt_attack_active", 0)) for r in rows], dtype=int)
    pred_attack = np.array([1 if r.get("pred_tactic") else 0 for r in rows], dtype=int)
    rewards = np.array([float(r.get("reward", 0.0)) for r in rows], dtype=float)
    cum_rewards = np.cumsum(rewards)

    tactic_set = sorted(
        {str(r.get("gt_tactic")) for r in rows if r.get("gt_tactic") is not None}
        | {str(r.get("pred_tactic")) for r in rows if r.get("pred_tactic") is not None}
    )
    tactic_to_y = {None: 0}
    for i, t in enumerate(tactic_set, start=1):
        tactic_to_y[t] = i

    gt_y = np.array([tactic_to_y.get(r.get("gt_tactic"), 0) for r in rows], dtype=float)
    pred_y = np.array([tactic_to_y.get(r.get("pred_tactic"), 0) for r in rows], dtype=float)

    fig, (ax0, ax1, ax2) = plt.subplots(
        3, 1, figsize=(11, 8), gridspec_kw={"height_ratios": [1.0, 1.4, 1.2]}, sharex=True
    )

    x_min = float(steps.min() - 0.5)
    x_max = float(steps.max() + 0.5)
    rew_pad = max(0.2, 0.1 * float(np.max(np.abs(rewards)) + 1e-8))
    cum_pad = max(0.5, 0.1 * float(np.max(np.abs(cum_rewards)) + 1e-8))

    def update(frame: int):
        ax0.cla()
        ax1.cla()
        ax2.cla()
        end = frame + 1
        x = steps[:end]
        cur = rows[frame]

        ax0.step(x, gt_attack[:end], where="post", linewidth=2.0, color="#ef5350", label="GT attack")
        ax0.step(x, pred_attack[:end], where="post", linewidth=2.0, color="#42a5f5", label="Pred attack")
        ax0.fill_between(x, 0, gt_attack[:end], color="#ef5350", alpha=0.16, step="post")
        ax0.fill_between(x, 0, pred_attack[:end], color="#42a5f5", alpha=0.12, step="post")
        ax0.axvline(float(steps[frame]), color="goldenrod", linestyle="--", linewidth=1.5)
        ax0.set_ylim(-0.1, 1.2)
        ax0.set_yticks([0, 1])
        ax0.set_ylabel("Attack")
        ax0.set_xlim(x_min, x_max)
        ax0.grid(alpha=0.3)
        ax0.legend(loc="upper right", fontsize=8)

        ax1.plot(x, gt_y[:end], color="#d32f2f", linewidth=2.2, marker="o", markersize=3, label="GT tactic")
        ax1.plot(
            x,
            pred_y[:end],
            color="#1976d2",
            linewidth=2.2,
            marker="s",
            markersize=3,
            linestyle="--",
            label="Pred tactic",
        )
        ax1.axvline(float(steps[frame]), color="goldenrod", linestyle="--", linewidth=1.5)
        yticks = list(range(0, len(tactic_set) + 1))
        ylabels = ["benign"] + tactic_set
        ax1.set_yticks(yticks)
        ax1.set_yticklabels(ylabels, fontsize=8)
        ax1.set_ylabel("Tactic")
        ax1.grid(alpha=0.3)
        ax1.legend(loc="upper right", fontsize=8)

        ax2.bar(x, rewards[:end], width=0.75, color="#90a4ae", alpha=0.75, label="step reward")
        ax2.plot(x, cum_rewards[:end], color="#2e7d32", linewidth=2.0, marker=".", label="cumulative reward")
        ax2.axvline(float(steps[frame]), color="goldenrod", linestyle="--", linewidth=1.5)
        ax2.set_ylim(float(rewards.min() - rew_pad), float(rewards.max() + rew_pad))
        ax2_t = ax2.twinx()
        ax2_t.set_ylim(float(cum_rewards.min() - cum_pad), float(cum_rewards.max() + cum_pad))
        ax2.set_xlabel("step")
        ax2.set_ylabel("step reward")
        ax2_t.set_ylabel("cum reward", color="#2e7d32")
        ax2.grid(alpha=0.25)
        ax2.legend(loc="upper left", fontsize=8)

        title = f"{algo} | {split} | eval_ep={int(cur.get('eval_episode_idx', 0))} | step={frame+1}/{len(rows)}"
        status = (
            f"action={cur.get('action_name')} | reward={float(cur.get('reward', 0.0)):.3f} | "
            f"gt={cur.get('gt_tactic')} | pred={cur.get('pred_tactic')}"
        )
        fig.suptitle(title, fontsize=12, y=0.98)
        fig.text(0.01, 0.01, status, fontsize=9)
        return ax0, ax1, ax2

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


def animate_2d_coord(rows: list[dict], args):
    algo = str(rows[0].get("algorithm", "model"))
    split = str(rows[0].get("split", "val"))
    steps = np.array([float(r.get("step_idx", i)) for i, r in enumerate(rows)], dtype=float)
    pos = np.array([float(r.get("stream_pos", i)) for i, r in enumerate(rows)], dtype=float)
    gt_attack = np.array([int(r.get("gt_attack_active", 0)) for r in rows], dtype=int)
    pred_attack = np.array([1 if r.get("pred_tactic") else 0 for r in rows], dtype=int)
    rewards = np.array([float(r.get("reward", 0.0)) for r in rows], dtype=float)

    x_pad = max(0.5, 0.05 * float(steps.max() - steps.min() + 1))
    y_pad = max(0.5, 0.05 * float(pos.max() - pos.min() + 1))

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10.5, 7.2), gridspec_kw={"height_ratios": [1.5, 1.0]})

    def update(frame: int):
        ax0.cla()
        ax1.cla()
        end = frame + 1
        x = steps[:end]
        y = pos[:end]
        cur = rows[frame]

        ax0.plot(x, y, color="#1e88e5", linewidth=2.0, label="trajectory")

        benign_idx = np.where(gt_attack[:end] == 0)[0]
        attack_idx = np.where(gt_attack[:end] == 1)[0]
        pred_idx = np.where(pred_attack[:end] == 1)[0]

        if len(benign_idx):
            ax0.scatter(x[benign_idx], y[benign_idx], c="#9e9e9e", s=35, alpha=0.8, label="GT benign")
        if len(attack_idx):
            ax0.scatter(x[attack_idx], y[attack_idx], c="#ef5350", s=44, alpha=0.9, label="GT attack")
        if len(pred_idx):
            ax0.scatter(x[pred_idx], y[pred_idx], c="#42a5f5", s=28, alpha=0.9, label="Pred attack")

        ax0.scatter(x[-1], y[-1], c="gold", s=180, marker="^", edgecolors="black", zorder=5)
        ax0.set_xlim(float(steps.min() - x_pad), float(steps.max() + x_pad))
        ax0.set_ylim(float(pos.min() - y_pad), float(pos.max() + y_pad))
        ax0.set_xlabel("step_idx")
        ax0.set_ylabel("stream_pos")
        ax0.grid(alpha=0.3)
        ax0.legend(loc="upper left", fontsize=8)
        ax0.set_title(
            f"{algo} | {split} | eval_ep={int(cur.get('eval_episode_idx', 0))} | step={frame+1}/{len(rows)}",
            fontsize=11,
        )

        ax1.bar(x, rewards[:end], color="#90a4ae", alpha=0.75, width=0.8, label="step reward")
        ax1.plot(x, np.cumsum(rewards[:end]), color="#2e7d32", linewidth=2.0, label="cumulative reward")
        ax1.axvline(float(steps[frame]), color="goldenrod", linestyle="--", linewidth=1.5)
        ax1.set_xlim(float(steps.min() - x_pad), float(steps.max() + x_pad))
        ax1.set_xlabel("step_idx")
        ax1.set_ylabel("reward")
        ax1.grid(alpha=0.25)
        ax1.legend(loc="upper left", fontsize=8)

        status = (
            f"action={cur.get('action_name')} | reward={float(cur.get('reward', 0.0)):.3f} | "
            f"gt={cur.get('gt_tactic')} | pred={cur.get('pred_tactic')}"
        )
        fig.text(0.01, 0.01, status, fontsize=9)
        return ax0, ax1

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


def animate_3d(rows: list[dict], args):
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
        ax.set_title(f"{algo} | {split} | eval_ep={int(cur.get('eval_episode_idx', 0))} | step={frame+1}/{len(rows)}")
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


def main():
    args = parse_args()
    all_rows = load_trace_rows(args.trace)
    episodes = split_by_episode(all_rows)
    selected_ep = choose_episode(episodes, args.episode_idx)
    rows = episodes[selected_ep]
    print(f"selected eval_episode_idx={selected_ep} (total episodes in trace: {len(episodes)})")

    if args.style == "2d":
        animate_2d(rows, args)
    elif args.style == "2d_coord":
        animate_2d_coord(rows, args)
    else:
        animate_3d(rows, args)


if __name__ == "__main__":
    main()

