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
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def parse_args():
    p = argparse.ArgumentParser(description="2D trace embedding animation.")
    p.add_argument("--trace", type=Path, required=True, help="input .jsonl.gz trace")
    p.add_argument(
        "--episode-idx",
        type=int,
        default=-1,
        help="-1 = auto-pick episode",
    )
    p.add_argument("--embedding", choices=("tsne", "pca"), default="tsne")
    p.add_argument(
        "--fit-label",
        choices=("pred", "gt"),
        default="gt",
        help="label for boundary fit (pred or gt)",
    )
    p.add_argument(
        "--fit-scope",
        choices=("global_prefix", "episode_prefix", "global_full"),
        default="global_full",
        help="boundary fit scope",
    )
    p.add_argument(
        "--boundary-model",
        choices=("knn", "logreg"),
        default="knn",
        help="boundary model (knn or logreg)",
    )
    p.add_argument("--knn-k", type=int, default=15, help="k for knn")
    p.add_argument(
        "--traj-tail",
        type=int,
        default=6,
        help="recent steps to highlight",
    )
    p.add_argument(
        "--show-fit-centroids",
        action="store_true",
        help="show fit centroids",
    )
    p.add_argument(
        "--view-mode",
        choices=("full", "boundary_only"),
        default="full",
        help="full or boundary_only",
    )
    p.add_argument("--interval-ms", type=int, default=260)
    p.add_argument("--fps", type=int, default=4)
    p.add_argument("--dpi", type=int, default=120)
    p.add_argument("--output", type=Path, default=Path("results/trace_embedding_2d.gif"))
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def load_rows(path: Path):
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    if not rows:
        raise RuntimeError(f"No rows in trace: {path}")
    return rows


def split_by_episode(rows: list[dict]) -> dict[int, list[dict]]:
    eps: dict[int, list[dict]] = {}
    for i, r in enumerate(rows):
        r["_global_idx"] = i
        ep = int(r.get("eval_episode_idx", -1))
        eps.setdefault(ep, []).append(r)
    for ep in eps:
        eps[ep].sort(key=lambda x: int(x.get("step_idx", 0)))
    return eps


def pick_episode(eps: dict[int, list[dict]], requested: int) -> int:
    if requested >= 0:
        if requested not in eps:
            raise RuntimeError(f"eval_episode_idx={requested} not found.")
        return requested

    best_ep = None
    best_score = -math.inf
    for ep, rows in eps.items():
        pred = [r.get("pred_tactic") for r in rows]
        gt = [int(r.get("gt_attack_active", 0)) for r in rows]
        pred_attack = [1 if p else 0 for p in pred]
        pred_changes = sum(1 for i in range(1, len(pred)) if pred[i] != pred[i - 1])
        overlap = sum(1 for g, p in zip(gt, pred_attack) if g == 1 and p == 1)
        score = 2.0 * pred_changes + overlap
        if score > best_score:
            best_score = score
            best_ep = ep
    if best_ep is None:
        raise RuntimeError("Could not auto-select episode.")
    return best_ep


def build_features(rows: list[dict]):
    action_names = sorted({str(r.get("action_name", "NA")) for r in rows})
    action_to_idx = {a: i for i, a in enumerate(action_names)}
    feats = []
    gt = []
    pred = []
    for r in rows:
        s3 = r.get("state3") or [0.0, 0.0, 0.0]
        action_idx = float(action_to_idx.get(str(r.get("action_name", "NA")), 0))
        feat = [
            float(s3[0]),
            float(s3[1]),
            float(s3[2]),
            float(r.get("step_idx", 0)),
            float(r.get("stream_pos", 0)),
            float(r.get("reward", 0.0)),
            float(r.get("step_event_f1", 0.0)),
            action_idx,
        ]
        feats.append(feat)
        gt.append(int(r.get("gt_attack_active", 0)))
        pred.append(1 if r.get("pred_tactic") else 0)
    return np.asarray(feats, dtype=float), np.asarray(gt, dtype=int), np.asarray(pred, dtype=int)


def embed_features(x: np.ndarray, method: str):
    x_scaled = StandardScaler().fit_transform(x)
    if method == "pca":
        return PCA(n_components=2, random_state=42).fit_transform(x_scaled)

    n = x_scaled.shape[0]
    # Keep perplexity valid for small samples.
    perp = min(30, max(5, (n - 1) // 3))
    model = TSNE(
        n_components=2,
        perplexity=float(perp),
        init="pca",
        learning_rate="auto",
        random_state=42,
    )
    return model.fit_transform(x_scaled)


def fit_boundary(x2: np.ndarray, y: np.ndarray, model_name: str, knn_k: int):
    uniq, counts = np.unique(y, return_counts=True)
    if len(uniq) < 2:
        return None
    if np.min(counts) < 2:
        return None
    if model_name == "knn":
        k = int(max(3, min(knn_k, len(x2))))
        clf = KNeighborsClassifier(n_neighbors=k, weights="distance")
    else:
        clf = LogisticRegression(random_state=42, max_iter=500)
    clf.fit(x2, y)
    return clf


def masked_probability_grid(clf, grid: np.ndarray, x_fit: np.ndarray):
    p = clf.predict_proba(grid)[:, 1]
    if len(x_fit) < 3:
        return p

    k = min(6, len(x_fit))
    nn_fit = NearestNeighbors(n_neighbors=k)
    nn_fit.fit(x_fit)
    d_fit, _ = nn_fit.kneighbors(x_fit)
    local_scale = float(np.quantile(d_fit[:, -1], 0.9))
    radius = max(local_scale * 1.8, 0.25)

    nn_grid = NearestNeighbors(n_neighbors=1)
    nn_grid.fit(x_fit)
    d_grid, _ = nn_grid.kneighbors(grid)
    far_mask = d_grid[:, 0] > radius
    p[far_mask] = np.nan
    return p


def main():
    args = parse_args()
    rows = load_rows(args.trace)
    eps = split_by_episode(rows)
    ep_idx = pick_episode(eps, args.episode_idx)
    ep_rows = eps[ep_idx]
    print(f"selected eval_episode_idx={ep_idx} (episodes={len(eps)})")

    x, y_gt, y_pred = build_features(rows)
    emb = embed_features(x, args.embedding)
    labels_for_fit = y_pred if args.fit_label == "pred" else y_gt

    ep_global = [int(r["_global_idx"]) for r in ep_rows]
    ep_emb = emb[ep_global]
    all_step_idx = np.array([int(r.get("step_idx", 0)) for r in rows], dtype=int)

    algo = str(ep_rows[0].get("algorithm", "model"))
    split = str(ep_rows[0].get("split", "test"))

    x_min, x_max = float(np.min(emb[:, 0])), float(np.max(emb[:, 0]))
    y_min, y_max = float(np.min(emb[:, 1])), float(np.max(emb[:, 1]))
    x_pad = max(0.5, 0.08 * (x_max - x_min + 1e-6))
    y_pad = max(0.5, 0.08 * (y_max - y_min + 1e-6))

    gx = np.linspace(x_min - x_pad, x_max + x_pad, 120)
    gy = np.linspace(y_min - y_pad, y_max + y_pad, 120)
    xx, yy = np.meshgrid(gx, gy)
    grid = np.c_[xx.ravel(), yy.ravel()]

    fig, ax = plt.subplots(figsize=(9.8, 7.4))

    def update(frame: int):
        ax.cla()
        end = frame + 1
        cur = ep_rows[frame]

        benign_idx = np.where(y_gt == 0)[0]
        attack_idx = np.where(y_gt == 1)[0]
        ax.scatter(
            emb[benign_idx, 0],
            emb[benign_idx, 1],
            c="#90a4ae",
            s=20,
            alpha=0.45,
            edgecolors="white",
            linewidths=0.25,
            label="All GT benign",
        )
        ax.scatter(
            emb[attack_idx, 0],
            emb[attack_idx, 1],
            c="#e53935",
            s=22,
            alpha=0.50,
            edgecolors="white",
            linewidths=0.25,
            label="All GT attack",
        )

        tr = ep_emb[:end]
        if args.view_mode == "full" and len(tr) > 0:
            ax.scatter(tr[:, 0], tr[:, 1], c="#607d8b", s=20, alpha=0.18, label="Episode history")

            cvals = np.linspace(0.0, 1.0, len(tr))
            ax.scatter(tr[:, 0], tr[:, 1], c=cvals, cmap="viridis", s=36, alpha=0.95, label="Episode points(time)")

            tail = max(2, int(args.traj_tail))
            t0 = max(0, len(tr) - tail)
            tr_tail = tr[t0:]
            ax.plot(
                tr_tail[:, 0],
                tr_tail[:, 1],
                color="#1565c0",
                linewidth=2.5,
                alpha=0.95,
                label=f"Recent tail({len(tr_tail)})",
            )

            for i in range(1, len(tr_tail)):
                x0, y0 = tr_tail[i - 1]
                x1, y1 = tr_tail[i]
                dx, dy = (x1 - x0), (y1 - y0)
                ax.quiver(
                    x0,
                    y0,
                    dx,
                    dy,
                    angles="xy",
                    scale_units="xy",
                    scale=1.0,
                    color="#0d47a1",
                    width=0.003,
                    alpha=0.85,
                )
            ax.scatter(tr[-1, 0], tr[-1, 1], c="gold", s=180, marker="*", edgecolors="black", zorder=5, label="Current")

        if args.fit_scope == "global_prefix":
            cur_step = int(cur.get("step_idx", 0))
            fit_idx = np.where(all_step_idx <= cur_step)[0]
        elif args.fit_scope == "global_full":
            fit_idx = np.arange(len(rows))
        else:
            fit_idx = np.array(ep_global[:end], dtype=int)
        x_fit = emb[fit_idx]
        y_fit = labels_for_fit[fit_idx]
        fit_b = np.where(y_fit == 0)[0]
        fit_a = np.where(y_fit == 1)[0]
        if len(fit_b):
            ax.scatter(x_fit[fit_b, 0], x_fit[fit_b, 1], c="#455a64", s=14, alpha=0.18, label="Fit benign")
        if len(fit_a):
            ax.scatter(x_fit[fit_a, 0], x_fit[fit_a, 1], c="#b71c1c", s=15, alpha=0.20, label="Fit attack")
        clf = fit_boundary(x_fit, y_fit, args.boundary_model, args.knn_k)
        if clf is not None:
            p = masked_probability_grid(clf, grid, x_fit).reshape(xx.shape)
            ax.contourf(
                xx,
                yy,
                p,
                levels=[0.0, 0.25, 0.5, 0.75, 1.0],
                colors=["#cfd8dc", "#b0bec5", "#90caf9", "#64b5f6"],
                alpha=0.33,
            )
            ax.contour(xx, yy, p, levels=[0.5], colors=["#0d47a1"], linewidths=2.6, linestyles="-")
        else:
            ax.text(
                0.02,
                0.93,
                "boundary unavailable (single-class prefix)",
                transform=ax.transAxes,
                fontsize=9,
                color="#6d4c41",
            )

        if args.show_fit_centroids and len(x_fit) > 0:
            for cls, color, name in [(0, "#455a64", "fit benign center"), (1, "#b71c1c", "fit attack center")]:
                cls_idx = np.where(y_fit == cls)[0]
                if len(cls_idx) == 0:
                    continue
                cxy = np.mean(x_fit[cls_idx], axis=0)
                ax.scatter(
                    cxy[0],
                    cxy[1],
                    marker="X",
                    s=150,
                    c=color,
                    edgecolors="white",
                    linewidths=0.8,
                    zorder=6,
                    label=name,
                )

        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.grid(alpha=0.25)
        ax.set_xlabel(f"{args.embedding.upper()}-1")
        ax.set_ylabel(f"{args.embedding.upper()}-2")
        ax.set_title(
            f"{algo} | {split} | eval_ep={ep_idx} | step={frame+1}/{len(ep_rows)} | "
            f"boundary={args.fit_label}/{args.fit_scope}/{args.boundary_model}",
            fontsize=11,
        )
        ax.text(
            0.02,
            0.97,
            f"fit_samples={len(fit_idx)}",
            transform=ax.transAxes,
            fontsize=9,
            va="top",
            color="#263238",
        )
        status = (
            f"action={cur.get('action_name')} | reward={float(cur.get('reward', 0.0)):.3f} | "
            f"gt={cur.get('gt_tactic')} | pred={cur.get('pred_tactic')}"
        )
        ax.text(0.01, 0.01, status, transform=ax.transAxes, fontsize=9, va="bottom")
        ax.legend(loc="upper right", fontsize=8)
        return ax,

    ani = FuncAnimation(fig, update, frames=len(ep_rows), interval=args.interval_ms, blit=False)
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

