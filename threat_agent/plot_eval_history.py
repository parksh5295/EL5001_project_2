#!/usr/bin/env python3
"""Plot training/eval curves from eval_history jsonl files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Plot eval history jsonl files.")
    p.add_argument("--eval-history-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument(
        "--metrics",
        type=str,
        default="event_micro_f1,segment_boundary_f1,attack_step_overlap_hit,attack_step_pred_coverage,majority_gain,avg_return",
    )
    return p.parse_args()


def try_import_matplotlib():
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415

        return plt
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required to draw plots. Install it with 'pip install matplotlib'."
        ) from e


def load_history(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
    return rows


def extract_series(rows: list[dict], split: str, metric: str):
    x = []
    y = []
    for r in rows:
        if r.get("stage") != "periodic_eval":
            continue
        if r.get("split") != split:
            continue
        m = (r.get("metrics") or {}).get(metric)
        if m is None:
            continue
        x.append(int(r.get("episode", 0)))
        y.append(float(m))
    return x, y


def main():
    args = parse_args()
    plt = try_import_matplotlib()
    out_dir = args.output_dir or (args.eval_history_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    files = sorted(args.eval_history_dir.glob("*_eval_history.jsonl"))
    if not files:
        raise RuntimeError(f"No *_eval_history.jsonl files found in {args.eval_history_dir}")

    for fp in files:
        rows = load_history(fp)
        algo = fp.stem.replace("_eval_history", "")
        for metric in metrics:
            x_val, y_val = extract_series(rows, "val", metric)
            if not x_val:
                continue
            plt.figure(figsize=(7, 4))
            plt.plot(x_val, y_val, label=f"{algo}/val", linewidth=1.6)
            plt.xlabel("episode")
            plt.ylabel(metric)
            plt.title(f"{algo} - {metric}")
            plt.grid(alpha=0.3)
            plt.legend()
            out_path = out_dir / f"{algo}__{metric}.png"
            plt.tight_layout()
            plt.savefig(out_path, dpi=130)
            plt.close()
            print(f"saved: {out_path}")


if __name__ == "__main__":
    main()

