#!/usr/bin/env python3
"""Run full comparison on stream RL environment."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str]):
    print(" ".join(cmd))
    r = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if r.stdout:
        print(r.stdout)
    if r.stderr:
        print(r.stderr)
    if r.returncode != 0:
        raise RuntimeError(f"Command failed ({r.returncode}): {' '.join(cmd)}")


def parse_args():
    p = argparse.ArgumentParser(description="Stream RL comparison runner.")
    p.add_argument("--stream-data", type=Path, default=Path("results/stream_events.ndjson"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tabular-episodes", type=int, default=3000)
    p.add_argument("--deep-episodes", type=int, default=1500)
    p.add_argument("--eval-episodes", type=int, default=100)
    p.add_argument("--output-json", type=Path, default=Path("results/stream_compare_summary.json"))
    p.add_argument("--output-csv", type=Path, default=Path("results/stream_compare_summary.csv"))
    return p.parse_args()


def main():
    args = parse_args()
    py = sys.executable
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "results"

    tab_path = out_dir / "stream_tabular_metrics.json"
    dqn_path = out_dir / "stream_dqn_metrics.json"
    rf_path = out_dir / "stream_reinforce_metrics.json"
    a2c_path = out_dir / "stream_a2c_metrics.json"

    run_cmd(
        [
            py,
            "-m",
            "threat_agent.train_stream_tabular",
            "--stream-data",
            str(args.stream_data),
            "--algorithm",
            "all",
            "--episodes",
            str(args.tabular_episodes),
            "--eval-episodes",
            str(args.eval_episodes),
            "--seed",
            str(args.seed),
            "--output",
            str(tab_path),
        ]
    )
    run_cmd(
        [
            py,
            "-m",
            "threat_agent.train_stream_dqn",
            "--stream-data",
            str(args.stream_data),
            "--episodes",
            str(args.deep_episodes),
            "--eval-episodes",
            str(args.eval_episodes),
            "--seed",
            str(args.seed),
            "--save-model",
            str(root / "checkpoints" / "stream_dqn.pt"),
            "--metrics-output",
            str(dqn_path),
        ]
    )
    run_cmd(
        [
            py,
            "-m",
            "threat_agent.train_stream_reinforce",
            "--stream-data",
            str(args.stream_data),
            "--episodes",
            str(args.deep_episodes),
            "--eval-episodes",
            str(args.eval_episodes),
            "--seed",
            str(args.seed),
            "--save-model",
            str(root / "checkpoints" / "stream_reinforce.pt"),
            "--metrics-output",
            str(rf_path),
        ]
    )
    run_cmd(
        [
            py,
            "-m",
            "threat_agent.train_stream_a2c",
            "--stream-data",
            str(args.stream_data),
            "--episodes",
            str(args.deep_episodes),
            "--eval-episodes",
            str(args.eval_episodes),
            "--seed",
            str(args.seed),
            "--save-model",
            str(root / "checkpoints" / "stream_a2c.pt"),
            "--metrics-output",
            str(a2c_path),
        ]
    )

    summary = {
        "tabular": json.loads(tab_path.read_text(encoding="utf-8")),
        "deep": {
            "stream_dqn": json.loads(dqn_path.read_text(encoding="utf-8")),
            "stream_reinforce": json.loads(rf_path.read_text(encoding="utf-8")),
            "stream_a2c": json.loads(a2c_path.read_text(encoding="utf-8")),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    rows = []
    for algo, split_map in summary["tabular"].items():
        for split_name, metric in split_map.items():
            rows.append(
                {
                    "algorithm": algo,
                    "split": split_name,
                    "accuracy": metric.get("accuracy"),
                    "balanced_accuracy": metric.get("balanced_accuracy"),
                    "macro_f1": metric.get("macro_f1"),
                    "avg_return": metric.get("avg_return"),
                    "avg_steps": metric.get("avg_steps"),
                    "declare_step1_ratio": metric.get("declare_step1_ratio"),
                    "majority_baseline_accuracy": metric.get("majority_baseline_accuracy"),
                    "majority_gain": metric.get("majority_gain"),
                    "avg_detection_delay": metric.get("avg_detection_delay"),
                }
            )
    for algo, entry in summary["deep"].items():
        for split_name in ("val", "test"):
            metric = entry[split_name]
            rows.append(
                {
                    "algorithm": algo,
                    "split": split_name,
                    "accuracy": metric.get("accuracy"),
                    "balanced_accuracy": metric.get("balanced_accuracy"),
                    "macro_f1": metric.get("macro_f1"),
                    "avg_return": metric.get("avg_return"),
                    "avg_steps": metric.get("avg_steps"),
                    "declare_step1_ratio": metric.get("declare_step1_ratio"),
                    "majority_baseline_accuracy": metric.get("majority_baseline_accuracy"),
                    "majority_gain": metric.get("majority_gain"),
                    "avg_detection_delay": metric.get("avg_detection_delay"),
                }
            )

    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "algorithm",
                "split",
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "avg_return",
                "avg_steps",
                "declare_step1_ratio",
                "majority_baseline_accuracy",
                "majority_gain",
                "avg_detection_delay",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved summary json: {args.output_json.resolve()}")
    print(f"Saved summary csv: {args.output_csv.resolve()}")


if __name__ == "__main__":
    main()

