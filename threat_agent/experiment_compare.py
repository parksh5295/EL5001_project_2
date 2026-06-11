#!/usr/bin/env python3
"""Run multi-algorithm comparison experiments for Threat Investigation Agent."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def run_command(command: list[str]):
    print(" ".join(command))
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(command)}")


def parse_args():
    p = argparse.ArgumentParser(description="Run comparison experiments across multiple algorithms.")
    p.add_argument("--dataset", type=Path, default=Path("results/threat_agent_data.json"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tabular-episodes", type=int, default=2000)
    p.add_argument("--deep-episodes", type=int, default=1000)
    p.add_argument("--eval-episodes", type=int, default=100)
    p.add_argument("--output-csv", type=Path, default=Path("results/compare_summary.csv"))
    p.add_argument("--output-json", type=Path, default=Path("results/compare_summary.json"))
    return p.parse_args()


def main():
    args = parse_args()
    py = sys.executable
    root = Path(__file__).resolve().parents[1]

    # 1) tabular baselines
    tab_path = root / "results" / "tabular_metrics.json"
    run_command(
        [
            py,
            "-m",
            "threat_agent.train_tabular",
            "--dataset",
            str(args.dataset),
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

    dqn_metric_path = root / "results" / "dqn_metrics.json"
    reinforce_metric_path = root / "results" / "reinforce_metrics.json"
    a2c_metric_path = root / "results" / "a2c_metrics.json"

    # 2) deep value-based
    run_command(
        [
            py,
            "-m",
            "threat_agent.train_dqn",
            "--dataset",
            str(args.dataset),
            "--episodes",
            str(args.deep_episodes),
            "--eval-episodes",
            str(args.eval_episodes),
            "--seed",
            str(args.seed),
            "--save-model",
            str(root / "checkpoints" / "threat_agent_dqn.pt"),
            "--metrics-output",
            str(dqn_metric_path),
        ]
    )

    # 3) deep policy-based
    run_command(
        [
            py,
            "-m",
            "threat_agent.train_reinforce",
            "--dataset",
            str(args.dataset),
            "--episodes",
            str(args.deep_episodes),
            "--eval-episodes",
            str(args.eval_episodes),
            "--seed",
            str(args.seed),
            "--save-model",
            str(root / "checkpoints" / "threat_agent_reinforce.pt"),
            "--metrics-output",
            str(reinforce_metric_path),
        ]
    )
    run_command(
        [
            py,
            "-m",
            "threat_agent.train_a2c",
            "--dataset",
            str(args.dataset),
            "--episodes",
            str(args.deep_episodes),
            "--eval-episodes",
            str(args.eval_episodes),
            "--seed",
            str(args.seed),
            "--save-model",
            str(root / "checkpoints" / "threat_agent_a2c.pt"),
            "--metrics-output",
            str(a2c_metric_path),
        ]
    )

    summary = {
        "tabular": json.loads(tab_path.read_text(encoding="utf-8")),
        "deep": {
            "dqn": json.loads(dqn_metric_path.read_text(encoding="utf-8")),
            "reinforce": json.loads(reinforce_metric_path.read_text(encoding="utf-8")),
            "a2c": json.loads(a2c_metric_path.read_text(encoding="utf-8")),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    rows = []
    for algo, entry in summary["deep"].items():
        for split_name in ("val", "test"):
            metric = entry[split_name]
            rows.append(
                {
                    "algorithm": algo,
                    "split": split_name,
                    "accuracy": metric["accuracy"],
                    "avg_return": metric["avg_return"],
                    "avg_steps": metric["avg_steps"],
                }
            )
    for algo, splits in summary["tabular"].items():
        for split_name, metric in splits.items():
            rows.append(
                {
                    "algorithm": algo,
                    "split": split_name,
                    "accuracy": metric["accuracy"],
                    "avg_return": metric["avg_return"],
                    "avg_steps": metric["avg_steps"],
                }
            )
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["algorithm", "split", "accuracy", "avg_return", "avg_steps"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved summary json: {args.output_json.resolve()}")
    print(f"Saved summary csv: {args.output_csv.resolve()}")


if __name__ == "__main__":
    main()

