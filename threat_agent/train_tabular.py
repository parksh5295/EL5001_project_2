#!/usr/bin/env python3
"""Train tabular RL baselines (MC/SARSA/Q-learning) on Threat Investigation Agent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from threat_agent.env import EnvConfig, ThreatInvestigationEnv
from threat_agent.tabular_agents import (
    MonteCarloControlAgent,
    QLearningAgent,
    SARSAAgent,
    TabularConfig,
    evaluate_tabular,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train tabular baselines for Threat Investigation Agent.")
    p.add_argument("--dataset", type=Path, default=Path("results/threat_agent_data.json"))
    p.add_argument("--algorithm", type=str, default="all", choices=["mc", "sarsa", "qlearning", "all"])
    p.add_argument("--episodes", type=int, default=3000)
    p.add_argument("--eval-episodes", type=int, default=200)
    p.add_argument("--max-steps", type=int, default=12)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--epsilon", type=float, default=0.2)
    p.add_argument("--epsilon-min", type=float, default=0.02)
    p.add_argument("--epsilon-decay", type=float, default=0.999)
    p.add_argument("--bins", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=Path("results/tabular_metrics.json"))
    return p.parse_args()


def main():
    args = parse_args()
    env_cfg = EnvConfig(
        max_steps=args.max_steps,
        investigate_cost=0.5,
        invalid_action_penalty=0.25,
        correct_declare_reward=10.0,
        wrong_declare_penalty=10.0,
        reveal_success_prob=0.9,
        seed=args.seed,
    )
    train_env = ThreatInvestigationEnv(args.dataset, split="train", config=env_cfg)
    val_env = ThreatInvestigationEnv(args.dataset, split="val", config=env_cfg)
    test_env = ThreatInvestigationEnv(args.dataset, split="test", config=env_cfg)
    tab_cfg = TabularConfig(
        gamma=args.gamma,
        alpha=args.alpha,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        bins=args.bins,
        seed=args.seed,
    )

    algo_map = {
        "mc": MonteCarloControlAgent,
        "sarsa": SARSAAgent,
        "qlearning": QLearningAgent,
    }
    selected = list(algo_map.keys()) if args.algorithm == "all" else [args.algorithm]
    report = {}
    for name in selected:
        agent = algo_map[name](train_env.action_size, tab_cfg)
        agent.train(train_env, episodes=args.episodes)
        report[name] = {
            "val": evaluate_tabular(agent, val_env, args.eval_episodes),
            "test": evaluate_tabular(agent, test_env, args.eval_episodes),
        }
        print(f"[{name}] val={report[name]['val']} test={report[name]['test']}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved metrics: {args.output.resolve()}")


if __name__ == "__main__":
    main()

