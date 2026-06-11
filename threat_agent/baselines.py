#!/usr/bin/env python3
"""Baselines for Threat Investigation Agent."""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import numpy as np

from threat_agent.env import EnvConfig, ThreatInvestigationEnv


def random_valid_action(mask: np.ndarray, rng: np.random.Generator) -> int:
    valid = np.where(mask > 0.0)[0]
    return int(rng.choice(valid))


def heuristic_action(env: ThreatInvestigationEnv, mask: np.ndarray) -> int:
    """Simple domain heuristic:
    network -> process -> registry -> user, then declare most frequent class prior.
    """
    investigate_priority = [2, 0, 1, 3]
    for act in investigate_priority:
        if mask[act] > 0.0:
            return act
    # fallback declare first tactic index
    return 4


def rollout(
    env: ThreatInvestigationEnv,
    policy: str,
    episodes: int,
    seed: int,
):
    rng = np.random.default_rng(seed)
    returns = []
    correct = 0
    steps = []

    for _ in range(episodes):
        _, info = env.reset()
        done = False
        ep_return = 0.0
        ep_steps = 0
        last_info = info
        while not done:
            mask = np.array(last_info["action_mask"], dtype=np.float32)
            if policy == "random":
                action = random_valid_action(mask, rng)
            elif policy == "heuristic":
                action = heuristic_action(env, mask)
            else:
                raise ValueError(policy)

            _, reward, terminated, truncated, last_info = env.step(action)
            ep_return += reward
            ep_steps += 1
            done = terminated or truncated

        returns.append(ep_return)
        steps.append(ep_steps)
        if last_info.get("correct"):
            correct += 1

    acc = correct / episodes if episodes else 0.0
    return {
        "episodes": episodes,
        "accuracy": acc,
        "avg_return": statistics.mean(returns) if returns else 0.0,
        "avg_steps": statistics.mean(steps) if steps else 0.0,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run baseline policies for Threat Investigation Agent.")
    parser.add_argument("--dataset", type=Path, default=Path("results/threat_agent_data.json"))
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--policy", type=str, default="both", choices=["random", "heuristic", "both"])
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = EnvConfig(seed=args.seed)
    env = ThreatInvestigationEnv(args.dataset, split=args.split, config=cfg)

    policies = ["random", "heuristic"] if args.policy == "both" else [args.policy]
    for p in policies:
        result = rollout(env, policy=p, episodes=args.episodes, seed=args.seed)
        print(f"[{p}] {result}")


if __name__ == "__main__":
    main()
