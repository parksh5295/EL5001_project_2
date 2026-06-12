#!/usr/bin/env python3
"""Baselines for StreamThreatEnv."""

from __future__ import annotations

import argparse
import random

import numpy as np

from threat_agent.stream_env import StreamEnvConfig, StreamThreatEnv
from threat_agent.stream_eval import StreamEval


def random_policy(env: StreamThreatEnv, info: dict):
    return random.randrange(env.action_size)


def heuristic_policy(env: StreamThreatEnv, state: np.ndarray, threshold: float = 0.25):
    # state layout: weak ratios [attack, benign, unknown], ...
    attack_ratio = state[0]
    benign_ratio = state[1]
    if attack_ratio >= threshold:
        # declare first attack tactic as heuristic
        return 2
    if benign_ratio >= 0.8:
        return 1
    return 0


def rollout(env: StreamThreatEnv, episodes: int, policy: str):
    ev = StreamEval(labels=env.labels)
    for _ in range(episodes):
        state, info = env.reset()
        done = False
        ep_return = 0.0
        steps = 0
        pred_label = None
        true_label = None
        declared_step = None
        detection_delay = None
        first_attack = info.get("first_attack_pos")
        while not done:
            if policy == "random":
                action = random_policy(env, info)
            elif policy == "heuristic":
                action = heuristic_policy(env, state)
            else:
                raise ValueError(policy)
            state, r, terminated, truncated, info = env.step(action)
            ep_return += r
            steps += 1
            if info.get("declared_label") is not None:
                pred_label = info.get("declared_label")
                true_label = info.get("true_label_at_declare")
                declared_step = info.get("declared_step")
                detection_delay = info.get("detection_delay")
            done = terminated or truncated

        if pred_label is None:
            pred_label = "benign"
            true_label = "benign"
        ev.add(
            true_label=true_label,
            pred_label=pred_label,
            steps=steps,
            declared_step=declared_step,
            episode_return=ep_return,
            first_attack_pos=first_attack,
            detection_delay=detection_delay,
        )
    return ev.summary()


def parse_args():
    p = argparse.ArgumentParser(description="Run stream baselines.")
    p.add_argument("--stream-data", default="results/stream_events.ndjson")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--policy", default="both", choices=["random", "heuristic", "both"])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = StreamEnvConfig(seed=args.seed)
    env = StreamThreatEnv(args.stream_data, split=args.split, config=cfg)
    policies = ["random", "heuristic"] if args.policy == "both" else [args.policy]
    for p in policies:
        print(f"[{p}] {rollout(env, args.episodes, p)}")


if __name__ == "__main__":
    main()

