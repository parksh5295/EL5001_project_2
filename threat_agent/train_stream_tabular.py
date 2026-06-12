#!/usr/bin/env python3
"""Train tabular baselines (MC/SARSA/Q-learning) on StreamThreatEnv."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from threat_agent.stream_env import StreamEnvConfig, StreamThreatEnv
from threat_agent.stream_eval import StreamEval


def discretize_state(state: np.ndarray, bins: int = 10):
    clipped = np.clip(state, 0.0, 1.0)
    return tuple(np.floor(clipped * bins).astype(np.int32).tolist())


@dataclass
class Cfg:
    gamma: float = 0.99
    alpha: float = 0.1
    epsilon: float = 0.2
    epsilon_min: float = 0.02
    epsilon_decay: float = 0.999
    bins: int = 10
    seed: int = 42


class BaseTab:
    def __init__(self, action_size: int, cfg: Cfg):
        self.action_size = action_size
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self.q = defaultdict(lambda: np.zeros(action_size, dtype=np.float32))

    def skey(self, s):
        return discretize_state(s, self.cfg.bins)

    def egreedy(self, key, eps):
        if self.rng.random() < eps:
            return self.rng.randrange(self.action_size)
        return int(np.argmax(self.q[key]))

    def greedy(self, key):
        return int(np.argmax(self.q[key]))


class MC(BaseTab):
    def __init__(self, action_size, cfg):
        super().__init__(action_size, cfg)
        self.rs = defaultdict(float)
        self.rc = defaultdict(int)

    def train(self, env, episodes: int):
        eps = self.cfg.epsilon
        for _ in range(episodes):
            s, _ = env.reset()
            done = False
            traj = []
            while not done:
                k = self.skey(s)
                a = self.egreedy(k, eps)
                ns, r, t, tr, _ = env.step(a)
                traj.append((k, a, r))
                s = ns
                done = t or tr
            g = 0.0
            vis = set()
            for k, a, r in reversed(traj):
                g = r + self.cfg.gamma * g
                pair = (k, a)
                if pair in vis:
                    continue
                vis.add(pair)
                self.rs[pair] += g
                self.rc[pair] += 1
                self.q[k][a] = self.rs[pair] / max(1, self.rc[pair])
            eps = max(self.cfg.epsilon_min, eps * self.cfg.epsilon_decay)


class SARSA(BaseTab):
    def train(self, env, episodes: int):
        eps = self.cfg.epsilon
        for _ in range(episodes):
            s, _ = env.reset()
            k = self.skey(s)
            a = self.egreedy(k, eps)
            done = False
            while not done:
                ns, r, t, tr, _ = env.step(a)
                done = t or tr
                nk = self.skey(ns)
                if done:
                    target = r
                else:
                    na = self.egreedy(nk, eps)
                    target = r + self.cfg.gamma * self.q[nk][na]
                self.q[k][a] += self.cfg.alpha * (target - self.q[k][a])
                if done:
                    break
                s, k, a = ns, nk, na
            eps = max(self.cfg.epsilon_min, eps * self.cfg.epsilon_decay)


class QLearning(BaseTab):
    def train(self, env, episodes: int):
        eps = self.cfg.epsilon
        for _ in range(episodes):
            s, _ = env.reset()
            done = False
            while not done:
                k = self.skey(s)
                a = self.egreedy(k, eps)
                ns, r, t, tr, _ = env.step(a)
                done = t or tr
                nk = self.skey(ns)
                target = r if done else r + self.cfg.gamma * float(np.max(self.q[nk]))
                self.q[k][a] += self.cfg.alpha * (target - self.q[k][a])
                s = ns
            eps = max(self.cfg.epsilon_min, eps * self.cfg.epsilon_decay)


def evaluate(agent: BaseTab, env: StreamThreatEnv, episodes: int):
    ev = StreamEval(labels=env.labels)
    for _ in range(episodes):
        s, info = env.reset()
        done = False
        ep_return = 0.0
        steps = 0
        pred, true = None, None
        declared_step, delay = None, None
        while not done:
            a = agent.greedy(agent.skey(s))
            s, r, t, tr, info = env.step(a)
            ep_return += r
            steps += 1
            if info.get("declared_label") is not None:
                pred = info["declared_label"]
                true = info["true_label_at_declare"]
                declared_step = info["declared_step"]
                delay = info.get("detection_delay")
            done = t or tr
        if pred is None:
            pred, true = "benign", "benign"
        ev.add(true, pred, steps, declared_step, ep_return, info.get("first_attack_pos"), delay)
    return ev.summary()


def parse_args():
    p = argparse.ArgumentParser(description="Train stream tabular algorithms.")
    p.add_argument("--stream-data", type=Path, default=Path("results/stream_events.ndjson"))
    p.add_argument("--algorithm", default="all", choices=["mc", "sarsa", "qlearning", "all"])
    p.add_argument("--episodes", type=int, default=3000)
    p.add_argument("--eval-episodes", type=int, default=100)
    p.add_argument("--window-size", type=int, default=25)
    p.add_argument("--max-steps", type=int, default=250)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--epsilon", type=float, default=0.2)
    p.add_argument("--epsilon-min", type=float, default=0.02)
    p.add_argument("--epsilon-decay", type=float, default=0.999)
    p.add_argument("--bins", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=Path("results/stream_tabular_metrics.json"))
    return p.parse_args()


def main():
    args = parse_args()
    env_cfg = StreamEnvConfig(window_size=args.window_size, max_steps=args.max_steps, seed=args.seed)
    train_env = StreamThreatEnv(args.stream_data, split="train", config=env_cfg)
    val_env = StreamThreatEnv(args.stream_data, split="val", config=env_cfg)
    test_env = StreamThreatEnv(args.stream_data, split="test", config=env_cfg)
    cfg = Cfg(
        gamma=args.gamma,
        alpha=args.alpha,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        bins=args.bins,
        seed=args.seed,
    )
    algo_map = {"mc": MC, "sarsa": SARSA, "qlearning": QLearning}
    selected = list(algo_map.keys()) if args.algorithm == "all" else [args.algorithm]
    report = {}
    for name in selected:
        ag = algo_map[name](train_env.action_size, cfg)
        ag.train(train_env, args.episodes)
        report[name] = {"val": evaluate(ag, val_env, args.eval_episodes), "test": evaluate(ag, test_env, args.eval_episodes)}
        print(f"[{name}] val={report[name]['val']} test={report[name]['test']}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved metrics: {args.output.resolve()}")


if __name__ == "__main__":
    main()

