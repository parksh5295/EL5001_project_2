#!/usr/bin/env python3
"""Tabular control baselines for Threat Investigation Agent.

Algorithms:
- Monte Carlo Control (first-visit)
- SARSA
- Q-learning
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

import numpy as np


def discretize_state(state: np.ndarray, bins: int = 10) -> tuple[int, ...]:
    """Discretize continuous state vector into a hashable tuple."""
    clipped = np.clip(state, 0.0, 1.0)
    return tuple(np.floor(clipped * bins).astype(np.int32).tolist())


@dataclass
class TabularConfig:
    gamma: float = 0.99
    alpha: float = 0.1
    epsilon: float = 0.1
    epsilon_min: float = 0.02
    epsilon_decay: float = 0.999
    bins: int = 10
    seed: int = 42


class BaseTabularAgent:
    def __init__(self, action_size: int, config: TabularConfig):
        self.action_size = action_size
        self.cfg = config
        self.rng = random.Random(config.seed)
        self.q: dict[tuple[int, ...], np.ndarray] = defaultdict(lambda: np.zeros(self.action_size, dtype=np.float32))

    def state_key(self, state: np.ndarray) -> tuple[int, ...]:
        return discretize_state(state, bins=self.cfg.bins)

    def epsilon_greedy(self, key: tuple[int, ...], mask: np.ndarray, epsilon: float) -> int:
        valid = np.where(mask > 0.0)[0]
        if len(valid) == 0:
            return 0
        if self.rng.random() < epsilon:
            return int(self.rng.choice(valid.tolist()))
        q = self.q[key].copy()
        q[mask <= 0.0] = -1e9
        return int(np.argmax(q))

    def greedy(self, key: tuple[int, ...], mask: np.ndarray) -> int:
        return self.epsilon_greedy(key, mask, epsilon=0.0)


class MonteCarloControlAgent(BaseTabularAgent):
    def __init__(self, action_size: int, config: TabularConfig):
        super().__init__(action_size, config)
        self.returns_sum: dict[tuple[tuple[int, ...], int], float] = defaultdict(float)
        self.returns_count: dict[tuple[tuple[int, ...], int], int] = defaultdict(int)

    def train(self, env, episodes: int):
        epsilon = self.cfg.epsilon
        for _ in range(episodes):
            state, info = env.reset()
            done = False
            trajectory: list[tuple[tuple[int, ...], int, float]] = []
            while not done:
                mask = np.array(info["action_mask"], dtype=np.float32)
                key = self.state_key(state)
                action = self.epsilon_greedy(key, mask, epsilon)
                nstate, reward, terminated, truncated, ninfo = env.step(action)
                trajectory.append((key, action, reward))
                state, info = nstate, ninfo
                done = terminated or truncated

            # first-visit MC update
            g = 0.0
            visited = set()
            for key, action, reward in reversed(trajectory):
                g = reward + self.cfg.gamma * g
                pair = (key, action)
                if pair in visited:
                    continue
                visited.add(pair)
                self.returns_sum[pair] += g
                self.returns_count[pair] += 1
                self.q[key][action] = self.returns_sum[pair] / max(1, self.returns_count[pair])

            epsilon = max(self.cfg.epsilon_min, epsilon * self.cfg.epsilon_decay)


class SARSAAgent(BaseTabularAgent):
    def train(self, env, episodes: int):
        epsilon = self.cfg.epsilon
        for _ in range(episodes):
            state, info = env.reset()
            mask = np.array(info["action_mask"], dtype=np.float32)
            key = self.state_key(state)
            action = self.epsilon_greedy(key, mask, epsilon)
            done = False

            while not done:
                nstate, reward, terminated, truncated, ninfo = env.step(action)
                done = terminated or truncated
                nkey = self.state_key(nstate)
                nmask = np.array(ninfo["action_mask"], dtype=np.float32)

                if done:
                    target = reward
                else:
                    naction = self.epsilon_greedy(nkey, nmask, epsilon)
                    target = reward + self.cfg.gamma * self.q[nkey][naction]
                self.q[key][action] += self.cfg.alpha * (target - self.q[key][action])

                if done:
                    break
                state, info, key, action = nstate, ninfo, nkey, naction

            epsilon = max(self.cfg.epsilon_min, epsilon * self.cfg.epsilon_decay)


class QLearningAgent(BaseTabularAgent):
    def train(self, env, episodes: int):
        epsilon = self.cfg.epsilon
        for _ in range(episodes):
            state, info = env.reset()
            done = False

            while not done:
                key = self.state_key(state)
                mask = np.array(info["action_mask"], dtype=np.float32)
                action = self.epsilon_greedy(key, mask, epsilon)
                nstate, reward, terminated, truncated, ninfo = env.step(action)
                done = terminated or truncated

                nkey = self.state_key(nstate)
                nmask = np.array(ninfo["action_mask"], dtype=np.float32)
                if done:
                    target = reward
                else:
                    nq = self.q[nkey].copy()
                    nq[nmask <= 0.0] = -1e9
                    target = reward + self.cfg.gamma * float(np.max(nq))
                self.q[key][action] += self.cfg.alpha * (target - self.q[key][action])

                state, info = nstate, ninfo

            epsilon = max(self.cfg.epsilon_min, epsilon * self.cfg.epsilon_decay)


def evaluate_tabular(agent: BaseTabularAgent, env, episodes: int):
    correct = 0
    total_return = 0.0
    total_steps = 0
    for _ in range(episodes):
        state, info = env.reset()
        done = False
        while not done:
            key = agent.state_key(state)
            mask = np.array(info["action_mask"], dtype=np.float32)
            action = agent.greedy(key, mask)
            state, reward, terminated, truncated, info = env.step(action)
            total_return += reward
            total_steps += 1
            done = terminated or truncated
        if info.get("correct"):
            correct += 1
    return {
        "accuracy": correct / episodes if episodes else 0.0,
        "avg_return": total_return / episodes if episodes else 0.0,
        "avg_steps": total_steps / episodes if episodes else 0.0,
    }

