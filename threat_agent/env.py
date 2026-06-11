#!/usr/bin/env python3
"""Threat Investigation Agent environment (Gym-style API without hard dependency)."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

CATEGORY_ORDER = ("process", "registry", "network", "user")


@dataclass
class EnvConfig:
    max_steps: int = 12
    investigate_cost: float = 0.5
    invalid_action_penalty: float = 0.25
    correct_declare_reward: float = 10.0
    wrong_declare_penalty: float = 10.0
    reveal_success_prob: float = 1.0
    early_bonus_scale: float = 0.0
    seed: int | None = None


class ThreatInvestigationEnv:
    """Minimal environment for Threat Investigation MDP.

    Action space:
      0..3                      -> Investigate category (process/registry/network/user)
      4..(4 + num_tactics - 1)  -> Declare tactic
    """

    def __init__(
        self,
        dataset_path: str | Path,
        split: str = "train",
        split_ratio: tuple[float, float, float] = (0.7, 0.15, 0.15),
        config: EnvConfig | None = None,
    ):
        self.dataset_path = Path(dataset_path)
        self.config = config or EnvConfig()
        self.rng = random.Random(self.config.seed)
        self.np_rng = np.random.default_rng(self.config.seed)

        data = json.loads(self.dataset_path.read_text(encoding="utf-8"))
        self.tactics: list[str] = data["tactics"]
        self.event_id_bins: list[int] = data.get("event_id_bins", [])
        episodes = data["episodes"]

        # deterministic split by sorted source_file for reproducibility
        episodes = sorted(episodes, key=lambda x: x["source_file"])
        n = len(episodes)
        n_train = int(n * split_ratio[0])
        n_val = int(n * split_ratio[1])
        if split == "train":
            self.episodes = episodes[:n_train]
        elif split == "val":
            self.episodes = episodes[n_train : n_train + n_val]
        elif split == "test":
            self.episodes = episodes[n_train + n_val :]
        else:
            raise ValueError(f"Unknown split: {split}")
        if not self.episodes:
            raise ValueError(f"No episodes in split '{split}'.")

        self.action_size = 4 + len(self.tactics)
        self.state_size = 8 + 2 + len(self.event_id_bins) + 1

        self.current_episode: dict[str, Any] | None = None
        self.reset_state()

    def reset_state(self):
        self.step_count = 0
        self.done = False
        self.revealed_index = {c: 0 for c in CATEGORY_ORDER}
        self.revealed_event_ids: list[int] = []
        self.last_reward = 0.0
        self.total_reward = 0.0
        self.investigate_count = 0

    def sample_episode(self) -> dict[str, Any]:
        return self.rng.choice(self.episodes)

    def reset(self, episode: dict[str, Any] | None = None):
        self.current_episode = episode if episode is not None else self.sample_episode()
        self.reset_state()
        return self._get_state(), self._get_info()

    def _card_totals(self) -> dict[str, int]:
        assert self.current_episode is not None
        return {c: len(self.current_episode["cards"][c]) for c in CATEGORY_ORDER}

    def get_action_mask(self) -> np.ndarray:
        """1 for valid actions, 0 for invalid."""
        totals = self._card_totals()
        mask = np.ones(self.action_size, dtype=np.float32)

        # Investigate actions: valid only if more cards remain
        for idx, cat in enumerate(CATEGORY_ORDER):
            if self.revealed_index[cat] >= totals[cat]:
                mask[idx] = 0.0

        # Declare actions always valid
        return mask

    def _event_histogram(self) -> np.ndarray:
        hist = np.zeros(len(self.event_id_bins) + 1, dtype=np.float32)
        if not self.revealed_event_ids:
            return hist
        id_to_idx = {eid: i for i, eid in enumerate(self.event_id_bins)}
        for eid in self.revealed_event_ids:
            idx = id_to_idx.get(eid, len(self.event_id_bins))
            hist[idx] += 1.0
        hist /= max(1, len(self.revealed_event_ids))
        return hist

    def _get_state(self) -> np.ndarray:
        totals = self._card_totals()
        base = []
        for c in CATEGORY_ORDER:
            revealed = self.revealed_index[c]
            total = totals[c]
            ratio = revealed / max(1, total)
            availability = 1.0 if total > 0 else 0.0
            base.extend([ratio, availability])

        progress = self.step_count / max(1, self.config.max_steps)
        budget_left = max(0, self.config.max_steps - self.step_count) / max(1, self.config.max_steps)

        state = np.concatenate(
            [
                np.array(base + [progress, budget_left], dtype=np.float32),
                self._event_histogram(),
            ]
        )
        return state

    def _get_info(self) -> dict[str, Any]:
        assert self.current_episode is not None
        return {
            "source_file": self.current_episode["source_file"],
            "tactic": self.current_episode["tactic"],
            "step_count": self.step_count,
            "investigate_count": self.investigate_count,
            "revealed_index": dict(self.revealed_index),
            "action_mask": self.get_action_mask().tolist(),
            "total_reward": self.total_reward,
        }

    def _investigate(self, category: str):
        assert self.current_episode is not None
        totals = self._card_totals()
        if self.revealed_index[category] >= totals[category]:
            reward = -self.config.invalid_action_penalty
            return reward, False, {"invalid_action": True, "category": category}

        reward = -self.config.investigate_cost
        if self.np_rng.random() <= self.config.reveal_success_prob:
            card = self.current_episode["cards"][category][self.revealed_index[category]]
            self.revealed_index[category] += 1
            if isinstance(card.get("event_id"), int):
                self.revealed_event_ids.append(card["event_id"])
            info = {"revealed_card": card, "category": category}
        else:
            info = {"reveal_failed": True, "category": category}

        self.investigate_count += 1
        return reward, False, info

    def _declare(self, tactic_idx: int):
        assert self.current_episode is not None
        declared = self.tactics[tactic_idx]
        correct = declared == self.current_episode["tactic"]
        reward = self.config.correct_declare_reward if correct else -self.config.wrong_declare_penalty
        if correct and self.config.early_bonus_scale > 0:
            remaining_ratio = max(0.0, (self.config.max_steps - self.step_count) / max(1, self.config.max_steps))
            reward += self.config.early_bonus_scale * remaining_ratio
        return reward, True, {"declared_tactic": declared, "correct": correct}

    def step(self, action: int):
        if self.done:
            raise RuntimeError("Episode already ended. Call reset().")
        if action < 0 or action >= self.action_size:
            raise ValueError(f"Invalid action: {action}")

        self.step_count += 1
        terminated = False
        info: dict[str, Any]

        if action <= 3:
            category = CATEGORY_ORDER[action]
            reward, terminated, info = self._investigate(category)
        else:
            tactic_idx = action - 4
            reward, terminated, info = self._declare(tactic_idx)

        truncated = False
        if not terminated and self.step_count >= self.config.max_steps:
            # force end if no declaration
            reward -= self.config.wrong_declare_penalty
            terminated = True
            truncated = True
            info["forced_termination"] = True

        self.last_reward = reward
        self.total_reward += reward
        self.done = terminated

        next_state = self._get_state()
        full_info = self._get_info()
        full_info.update(info)
        return next_state, reward, terminated, truncated, full_info
