#!/usr/bin/env python3
"""Stream RL environment using mixed stream_events.ndjson."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_EVENT_ID_BINS = [
    1,
    3,
    7,
    10,
    11,
    12,
    13,
    14,
    22,
    4624,
    4662,
    4672,
    4688,
    5145,
    5156,
]


@dataclass
class StreamEnvConfig:
    window_size: int = 25
    max_steps: int = 300
    declare_attack_reward: float = 8.0
    declare_benign_reward: float = 4.0
    wrong_tactic_penalty: float = 6.0
    false_alarm_penalty: float = 6.0
    false_negative_penalty: float = 8.0
    wait_cost: float = 0.02
    wait_attack_extra_cost: float = 0.05
    miss_penalty: float = 8.0
    early_attack_bonus_scale: float = 1.5
    event_id_bins: list[int] | None = None
    seed: int | None = None


class StreamThreatEnv:
    """Action space:
    0: WAIT
    1: DECLARE_BENIGN
    2..: DECLARE_ATTACK(tactic_i)
    """

    def __init__(
        self,
        stream_path: str | Path,
        split: str = "train",
        split_ratio: tuple[float, float, float] = (0.7, 0.15, 0.15),
        config: StreamEnvConfig | None = None,
    ):
        self.stream_path = Path(stream_path)
        self.cfg = config or StreamEnvConfig()
        self.rng = random.Random(self.cfg.seed)
        self.np_rng = np.random.default_rng(self.cfg.seed)

        rows = []
        with self.stream_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        if not rows:
            raise ValueError(f"No stream rows in {self.stream_path}")

        # group by stream_id and order by stream_pos
        by_stream: dict[str, list[dict]] = {}
        for r in rows:
            by_stream.setdefault(r["stream_id"], []).append(r)
        all_streams = []
        for sid, seq in by_stream.items():
            seq = sorted(seq, key=lambda x: int(x.get("stream_pos", 0)))
            all_streams.append({"stream_id": sid, "events": seq})
        all_streams = sorted(all_streams, key=lambda x: x["stream_id"])

        # tactics from gt_tactic except benign
        tactic_set = sorted({r.get("gt_tactic") for r in rows if r.get("gt_tactic") and r.get("gt_tactic") != "benign"})
        self.tactics = tactic_set
        self.labels = ["benign"] + self.tactics

        n = len(all_streams)
        if n < 3:
            n_train, n_val = n, 0
        else:
            n_train = max(1, int(n * split_ratio[0]))
            n_val = max(1, int(n * split_ratio[1]))
            # ensure at least one sample for test split
            while n_train + n_val >= n and n_train > 1:
                n_train -= 1
            while n_train + n_val >= n and n_val > 1:
                n_val -= 1
            if n_train + n_val >= n:
                n_train = max(1, n - 2)
                n_val = 1

        if split == "train":
            self.streams = all_streams[:n_train]
        elif split == "val":
            self.streams = all_streams[n_train : n_train + n_val]
        elif split == "test":
            self.streams = all_streams[n_train + n_val :]
        else:
            raise ValueError(f"Unknown split: {split}")
        if not self.streams:
            # fallback for very small datasets: evaluate on full set
            self.streams = all_streams

        self.event_id_bins = self.cfg.event_id_bins or DEFAULT_EVENT_ID_BINS
        # weak ratios(3) + tactic ratios + event histogram + progress(2)
        self.state_size = 3 + len(self.tactics) + len(self.event_id_bins) + 1 + 2
        self.action_size = 2 + len(self.tactics)

        self.current_stream: dict[str, Any] | None = None
        self.stream_events: list[dict] = []
        self.idx = 0
        self.step_count = 0
        self.done = False

    def reset(self, stream: dict | None = None):
        self.current_stream = stream if stream is not None else self.rng.choice(self.streams)
        self.stream_events = self.current_stream["events"]
        self.idx = min(self.cfg.window_size - 1, len(self.stream_events) - 1)
        self.step_count = 0
        self.done = False
        return self._get_state(), self._get_info()

    def _window(self):
        start = max(0, self.idx - self.cfg.window_size + 1)
        return self.stream_events[start : self.idx + 1]

    def _weak_ratio_features(self, win: list[dict]):
        if not win:
            return [0.0, 0.0, 0.0]
        c_attack = sum(1 for e in win if e.get("weak_label") == "attack-like")
        c_benign = sum(1 for e in win if e.get("weak_label") == "benign-like")
        c_unknown = len(win) - c_attack - c_benign
        n = float(len(win))
        return [c_attack / n, c_benign / n, c_unknown / n]

    def _tactic_ratio_features(self, win: list[dict]):
        if not win:
            return [0.0] * len(self.tactics)
        counts = {t: 0 for t in self.tactics}
        total = 0
        for e in win:
            for t in e.get("weak_tactic_candidates", []) or []:
                if t in counts:
                    counts[t] += 1
                    total += 1
        if total == 0:
            return [0.0] * len(self.tactics)
        return [counts[t] / total for t in self.tactics]

    def _event_hist_features(self, win: list[dict]):
        if not win:
            return [0.0] * (len(self.event_id_bins) + 1)
        bins = {eid: i for i, eid in enumerate(self.event_id_bins)}
        arr = np.zeros(len(self.event_id_bins) + 1, dtype=np.float32)
        for e in win:
            eid = e.get("event_id")
            try:
                eid = int(eid)
            except Exception:
                eid = None
            idx = bins.get(eid, len(self.event_id_bins))
            arr[idx] += 1.0
        arr /= max(1, len(win))
        return arr.tolist()

    def _get_state(self):
        win = self._window()
        weak = self._weak_ratio_features(win)
        tact = self._tactic_ratio_features(win)
        hist = self._event_hist_features(win)
        progress = self.idx / max(1, len(self.stream_events) - 1)
        budget_left = max(0, self.cfg.max_steps - self.step_count) / max(1, self.cfg.max_steps)
        return np.array(weak + tact + hist + [progress, budget_left], dtype=np.float32)

    def _current_gt(self):
        e = self.stream_events[self.idx]
        return e.get("gt_attack_active", 0), e.get("gt_tactic", "benign")

    def _first_attack_pos(self):
        for i, e in enumerate(self.stream_events):
            if int(e.get("gt_attack_active", 0)) == 1:
                return i + 1
        return None

    def _advance(self):
        if self.idx < len(self.stream_events) - 1 and self.step_count < self.cfg.max_steps:
            self.idx += 1
            return True
        return False

    def _get_info(self):
        attack_active, gt_tactic = self._current_gt()
        return {
            "stream_id": self.current_stream["stream_id"] if self.current_stream else None,
            "stream_pos": int(self.stream_events[self.idx]["stream_pos"]),
            "attack_active": int(attack_active),
            "gt_tactic": gt_tactic,
            "first_attack_pos": self._first_attack_pos(),
        }

    def step(self, action: int):
        if self.done:
            raise RuntimeError("Episode already ended.")
        if action < 0 or action >= self.action_size:
            raise ValueError(f"Invalid action: {action}")

        self.step_count += 1
        attack_active, gt_tactic = self._current_gt()
        info = self._get_info()
        terminated = False
        truncated = False
        declared_label = None

        if action == 0:  # WAIT
            reward = -self.cfg.wait_cost - (self.cfg.wait_attack_extra_cost if attack_active else 0.0)
            moved = self._advance()
            if not moved:
                # no declaration until end
                if self._first_attack_pos() is None:
                    reward += self.cfg.declare_benign_reward
                else:
                    reward -= self.cfg.miss_penalty
                terminated = True
                truncated = True
        elif action == 1:  # DECLARE_BENIGN
            declared_label = "benign"
            if attack_active:
                reward = -self.cfg.false_negative_penalty
                correct = False
            else:
                reward = self.cfg.declare_benign_reward
                correct = True
            terminated = True
            info["correct"] = correct
        else:  # DECLARE_ATTACK(tactic)
            tactic = self.tactics[action - 2]
            declared_label = tactic
            if not attack_active:
                reward = -self.cfg.false_alarm_penalty
                correct = False
            elif gt_tactic == tactic:
                # reward early correct attack declaration
                first_attack = self._first_attack_pos() or (self.idx + 1)
                delay = max(0, (self.idx + 1) - first_attack)
                reward = self.cfg.declare_attack_reward + self.cfg.early_attack_bonus_scale / (1.0 + delay)
                correct = True
                info["detection_delay"] = delay
            else:
                reward = -self.cfg.wrong_tactic_penalty
                correct = False
            terminated = True
            info["correct"] = correct

        if self.step_count >= self.cfg.max_steps and not terminated:
            terminated = True
            truncated = True
            reward = reward - self.cfg.miss_penalty if self._first_attack_pos() is not None else reward

        self.done = terminated
        if declared_label is not None:
            info["declared_label"] = declared_label
            info["declared_step"] = self.step_count
            info["true_label_at_declare"] = gt_tactic if attack_active else "benign"

        return self._get_state(), float(reward), terminated, truncated, info

