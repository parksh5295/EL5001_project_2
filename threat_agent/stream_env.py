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
    event_f1_reward_scale: float = 5.0
    boundary_bonus: float = 2.0
    boundary_tolerance: int = 1
    invalid_op_penalty: float = 1.5
    wait_cost: float = 0.02
    event_id_bins: list[int] | None = None
    seed: int | None = None


class StreamThreatEnv:
    """Action space:
    0: WAIT
    1..N: START_<tactic_i>
    N+1..2N: END_<tactic_i>
    """

    def __init__(
        self,
        stream_path: str | Path,
        split: str = "train",
        split_ratio: tuple[float, float, float] = (0.7, 0.15, 0.15),
        config: StreamEnvConfig | None = None,
        tactics: list[str] | None = None,
    ):
        self.stream_path = Path(stream_path)
        self.cfg = config or StreamEnvConfig()
        self.rng = random.Random(self.cfg.seed)
        self.np_rng = np.random.default_rng(self.cfg.seed)
        valid_splits = {"train", "val", "test"}
        has_any_row = False
        has_predefined_split = True
        tactic_set: set[str] = set()

        # Pass 1: detect split tagging and collect global tactic label space
        with self.stream_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                has_any_row = True
                row = json.loads(line)
                row_split = row.get("dataset_split")
                if row_split not in valid_splits:
                    has_predefined_split = False
                gt_tactic = row.get("gt_tactic")
                if gt_tactic and gt_tactic != "benign":
                    tactic_set.add(gt_tactic)

        if not has_any_row:
            raise ValueError(f"No stream rows in {self.stream_path}")

        # Pass 2: build per-stream byte range index so we can lazy-load stream events on reset().
        all_streams: list[dict[str, Any]] = []
        seen_stream_ids: set[str] = set()
        current_sid: str | None = None
        current_split: str | None = None
        current_start = 0
        non_contiguous = False
        with self.stream_path.open("rb") as f:
            while True:
                line_start = f.tell()
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line.decode("utf-8"))
                if has_predefined_split and row.get("dataset_split") != split:
                    continue
                sid = str(row["stream_id"])
                row_split = row.get("dataset_split")

                if current_sid is None:
                    current_sid = sid
                    current_split = row_split
                    current_start = line_start
                    if sid in seen_stream_ids:
                        non_contiguous = True
                    seen_stream_ids.add(sid)
                    continue

                if sid != current_sid:
                    all_streams.append(
                        {
                            "stream_id": current_sid,
                            "dataset_split": current_split,
                            "byte_start": current_start,
                            "byte_end": line_start,
                        }
                    )
                    current_sid = sid
                    current_split = row_split
                    current_start = line_start
                    if sid in seen_stream_ids:
                        non_contiguous = True
                    seen_stream_ids.add(sid)

            if current_sid is not None:
                all_streams.append(
                    {
                        "stream_id": current_sid,
                        "dataset_split": current_split,
                        "byte_start": current_start,
                        "byte_end": f.tell(),
                    }
                )

        if non_contiguous:
            # Fallback for legacy datasets where a stream_id appears in multiple distant chunks.
            by_stream: dict[str, list[dict]] = {}
            with self.stream_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    if has_predefined_split and row.get("dataset_split") != split:
                        continue
                    by_stream.setdefault(row["stream_id"], []).append(row)
            all_streams = []
            for sid, seq in by_stream.items():
                seq = sorted(seq, key=lambda x: int(x.get("stream_pos", 0)))
                all_streams.append({"stream_id": sid, "dataset_split": seq[0].get("dataset_split"), "events": seq})

        all_streams = sorted(all_streams, key=lambda x: x["stream_id"])

        self.tactics = sorted(tactics) if tactics is not None else sorted(tactic_set)
        self.labels = ["benign"] + self.tactics
        if has_predefined_split:
            if split not in {"train", "val", "test"}:
                raise ValueError(f"Unknown split: {split}")
            # all_streams are already filtered by split in pass 2.
            self.streams = all_streams
            if not self.streams:
                raise ValueError(f"No streams found for split='{split}' in dataset_split-tagged data.")
        else:
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
        self.action_size = 1 + 2 * len(self.tactics)
        self.action_to_op: dict[int, tuple[str, str | None]] = {0: ("wait", None)}
        for i, tactic in enumerate(self.tactics):
            self.action_to_op[1 + i] = ("start", tactic)
            self.action_to_op[1 + len(self.tactics) + i] = ("end", tactic)

        self.current_stream: dict[str, Any] | None = None
        self.stream_events: list[dict] = []
        self._cached_stream_id: str | None = None
        self._cached_events: list[dict] | None = None
        self.active_tactics: set[str] = set()
        self._gt_tactic_sets: list[set[str]] = []
        self._gt_start_points: dict[str, set[int]] = {}
        self._gt_end_points: dict[str, set[int]] = {}
        self._pred_trace: list[list[str]] = []
        self._gt_trace: list[list[str]] = []
        self._boundary_tp = 0
        self._boundary_fp = 0
        self._boundary_fn = 0
        self.idx = 0
        self.step_count = 0
        self.done = False

    def _load_stream_events(self, stream: dict[str, Any]) -> list[dict]:
        if "events" in stream:
            return stream["events"]
        sid = stream["stream_id"]
        if sid == self._cached_stream_id and self._cached_events is not None:
            return self._cached_events

        events: list[dict] = []
        with self.stream_path.open("rb") as f:
            f.seek(int(stream["byte_start"]))
            end = int(stream["byte_end"])
            while f.tell() < end:
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                events.append(json.loads(line.decode("utf-8")))
        events = sorted(events, key=lambda x: int(x.get("stream_pos", 0)))
        self._cached_stream_id = sid
        self._cached_events = events
        return events

    def reset(self, stream: dict | None = None):
        self.current_stream = stream if stream is not None else self.rng.choice(self.streams)
        self.stream_events = self._load_stream_events(self.current_stream)
        self.idx = min(self.cfg.window_size - 1, len(self.stream_events) - 1)
        self._gt_tactic_sets = []
        for e in self.stream_events:
            if int(e.get("gt_attack_active", 0)) == 1:
                gt_tactic = e.get("gt_tactic")
                if gt_tactic and gt_tactic in self.tactics:
                    self._gt_tactic_sets.append({gt_tactic})
                else:
                    self._gt_tactic_sets.append(set())
            else:
                self._gt_tactic_sets.append(set())
        self._gt_start_points = {t: set() for t in self.tactics}
        self._gt_end_points = {t: set() for t in self.tactics}
        prev_set: set[str] = set()
        for i, cur_set in enumerate(self._gt_tactic_sets):
            for t in cur_set - prev_set:
                self._gt_start_points[t].add(i)
            for t in prev_set - cur_set:
                self._gt_end_points[t].add(i)
            prev_set = set(cur_set)
        for t in prev_set:
            self._gt_end_points[t].add(len(self._gt_tactic_sets))
        self.active_tactics = set()
        self._pred_trace = []
        self._gt_trace = []
        self._boundary_tp = 0
        self._boundary_fp = 0
        self._boundary_fn = 0
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

    def _current_gt_set(self) -> set[str]:
        if not self._gt_tactic_sets:
            return set()
        return set(self._gt_tactic_sets[self.idx])

    def _first_attack_pos(self):
        for i, e in enumerate(self.stream_events):
            if int(e.get("gt_attack_active", 0)) == 1:
                return i + 1
        return None

    @staticmethod
    def _f1_for_sets(pred: set[str], gt: set[str]) -> float:
        if not pred and not gt:
            return 1.0
        tp = len(pred & gt)
        prec = tp / len(pred) if pred else 0.0
        rec = tp / len(gt) if gt else 0.0
        if prec + rec == 0:
            return 0.0
        return (2.0 * prec * rec) / (prec + rec)

    def _is_near_boundary(self, points: set[int]) -> bool:
        tol = max(0, int(self.cfg.boundary_tolerance))
        return any(abs(self.idx - p) <= tol for p in points)

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
            "active_tactics": sorted(self.active_tactics),
            "gt_active_tactics": sorted(self._current_gt_set()),
        }

    def step(self, action: int):
        if self.done:
            raise RuntimeError("Episode already ended.")
        if action < 0 or action >= self.action_size:
            raise ValueError(f"Invalid action: {action}")

        self.step_count += 1
        info = self._get_info()
        terminated = False
        truncated = False
        reward = 0.0
        op, tactic = self.action_to_op[action]
        boundary_hit = False
        invalid_op = False

        if op == "wait":
            reward -= self.cfg.wait_cost
        elif op == "start":
            if tactic is None or tactic in self.active_tactics:
                invalid_op = True
                reward -= self.cfg.invalid_op_penalty
            else:
                self.active_tactics.add(tactic)
                if self._is_near_boundary(self._gt_start_points.get(tactic, set())):
                    boundary_hit = True
                    reward += self.cfg.boundary_bonus
                    self._boundary_tp += 1
                else:
                    self._boundary_fp += 1
        elif op == "end":
            if tactic is None or tactic not in self.active_tactics:
                invalid_op = True
                reward -= self.cfg.invalid_op_penalty
            else:
                self.active_tactics.remove(tactic)
                if self._is_near_boundary(self._gt_end_points.get(tactic, set())):
                    boundary_hit = True
                    reward += self.cfg.boundary_bonus
                    self._boundary_tp += 1
                else:
                    self._boundary_fp += 1

        gt_set = self._current_gt_set()
        pred_set = set(self.active_tactics)
        step_f1 = self._f1_for_sets(pred_set, gt_set)
        reward += self.cfg.event_f1_reward_scale * step_f1

        self._pred_trace.append(sorted(pred_set))
        self._gt_trace.append(sorted(gt_set))

        moved = self._advance()
        if not moved:
            terminated = True
            truncated = True

        if self.step_count >= self.cfg.max_steps and not terminated:
            terminated = True
            truncated = True

        self.done = terminated
        info.update(
            {
                "action_name": op if tactic is None else f"{op}:{tactic}",
                "boundary_hit": boundary_hit,
                "invalid_op": invalid_op,
                "pred_active_tactics": sorted(pred_set),
                "gt_active_tactics": sorted(gt_set),
                "step_event_f1": step_f1,
            }
        )
        if terminated:
            self._boundary_fn = 0
            for t in self.tactics:
                expected = len(self._gt_start_points.get(t, set())) + len(self._gt_end_points.get(t, set()))
                matched = min(expected, self._boundary_tp)
                # conservative residual estimate
                self._boundary_fn += max(0, expected - matched)
            info["episode_pred_trace"] = self._pred_trace
            info["episode_gt_trace"] = self._gt_trace
            info["boundary_tp"] = self._boundary_tp
            info["boundary_fp"] = self._boundary_fp
            info["boundary_fn"] = self._boundary_fn

        return self._get_state(), float(reward), terminated, truncated, info

