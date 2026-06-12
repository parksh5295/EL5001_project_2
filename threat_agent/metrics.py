#!/usr/bin/env python3
"""Evaluation metrics for Threat Investigation Agent experiments."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


@dataclass
class EpisodeRecord:
    true_label: str
    pred_label: str | None
    steps: int
    declared_step: int | None
    episode_return: float


@dataclass
class EvalAccumulator:
    labels: list[str]
    records: list[EpisodeRecord] = field(default_factory=list)

    def add(
        self,
        true_label: str,
        pred_label: str | None,
        steps: int,
        declared_step: int | None,
        episode_return: float,
    ):
        self.records.append(
            EpisodeRecord(
                true_label=true_label,
                pred_label=pred_label,
                steps=steps,
                declared_step=declared_step,
                episode_return=episode_return,
            )
        )

    def _per_class_counts(self):
        tp = Counter({k: 0 for k in self.labels})
        fp = Counter({k: 0 for k in self.labels})
        fn = Counter({k: 0 for k in self.labels})

        for rec in self.records:
            y = rec.true_label
            p = rec.pred_label
            for c in self.labels:
                if y == c and p == c:
                    tp[c] += 1
                elif y != c and p == c:
                    fp[c] += 1
                elif y == c and p != c:
                    fn[c] += 1
        return tp, fp, fn

    def summary(self) -> dict:
        n = len(self.records)
        if n == 0:
            return {
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "macro_f1": 0.0,
                "per_class_recall": {k: 0.0 for k in self.labels},
                "avg_steps": 0.0,
                "avg_return": 0.0,
                "declare_step1_ratio": 0.0,
                "majority_baseline_accuracy": 0.0,
                "majority_gain": 0.0,
            }

        correct = sum(1 for r in self.records if r.pred_label == r.true_label)
        avg_steps = sum(r.steps for r in self.records) / n
        avg_return = sum(r.episode_return for r in self.records) / n
        step1 = sum(1 for r in self.records if r.declared_step == 1)
        declare_step1_ratio = step1 / n

        tp, fp, fn = self._per_class_counts()
        recalls = {}
        f1s = []
        for c in self.labels:
            recall_den = tp[c] + fn[c]
            recall = tp[c] / recall_den if recall_den > 0 else 0.0
            precision_den = tp[c] + fp[c]
            precision = tp[c] / precision_den if precision_den > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            recalls[c] = recall
            f1s.append(f1)

        balanced_accuracy = sum(recalls.values()) / len(self.labels) if self.labels else 0.0
        macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0

        true_counter = Counter(r.true_label for r in self.records)
        majority_label, majority_count = true_counter.most_common(1)[0]
        majority_baseline_accuracy = majority_count / n

        return {
            "accuracy": correct / n,
            "balanced_accuracy": balanced_accuracy,
            "macro_f1": macro_f1,
            "per_class_recall": recalls,
            "avg_steps": avg_steps,
            "avg_return": avg_return,
            "declare_step1_ratio": declare_step1_ratio,
            "majority_baseline_accuracy": majority_baseline_accuracy,
            "majority_baseline_label": majority_label,
            "majority_gain": (correct / n) - majority_baseline_accuracy,
        }

