#!/usr/bin/env python3
"""Metrics for stream-level declaration evaluation."""

from __future__ import annotations

from collections import Counter


class StreamEval:
    def __init__(self, labels: list[str]):
        self.labels = labels
        self.records = []

    def add(
        self,
        true_label: str,
        pred_label: str | None,
        steps: int,
        declared_step: int | None,
        episode_return: float,
        first_attack_pos: int | None,
        detection_delay: int | None,
    ):
        self.records.append(
            {
                "true_label": true_label,
                "pred_label": pred_label,
                "steps": steps,
                "declared_step": declared_step,
                "episode_return": episode_return,
                "first_attack_pos": first_attack_pos,
                "detection_delay": detection_delay,
            }
        )

    def _per_class(self):
        tp = Counter({k: 0 for k in self.labels})
        fp = Counter({k: 0 for k in self.labels})
        fn = Counter({k: 0 for k in self.labels})
        for r in self.records:
            y, p = r["true_label"], r["pred_label"]
            for c in self.labels:
                if y == c and p == c:
                    tp[c] += 1
                elif y != c and p == c:
                    fp[c] += 1
                elif y == c and p != c:
                    fn[c] += 1
        return tp, fp, fn

    def summary(self):
        n = len(self.records)
        if n == 0:
            return {}
        correct = sum(1 for r in self.records if r["true_label"] == r["pred_label"])
        avg_steps = sum(r["steps"] for r in self.records) / n
        avg_return = sum(r["episode_return"] for r in self.records) / n
        step1_ratio = sum(1 for r in self.records if r["declared_step"] == 1) / n

        tp, fp, fn = self._per_class()
        recalls = {}
        f1s = []
        for c in self.labels:
            rec_den = tp[c] + fn[c]
            pre_den = tp[c] + fp[c]
            recall = tp[c] / rec_den if rec_den else 0.0
            precision = tp[c] / pre_den if pre_den else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0
            recalls[c] = recall
            f1s.append(f1)

        bal_acc = sum(recalls.values()) / len(self.labels)
        macro_f1 = sum(f1s) / len(f1s)

        true_counter = Counter(r["true_label"] for r in self.records)
        majority_label, majority_count = true_counter.most_common(1)[0]
        majority_acc = majority_count / n

        # delay stats on attack streams where attack correctly declared
        delays = [
            r["detection_delay"]
            for r in self.records
            if r["detection_delay"] is not None and r["pred_label"] != "benign"
        ]
        avg_detection_delay = sum(delays) / len(delays) if delays else None

        return {
            "accuracy": correct / n,
            "balanced_accuracy": bal_acc,
            "macro_f1": macro_f1,
            "per_class_recall": recalls,
            "avg_steps": avg_steps,
            "avg_return": avg_return,
            "declare_step1_ratio": step1_ratio,
            "majority_baseline_accuracy": majority_acc,
            "majority_baseline_label": majority_label,
            "majority_gain": (correct / n) - majority_acc,
            "avg_detection_delay": avg_detection_delay,
        }

