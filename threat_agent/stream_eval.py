#!/usr/bin/env python3
"""Metrics for stream-level segment evaluation."""

from __future__ import annotations

from collections import Counter
from statistics import mean


class StreamEval:
    def __init__(self, labels: list[str]):
        self.labels = labels
        self.tactics = [l for l in labels if l != "benign"]
        self.records = []
        self.segment_records = []

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
        if self.segment_records:
            return self._segment_summary()
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

    def add_segment_episode(
        self,
        pred_trace: list[list[str]],
        gt_trace: list[list[str]],
        steps: int,
        episode_return: float,
        boundary_tp: int = 0,
        boundary_fp: int = 0,
        boundary_fn: int = 0,
    ):
        self.segment_records.append(
            {
                "pred_trace": pred_trace,
                "gt_trace": gt_trace,
                "steps": steps,
                "episode_return": episode_return,
                "boundary_tp": boundary_tp,
                "boundary_fp": boundary_fp,
                "boundary_fn": boundary_fn,
            }
        )

    @staticmethod
    def _f1(precision: float, recall: float) -> float:
        if precision + recall == 0:
            return 0.0
        return 2.0 * precision * recall / (precision + recall)

    def _segment_summary(self):
        n = len(self.segment_records)
        if n == 0:
            return {}

        tp = Counter({k: 0 for k in self.tactics})
        fp = Counter({k: 0 for k in self.tactics})
        fn = Counter({k: 0 for k in self.tactics})
        benign_tp = benign_fp = benign_fn = 0

        exact_match_total = 0
        total_steps = 0
        episode_returns = []
        episode_steps = []
        btp = bfp = bfn = 0

        for rec in self.segment_records:
            pred_trace = rec["pred_trace"]
            gt_trace = rec["gt_trace"]
            episode_returns.append(rec["episode_return"])
            episode_steps.append(rec["steps"])
            btp += int(rec.get("boundary_tp", 0))
            bfp += int(rec.get("boundary_fp", 0))
            bfn += int(rec.get("boundary_fn", 0))

            for pred_set_raw, gt_set_raw in zip(pred_trace, gt_trace):
                pred_set = set(pred_set_raw)
                gt_set = set(gt_set_raw)
                total_steps += 1
                if pred_set == gt_set:
                    exact_match_total += 1

                pred_benign = len(pred_set) == 0
                gt_benign = len(gt_set) == 0
                if pred_benign and gt_benign:
                    benign_tp += 1
                elif pred_benign and not gt_benign:
                    benign_fp += 1
                elif not pred_benign and gt_benign:
                    benign_fn += 1

                for t in self.tactics:
                    in_pred = t in pred_set
                    in_gt = t in gt_set
                    if in_pred and in_gt:
                        tp[t] += 1
                    elif in_pred and not in_gt:
                        fp[t] += 1
                    elif not in_pred and in_gt:
                        fn[t] += 1

        per_class_recall = {}
        per_class_f1 = {}
        recalls_for_bal_acc = []
        f1_for_macro = []

        benign_recall_den = benign_tp + benign_fn
        benign_prec_den = benign_tp + benign_fp
        benign_recall = benign_tp / benign_recall_den if benign_recall_den else 0.0
        benign_prec = benign_tp / benign_prec_den if benign_prec_den else 0.0
        benign_f1 = self._f1(benign_prec, benign_recall)
        per_class_recall["benign"] = benign_recall
        per_class_f1["benign"] = benign_f1
        recalls_for_bal_acc.append(benign_recall)
        f1_for_macro.append(benign_f1)

        for t in self.tactics:
            rec_den = tp[t] + fn[t]
            pre_den = tp[t] + fp[t]
            recall = tp[t] / rec_den if rec_den else 0.0
            precision = tp[t] / pre_den if pre_den else 0.0
            f1 = self._f1(precision, recall)
            per_class_recall[t] = recall
            per_class_f1[t] = f1
            recalls_for_bal_acc.append(recall)
            f1_for_macro.append(f1)

        micro_tp = sum(tp.values())
        micro_fp = sum(fp.values())
        micro_fn = sum(fn.values())
        micro_precision = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) else 0.0
        micro_recall = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) else 0.0
        micro_f1 = self._f1(micro_precision, micro_recall)

        boundary_precision = btp / (btp + bfp) if (btp + bfp) else 0.0
        boundary_recall = btp / (btp + bfn) if (btp + bfn) else 0.0
        boundary_f1 = self._f1(boundary_precision, boundary_recall)

        majority_baseline_accuracy = max(0.0, benign_recall)
        accuracy = exact_match_total / total_steps if total_steps else 0.0
        balanced_accuracy = mean(recalls_for_bal_acc) if recalls_for_bal_acc else 0.0
        macro_f1 = mean(f1_for_macro) if f1_for_macro else 0.0

        return {
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "macro_f1": macro_f1,
            "per_class_recall": per_class_recall,
            "per_class_f1": per_class_f1,
            "event_micro_f1": micro_f1,
            "event_macro_f1": macro_f1,
            "segment_boundary_precision": boundary_precision,
            "segment_boundary_recall": boundary_recall,
            "segment_boundary_f1": boundary_f1,
            "avg_steps": mean(episode_steps) if episode_steps else 0.0,
            "avg_return": mean(episode_returns) if episode_returns else 0.0,
            "declare_step1_ratio": 0.0,
            "majority_baseline_accuracy": majority_baseline_accuracy,
            "majority_baseline_label": "benign",
            "majority_gain": accuracy - majority_baseline_accuracy,
            "avg_detection_delay": None,
        }

