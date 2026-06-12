#!/usr/bin/env python3
"""Analyze evidence-card coverage in threat_agent dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

CATEGORIES = ("process", "registry", "network", "user")


def parse_args():
    p = argparse.ArgumentParser(description="Analyze evidence coverage in dataset JSON.")
    p.add_argument("--dataset", type=Path, default=Path("results/threat_agent_data.json"))
    return p.parse_args()


def main():
    args = parse_args()
    data = json.loads(args.dataset.read_text(encoding="utf-8"))
    episodes = data["episodes"]

    non_empty = Counter()
    card_totals = Counter()
    tactic_counter = Counter()

    for ep in episodes:
        tactic_counter[ep["tactic"]] += 1
        for c in CATEGORIES:
            n = len(ep["cards"][c])
            card_totals[c] += n
            if n > 0:
                non_empty[c] += 1

    print(f"Episodes: {len(episodes)}")
    print("Tactic distribution (episodes):")
    for k, v in tactic_counter.most_common():
        print(f"  {k}: {v}")
    print("Card coverage:")
    for c in CATEGORIES:
        pct = 100.0 * non_empty[c] / max(1, len(episodes))
        avg = card_totals[c] / max(1, len(episodes))
        print(f"  {c:<8} non-empty={non_empty[c]:>4}/{len(episodes)} ({pct:5.1f}%) avg_cards={avg:8.2f}")


if __name__ == "__main__":
    main()
