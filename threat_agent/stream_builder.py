#!/usr/bin/env python3
"""Build quasi-realistic mixed event streams from weak-labeled events.

Creates stream episodes by interleaving:
- attack events (from EVTX source_file blocks)
- benign-like background events
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Build mixed stream episodes from weak-labeled events.")
    p.add_argument("--input", type=Path, default=Path("results/events_weak_labeled.ndjson"))
    p.add_argument("--output", type=Path, default=Path("results/stream_events.ndjson"))
    p.add_argument("--summary-json", type=Path, default=Path("results/stream_summary.json"))
    p.add_argument("--num-streams", type=int, default=20)
    p.add_argument("--events-per-stream", type=int, default=800)
    p.add_argument("--attack-blocks-min", type=int, default=1)
    p.add_argument("--attack-blocks-max", type=int, default=3)
    p.add_argument("--benign-ratio", type=float, default=0.7)
    p.add_argument("--max-attack-events-per-block", type=int, default=160)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _sort_key(event: dict):
    # time is often a string; fallback to source order if unavailable
    return (str(event.get("time") or ""), int(event.get("event_id") or 0))


def _sample_block_events(rng: random.Random, block: list[dict], max_events: int) -> list[dict]:
    if len(block) <= max_events:
        return [deepcopy(e) for e in block]
    # pick contiguous slice to keep local temporal consistency
    start = rng.randint(0, len(block) - max_events)
    return [deepcopy(e) for e in block[start : start + max_events]]


def load_events(path: Path):
    events = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return events


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)

    events = load_events(args.input)
    if not events:
        raise RuntimeError(f"No events found in {args.input}")

    by_source = defaultdict(list)
    benign_pool = []
    background_pool = []
    for ev in events:
        src = ev.get("source_file") or "unknown_source"
        by_source[src].append(ev)
        if ev.get("weak_label") == "benign-like":
            benign_pool.append(ev)
        if ev.get("weak_label") != "attack-like":
            background_pool.append(ev)

    attack_sources = [
        src
        for src, rows in by_source.items()
        if any(r.get("weak_label") == "attack-like" for r in rows)
    ]
    if not attack_sources:
        raise RuntimeError("No attack-like source blocks found. Run stream_labeler first.")
    if not benign_pool:
        # fallback: unknown as pseudo-benign background
        benign_pool = list(background_pool)
    if not benign_pool:
        raise RuntimeError("No benign-like or unknown pool available for background.")

    # sort each source by time-like key
    for src in list(by_source.keys()):
        by_source[src] = sorted(by_source[src], key=_sort_key)

    total_written = 0
    per_stream_stats = []
    with args.output.open("w", encoding="utf-8", newline="\n") as fout:
        for stream_idx in range(args.num_streams):
            stream_id = f"stream_{stream_idx:04d}"
            target_n = args.events_per_stream
            attack_n = max(1, int(target_n * (1.0 - args.benign_ratio)))
            benign_n = max(1, target_n - attack_n)

            num_attack_blocks = rng.randint(args.attack_blocks_min, args.attack_blocks_max)
            chosen_sources = rng.sample(attack_sources, k=min(num_attack_blocks, len(attack_sources)))

            attack_sequences: list[list[dict]] = []
            # distribute attack budget across selected blocks
            if chosen_sources:
                each_cap = max(1, attack_n // len(chosen_sources))
                for src in chosen_sources:
                    block_events = _sample_block_events(
                        rng,
                        [e for e in by_source[src] if e.get("weak_label") == "attack-like"],
                        max_events=min(args.max_attack_events_per_block, each_cap),
                    )
                    if block_events:
                        attack_sequences.append(block_events)

            # background sample
            benign_sample = [deepcopy(rng.choice(benign_pool)) for _ in range(benign_n)]

            # interleave sequences
            attack_ptr = [0] * len(attack_sequences)
            benign_ptr = 0
            stream_pos = 0
            attack_written = 0
            benign_written = 0
            stream_label_counts = Counter()

            while stream_pos < target_n:
                has_attack_left = any(attack_ptr[i] < len(attack_sequences[i]) for i in range(len(attack_sequences)))
                has_benign_left = benign_ptr < len(benign_sample)
                if not has_attack_left and not has_benign_left:
                    break

                choose_benign = has_benign_left and (not has_attack_left or rng.random() < args.benign_ratio)
                if choose_benign:
                    ev = benign_sample[benign_ptr]
                    benign_ptr += 1
                    gt_attack_active = 0
                    gt_tactic = "benign"
                    benign_written += 1
                else:
                    available = [i for i in range(len(attack_sequences)) if attack_ptr[i] < len(attack_sequences[i])]
                    sel = rng.choice(available)
                    ev = deepcopy(attack_sequences[sel][attack_ptr[sel]])
                    attack_ptr[sel] += 1
                    gt_attack_active = 1
                    gt_tactic = ev.get("scenario_tactic") or "unknown_attack"
                    attack_written += 1

                stream_pos += 1
                ev_for_stream = deepcopy(ev)
                # Keep scenario tactic only as hidden ground truth (reward/eval), not as observation payload.
                ev_for_stream.pop("scenario_tactic", None)
                out = {
                    "stream_id": stream_id,
                    "stream_pos": stream_pos,
                    "synthetic_time": stream_pos,  # monotonic synthetic timestamp
                    "gt_attack_active": gt_attack_active,
                    "gt_tactic": gt_tactic,
                    **ev_for_stream,
                }
                fout.write(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
                fout.write("\n")
                total_written += 1
                stream_label_counts[out.get("weak_label", "unknown")] += 1

            # pad remaining positions with generic background to satisfy target length
            while stream_pos < target_n:
                stream_pos += 1
                ev = deepcopy(rng.choice(background_pool))
                ev_for_stream = deepcopy(ev)
                # Keep scenario tactic only as hidden ground truth (reward/eval), not as observation payload.
                ev_for_stream.pop("scenario_tactic", None)
                out = {
                    "stream_id": stream_id,
                    "stream_pos": stream_pos,
                    "synthetic_time": stream_pos,
                    "gt_attack_active": 0,
                    "gt_tactic": "benign",
                    **ev_for_stream,
                }
                fout.write(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
                fout.write("\n")
                total_written += 1
                benign_written += 1
                stream_label_counts[out.get("weak_label", "unknown")] += 1

            per_stream_stats.append(
                {
                    "stream_id": stream_id,
                    "events": stream_pos,
                    "attack_events": attack_written,
                    "benign_events": benign_written,
                    "weak_label_counts": dict(stream_label_counts),
                    "attack_sources": chosen_sources,
                }
            )

    summary = {
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "num_streams": args.num_streams,
        "events_per_stream_target": args.events_per_stream,
        "total_events_written": total_written,
        "seed": args.seed,
        "per_stream_stats": per_stream_stats,
    }
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Streams built: {args.num_streams}")
    print(f"Total events written: {total_written}")
    print(f"Output: {args.output.resolve()}")
    print(f"Summary: {args.summary_json.resolve()}")


if __name__ == "__main__":
    main()

