#!/usr/bin/env python3
"""Build synthetic streams using extracted run units as attack blocks."""

from __future__ import annotations

import argparse
import contextlib
import json
import random
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

SPLITS = ("train", "val", "test")
CONF_LEVEL = {"low": 0, "medium": 1, "high": 2}


def parse_args():
    p = argparse.ArgumentParser(description="Build stream episodes from confident run units.")
    p.add_argument("--events-input", type=Path, default=Path("results/events_weak_labeled.ndjson"))
    p.add_argument("--runs-input", type=Path, default=Path("results/confident_runs.ndjson"))
    p.add_argument("--output", type=Path, default=Path("results/stream_events_runs.ndjson"))
    p.add_argument("--summary-json", type=Path, default=Path("results/stream_runs_summary.json"))
    p.add_argument("--num-streams", type=int, default=3000)
    p.add_argument("--events-per-stream", type=int, default=120)
    p.add_argument("--attack-runs-min", type=int, default=1)
    p.add_argument("--attack-runs-max", type=int, default=3)
    p.add_argument("--benign-gap-min", type=int, default=3)
    p.add_argument("--benign-gap-max", type=int, default=12)
    p.add_argument("--intra-run-benign-prob", type=float, default=0.15)
    p.add_argument("--intra-run-benign-max", type=int, default=2)
    p.add_argument("--max-events-per-run", type=int, default=200)
    p.add_argument("--min-confidence", choices=("low", "medium", "high"), default="medium")
    p.add_argument("--split-mode", choices=("source", "run"), default="source")
    p.add_argument("--split-ratio", type=str, default="0.7,0.15,0.15")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def parse_split_ratio(raw: str) -> tuple[float, float, float]:
    vals = [v.strip() for v in raw.split(",")]
    if len(vals) != 3:
        raise ValueError("--split-ratio must contain three values: train,val,test")
    ratios = tuple(float(v) for v in vals)
    if any(v < 0 for v in ratios):
        raise ValueError("--split-ratio values must be >= 0")
    total = sum(ratios)
    if total <= 0:
        raise ValueError("--split-ratio sum must be > 0")
    return (ratios[0] / total, ratios[1] / total, ratios[2] / total)


def split_counts(total: int, ratios: tuple[float, float, float]) -> dict[str, int]:
    counts = {s: int(total * r) for s, r in zip(SPLITS, ratios)}
    rem = total - sum(counts.values())
    order = sorted(SPLITS, key=lambda s: ratios[SPLITS.index(s)], reverse=True)
    i = 0
    while rem > 0:
        counts[order[i % len(order)]] += 1
        rem -= 1
        i += 1
    return counts


def sample_split(rng: random.Random, ratios: tuple[float, float, float]) -> str:
    u = rng.random()
    t = ratios[0]
    if u < t:
        return "train"
    t += ratios[1]
    if u < t:
        return "val"
    return "test"


def _sort_key(ev: dict):
    return (str(ev.get("time") or ""), int(ev.get("event_id") or 0))


def load_events_by_source(path: Path) -> dict[str, list[dict]]:
    by_source = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            ev = json.loads(s)
            src = str(ev.get("source_file") or "unknown_source")
            by_source[src].append(ev)
    for src in list(by_source.keys()):
        by_source[src] = sorted(by_source[src], key=_sort_key)
    return dict(by_source)


def load_runs(path: Path, min_conf: str) -> list[dict]:
    out = []
    threshold = CONF_LEVEL[min_conf]
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            row = json.loads(s)
            lv = str(row.get("confidence_level") or "low").lower()
            if CONF_LEVEL.get(lv, 0) < threshold:
                continue
            out.append(row)
    return out


def attach_run_events(run_rows: list[dict], events_by_source: dict[str, list[dict]], max_events_per_run: int) -> list[dict]:
    ready = []
    for r in run_rows:
        src = str(r["source_file"])
        if src not in events_by_source:
            continue
        rows = events_by_source[src]
        st = int(r["start_idx"])
        ed = int(r["end_idx"])
        if st < 0 or ed < st or ed >= len(rows):
            continue
        run_events = [deepcopy(x) for x in rows[st : ed + 1]]
        if len(run_events) > max_events_per_run:
            run_events = run_events[:max_events_per_run]
        tactic = str(r.get("dominant_scenario_tactic") or "")
        if not tactic:
            # fallback: scenario_tactic_counts max
            counts = r.get("scenario_tactic_counts") or {}
            if counts:
                tactic = max(counts.items(), key=lambda kv: kv[1])[0]
        if not tactic:
            continue
        ready.append(
            {
                "source_file": src,
                "run_id_in_source": int(r.get("run_id_in_source", 0)),
                "confidence_level": str(r.get("confidence_level") or "low"),
                "confidence_score": float(r.get("confidence_score") or 0.0),
                "tactic": tactic,
                "events": run_events,
            }
        )
    return ready


def choose_split_pools(
    rng: random.Random,
    runs: list[dict],
    events_by_source: dict[str, list[dict]],
    ratios: tuple[float, float, float],
    split_mode: str,
):
    split_runs = {s: [] for s in SPLITS}
    split_benign = {s: [] for s in SPLITS}

    if split_mode == "source":
        sources = sorted(events_by_source.keys())
        rng.shuffle(sources)
        cnt = split_counts(len(sources), ratios)
        source_to_split = {}
        pos = 0
        for s in SPLITS:
            for src in sources[pos : pos + cnt[s]]:
                source_to_split[src] = s
            pos += cnt[s]
        for r in runs:
            s = source_to_split.get(r["source_file"], "train")
            split_runs[s].append(r)
        for src, rows in events_by_source.items():
            s = source_to_split.get(src, "train")
            for ev in rows:
                if str(ev.get("weak_label") or "") == "benign-like":
                    split_benign[s].append(ev)
    else:
        for r in runs:
            s = sample_split(rng, ratios)
            split_runs[s].append(r)
        all_benign = [
            ev
            for rows in events_by_source.values()
            for ev in rows
            if str(ev.get("weak_label") or "") == "benign-like"
        ]
        for ev in all_benign:
            s = sample_split(rng, ratios)
            split_benign[s].append(ev)

    # fallback benign pool if empty
    fallback_bg = [
        ev
        for rows in events_by_source.values()
        for ev in rows
        if str(ev.get("weak_label") or "") != "attack-like"
    ]
    for s in SPLITS:
        if not split_benign[s]:
            split_benign[s] = fallback_bg
    return split_runs, split_benign


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    ratios = parse_split_ratio(args.split_ratio)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)

    events_by_source = load_events_by_source(args.events_input)
    run_rows = load_runs(args.runs_input, args.min_confidence)
    runs = attach_run_events(run_rows, events_by_source, args.max_events_per_run)
    if not runs:
        raise RuntimeError("No valid runs found after confidence filter and event attachment.")

    split_runs, split_benign = choose_split_pools(
        rng=rng,
        runs=runs,
        events_by_source=events_by_source,
        ratios=ratios,
        split_mode=args.split_mode,
    )
    streams_per_split = split_counts(args.num_streams, ratios)

    split_output_paths = {
        s: args.output.with_name(f"{args.output.stem}_{s}{args.output.suffix}") for s in SPLITS
    }
    total_written = 0
    per_stream_stats = []
    stream_global_idx = 0

    with contextlib.ExitStack() as stack:
        fout = stack.enter_context(args.output.open("w", encoding="utf-8", newline="\n"))
        split_writers = {
            s: stack.enter_context(split_output_paths[s].open("w", encoding="utf-8", newline="\n"))
            for s in SPLITS
        }

        for split in SPLITS:
            run_pool = split_runs[split]
            benign_pool = split_benign[split]
            if streams_per_split[split] <= 0:
                continue
            if not run_pool:
                raise RuntimeError(f"No run units available for split={split}.")
            if not benign_pool:
                raise RuntimeError(f"No benign pool available for split={split}.")

            for _ in range(streams_per_split[split]):
                stream_id = f"{split}_runstream_{stream_global_idx:06d}"
                stream_global_idx += 1
                target_n = args.events_per_stream
                stream_pos = 0
                attack_events = 0
                benign_events = 0
                run_count = rng.randint(args.attack_runs_min, args.attack_runs_max)
                chosen_runs = [rng.choice(run_pool) for _ in range(max(1, run_count))]
                lbl_counter = Counter()

                # optional initial benign gap
                init_gap = rng.randint(args.benign_gap_min, args.benign_gap_max)
                for _g in range(init_gap):
                    if stream_pos >= target_n:
                        break
                    ev = deepcopy(rng.choice(benign_pool))
                    ev_for_stream = deepcopy(ev)
                    ev_for_stream.pop("scenario_tactic", None)
                    out = {
                        "stream_id": stream_id,
                        "dataset_split": split,
                        "stream_pos": stream_pos + 1,
                        "synthetic_time": stream_pos + 1,
                        "gt_attack_active": 0,
                        "gt_tactic": "benign",
                        "run_source_file": "",
                        "run_id_in_source": -1,
                        **ev_for_stream,
                    }
                    line = json.dumps(out, ensure_ascii=False, separators=(",", ":"))
                    fout.write(line + "\n")
                    split_writers[split].write(line + "\n")
                    stream_pos += 1
                    benign_events += 1
                    total_written += 1
                    lbl_counter[out.get("weak_label", "unknown")] += 1

                for run in chosen_runs:
                    if stream_pos >= target_n:
                        break
                    tactic = run["tactic"]
                    run_events = run["events"]
                    for ev in run_events:
                        if stream_pos >= target_n:
                            break
                        ev_for_stream = deepcopy(ev)
                        ev_for_stream.pop("scenario_tactic", None)
                        out = {
                            "stream_id": stream_id,
                            "dataset_split": split,
                            "stream_pos": stream_pos + 1,
                            "synthetic_time": stream_pos + 1,
                            "gt_attack_active": 1,
                            "gt_tactic": tactic,
                            "run_source_file": run["source_file"],
                            "run_id_in_source": run["run_id_in_source"],
                            **ev_for_stream,
                        }
                        line = json.dumps(out, ensure_ascii=False, separators=(",", ":"))
                        fout.write(line + "\n")
                        split_writers[split].write(line + "\n")
                        stream_pos += 1
                        attack_events += 1
                        total_written += 1
                        lbl_counter[out.get("weak_label", "unknown")] += 1

                        # inject benign-like events *inside* attack progression
                        if (
                            stream_pos < target_n
                            and args.intra_run_benign_prob > 0
                            and rng.random() < args.intra_run_benign_prob
                        ):
                            k = rng.randint(1, max(1, args.intra_run_benign_max))
                            for _ in range(k):
                                if stream_pos >= target_n:
                                    break
                                bev = deepcopy(rng.choice(benign_pool))
                                bev_for_stream = deepcopy(bev)
                                bev_for_stream.pop("scenario_tactic", None)
                                out_b = {
                                    "stream_id": stream_id,
                                    "dataset_split": split,
                                    "stream_pos": stream_pos + 1,
                                    "synthetic_time": stream_pos + 1,
                                    "gt_attack_active": 1,
                                    "gt_tactic": tactic,
                                    "run_source_file": run["source_file"],
                                    "run_id_in_source": run["run_id_in_source"],
                                    "intra_attack_injected_benign": 1,
                                    **bev_for_stream,
                                }
                                line_b = json.dumps(out_b, ensure_ascii=False, separators=(",", ":"))
                                fout.write(line_b + "\n")
                                split_writers[split].write(line_b + "\n")
                                stream_pos += 1
                                attack_events += 1
                                total_written += 1
                                lbl_counter[out_b.get("weak_label", "unknown")] += 1

                    # inter-run benign gap
                    gap = rng.randint(args.benign_gap_min, args.benign_gap_max)
                    for _g in range(gap):
                        if stream_pos >= target_n:
                            break
                        ev = deepcopy(rng.choice(benign_pool))
                        ev_for_stream = deepcopy(ev)
                        ev_for_stream.pop("scenario_tactic", None)
                        out = {
                            "stream_id": stream_id,
                            "dataset_split": split,
                            "stream_pos": stream_pos + 1,
                            "synthetic_time": stream_pos + 1,
                            "gt_attack_active": 0,
                            "gt_tactic": "benign",
                            "run_source_file": "",
                            "run_id_in_source": -1,
                            **ev_for_stream,
                        }
                        line = json.dumps(out, ensure_ascii=False, separators=(",", ":"))
                        fout.write(line + "\n")
                        split_writers[split].write(line + "\n")
                        stream_pos += 1
                        benign_events += 1
                        total_written += 1
                        lbl_counter[out.get("weak_label", "unknown")] += 1

                while stream_pos < target_n:
                    ev = deepcopy(rng.choice(benign_pool))
                    ev_for_stream = deepcopy(ev)
                    ev_for_stream.pop("scenario_tactic", None)
                    out = {
                        "stream_id": stream_id,
                        "dataset_split": split,
                        "stream_pos": stream_pos + 1,
                        "synthetic_time": stream_pos + 1,
                        "gt_attack_active": 0,
                        "gt_tactic": "benign",
                        "run_source_file": "",
                        "run_id_in_source": -1,
                        **ev_for_stream,
                    }
                    line = json.dumps(out, ensure_ascii=False, separators=(",", ":"))
                    fout.write(line + "\n")
                    split_writers[split].write(line + "\n")
                    stream_pos += 1
                    benign_events += 1
                    total_written += 1
                    lbl_counter[out.get("weak_label", "unknown")] += 1

                per_stream_stats.append(
                    {
                        "stream_id": stream_id,
                        "dataset_split": split,
                        "events": stream_pos,
                        "attack_events": attack_events,
                        "benign_events": benign_events,
                        "weak_label_counts": dict(lbl_counter),
                        "num_runs_chosen": len(chosen_runs),
                        "run_sources": [r["source_file"] for r in chosen_runs],
                        "run_tactics": [r["tactic"] for r in chosen_runs],
                    }
                )

    summary = {
        "events_input": str(args.events_input.resolve()),
        "runs_input": str(args.runs_input.resolve()),
        "output": str(args.output.resolve()),
        "split_output_files": {s: str(split_output_paths[s].resolve()) for s in SPLITS},
        "split_mode": args.split_mode,
        "split_ratio": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
        "num_streams": args.num_streams,
        "streams_per_split": streams_per_split,
        "events_per_stream_target": args.events_per_stream,
        "total_events_written": total_written,
        "seed": args.seed,
        "min_confidence": args.min_confidence,
        "num_run_units_total": len(runs),
        "num_run_units_per_split": {s: len(split_runs[s]) for s in SPLITS},
        "per_stream_stats": per_stream_stats,
    }
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"run-based streams built: {args.num_streams}")
    print(f"total events written: {total_written}")
    print(f"output: {args.output.resolve()}")
    print(f"summary: {args.summary_json.resolve()}")


if __name__ == "__main__":
    main()

