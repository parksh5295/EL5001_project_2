#!/usr/bin/env python3
"""Extract source-timeline run candidates for inspection.

This script intentionally reports multiple run definitions:
1) weak_attack_runs: contiguous weak_label == attack-like
2) deterministic_attack_runs: contiguous scenario_tactic != benign/unknown
3) deterministic_tactic_runs: contiguous same non-benign scenario_tactic
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


BENIGN_TACTIC_NAMES = {"", "none", "null", "unknown", "benign"}


@dataclass
class EventRow:
    idx: int
    time: str
    source_file: str
    scenario_tactic: str
    weak_label: str
    event_id: int | None
    weak_rules: list[str]


def parse_args():
    p = argparse.ArgumentParser(description="Summarize per-source timeline runs.")
    p.add_argument("--input", type=Path, default=Path("results/events_weak_labeled.ndjson"))
    p.add_argument("--output-json", type=Path, default=Path("results/source_run_summary.json"))
    p.add_argument("--output-ndjson", type=Path, default=Path("results/source_runs.ndjson"))
    p.add_argument("--min-run-len", type=int, default=1)
    p.add_argument("--merge-gap", type=int, default=0, help="Merge adjacent weak runs if gap <= this size.")
    p.add_argument("--top-k-sources", type=int, default=50)
    return p.parse_args()


def _norm_tactic(raw: object) -> str:
    t = str(raw or "").strip()
    if t.lower() in BENIGN_TACTIC_NAMES:
        return ""
    return t


def _parse_event_id(raw: object) -> int | None:
    try:
        return int(raw)  # type: ignore[arg-type]
    except Exception:
        return None


def _is_attack_weak(ev: EventRow) -> bool:
    return ev.weak_label == "attack-like"


def _is_attack_by_tactic(ev: EventRow) -> bool:
    return ev.scenario_tactic != ""


def _collect_runs(rows: list[EventRow], predicate) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = -1
    for i, row in enumerate(rows):
        ok = bool(predicate(row))
        if ok and start < 0:
            start = i
        if (not ok) and start >= 0:
            runs.append((start, i - 1))
            start = -1
    if start >= 0:
        runs.append((start, len(rows) - 1))
    return runs


def _merge_runs(runs: list[tuple[int, int]], merge_gap: int) -> list[tuple[int, int]]:
    if not runs:
        return []
    merged = [runs[0]]
    for st, ed in runs[1:]:
        pst, ped = merged[-1]
        gap = st - ped - 1
        if gap <= merge_gap:
            merged[-1] = (pst, ed)
        else:
            merged.append((st, ed))
    return merged


def _collect_tactic_runs(rows: list[EventRow]) -> list[tuple[int, int, str]]:
    out: list[tuple[int, int, str]] = []
    cur_tactic = ""
    start = -1
    for i, row in enumerate(rows):
        t = row.scenario_tactic
        if t == "":
            if start >= 0:
                out.append((start, i - 1, cur_tactic))
                start = -1
                cur_tactic = ""
            continue
        if start < 0:
            start = i
            cur_tactic = t
            continue
        if t != cur_tactic:
            out.append((start, i - 1, cur_tactic))
            start = i
            cur_tactic = t
    if start >= 0:
        out.append((start, len(rows) - 1, cur_tactic))
    return out


def _run_stats(rows: list[EventRow], st: int, ed: int) -> dict:
    sub = rows[st : ed + 1]
    weak_counter = Counter(r.weak_label for r in sub)
    rule_counter = Counter()
    eid_counter = Counter()
    for r in sub:
        for rule in r.weak_rules:
            rule_counter[rule] += 1
        if r.event_id is not None:
            eid_counter[r.event_id] += 1
    return {
        "start_idx": st,
        "end_idx": ed,
        "length": ed - st + 1,
        "start_time": sub[0].time,
        "end_time": sub[-1].time,
        "weak_label_counts": dict(weak_counter),
        "top_weak_rules": rule_counter.most_common(5),
        "top_event_ids": eid_counter.most_common(5),
        "attack_like_ratio": (weak_counter.get("attack-like", 0) / max(1, len(sub))),
        "event_preview": [
            {
                "time": sub[0].time,
                "event_id": sub[0].event_id,
                "weak_label": sub[0].weak_label,
                "scenario_tactic": sub[0].scenario_tactic,
            },
            {
                "time": sub[-1].time,
                "event_id": sub[-1].event_id,
                "weak_label": sub[-1].weak_label,
                "scenario_tactic": sub[-1].scenario_tactic,
            },
        ],
    }


def main():
    args = parse_args()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_ndjson.parent.mkdir(parents=True, exist_ok=True)

    by_source: dict[str, list[EventRow]] = defaultdict(list)
    with args.input.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            src = str(raw.get("source_file") or "unknown_source")
            by_source[src].append(
                EventRow(
                    idx=i,
                    time=str(raw.get("time") or ""),
                    source_file=src,
                    scenario_tactic=_norm_tactic(raw.get("scenario_tactic")),
                    weak_label=str(raw.get("weak_label") or "unknown"),
                    event_id=_parse_event_id(raw.get("event_id")),
                    weak_rules=list(raw.get("weak_rules") or []),
                )
            )

    source_rows = []
    for src, rows in by_source.items():
        rows.sort(key=lambda r: (r.time, r.idx))
        weak_runs = _collect_runs(rows, _is_attack_weak)
        weak_runs = _merge_runs(weak_runs, max(0, int(args.merge_gap)))
        weak_runs = [(st, ed) for st, ed in weak_runs if (ed - st + 1) >= args.min_run_len]
        deterministic_attack_runs = _collect_runs(rows, _is_attack_by_tactic)
        deterministic_attack_runs = [(st, ed) for st, ed in deterministic_attack_runs if (ed - st + 1) >= args.min_run_len]
        tactic_runs = [r for r in _collect_tactic_runs(rows) if (r[1] - r[0] + 1) >= args.min_run_len]

        weak_run_stats = [_run_stats(rows, st, ed) for st, ed in weak_runs]
        deterministic_attack_run_stats = [_run_stats(rows, st, ed) for st, ed in deterministic_attack_runs]
        tactic_run_stats = []
        for st, ed, tactic in tactic_runs:
            rs = _run_stats(rows, st, ed)
            rs["scenario_tactic"] = tactic
            tactic_run_stats.append(rs)

        weak_counter = Counter(r.weak_label for r in rows)
        source_row = {
            "source_file": src,
            "num_events": len(rows),
            "time_start": rows[0].time if rows else None,
            "time_end": rows[-1].time if rows else None,
            "scenario_tactic_set": sorted({r.scenario_tactic for r in rows if r.scenario_tactic}),
            "weak_label_counts": dict(weak_counter),
            "run_definition": {
                "weak_attack_runs": "contiguous weak_label == attack-like (heuristic, noisy 가능)",
                "deterministic_attack_runs": "contiguous scenario_tactic != benign/unknown (source timeline deterministic)",
                "deterministic_tactic_runs": "contiguous same scenario_tactic != benign/unknown (source timeline deterministic)",
            },
            "weak_attack_runs": weak_run_stats,
            "deterministic_attack_runs": deterministic_attack_run_stats,
            "deterministic_tactic_runs": tactic_run_stats,
            "num_weak_attack_runs": len(weak_run_stats),
            "num_deterministic_attack_runs": len(deterministic_attack_run_stats),
            "num_deterministic_tactic_runs": len(tactic_run_stats),
        }
        source_rows.append(source_row)

    source_rows.sort(key=lambda x: x["num_events"], reverse=True)
    top = source_rows[: max(1, int(args.top_k_sources))]

    with args.output_ndjson.open("w", encoding="utf-8", newline="\n") as f:
        for row in source_rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")

    summary = {
        "input": str(args.input.resolve()),
        "num_sources": len(source_rows),
        "num_events_total": sum(r["num_events"] for r in source_rows),
        "params": {
            "min_run_len": args.min_run_len,
            "merge_gap": args.merge_gap,
        },
        "top_sources_preview": top,
        "output_ndjson": str(args.output_ndjson.resolve()),
    }
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"source run summary written: {args.output_json.resolve()}")
    print(f"source run ndjson written: {args.output_ndjson.resolve()}")


if __name__ == "__main__":
    main()

