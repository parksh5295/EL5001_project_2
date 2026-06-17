#!/usr/bin/env python3
"""Extract run candidates from source timelines with multi-signal confidence."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


SUSPICIOUS_CMD_TOKENS = {
    " -enc ",
    "encodedcommand",
    "mimikatz",
    "procdump",
    "lsass",
    "dcsync",
    "sekurlsa",
    "regsvr32",
    "rundll32",
    "wmic",
    "schtasks /create",
    "powershell",
}

BENIGN_TACTICS = {"", "none", "null", "unknown", "benign"}


@dataclass
class EventRow:
    idx: int
    time_raw: str
    time_dt: datetime | None
    source_file: str
    scenario_tactic: str
    weak_label: str
    weak_rules: list[str]
    weak_tactic_candidates: list[str]
    event_id: int | None
    host: str
    user: str
    process: str
    parent_process: str
    command_line: str


def parse_args():
    p = argparse.ArgumentParser(description="Extract confident attack runs from source timelines.")
    p.add_argument("--input", type=Path, default=Path("results/events_weak_labeled.ndjson"))
    p.add_argument("--output-ndjson", type=Path, default=Path("results/confident_runs.ndjson"))
    p.add_argument("--output-json", type=Path, default=Path("results/confident_runs_summary.json"))
    p.add_argument("--min-run-len", type=int, default=3)
    p.add_argument("--start-threshold", type=float, default=1.4)
    p.add_argument("--end-threshold", type=float, default=0.2)
    p.add_argument("--end-patience", type=int, default=3)
    p.add_argument("--max-gap-sec", type=float, default=20.0)
    p.add_argument("--merge-gap-events", type=int, default=2)
    return p.parse_args()


def _norm_tactic(raw: object) -> str:
    t = str(raw or "").strip()
    return "" if t.lower() in BENIGN_TACTICS else t


def _parse_dt(raw: object) -> datetime | None:
    s = str(raw or "").strip()
    if not s:
        return None
    # Support "YYYY-mm-dd HH:MM:SS.ffffff" style.
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _parse_int(raw: object) -> int | None:
    try:
        return int(raw)  # type: ignore[arg-type]
    except Exception:
        return None


def _to_list(raw: object) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw if str(x).strip()]
    if isinstance(raw, str):
        return [x.strip() for x in raw.replace("|", ",").split(",") if x.strip()]
    s = str(raw).strip()
    return [s] if s else []


def _event_evidence(ev: EventRow) -> float:
    score = 0.0
    if ev.weak_label == "attack-like":
        score += 2.0
    elif ev.weak_label == "benign-like":
        score -= 1.0
    score += 0.7 * min(3, len(ev.weak_rules))
    score += 0.3 * min(3, len(ev.weak_tactic_candidates))
    cmd = f" {ev.command_line.lower()} "
    if any(tok in cmd for tok in SUSPICIOUS_CMD_TOKENS):
        score += 0.5
    if ev.scenario_tactic:
        score += 0.2
    return score


def _chain_break(prev_ev: EventRow, cur_ev: EventRow) -> float:
    b = 0.0
    if prev_ev.host and cur_ev.host and prev_ev.host != cur_ev.host:
        b += 1.0
    if prev_ev.user and cur_ev.user and prev_ev.user != cur_ev.user:
        b += 1.0
    if prev_ev.process and cur_ev.process and prev_ev.process != cur_ev.process:
        b += 0.5
    if prev_ev.parent_process and cur_ev.parent_process and prev_ev.parent_process != cur_ev.parent_process:
        b += 0.4
    if prev_ev.scenario_tactic != cur_ev.scenario_tactic:
        b += 0.8
    return b


def _gap_seconds(prev_ev: EventRow, cur_ev: EventRow) -> float | None:
    if prev_ev.time_dt is None or cur_ev.time_dt is None:
        return None
    return (cur_ev.time_dt - prev_ev.time_dt).total_seconds()


def _confidence_from_run(events: list[EventRow], evidences: list[float]) -> tuple[float, str]:
    n = max(1, len(events))
    weak_attack_ratio = sum(1 for e in events if e.weak_label == "attack-like") / n
    rule_div = len({r for e in events for r in e.weak_rules})
    mean_evidence = sum(evidences) / max(1, len(evidences))
    length_score = min(1.0, n / 20.0)
    score = (
        0.45 * weak_attack_ratio
        + 0.20 * min(1.0, rule_div / 3.0)
        + 0.25 * min(1.0, max(0.0, mean_evidence) / 3.0)
        + 0.10 * length_score
    )
    if score >= 0.67:
        return score, "high"
    if score >= 0.45:
        return score, "medium"
    return score, "low"


def _merge_runs(runs: list[tuple[int, int]], gap_events: int) -> list[tuple[int, int]]:
    if not runs:
        return []
    runs = sorted(runs)
    merged = [runs[0]]
    for st, ed in runs[1:]:
        pst, ped = merged[-1]
        if st - ped - 1 <= gap_events:
            merged[-1] = (pst, ed)
        else:
            merged.append((st, ed))
    return merged


def main():
    args = parse_args()
    args.output_ndjson.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

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
                    time_raw=str(raw.get("time") or ""),
                    time_dt=_parse_dt(raw.get("time")),
                    source_file=src,
                    scenario_tactic=_norm_tactic(raw.get("scenario_tactic")),
                    weak_label=str(raw.get("weak_label") or "unknown"),
                    weak_rules=_to_list(raw.get("weak_rules")),
                    weak_tactic_candidates=_to_list(raw.get("weak_tactic_candidates")),
                    event_id=_parse_int(raw.get("event_id")),
                    host=str(raw.get("host") or ""),
                    user=str(raw.get("user") or ""),
                    process=str(raw.get("process") or ""),
                    parent_process=str(raw.get("parent_process") or ""),
                    command_line=str(raw.get("command_line") or ""),
                )
            )

    run_rows = []
    confidence_counter = Counter()
    source_counter = Counter()

    for src, rows in by_source.items():
        rows.sort(key=lambda e: (e.time_raw, e.idx))
        evidences = [_event_evidence(e) for e in rows]

        candidate_runs: list[tuple[int, int]] = []
        in_run = False
        start = -1
        low_streak = 0

        for i, ev in enumerate(rows):
            ev_score = evidences[i]
            prev = rows[i - 1] if i > 0 else None
            gap = _gap_seconds(prev, ev) if prev is not None else None
            chain_break = _chain_break(prev, ev) if prev is not None else 0.0

            if not in_run:
                if ev_score >= args.start_threshold:
                    in_run = True
                    start = i
                    low_streak = 0
                continue

            # in run: evaluate end conditions
            should_end = False
            if gap is not None and gap > args.max_gap_sec:
                should_end = True
            if chain_break >= 1.8 and ev_score < args.start_threshold:
                should_end = True
            if ev_score <= args.end_threshold:
                low_streak += 1
            else:
                low_streak = 0
            if low_streak >= args.end_patience:
                should_end = True

            if should_end:
                end = i - 1
                if start >= 0 and end >= start:
                    candidate_runs.append((start, end))
                in_run = False
                start = -1
                low_streak = 0
                if ev_score >= args.start_threshold:
                    in_run = True
                    start = i

        if in_run and start >= 0:
            candidate_runs.append((start, len(rows) - 1))

        candidate_runs = _merge_runs(candidate_runs, max(0, int(args.merge_gap_events)))
        candidate_runs = [(st, ed) for st, ed in candidate_runs if (ed - st + 1) >= args.min_run_len]

        for ridx, (st, ed) in enumerate(candidate_runs, start=1):
            run_events = rows[st : ed + 1]
            run_evidences = evidences[st : ed + 1]
            confidence_score, confidence_level = _confidence_from_run(run_events, run_evidences)
            confidence_counter[confidence_level] += 1
            source_counter[src] += 1
            tactic_counter = Counter(e.scenario_tactic for e in run_events if e.scenario_tactic)
            weak_counter = Counter(e.weak_label for e in run_events)
            run_rows.append(
                {
                    "source_file": src,
                    "run_id_in_source": ridx,
                    "start_idx": st,
                    "end_idx": ed,
                    "length": ed - st + 1,
                    "start_time": run_events[0].time_raw,
                    "end_time": run_events[-1].time_raw,
                    "dominant_scenario_tactic": tactic_counter.most_common(1)[0][0] if tactic_counter else "",
                    "scenario_tactic_counts": dict(tactic_counter),
                    "weak_label_counts": dict(weak_counter),
                    "attack_like_ratio": weak_counter.get("attack-like", 0) / max(1, len(run_events)),
                    "confidence_score": round(confidence_score, 4),
                    "confidence_level": confidence_level,
                    "top_weak_rules": Counter(r for e in run_events for r in e.weak_rules).most_common(5),
                    "top_event_ids": Counter(e.event_id for e in run_events if e.event_id is not None).most_common(5),
                    "event_preview": [
                        {
                            "time": run_events[0].time_raw,
                            "event_id": run_events[0].event_id,
                            "weak_label": run_events[0].weak_label,
                            "scenario_tactic": run_events[0].scenario_tactic,
                            "process": run_events[0].process,
                        },
                        {
                            "time": run_events[-1].time_raw,
                            "event_id": run_events[-1].event_id,
                            "weak_label": run_events[-1].weak_label,
                            "scenario_tactic": run_events[-1].scenario_tactic,
                            "process": run_events[-1].process,
                        },
                    ],
                }
            )

    run_rows.sort(key=lambda r: (r["source_file"], r["start_idx"]))
    with args.output_ndjson.open("w", encoding="utf-8", newline="\n") as f:
        for row in run_rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")

    summary = {
        "input": str(args.input.resolve()),
        "num_sources": len(by_source),
        "num_runs": len(run_rows),
        "confidence_counts": dict(confidence_counter),
        "sources_with_runs": len(source_counter),
        "params": {
            "min_run_len": args.min_run_len,
            "start_threshold": args.start_threshold,
            "end_threshold": args.end_threshold,
            "end_patience": args.end_patience,
            "max_gap_sec": args.max_gap_sec,
            "merge_gap_events": args.merge_gap_events,
        },
        "top_runs_preview": run_rows[:100],
        "output_ndjson": str(args.output_ndjson.resolve()),
    }
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"confident runs written: {len(run_rows)}")
    print(f"summary: {args.output_json.resolve()}")
    print(f"runs: {args.output_ndjson.resolve()}")


if __name__ == "__main__":
    main()

