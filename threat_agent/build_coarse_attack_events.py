#!/usr/bin/env python3
"""Collapse fine-grained tactics into 1~2 coarse attack groups."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


TWO_GROUP_MAP = {
    "Credential Access": "access_lateral",
    "Privilege Escalation": "access_lateral",
    "Lateral Movement": "access_lateral",
    "Persistence": "access_lateral",
    "Execution": "execution_control",
    "Defense Evasion": "execution_control",
    "Discovery": "execution_control",
    "AutomatedTestingTools": "execution_control",
    "Command and Control": "execution_control",
    "Other": "execution_control",
}


def parse_args():
    p = argparse.ArgumentParser(description="Build coarse-labeled weak event file.")
    p.add_argument("--input", type=Path, default=Path("results/events_weak_labeled.ndjson"))
    p.add_argument("--output", type=Path, default=Path("results/events_weak_labeled_coarse.ndjson"))
    p.add_argument("--summary-json", type=Path, default=Path("results/events_coarse_summary.json"))
    p.add_argument(
        "--mode",
        choices=("one", "two"),
        default="two",
        help="'one': all attacks -> attack_generic, 'two': two coarse groups",
    )
    p.add_argument(
        "--drop-unmapped-attack",
        action="store_true",
        help="If set, attack-like rows with unmapped tactic are dropped. Otherwise fallback mapping is used.",
    )
    return p.parse_args()


def _to_list(raw):
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, str):
        return [x.strip() for x in raw.replace("|", ",").split(",") if x.strip()]
    return [str(raw)]


def _map_tactic(name: str, mode: str) -> str | None:
    if not name:
        return None
    if mode == "one":
        return "attack_generic"
    return TWO_GROUP_MAP.get(name)


def _map_candidates(raw, mode: str):
    mapped = []
    seen = set()
    for t in _to_list(raw):
        coarse = _map_tactic(t, mode)
        if coarse and coarse not in seen:
            mapped.append(coarse)
            seen.add(coarse)
    return mapped


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)

    n_in = 0
    n_out = 0
    n_drop = 0
    fine_attack = Counter()
    coarse_attack = Counter()
    source_to_fine = defaultdict(Counter)
    source_to_coarse = defaultdict(Counter)

    with args.input.open("r", encoding="utf-8") as fin, args.output.open("w", encoding="utf-8", newline="\n") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            ev = json.loads(line)

            src = str(ev.get("source_file") or "unknown_source")
            weak = str(ev.get("weak_label") or "")
            fine = str(ev.get("scenario_tactic") or "")
            if fine:
                source_to_fine[src][fine] += 1
                if weak == "attack-like":
                    fine_attack[fine] += 1

            coarse = _map_tactic(fine, args.mode)

            if weak == "attack-like" and coarse is None:
                if args.drop_unmapped_attack:
                    n_drop += 1
                    continue
                coarse = "execution_control" if args.mode == "two" else "attack_generic"

            if coarse is not None:
                ev["scenario_tactic"] = coarse
                if weak == "attack-like":
                    coarse_attack[coarse] += 1
                    source_to_coarse[src][coarse] += 1
            else:
                # benign/unknown rows keep benign-like semantics
                if "scenario_tactic" in ev and weak != "attack-like":
                    ev["scenario_tactic"] = "benign"

            ev["weak_tactic_candidates"] = _map_candidates(ev.get("weak_tactic_candidates"), args.mode)

            fout.write(json.dumps(ev, ensure_ascii=False, separators=(",", ":")))
            fout.write("\n")
            n_out += 1

    source_top = []
    for src, cnt in sorted(source_to_coarse.items(), key=lambda kv: sum(kv[1].values()), reverse=True):
        source_top.append({"source_file": src, "coarse_top": cnt.most_common(3), "fine_top": source_to_fine[src].most_common(3)})

    summary = {
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "mode": args.mode,
        "rows_in": n_in,
        "rows_out": n_out,
        "rows_dropped": n_drop,
        "fine_attack_tactic_counts": dict(fine_attack),
        "coarse_attack_tactic_counts": dict(coarse_attack),
        "source_mapping_preview": source_top[:50],
    }
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"coarse events written: {n_out} (dropped={n_drop})")
    print(f"output: {args.output.resolve()}")
    print(f"summary: {args.summary_json.resolve()}")


if __name__ == "__main__":
    main()

