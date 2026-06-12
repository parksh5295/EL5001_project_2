#!/usr/bin/env python3
"""Weak event labeler for EVTX-derived NDJSON.

Input schema example (events.ndjson):
{
  "time": "...",
  "event_id": 3,
  "provider": "Microsoft-Windows-Sysmon",
  "process": "powershell.exe",
  "command_line": "...",
  "source_file": "...evtx",
  "scenario_tactic": "Execution",
  ...
}
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

SUSPICIOUS_PROC = {
    "powershell.exe",
    "pwsh.exe",
    "cmd.exe",
    "wscript.exe",
    "cscript.exe",
    "rundll32.exe",
    "regsvr32.exe",
    "mshta.exe",
    "wmic.exe",
    "certutil.exe",
    "procdump.exe",
    "mimikatz.exe",
}

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

TACTIC_HINT_BY_EVENT_ID = {
    1: "Execution",
    3: "Command and Control",
    7: "Defense Evasion",
    10: "Credential Access",
    11: "Defense Evasion",
    12: "Persistence",
    13: "Persistence",
    14: "Persistence",
    4624: "Lateral Movement",
    4662: "Credential Access",
    4672: "Privilege Escalation",
    4688: "Execution",
    4698: "Persistence",
    5145: "Lateral Movement",
    5156: "Command and Control",
    7045: "Persistence",
}


def _text(v: object | None) -> str:
    return (str(v or "")).strip()


def label_event(event: dict) -> tuple[str, list[str], list[str]]:
    """Return (weak_label, weak_rule_ids, weak_tactic_candidates)."""
    rules: list[str] = []
    tactics: list[str] = []

    eid = event.get("event_id")
    try:
        eid_int = int(eid) if eid is not None else None
    except Exception:
        eid_int = None

    provider = _text(event.get("provider")).lower()
    process = _text(event.get("process")).lower()
    cmd = f" {_text(event.get('command_line')).lower()} "

    # Attack-like rules
    if any(tok in cmd for tok in SUSPICIOUS_CMD_TOKENS):
        rules.append("A1_suspicious_command")
    if eid_int == 10:
        rules.append("A2_process_access")
    if eid_int in {12, 13, 14, 4698, 7045}:
        rules.append("A3_persistence_event")
    if eid_int == 3 and ("sysmon" in provider) and (process in SUSPICIOUS_PROC):
        rules.append("A4_suspicious_network_process")
    if eid_int in {4662, 4672}:
        rules.append("A5_privilege_or_directory_access")

    # Benign-like rules (only if no strong attack-like evidence later)
    benign_rules: list[str] = []
    if eid_int in {4624, 4634}:
        benign_rules.append("B1_logon_logoff")
    if eid_int in {7036, 7040}:
        benign_rules.append("B2_service_state_change")
    if process in {"explorer.exe", "svchost.exe", "runtimebroker.exe"} and not _text(event.get("command_line")):
        benign_rules.append("B3_common_system_process")

    if eid_int in TACTIC_HINT_BY_EVENT_ID:
        tactics.append(TACTIC_HINT_BY_EVENT_ID[eid_int])

    # Deduplicate while preserving order
    tactics = list(dict.fromkeys(tactics))

    if rules:
        return "attack-like", rules, tactics
    if benign_rules:
        return "benign-like", benign_rules, tactics
    return "unknown", [], tactics


def parse_args():
    p = argparse.ArgumentParser(description="Apply weak labels to NDJSON events.")
    p.add_argument("--input", type=Path, default=Path("events.ndjson"))
    p.add_argument("--output", type=Path, default=Path("results/events_weak_labeled.ndjson"))
    p.add_argument(
        "--summary-json",
        type=Path,
        default=Path("results/events_weak_label_summary.json"),
    )
    return p.parse_args()


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    label_counter = Counter()
    tactic_hint_counter = Counter()

    with args.input.open("r", encoding="utf-8") as fin, args.output.open(
        "w", encoding="utf-8", newline="\n"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            weak_label, weak_rules, weak_tactics = label_event(event)
            event["weak_label"] = weak_label
            event["weak_rules"] = weak_rules
            event["weak_tactic_candidates"] = weak_tactics

            fout.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")))
            fout.write("\n")

            count += 1
            label_counter[weak_label] += 1
            for t in weak_tactics:
                tactic_hint_counter[t] += 1

    summary = {
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "events": count,
        "weak_label_counts": dict(label_counter),
        "weak_tactic_candidate_counts": dict(tactic_hint_counter),
    }
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Labeled events: {count}")
    print(f"Output: {args.output.resolve()}")
    print(f"Summary: {args.summary_json.resolve()}")
    print(f"Weak labels: {dict(label_counter)}")


if __name__ == "__main__":
    main()

