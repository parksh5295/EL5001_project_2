#!/usr/bin/env python3
"""Extract all EVTX files in this repo to a single NDJSON file."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

SCRIPT_DIR = Path(__file__).resolve().parent
EVTX_LIB_DIR = SCRIPT_DIR / "EVTX_ATT&CK_Metadata"
DEFAULT_OUTPUT = SCRIPT_DIR / "events.ndjson"

SKIP_DIRS = {".git", "EVTX_ATT&CK_Metadata", "winlogbeat", ".vscode"}
TACTIC_DIRS = {
    "AutomatedTestingTools",
    "Command and Control",
    "Credential Access",
    "Defense Evasion",
    "Discovery",
    "Execution",
    "Lateral Movement",
    "Other",
    "Persistence",
    "Privilege Escalation",
}

NS = "{http://schemas.microsoft.com/win/2004/08/events/event}"
PROCESS_FIELDS = (
    "NewProcessName",
    "Image",
    "ProcessName",
    "TargetImage",
    "SourceImage",
)


def _import_evtx():
    sys.path.insert(0, str(EVTX_LIB_DIR))
    from Evtx.Evtx import Evtx

    return Evtx


def scenario_tactic_for(path: Path) -> str:
    for part in path.parts:
        if part in TACTIC_DIRS:
            return part
    if path.name == "UACME_59_Sysmon.evtx":
        return "Privilege Escalation"
    return "Unknown"


def basename_process(value: str | None) -> str | None:
    if not value:
        return None
    return Path(value.replace("\\", "/")).name


def collect_event_fields(event_data: ET.Element | None) -> dict[str, str | None]:
    fields: dict[str, str | None] = {}
    if event_data is None:
        return fields

    for data in event_data.findall(f"{NS}Data"):
        name = data.get("Name")
        if name:
            fields[name] = data.text
    return fields


def pick_process(fields: dict[str, str | None]) -> str | None:
    for key in PROCESS_FIELDS:
        process = basename_process(fields.get(key))
        if process:
            return process
    return None


def parse_record(xml_str: str, source_file: str, scenario_tactic: str) -> dict:
    root = ET.fromstring(xml_str)
    system = root.find(f"{NS}System")

    record: dict = {
        "time": None,
        "event_id": None,
        "host": None,
        "channel": None,
        "provider": None,
        "process": None,
        "parent_process": None,
        "command_line": None,
        "user": None,
        "source_file": source_file,
        "scenario_tactic": scenario_tactic,
    }

    host = None
    if system is not None:
        computer = system.find(f"{NS}Computer")
        if computer is not None and computer.text:
            host = computer.text.strip()

        for child in system:
            tag = child.tag.replace(NS, "")
            if tag == "Provider":
                record["provider"] = child.get("Name")
            elif tag == "TimeCreated":
                record["time"] = child.get("SystemTime")
            elif tag == "Channel":
                record["channel"] = (child.text or "").strip() or None
            elif tag == "EventID":
                record["event_id"] = int((child.text or "0").strip())

    fields = collect_event_fields(root.find(f"{NS}EventData"))
    record["host"] = fields.get("Computer") or host
    record["process"] = pick_process(fields)
    record["parent_process"] = basename_process(fields.get("ParentImage"))
    record["command_line"] = fields.get("CommandLine")
    record["user"] = fields.get("User") or fields.get("TargetUserName")

    return record


def iter_evtx_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in SKIP_DIRS]
        for filename in filenames:
            if filename.lower().endswith(".evtx"):
                files.append(Path(dirpath) / filename)
    return sorted(files)


def convert_repo(
    repo_root: Path,
    output_path: Path,
    exclude_files: set[str] | None = None,
) -> tuple[int, int]:
    Evtx = _import_evtx()
    exclude_files = exclude_files or set()

    evtx_files = [
        path
        for path in iter_evtx_files(repo_root)
        if path.name not in exclude_files
    ]

    event_count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="\n") as out:
        for evtx_path in evtx_files:
            scenario_tactic = scenario_tactic_for(evtx_path.relative_to(repo_root))
            source_file = evtx_path.name

            with Evtx(str(evtx_path)) as log:
                for record in log.records():
                    event = parse_record(
                        record.xml(),
                        source_file=source_file,
                        scenario_tactic=scenario_tactic,
                    )
                    out.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")))
                    out.write("\n")
                    event_count += 1

    return len(evtx_files), event_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert all EVTX files in this repository to one NDJSON file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output NDJSON path (default: {DEFAULT_OUTPUT.name})",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="EVTX file names to skip (e.g. CA_PetiPotam_etw_rpc_efsr_5_6.evtx)",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=SCRIPT_DIR,
        help="Repository root to scan (default: script directory)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    file_count, event_count = convert_repo(
        repo_root=args.repo_root.resolve(),
        output_path=args.output.resolve(),
        exclude_files=set(args.exclude),
    )

    output_path = args.output.resolve()
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"EVTX files processed: {file_count}")
    print(f"Events written: {event_count}")
    print(f"Output: {output_path}")
    print(f"Size: {size_mb:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
