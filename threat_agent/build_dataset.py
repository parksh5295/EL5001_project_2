#!/usr/bin/env python3
"""Build Threat Investigation episodes from EVTX files.

Output format (JSON):
{
  "tactics": [...],
  "event_id_bins": [...],
  "episodes": [
    {
      "source_file": "...evtx",
      "relative_path": "...",
      "tactic": "Privilege Escalation",
      "cards": {
        "process": [...],
        "registry": [...],
        "network": [...],
        "user": [...]
      },
      "all_event_ids": [...]
    }
  ]
}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

NS = "{http://schemas.microsoft.com/win/2004/08/events/event}"

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

SKIP_DIRS = {".git", "EVTX_ATT&CK_Metadata", "winlogbeat", ".vscode"}
CATEGORY_ORDER = ("process", "registry", "network", "user")

# Fixed bins for state vector histogram
DEFAULT_EVENT_ID_BINS = [
    1,
    3,
    7,
    10,
    11,
    12,
    13,
    14,
    4624,
    4657,
    4662,
    4672,
    4688,
    5145,
    5156,
]


def _import_evtx(evtx_lib_dir: Path):
    sys.path.insert(0, str(evtx_lib_dir))
    from Evtx.Evtx import Evtx

    return Evtx


def scenario_tactic_for(path: Path) -> str:
    for part in path.parts:
        if part in TACTIC_DIRS:
            return part
    if path.name == "UACME_59_Sysmon.evtx":
        return "Privilege Escalation"
    return "Unknown"


def iter_evtx_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in SKIP_DIRS]
        for filename in filenames:
            if filename.lower().endswith(".evtx"):
                files.append(Path(dirpath) / filename)
    return sorted(files)


def normalize_text(value: str | None) -> str | None:
    if value is None:
        return None
    v = value.strip()
    return v or None


def collect_event_fields(event_data: ET.Element | None) -> dict[str, str | None]:
    fields: dict[str, str | None] = {}
    if event_data is None:
        return fields
    for data in event_data.findall(f"{NS}Data"):
        name = data.get("Name")
        if name:
            fields[name] = normalize_text(data.text)
    return fields


def parse_xml(xml_str: str) -> tuple[dict, dict[str, str | None]]:
    root = ET.fromstring(xml_str)
    system = root.find(f"{NS}System")

    event = {
        "time": None,
        "event_id": None,
        "provider": None,
        "channel": None,
        "host": None,
    }

    if system is not None:
        computer = system.find(f"{NS}Computer")
        if computer is not None and computer.text:
            event["host"] = computer.text.strip()

        for child in system:
            tag = child.tag.replace(NS, "")
            if tag == "Provider":
                event["provider"] = child.get("Name")
            elif tag == "TimeCreated":
                event["time"] = child.get("SystemTime")
            elif tag == "Channel":
                event["channel"] = normalize_text(child.text)
            elif tag == "EventID":
                try:
                    event["event_id"] = int((child.text or "0").strip())
                except ValueError:
                    event["event_id"] = None

    fields = collect_event_fields(root.find(f"{NS}EventData"))
    if fields.get("Computer"):
        event["host"] = fields["Computer"]
    return event, fields


def is_registry_event(event_id: int | None, fields: dict[str, str | None]) -> bool:
    if event_id in {12, 13, 14, 4657}:
        return True
    target = (fields.get("TargetObject") or fields.get("ObjectName") or "").upper()
    return "\\REGISTRY\\" in target


def is_network_event(event_id: int | None, fields: dict[str, str | None]) -> bool:
    if event_id in {3, 5156, 5158}:
        return True
    keys = {"DestinationIp", "SourceIp", "DestAddress", "SourceAddress", "DestinationPort"}
    return any(fields.get(k) for k in keys)


def is_process_event(event_id: int | None, fields: dict[str, str | None]) -> bool:
    if event_id in {1, 5, 6, 7, 8, 10, 11, 4688}:
        return True
    keys = {
        "Image",
        "NewProcessName",
        "ProcessName",
        "ParentImage",
        "SourceImage",
        "TargetImage",
        "CommandLine",
    }
    return any(fields.get(k) for k in keys)


def is_user_event(event_id: int | None, fields: dict[str, str | None]) -> bool:
    if event_id in {4624, 4625, 4634, 4648, 4672}:
        return True
    keys = {
        "User",
        "TargetUserName",
        "SubjectUserName",
        "AccountName",
        "SubjectDomainName",
    }
    return any(fields.get(k) for k in keys)


def summarize_card(
    category: str,
    event: dict,
    fields: dict[str, str | None],
) -> str:
    event_id = event.get("event_id")
    if category == "process":
        proc = fields.get("Image") or fields.get("NewProcessName") or fields.get("ProcessName")
        parent = fields.get("ParentImage") or fields.get("CreatorProcessName")
        cmd = fields.get("CommandLine")
        return f"EID={event_id} process={proc} parent={parent} cmd={cmd}"
    if category == "registry":
        target = fields.get("TargetObject") or fields.get("ObjectName")
        op = fields.get("Details") or fields.get("OperationType")
        return f"EID={event_id} target={target} op={op}"
    if category == "network":
        src = fields.get("SourceIp") or fields.get("SourceAddress")
        dst = fields.get("DestinationIp") or fields.get("DestAddress")
        dport = fields.get("DestinationPort") or fields.get("DestPort")
        img = fields.get("Image")
        return f"EID={event_id} image={img} {src}->{dst}:{dport}"
    user = (
        fields.get("User")
        or fields.get("TargetUserName")
        or fields.get("SubjectUserName")
        or fields.get("AccountName")
    )
    domain = fields.get("SubjectDomainName") or fields.get("AccountDomain")
    return f"EID={event_id} user={domain}\\{user}"


def card_fields_for_category(category: str, fields: dict[str, str | None]) -> dict[str, str | None]:
    if category == "process":
        keys = [
            "Image",
            "NewProcessName",
            "ProcessName",
            "ParentImage",
            "CreatorProcessName",
            "SourceImage",
            "TargetImage",
            "CommandLine",
        ]
    elif category == "registry":
        keys = ["TargetObject", "ObjectName", "Details", "OperationType", "NewValue", "OldValue"]
    elif category == "network":
        keys = [
            "Image",
            "SourceIp",
            "DestinationIp",
            "SourcePort",
            "DestinationPort",
            "SourceAddress",
            "DestAddress",
            "Protocol",
        ]
    else:
        keys = [
            "User",
            "TargetUserName",
            "SubjectUserName",
            "SubjectDomainName",
            "AccountName",
            "AccountDomain",
            "LogonType",
        ]
    return {k: fields.get(k) for k in keys if fields.get(k) is not None}


def build_episode(evtx_path: Path, repo_root: Path, Evtx) -> dict:
    cards: dict[str, list[dict]] = {k: [] for k in CATEGORY_ORDER}
    all_event_ids: list[int] = []
    relative = evtx_path.relative_to(repo_root)
    tactic = scenario_tactic_for(relative)

    with Evtx(str(evtx_path)) as log:
        for record in log.records():
            event, fields = parse_xml(record.xml())
            event_id = event["event_id"]
            if isinstance(event_id, int):
                all_event_ids.append(event_id)

            category_flags = {
                "process": is_process_event(event_id, fields),
                "registry": is_registry_event(event_id, fields),
                "network": is_network_event(event_id, fields),
                "user": is_user_event(event_id, fields),
            }

            for category, enabled in category_flags.items():
                if not enabled:
                    continue
                cards[category].append(
                    {
                        "time": event.get("time"),
                        "event_id": event_id,
                        "provider": event.get("provider"),
                        "channel": event.get("channel"),
                        "host": event.get("host"),
                        "summary": summarize_card(category, event, fields),
                        "fields": card_fields_for_category(category, fields),
                    }
                )

    return {
        "source_file": evtx_path.name,
        "relative_path": str(relative).replace("\\", "/"),
        "tactic": tactic,
        "cards": cards,
        "all_event_ids": all_event_ids,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Threat Investigation episode dataset from EVTX.")
    parser.add_argument(
        "--evtx-root",
        type=Path,
        default=Path("evtx_samples"),
        help="Directory to scan for EVTX files.",
    )
    parser.add_argument(
        "--evtx-lib-dir",
        type=Path,
        default=Path("evtx_samples/EVTX_ATT&CK_Metadata"),
        help="Path that contains Evtx parser package.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("results/threat_agent_data.json"),
        help="Output dataset JSON path.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Optional cap on number of EVTX files for quick experiments (0 means all).",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="EVTX file names to skip.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.evtx_root.resolve()
    evtx_lib_dir = args.evtx_lib_dir.resolve()
    output = args.output.resolve()
    Evtx = _import_evtx(evtx_lib_dir)

    files = [p for p in iter_evtx_files(repo_root) if p.name not in set(args.exclude)]
    if args.max_files > 0:
        files = files[: args.max_files]

    episodes = [build_episode(path, repo_root, Evtx) for path in files]
    tactics = sorted({ep["tactic"] for ep in episodes if ep["tactic"] != "Unknown"})

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "tactics": tactics,
                "event_id_bins": DEFAULT_EVENT_ID_BINS,
                "episodes": episodes,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )

    total_cards = sum(
        len(ep["cards"]["process"])
        + len(ep["cards"]["registry"])
        + len(ep["cards"]["network"])
        + len(ep["cards"]["user"])
        for ep in episodes
    )
    print(f"EVTX files processed: {len(episodes)}")
    print(f"Tactics: {len(tactics)} -> {tactics}")
    print(f"Total cards: {total_cards}")
    print(f"Output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
