# Weak Labeling Rules (Event Level)

This document defines weak labeling rules for event-level triage:

- `attack-like`
- `benign-like`
- `unknown`

These are heuristics, not ground-truth labels.

---

## 1) Important Clarification

Current dataset has:

- Event metadata (`event_id`, `provider`, fields)
- File-level tactic label (`source_file` -> tactic)

Current dataset does **not** have:

- Event-level tactic ground truth
- Event-level attack/benign ground truth
- Exact attack start/end timestamp labels

So this rule table provides **weak labels** only.

---

## 2) Label Priority (Conflict Resolution)

If multiple rules match the same event:

1. `attack-like` (highest priority)
2. `benign-like`
3. `unknown` (default)

If both attack-like and benign-like signals appear, assign `attack-like`.

---

## 3) Attack-like Rules

| Rule ID | Condition (example) | Label |
|---|---|---|
| A1 | Sysmon `event_id=10` with `SourceImage` accessing `TargetImage` (especially `lsass.exe`, `winlogon.exe`) | attack-like |
| A2 | Sysmon `event_id=13/12/14` with registry autorun paths (`\\Run`, `\\RunOnce`, services keys) | attack-like |
| A3 | Sysmon `event_id=11` creating suspicious file in startup/temp + suspicious parent process | attack-like |
| A4 | Sysmon `event_id=3` outbound to rare external IP/port from script tools (`powershell`, `cmd`, `wscript`) | attack-like |
| A5 | Sysmon `event_id=7` with suspicious DLL load chain from uncommon path | attack-like |
| A6 | Security `event_id=4662` with DCSync-like access patterns in same short window | attack-like |
| A7 | Explicit known suspicious command patterns in `CommandLine` (dumping, encoded scripts, remote admin abuse) | attack-like |

---

## 4) Benign-like Rules

| Rule ID | Condition (example) | Label |
|---|---|---|
| B1 | Security `event_id=4624` interactive/service logon with typical host/user behavior | benign-like |
| B2 | Security `event_id=4634` normal logoff following normal logon | benign-like |
| B3 | Common process spawn chain (`explorer.exe -> chrome.exe`, office apps) without suspicious indicators | benign-like |
| B4 | Internal/local expected network activity from known system processes | benign-like |
| B5 | Routine software update/installer activity without malicious signals | benign-like |

---

## 5) Unknown Rules

Assign `unknown` when:

- Event has weak context only (single event insufficient)
- Rare provider/event with no clear detection logic
- Benign/attack signals both weak or contradictory

---

## 6) Practical Usage in This Project

Recommended event labeling flow:

1. Parse event fields from EVTX
2. Apply attack-like rules
3. If no match, apply benign-like rules
4. If still no match, assign unknown
5. Store `weak_label` and `matched_rule_ids`

Suggested output fields:

- `weak_label`: one of `attack-like | benign-like | unknown`
- `weak_rules`: list of matched rule IDs (e.g., `["A2","A7"]`)
- `weak_confidence`: low/medium/high (optional)

---

## 7) Cautions

- Do not treat weak labels as true labels in final claims.
- Report both weak-label metrics and uncertainty.
- Keep an `unknown` class to reduce noisy supervision.
- Prefer window-level or sequence-level evaluation for realistic SOC settings.
