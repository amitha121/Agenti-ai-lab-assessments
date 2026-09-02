"""
Task 7: Security Log / Alert Analysis Agent
==============================================
An agent that reads raw security logs, flags suspicious lines with a
lightweight heuristic pre-filter (the way a SIEM correlation rule would),
then hands the flagged lines to an LLM acting as a SOC analyst to:
  - identify distinct security incidents (grouping related log lines,
    e.g. 5 repeated failed-login lines = ONE brute-force incident)
  - classify each incident's threat type
  - assign a severity: Low / Medium / High / Critical
  - suggest concrete mitigation steps

Two-stage design (this mirrors real SOC tooling):
  STAGE 1 - HEURISTIC TRIAGE : cheap keyword/pattern rules flag which log
             lines are worth an analyst's (or LLM's) attention, filtering
             out obvious normal traffic (e.g. ALLOW rules). This keeps
             the LLM prompt small and focused, and mirrors how a SIEM's
             correlation rules reduce alert volume before Tier 1 triage.
  STAGE 2 - LLM ANALYSIS      : the flagged lines are sent to the LLM,
             which reasons about them together (so it can correlate,
             e.g., 5 failed-SSH lines from the same IP into one
             brute-force incident) and returns structured findings.

Run:
    python security_log_agent.py
"""

import json
import os
import re
import sys
from datetime import date

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "shared"))
from llm_client import chat  # noqa: E402

LOG_PATH = os.path.join(os.path.dirname(__file__), "sample_logs.txt")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "reports")

# Keywords that make a log line worth escalating to the LLM for analysis.
# (In a real SIEM these would be proper correlation rules, not just
# keyword matching -- this is a simplified stand-in for that layer.)
SUSPICIOUS_PATTERNS = [
    r"failed password", r"quarantin", r"deny\b", r"sql injection",
    r"port scan", r"mfa method removed", r"dlp", r"unauthorized",
    r"new device", r"base64 encoded command", r"malware", r"loader signature",
]


# ---------------------------------------------------------------------------
# Stage 1: Heuristic triage
# ---------------------------------------------------------------------------
def load_logs(path=LOG_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def flag_suspicious_lines(lines):
    pattern = re.compile("|".join(SUSPICIOUS_PATTERNS), re.IGNORECASE)
    flagged = []
    for i, line in enumerate(lines, 1):
        if pattern.search(line):
            flagged.append((i, line))
    print(f"[Triage] {len(flagged)} of {len(lines)} log lines flagged "
          f"as worth analyzing (filtered out routine ALLOW/normal traffic).")
    return flagged


# ---------------------------------------------------------------------------
# Stage 2: LLM analysis - identify incidents, classify, recommend mitigation
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a SOC Tier 2 analyst. You will be given a set of
flagged security log lines (with line numbers). Group related lines into
distinct INCIDENTS (e.g. multiple repeated failed-login lines from the same
source are ONE brute-force incident, not five separate ones). For each
incident, determine:
  - threat_type: a short label (e.g. "Brute Force Login Attempt",
    "Malware Execution", "SQL Injection Attempt", "Internal Network Scan",
    "Possible Data Exfiltration")
  - severity: exactly one of Low, Medium, High, Critical
  - related_lines: the line numbers that are part of this incident
  - description: 1-2 sentences explaining what happened
  - mitigation: 2-4 concrete, specific steps an analyst should take right now

Respond with ONLY valid JSON: a list of incident objects with keys
threat_type, severity, related_lines, description, mitigation (mitigation
is a list of strings). No markdown fences, no extra text."""


def analyze_incidents(flagged_lines):
    numbered_text = "\n".join(f"[{i}] {line}" for i, line in flagged_lines)
    prompt = f"Flagged log lines:\n{numbered_text}\n\nIdentify the incidents."

    print("\n[Analysis] Sending flagged lines to LLM for incident "
          "classification...")
    raw = chat(prompt, system=SYSTEM_PROMPT, temperature=0, max_tokens=2000)

    # Defensive parsing: strip accidental markdown fences if present
    cleaned = re.sub(r"^```json|```$|^```", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        incidents = json.loads(cleaned)
    except json.JSONDecodeError:
        print("[Analysis] Warning: could not parse JSON, printing raw output instead.")
        print(raw)
        return []
    return incidents


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
SEVERITY_ORDER = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}


def print_incidents(incidents):
    incidents = sorted(incidents, key=lambda x: SEVERITY_ORDER.get(x.get("severity", "Low"), 4))
    print(f"\n{'='*70}\nSECURITY INCIDENT SUMMARY  ({len(incidents)} incident(s) found)\n{'='*70}")
    for idx, inc in enumerate(incidents, 1):
        print(f"\n[{idx}] {inc.get('threat_type', 'Unknown')}  "
              f"-- Severity: {inc.get('severity', 'Unknown')}")
        print(f"    Log lines: {inc.get('related_lines', [])}")
        print(f"    What happened: {inc.get('description', '')}")
        print("    Mitigation steps:")
        for step in inc.get("mitigation", []):
            print(f"      - {step}")


def save_report(incidents):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"incident_report_{date.today().isoformat()}.md")
    incidents = sorted(incidents, key=lambda x: SEVERITY_ORDER.get(x.get("severity", "Low"), 4))
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Security Incident Report\n\n*Generated {date.today().isoformat()}*\n\n")
        f.write(f"**{len(incidents)} incident(s) identified**\n\n")
        for idx, inc in enumerate(incidents, 1):
            f.write(f"## {idx}. {inc.get('threat_type', 'Unknown')} "
                    f"— Severity: {inc.get('severity', 'Unknown')}\n\n")
            f.write(f"**Related log lines:** {inc.get('related_lines', [])}\n\n")
            f.write(f"**Description:** {inc.get('description', '')}\n\n")
            f.write("**Mitigation steps:**\n")
            for step in inc.get("mitigation", []):
                f.write(f"- {step}\n")
            f.write("\n")
    print(f"\n[Save] Report written to {path}")
    return path


if __name__ == "__main__":
    lines = load_logs()
    print(f"Loaded {len(lines)} log lines from {LOG_PATH}")

    flagged = flag_suspicious_lines(lines)
    incidents = analyze_incidents(flagged)

    if incidents:
        print_incidents(incidents)
        save_report(incidents)
    else:
        print("No incidents parsed.")

    print("\nDone. Replace sample_logs.txt with your own logs and re-run.")
