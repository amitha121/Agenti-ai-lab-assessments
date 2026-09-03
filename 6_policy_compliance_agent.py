"""
TASK 6: Policy Compliance Agent
----------------------------------
An agent that checks SYNTHETIC "employee action" records against a small
set of company policy RULES, and uses the LLM to explain any violations
in plain English (useful for a compliance/audit report).

Design:
    1. generate_synthetic_records() -> creates fake employee action logs
       (e.g. "sent file X to personal email", "logged in from new device")
    2. RuleEngine -> pure Python, deterministic rule checks (no LLM here -
       compliance rules should NOT depend on a probabilistic model)
    3. ComplianceAgent -> runs the rules on each record, and for any
       VIOLATIONS, asks the LLM to write a clear human-readable explanation
       and a suggested remediation step
    4. Prints a final compliance report

This version uses OLLAMA (free, runs locally, no API key/billing needed)
ONLY for the explanation-writing step. The actual rule evaluation is
plain Python so it stays deterministic and auditable.

HOW TO RUN:
    1. Install Ollama: https://ollama.com
    2. ollama pull llama3.2
    3. pip install ollama
    4. python 6_policy_compliance_agent.py
"""

import random
import ollama

MODEL_NAME = "llama3.2"


def ask_llm(prompt: str) -> str:
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"].strip()


# ---------------------------------------------------------------
# 1. SYNTHETIC DATA GENERATION
# ---------------------------------------------------------------
def generate_synthetic_records(n: int = 8) -> list:
    """Creates fake employee action logs for demo/testing purposes."""
    random.seed(42)  # reproducible output for grading/demo

    employees = ["A. Sharma", "R. Iyer", "K. Verma", "S. Rao", "M. Das"]
    actions = [
        {"action": "emailed_file_to_personal_account", "file_type": "customer_data.csv"},
        {"action": "logged_in_from_new_country", "country": "Unknown VPN exit node"},
        {"action": "downloaded_bulk_records", "record_count": 15000},
        {"action": "shared_password_with_colleague", "system": "internal CRM"},
        {"action": "accessed_system_outside_work_hours", "time": "2:47 AM"},
        {"action": "used_approved_device_normal_login", "device": "company laptop"},
        {"action": "updated_own_profile_info", "field": "phone number"},
        {"action": "submitted_expense_report", "amount": 1200},
    ]

    records = []
    for i in range(n):
        record = {
            "record_id": f"REC-{1000+i}",
            "employee": random.choice(employees),
            **random.choice(actions),
        }
        records.append(record)
    return records


# ---------------------------------------------------------------
# 2. RULE ENGINE (deterministic, no LLM)
# ---------------------------------------------------------------
class RuleEngine:
    """Each rule is a simple function: record -> (violated: bool, rule_name, severity)."""

    def __init__(self):
        self.rules = [
            self.rule_no_personal_email_file_transfer,
            self.rule_no_bulk_download,
            self.rule_no_password_sharing,
            self.rule_flag_off_hours_access,
        ]

    def evaluate(self, record: dict) -> list:
        violations = []
        for rule in self.rules:
            violated, name, severity = rule(record)
            if violated:
                violations.append({"rule": name, "severity": severity})
        return violations

    # --- individual rules ---
    @staticmethod
    def rule_no_personal_email_file_transfer(record):
        if record.get("action") == "emailed_file_to_personal_account":
            return True, "No sending company files to personal email", "HIGH"
        return False, "", ""

    @staticmethod
    def rule_no_bulk_download(record):
        if record.get("action") == "downloaded_bulk_records" and record.get("record_count", 0) > 10000:
            return True, "No bulk downloads over 10,000 records without approval", "HIGH"
        return False, "", ""

    @staticmethod
    def rule_no_password_sharing(record):
        if record.get("action") == "shared_password_with_colleague":
            return True, "Passwords must never be shared between employees", "HIGH"
        return False, "", ""

    @staticmethod
    def rule_flag_off_hours_access(record):
        if record.get("action") == "accessed_system_outside_work_hours":
            return True, "Flag system access outside standard work hours for review", "MEDIUM"
        return False, "", ""


# ---------------------------------------------------------------
# 3. COMPLIANCE AGENT (rules + LLM explanation)
# ---------------------------------------------------------------
class ComplianceAgent:
    def __init__(self):
        self.rule_engine = RuleEngine()

    def review_record(self, record: dict) -> dict:
        violations = self.rule_engine.evaluate(record)
        explanation = None
        if violations:
            explanation = self._explain_violations(record, violations)
        return {
            "record": record,
            "violations": violations,
            "explanation": explanation,
        }

    def _explain_violations(self, record: dict, violations: list) -> str:
        prompt = (
            f"An employee action was flagged by our compliance rule engine.\n"
            f"Record: {record}\n"
            f"Violated rules: {[v['rule'] for v in violations]}\n\n"
            "In 2-3 short sentences, explain in plain English why this is a "
            "concern and suggest ONE concrete remediation step."
        )
        return ask_llm(prompt)


# ---------------------------------------------------------------
# ORCHESTRATION
# ---------------------------------------------------------------
def run_compliance_check():
    print("=== Generating synthetic employee action records ===")
    records = generate_synthetic_records(n=8)
    for r in records:
        print(f"  {r}")

    print("\n=== Running Policy Compliance Agent ===")
    agent = ComplianceAgent()

    flagged_count = 0
    for record in records:
        result = agent.review_record(record)
        print(f"\n--- {record['record_id']} ({record['employee']}) ---")
        if result["violations"]:
            flagged_count += 1
            print(f"STATUS: VIOLATION FOUND")
            for v in result["violations"]:
                print(f"  - Rule broken: {v['rule']}  [Severity: {v['severity']}]")
            print(f"  Explanation: {result['explanation']}")
        else:
            print("STATUS: Compliant, no action needed.")

    print(f"\n=== SUMMARY: {flagged_count} of {len(records)} records flagged ===")


if __name__ == "__main__":
    run_compliance_check()
