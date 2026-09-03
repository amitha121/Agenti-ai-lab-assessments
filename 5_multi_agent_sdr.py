"""
TASK 5: Multi-Agent SDR (Sales Development Rep) System
---------------------------------------------------------
Three "agents" cooperate in a pipeline, each with a single clear job:

    1. LeadGenAgent       -> generates a list of candidate leads (synthetic,
                              for demo purposes) matching a target profile
    2. QualificationAgent -> scores/qualifies each lead against simple rules
                              PLUS an LLM judgement, and keeps only qualified ones
    3. EmailAgent         -> writes a personalized outreach email for each
                              qualified lead using the LLM

Each agent is just a Python class with one job -- this keeps the design
easy to explain: "Agent" here means a focused unit that takes input,
does one kind of reasoning/action, and hands off output to the next agent.

This version uses OLLAMA (free, runs locally, no API key/billing needed).

HOW TO RUN:
    1. Install Ollama: https://ollama.com
    2. ollama pull llama3.2
    3. pip install ollama
    4. python 5_multi_agent_sdr.py
"""

import ollama
import json

MODEL_NAME = "llama3.2"


def ask_llm(prompt: str) -> str:
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"].strip()


# ---------------------------------------------------------------
# AGENT 1: LEAD GENERATION
# ---------------------------------------------------------------
class LeadGenAgent:
    """Generates a list of candidate leads for a given target customer profile.
    In a real system this would query LinkedIn/Apollo/a CRM API. Here we use
    the LLM to generate realistic SYNTHETIC leads for demo purposes."""

    def generate_leads(self, target_profile: str, num_leads: int = 5) -> list:
        prompt = (
            f"Generate {num_leads} realistic but FICTIONAL sales leads that "
            f"match this target customer profile: '{target_profile}'.\n"
            "For each lead include: name, job_title, company, company_size, "
            "industry, and a one-line note about their likely pain point.\n"
            "Respond ONLY as a JSON list of objects with those exact keys, "
            "no extra text."
        )
        raw = ask_llm(prompt)
        return self._safe_parse_json(raw)

    @staticmethod
    def _safe_parse_json(raw: str) -> list:
        # LLMs sometimes wrap JSON in ```json fences - strip those if present.
        cleaned = raw.strip().strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            print("Warning: could not parse leads as JSON. Raw output was:")
            print(raw)
            return []


# ---------------------------------------------------------------
# AGENT 2: QUALIFICATION
# ---------------------------------------------------------------
class QualificationAgent:
    """Applies simple RULES (company size, industry match) plus an LLM
    judgement call to decide whether a lead is worth pursuing."""

    def __init__(self, min_company_size: int = 50, target_industries=None):
        self.min_company_size = min_company_size
        self.target_industries = target_industries or []

    def qualify(self, lead: dict) -> dict:
        reasons = []
        rule_pass = True

        # --- Rule 1: company size ---
        size = self._extract_size_number(lead.get("company_size", "0"))
        if size < self.min_company_size:
            rule_pass = False
            reasons.append(f"Company size {size} is below minimum {self.min_company_size}")

        # --- Rule 2: industry match (if a target list was given) ---
        if self.target_industries:
            industry = (lead.get("industry") or "").lower()
            if not any(t.lower() in industry for t in self.target_industries):
                rule_pass = False
                reasons.append(f"Industry '{lead.get('industry')}' not in target list")

        # --- LLM judgement: does the pain point suggest genuine interest? ---
        prompt = (
            f"Lead details: {json.dumps(lead)}\n\n"
            "Based on the job title, company, and pain point, would this "
            "person likely be a good sales lead for a B2B SaaS product? "
            "Answer with just YES or NO and one short reason."
        )
        llm_verdict = ask_llm(prompt)
        llm_pass = llm_verdict.strip().upper().startswith("YES")

        qualified = rule_pass and llm_pass
        return {
            "lead": lead,
            "qualified": qualified,
            "rule_pass": rule_pass,
            "llm_verdict": llm_verdict,
            "reasons": reasons,
        }

    @staticmethod
    def _extract_size_number(size_field) -> int:
        """Company size might come as '200' or '100-500 employees' - grab a number."""
        import re
        digits = re.findall(r"\d+", str(size_field))
        return int(digits[0]) if digits else 0


# ---------------------------------------------------------------
# AGENT 3: EMAILING
# ---------------------------------------------------------------
class EmailAgent:
    """Writes a short, personalized outreach email for a qualified lead."""

    def write_email(self, lead: dict, product_name: str, product_pitch: str) -> str:
        prompt = (
            f"Write a short (under 120 words), friendly, non-pushy cold "
            f"outreach email to this lead:\n{json.dumps(lead)}\n\n"
            f"The email is selling: {product_name} - {product_pitch}\n"
            "Reference their likely pain point naturally. Include a subject "
            "line at the top like 'Subject: ...'. End with a simple "
            "call-to-action to book a 15-minute call. Sign off as 'Amitha'."
        )
        return ask_llm(prompt)


# ---------------------------------------------------------------
# ORCHESTRATION: run the full SDR pipeline
# ---------------------------------------------------------------
def run_sdr_pipeline():
    target_profile = input(
        "Describe your target customer (e.g. 'IT managers at mid-size "
        "companies who care about cybersecurity'): "
    ).strip() or "IT managers at mid-size companies who care about cybersecurity"

    product_name = input("Product name (e.g. 'SecureGate'): ").strip() or "SecureGate"
    product_pitch = input("One-line product pitch: ").strip() or \
        "an AI-powered zero-trust gateway that protects RAG/AI pipelines from data leaks"

    print("\n=== STEP 1: Lead Generation ===")
    leadgen = LeadGenAgent()
    leads = leadgen.generate_leads(target_profile, num_leads=5)
    for i, lead in enumerate(leads, 1):
        print(f"{i}. {lead.get('name')} - {lead.get('job_title')} at {lead.get('company')} "
              f"({lead.get('company_size')}, {lead.get('industry')})")

    print("\n=== STEP 2: Qualification ===")
    qualifier = QualificationAgent(
        min_company_size=50,
        target_industries=["technology", "software", "IT", "finance", "healthcare"],
    )
    qualified_leads = []
    for lead in leads:
        result = qualifier.qualify(lead)
        status = "QUALIFIED" if result["qualified"] else "REJECTED"
        print(f"\n- {lead.get('name')} ({lead.get('company')}): {status}")
        print(f"  Rule check passed: {result['rule_pass']}  | Reasons: {result['reasons']}")
        print(f"  LLM verdict: {result['llm_verdict']}")
        if result["qualified"]:
            qualified_leads.append(lead)

    print(f"\n{len(qualified_leads)} of {len(leads)} leads qualified.")

    print("\n=== STEP 3: Personalized Emails ===")
    emailer = EmailAgent()
    for lead in qualified_leads:
        print(f"\n--- Email for {lead.get('name')} ({lead.get('company')}) ---")
        email = emailer.write_email(lead, product_name, product_pitch)
        print(email)


if __name__ == "__main__":
    run_sdr_pipeline()
