"""
Task 8: Multi-Agent Collaboration System
===========================================
Three specialized agents that hand off work to each other in sequence to
complete one task automatically, each with its own role, system prompt,
and responsibility -- this is what distinguishes it from Task 6 (a single
agent doing search+summarize+report itself):

  1. RESEARCH AGENT : searches the web for raw information on the topic
                       and gathers source snippets. Its ONLY job is
                       gathering -- it does not analyze or judge.
  2. ANALYST AGENT   : takes the Research Agent's raw findings and does
                       the thinking -- identifies patterns, agreements/
                       disagreements between sources, risks, and the 3-5
                       most important insights. It does NOT write prose,
                       it produces structured analysis.
  3. REPORT AGENT     : takes the Analyst Agent's structured insights and
                       writes the final polished, structured report with
                       a references section. It does NOT re-analyze the
                       raw sources -- it only knows what the Analyst told it.

Each agent only sees what the previous agent handed it (not the full raw
history), which mirrors how real multi-agent systems scope context per
role. Every handoff is printed so you can see the collaboration happen.

Run:
    python multi_agent_system.py
"""

import os
import re
import sys
from datetime import date

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "shared"))
from llm_client import chat  # noqa: E402

try:
    from ddgs import DDGS
except ImportError:
    DDGS = None

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "reports")
NUM_RESULTS = 6


# ---------------------------------------------------------------------------
# AGENT 1: Research Agent -- gathers raw information, does not analyze
# ---------------------------------------------------------------------------
class ResearchAgent:
    role = "Research Agent"

    def run(self, topic):
        print(f"\n[{self.role}] Searching for information on: \"{topic}\"")
        if DDGS is None:
            raise RuntimeError("Run: pip install ddgs")

        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(topic, max_results=NUM_RESULTS):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                })

        print(f"[{self.role}] Gathered {len(results)} sources:")
        for r in results:
            print(f"    - {r['title']}  ({r['url']})")

        return results  # handed off to the Analyst Agent


# ---------------------------------------------------------------------------
# AGENT 2: Analyst Agent -- turns raw sources into structured insights
# ---------------------------------------------------------------------------
class AnalystAgent:
    role = "Analyst Agent"

    system_prompt = (
        "You are an analyst. You will receive raw search results (title, "
        "URL, snippet) on a research topic. Extract 3-6 KEY INSIGHTS -- "
        "the most important, non-redundant facts or claims relevant to the "
        "topic. For each insight, note which source number(s) support it, "
        "and flag if two sources seem to disagree. Respond as a numbered "
        "list of insights only, no preamble. Format each as:\n"
        "N. <insight> (Source: [n])"
    )

    def run(self, topic, sources):
        print(f"\n[{self.role}] Analyzing {len(sources)} sources from the Research Agent...")
        numbered = "\n".join(
            f"[{i+1}] {s['title']}\n{s['snippet']}\nURL: {s['url']}"
            for i, s in enumerate(sources)
        )
        prompt = f"Research topic: {topic}\n\nSources:\n{numbered}\n\nExtract key insights."
        insights = chat(prompt, system=self.system_prompt, temperature=0.2, max_tokens=800)

        print(f"[{self.role}] Insights extracted:\n{insights}")
        return {"insights": insights, "sources": sources}  # handed off to Report Agent


# ---------------------------------------------------------------------------
# AGENT 3: Report Agent -- writes the final structured, referenced report
# ---------------------------------------------------------------------------
class ReportAgent:
    role = "Report Agent"

    system_prompt = (
        "You are a report writer. You will receive a list of pre-analyzed "
        "insights (already extracted by an analyst) and a source list. "
        "Write a polished structured research report using ONLY the given "
        "insights -- do not add new claims. Use this exact structure:\n"
        "## Introduction\n(2-3 sentences)\n\n"
        "## Key Findings\n(prose paragraphs organized by theme, citing "
        "sources inline like [1])\n\n"
        "## Conclusion\n(2-3 sentences)\n\n"
        "## References\n(numbered: [n] Title - URL)"
    )

    def run(self, topic, analysis):
        print(f"\n[{self.role}] Composing final report from the Analyst Agent's insights...")
        source_list = "\n".join(
            f"[{i+1}] {s['title']} - {s['url']}"
            for i, s in enumerate(analysis["sources"])
        )
        prompt = (
            f"Research topic: {topic}\n\n"
            f"Analyst's insights:\n{analysis['insights']}\n\n"
            f"Source list:\n{source_list}\n\n"
            "Write the final report now."
        )
        report = chat(prompt, system=self.system_prompt, temperature=0.3, max_tokens=1500)
        print(f"[{self.role}] Report complete.")
        return report


# ---------------------------------------------------------------------------
# Orchestrator: runs the three agents in sequence, passing outputs forward
# ---------------------------------------------------------------------------
def save_report(topic, report_text):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    safe_name = re.sub(r"[^a-zA-Z0-9]+", "_", topic.lower()).strip("_")[:50]
    path = os.path.join(OUTPUT_DIR, f"{safe_name}_{date.today().isoformat()}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Multi-Agent Research Report: {topic}\n\n")
        f.write(f"*Generated {date.today().isoformat()} by Research -> Analyst -> Report agent pipeline*\n\n")
        f.write(report_text)
    print(f"\n[Orchestrator] Report saved to {path}")
    return path


def run_pipeline(topic):
    print(f"\n{'='*70}\nMULTI-AGENT PIPELINE: {topic}\n{'='*70}")

    research_agent = ResearchAgent()
    analyst_agent = AnalystAgent()
    report_agent = ReportAgent()

    sources = research_agent.run(topic)
    if not sources:
        print("[Orchestrator] Research Agent found nothing -- stopping pipeline.")
        return None

    analysis = analyst_agent.run(topic, sources)
    final_report = report_agent.run(topic, analysis)

    print(f"\n{'='*70}\nFINAL REPORT\n{'='*70}\n{final_report}")
    save_report(topic, final_report)
    return final_report


if __name__ == "__main__":
    topic = "AI agent security risks in enterprise RAG pipelines"
    run_pipeline(topic)

    print("\nDone. Call run_pipeline('your topic') to research anything else.")
