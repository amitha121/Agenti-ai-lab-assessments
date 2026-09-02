"""
Task 6: Research Agent (Search -> Summarize -> Structured Report)
=====================================================================
An agent that searches the web for information on a topic, summarizes
each finding, and compiles everything into a structured research report
complete with a references section.

Pipeline:
  1. SEARCH      : Query DuckDuckGo (free, no API key needed at all) for
                    the topic, get back a handful of results (title, URL,
                    snippet).
  2. SUMMARIZE    : For each result, ask the LLM to extract the key point
                    relevant to the research topic.
  3. SYNTHESIZE   : Feed all per-source summaries to the LLM and have it
                    write one structured report: Introduction, Key
                    Findings (grouped by theme), Conclusion, References.
  4. SAVE          : Write the final report to a markdown file.

Run:
    python research_agent.py
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
# 1. Search
# ---------------------------------------------------------------------------
def search_web(query, max_results=NUM_RESULTS):
    """Free web search via DuckDuckGo -- no API key required."""
    if DDGS is None:
        raise RuntimeError(
            "The 'ddgs' package is not installed. Run: pip install ddgs"
        )
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append({
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
            })
    print(f"\n[Search] Found {len(results)} results for: \"{query}\"")
    for r in results:
        print(f"    - {r['title']}  ({r['url']})")
    return results


# ---------------------------------------------------------------------------
# 2. Summarize each source
# ---------------------------------------------------------------------------
def summarize_source(topic, result):
    prompt = (
        f"Research topic: {topic}\n\n"
        f"Source title: {result['title']}\n"
        f"Source snippet: {result['snippet']}\n\n"
        "In 1-2 sentences, extract the key fact or claim from this snippet "
        "that is relevant to the research topic. If the snippet has "
        "nothing relevant, say 'Not directly relevant.'"
    )
    return chat(prompt, temperature=0.2, max_tokens=150)


def summarize_all_sources(topic, results):
    print("\n[Summarize] Extracting key points per source...")
    summaries = []
    for i, r in enumerate(results, 1):
        point = summarize_source(topic, r)
        print(f"    [{i}] {point}")
        summaries.append({**r, "key_point": point})
    return summaries


# ---------------------------------------------------------------------------
# 3. Synthesize into a structured report
# ---------------------------------------------------------------------------
def synthesize_report(topic, summaries):
    print("\n[Synthesize] Writing structured report...")
    numbered_findings = "\n".join(
        f"[{i+1}] {s['key_point']} (Source: {s['title']}, {s['url']})"
        for i, s in enumerate(summaries)
    )

    system = (
        "You are a research analyst. Write a structured research report "
        "using ONLY the findings given below -- do not invent facts. Use "
        "this exact structure with markdown headers:\n"
        "## Introduction\n(2-3 sentences framing the topic)\n\n"
        "## Key Findings\n(grouped into 2-4 themed bullet sections, each "
        "citing sources inline like [1], [2])\n\n"
        "## Conclusion\n(2-3 sentences synthesizing the findings)\n\n"
        "## References\n(numbered list: [n] Title - URL, matching the "
        "citation numbers used above)"
    )
    prompt = f"""Research topic: {topic}

Findings:
{numbered_findings}

Write the report now."""
    return chat(prompt, system=system, temperature=0.3, max_tokens=1500)


# ---------------------------------------------------------------------------
# 4. Save
# ---------------------------------------------------------------------------
def save_report(topic, report_text):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    safe_name = re.sub(r"[^a-zA-Z0-9]+", "_", topic.lower()).strip("_")[:50]
    filename = f"{safe_name}_{date.today().isoformat()}.md"
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Research Report: {topic}\n\n")
        f.write(f"*Generated {date.today().isoformat()}*\n\n")
        f.write(report_text)
    print(f"\n[Save] Report written to {path}")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def research(topic):
    print(f"\n{'='*70}\nRESEARCHING: {topic}\n{'='*70}")
    results = search_web(topic)
    if not results:
        print("No search results found -- try a different query.")
        return None
    summaries = summarize_all_sources(topic, results)
    report = synthesize_report(topic, summaries)
    print(f"\n[FINAL REPORT]\n{report}")
    save_report(topic, report)
    return report


if __name__ == "__main__":
    topic = "zero trust architecture for AI agent security"
    research(topic)

    print("\nDone. Call research('your topic') to research anything else.")
