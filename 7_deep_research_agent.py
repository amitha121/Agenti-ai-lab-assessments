"""
TASK 7: Deep Research Agent Workflow
----------------------------------------
Implements a PLAN -> RESEARCH -> DRAFT -> REFLECT -> REVISE loop for content
generation on a given topic. This mirrors how "deep research" agents work:
they don't just answer in one shot, they plan sub-questions, gather
information for each, draft an answer, critique their own draft, and
improve it before presenting a final report.

Pipeline:
    1. PLAN      -> break the research topic into 3-4 sub-questions
    2. RESEARCH  -> answer each sub-question individually (using the LLM's
                    own knowledge here; in a real system this step would
                    call a web-search or RAG tool per sub-question)
    3. DRAFT     -> combine the sub-answers into one structured draft report
    4. REFLECT   -> ask the LLM to critique its own draft (gaps, unclear
                    parts, missing evidence)
    5. REVISE    -> produce a final, improved report based on that critique

This version uses OLLAMA (free, runs locally, no API key/billing needed).

HOW TO RUN:
    1. Install Ollama: https://ollama.com
    2. ollama pull llama3.2
    3. pip install ollama
    4. python 7_deep_research_agent.py
"""

import re
import ollama

MODEL_NAME = "llama3.2"


def ask_llm(prompt: str) -> str:
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"].strip()


# ---------------------------------------------------------------
# STEP 1: PLAN
# ---------------------------------------------------------------
def plan_subquestions(topic: str) -> list:
    prompt = (
        f"You are a research planner. Break the research topic '{topic}' "
        "into 3-4 focused sub-questions that together would give a "
        "well-rounded understanding of the topic. Output ONLY a numbered "
        "list, nothing else."
    )
    raw = ask_llm(prompt)
    questions = []
    for line in raw.splitlines():
        line = line.strip()
        match = re.match(r"^\d+[\.\)]\s*(.+)", line)
        if match:
            questions.append(match.group(1).strip())
    if not questions:
        questions = [line.strip("- ").strip() for line in raw.splitlines() if line.strip()]
    return questions


# ---------------------------------------------------------------
# STEP 2: RESEARCH each sub-question
# ---------------------------------------------------------------
def research_subquestion(topic: str, subquestion: str) -> str:
    prompt = (
        f"Research topic: {topic}\n"
        f"Sub-question: {subquestion}\n\n"
        "Give a clear, factual, well-organized answer in 3-5 sentences. "
        "If you are uncertain about a fact, say so rather than guessing."
    )
    return ask_llm(prompt)


# ---------------------------------------------------------------
# STEP 3: DRAFT the combined report
# ---------------------------------------------------------------
def draft_report(topic: str, research_notes: dict) -> str:
    notes_block = "\n\n".join(
        f"Sub-question: {q}\nFindings: {a}" for q, a in research_notes.items()
    )
    prompt = (
        f"Using the research notes below, write a structured report on "
        f"'{topic}' with a short introduction, one section per sub-question "
        "(with a heading), and a brief conclusion.\n\n"
        f"Research notes:\n{notes_block}"
    )
    return ask_llm(prompt)


# ---------------------------------------------------------------
# STEP 4: REFLECT - self-critique the draft
# ---------------------------------------------------------------
def reflect_on_draft(topic: str, draft: str) -> str:
    prompt = (
        f"You are reviewing a draft research report on '{topic}'.\n\n"
        f"Draft:\n{draft}\n\n"
        "Critically evaluate this draft: point out gaps, unclear "
        "explanations, unsupported claims, or missing perspectives. "
        "List 3-5 specific, actionable improvement points."
    )
    return ask_llm(prompt)


# ---------------------------------------------------------------
# STEP 5: REVISE - produce the final improved report
# ---------------------------------------------------------------
def revise_report(topic: str, draft: str, critique: str) -> str:
    prompt = (
        f"Topic: {topic}\n\n"
        f"Original draft:\n{draft}\n\n"
        f"Critique/improvement points:\n{critique}\n\n"
        "Rewrite the report addressing the critique. Keep the same "
        "structure (introduction, sections, conclusion) but improve clarity, "
        "fill gaps where possible, and note remaining uncertainty honestly."
    )
    return ask_llm(prompt)


# ---------------------------------------------------------------
# ORCHESTRATION
# ---------------------------------------------------------------
def run_deep_research(topic: str):
    print(f"\n=== RESEARCH TOPIC ===\n{topic}")

    print("\n=== STEP 1: Planning sub-questions ===")
    subquestions = plan_subquestions(topic)
    for i, q in enumerate(subquestions, 1):
        print(f"{i}. {q}")

    print("\n=== STEP 2: Researching each sub-question ===")
    research_notes = {}
    for q in subquestions:
        print(f"\n-> Researching: {q}")
        answer = research_subquestion(topic, q)
        print(f"   {answer}")
        research_notes[q] = answer

    print("\n=== STEP 3: Drafting the report ===")
    draft = draft_report(topic, research_notes)
    print(draft)

    print("\n=== STEP 4: Self-reflection / critique ===")
    critique = reflect_on_draft(topic, draft)
    print(critique)

    print("\n=== STEP 5: Final revised report ===")
    final_report = revise_report(topic, draft, critique)
    print(final_report)

    return final_report


def main():
    topic = input("Enter a research topic: ").strip()
    if not topic:
        print("No topic entered. Exiting.")
        return
    run_deep_research(topic)


if __name__ == "__main__":
    main()
