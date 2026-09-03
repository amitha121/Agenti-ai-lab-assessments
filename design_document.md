# Agentic AI Systems — Design Document
### Tasks 5, 6 & 7

---

## 5. Multi-Agent SDR (Sales Development Rep) System

**File:** `5_multi_agent_sdr.py`

**Goal:** Automate the early sales pipeline — find leads, decide which are
worth pursuing, and draft outreach emails — using three cooperating agents.

### Architecture

```
 Target Customer Profile
          │
          ▼
 ┌─────────────────┐
 │  LeadGenAgent    │  generates candidate leads matching the profile
 └────────┬─────────┘
          │ list of leads
          ▼
 ┌─────────────────┐
 │ QualificationAgent│  rule checks (company size, industry) + LLM judgement
 └────────┬─────────┘
          │ qualified leads only
          ▼
 ┌─────────────────┐
 │   EmailAgent     │  writes a personalized outreach email per lead
 └────────┬─────────┘
          │
          ▼
   Ready-to-send emails
```

### Agent responsibilities

| Agent | Input | Job | Output |
|---|---|---|---|
| **LeadGenAgent** | Target customer profile (text) | Generates realistic candidate leads (synthetic, for demo) with name, title, company, size, industry, pain point | List of lead records (JSON) |
| **QualificationAgent** | One lead record | Applies deterministic rules (min. company size, target industry) AND asks the LLM whether the pain point suggests genuine buying interest | Qualified / Rejected + reasons |
| **EmailAgent** | A qualified lead + product info | Writes a short, personalized cold outreach email referencing the lead's likely pain point | Email text (subject + body) |

### Why this design

- **Separation of concerns:** each agent has exactly one job, so it's easy
  to test, debug, or swap out independently (e.g. replace LeadGenAgent with
  a real LinkedIn/Apollo API call later).
- **Rules + LLM combined in qualification:** hard business rules (company
  size, industry) are deterministic and auditable; the LLM adds judgement
  for things that are hard to encode as a rule (e.g. "does this pain point
  sound genuine?").
- **Synthetic data:** lead generation is simulated by the LLM for demo
  purposes since we don't have access to a real lead database/API.

---

## 6. Policy Compliance Agent

**File:** `6_policy_compliance_agent.py`

**Goal:** Automatically review employee activity logs against company
policy rules and flag violations with a plain-English explanation.

### Architecture

```
 Synthetic Employee Action Records
          │
          ▼
 ┌─────────────────┐
 │   RuleEngine     │  pure Python, deterministic checks (no LLM)
 └────────┬─────────┘
          │ violations found?
          ▼
 ┌─────────────────┐
 │ ComplianceAgent  │  if violated → asks LLM to explain + suggest fix
 └────────┬─────────┘
          │
          ▼
     Compliance Report
```

### Key design decision: rules are NOT decided by the LLM

Compliance rule *evaluation* (rule_no_bulk_download, rule_no_password_sharing,
etc.) is written as plain, deterministic Python — **not** left to the LLM.
This matters because:
- Compliance decisions must be **consistent and auditable** (same input →
  same output, every time).
- LLMs can be inconsistent or "hallucinate" a judgement.

The LLM is only used **after** a rule has already fired, to translate a
technical rule violation into a clear, human-readable explanation and
suggested remediation — a task well suited to language generation.

### Rules implemented (demo set)

| Rule | Severity | Trigger |
|---|---|---|
| No personal-email file transfer | HIGH | Employee emails a company file to a personal account |
| No bulk download without approval | HIGH | Download of >10,000 records in one action |
| No password sharing | HIGH | Employee shares login credentials with a colleague |
| Flag off-hours access | MEDIUM | System accessed outside standard work hours |

### Synthetic data

`generate_synthetic_records()` creates fake employee action logs (fixed
random seed for reproducible demo/grading output) so the agent can be
tested without real company data.

---

## 7. Deep Research Agent Workflow

**File:** `7_deep_research_agent.py`

**Goal:** Produce a well-researched report on a topic using a
**Plan → Research → Draft → Reflect → Revise** loop, rather than answering
in a single LLM call.

### Architecture

```
      Research Topic
            │
            ▼
   ┌─────────────────┐
   │   1. PLAN        │  break topic into 3–4 sub-questions
   └────────┬─────────┘
            ▼
   ┌─────────────────┐
   │   2. RESEARCH    │  answer each sub-question individually
   └────────┬─────────┘
            ▼
   ┌─────────────────┐
   │   3. DRAFT       │  combine sub-answers into a structured report
   └────────┬─────────┘
            ▼
   ┌─────────────────┐
   │   4. REFLECT     │  LLM critiques its own draft (gaps, weak claims)
   └────────┬─────────┘
            ▼
   ┌─────────────────┐
   │   5. REVISE      │  produce final report addressing the critique
   └────────┬─────────┘
            ▼
       Final Report
```

### Why "reflection" matters

Single-shot LLM answers often miss nuance or state things too confidently.
Adding an explicit **self-critique step** (step 4) before the final answer
mimics how a human researcher would draft, then review their own work
before submitting it — this generally produces a more balanced, complete
report than asking the LLM to "just write a report" in one call.

### Notes for a real deployment

In this assignment version, "research" (step 2) uses the LLM's own trained
knowledge. In a production deep-research agent, step 2 would instead call
a **web search tool** or the **RAG pipeline from Task 4** per sub-question,
so answers are grounded in current, retrievable sources rather than the
model's memory.

---

## Common design notes across all three systems

- All three use **Ollama** (`llama3.2`, running locally) — free, no API
  key, no billing, works offline once the model is downloaded.
- All three follow the same basic agent pattern: **break a big task into
  smaller steps, hand off each step's output as the next step's input**
  (multi-agent pipeline / prompt chaining / plan-execute-reflect).
- Deterministic logic (business rules) is kept in plain Python wherever
  possible; the LLM is reserved for tasks that genuinely need language
  understanding or generation — this makes each system easier to test,
  debug, and trust.

## Setup (same as Tasks 1-4)

```bash
# one-time setup
# 1. install Ollama from https://ollama.com
# 2. ollama pull llama3.2
pip install ollama

# run any script
python 5_multi_agent_sdr.py
python 6_policy_compliance_agent.py
python 7_deep_research_agent.py
```
