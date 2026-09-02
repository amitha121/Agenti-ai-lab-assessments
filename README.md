# Agentic AI Workflows

Eight independent, runnable Python projects across two assignments.

| Folder | Task | What it demonstrates |
|---|---|---|
| `task1_text_to_sql/` | Text-to-SQL Workflow | Schema retrieval → SQL generation → safe execution |
| `task2_rag_qa/` | RAG-Based QA System | Indexing → retrieval → grounded generation |
| `task3_prompt_chaining/` | Prompt Chaining for Summarization | Map → Reduce → Critique → Refine, chained LLM calls |
| `task4_sql_agent/` | SQL Agent with Tool Use | ReAct loop: Thought → Action (tool call) → Observation, repeated |
| `task5_pdf_qa_agent/` | PDF/Document QA Agent | Extract PDF text → chunk/embed → retrieve → cited answers |
| `task6_research_agent/` | Research Agent | Web search → per-source summarize → structured report + references |
| `task7_security_log_agent/` | Security Log/Alert Analyzer | Heuristic triage → LLM incident classification → severity + mitigation |
| `task8_multi_agent_system/` | Multi-Agent Collaboration | Research Agent → Analyst Agent → Report Agent, sequential handoff |

All tasks use **one free LLM API key (Groq)** for generation. Retrieval-based
tasks (1, 2, 4, 5) also use a **free local embedding model**
(`sentence-transformers`, no key needed). Tasks 6 and 8 use **DuckDuckGo
search** (`ddgs` package), which is also free and needs no API key.

---

## 1. Get your free API key (~1 minute)

1. Go to **https://console.groq.com**
2. Sign up / log in (Google or GitHub — no credit card)
3. Left sidebar → **API Keys** → **Create API Key**
4. Copy the key (starts with `gsk_...`)

## 2. Set up the project

```bash
cd agentic_ai_workflows
pip install -r requirements.txt

copy .env.example .env        # Windows
# or: cp .env.example .env    # Mac/Linux

# then open .env and paste your key in place of "your_key_here"
```

> **Embedding model note:** the first time you run task 1, 2, 4, or 5,
> `sentence-transformers` auto-downloads a small (~90MB) model from Hugging
> Face. Needs internet once, then it's cached and works offline.

## 3. Run each task

```bash
cd task1_text_to_sql        && python text_to_sql.py
cd ../task2_rag_qa          && python rag_qa.py
cd ../task3_prompt_chaining && python prompt_chaining_summarization.py
cd ../task4_sql_agent       && python sql_agent.py
cd ../task5_pdf_qa_agent    && python pdf_qa_agent.py
cd ../task6_research_agent  && python research_agent.py
cd ../task7_security_log_agent && python security_log_agent.py
cd ../task8_multi_agent_system  && python multi_agent_system.py
```

Each script prints every intermediate step so you can screenshot the
pipeline in action for your report.

---

## Assignment 2 — how each new agent works

### Task 5 — PDF/Document QA Agent
Same 3-stage RAG pattern as Task 2, but for PDFs: `pypdf` extracts text
page-by-page, chunks are embedded locally, and the LLM answers citing
`(filename, page N)`. A sample PDF — an SOC incident-response playbook —
ships in `documents/` so it runs out of the box. Drop your own PDFs in and
delete `pdf_vector_store.pkl` to re-index.

### Task 6 — Research Agent
`search_web()` queries DuckDuckGo (no key needed) → each result is
summarized into one key point by the LLM → all key points are synthesized
into one structured report (Introduction / Key Findings / Conclusion /
References) → saved as a `.md` file in `reports/`.

### Task 7 — Security Log/Alert Analyzer
A two-stage design mirroring real SOC tooling: **Stage 1** is a cheap
keyword-based heuristic filter (a stand-in for SIEM correlation rules)
that flags which log lines are worth attention, dropping routine
ALLOW/normal traffic. **Stage 2** sends only the flagged lines to the LLM,
which groups related lines into distinct incidents (e.g. 5 repeated failed
SSH logins from the same IP = one brute-force incident, not five), and for
each incident returns a threat type, severity (Low/Medium/High/Critical),
and concrete mitigation steps. A realistic `sample_logs.txt` is included
(brute-force SSH, malware quarantine, internal SMB scan, credential
takeover pattern, DLP data-exfiltration signal, SQLi, port scan).

### Task 8 — Multi-Agent Collaboration System
Three agents with distinct roles and scoped context, run in sequence by an
orchestrator:
- **Research Agent** — only gathers raw sources, does no analysis.
- **Analyst Agent** — only sees the Research Agent's raw sources; extracts
  3-6 key insights, noting which sources support each and flagging
  disagreements between sources.
- **Report Agent** — only sees the Analyst Agent's insights (not the raw
  sources); writes the final structured, referenced report.

This differs from Task 6 (one agent doing search+summarize+report itself)
by genuinely separating responsibilities across agents that only see what
the previous agent handed them — closer to how production multi-agent
systems scope context per role.

---

## Assignment 1 — how tasks 1-4 work

### Task 1 — Text-to-SQL
Schema-card retrieval (cosine similarity) picks relevant tables, the LLM
writes SQL using only that schema, a regex guard blocks anything that
isn't `SELECT`, then it runs against an auto-generated SQLite sample DB.

### Task 2 — RAG QA
Chunks + embeds 3 sample docs (SOC ops, Zero Trust, RAG basics), retrieves
top-k by cosine similarity, answers with source citations. Vectors cache
to disk so re-runs are instant.

### Task 3 — Prompt Chaining for Summarization
Four chained LLM calls: `chunk summaries → combined draft → critique →
refined final summary` (map-reduce-critique-refine).

### Task 4 — SQL Agent with Tool Use (ReAct)
Unlike Task 1's one-shot pipeline, this is a real agent loop: the LLM
doesn't see the schema upfront. It calls `list_tables()`, then
`get_schema()`, then `run_query()` — reading each result before deciding
its next move — using native function calling (Groq/OpenAI `tools`
parameter). A `MAX_STEPS` cap (default 8) prevents infinite loops.

---

## Troubleshooting

- **`GROQ_API_KEY not set`**: make sure `.env` exists (not `.env.example`)
  and has your real key.
- **`ModuleNotFoundError: ddgs`**: run `pip install ddgs` — this only
  affects tasks 6 and 8.
- **Search returns nothing (tasks 6, 8)**: DuckDuckGo occasionally
  rate-limits automated queries — wait a minute and retry, or try a more
  specific search topic.
- **Rate limit errors from Groq**: the free tier has per-minute limits;
  wait a few seconds and retry, or switch `DEFAULT_MODEL` in
  `shared/llm_client.py` to `"llama-3.1-8b-instant"`.
