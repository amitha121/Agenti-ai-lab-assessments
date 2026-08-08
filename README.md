# Agentic AI Workflows — Text-to-SQL, RAG QA, Prompt Chaining

Three independent, runnable Python workflows:

| Folder | Task | What it demonstrates |
|---|---|---|
| `task1_text_to_sql/` | Text-to-SQL Workflow | Schema retrieval → SQL generation → safe execution |
| `task2_rag_qa/` | RAG-Based QA System | Indexing → retrieval → grounded generation |
| `task3_prompt_chaining/` | Prompt Chaining for Summarization | Map → Reduce → Critique → Refine, chained LLM calls |

All three use **one free LLM API key (Groq)** for generation, and a **free local
embedding model** (`sentence-transformers`, no key needed) for retrieval.

---

## 1. Get your free API key (~1 minute)

1. Go to **https://console.groq.com**
2. Sign up / log in (Google or GitHub login works — no credit card)
3. Left sidebar → **API Keys** → **Create API Key**
4. Copy the key (starts with `gsk_...`)

Groq's free tier is generous and fast enough for all three scripts below.

## 2. Set up the project

```bash
# from inside agentic_ai_workflows/
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env            # Windows: copy .env.example .env
# then open .env and paste your key in place of "your_key_here"
```

> **Note on the embedding model:** the first time you run task 1 or task 2,
> `sentence-transformers` will auto-download a small (~90MB) model
> (`all-MiniLM-L6-v2`) from Hugging Face. This needs a normal internet
> connection once — after that it's cached locally and works offline.

## 3. Run each task

```bash
cd task1_text_to_sql && python text_to_sql.py
cd ../task2_rag_qa   && python rag_qa.py
cd ../task3_prompt_chaining && python prompt_chaining_summarization.py
```

Each script prints every intermediate step (retrieval scores, generated SQL,
retrieved chunks, chain steps, etc.) so you can screenshot the pipeline in
action for your report.

---

## How each pipeline works

### Task 1 — Text-to-SQL
1. **Retrieval**: each table is described as a short "schema card"; the
   question and all schema cards are embedded, cosine similarity picks the
   relevant table(s) — this is what a real system does when a database has
   too many tables to fit in one prompt.
2. **Generation**: the LLM writes SQL using only the retrieved schema.
3. **Safety + execution**: a regex guard blocks anything that isn't a
   `SELECT`, then the query runs against a local SQLite sample database
   (auto-created on first run) and results print as a table.

### Task 2 — RAG QA
1. **Indexing**: all `.txt` files in `documents/` are chunked (120 words,
   30-word overlap) and embedded; vectors are cached to `vector_store.pkl`
   so re-runs are instant.
2. **Retrieval**: the question is embedded and compared against every chunk
   via cosine similarity; top-3 chunks are kept.
3. **Generation**: the LLM answers using only the retrieved chunks and
   cites the source file for each claim.

Sample documents cover SOC operations, Zero Trust architecture, and RAG
pipeline basics — swap in your own `.txt` files any time (delete
`vector_store.pkl` afterward to re-index).

### Task 3 — Prompt Chaining for Summarization
Four chained LLM calls, each consuming the previous step's output:
`chunk-level summaries → combined draft → critique → refined final summary`.
This is a **map-reduce-critique-refine** chain — a common pattern for
producing higher-quality output than a single "summarize this" prompt.

---

## Troubleshooting

- **`GROQ_API_KEY not set`**: make sure `.env` exists (not just
  `.env.example`) and contains your real key, or `export GROQ_API_KEY=...`
  in your terminal before running.
- **Embedding model download fails**: you need internet access on first run
  only; check your connection or try again — it's a one-time ~90MB download.
- **Rate limit errors from Groq**: the free tier has per-minute limits; wait
  a few seconds and re-run, or switch `DEFAULT_MODEL` in
  `shared/llm_client.py` to `"llama-3.1-8b-instant"` for higher throughput.
