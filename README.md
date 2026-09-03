# Applied Agentic AI (LLM & RAG) — Assignment Files

All 7 programs use **Ollama** — a free, local LLM runner. No API key, no
billing, no signup required.

| File | Task |
|---|---|
| `1_llm_workflow.py` | Basic LLM chat (accepts input, returns LLM response) |
| `2_prompt_chaining.py` | Multi-step chain: summary → key points → 3 questions |
| `3_agentic_ai.py` | Simple agent: plans steps, executes each, shows final output |
| `4_rag_qa.py` | RAG: answers questions from a PDF/TXT you provide |
| `5_multi_agent_sdr.py` | Multi-agent SDR system: lead gen → qualification → emailing |
| `6_policy_compliance_agent.py` | Policy compliance agent: rule engine + synthetic data + LLM explanations |
| `7_deep_research_agent.py` | Deep research agent: plan → research → draft → reflect → revise |
| `design_document.md` | Architecture explanation for Tasks 5, 6, and 7 |

## 1. Setup (do this once)

1. **Install Ollama** (free): https://ollama.com — download the installer
   for Windows/Mac/Linux and run it.
2. **Pull a model** (one-time, ~2GB download):
   ```
   ollama pull llama3.2
   ```
3. **Install Python packages:**
   ```
   pip install -r requirements.txt
   ```

That's it — no API key, no billing information needed anywhere.

> **Ollama must be running** in the background when you run the scripts.
> Installing it usually starts it automatically; if a script can't connect,
> just open the Ollama app once or run `ollama serve` in a terminal.

## 2. Running each script

```bash
python 1_llm_workflow.py
python 2_prompt_chaining.py
python 3_agentic_ai.py
python 4_rag_qa.py path/to/your_document.pdf
python 5_multi_agent_sdr.py
python 6_policy_compliance_agent.py
python 7_deep_research_agent.py
```

For Task 4, point it at any PDF or TXT file (a short article, notes file,
or report works fine as a demo document).

## 3. What to say in your submission/demo

- **Task 1** — straightforward "send text to the LLM, print the reply" loop.
- **Task 2** — "prompt chaining": each step's *output* becomes the next
  step's *input* (summary → key points → questions).
- **Task 3** — Plan → Execute → Final Output, the core loop behind more
  advanced agent frameworks like LangChain or CrewAI.
- **Task 4** — RAG pipeline: Load document → Chunk it → Retrieve the most
  relevant chunks (TF-IDF similarity) → Feed only those chunks to the LLM
  so answers are grounded in your document, not guessed.
- **Tasks 5, 6, 7** — see `design_document.md` for full architecture
  diagrams and design reasoning for each.

## 4. Common issues

- `ConnectionError` / can't connect to Ollama → make sure Ollama is
  installed and running (open the app, or run `ollama serve`).
- Model not found → run `ollama pull llama3.2` again.
- `ModuleNotFoundError` → run `pip install -r requirements.txt` again.
- PDF extraction returns empty text (Task 4) → the PDF might be a scanned
  image rather than real text; try a text-based PDF or a `.txt` file.
- Ollama responses are slower than a cloud API — this is normal since the
  model runs on your own CPU/GPU. Use a smaller model (e.g. `llama3.2:1b`)
  for faster (but less capable) responses if needed.
