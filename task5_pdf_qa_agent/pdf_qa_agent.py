"""
Task 5: PDF/Document QA Agent
===============================
An agent that reads PDF documents, retrieves the relevant passages for a
user's question, and answers using an LLM -- grounded in the actual PDF
content rather than the model's own memory.

Pipeline (same 3-stage RAG pattern as Task 2, but for PDFs):
  1. INGEST     : Extract text from every PDF in documents/ using pypdf,
                   keeping track of which PAGE each piece of text came
                   from (important for citing sources accurately).
  2. INDEX       : Chunk each page's text, embed chunks locally
                   (sentence-transformers, free, no key needed), cache
                   to disk.
  3. RETRIEVE +   : Embed the question, cosine-similarity search across
     GENERATE      all chunks, feed top matches to the LLM, which answers
                   citing "filename, page N" for each claim.

A sample PDF (an SOC incident-response playbook) is included so the
script works out of the box -- drop your own PDFs into documents/ and
delete pdf_vector_store.pkl to re-index.

Run:
    python pdf_qa_agent.py
"""

import glob
import os
import pickle
import sys

import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "shared"))
from llm_client import chat  # noqa: E402

DOCS_DIR = os.path.join(os.path.dirname(__file__), "documents")
CACHE_PATH = os.path.join(os.path.dirname(__file__), "pdf_vector_store.pkl")

CHUNK_WORDS = 150
CHUNK_OVERLAP = 40
TOP_K = 4


# ---------------------------------------------------------------------------
# 1. Ingest: extract text page-by-page from every PDF
# ---------------------------------------------------------------------------
def extract_pdf_pages(path):
    """Returns a list of (page_number, page_text) tuples."""
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((i, text))
    return pages


def chunk_text(text, chunk_words=CHUNK_WORDS, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_words
        chunks.append(" ".join(words[start:end]))
        start += chunk_words - overlap
    return chunks


# ---------------------------------------------------------------------------
# 2. Index: chunk + embed every page, cache to disk
# ---------------------------------------------------------------------------
def build_index(model):
    records = []
    pdf_paths = sorted(glob.glob(os.path.join(DOCS_DIR, "*.pdf")))
    for path in pdf_paths:
        source = os.path.basename(path)
        pages = extract_pdf_pages(path)
        for page_num, page_text in pages:
            for i, chunk in enumerate(chunk_text(page_text)):
                records.append({
                    "text": chunk, "source": source,
                    "page": page_num, "chunk_id": i,
                })

    print(f"[Ingest+Index] {len(records)} chunks from {len(pdf_paths)} PDF(s)")
    if not records:
        return records

    texts = [r["text"] for r in records]
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    for r, v in zip(records, vectors):
        r["vector"] = v

    with open(CACHE_PATH, "wb") as f:
        pickle.dump(records, f)
    return records


def load_or_build_index(model):
    if os.path.exists(CACHE_PATH):
        print(f"[Index] Loaded cached vector store ({CACHE_PATH}).")
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)
    return build_index(model)


# ---------------------------------------------------------------------------
# 3. Retrieve + Generate
# ---------------------------------------------------------------------------
def retrieve(question, model, records, top_k=TOP_K):
    q_vec = model.encode([question], normalize_embeddings=True)[0]
    scored = [(float(np.dot(r["vector"], q_vec)), r) for r in records]
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]
    print("\n[Retrieval] Top matches:")
    for score, r in top:
        preview = r["text"][:60].replace("\n", " ")
        print(f"    {score:.3f}  {r['source']} p.{r['page']}  \"{preview}...\"")
    return [r for _, r in top]


def generate_answer(question, retrieved_chunks):
    context_block = "\n\n".join(
        f"[Source: {c['source']}, page {c['page']}]\n{c['text']}"
        for c in retrieved_chunks
    )
    system = (
        "You are a document assistant. Answer the question using ONLY the "
        "provided excerpts from the PDF. If the excerpts don't contain the "
        "answer, say so plainly. Cite the source as (filename, page N) for "
        "every claim."
    )
    prompt = f"""Document excerpts:
{context_block}

Question: {question}

Answer:"""
    return chat(prompt, system=system, temperature=0.2)


def answer_question(question, model, records):
    print(f"\n=== Question: {question} ===")
    if not records:
        print("No PDFs indexed. Add a .pdf file to documents/ and re-run.")
        return
    retrieved = retrieve(question, model, records)
    answer = generate_answer(question, retrieved)
    print(f"\n[Generation] Answer:\n{answer}")


if __name__ == "__main__":
    print("Loading embedding model (first run downloads ~90MB, then cached)...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    index_records = load_or_build_index(embed_model)

    sample_questions = [
        "What containment actions must be taken within 15 minutes of confirming credential compromise?",
        "How is severity classified in this playbook, from Low to Critical?",
        "What happens during the post-incident review phase?",
    ]

    for q in sample_questions:
        answer_question(q, embed_model, index_records)

    print("\nDone. Drop your own PDFs into documents/, delete "
          "pdf_vector_store.pkl to re-index, and edit `sample_questions` "
          "above to ask your own things.")
