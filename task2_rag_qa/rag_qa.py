"""
Task 2: RAG-Based Question Answering System
=============================================
Implements the three classic RAG stages end-to-end:

  1. INDEXING   : Read all .txt files in documents/, split them into
                   overlapping chunks, embed each chunk with a free local
                   sentence-transformers model, and store the vectors in
                   memory (and cached to disk as vector_store.pkl so you
                   don't have to re-embed every run).
  2. RETRIEVAL  : Embed the user's question, rank all chunks by cosine
                   similarity, keep the top-k most relevant chunks.
  3. GENERATION : Feed the retrieved chunks + question to the LLM (Groq)
                   and ask it to answer using ONLY that context, citing
                   which source file each fact came from.

Run:
    python rag_qa.py
"""

import glob
import os
import pickle
import sys

import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "shared"))
from llm_client import chat  # noqa: E402

DOCS_DIR = os.path.join(os.path.dirname(__file__), "documents")
CACHE_PATH = os.path.join(os.path.dirname(__file__), "vector_store.pkl")

CHUNK_WORDS = 120
CHUNK_OVERLAP = 30
TOP_K = 3


# ---------------------------------------------------------------------------
# 1. Indexing
# ---------------------------------------------------------------------------
def chunk_text(text, chunk_words=CHUNK_WORDS, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_words
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_words - overlap
    return chunks


def build_index(model):
    """Chunk every document, embed every chunk, return the index."""
    records = []  # each: {"text":..., "source":..., "chunk_id":...}
    for path in sorted(glob.glob(os.path.join(DOCS_DIR, "*.txt"))):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        source = os.path.basename(path)
        for i, chunk in enumerate(chunk_text(text)):
            records.append({"text": chunk, "source": source, "chunk_id": i})

    texts = [r["text"] for r in records]
    print(f"[Indexing] {len(records)} chunks from "
          f"{len(glob.glob(os.path.join(DOCS_DIR, '*.txt')))} documents")

    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    for r, v in zip(records, vectors):
        r["vector"] = v

    with open(CACHE_PATH, "wb") as f:
        pickle.dump(records, f)
    return records


def load_or_build_index(model):
    if os.path.exists(CACHE_PATH):
        print("[Indexing] Loaded cached vector store "
              f"({CACHE_PATH}). Delete this file to re-index.")
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)
    return build_index(model)


# ---------------------------------------------------------------------------
# 2. Retrieval
# ---------------------------------------------------------------------------
def retrieve(question, model, records, top_k=TOP_K):
    q_vec = model.encode([question], normalize_embeddings=True)[0]
    scored = []
    for r in records:
        score = float(np.dot(r["vector"], q_vec))
        scored.append((score, r))
    scored.sort(key=lambda x: x[0], reverse=True)

    top = scored[:top_k]
    print("\n[Retrieval] Top chunks:")
    for score, r in top:
        preview = r["text"][:70].replace("\n", " ")
        print(f"    {score:.3f}  {r['source']} #{r['chunk_id']}  \"{preview}...\"")
    return [r for _, r in top]


# ---------------------------------------------------------------------------
# 3. Generation
# ---------------------------------------------------------------------------
def generate_answer(question, retrieved_chunks):
    context_block = "\n\n".join(
        f"[Source: {c['source']} chunk {c['chunk_id']}]\n{c['text']}"
        for c in retrieved_chunks
    )
    system = (
        "You are a precise assistant. Answer the question using ONLY the "
        "provided context. If the context does not contain the answer, say "
        "so plainly instead of guessing. Cite the source file for each "
        "claim in square brackets, e.g. [soc_operations.txt]."
    )
    prompt = f"""Context:
{context_block}

Question: {question}

Answer:"""
    return chat(prompt, system=system, temperature=0.2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def answer_question(question, model, records):
    print(f"\n=== Question: {question} ===")
    retrieved = retrieve(question, model, records)
    answer = generate_answer(question, retrieved)
    print(f"\n[Generation] Answer:\n{answer}")


if __name__ == "__main__":
    print("Loading embedding model (first run downloads ~90MB, then cached)...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    index_records = load_or_build_index(embed_model)

    sample_questions = [
        "What does a zero-trust middleware gateway do in a RAG pipeline?",
        "What are the six phases of a SOC incident response workflow?",
        "Why is choosing top-k important during retrieval?",
    ]

    for q in sample_questions:
        answer_question(q, embed_model, index_records)

    print("\nDone. Add your own .txt files to documents/, delete "
          "vector_store.pkl to re-index, and edit `sample_questions` "
          "above to ask your own things.")
