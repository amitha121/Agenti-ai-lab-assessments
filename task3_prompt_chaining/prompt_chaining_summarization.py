"""
Task 3: Prompt Chaining for Summarization
============================================
Demonstrates a multi-step prompt pipeline, where each step's OUTPUT
becomes the next step's INPUT. This is the core idea of "prompt
chaining": break one hard task into several small, reliable LLM calls
instead of one big unreliable one.

Pipeline (4 chained steps):
  Step 1 (MAP)      : Split the article into chunks, summarize each
                        chunk independently into 2-3 bullet points.
  Step 2 (REDUCE)    : Combine all chunk-level bullets into a single
                        cohesive draft summary.
  Step 3 (CRITIQUE)  : Ask the LLM to critique the draft -- what's
                        missing, unclear, or redundant.
  Step 4 (REFINE)    : Produce the final polished summary, incorporating
                        the critique feedback.

Every intermediate output is printed, so you can see exactly how the
summary evolves at each step of the chain -- useful for a report/screenshots.

Run:
    python prompt_chaining_summarization.py
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "shared"))
from llm_client import chat  # noqa: E402

ARTICLE_PATH = os.path.join(os.path.dirname(__file__), "sample_article.txt")
CHUNK_WORDS = 180


def chunk_text(text, chunk_words=CHUNK_WORDS):
    words = text.split()
    return [
        " ".join(words[i:i + chunk_words])
        for i in range(0, len(words), chunk_words)
    ]


# ---------------------------------------------------------------------------
# Step 1: MAP - summarize each chunk independently
# ---------------------------------------------------------------------------
def step1_map_summarize(chunks):
    print("\n[Step 1/4] Summarizing each chunk independently (MAP)...")
    bullet_sets = []
    for i, chunk in enumerate(chunks, 1):
        prompt = (
            f"Summarize the following text into 2-3 concise bullet points, "
            f"capturing only the key facts:\n\n{chunk}"
        )
        bullets = chat(prompt, temperature=0.2)
        bullet_sets.append(bullets)
        print(f"\n  -- Chunk {i} bullets --\n{bullets}")
    return bullet_sets


# ---------------------------------------------------------------------------
# Step 2: REDUCE - combine bullet sets into one draft summary
# ---------------------------------------------------------------------------
def step2_reduce_combine(bullet_sets):
    print("\n[Step 2/4] Combining chunk summaries into one draft (REDUCE)...")
    combined_bullets = "\n\n".join(
        f"From chunk {i+1}:\n{b}" for i, b in enumerate(bullet_sets)
    )
    prompt = (
        "The following are bullet-point summaries of consecutive sections "
        "of one article. Combine them into a single, cohesive summary "
        "paragraph (5-7 sentences) that flows naturally and avoids "
        f"repetition:\n\n{combined_bullets}"
    )
    draft = chat(prompt, temperature=0.3)
    print(f"\n  -- Draft summary --\n{draft}")
    return draft


# ---------------------------------------------------------------------------
# Step 3: CRITIQUE - self-critique the draft
# ---------------------------------------------------------------------------
def step3_critique(draft, original_article):
    print("\n[Step 3/4] Critiquing the draft summary (CRITIQUE)...")
    prompt = f"""Original article:
{original_article}

Draft summary:
{draft}

Critique the draft summary against the original article. List up to 3
specific issues: anything important that's missing, anything unclear,
or anything redundant. Be concise and specific."""
    critique = chat(prompt, temperature=0.3)
    print(f"\n  -- Critique --\n{critique}")
    return critique


# ---------------------------------------------------------------------------
# Step 4: REFINE - produce the final summary using the critique
# ---------------------------------------------------------------------------
def step4_refine(draft, critique):
    print("\n[Step 4/4] Producing final polished summary (REFINE)...")
    prompt = f"""Draft summary:
{draft}

Critique of the draft:
{critique}

Rewrite the draft summary to address the critique. Output ONLY the final
polished summary (5-8 sentences), nothing else."""
    final = chat(prompt, temperature=0.2)
    print(f"\n  -- FINAL SUMMARY --\n{final}")
    return final


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    with open(ARTICLE_PATH, "r", encoding="utf-8") as f:
        article = f.read()

    print(f"Loaded article ({len(article.split())} words) from {ARTICLE_PATH}")

    chunks = chunk_text(article)
    print(f"Split into {len(chunks)} chunk(s) of ~{CHUNK_WORDS} words each")

    bullet_sets = step1_map_summarize(chunks)
    draft = step2_reduce_combine(bullet_sets)
    critique = step3_critique(draft, article)
    final_summary = step4_refine(draft, critique)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(final_summary)
