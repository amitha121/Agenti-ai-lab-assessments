"""
llm_client.py
--------------
Tiny shared wrapper around the Groq API so all three tasks call the
LLM the same way. Groq is used because it gives a genuinely FREE API
key (no credit card) with a generous free tier, and is OpenAI-SDK
compatible so the code stays simple.

HOW TO GET YOUR FREE KEY (takes ~1 minute):
  1. Go to https://console.groq.com
  2. Sign up / log in (Google or GitHub login works)
  3. Left sidebar -> "API Keys" -> "Create API Key"
  4. Copy the key

HOW TO USE THE KEY (pick one):
  Option A - .env file (easiest):
      Copy .env.example to .env and paste your key in, e.g.:
          GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx

  Option B - environment variable:
      Windows (PowerShell):  $env:GROQ_API_KEY="gsk_xxxx..."
      Mac/Linux:              export GROQ_API_KEY="gsk_xxxx..."

No other file needs to be touched — every script imports get() from here.
"""

import os
from openai import OpenAI

# Optional convenience: load a .env file if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Free-tier friendly, strong general-purpose model on Groq.
# Swap to "llama-3.1-8b-instant" for an even faster/cheaper option.
DEFAULT_MODEL = "llama-3.3-70b-versatile"


def get_client() -> OpenAI:
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY not set. Get a free key at https://console.groq.com "
            "and add it to a .env file or your environment (see README.md)."
        )
    return OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)


def chat(prompt: str, system: str | None = None, model: str = DEFAULT_MODEL,
          temperature: float = 0.2, max_tokens: int = 1024) -> str:
    """Send one prompt to the LLM and return the plain-text reply."""
    client = get_client()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()
