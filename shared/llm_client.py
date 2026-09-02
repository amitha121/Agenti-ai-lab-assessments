"""
llm_client.py
--------------
Tiny shared wrapper around the Groq API so all tasks call the LLM the
same way. Groq is used because it gives a genuinely FREE API key (no
credit card) with a generous free tier, and is OpenAI-SDK compatible.

HOW TO GET YOUR FREE KEY (~1 minute):
  1. Go to https://console.groq.com
  2. Sign up / log in (Google or GitHub login works)
  3. Left sidebar -> API Keys -> Create API Key
  4. Copy the key

HOW TO USE THE KEY:
  Copy .env.example to .env and paste your key in, e.g.:
      GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
"""

import os
from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"


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
