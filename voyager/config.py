"""Model configuration.

All five topologies must share the SAME model so that performance differences
are attributable to topology, not to the underlying LLM. Change the model in
one place here.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

# Same model across every topology — this is a methodological requirement.
DEFAULT_MODEL = "qwen/qwen3-32b"


def get_llm(model: str = DEFAULT_MODEL, temperature: float = 0.0) -> ChatGroq:
    """Return the shared chat model. Temperature 0 for reproducible evals."""
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY not set. Copy .env.example to .env and add your free "
            "Groq key from https://console.groq.com/keys"
        )
    return ChatGroq(model=model, temperature=temperature)
