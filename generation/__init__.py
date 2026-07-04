"""
Generation module for RedSea GPT.

Lightweight, always-importable symbols are exposed here directly. The full RAG
chain (``RedSeaGPT``) pulls in the vector-store / embedding stack, so it is
imported lazily to keep test and config imports cheap and dependency-light.
"""

from .llm_config import create_llm
from .prompts import create_rag_prompt

__all__ = [
    "create_llm",
    "create_rag_prompt",
    "create_rag_chain",
    "RedSeaGPT",
]


def __getattr__(name):
    # Lazy import for the heavy RAG chain (needs chromadb / embeddings).
    if name in ("create_rag_chain", "RedSeaGPT"):
        from .rag_chain import create_rag_chain, RedSeaGPT
        return {"create_rag_chain": create_rag_chain, "RedSeaGPT": RedSeaGPT}[name]
    raise AttributeError(f"module 'generation' has no attribute {name!r}")
