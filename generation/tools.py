"""
LangChain tools wrapping Red Sea retrieval operations.

These are genuine ``@tool``-decorated functions (each has a name, description,
and typed argschema), so the retrieval layer is a first-class *tool* that an
agent or graph can invoke — not a hidden method call. The CRAG graph (see
``graph.py``) is built on top of these tools, and an agentic ReAct loop could
bind the same tools to an LLM.

Exposing retrieval as tools (rather than a monolithic retriever object) is what
makes the "tool use" claim literally true and inspectable in the code.
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_core.documents import Document
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


def _make_retrieval_tools(hybrid_retriever, reranker=None):
    """Build tool functions bound to a configured retriever + optional reranker.

    Factory pattern because LangChain ``@tool`` closures need to capture the
    concrete retriever instance. Returns a dict of tool callables keyed by name.
    """

    @tool("redsea_hybrid_search")
    def redsea_hybrid_search(query: str, top_k: int = 5) -> str:
        """Search the Egyptian Red Sea scientific corpus for passages relevant to a question.

        Uses hybrid retrieval (dense embeddings + BM25 keyword, fused via Reciprocal
        Rank Fusion) and optional cross-encoder re-ranking. Returns numbered passages
        with source filename and page, ready to cite as [1], [2], ...

        Args:
            query: A natural-language question or search phrase about the Red Sea.
            top_k: Number of passages to return (default 5).
        """
        docs: list[Document] = hybrid_retriever.retrieve(query, top_k=max(top_k, 5))
        if reranker is not None:
            docs = reranker.rerank(query, docs, top_k=top_k)
        return _format_as_context(docs)

    @tool("redsea_semantic_search")
    def redsea_semantic_search(query: str, top_k: int = 5) -> str:
        """Semantic-only search of the Red Sea corpus (dense vector similarity, no reranking).

        Use this when you want fast, un-ranked semantic neighbours. For best
        precision prefer redsea_hybrid_search.

        Args:
            query: A natural-language question or search phrase about the Red Sea.
            top_k: Number of passages to return (default 5).
        """
        docs = hybrid_retriever.retrieve(query, top_k=top_k)
        return _format_as_context(docs)

    return {
        "redsea_hybrid_search": redsea_hybrid_search,
        "redsea_semantic_search": redsea_semantic_search,
    }


def _format_as_context(docs: list[Document]) -> str:
    """Format retrieved docs as numbered, provenance-carrying passages."""
    from .utils import clean_source_path

    parts = []
    for i, d in enumerate(docs, start=1):
        src = clean_source_path(d.metadata.get("source", "Unknown"))
        page = d.metadata.get("page", "?")
        parts.append(f"[{i}] (Source: {src}, page {page})\n{d.page_content.strip()}")
    return "\n\n---\n\n".join(parts) if parts else "(no passages found)"
