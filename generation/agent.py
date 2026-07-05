"""
``RedSeaAgent`` — the agentic RAG interface, same contract as ``RedSeaGPT``.

This wraps the LangGraph CRAG graph (``graph.py``) behind the same
``.query(question, return_source_docs=...)`` API the rest of the codebase
expects, so the evaluation runner and the A/B harness can swap the baseline
``RedSeaGPT`` for the agentic ``RedSeaAgent`` with one constructor call.

It assembles the pieces the graph needs:
  * a provider-agnostic LLM (``create_llm``),
  * a hybrid retriever (dense + BM25 + RRF),
  * an optional cross-encoder reranker,
  * retrieval exposed as LangChain ``@tool`` functions (``tools.py``).

The agent keeps the LLM-driven self-correction loop (grade -> rewrite -> retry)
and the post-generation grounding verifier, both of which the baseline lacks.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from .graph import build_graph
from .llm_config import create_llm
from .memory import ConversationMemory, resolve_query_with_history
from .reranker import Reranker
from .retrievers import HybridRetriever
from .tools import _make_retrieval_tools
from .utils import clean_source_path

logger = logging.getLogger(__name__)


class RedSeaAgent:
    """Agentic Red Sea naturalist orchestrated by a LangGraph CRAG graph.

    Same ``.query()`` contract as ``RedSeaGPT`` (baseline) so the two can be
    A/B tested head-to-head.
    """

    def __init__(
        self,
        vectordb_path: str = "chroma_redsea",
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
        llm_config: Optional[Dict[str, Any]] = None,
        retrieval_k: int = 5,
        use_reranker: bool = True,
        max_retrieval_rounds: int = 2,
        recursion_limit: int = 18,
        strict_verify: bool = False,
    ):
        self.retrieval_k = retrieval_k

        # Embeddings + vectorstore
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vectorstore = Chroma(
            persist_directory=vectordb_path, embedding_function=self.embeddings
        )

        # LLM (provider-agnostic)
        self.llm = create_llm(**(llm_config or {}))

        # Hybrid retriever: build the BM25 corpus from a snapshot of the collection.
        self.hybrid = HybridRetriever(self.vectorstore, dense_k=max(retrieval_k * 4, 20))
        self._populate_bm25_corpus()

        # Optional cross-encoder reranker (degrades to identity if unavailable)
        self.reranker = Reranker() if use_reranker else None

        # Expose retrieval as genuine LangChain tools (inspectable, reusable)
        self.tools = _make_retrieval_tools(self.hybrid, self.reranker)

        # Build + compile the LangGraph CRAG graph
        self.graph_app, self._recursion_limit = build_graph(
            self.llm,
            self.hybrid,
            self.reranker,
            max_rounds=max_retrieval_rounds,
            retrieval_k=retrieval_k,
            recursion_limit=recursion_limit,
            strict_verify=strict_verify,
        )

    def _populate_bm25_corpus(self) -> None:
        """Snapshot the Chroma collection into an in-memory BM25 index.

        The collection is ~5.5k chunks; this is a one-time cost (a second or
        two) and makes subsequent sparse queries very fast. We pull all docs in
        pages to respect any collection-size limits.
        """
        coll = self.vectorstore._collection
        docs: List[Document] = []
        batch = 2000
        offset = 0
        total = coll.count()
        while offset < total:
            res = coll.get(include=["documents", "metadatas"], limit=batch, offset=offset)
            for txt, meta in zip(res.get("documents", []), res.get("metadatas", [])):
                docs.append(Document(page_content=txt, metadata=meta or {}))
            offset += batch
        self.hybrid.set_corpus(docs)
        logger.info("BM25 corpus populated: %d chunks.", len(docs))

    def query(self, question: str, return_source_docs: bool = False,
              memory: Optional[ConversationMemory] = None):
        """Run the agentic graph for a question.

        Returns the same dict shape as ``RedSeaGPT.query(return_source_docs=True)``
        so the two are interchangeable in evaluation harnesses.

        Multiturn: when ``memory`` is non-empty, the latest message is rewritten
        into a self-contained question (so the graph's classify/retrieve nodes
        see the real intent) before invocation. Falls back to the raw question
        on any failure.
        """
        history_block = ""
        resolved_question = question
        if memory is not None and not memory.is_empty:
            resolved_question = resolve_query_with_history(self.llm, question, memory)
            history_block = memory.format_for_prompt()
        try:
            final = self.graph_app.invoke(
                {"question": resolved_question, "history": history_block},
                config={"recursion_limit": self._recursion_limit},
            )
        except Exception as exc:  # noqa: BLE001
            # Never crash the eval loop; surface as a structured error.
            logger.exception("Agent graph failed for question: %s", question)
            if return_source_docs:
                return {
                    "answer": f"ERROR: agent failed: {exc.__class__.__name__}",
                    "sources": [],
                    "retrieved_chunks": [],
                    "question": question,
                    "resolved_question": resolved_question,
                    "confidence": 0.0,
                    "refusal": False,
                    "retrieval_method": "graph",
                    "num_sources": 0,
                    "error": str(exc),
                }
            return f"ERROR: agent failed: {exc.__class__.__name__}"

        answer = final.get("answer", "") or ""
        refused = bool(final.get("refused", False))
        docs: List[Document] = final.get("documents", []) or []

        sources = self._format_sources_list(docs)
        chunks = [
            {
                "citation_id": i,
                "source": clean_source_path(d.metadata.get("source", "Unknown")),
                "page": d.metadata.get("page"),
                "page_content": d.page_content,
            }
            for i, d in enumerate(docs, start=1)
        ]

        if return_source_docs:
            return {
                "answer": answer,
                "sources": sources,
                "retrieved_chunks": chunks,
                "question": question,
                "resolved_question": resolved_question,
                "confidence": _avg_relevance(final),
                "refusal": refused,
                "retrieval_method": "graph_crag",
                "num_sources": len(docs),
                "verification": final.get("verification", {}),
                "route": final.get("route"),
                "retrieval_rounds": final.get("retrieval_rounds", 0),
                "trace": final.get("trace", {}),
            }
        return answer

    @staticmethod
    def _format_sources_list(docs: List[Document]) -> List[Dict[str, Any]]:
        sources = []
        for i, doc in enumerate(docs, start=1):
            sources.append({
                "citation_id": i,
                "source": clean_source_path(doc.metadata.get("source", "Unknown")),
                "page": doc.metadata.get("page", "Unknown"),
                "content": doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""),
            })
        return sources


def _avg_relevance(final: Dict[str, Any]) -> float:
    """Proxy confidence: fraction of retrieved docs graded relevant."""
    grades = final.get("doc_grades") or []
    if not grades:
        return 0.0
    return round(sum(1 for g in grades if g) / len(grades), 3)
