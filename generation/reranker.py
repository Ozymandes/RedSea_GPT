"""
Cross-encoder re-ranking for RedSea GPT.

A bi-encoder (our ``all-mpnet`` embedder) produces query and doc embeddings
*separately* and compares by cosine similarity — fast, but it never sees the
query and document together. A **cross-encoder** feeds the query and each
candidate document through a transformer *jointly*, producing a much sharper
relevance score. Re-ranking the top-N fused candidates with a cross-encoder is
one of the most reliable precision lifts in modern RAG.

This module loads ``BAAI/bge-reranker-base`` (a 278MB, 6-layer cross-encoder
that scores English Q/P pairs well) on first use, and **degrades gracefully**:
if the model cannot be downloaded/loaded (offline, no disk, slow connection),
the reranker becomes identity and retrieval simply keeps the fused order.
That keeps the system runnable from a clean clone without a mandatory download.

Set ``RERANKER_ENABLED=0`` to disable cross-encoder re-ranking entirely (useful
for ablations / A-B comparisons).
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import List, Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "BAAI/bge-reranker-base"


def _is_disabled() -> bool:
    return os.getenv("RERANKER_ENABLED", "1").strip() in ("0", "false", "False", "no")


@lru_cache(maxsize=1)
def _load_cross_encoder(model_name: str):
    """Load the CrossEncoder once, process-wide. Returns None if unavailable."""
    if _is_disabled():
        logger.info("Reranker disabled via RERANKER_ENABLED=0; skipping.")
        return None
    try:
        from sentence_transformers import CrossEncoder  # noqa
    except Exception as exc:  # pragma: no cover
        logger.warning("sentence-transformers not importable; reranking disabled (%s)", exc)
        return None
    try:
        # local_files_only is governed by HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE
        # in the environment; if offline and the model is absent, this raises.
        ce = CrossEncoder(model_name)
        logger.info("Loaded cross-encoder reranker '%s'.", model_name)
        return ce
    except Exception as exc:  # pragma: no cover
        logger.warning(
            "Could not load cross-encoder '%s' (reranking disabled). "
            "System still works with fused retrieval only. Reason: %s",
            model_name, exc,
        )
        return None


class Reranker:
    """Re-rank Documents against a query using a cross-encoder, with fallback."""

    def __init__(self, model_name: str = _DEFAULT_MODEL):
        self.model_name = model_name

    def rerank(
        self,
        query: str,
        docs: List[Document],
        top_k: Optional[int] = None,
    ) -> List[Document]:
        """Return docs re-ranked by cross-encoder score. Falls back to input order."""
        if not docs:
            return []
        ce = _load_cross_encoder(self.model_name)
        if ce is None:
            # Graceful fallback: keep fused order, just truncate.
            return docs[:top_k] if top_k else docs

        pairs = [(query, d.page_content) for d in docs]
        try:
            scores = ce.predict(pairs)
        except Exception as exc:  # pragma: no cover
            logger.warning("Cross-encoder predict failed; using fused order. (%s)", exc)
            return docs[:top_k] if top_k else docs

        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        result = [d for d, _ in ranked]
        return result[:top_k] if top_k else result
