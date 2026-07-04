"""
Hybrid retrieval for RedSea GPT: dense (Chroma) + sparse (BM25), fused with
Reciprocal Rank Fusion (RRF).

Why hybrid? Dense retrieval captures semantic similarity (so "salty" matches
"saline"); BM25 captures exact terminology (species names, numbers like "40.6‰",
acronyms) that dense models can blur. Fusing their ranked lists with RRF
consistently beats either retriever alone — this is the standard finding across
MS MARCO and BEIR. RRF is parameter-light (just ``k`` ~ 60) and needs no
training, so it stays transparent and reproducible.

This module is deliberately self-contained and testable: it takes a vectorstore
+ a corpus of documents and returns fused, provenance-carrying chunks.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# RRF constant (Cormack et al. 2009). 60 is the canonical default; it makes the
# score of a rank-1 hit ~0.0165, so lower ranks decay smoothly without vanishing.
RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[str]],
    k: int = RRF_K,
) -> List[Tuple[str, float]]:
    """Fuse multiple ranked lists of ids via Reciprocal Rank Fusion.

    Args:
        ranked_lists: each inner sequence is a ranked list of *doc ids* (best first).
        k: RRF smoothing constant.

    Returns:
        ``(id, rrf_score)`` pairs sorted best-first.

    Formula (Cormack, Clarke & Buettcher, 2009):
        score(d) = sum_i  1 / (k + rank_i(d))
    """
    scores: Dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


class HybridRetriever:
    """Dense + sparse retrieval with RRF fusion.

    The sparse (BM25) index is built lazily from a snapshot of the corpus so the
    first query pays a one-time indexing cost and subsequent queries are fast.
    """

    def __init__(
        self,
        vectorstore,
        all_docs: Optional[List[Document]] = None,
        dense_k: int = 20,
        sparse_k: int = 20,
    ):
        self.vectorstore = vectorstore
        self.dense_k = dense_k
        self.sparse_k = sparse_k
        self._bm25 = None
        self._corpus: List[Document] = []
        self._doc_ids: List[str] = []
        if all_docs:
            self._index_bm25(all_docs)

    def _index_bm25(self, docs: List[Document]) -> None:
        """Tokenise and index the corpus for BM25. Assigns a stable id per doc."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:  # pragma: no cover
            logger.warning("rank-bm25 not installed; sparse retrieval disabled.")
            self._bm25 = None
            return
        self._corpus = list(docs)
        self._doc_ids = [str(i) for i in range(len(docs))]
        tokenized = [self._tokenize(d.page_content) for d in docs]
        self._bm25 = BM25Okapi(tokenized)
        logger.info("BM25 index built over %d chunks.", len(docs))

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return text.lower().split()

    def set_corpus(self, docs: List[Document]) -> None:
        """(Re)build the BM25 index from a full corpus snapshot."""
        self._index_bm25(docs)

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """Run dense + (optional) sparse retrieval, fuse with RRF, return top_k docs.

        Returns provenance-carrying Documents in fused order.
        """
        # --- dense retrieval (Chroma / vectorstore) ---
        dense_docs = self.vectorstore.similarity_search(query, k=self.dense_k)
        dense_id_by_doc = {id(d): str(i) for i, d in enumerate(dense_docs)}
        dense_ranked = [dense_id_by_doc[id(d)] for d in dense_docs]

        ranked_lists = [dense_ranked]
        doc_lookup: Dict[str, Document] = {dense_id_by_doc[id(d)]: d for d in dense_docs}

        # --- sparse retrieval (BM25) ---
        if self._bm25 is not None and self._corpus:
            tokens = self._tokenize(query)
            scores = self._bm25.get_scores(tokens)
            sparse_order = sorted(
                range(len(scores)), key=lambda i: scores[i], reverse=True
            )[: self.sparse_k]
            sparse_ranked = []
            for i in sparse_order:
                if scores[i] <= 0:
                    continue
                did = f"s{i}"
                doc_lookup[did] = self._corpus[i]
                sparse_ranked.append(did)
            if sparse_ranked:
                ranked_lists.append(sparse_ranked)

        fused = reciprocal_rank_fusion(ranked_lists)[:top_k]
        out: List[Document] = []
        for doc_id, _score in fused:
            d = doc_lookup.get(doc_id)
            if d is not None:
                out.append(d)
        return out

    def retrieve_multi(self, queries: List[str], top_k: int = 5) -> List[Document]:
        """Fuse across multiple query variants (sub-queries / HyDE) via RRF.

        Each variant produces its own fused dense+sparse list; the variant lists
        are then fused together by RRF as well. De-duplication by content key
        keeps the result set compact.
        """
        if not queries:
            return []
        per_variant_ranked: List[List[str]] = []
        lookup: Dict[str, Document] = {}
        for vq in queries:
            docs = self.retrieve(vq, top_k=max(top_k, self.dense_k))
            ids = []
            for i, d in enumerate(docs):
                key = f"{d.metadata.get('source','')}|{d.metadata.get('page','')}|{d.page_content[:120]}"
                lookup[key] = d
                ids.append(key)
            per_variant_ranked.append(ids)
        fused = reciprocal_rank_fusion(per_variant_ranked)[:top_k]
        return [lookup[k] for k, _ in fused if k in lookup]
