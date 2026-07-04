"""
Query rewriting for better retrieval: sub-query decomposition + HyDE.

Two complementary transformations improve recall before retrieval:

1. **Sub-query decomposition** — split a multi-part or vague question into a few
   focused sub-queries. ("Why are Red Sea corals heat-tolerant and where do they
   live?" -> ["mechanisms of heat tolerance in Red Sea corals",
              "distribution of heat-tolerant corals in the Red Sea"])

2. **HyDE (Hypothetical Document Embeddings)** — generate a *hypothetical
   answer* and retrieve against it. A plausible-sounding answer paragraph is a
   closer neighbour to the real source text than the terse question is. This is
   the Gao et al. 2023 trick and consistently helps dense retrieval.

Both are LLM-driven but cheap (small prompts). Malformed output degrades
gracefully: if parsing fails we fall back to the original question, so a
rewriter failure can never break the pipeline.
"""

from __future__ import annotations

import json
import re
from typing import List

from langchain_core.language_models.llms import BaseLLM


SUBQUERY_PROMPT = """You are a search query optimizer for a scientific library about the Egyptian Red Sea.
Break the user's question into 2-3 focused sub-queries that, each on its own, would retrieve the relevant evidence.
Keep them as search phrases, not full sentences. Do not answer the question.

Question: {question}

Respond with ONLY a JSON object, no commentary:
{{"subqueries": ["...", "...", "..."]}}"""


HYDE_PROMPT = """Write a short (3-4 sentence) plausible scientific paragraph that would answer this question about the Egyptian Red Sea reef ecosystem. It will be used only as a search query to retrieve the real evidence, so include the key technical terms and entities the answer would mention. Do not add citations.

Question: {question}

Paragraph:"""


def _extract_json(raw: str):
    """Best-effort JSON object extraction from an LLM response."""
    raw = raw.strip()
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        return None


def generate_subqueries(llm: BaseLLM, question: str, max_n: int = 3) -> List[str]:
    """Decompose a question into focused sub-queries (graceful fallback)."""
    try:
        out = llm.invoke(SUBQUERY_PROMPT.format(question=question))
        text = getattr(out, "content", out)
        if isinstance(text, dict) and "content" in text:
            text = text["content"]
        parsed = _extract_json(str(text)) or {}
        subs = [s.strip() for s in parsed.get("subqueries", []) if isinstance(s, str) and s.strip()]
        return subs[:max_n]
    except Exception:  # noqa: BLE001
        return []


def generate_hyde(llm: BaseLLM, question: str) -> str:
    """Generate a HyDE hypothetical document (graceful fallback)."""
    try:
        out = llm.invoke(HYDE_PROMPT.format(question=question))
        text = getattr(out, "content", out)
        if isinstance(text, dict) and "content" in text:
            text = text["content"]
        text = str(text).strip()
        return text
    except Exception:  # noqa: BLE001
        return ""


def rewrite_for_retrieval(llm: BaseLLM, question: str) -> List[str]:
    """Return the full set of retrieval variants: original + sub-queries + HyDE.

    The original question is always included so a weak/failed rewriter never
    reduces retrieval quality below baseline.
    """
    variants = [question]
    variants.extend(generate_subqueries(llm, question))
    hyde = generate_hyde(llm, question)
    if hyde:
        variants.append(hyde)
    return variants
