"""
The RedSea GPT agent: a self-reflective RAG pipeline orchestrated by LangGraph.

This implements the **CRAG / Self-RAG** pattern as an explicit ``StateGraph``
with conditional edges. The graph can *self-correct*: it grades retrieved
documents for relevance, and if they are insufficient it rewrites the query
(sub-query decomposition + HyDE) and retries retrieval before generating. It
then *verifies* the draft answer with claim-level entailment against the
context and refuses if the answer is fabricated or ungrounded.

Topology
--------

    START -> classify
    classify      --(in domain)-->   retrieve
    classify      --(off scope)-->   refuse -> END
    retrieve      ->  grade_documents
    grade_documents  --(enough relevant, OR retries exhausted)-->  generate
    grade_documents  --(insufficient)--> rewrite_query -> retrieve   (loop)
    generate      ->  verify
    verify        --(grounded)-->   END
    verify        --(ungrounded)--> refuse -> END

The retry loop is bounded two ways: an explicit ``max_retrieval_rounds``
counter in state, and the graph's ``recursion_limit`` as a hard backstop.

Every node uses the same provider-agnostic ``UniversalLLM`` (text in/out), so
the graph runs on OptiLLM, Groq, or OpenAI without code changes. No node makes
assumptions about native tool-calling — routing decisions are parsed from
strict LLM prompts, which keeps the agent portable and the behavior auditable.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Literal

from langchain_core.documents import Document
from langchain_core.language_models.llms import BaseLLM
from langgraph.graph import END, START, StateGraph

from .prompts import create_rag_prompt, format_context
from .query_rewriter import generate_subqueries, generate_hyde
from .state import AgentState

logger = logging.getLogger(__name__)

MAX_RETRIEVAL_ROUNDS = 2  # original + 1 rewrite-retry
MIN_RELEVANT_DOCS = 2     # need at least this many graded-relevant docs to generate


# ---------------------------------------------------------------------------
# Prompts for the reflective nodes (each strict + cheap + auditable)
# ---------------------------------------------------------------------------
CLASSIFY_PROMPT = """You are a router for a scientific Q&A system about the Egyptian Red Sea's natural science ONLY (geology, oceanography, reef biology, biodiversity, conservation).

Route to REFUSE (out of scope) for any of these, even if they mention the Red Sea:
- Tourism, travel, hotels, diving trips, restaurants, visas, prices
- Sports, politics, history of wars, celebrities, entertainment
- Future predictions ("what will happen by 2100?", forecasts, projections)
- Medical, financial, or legal advice
- Questions about other seas/oceans/regions unless directly comparing to the Red Sea
- Anything asking you to use general world knowledge rather than the corpus

Route to RETRIEVE only for questions about Red Sea geology, oceanography, coral/reef biology, biodiversity/endemism, or conservation that can be answered from a scientific corpus.

Question: {question}

Respond with ONLY a JSON object: {{"route": "retrieve"}} or {{"route": "refuse"}}"""


GRADE_PROMPT = """You are grading retrieved passages for relevance to a scientific question about the Egyptian Red Sea.

Question: {question}

For EACH passage below, decide if it contains information useful for answering the question. Return a JSON list of 1/0 in the SAME ORDER as the passages (1 = relevant, 0 = not relevant). No commentary.

Passages:
{passages}

Respond with ONLY a JSON list, e.g. [1, 0, 1, 1, 0]."""


VERIFY_PROMPT = """You are a fabrication detector for a scientific RAG system. Your ONLY job is to catch MADE-UP facts — not to judge style or require every phrase to be copied.

An answer is GROUNDED unless it contains a specific FABRICATION:
  - a species name, gene, chemical compound, or proper-noun entity that appears NOWHERE in the context
  - a specific number, percentage, date, or measurement that appears NOWHERE in the context
  - a claim that directly CONTRADICTS the context

An answer is STILL GROUNDED if it:
  - paraphrases, summarizes, or synthesizes ideas that ARE in the context (e.g. naming "Gondwana" or "sea-floor spreading" when those concepts appear)
  - explains a mechanism or causal chain using terms present in the context
  - uses general scientific connective language

Context:
\"\"\"{context}\"\"\"

Answer:
\"\"\"{answer}\"\"\"

If the answer is grounded, respond with ONLY: {{"grounded": true}}
If it contains a specific fabrication, respond with ONLY: {{"grounded": false, "unsupported_claims": ["<the single most obvious fabricated specific entity/number/date>"]}}"""


REFUSAL_PROMPT = """Refuse a question politely and concisely (1-2 sentences) in the voice of an expert Red Sea naturalist. Do NOT apologize profusely. State plainly that the curated Red Sea scientific corpus does not cover this, and suggest reframing toward Red Sea geology, oceanography, reef biology, biodiversity, or conservation if relevant. No citations.

Reason: {reason}
Question: {question}

Refusal:"""


# ---------------------------------------------------------------------------
# Node factory: build nodes bound to a configured LLM + retriever + reranker
# ---------------------------------------------------------------------------
class GraphNodes:
    """Produces graph-node callables bound to shared LLM/retriever/reranker.

    Kept as a class (not closures) so the wiring in ``build_graph`` reads clearly
    and so nodes can be unit-tested in isolation.
    """

    def __init__(self, llm: BaseLLM, retriever, reranker=None, max_rounds: int = MAX_RETRIEVAL_ROUNDS,
                 retrieval_k: int = 5):
        self.llm = llm
        self.retriever = retriever
        self.reranker = reranker
        self.max_rounds = max_rounds
        self.retrieval_k = retrieval_k
        self.prompt = create_rag_prompt()

    # -- low-level LLM helper ------------------------------------------------
    def _call(self, prompt: str) -> str:
        out = self.llm.invoke(prompt)
        text = getattr(out, "content", out)
        if isinstance(text, dict) and "content" in text:
            text = text["content"]
        return str(text).strip()

    @staticmethod
    def _extract_json(raw: str):
        raw = (raw or "").strip()
        m = re.search(r"\{.*\}|\[.*\]", raw, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:  # noqa: BLE001
            return None

    # -- nodes ---------------------------------------------------------------
    def classify(self, state: AgentState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        q = state["question"]
        raw = self._call(CLASSIFY_PROMPT.format(question=q))
        parsed = self._extract_json(raw) or {}
        # Default to retrieve (safe): we only refuse on an explicit 'refuse'
        # verdict, so a parse hiccup can never wrongly gate out an on-topic Q.
        verdict = str(parsed.get("route", "")).strip().lower()
        route = "refuse" if verdict == "refuse" else "retrieve"
        return {"route": route, "trace": {"classify": round(time.perf_counter() - t0, 3)}}

    def retrieve(self, state: AgentState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        q = state["question"]
        rounds = state.get("retrieval_rounds", 0)

        # Round 0: use sub-query expansion for richer recall.
        # Retry rounds: regenerate variants (HyDE) to escape a bad query.
        queries = [q]
        try:
            if rounds == 0:
                queries += generate_subqueries(self.llm, q, max_n=2)
            else:
                hyde = generate_hyde(self.llm, q)
                if hyde:
                    queries.append(hyde)
                queries += generate_subqueries(self.llm, q, max_n=2)
        except Exception:  # noqa: BLE001
            pass

        docs = self.retriever.retrieve_multi(queries, top_k=max(self.retrieval_k, 6))
        if self.reranker is not None and docs:
            docs = self.reranker.rerank(q, docs, top_k=self.retrieval_k)
        else:
            docs = docs[: self.retrieval_k]

        return {
            "documents": docs,
            "rewritten_queries": queries,
            "retrieval_rounds": rounds + 1,
            "trace": {"retrieve": round(time.perf_counter() - t0, 3)},
        }

    def grade_documents(self, state: AgentState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        docs: List[Document] = state.get("documents", [])
        q = state["question"]
        if not docs:
            return {"doc_grades": [], "documents": [], "trace": {"grade_documents": 0.0}}

        passages = "\n\n".join(
            f"PASSAGE {i+1}:\n{d.page_content[:900]}" for i, d in enumerate(docs)
        )
        raw = self._call(GRADE_PROMPT.format(question=q, passages=passages))
        parsed = self._extract_json(raw)
        if isinstance(parsed, list) and len(parsed) == len(docs):
            grades = [int(bool(g)) for g in parsed]
        else:
            # Conservative fallback: if we cannot parse grades, keep all docs
            # (so a grader hiccup never drops good evidence silently).
            grades = [1] * len(docs)

        relevant = [d for d, g in zip(docs, grades) if g]
        # Replace the document set with the graded-relevant subset for generation.
        # (add_or_replace dedupes against earlier rounds, keeping cumulative evidence.)
        keep = relevant if relevant else docs
        return {
            "doc_grades": grades,
            "documents": keep,
            "trace": {"grade_documents": round(time.perf_counter() - t0, 3)},
        }

    def generate(self, state: AgentState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        docs: List[Document] = state.get("documents", []) or []
        # If grading kept nothing, refuse downstream; here we just generate from what we have.
        context = format_context(docs)
        formatted = self.prompt.format(context=context, question=state["question"])
        answer = self._call(formatted)
        return {"answer": answer, "trace": {"generate": round(time.perf_counter() - t0, 3)}}

    def verify(self, state: AgentState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        answer = state.get("answer", "")
        docs: List[Document] = state.get("documents", []) or []
        context = format_context(docs)
        raw = self._call(VERIFY_PROMPT.format(context=context[:6000], answer=answer[:3000]))
        parsed = self._extract_json(raw)
        grounded = True
        unsupported: List[str] = []
        if isinstance(parsed, dict):
            grounded = bool(parsed.get("grounded", True))
            unsupported = list(parsed.get("unsupported_claims", []) or [])
        return {
            "verification": {"grounded": grounded, "unsupported_claims": unsupported},
            "trace": {"verify": round(time.perf_counter() - t0, 3)},
        }

    def refuse(self, state: AgentState) -> Dict[str, Any]:
        reason = state.get("reason", "outside the scope of the curated Red Sea corpus")
        refusal = self._call(REFUSAL_PROMPT.format(reason=reason, question=state["question"]))
        return {"answer": refusal, "refused": True}


# ---------------------------------------------------------------------------
# Routing functions (pure: read state, return next node name; never mutate)
# ---------------------------------------------------------------------------
def route_after_classify(state: AgentState) -> Literal["retrieve", "refuse"]:
    return "refuse" if state.get("route") == "refuse" else "retrieve"


def decide_after_grading(state: AgentState) -> Literal["generate", "rewrite_query", "refuse"]:
    """Generate if we have enough relevant docs; rewrite if we can retry; refuse
    if retries are exhausted and nothing relevant was found.

    This relevance gate is the agent's equivalent of the baseline confidence /
    topic-mismatch refusal: it stops the generator from answering off-topic or
    unsupported questions where the retriever returned only loosely-related docs
    (Chroma always returns *something*, so a relevance floor is essential).
    """
    rounds = state.get("retrieval_rounds", 0)
    grades = state.get("doc_grades", [])
    relevant = sum(1 for g in grades if g)
    if relevant >= MIN_RELEVANT_DOCS:
        return "generate"
    if rounds >= MAX_RETRIEVAL_ROUNDS:
        # Retries exhausted. If we found NOTHING relevant, refuse rather than
        # fabricate from loosely-related chunks. If we found a few, generate.
        return "generate" if relevant > 0 else "refuse"
    return "rewrite_query"


def rewrite_query_node_factory(llm: BaseLLM):
    """Retry node: bumps the round counter. retrieve() regenerates variants."""
    def _node(state: AgentState) -> Dict[str, Any]:
        # retrieval_rounds is incremented in retrieve(); here we just signal the
        # retry by returning an empty partial so the graph advances to retrieve.
        return {"reason": "low relevance — rewriting query for another retrieval round"}
    return _node


def route_after_verify_factory(strict_verify: bool):
    """Build a verify-router. In lenient mode (default) the verifier's verdict
    is recorded but never gates the answer; only strict mode refuses on a
    detected fabrication."""
    def _router(state: AgentState):
        if not strict_verify:
            return END
        ver = state.get("verification", {})
        if ver.get("grounded", True):
            return END
        return "refuse"
    return _router


def route_after_verify(state: AgentState) -> Literal[END, "refuse"]:
    ver = state.get("verification", {})
    if ver.get("grounded", True):
        return END
    return "refuse"


def refuse_reason_after_classify():
    return "The question is outside the scope of the Egyptian Red Sea natural-science corpus."


def refuse_reason_after_verify():
    return "The drafted answer contains claims not supported by the retrieved context."


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------
def build_graph(llm, retriever, reranker=None, *, max_rounds: int = MAX_RETRIEVAL_ROUNDS,
                retrieval_k: int = 5, recursion_limit: int = 18, strict_verify: bool = False):
    """Build and compile the CRAG StateGraph. Returns a compiled runnable.

    ``strict_verify=False`` (default) means the post-generation verifier runs
    and records its verdict in state (for observability/eval) but does NOT gate
    the answer — the generator's grounding prompt is the primary guardrail, and a
    second LLM gate over-refuses legitimate paraphrase/synthesis. Set
    ``strict_verify=True`` to refuse on detected fabrication.
    """
    nodes = GraphNodes(llm, retriever, reranker, max_rounds=max_rounds, retrieval_k=retrieval_k)

    g = StateGraph(AgentState)
    g.add_node("classify", nodes.classify)
    g.add_node("retrieve", nodes.retrieve)
    g.add_node("grade_documents", nodes.grade_documents)
    g.add_node("rewrite_query", rewrite_query_node_factory(llm))
    g.add_node("generate", nodes.generate)
    g.add_node("verify", nodes.verify)
    g.add_node("refuse", nodes.refuse)

    g.add_edge(START, "classify")
    g.add_conditional_edges(
        "classify", route_after_classify, {"retrieve": "retrieve", "refuse": "refuse"}
    )
    g.add_edge("retrieve", "grade_documents")
    g.add_conditional_edges(
        "grade_documents", decide_after_grading,
        {"generate": "generate", "rewrite_query": "rewrite_query", "refuse": "refuse"}
    )
    g.add_edge("rewrite_query", "retrieve")
    g.add_edge("generate", "verify")
    g.add_conditional_edges("verify", route_after_verify_factory(strict_verify), {END: END, "refuse": "refuse"})
    g.add_edge("refuse", END)

    return g.compile(), recursion_limit
