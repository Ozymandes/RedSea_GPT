"""
Tests for the agentic RAG components (LangGraph CRAG, hybrid retrieval, reranker,
query rewriting, A/B comparison).

These run WITHOUT network access and WITHOUT the embedding model by testing the
pure logic (RRF fusion, graph state reducers, reranker fallback, faithfulness
claim extraction, A/B pairing). The heavy end-to-end path is exercised manually
via the eval runner.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion (pure, no deps)
# ---------------------------------------------------------------------------
def test_rrf_basic_agreement_wins():
    from generation.retrievers import reciprocal_rank_fusion
    # doc A: rank1 + rank2 -> should beat doc C: rank3 only
    fused = reciprocal_rank_fusion([["A", "B", "C"], ["B", "A", "D"]])
    ids = [d for d, _ in fused]
    assert ids[0] in ("A", "B"), "agreement docs should rank first"
    assert "D" in ids[-1], "single-list tail doc should rank last"


def test_rrf_single_list_preserves_order():
    from generation.retrievers import reciprocal_rank_fusion
    fused = reciprocal_rank_fusion([["x", "y", "z"]])
    assert [d for d, _ in fused] == ["x", "y", "z"]


def test_rrf_empty():
    from generation.retrievers import reciprocal_rank_fusion
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[]]) == []


def test_rrf_constant_matches_cormack():
    """score(d at rank1 in one list) = 1/(60+1)."""
    from generation.retrievers import reciprocal_rank_fusion, RRF_K
    fused = dict(reciprocal_rank_fusion([["only"]]))
    assert abs(fused["only"] - 1.0 / (RRF_K + 1)) < 1e-9


# ---------------------------------------------------------------------------
# Graph state reducer
# ---------------------------------------------------------------------------
def test_add_or_replace_dedupes():
    from generation.state import add_or_replace
    from langchain_core.documents import Document
    d1 = Document(page_content="alpha beta", metadata={"source": "s.pdf", "page": 1})
    d1b = Document(page_content="alpha beta", metadata={"source": "s.pdf", "page": 1})
    d2 = Document(page_content="gamma delta", metadata={"source": "s.pdf", "page": 2})
    merged = add_or_replace([d1], [d1b, d2])
    assert len(merged) == 2, "dedup by (source,page,content) should drop the duplicate"


def test_add_or_replace_none_right():
    from generation.state import add_or_replace
    from langchain_core.documents import Document
    d = Document(page_content="x", metadata={})
    assert add_or_replace([d], None) == [d]


# ---------------------------------------------------------------------------
# Reranker graceful fallback
# ---------------------------------------------------------------------------
def test_reranker_fallback_keeps_order(monkeypatch):
    from generation import reranker as rmod
    from langchain_core.documents import Document
    # force the cross-encoder to be unavailable
    monkeypatch.setenv("RERANKER_ENABLED", "0")
    rmod._load_cross_encoder.cache_clear()
    rr = rmod.Reranker()
    docs = [Document(page_content="a"), Document(page_content="b")]
    out = rr.rerank("query", docs, top_k=2)
    assert out == docs, "disabled reranker must fall back to input order"


def test_reranker_empty():
    from generation import reranker as rmod
    rmod._load_cross_encoder.cache_clear()
    rr = rmod.Reranker()
    assert rr.rerank("q", [], top_k=5) == []


# ---------------------------------------------------------------------------
# Tools are real @tool objects with a name + schema
# ---------------------------------------------------------------------------
def test_retrieval_tools_are_langchain_tools():
    from generation.tools import _make_retrieval_tools

    class FakeRetriever:
        def retrieve(self, q, top_k=5):
            from langchain_core.documents import Document
            return [Document(page_content="snippet " + q, metadata={"source": "a.pdf", "page": 1})]

    tools = _make_retrieval_tools(FakeRetriever())
    assert set(tools.keys()) == {"redsea_hybrid_search", "redsea_semantic_search"}
    for name, t in tools.items():
        # LangChain BaseTool
        assert hasattr(t, "name"), f"{name} must be a LangChain tool"
        assert t.name == name
        assert hasattr(t, "description")


# ---------------------------------------------------------------------------
# Faithfulness metric: claim extraction + scoring (mocked LLM)
# ---------------------------------------------------------------------------
class _MockLLM:
    """Returns scripted responses for claim extraction / judging."""
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def invoke(self, prompt, **kw):
        self.calls.append(prompt)
        if not self._responses:
            return ""
        resp = self._responses.pop(0)
        class _Out:
            content = resp
        return _Out()


def test_faithfulness_llm_supported(monkeypatch):
    from evaluation.metrics_v2 import evaluate_faithfulness_llm
    llm = _MockLLM([
        '{"claims": ["The Red Sea is salty", "It lies between Africa and Arabia"]}',
        '{"verdicts": [{"claim":"The Red Sea is salty","verdict":"supported","evidence":"salty"},'
        '{"claim":"It lies between Africa and Arabia","verdict":"supported","evidence":"between"}]}',
    ])
    res = evaluate_faithfulness_llm(llm, "Why is it salty?", "The Red Sea is salty. It lies between Africa and Arabia.", "context...")
    assert res["faithfulness"] == 1.0
    assert res["supported"] == 2
    assert res["method"] == "llm_claim_entailment"


def test_faithfulness_llm_partial(monkeypatch):
    from evaluation.metrics_v2 import evaluate_faithfulness_llm
    llm = _MockLLM([
        '{"claims": ["claim A", "claim B"]}',
        '{"verdicts": [{"claim":"claim A","verdict":"supported","evidence":""},'
        '{"claim":"claim B","verdict":"unsupported","evidence":""}]}',
    ])
    res = evaluate_faithfulness_llm(llm, "q", "claim A. claim B.", "ctx")
    assert res["faithfulness"] == 0.5
    assert res["supported"] == 1 and res["unsupported"] == 1


def test_faithfulness_llm_no_claims():
    from evaluation.metrics_v2 import evaluate_faithfulness_llm
    llm = _MockLLM(['{"claims": []}'])
    res = evaluate_faithfulness_llm(llm, "q", "some answer", "ctx")
    assert res["faithfulness"] is None  # fail closed, never silently 1.0


# ---------------------------------------------------------------------------
# A/B comparison pairing logic
# ---------------------------------------------------------------------------
def test_ab_paired_verdicts():
    from evaluation.run_ab_eval import build_comparison
    q = [{"id": "q1", "group": "g", "question": "what?", "expected_behavior": "answer"}]
    baseline = [{"id": "q1", "passed": False, "latency_ms": 1000, "expected_behavior": "answer",
                 "actually_refused": False, "group": "g", "category": "c", "question": "what?", "num_sources": 3, "metrics": {}}]
    agent = [{"id": "q1", "passed": True, "latency_ms": 2000, "expected_behavior": "answer",
              "actually_refused": False, "group": "g", "category": "c", "question": "what?", "num_sources": 4, "metrics": {}}]
    comp = build_comparison(baseline, agent, q)
    assert comp["paired"]["agent_only_pass"] == 1
    assert comp["paired"]["baseline_only_pass"] == 0
    assert comp["per_question"][0]["verdict"] == "agent_win"
    assert comp["headline"]["pass_rate_delta"] > 0


def test_sign_test_low_power_flag():
    from evaluation.run_ab_eval import build_comparison
    q = [{"id": f"q{i}", "group": "g", "question": "?", "expected_behavior": "answer"} for i in range(20)]
    baseline = [{"id": f"q{i}", "passed": True, "latency_ms": 1, "expected_behavior": "answer",
                 "actually_refused": False, "group": "g", "category": "c", "question": "?", "num_sources": 1, "metrics": {}} for i in range(20)]
    agent = [{"id": f"q{i}", "passed": True, "latency_ms": 1, "expected_behavior": "answer",
              "actually_refused": False, "group": "g", "category": "c", "question": "?", "num_sources": 1, "metrics": {}} for i in range(20)]
    # no discordant pairs -> low power caveat True
    comp = build_comparison(baseline, agent, q)
    assert comp["paired"]["discordant_pairs"] == 0
    assert comp["paired"]["low_power_caveat"] is True


# ---------------------------------------------------------------------------
# Query rewriter graceful fallback
# ---------------------------------------------------------------------------
def test_query_rewriter_fallback_returns_original():
    from generation.query_rewriter import rewrite_for_retrieval
    llm = _MockLLM([])  # invoke returns "" -> no subqueries/hyde parsed
    variants = rewrite_for_retrieval(llm, "Why is the Red Sea salty?")
    assert variants[0] == "Why is the Red Sea salty?", "original question must always be included"
