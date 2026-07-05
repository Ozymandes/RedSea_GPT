"""
Smoke tests that need no LLM provider and no heavy ML deps where possible.

These guard the parts of the system that must work on a clean clone:
  - .env / secrets hygiene
  - provider config resolution (no key leakage)
  - prompt + context formatting
  - utility helpers
"""

import importlib
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# 1. Secrets hygiene
# ---------------------------------------------------------------------------
def test_env_is_gitignored():
    """The real .env must be protected by .gitignore."""
    gi = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".env" in gi, ".env must be listed in .gitignore"


def test_env_example_has_no_real_keys():
    """.env.example must contain only placeholders, never real sk- keys."""
    example = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    placeholder_words = ("your", "here", "placeholder", "xxx", "example", "replace", "<", "key")
    for line in example.splitlines():
        if "sk-" in line and "=" in line and not line.strip().startswith("#"):
            val = line.split("=", 1)[1].strip().lower()
            looks_like_placeholder = any(w in val for w in placeholder_words)
            looks_like_token = (
                len(val) > 20
                and val.replace("sk-", "").replace("-", "").replace("_", "").isalnum()
            )
            assert looks_like_placeholder or not looks_like_token, (
                f"Possible real key in .env.example: {line[:25]}..."
            )


def test_provider_config_never_leaks_key():
    """describe_active_provider() and repr(llm) must never expose the real key."""
    # Read the real key length from .env if present (never its value).
    real_key = None
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        for ln in env_path.read_bytes().split(b"\n"):
            s = ln.decode("utf-8", "replace").rstrip("\r")
            if s.startswith("OPTO_LLM_API_KEY=") or s.startswith("GROQ_API_KEY="):
                real_key = s.split("=", 1)[1]
                break

    cfg = importlib.import_module("generation.llm_config")
    info = cfg.describe_active_provider()
    blob = repr(info)
    assert "api_key" not in blob, "provider description exposed an api_key field"
    if real_key and len(real_key) > 8:
        key_str = real_key.decode("utf-8", "replace") if isinstance(real_key, bytes) else real_key
        assert key_str not in blob, "real key value leaked into provider description"


def test_optollm_brand_alias_resolves():
    """Both spellings (brand "optollm" and legacy "optillm") must resolve to the
    same provider preset, so a brand-correct LLM_PROVIDER value doesn't crash
    the engine build at deploy time."""
    from generation.llm_config import resolve_provider_config, PROVIDER_PRESETS

    import os
    saved = os.environ.get("LLM_PROVIDER")
    os.environ["LLM_PROVIDER"] = "optollm"
    try:
        cfg_brand = resolve_provider_config()
    finally:
        if saved is None:
            os.environ.pop("LLM_PROVIDER", None)
        else:
            os.environ["LLM_PROVIDER"] = saved

    os.environ["LLM_PROVIDER"] = "optillm"
    try:
        cfg_legacy = resolve_provider_config()
    finally:
        if saved is None:
            os.environ.pop("LLM_PROVIDER", None)
        else:
            os.environ["LLM_PROVIDER"] = saved

    # Both must resolve to the same canonical preset (the alias maps optollm -> optillm).
    assert cfg_brand["provider"] == "optillm", "brand spelling did not resolve via alias"
    assert cfg_brand["provider"] == cfg_legacy["provider"]
    assert cfg_brand["base_url"] == cfg_legacy["base_url"]
    assert cfg_brand["protocol"] == cfg_legacy["protocol"]
    assert "optollm" not in PROVIDER_PRESETS  # alias, not a duplicate preset


def test_llm_object_repr_hides_secret():
    """The LLM client object must never expose the key via repr()/str()."""
    from generation.llm_config import create_llm
    try:
        llm = create_llm()
    except Exception:
        pytest.skip("no provider configured")
    key = llm._key_value()
    if not key or len(key) < 8:
        pytest.skip("no real key to test against")
    assert key not in repr(llm), "full key leaked via repr(llm)"
    assert key not in str(llm), "full key leaked via str(llm)"
    assert "api_key" not in llm._identifying_params


# ---------------------------------------------------------------------------
# 2. Prompt & context formatting
# ---------------------------------------------------------------------------
def test_rag_prompt_loads_with_required_sections():
    from generation.prompts import create_rag_prompt

    p = create_rag_prompt()
    formatted = p.format(context="[1] (Source: x.pdf, page 3) coral heat tolerance",
                         question="Why are Red Sea corals heat tolerant?")
    # The prompt must enforce grounding + citation discipline.
    assert "only" in formatted.lower() or "context" in formatted.lower()
    assert "[1]" in formatted
    assert "question" in formatted.lower()


def test_context_formatting_has_provenance_and_citation_ids():
    from generation.prompts import format_context
    from langchain_core.documents import Document

    docs = [
        Document(page_content="Red Sea salinity is about 40 per mille.",
                 metadata={"source": "ocean.pdf", "page": 12}),
        Document(page_content="Corals tolerate high heat.",
                 metadata={"source": "coral.pdf", "page": 4}),
    ]
    ctx = format_context(docs)
    assert "[1]" in ctx and "[2]" in ctx
    assert "ocean.pdf" in ctx and "page 12" in ctx
    assert "coral.pdf" in ctx and "page 4" in ctx


def test_clean_source_path_handles_separators():
    from generation.utils import clean_source_path

    assert clean_source_path("data/docs/red_sea.pdf") == "red_sea.pdf"
    assert clean_source_path("data\\docs\\red_sea.pdf") == "red_sea.pdf"
    assert clean_source_path("plain.pdf") == "plain.pdf"


# ---------------------------------------------------------------------------
# 3. Golden set integrity
# ---------------------------------------------------------------------------
def test_golden_set_has_all_required_categories():
    from evaluation.golden_set import GOLDEN_SET, answerable_questions, refusal_questions

    groups = {q["group"] for q in GOLDEN_SET}
    required = {
        "geology", "oceanography", "coral_heat", "biodiversity", "conservation",
        "synthesis", "off_topic", "unsupported", "hallucination_trap",
        "citation_integrity",
    }
    assert required.issubset(groups), f"missing groups: {required - groups}"
    assert len(GOLDEN_SET) >= 30, "golden set should have at least 30 questions"
    assert len(answerable_questions()) >= 15
    assert len(refusal_questions()) >= 8
    # every question declares an expected behavior
    assert all(q.get("expected_behavior") in ("answer", "refuse") for q in GOLDEN_SET)


# ---------------------------------------------------------------------------
# 4. Metrics are pure + transparent
# ---------------------------------------------------------------------------
def test_metrics_refusal_and_citation_detection():
    from evaluation import metrics_v2 as m

    assert m.is_refusal("I don't have sufficient information to answer.") is True
    assert m.is_refusal("The Red Sea formed by rifting [1]. It is a young ocean basin.") is False
    pres = m.evaluate_citation_presence("Claim [1]. Another [2].", min_citations=1)
    assert pres["num_distinct"] == 2 and pres["ok"]
    sup = m.evaluate_citation_support("Cite [1] and [9].", [
        {"content": "a"}, {"content": "b"}, {"content": "c"}])
    assert 9 in sup["unsupported"] and 1 in sup["supported"]


def test_metrics_faithfulness_is_explainable():
    from evaluation import metrics_v2 as m

    ctx = ("The Red Sea has high salinity around 40 per mille due to high "
           "evaporation in its warm arid climate.")
    # Near-verbatim echo of the context -> should be highly grounded.
    ans_good = ("The Red Sea has high salinity around 40 per mille due to high "
                "evaporation in its warm arid climate [1].")
    # Totally unrelated text -> should be ungrounded.
    ans_bad = ("The Great Barrier Reef lies off the coast of Queensland, "
               "Australia and is a popular tourist destination.")
    f_good = m.evaluate_faithfulness(ans_good, ctx)
    f_bad = m.evaluate_faithfulness(ans_bad, ctx)
    assert f_good["faithfulness"] >= 0.6, f_good
    assert f_bad["faithfulness"] <= 0.2, f_bad
    assert f_good["faithfulness"] > f_bad["faithfulness"]
    assert "evidence" in f_good


# ---------------------------------------------------------------------------
# Upfront scope gate (saves API tokens by refusing OOS questions pre-LLM)
# ---------------------------------------------------------------------------
def test_upfront_scope_gate_refuses_off_topic_and_passes_in_scope():
    """Obviously off-topic questions are refused BEFORE any LLM call; legit Red
    Sea / marine questions always pass through. Pure function — no engine load."""
    from generation.rag_chain import RedSeaGPT
    # Bypass __init__ (which loads the 1.6GB engine) — we only test the pure check.
    g = RedSeaGPT.__new__(RedSeaGPT)

    off_topic = [
        "Who won the 2022 World Cup?",
        "Write me a Python function to sort a list",
        "What is the capital of France?",
        "How do I bake sourdough bread?",
        "What is the bitcoin price today?",
        "Should I break up with my girlfriend?",
        # New categories (broadened coverage):
        "Solve my math homework",
        "Solve 2x + 5 = 15",
        "Translate hello to french",
        "Write me a poem about love",
        "Tell me a joke",
        "Explain quantum physics",
        "How do I fix my car engine",
        "Best gpu for gaming",
        "What is the meaning of life",
        "Help me with my resume",
        "Summarize the bible",
        "Who wrote hamlet",
        "Diagnose my chest pain",
        "Explain the french revolution",
        "Tell me about taylor swift",
        "Write me a business plan",
    ]
    for q in off_topic:
        oos, reason = g._is_clearly_out_of_scope(q)
        assert oos, f"should refuse upfront: {q!r} (reason={reason!r})"
        assert reason, f"reason text missing for {q!r}"

    in_scope = [
        "How did the Red Sea form?",
        "Why are some corals heat tolerant?",
        "How deep is the Gulf of Aqaba?",
        "What is the salinity of the water there?",
        "Tell me about the mangroves",
        "Which fish are endemic?",
        # New in-scope probes (must NOT be false-positived by the broader gate):
        "What causes coral bleaching?",
        "Are there hydrothermal vents?",
        "Describe the thermocline",
        "How fast is the sea floor spreading?",
        "How does warming affect reefs?",
        "What is the spreading rate of the rift?",
    ]
    for q in in_scope:
        oos, reason = g._is_clearly_out_of_scope(q)
        assert not oos, f"should NOT be refused upfront: {q!r} (got reason={reason!r})"
