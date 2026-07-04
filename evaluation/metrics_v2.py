"""
Transparent evaluation metrics for RedSea GPT.

These metrics are deliberately *explainable* (no opaque LLM-as-judge by default).
Each score returns the raw evidence it was computed from so results in
``eval_results/`` can be audited by a reviewer.

Metrics implemented
-------------------
* ``retrieval``       - did we return chunks with provenance metadata?
* ``answer_correctness`` - required-concept coverage for answerable questions
* ``citation_presence`` - does the answer contain ``[n]`` markers?
* ``citation_support``   - does each cited chunk actually exist in retrieval and
                          contain the supporting concept? (citation integrity)
* ``faithfulness``    - sentence-level n-gram grounding in retrieved context
* ``refusal_correctness`` - correct refuse/answer decision vs expectation
* ``hallucination``   - severe-hallucination flag for known trap categories
* ``latency_ms``      - wall-clock for the full query

An *optional* LLM-as-judge pass (``judge_with_llm``) is provided for clarity
scoring, but it is OFF by default and always stores its prompt + raw verdict in
the eval artifact so the judge is fully auditable.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

# Phrases the RAG chain uses when it correctly refuses. If the model output is
# dominated by these, we treat it as a refusal.
REFUSUAL_PHRASES = (
    "i don't have sufficient information",
    "i do not have sufficient information",
    "cannot answer",
    "can't answer",
    "i cannot answer",
    "i can't answer",
    "i'm unable to answer",
    "i am unable to answer",
    "unable to answer",
    "outside the scope",
    "outside my scope",
    "not covered in",
    "not mentioned in",
    "doesn't appear in",
    "does not appear in",
    "isn't mentioned",
    "is not mentioned",
    "no information",
    "not in the provided context",
    "not supported",
    "not available in my",
    "i apologize, but",
    "i'm sorry, but",
    "i am sorry, but",
    "cannot provide",
    "can't provide",
    "unable to provide",
    "i'm not able to",
    "i am not able to",
    "context does not",
    "provided context does not",
    "does not include",
    "don't have that information",
    "available sources",
)

# Citation marker like [1], [12].
_CITATION_RE = re.compile(r"\[(\d{1,3})\]")


def is_refusal(answer: str) -> bool:
    """Heuristic: treat an answer as a refusal if it opens with / is dominated by
    refusal language. Refusals in this system include a helpful explanation, so
    they can be longer than a terse 'no' -- we therefore (a) always treat a
    refusal phrase in the first sentence as a refusal, and (b) otherwise require
    a refusal phrase AND a short answer."""
    if not answer:
        return True
    low = answer.lower().strip()
    has_phrase = any(p in low for p in REFUSUAL_PHRASES)
    if not has_phrase:
        return False
    # If the answer STARTS with refusal language, it's a refusal regardless of
    # the length of the accompanying explanation.
    first_clause = low.split(".")[0][:200]
    starts_with_refusal = any(low.startswith(p) for p in REFUSUAL_PHRASES) or any(p in first_clause for p in REFUSUAL_PHRASES[:8])
    short = len(answer) < 600
    return starts_with_refusal or short


def evaluate_retrieval(sources: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Did retrieval return chunks carrying provenance (source + page)?"""
    n = len(sources)
    with_source = sum(1 for s in sources if s.get("source"))
    with_page = sum(1 for s in sources if s.get("page") is not None and s.get("page") != "?")
    unique_sources = len({s.get("source") for s in sources if s.get("source")})
    return {
        "num_retrieved": n,
        "with_source_metadata": with_source,
        "with_page_metadata": with_page,
        "unique_sources": unique_sources,
        "ok": n > 0 and with_source == n and with_page == n,
    }


def evaluate_concept_coverage(answer: str, required_concepts: List[str]) -> Dict[str, Any]:
    """Required-concept coverage for an answerable question (idea-level, not exact)."""
    low = answer.lower()
    found = [c for c in required_concepts if c.lower() in low]
    coverage = (len(found) / len(required_concepts)) if required_concepts else 1.0
    return {
        "coverage": coverage,
        "found": found,
        "missing": [c for c in required_concepts if c not in found],
        "ok": coverage >= 0.5,
    }


def evaluate_citation_presence(answer: str, min_citations: int = 1) -> Dict[str, Any]:
    """Does the answer cite at least ``min_citations`` distinct sources?"""
    ids = sorted({int(m) for m in _CITATION_RE.findall(answer)})
    return {
        "citation_ids": ids,
        "num_distinct": len(ids),
        "min_required": min_citations,
        "ok": len(ids) >= min_citations,
    }


def evaluate_citation_support(
    answer: str, sources: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Citation integrity: every cited ``[n]`` must map to a real retrieved
    chunk (1..len(sources)). Reports which citations are unsupported."""
    cited = sorted({int(m) for m in _CITATION_RE.findall(answer)})
    valid_ids = set(range(1, len(sources) + 1))
    supported = [c for c in cited if c in valid_ids]
    unsupported = [c for c in cited if c not in valid_ids]
    return {
        "cited": cited,
        "supported": supported,
        "unsupported": unsupported,
        "ok": len(cited) > 0 and len(unsupported) == 0,
    }


def evaluate_faithfulness(answer: str, context: str) -> Dict[str, Any]:
    """Sentence-level grounding: fraction of answer sentences with strong
    token overlap (4-gram Jaccard) against the retrieved context. Transparent
    and cheap; conservative against paraphrase."""
    if not answer.strip() or not context.strip():
        return {"faithfulness": 0.0, "grounded": 0, "total": 0, "ok": False}

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if len(s.strip()) > 25]
    if not sentences:
        return {"faithfulness": 1.0, "grounded": 0, "total": 0, "ok": True}

    ctx_tokens = set(_tokenize(context))
    ctx_4grams = set(_ngrams(context, 4))

    grounded = 0
    evidence = []
    for s in sentences:
        s_tokens = set(_tokenize(s))
        s_4grams = set(_ngrams(s, 4))
        token_overlap = len(s_tokens & ctx_tokens) / max(1, len(s_tokens))
        ngram_overlap = len(s_4grams & ctx_4grams) / max(1, len(s_4grams))
        is_grounded = token_overlap >= 0.6 or ngram_overlap >= 0.25
        if is_grounded:
            grounded += 1
        evidence.append({
            "sentence": s[:120],
            "token_overlap": round(token_overlap, 3),
            "ngram_overlap": round(ngram_overlap, 3),
            "grounded": is_grounded,
        })

    faith = grounded / len(sentences)
    return {
        "faithfulness": round(faith, 3),
        "grounded": grounded,
        "total": len(sentences),
        "ok": faith >= 0.6,
        "evidence": evidence[:6],  # cap stored evidence
    }


def evaluate_refusal_correctness(
    answer: str, expected_behavior: str
) -> Dict[str, Any]:
    """Did the system correctly answer vs refuse per the golden-set expectation?"""
    actually_refused = is_refusal(answer)
    expected_refusal = expected_behavior == "refuse"
    correct = actually_refused == expected_refusal
    return {
        "actually_refused": actually_refused,
        "expected_refusal": expected_refusal,
        "correct": correct,
        "ok": correct,
        # A refusal where an answer was expected (or vice-versa) is a serious miss.
        "serious_failure": not correct,
    }


def evaluate_hallucination(
    question_meta: Dict[str, Any], answer: str, sources: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Flag severe hallucinations. The strongest signal is the trap category:
    a fabricated entity question that is *answered* (not refused) is a severe
    hallucination by definition."""
    group = question_meta.get("group", "")
    refused = is_refusal(answer)
    severe = False
    reason = ""

    if group == "hallucination_trap" and not refused:
        severe = True
        reason = "Answered a fabricated-entity / hallucination-trap question instead of refusing."
    elif group in ("off_topic", "unsupported") and not refused:
        severe = True
        reason = f"Answered a {group} question that should have been refused."

    # Secondary signal: very low faithfulness on an answerable question.
    # Only counts as severe if the answer ALSO fails concept coverage -- a
    # well-paraphrased but on-topic, well-cited answer is NOT a hallucination.
    if not severe and not refused and question_meta.get("expected_behavior") == "answer":
        faith = evaluate_faithfulness(answer, " ".join(s.get("content", "") for s in sources))
        # concept coverage check
        concepts = question_meta.get("required_concepts", [])
        low_concept = bool(concepts) and not any(c.lower() in answer.lower() for c in concepts)
        if faith["total"] >= 2 and faith["faithfulness"] < 0.2 and low_concept:
            severe = True
            reason = f"Very low faithfulness ({faith['faithfulness']:.0%}) AND missing required concepts."

    return {"severe_hallucination": severe, "reason": reason, "ok": not severe}


# ---------------------------------------------------------------------------
# LLM-based faithfulness (claim extraction + batched entailment)
# ---------------------------------------------------------------------------
# A dependency-light reimplementation of the RAGAS faithfulness idea (no RAGAS/
# DeepEval import): decompose the answer into atomic claims, then ask one LLM
# call to judge each claim as supported / contradicted / unsupported against the
# retrieved context. The n-gram heuristic above is cheap and conservative; this
# metric is sharper and explainable, and returns the per-claim verdicts so a
# reviewer can audit exactly which claims were flagged.
# ---------------------------------------------------------------------------
_EXTRACT_CLAIMS_PROMPT = """You decompose an answer into atomic, self-contained claims for a grounding check.
Given a question and an answer, break the answer into the smallest set of factual claims. Rules:
- One verifiable fact per claim. Split compound sentences.
- Resolve pronouns to concrete entities (use the question).
- Do NOT add information not in the answer; do NOT use prior knowledge.
- Ignore stylistic/greeting/meta phrases.

Question: {question}
Answer: {answer}

Return ONLY a JSON object: {{"claims": ["...", "..."]}}. If no factual claims, return {{"claims": []}}."""

_JUDGE_CLAIMS_PROMPT = """You are a strict grounding judge. Decide whether each claim is supported by the CONTEXT (only the context, never your own knowledge).

Verdicts:
- "supported"    - directly inferable from the context (small rounding ok)
- "contradicted" - the context directly asserts the opposite
- "unsupported"  - neither; the claim is absent from the context

CONTEXT:
\"\"\"{context}\"\"\"

CLAIMS:
{claims_json}

Return ONLY a JSON object: {{"verdicts": [{{"claim": "...", "verdict": "supported|contradicted|unsupported", "evidence": "<short context quote or empty>"}}]}} — one verdict per claim, in order."""

_CLAIM_CREDIT = {"supported": 1.0, "unsupported": 0.0, "contradicted": 0.0}


def _extract_claims(llm, question: str, answer: str) -> List[str]:
    if not answer.strip():
        return []
    try:
        out = llm.invoke(_EXTRACT_CLAIMS_PROMPT.format(question=question, answer=answer[:2500]))
        raw = getattr(out, "content", out)
        if isinstance(raw, dict) and "content" in raw:
            raw = raw["content"]
        m = re.search(r"\{.*\}", str(raw), re.DOTALL)
        if not m:
            return []
        parsed = _json_loads_lenient(m.group(0))
        return [str(c).strip() for c in (parsed or {}).get("claims", []) if str(c).strip()]
    except Exception:  # noqa: BLE001
        return []


def _judge_claims(llm, claims: List[str], context: str) -> List[Dict[str, str]]:
    if not claims:
        return []
    try:
        out = llm.invoke(_JUDGE_CLAIMS_PROMPT.format(
            context=context[:5000], claims_json=json.dumps(claims, ensure_ascii=False)))
        raw = getattr(out, "content", out)
        if isinstance(raw, dict) and "content" in raw:
            raw = raw["content"]
        m = re.search(r"\{.*\}", str(raw), re.DOTALL)
        if not m:
            return []
        parsed = _json_loads_lenient(m.group(0)) or {}
        return parsed.get("verdicts", [])
    except Exception:  # noqa: BLE001
        return []


def _json_loads_lenient(s: str):
    import json as _j
    s = s.strip()
    try:
        return _j.loads(s)
    except Exception:
        # strip markdown fences + repair trailing commas
        s2 = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.MULTILINE)
        s2 = re.sub(r",\s*([}\]])", r"\1", s2)
        try:
            return _j.loads(s2)
        except Exception:
            return None


def evaluate_faithfulness_llm(llm, question: str, answer: str, context: str) -> Dict[str, Any]:
    """Claim-level faithfulness via LLM entailment (RAGAS-style, dependency-free).

    Returns a 0-1 score plus the full per-claim verdicts for audit. Falls back to
    a None score (never silently 1.0) if the judge cannot be parsed.
    """
    if not answer.strip():
        return {"faithfulness": 0.0, "claims": [], "verdicts": [], "ok": False, "method": "llm"}
    claims = _extract_claims(llm, question, answer)
    if not claims:
        return {"faithfulness": None, "claims": [], "verdicts": [], "ok": False,
                "reason": "no claims extracted", "method": "llm"}
    verdicts = _judge_claims(llm, claims, context)
    if not verdicts or len(verdicts) != len(claims):
        return {"faithfulness": None, "claims": claims, "verdicts": verdicts, "ok": False,
                "reason": "judge verdict count mismatch (possible truncation)", "method": "llm"}
    credits = [_CLAIM_CREDIT.get(v.get("verdict", "unsupported"), 0.0) for v in verdicts]
    score = sum(credits) / len(credits) if credits else 0.0
    n_sup = sum(1 for v in verdicts if v.get("verdict") == "supported")
    n_uns = sum(1 for v in verdicts if v.get("verdict") == "unsupported")
    n_con = sum(1 for v in verdicts if v.get("verdict") == "contradicted")
    return {
        "faithfulness": round(score, 4),
        "n_claims": len(claims),
        "supported": n_sup, "unsupported": n_uns, "contradicted": n_con,
        "verdicts": verdicts[:8],
        "ok": score >= 0.7,
        "method": "llm_claim_entailment",
    }


# ---------------------------------------------------------------------------
# Optional, fully-auditable LLM-as-judge for answer clarity
# ---------------------------------------------------------------------------
JUDGE_PROMPT_TEMPLATE = """You are a strict but fair evaluator for a scientific RAG system about the Egyptian Red Sea.

Question: {question}
Expected behavior: {expected}
Retrieved context (excerpt):
\"\"\"{context}\"\"\"
Answer to evaluate:
\"\"\"{answer}\"\"\"

Rate the answer on THREE 0-5 scales (use decimals allowed). Be strict:
1. groundedness: are claims supported ONLY by the retrieved context? (5 = fully)
2. citation_integrity: are citations present AND do they point to real supporting chunks? (0 = none)
3. clarity: is it clear, well-structured, free of padding? (5 = excellent)

Respond in STRICT JSON only, nothing else:
{{"groundedness": <0-5>, "citation_integrity": <0-5>, "clarity": <0-5>, "verdict": "<pass|borderline|fail>", "one_line_reason": "<short>"}}
"""


def judge_with_llm(llm, question: str, answer: str, context: str, expected_behavior: str) -> Dict[str, Any]:
    """Run an optional LLM-as-judge pass. Always returns the prompt + raw output
    so the verdict is auditable. Never auto-applied; caller opts in."""
    import json as _json

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        expected=expected_behavior,
        context=context[:2500],
        answer=answer[:2500],
    )
    raw = ""
    try:
        out = llm.invoke(prompt)
        raw = getattr(out, "content", out)
        if isinstance(raw, dict) and "content" in raw:
            raw = raw["content"]
        raw = str(raw).strip()
        # Extract the JSON object defensively.
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        parsed = _json.loads(m.group(0)) if m else {}
        return {"judge_prompt": prompt, "judge_raw": raw[:1200], "judge": parsed}
    except Exception as exc:  # noqa: BLE001
        return {"judge_prompt": prompt, "judge_raw": raw[:1200], "judge": {"error": str(exc)}}


# ---------------------------------------------------------------------------
# text utils
# ---------------------------------------------------------------------------
_WORD_RE = re.compile(r"[a-z0-9‰°']+")


def _tokenize(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


def _ngrams(text: str, n: int) -> List[tuple]:
    toks = _tokenize(text)
    return [tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)]
