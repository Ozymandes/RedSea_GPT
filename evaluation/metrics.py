"""
Evaluation Metrics for RedSea GPT

Implements metrics for evaluating retrieval and generation quality.
"""

from typing import List, Dict, Any, TypedDict


def evaluate_answer_relevance(
    answer: str,
    question: str,
    expected_keywords: List[str],
) -> Dict[str, Any]:
    """
    Evaluate the relevance of an answer using keyword matching.

    Args:
        answer: Generated answer
        question: Original question
        expected_keywords: List of expected keywords

    Returns:
        Dictionary with relevance metrics
    """
    answer_lower = answer.lower()

    # Check keyword presence
    keywords_found = [
        kw for kw in expected_keywords
        if kw.lower() in answer_lower
    ]

    keyword_coverage = len(keywords_found) / len(expected_keywords) if expected_keywords else 0

    return {
        "keyword_coverage": keyword_coverage,
        "keywords_found": keywords_found,
        "keywords_total": len(expected_keywords),
        "answer_length": len(answer),
    }


def evaluate_retrieval_quality(
    retrieved_docs: List[Dict[str, Any]],
    question: str,
) -> Dict[str, Any]:
    """
    Evaluate the quality of retrieved documents.

    Args:
        retrieved_docs: List of retrieved documents
        question: The query question

    Returns:
        Dictionary with retrieval metrics
    """
    if not retrieved_docs:
        return {
            "num_retrieved": 0,
            "avg_content_length": 0,
            "unique_sources": 0,
        }

    # Content length metrics
    content_lengths = [doc.get("content_length", 0) for doc in retrieved_docs]
    avg_content_length = sum(content_lengths) / len(content_lengths)

    # Source diversity
    sources = [doc.get("source", "Unknown") for doc in retrieved_docs]
    unique_sources = len(set(sources))

    return {
        "num_retrieved": len(retrieved_docs),
        "avg_content_length": avg_content_length,
        "unique_sources": unique_sources,
        "source_diversity": unique_sources / len(retrieved_docs) if retrieved_docs else 0,
    }


def evaluate_faithfulness(
    answer: str,
    context: str,
) -> Dict[str, Any]:
    """
    Basic faithfulness check - does answer stay grounded in context?

    This is a simplified version. For production, you'd want to use
    more sophisticated methods like NLI-based fact checking.

    Args:
        answer: Generated answer
        context: Retrieved context

    Returns:
        Dictionary with faithfulness metrics
    """
    # Split answer into sentences
    sentences = [s.strip() for s in answer.split(".") if s.strip()]

    # Check if sentences have content from context
    context_words = set(context.lower().split())

    sentences_with_context = 0
    for sentence in sentences:
        sentence_words = set(sentence.lower().split())
        overlap = len(sentence_words & context_words)
        # If > 30% of sentence words overlap with context
        if len(sentence_words) > 0 and overlap / len(sentence_words) > 0.3:
            sentences_with_context += 1

    faithfulness = sentences_with_context / len(sentences) if sentences else 0

    return {
        "faithfulness": faithfulness,
        "total_sentences": len(sentences),
        "grounded_sentences": sentences_with_context,
    }


def calculate_evaluation_summary(
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Calculate summary statistics across all evaluation results.

    Args:
        results: List of evaluation results

    Returns:
        Dictionary with summary statistics
    """
    if not results:
        return {}

    # Average keyword coverage
    coverages = [r["metrics"]["relevance"]["keyword_coverage"] for r in results]
    avg_coverage = sum(coverages) / len(coverages) if coverages else 0

    # By difficulty
    difficulty_performance = {}
    for result in results:
        diff = result["question"]["difficulty"]
        if diff not in difficulty_performance:
            difficulty_performance[diff] = []
        difficulty_performance[diff].append(
            result["metrics"]["relevance"]["keyword_coverage"]
        )

    difficulty_avg = {
        diff: sum(scores) / len(scores)
        for diff, scores in difficulty_performance.items()
    }

    # By category
    category_performance = {}
    for result in results:
        cat = result["question"]["category"]
        if cat not in category_performance:
            category_performance[cat] = []
        category_performance[cat].append(
            result["metrics"]["relevance"]["keyword_coverage"]
        )

    category_avg = {
        cat: sum(scores) / len(scores)
        for cat, scores in category_performance.items()
    }

    return {
        "total_questions": len(results),
        "avg_keyword_coverage": avg_coverage,
        "by_difficulty": difficulty_avg,
        "by_category": category_avg,
    }


class EvaluationResult:
    """
    Container for evaluation results.
    """

    def __init__(
        self,
        question: Dict[str, Any],
        answer: str,
        sources: List[Dict[str, Any]],
        metrics: Dict[str, Any],
    ):
        self.question = question
        self.answer = answer
        self.sources = sources
        self.metrics = metrics

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": self.sources,
            "metrics": self.metrics,
        }

    def print_summary(self) -> None:
        """Print a summary of this evaluation result."""
        print(f"\n❓ Question ({self.question['difficulty']}): {self.question['question']}")
        print(f"📝 Category: {self.question['category']}")
        print(f"✅ Keyword Coverage: {self.metrics['relevance']['keyword_coverage']:.2%}")
        print(f"📚 Sources: {len(self.sources)} documents")

    def is_passed(self, threshold: float = 0.5) -> bool:
        """
        Check if the evaluation passed the threshold.

        Args:
            threshold: Minimum keyword coverage threshold

        Returns:
            True if passed, False otherwise
        """
        return self.metrics["relevance"]["keyword_coverage"] >= threshold
