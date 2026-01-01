"""
Evaluation Module for RedSea GPT

Provides tools for evaluating RAG system performance.
"""

from .questions import TEST_QUESTIONS, get_questions_by_category
from .metrics import evaluate_answer_relevance, evaluate_retrieval_quality
from .run_evaluation import run_evaluation, EvaluationResult

__all__ = [
    "TEST_QUESTIONS",
    "get_questions_by_category",
    "evaluate_answer_relevance",
    "evaluate_retrieval_quality",
    "run_evaluation",
    "EvaluationResult",
]
