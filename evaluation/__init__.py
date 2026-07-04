"""
Evaluation module for RedSea GPT.

Exposes the lightweight, dependency-free evaluation pieces directly. The full
evaluation runner (``run_evaluation``) imports the RAG pipeline, so it is loaded
lazily to keep imports cheap and tests isolated.
"""

from .questions import TEST_QUESTIONS, get_questions_by_category
from .metrics import evaluate_answer_relevance, evaluate_retrieval_quality

__all__ = [
    "TEST_QUESTIONS",
    "get_questions_by_category",
    "evaluate_answer_relevance",
    "evaluate_retrieval_quality",
    "run_evaluation",
    "EvaluationResult",
]


def __getattr__(name):
    if name in ("run_evaluation", "EvaluationResult"):
        from .run_evaluation import run_evaluation, EvaluationResult
        return {"run_evaluation": run_evaluation, "EvaluationResult": EvaluationResult}[name]
    raise AttributeError(f"module 'evaluation' has no attribute {name!r}")
