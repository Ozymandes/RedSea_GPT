"""
Logging Utilities for RedSea GPT

Helper functions for logging events.
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any, List


# Get loggers
request_logger = logging.getLogger("redsea.requests")
response_logger = logging.getLogger("redsea.responses")


def log_request(
    question: str,
    session_id: Optional[str] = None,
    **kwargs
):
    """Log incoming request"""
    request_logger.info("", extra={
        "event": "request",
        "question": question,
        "session_id": session_id,
        "timestamp": datetime.utcnow().isoformat(),
        **kwargs
    })


def log_response(
    question: str,
    answer: str,
    confidence: float,
    grounding_rate: float,
    num_sources: int,
    retrieval_method: str,
    refused: bool,
    session_id: Optional[str] = None,
    latency_ms: float = 0,
    sources: Optional[List[Dict]] = None,
    **kwargs
):
    """Log response"""
    response_logger.info("", extra={
        "event": "response",
        "question": question,
        "answer": answer[:500],  # Truncate long answers
        "confidence": confidence,
        "grounding_rate": grounding_rate,
        "num_sources": num_sources,
        "retrieval_method": retrieval_method,
        "refused": refused,
        "session_id": session_id,
        "latency_ms": latency_ms,
        "timestamp": datetime.utcnow().isoformat(),
        **kwargs
    })


class RequestTimer:
    """Context manager for timing requests"""

    def __init__(self, question: str, session_id: Optional[str] = None):
        self.question = question
        self.session_id = session_id
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = datetime.utcnow()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.utcnow()

    def get_latency_ms(self) -> float:
        """Get latency in milliseconds"""
        if self.start_time and self.end_time:
            delta = self.end_time - self.start_time
            return delta.total_seconds() * 1000
        return 0
