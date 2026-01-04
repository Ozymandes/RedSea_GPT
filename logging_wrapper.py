"""
Simple logging wrapper for RedSea GPT

Wraps the query method to add logging without modifying the core RAG pipeline.
"""

import time
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional


class SimpleLogger:
    """Simple file-based logger"""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Create log files
        self.request_log = open(self.log_dir / "requests.log", "a", encoding="utf-8")
        self.response_log = open(self.log_dir / "responses.log", "a", encoding="utf-8")

    def log_request(self, question: str, session_id: Optional[str] = None):
        """Log incoming request"""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": "request",
            "question": question,
            "session_id": session_id or "default",
        }
        self.request_log.write(json.dumps(entry) + "\n")
        self.request_log.flush()

    def log_response(
        self,
        question: str,
        answer: str,
        confidence: float,
        latency_ms: float,
        session_id: Optional[str] = None,
        **kwargs
    ):
        """Log response"""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": "response",
            "question": question,
            "answer": answer[:500],  # Truncate
            "confidence": confidence,
            "latency_ms": latency_ms,
            "session_id": session_id or "default",
            **kwargs
        }
        self.response_log.write(json.dumps(entry) + "\n")
        self.response_log.flush()

    def close(self):
        """Close log files"""
        self.request_log.close()
        self.response_log.close()


class LoggedRedSeaGPT:
    """Wrapper that adds logging to RedSeaGPT"""

    def __init__(self, gpt_instance, enable_logging: bool = True):
        self.gpt = gpt_instance
        self.enable_logging = enable_logging
        self.logger = SimpleLogger() if enable_logging else None
        self.session_id = f"session_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    def query(self, question: str, return_source_docs: bool = False) -> Any:
        """Query with logging"""
        start_time = time.time()

        # Log request
        if self.logger:
            self.logger.log_request(question, self.session_id)

        # Call actual query
        result = self.gpt.query(question, return_source_docs=return_source_docs)

        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000

        # Log response
        if self.logger:
            if isinstance(result, dict):
                self.logger.log_response(
                    question=question,
                    answer=result.get("answer", ""),
                    confidence=result.get("confidence", 0),
                    latency_ms=latency_ms,
                    session_id=self.session_id,
                    num_sources=result.get("num_sources", 0),
                )
            else:
                self.logger.log_response(
                    question=question,
                    answer=str(result),
                    confidence=0,
                    latency_ms=latency_ms,
                    session_id=self.session_id,
                )

        return result

    def __getattr__(self, name):
        """Proxy all other attributes to the wrapped instance"""
        return getattr(self.gpt, name)
