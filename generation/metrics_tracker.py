"""
Simple Metrics Tracker for RedSea GPT

Tracks performance metrics over time.
"""

from typing import Dict, Optional
from datetime import datetime, timedelta
from threading import Lock
from collections import defaultdict


class MetricsTracker:
    """Track and aggregate metrics"""

    def __init__(self, window_seconds: int = 3600):
        self.window_seconds = window_seconds
        self.lock = Lock()

        # Metrics storage
        self.queries = []  # List of (timestamp, metrics_dict)
        self.session_metrics = defaultdict(list)

    def record_request(
        self,
        confidence: float,
        grounding_rate: float,
        refused: bool,
        latency_ms: float,
        error: bool,
        session_id: Optional[str] = None,
    ):
        """Record a single request"""
        with self.lock:
            now = datetime.utcnow()
            metrics = {
                "timestamp": now,
                "confidence": confidence,
                "grounding_rate": grounding_rate,
                "refused": refused,
                "latency_ms": latency_ms,
                "error": error,
            }

            self.queries.append(metrics)
            if session_id:
                self.session_metrics[session_id].append(metrics)

            # Clean old queries
            self._cleanup_old_queries()

    def _cleanup_old_queries(self):
        """Remove queries outside the time window"""
        cutoff = datetime.utcnow() - timedelta(seconds=self.window_seconds)
        self.queries = [q for q in self.queries if q["timestamp"] > cutoff]

    def get_metrics(self, session_id: Optional[str] = None) -> Dict:
        """Get aggregated metrics"""
        with self.lock:
            if session_id:
                queries = self.session_metrics.get(session_id, [])
            else:
                queries = self.queries

            if not queries:
                return {
                    "total_queries": 0,
                    "successful_queries": 0,
                    "refused_queries": 0,
                    "error_queries": 0,
                    "avg_confidence": 0.0,
                    "avg_grounding_rate": 0.0,
                    "avg_latency_ms": 0.0,
                }

            total = len(queries)
            successful = sum(1 for q in queries if not q["error"] and not q["refused"])
            refused = sum(1 for q in queries if q["refused"])
            errors = sum(1 for q in queries if q["error"])

            avg_confidence = sum(q["confidence"] for q in queries) / total
            avg_grounding = sum(q["grounding_rate"] for q in queries) / total
            avg_latency = sum(q["latency_ms"] for q in queries) / total

            return {
                "total_queries": total,
                "successful_queries": successful,
                "refused_queries": refused,
                "error_queries": errors,
                "avg_confidence": avg_confidence,
                "avg_grounding_rate": avg_grounding,
                "avg_latency_ms": avg_latency,
            }

    def flush_metrics(self):
        """Flush metrics to log file"""
        from .log_utils import log_metrics

        metrics = self.get_metrics()
        log_metrics(**metrics)

        # Clear after flushing
        with self.lock:
            self.queries = []
            self.session_metrics = defaultdict(list)
