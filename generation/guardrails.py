"""
Simple Guardrails for RedSea GPT

Implements:
- Rate limiting (sliding window)
- Content moderation (basic pattern matching)
"""

import time
import re
from typing import Tuple, Optional, Dict
from collections import deque
from threading import Lock


class RateLimiter:
    """Simple sliding window rate limiter"""

    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, deque] = {}
        self.lock = Lock()

    def is_allowed(self, identifier: str) -> Tuple[bool, Optional[str]]:
        """Check if request is allowed"""
        with self.lock:
            now = time.time()

            # Initialize deque for new identifier
            if identifier not in self.requests:
                self.requests[identifier] = deque()

            queue = self.requests[identifier]

            # Remove old requests outside window
            while queue and queue[0] < now - self.window_seconds:
                queue.popleft()

            # Check limit
            if len(queue) >= self.max_requests:
                return False, f"Rate limit exceeded: {self.max_requests} requests per {self.window_seconds} seconds"

            # Add current request
            queue.append(now)
            return True, None


class ContentModerator:
    """Basic content moderation using pattern matching"""

    # Patterns to block
    MALICIOUS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",  # onclick=, onload=, etc.
    ]

    OFF_TOPIC_PATTERNS = [
        r"ignore.*(?:your|previous).*(?:instructions|system)",
        r"forget.*everything",
        r"disregard.*context",
    ]

    def check(self, text: str) -> Tuple[bool, Optional[str]]:
        """Check if content is safe"""
        text_lower = text.lower()

        # Check for malicious patterns
        for pattern in self.MALICIOUS_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return False, "Request contains potentially malicious content"

        # Check for prompt injection
        for pattern in self.OFF_TOPIC_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return False, "Request appears to be a prompt injection attempt"

        return True, None


class RequestValidator:
    """Combines rate limiting and content moderation"""

    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.rate_limiter = RateLimiter(max_requests, window_seconds)
        self.moderator = ContentModerator()

    def validate(self, question: str, identifier: str) -> Tuple[bool, Optional[str]]:
        """
        Validate a request

        Returns:
            (is_valid, error_message)
        """
        # Check rate limit
        allowed, error = self.rate_limiter.is_allowed(identifier)
        if not allowed:
            return False, error

        # Check content
        safe, error = self.moderator.check(question)
        if not safe:
            return False, error

        return True, None
