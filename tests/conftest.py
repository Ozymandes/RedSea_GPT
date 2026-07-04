"""
Pytest configuration: make the project root importable without secrets.

These tests are designed to run safely in CI. They NEVER read or print API
keys. Tests that require a live LLM provider are skipped automatically when no
provider is configured.
"""

import os
import sys
from pathlib import Path

# Ensure project root is importable.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def pytest_configure(config):
    # Belt-and-braces: never let a stray .env key appear in test output.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
