"""
Simple Logging Configuration for RedSea GPT

Implements structured JSON logging to separate files.
"""

import logging
import json
from datetime import datetime
from pathlib import Path


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter"""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        return json.dumps(log_data)


def setup_logging(log_dir: str = "logs"):
    """Setup separate log files for different event types"""
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    # Create separate loggers (only requests and responses)
    loggers = {
        "requests": ("requests.log", logging.INFO),
        "responses": ("responses.log", logging.INFO),
    }

    for name, (filename, level) in loggers.items():
        logger = logging.getLogger(f"redsea.{name}")
        logger.setLevel(level)
        logger.propagate = False

        # File handler
        handler = logging.FileHandler(log_path / filename)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)

    return logging.getLogger("redsea.requests")
