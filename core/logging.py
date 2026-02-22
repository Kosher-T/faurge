"""
core/logging.py
Structured JSON-lines logging setup for Faurge.
"""

import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path

# Need to safely import DIRS and log settings without circular import.
# Since settings.py configures DIRS, we can just fetch LOG_LEVEL and LOG_MAX_BYTES from os
# or defaults here, but it's cleaner to let settings pass them or grab them directly.
from core import defaults

class JsonLinesFormatter(logging.Formatter):
    """Format log records as single-line JSON strings."""
    def format(self, record):
        log_obj = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add extra args if any were passed
        if hasattr(record, "extra_data"):
            log_obj["extra"] = record.extra_data
            
        # Exception info
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_obj)

def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger instance with console and JSON-file rotating handlers.
    """
    logger = logging.getLogger(name)

    # Only configure if no handlers exist to avoid duplicate logs
    if not logger.handlers:
        base_dir = Path(__file__).resolve().parent.parent
        log_dir = base_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "faurge.log"
        
        log_level_str = os.getenv("FAURGE_LOG_LEVEL", defaults.LOG_LEVEL).upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
        logger.setLevel(log_level)

        # 1. Console Handler (Standard text)
        console_fmt = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        ch.setFormatter(console_fmt)
        logger.addHandler(ch)

        # 2. JSON-Lines Rotating File Handler
        try:
            max_bytes = int(os.getenv("FAURGE_LOG_MAX_BYTES", defaults.LOG_MAX_BYTES))
            backup_count = int(os.getenv("FAURGE_LOG_BACKUP_COUNT", defaults.LOG_BACKUP_COUNT))
        except ValueError:
            max_bytes = defaults.LOG_MAX_BYTES
            backup_count = defaults.LOG_BACKUP_COUNT

        fh = RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count
        )
        fh.setLevel(log_level)
        fh.setFormatter(JsonLinesFormatter())
        logger.addHandler(fh)

        # Don't propagate to root logger
        logger.propagate = False

    return logger

# Module-level convenience logger
logger = get_logger("faurge.core")
