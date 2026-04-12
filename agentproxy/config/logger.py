"""Centralized logging config for AgentProxy.

Usage in any module:
    import logging
    logger = logging.getLogger(__name__)
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from agentproxy.config.settings import BASE_DIR, get_settings

LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "agentproxy.log"

CONSOLE_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
FILE_FORMAT = "%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging() -> None:
    """Configure root logger with console + rotating file handlers.

    Call once at startup (e.g. in main.py).
    """
    settings = get_settings()
    level = logging.DEBUG if settings.debug else logging.INFO

    LOG_DIR.mkdir(exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # capture everything; handlers filter

    # --- Console: respects debug setting ---
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(CONSOLE_FORMAT, datefmt=DATE_FORMAT))
    root.addHandler(console)

    # --- File: always DEBUG, rotates at 5 MB, keeps 3 backups ---
    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(FILE_FORMAT, datefmt=DATE_FORMAT))
    root.addHandler(file_handler)

    # Quiet noisy third-party loggers
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("google").setLevel(logging.WARNING)
    logging.getLogger("google_genai").setLevel(logging.WARNING)
